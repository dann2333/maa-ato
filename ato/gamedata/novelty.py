"""Turn "what changed" into "what to do about it".

Two things generate novelty, and they are not equally dangerous:

* a **snapshot diff** — new or rebalanced content. The simulator keeps working;
  it is simply modelling content the policy has never met.
* a **:class:`~ato.sim.registry.NoveltyEvent`** — a mechanic key the simulator
  does not implement. Nothing refuses, nothing crashes outside strict mode: the
  battle still produces a number, and that number is wrong. Every reward
  derived from it poisons training.

So the priority function ranks mechanism novelty above content novelty of the
same size, and the ingestion plan routes it to a different place: content is
data (or at worst a training run), a missing mechanic is a code change.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ato.gamedata.diff import Change, ChangeKind, EntityType, SnapshotDiff, combat_delta
from ato.sim.registry import MechanismKind, NoveltyEvent


class ItemKind(enum.StrEnum):
    NEW_OPERATOR = "new_operator"
    NEW_ENEMY = "new_enemy"
    NEW_STAGE = "new_stage"
    NEW_SKILL = "new_skill"
    NEW_RANGE = "new_range"
    NEW_MODULE = "new_module"
    REBALANCE = "rebalance"
    CHANGED_LEVEL = "changed_level"
    REMOVED = "removed"


#: A mechanism item's kind is the :class:`MechanismKind` itself, so the two
#: namespaces share one flat ``kind`` field without colliding.
MECHANISM_KINDS = frozenset(str(k) for k in MechanismKind)

# --- mechanism weights -----------------------------------------------------
#
# Ordered by how much of a battle one unmodelled key silently falsifies.
MECHANISM_WEIGHT: dict[str, float] = {
    # A rune rescales *every* unit in the battle (Adverse mode: enemy atk/def/hp
    # x1.2, +1 life point, x2 DP). Miss one and every number in the fight is off.
    str(MechanismKind.RUNE): 1.00,
    # Checkpoints decide where enemies actually go; get them wrong and every
    # block/leak decision the policy learns is learned against a fictional route.
    str(MechanismKind.CHECKPOINT): 0.90,
    # Wave actions decide what spawns and when — wrong pressure, wrong DP curve.
    str(MechanismKind.WAVE_ACTION): 0.85,
    # An enemy ability distorts the fights that contain that enemy.
    str(MechanismKind.ENEMY_ABILITY): 0.80,
    # A skill blackboard key distorts one operator's kit; the rest of the battle
    # stays honest, and the error is bounded by that operator's uptime.
    str(MechanismKind.SKILL_KEY): 0.75,
    # Traits are passive and usually small, but they are always on.
    str(MechanismKind.TRAIT): 0.70,
    # Most new tile keys are decoration. The load-bearing ones (healing floors,
    # conveyors) also announce themselves through a rune or an ability key, so
    # under-weighting tiles does not lose the mechanic — only its label.
    str(MechanismKind.TILE): 0.30,
}
#: An unrecognised mechanism kind is treated as mid-severity rather than ignored.
DEFAULT_MECHANISM_WEIGHT = 0.70
#: A key met in many places corrupts many battles; small, capped, monotonic.
FREQUENCY_BONUS = 0.02
MAX_FREQUENCY_BONUS = 0.10

# --- content weights -------------------------------------------------------
#
# Base for a 1-star; rarity adds on top. A 6-star is a new tactical primitive
# (new range shapes, new skill categories, often a new mechanic); a 1-star is a
# stat block the policy already generalises over.
W_NEW_OPERATOR = 0.40
RARITY_STEP = 0.07
#: Tokens and traps are summoned by someone else's kit and are never chosen by
#: the policy, so they arrive as data, not as a new decision.
W_NEW_TOKEN = 0.20
W_NEW_ENEMY = 0.40
#: Elites and bosses are where high-difficulty content is decided, and bosses
#: are where new mechanics ship. A trash mob is a stat variant of one we know.
ELITE_BONUS = 0.20
BOSS_BONUS = 0.40
#: A new stage extends the curriculum; it invalidates nothing already learned.
W_NEW_STAGE = 0.30
#: ...unless it is an Adverse/high-difficulty variant, where simulator error and
#: policy error both show up first.
HARD_STAGE_BONUS = 0.10
#: A new skill id usually ships with a new operator (counted there already).
W_NEW_SKILL = 0.35
#: A module rewrites an existing operator's stats and sometimes their talent, so
#: everything the policy believes about that operator's numbers is stale.
W_NEW_MODULE = 0.60
#: Pure geometry — the simulator consumes a new range with zero code.
W_NEW_RANGE = 0.10
#: A retuned level file means that stage was redesigned under a name we already
#: trained on, which is worse than a new stage: old plans look valid and are not.
W_CHANGED_LEVEL = 0.65
#: Disappearing content is nearly always a table restructure or a bad fetch.
#: Acting on it (forgetting a trained operator) is not reversible, so a human
#: confirms it.
W_REMOVED = 0.60
#: A rebalance is scaled by how far the numbers actually moved.
W_REBALANCE = 0.70
#: Below this, a scaled rebalance still counts as a real change but does not
#: dominate; the floor keeps a 1% tweak from ranking as zero.
REBALANCE_FLOOR = 0.35
#: A 20% swing on a combat number changes which strategies win. Under roughly
#: 15% the value estimates degrade gracefully and the next scheduled run absorbs
#: the new numbers, which is why the retrain threshold sits where it does.
REBALANCE_FULL_SCALE = 0.20

# --- routing ---------------------------------------------------------------
#: At or above this, an item is worth its own training run rather than waiting
#: for the next scheduled one.
RETRAIN_THRESHOLD = 0.60
#: Mechanism items at or above this actively falsify simulation results; nothing
#: may be trained on affected content until a handler exists.
SIM_CORRUPTION_PRIORITY = 0.75
#: How much supporting detail an item carries into the report.
MAX_EVIDENCE_FIELDS = 8
MAX_EVIDENCE_CONTEXTS = 5

BUCKETS = ("auto", "retrain", "code", "ask_human")


@dataclass
class NoveltyItem:
    """One thing the system has to do something about."""

    kind: str
    key: str
    name: str = ""
    priority: float = 0.0
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    #: True when the system cannot settle this on its own — acting would be
    #: irreversible or the data does not say enough to act at all.
    needs_human: bool = False

    @property
    def is_mechanism(self) -> bool:
        return self.kind in MECHANISM_KINDS

    def label(self) -> str:
        return f"{self.key} ({self.name})" if self.name else self.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "key": self.key,
            "name": self.name,
            "priority": self.priority,
            "reason": self.reason,
            "evidence": self.evidence,
            "needs_human": self.needs_human,
        }


def _sorted(items: Iterable[NoveltyItem]) -> list[NoveltyItem]:
    return sorted(items, key=lambda i: (-i.priority, str(i.kind), i.key))


def _round(value: float) -> float:
    # Priorities are compared against thresholds; keep them off float dust.
    return round(min(max(value, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# From a snapshot diff
# ---------------------------------------------------------------------------


def _added_item(c: Change) -> NoveltyItem | None:
    d = c.detail
    if c.entity_type == EntityType.OPERATOR:
        rarity = int(d.get("rarity") or 0)
        if d.get("is_token"):
            return NoveltyItem(
                ItemKind.NEW_OPERATOR,
                c.entity_id,
                c.name,
                _round(W_NEW_TOKEN),
                f"new {d.get('profession', '').lower() or 'token'} unit — summoned, never chosen",
            )
        return NoveltyItem(
            ItemKind.NEW_OPERATOR,
            c.entity_id,
            c.name,
            _round(W_NEW_OPERATOR + RARITY_STEP * rarity),
            f"new {rarity}* {d.get('profession', '?').lower()}/{d.get('sub_profession', '?')} "
            "operator the policy has never deployed",
        )
    if c.entity_type == EntityType.ENEMY:
        tier = str(d.get("enemy_level") or "").upper()
        bonus = BOSS_BONUS if tier == "BOSS" else ELITE_BONUS if tier == "ELITE" else 0.0
        return NoveltyItem(
            ItemKind.NEW_ENEMY,
            c.entity_id,
            c.name,
            _round(W_NEW_ENEMY + bonus),
            f"new {tier.lower() or 'normal'} enemy — stats are data, its abilities may not be",
        )
    if c.entity_type == EntityType.STAGE:
        hard = str(d.get("difficulty") or "NORMAL").upper() not in ("", "NORMAL")
        return NoveltyItem(
            ItemKind.NEW_STAGE,
            c.entity_id,
            c.name,
            _round(W_NEW_STAGE + (HARD_STAGE_BONUS if hard else 0.0)),
            f"new {d.get('stage_type', '?').lower()} stage "
            f"({d.get('difficulty', 'NORMAL')}) — extends the curriculum",
        )
    if c.entity_type == EntityType.SKILL:
        return NoveltyItem(
            ItemKind.NEW_SKILL, c.entity_id, c.name, _round(W_NEW_SKILL),
            "new skill — its blackboard keys decide whether the simulator models it",
        )
    if c.entity_type == EntityType.MODULE:
        return NoveltyItem(
            ItemKind.NEW_MODULE, c.entity_id, c.name, _round(W_NEW_MODULE),
            "new module — an existing operator's numbers change under it",
        )
    if c.entity_type == EntityType.RANGE:
        return NoveltyItem(
            ItemKind.NEW_RANGE, c.entity_id, c.name, _round(W_NEW_RANGE),
            "new attack range — pure geometry, consumed as-is",
        )
    # A level file only ever appears because it was fetched, not because the
    # game gained one; see ato.gamedata.diff._diff_levels.
    return None


def _item_from_change(c: Change) -> NoveltyItem | None:
    """One item per change, or None for changes that must not cost anything."""
    evidence: dict[str, Any] = {"entity_type": str(c.entity_type), "change": str(c.kind)}
    evidence.update({k: v for k, v in c.detail.items() if k not in ("combat", "fields")})

    if c.kind == ChangeKind.ADDED:
        item = _added_item(c)
        if item is None:
            return None
        item.evidence = evidence
        return item

    if c.kind == ChangeKind.REMOVED:
        return NoveltyItem(
            ItemKind.REMOVED,
            c.entity_id,
            c.name,
            _round(W_REMOVED),
            f"{c.entity_type} disappeared from the tables — restructure, rename or bad fetch",
            evidence,
            needs_human=True,
        )

    # MODIFIED. A change that touches no combat-relevant leaf is decoration:
    # it produces no item at all, so flavour text can never trigger retraining.
    if not c.detail.get("combat_count"):
        return None

    delta = combat_delta(c)
    evidence["combat"] = c.detail.get("combat", {})
    evidence["fields"] = list(c.fields[:MAX_EVIDENCE_FIELDS])
    evidence["delta"] = round(delta, 3)
    if c.entity_type == EntityType.LEVEL:
        return NoveltyItem(
            ItemKind.CHANGED_LEVEL, c.entity_id, c.name, _round(W_CHANGED_LEVEL),
            "level file retuned — plans and value estimates for this stage are stale",
            evidence,
        )
    scale = REBALANCE_FLOOR + (1.0 - REBALANCE_FLOOR) * min(1.0, delta / REBALANCE_FULL_SCALE)
    return NoveltyItem(
        ItemKind.REBALANCE,
        c.entity_id,
        c.name,
        _round(W_REBALANCE * scale),
        f"{c.entity_type} rebalanced: {', '.join(c.combat_names())} "
        f"({c.detail['combat_count']} combat field(s), largest move {delta:.0%})",
        evidence,
    )


def from_diff(diff: SnapshotDiff) -> list[NoveltyItem]:
    """Work items implied by a snapshot diff, highest priority first."""
    items = [item for item in map(_item_from_change, diff.changes) if item is not None]
    return _sorted(items)


# ---------------------------------------------------------------------------
# From runtime novelty events
# ---------------------------------------------------------------------------


def from_events(events: Iterable[NoveltyEvent]) -> list[NoveltyItem]:
    """Work items implied by mechanics the simulator met but does not model."""
    counts: dict[tuple[str, str], int] = {}
    contexts: dict[tuple[str, str], list[str]] = {}
    for ev in events:
        ident = (str(ev.kind), str(ev.key))
        counts[ident] = counts.get(ident, 0) + 1
        ctxs = contexts.setdefault(ident, [])
        if ev.context and ev.context not in ctxs:
            ctxs.append(ev.context)

    items: list[NoveltyItem] = []
    for (kind, key), count in counts.items():
        base = MECHANISM_WEIGHT.get(kind, DEFAULT_MECHANISM_WEIGHT)
        seen = contexts[(kind, key)]
        items.append(
            NoveltyItem(
                kind=kind,
                key=key,
                priority=_round(base + min(MAX_FREQUENCY_BONUS, FREQUENCY_BONUS * (count - 1))),
                reason=f"unmodelled {kind} key met {count}x — battles containing it are "
                "simulated wrong, not refused",
                evidence={"occurrences": count, "contexts": seen[:MAX_EVIDENCE_CONTEXTS]},
                # Without a single context there is nothing to read the mechanic's
                # meaning off: no stage, no unit, no blackboard. Only a person
                # looking at the game can say what it does.
                needs_human=not seen,
            )
        )
    return _sorted(items)


def merge(*groups: Iterable[NoveltyItem]) -> list[NoveltyItem]:
    """Combine item lists, keeping one entry per ``(kind, key)``."""
    out: dict[tuple[str, str], NoveltyItem] = {}
    for group in groups:
        for item in group:
            ident = (str(item.kind), item.key)
            prev = out.get(ident)
            if prev is None:
                # Copy: merging mutates the accumulator, never the caller's items.
                out[ident] = replace(item, evidence=dict(item.evidence))
                continue
            prev.priority = max(prev.priority, item.priority)
            prev.needs_human = prev.needs_human or item.needs_human
            prev.name = prev.name or item.name
            if item.reason and item.reason not in prev.reason:
                prev.reason = f"{prev.reason}; {item.reason}" if prev.reason else item.reason
            for k, v in item.evidence.items():
                prev.evidence.setdefault(k, v)
    return _sorted(out.values())


# ---------------------------------------------------------------------------
# Ingestion plan
# ---------------------------------------------------------------------------


def bucket_of(item: NoveltyItem) -> str:
    """Which pipeline stage owns this item.

    One rule, because the priority function already encodes severity: anything
    the system cannot settle goes to a human, any missing mechanic is code, and
    what is left is either worth a training run or is just data.
    """
    if item.needs_human:
        return "ask_human"
    if item.is_mechanism:
        return "code"
    if item.priority >= RETRAIN_THRESHOLD:
        return "retrain"
    return "auto"


@dataclass
class IngestionPlan:
    """Novelty split by what has to happen before the system is current again."""

    label: str = ""
    #: Pure data: new numbers flow into the simulator, no code, no training.
    auto: list[NoveltyItem] = field(default_factory=list)
    #: Needs a training run: content the policy has never had to decide about.
    retrain: list[NoveltyItem] = field(default_factory=list)
    #: Needs a handler registered in ``ato.sim.registry`` before it can be
    #: simulated honestly.
    code: list[NoveltyItem] = field(default_factory=list)
    #: Irreversible or under-determined; a person decides.
    ask_human: list[NoveltyItem] = field(default_factory=list)

    @property
    def items(self) -> list[NoveltyItem]:
        return _sorted([i for b in BUCKETS for i in getattr(self, b)])

    @property
    def blocking(self) -> list[NoveltyItem]:
        """Mechanisms severe enough that training on affected content is invalid.

        Scans every bucket: a mechanic no one can explain (``ask_human``)
        falsifies simulation exactly as much as one waiting on a handler.
        """
        return [
            i for i in self.items if i.is_mechanism and i.priority >= SIM_CORRUPTION_PRIORITY
        ]

    def counts(self) -> dict[str, int]:
        return {b: len(getattr(self, b)) for b in BUCKETS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "counts": self.counts(),
            "blocking": [i.key for i in self.blocking],
            "buckets": {b: [i.to_dict() for i in getattr(self, b)] for b in BUCKETS},
        }

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_text(self, limit: int = 20) -> str:
        counts = self.counts()
        head = ", ".join(f"{b} {counts[b]}" for b in BUCKETS)
        lines = [
            f"ingestion plan — {self.label or 'unlabelled'}",
            f"  {len(self.items)} items: {head}",
        ]
        if not self.items:
            lines.append("  nothing to ingest — the snapshots agree and the simulator met "
                         "no unknown mechanic")
        if self.blocking:
            lines += [
                "",
                f"  BLOCKING ({len(self.blocking)}): unmodelled mechanics falsify simulation.",
                "  Do not train on affected content until a handler is registered.",
            ]
        for bucket in BUCKETS:
            items: list[NoveltyItem] = getattr(self, bucket)
            if not items:
                continue
            lines += ["", f"  {bucket} ({len(items)}) — {_BUCKET_BLURB[bucket]}"]
            for item in items[:limit]:
                lines.append(f"    {item.priority:5.2f}  {item.kind:<14} {item.label()}")
                lines.append(f"           {item.reason}")
                ctxs = item.evidence.get("contexts")
                if ctxs:
                    lines.append(f"           seen in: {', '.join(map(str, ctxs))}")
            if len(items) > limit:
                lines.append(f"    ... {len(items) - limit} more")
        return "\n".join(lines)


_BUCKET_BLURB = {
    "auto": "data only: new numbers, zero code, zero training",
    "retrain": "needs a training run before the policy is current",
    "code": "needs a mechanic handler in ato.sim.registry",
    "ask_human": "cannot be settled from the data alone",
}


def plan_ingestion(items: Iterable[NoveltyItem], *, label: str = "") -> IngestionPlan:
    """Bucket novelty into what the continual-learning loop must do with it."""
    plan = IngestionPlan(label=label)
    for item in _sorted(items):
        getattr(plan, bucket_of(item)).append(item)
    return plan


def write_report(plan: IngestionPlan, path: str | Path) -> Path:
    """Write the plan as JSON (``.json`` suffix) or as the text report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = plan.to_json() if out.suffix == ".json" else plan.to_text()
    out.write_text(body + "\n", "utf-8")
    return out
