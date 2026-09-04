"""Work out what a game update actually changed.

Two snapshots straddle a game update; the continual-learning loop starts by
asking this module what is new. Two properties matter more than completeness:

* **Cheap.** The tables are 15 MB+ each. Byte-identical tables are skipped via
  the manifest hashes, entities are compared by key first, and only entities
  that genuinely differ are walked — with a bounded depth and a cap on the
  number of leaf paths recorded per entity.
* **Honest about significance.** Every changed leaf is classified as
  combat-relevant or cosmetic, so an operator's rewritten flavour text can
  never be reported as a rebalance and can never cost a training run.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ato.gamedata.tables import GameData


class ChangeKind(enum.StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class EntityType(enum.StrEnum):
    OPERATOR = "operator"
    ENEMY = "enemy"
    SKILL = "skill"
    STAGE = "stage"
    RANGE = "range"
    MODULE = "module"
    LEVEL = "level"


#: Leaf names whose value feeds combat arithmetic. A change to one of these
#: invalidates whatever the policy learned about that unit; anything not listed
#: here (and not inside a combat container) is assumed decorative.
COMBAT_FIELDS = frozenset({
    "maxHp", "atk", "def", "magicResistance", "cost", "blockCnt", "moveSpeed",
    "attackSpeed", "baseAttackTime", "respawnTime", "hpRecoveryPerSec",
    "spRecoveryPerSec", "maxDeployCount", "tauntLevel", "massLevel", "baseForceLevel",
    "epDamageResistance", "epResistance", "lifePointReduce", "rangeRadius",
    "applyWay", "motion", "position", "profession", "subProfessionId",
    "spCost", "initSp", "increment", "spType", "duration", "durationType",
    "skillType", "rangeId", "maxLevel", "grids", "direction",
    "stunImmune", "silenceImmune", "sleepImmune", "frozenImmune", "levitateImmune",
    "disarmedCombatImmune", "fearedImmune", "palsyImmune", "attractImmune",
    "teleportImmune", "groundBoundImmune",
    # Stage-level: pointing a stage at a different map invalidates every plan
    # ever made for it, and the difficulty flag selects which runes apply.
    "levelId", "difficulty",
})

#: Containers whose numeric leaves are all kit numbers. Blackboards carry the
#: values that actually decide how much a skill or talent does, and they are
#: named per effect (``atk_scale``), so they cannot be enumerated by leaf name.
COMBAT_CONTAINERS = frozenset({
    "attributes", "attributesKeyFrames", "favorKeyFrames", "potentialRanks",
    "phases", "talents", "skills", "trait", "blackboard", "attributeBlackboard",
    "talentBlackboard", "spData", "attributeModifiers", "buff", "levels",
})

#: Text and art. Checked *first*, so a description rewritten inside a talent's
#: blackboard-bearing candidate is still recognised as cosmetic.
COSMETIC_FIELDS = frozenset({
    "name", "description", "desc", "appellation", "itemDesc", "itemUsage",
    "itemObtainApproach", "storyText", "displayNumber", "sortId", "sortIndex",
    "iconId", "prefabId", "prefabKey", "characterPrefabKey", "portraitId",
    "nationId", "groupId", "teamId", "abilityList", "ability", "enemyIndex",
    "hideInHandbook", "hideInStage", "invisibleDetail", "uniEquipName",
})

#: Presentation keys inside a level file. Everything else in a level file is
#: simulator input by construction, so the rule there is inverted.
LEVEL_COSMETIC_FIELDS = frozenset({
    "loadingPicId", "bgmEvent", "environmentSe", "storyId", "opera", "previewPic",
    "displayDetailRewards", "levelDesc",
})

#: Entity subtrees are shallow; below this depth a diff is blackboard noise and
#: the whole subtree is reported as one path.
MAX_DEPTH = 7
#: More changed leaves than this means the entity was rewritten, not tweaked —
#: recording them all buys nothing and bloats the report.
MAX_PATHS = 24
#: Sample of before/after pairs kept for combat leaves (the count is exact).
MAX_COMBAT_SAMPLES = 16
#: Hard bound on one entity's traversal, so a pathological entry cannot stall
#: a diff of thousands of them.
MAX_NODES = 20_000

_IDENT = re.compile(r"[^.\[\]]+")
#: ``enemy_database`` boxes every field as ``{m_defined, m_value}``; the box is
#: not part of a field's identity, so path names see through it.
_WRAPPER_SEGMENTS = frozenset({"m_value", "m_defined"})
_MISSING = object()
#: Fields a record can be indexed by inside a list, so that a reordered list of
#: blackboard entries or key frames is not reported as a change.
_LIST_KEY_FIELDS = ("key", "attributeType", "id", "level", "phase")
_NON_PLAYABLE_PROFESSIONS = frozenset({"TOKEN", "TRAP"})


def rarity_index(raw: Any) -> int:
    """Stars, 1-6. New tables store ``TIER_5``; older ones a 0-based int."""
    if isinstance(raw, str):
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else 0
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw + 1
    return 0


def leaf_name(path: str) -> str:
    """Field a leaf path names: ``phases[1].data.atk`` -> ``atk``, box removed."""
    segs = [s for s in _IDENT.findall(path) if s not in _WRAPPER_SEGMENTS]
    return segs[-1] if segs else path


def is_combat_path(path: str, entity_type: str = "") -> bool:
    """True if a changed leaf can move a battle's outcome."""
    leaf = leaf_name(path)
    if leaf in COSMETIC_FIELDS:
        return False
    if entity_type == EntityType.LEVEL:
        return leaf not in LEVEL_COSMETIC_FIELDS
    return any(s in COMBAT_FIELDS or s in COMBAT_CONTAINERS for s in _IDENT.findall(path))


@dataclass
class Change:
    """One entity that appeared, disappeared or moved between two snapshots."""

    kind: ChangeKind
    entity_type: str
    entity_id: str
    name: str = ""
    #: Changed leaf paths, capped at ``MAX_PATHS``; empty for ADDED/REMOVED.
    fields: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rebalance(self) -> bool:
        return self.kind == ChangeKind.MODIFIED and bool(self.detail.get("combat_count"))

    @property
    def is_cosmetic(self) -> bool:
        return self.kind == ChangeKind.MODIFIED and not self.detail.get("combat_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "entity_type": str(self.entity_type),
            "entity_id": self.entity_id,
            "name": self.name,
            "fields": list(self.fields),
            "detail": self.detail,
        }

    def label(self) -> str:
        return f"{self.entity_id} ({self.name})" if self.name else self.entity_id

    def combat_names(self, limit: int = 3) -> list[str]:
        """Names of the combat leaves that moved — the readable form of ``fields``."""
        names = dict.fromkeys(leaf_name(p) for p in self.detail.get("combat", {}))
        return list(names)[:limit]


def combat_delta(change: Change) -> float:
    """Largest relative move among an entity's combat leaves.

    Measured against the *old* value, so a x1.2 buff reads as 20%. A categorical
    flip (``WALK`` -> ``FLY``, a phase appearing) has no relative size but
    changes everything, so it saturates at 1.0.
    """
    worst = 0.0
    for before, after in change.detail.get("combat", {}).values():
        numeric = (
            isinstance(before, (int, float))
            and isinstance(after, (int, float))
            and not isinstance(before, bool)
            and not isinstance(after, bool)
        )
        if not numeric:
            return 1.0
        base = abs(float(before)) or abs(float(after))
        if base > 0.0:
            worst = max(worst, abs(float(after) - float(before)) / base)
    return min(worst, 1.0)


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------


@dataclass
class _Walk:
    entity_type: str
    max_paths: int = MAX_PATHS
    paths: list[str] = field(default_factory=list)
    combat: dict[str, list[Any]] = field(default_factory=dict)
    combat_count: int = 0
    total: int = 0
    nodes: int = 0
    truncated: bool = False


def _summarise(value: Any) -> Any:
    """JSON-safe, bounded rendering of one side of a changed leaf."""
    if value is _MISSING:
        return "<absent>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, dict):
        return f"<{len(value)} fields>"
    if isinstance(value, list):
        return f"<{len(value)} items>"
    return repr(value)[:80]


def _record(w: _Walk, path: str, old: Any, new: Any) -> None:
    w.total += 1
    if len(w.paths) < w.max_paths:
        w.paths.append(path)
    else:
        w.truncated = True
    if is_combat_path(path, w.entity_type):
        w.combat_count += 1
        if len(w.combat) < MAX_COMBAT_SAMPLES:
            w.combat[path] = [_summarise(old), _summarise(new)]


def _index_list(seq: list[Any]) -> dict[str, Any] | None:
    """Index a list of records by their own key; None if that is not possible."""
    out: dict[str, Any] = {}
    for item in seq:
        if not isinstance(item, dict):
            return None
        key = None
        for f in _LIST_KEY_FIELDS:
            v = item.get(f)
            if isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)):
                key = str(v)
                break
        if key is None or key in out:
            return None
        out[key] = item
    return out or None


def _child(path: str, key: str, *, bracket: bool = False) -> str:
    if bracket:
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _walk(old: Any, new: Any, path: str, depth: int, w: _Walk) -> None:
    if old is not _MISSING and new is not _MISSING and old == new:
        return
    w.nodes += 1
    if w.nodes > MAX_NODES:
        w.truncated = True
        return
    if depth < MAX_DEPTH and isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            _walk(old.get(k, _MISSING), new.get(k, _MISSING), _child(path, k), depth + 1, w)
        return
    if depth < MAX_DEPTH and isinstance(old, list) and isinstance(new, list):
        oi, ni = _index_list(old), _index_list(new)
        if oi is not None and ni is not None:
            for k in sorted(set(oi) | set(ni)):
                sub = _child(path, k, bracket=True)
                _walk(oi.get(k, _MISSING), ni.get(k, _MISSING), sub, depth + 1, w)
        else:
            for i in range(max(len(old), len(new))):
                a = old[i] if i < len(old) else _MISSING
                b = new[i] if i < len(new) else _MISSING
                _walk(a, b, _child(path, str(i), bracket=True), depth + 1, w)
        return
    _record(w, path or "$", old, new)


def diff_entity(
    entity_type: str,
    entity_id: str,
    old: Any,
    new: Any,
    *,
    name: str = "",
    detail: dict[str, Any] | None = None,
    max_paths: int = MAX_PATHS,
) -> Change | None:
    """MODIFIED change for one entity, or None when the two versions are equal."""
    if old == new:
        return None
    w = _Walk(entity_type=entity_type, max_paths=max_paths)
    _walk(old, new, "", 0, w)
    if not w.total and not w.truncated:
        return None
    body = dict(detail or {})
    body.update(
        combat=w.combat,
        combat_count=w.combat_count,
        paths=w.total,
        truncated=w.truncated,
    )
    return Change(
        kind=ChangeKind.MODIFIED,
        entity_type=entity_type,
        entity_id=entity_id,
        name=name,
        fields=tuple(w.paths),
        detail=body,
    )


# ---------------------------------------------------------------------------
# Per-table metadata: what an entity is called and what a consumer needs to
# know about it without re-opening the table.
# ---------------------------------------------------------------------------

Meta = Callable[[str, Any], tuple[str, dict[str, Any]]]


def _operator_meta(char_id: str, entry: Any) -> tuple[str, dict[str, Any]]:
    prof = str(entry.get("profession") or "")
    return str(entry.get("name") or char_id), {
        "rarity": rarity_index(entry.get("rarity")),
        "profession": prof,
        "sub_profession": str(entry.get("subProfessionId") or ""),
        # Tokens and traps are summoned by someone else's kit, never deployed
        # by the policy — they must not be priced like a new operator.
        "is_token": prof in _NON_PLAYABLE_PROFESSIONS,
    }


def _enemy_name(levels: Any) -> str:
    """Level 0's ``name``, unwrapped from the ``{m_defined, m_value}`` box."""
    for entry in levels or ():
        node = (entry.get("enemyData") or {}).get("name")
        if isinstance(node, dict):
            node = node.get("m_value") if node.get("m_defined") else None
        if node:
            return str(node)
    return ""


def _enemy_meta_fn(handbook: dict[str, Any]) -> Meta:
    hb = handbook.get("enemyData", handbook) if isinstance(handbook, dict) else {}

    def meta(key: str, levels: Any) -> tuple[str, dict[str, Any]]:
        entry = hb.get(key) or {}
        return str(entry.get("name") or _enemy_name(levels) or key), {
            "enemy_level": str(entry.get("enemyLevel") or ""),
            "levels": len(levels or ()),
        }

    return meta


def _skill_meta(skill_id: str, entry: Any) -> tuple[str, dict[str, Any]]:
    levels = entry.get("levels") or []
    name = str(levels[0].get("name") or skill_id) if levels else skill_id
    return name, {"levels": len(levels)}


def _stage_meta(stage_id: str, entry: Any) -> tuple[str, dict[str, Any]]:
    return str(entry.get("code") or entry.get("name") or stage_id), {
        "code": str(entry.get("code") or ""),
        "difficulty": str(entry.get("difficulty") or ""),
        "stage_type": str(entry.get("stageType") or ""),
        "level_id": str(entry.get("levelId") or ""),
        "ap_cost": entry.get("apCost"),
    }


def _range_meta(range_id: str, entry: Any) -> tuple[str, dict[str, Any]]:
    return "", {"grids": len(entry.get("grids") or ())}


def _module_meta(equip_id: str, entry: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(entry, dict):
        return "", {}
    return str(entry.get("uniEquipName") or ""), {"char_id": str(entry.get("charId") or "")}


def _diff_mapping(
    entity_type: str,
    table: str,
    old_map: dict[str, Any],
    new_map: dict[str, Any],
    meta: Meta,
    *,
    max_paths: int = MAX_PATHS,
) -> Iterator[Change]:
    """Key-first diff of one table: added, removed, then modified entities."""
    old_keys, new_keys = set(old_map), set(new_map)
    for key in sorted(new_keys - old_keys):
        name, detail = meta(key, new_map[key])
        yield Change(ChangeKind.ADDED, entity_type, key, name, (), {"table": table, **detail})
    for key in sorted(old_keys - new_keys):
        name, detail = meta(key, old_map[key])
        yield Change(ChangeKind.REMOVED, entity_type, key, name, (), {"table": table, **detail})
    for key in sorted(old_keys & new_keys):
        old_entry, new_entry = old_map[key], new_map[key]
        if old_entry == new_entry:
            continue
        name, detail = meta(key, new_entry)
        change = diff_entity(
            entity_type,
            key,
            old_entry,
            new_entry,
            name=name,
            detail={"table": table, **detail},
            max_paths=max_paths,
        )
        if change is not None:
            yield change


# ---------------------------------------------------------------------------
# Snapshot diff
# ---------------------------------------------------------------------------


@dataclass
class SnapshotDiff:
    """Everything that changed between two pinned snapshots."""

    old_version: str
    new_version: str
    server: str = "cn"
    changes: list[Change] = field(default_factory=list)
    #: Tables the manifests prove identical — not parsed at all.
    unchanged_tables: tuple[str, ...] = ()
    #: Tables missing from one side, with why. They are *not* reported as mass
    #: additions or removals: an absent table is a snapshot gap, not a game change.
    skipped_tables: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.changes)

    def of_type(self, entity_type: str) -> list[Change]:
        return [c for c in self.changes if c.entity_type == entity_type]

    def added(self, entity_type: str) -> list[Change]:
        return [
            c for c in self.changes
            if c.kind == ChangeKind.ADDED and c.entity_type == entity_type
        ]

    @property
    def new_operators(self) -> list[Change]:
        return self.added(EntityType.OPERATOR)

    @property
    def new_enemies(self) -> list[Change]:
        return self.added(EntityType.ENEMY)

    @property
    def new_stages(self) -> list[Change]:
        return self.added(EntityType.STAGE)

    @property
    def removed(self) -> list[Change]:
        return [c for c in self.changes if c.kind == ChangeKind.REMOVED]

    @property
    def rebalanced(self) -> list[Change]:
        """Existing entities whose combat numbers moved — the retraining trigger."""
        return [c for c in self.changes if c.is_rebalance]

    @property
    def cosmetic(self) -> list[Change]:
        """Modified entities with no combat-relevant leaf. Deliberately inert."""
        return [c for c in self.changes if c.is_cosmetic]

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for c in self.changes:
            row = out.setdefault(
                str(c.entity_type), {"added": 0, "removed": 0, "modified": 0, "rebalanced": 0}
            )
            row[str(c.kind)] += 1
            if c.is_rebalance:
                row["rebalanced"] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "server": self.server,
            "counts": self.counts(),
            "unchanged_tables": list(self.unchanged_tables),
            "skipped_tables": self.skipped_tables,
            "changes": [c.to_dict() for c in self.changes],
        }

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self, limit: int = 8) -> str:
        lines = [
            f"gamedata diff  {self.server}  {self.old_version} -> {self.new_version}",
            f"  tables: {len(self.unchanged_tables)} unchanged (hash), "
            f"{len(self.skipped_tables)} unavailable"
            + (f" ({', '.join(sorted(self.skipped_tables))})" if self.skipped_tables else ""),
        ]
        counts = self.counts()
        for etype in sorted(counts):
            row = counts[etype]
            lines.append(
                f"  {etype:<9} +{row['added']:<4} -{row['removed']:<4} "
                f"~{row['modified']:<4} ({row['rebalanced']} rebalanced)"
            )
        if not counts:
            lines.append("  no content changes")
        lines.append(
            f"  totals: {len(self.changes)} changes, {len(self.rebalanced)} rebalanced, "
            f"{len(self.cosmetic)} cosmetic-only"
        )

        def listing(title: str, changes: list[Change], fmt: Callable[[Change], str]) -> None:
            if not changes:
                return
            shown = ", ".join(fmt(c) for c in changes[:limit])
            more = f" (+{len(changes) - limit} more)" if len(changes) > limit else ""
            lines.append(f"  {title}: {shown}{more}")

        listing(
            "new operators",
            self.new_operators,
            lambda c: f"{c.label()} {c.detail.get('rarity', 0)}*",
        )
        listing(
            "new enemies",
            self.new_enemies,
            lambda c: f"{c.label()} {c.detail.get('enemy_level', '')}".rstrip(),
        )
        listing("new stages", self.new_stages, lambda c: c.label())
        listing("removed", self.removed, lambda c: f"{c.entity_type} {c.label()}")
        listing(
            "rebalanced",
            self.rebalanced,
            lambda c: f"{c.label()} [{', '.join(c.combat_names())}] {combat_delta(c):.0%}",
        )
        return "\n".join(lines)


def _same_hash(old: GameData, new: GameData, table: str) -> bool:
    """True when both manifests carry the same *non-empty* hash for a table.

    Hand-made snapshots leave the hash empty; two empty strings are not evidence
    of equality, and treating them as such would silently skip every table.
    """
    om, nm = old.manifest, new.manifest
    if om is None or nm is None:
        return False
    o, n = om.files.get(table), nm.files.get(table)
    return bool(o and n and o.sha256 and o.sha256 == n.sha256)


def _mapping(gd: GameData, table: str) -> dict[str, Any]:
    """The entity mapping inside a table, whatever shape the table has."""
    raw = gd.table(table)
    if table == "stage_table":
        return raw["stages"]
    if table == "enemy_database":
        return {e["Key"]: e["Value"] for e in raw["enemies"]}
    if table == "uniequip_table":
        return raw.get("equipDict") or {}
    return raw


def _optional(gd: GameData, table: str) -> dict[str, Any]:
    try:
        return gd.table(table)
    except (FileNotFoundError, KeyError):
        return {}


def _diff_levels(old: GameData, new: GameData, *, max_paths: int) -> Iterator[Change]:
    """Diff the level files present in *both* snapshots.

    Levels are fetched lazily, so a file present on one side only says something
    about what has been simulated, not about what the game changed — reporting
    that as new content would be a lie.
    """
    o_root, n_root = old.root / "levels", new.root / "levels"
    if not o_root.is_dir() or not n_root.is_dir():
        return
    for path in sorted(o_root.rglob("*.json")):
        other = n_root / path.relative_to(o_root)
        if not other.is_file() or path.read_bytes() == other.read_bytes():
            continue
        level_id = path.relative_to(o_root).with_suffix("").as_posix()
        change = diff_entity(
            EntityType.LEVEL,
            level_id,
            json.loads(path.read_text("utf-8")),
            json.loads(other.read_text("utf-8")),
            detail={"table": "levels"},
            max_paths=max_paths,
        )
        if change is not None:
            yield change


def diff_snapshots(
    old: GameData,
    new: GameData,
    *,
    include_levels: bool = True,
    max_paths: int = MAX_PATHS,
) -> SnapshotDiff:
    """Compare two snapshots table by table."""
    diff = SnapshotDiff(old_version=old.version, new_version=new.version, server=new.server)
    unchanged: list[str] = []

    enemy_meta = _enemy_meta_fn(_optional(new, "enemy_handbook_table"))
    plan: tuple[tuple[str, str, Meta], ...] = (
        ("character_table", EntityType.OPERATOR, _operator_meta),
        ("enemy_database", EntityType.ENEMY, enemy_meta),
        ("skill_table", EntityType.SKILL, _skill_meta),
        ("stage_table", EntityType.STAGE, _stage_meta),
        ("range_table", EntityType.RANGE, _range_meta),
        ("uniequip_table", EntityType.MODULE, _module_meta),
        ("battle_equip_table", EntityType.MODULE, _module_meta),
    )

    for table, entity_type, meta in plan:
        if _same_hash(old, new, table):
            unchanged.append(table)
            continue
        try:
            old_map, new_map = _mapping(old, table), _mapping(new, table)
        except FileNotFoundError as exc:
            diff.skipped_tables[table] = str(exc)
            continue
        except (KeyError, TypeError, AttributeError) as exc:
            # A table whose shape moved is news, but it must not abort the diff
            # of the other tables — it is reported, never silently dropped.
            diff.skipped_tables[table] = f"unreadable ({type(exc).__name__}: {exc})"
            continue
        diff.changes.extend(
            _diff_mapping(entity_type, table, old_map, new_map, meta, max_paths=max_paths)
        )

    if include_levels:
        diff.changes.extend(_diff_levels(old, new, max_paths=max_paths))

    diff.unchanged_tables = tuple(unchanged)
    return diff
