"""The trust ledger: what the simulator has earned the right to be used for.

Conservative by construction. A stage with no evidence is ``UNKNOWN``, and
``UNKNOWN`` grants nothing -- it does not default to "probably fine". Evidence
expires when the game data version changes, because a balance patch can
invalidate a calibration that was correct last week.

The asymmetry is deliberate: one bad comparison downgrades a scope immediately,
while promoting it needs several consistent good ones. Being wrongly optimistic
about the simulator is the failure mode that silently poisons training; being
wrongly pessimistic only costs time.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ato.fidelity.divergence import Divergence, Usable


class TrustLevel(enum.IntEnum):
    """What a scope's accumulated evidence licenses."""

    UNKNOWN = 0        # never checked -- grants nothing
    UNTRUSTED = 1      # checked and found wrong
    TRIAGE = 2         # may rank options, may not be believed
    IMITATION = 3      # outcomes agree; may generate demonstrations
    PLANNING = 4       # timings agree to ~1s; may drive search
    TRAINING = 5       # timings agree to ~0.5s; may generate reward

    @property
    def allows_planning(self) -> bool:
        return self >= TrustLevel.PLANNING

    @property
    def allows_training(self) -> bool:
        return self >= TrustLevel.TRAINING

    @property
    def allows_imitation(self) -> bool:
        return self >= TrustLevel.IMITATION


#: How many consistent observations a scope needs before it may be promoted to
#: each level. Higher stakes, more evidence.
PROMOTION_SAMPLES = {
    TrustLevel.TRIAGE: 1,
    TrustLevel.IMITATION: 3,
    TrustLevel.PLANNING: 5,
    TrustLevel.TRAINING: 8,
}

_USABLE_TO_TRUST = {
    Usable.NOTHING: TrustLevel.UNTRUSTED,
    Usable.TRIAGE: TrustLevel.TRIAGE,
    Usable.IMITATION: TrustLevel.IMITATION,
    Usable.PLANNING: TrustLevel.PLANNING,
    Usable.TRAINING: TrustLevel.TRAINING,
}


@dataclass
class FidelityRecord:
    """Accumulated evidence about one scope (a stage id, or a mechanic key)."""

    scope: str
    kind: str = "stage"                  # stage | mechanic
    samples: int = 0
    #: Worst verdict seen since the last reset. Trust is bounded by the worst
    #: observation, not the average: a simulator that is usually right and
    #: occasionally very wrong is not safe to train on.
    worst: int = int(Usable.TRAINING)
    #: Running mean of the scalar divergence score, for trend reporting.
    mean_score: float = 0.0
    #: Number of consecutive observations at or above the current best tier.
    streak: int = 0
    gamedata_version: str = ""
    last_summary: str = ""
    #: Set when evidence came from a real device or a video rather than another
    #: simulator run. Sim-vs-sim agreement says nothing about reality, so it can
    #: never promote past PLANNING.
    grounded: bool = False

    @property
    def trust(self) -> TrustLevel:
        if self.samples == 0:
            return TrustLevel.UNKNOWN
        cap = _USABLE_TO_TRUST[Usable(self.worst)]
        if not self.grounded:
            # Agreement between two simulator runs is a robustness check, not
            # evidence about the real game. It cannot license reward generation.
            cap = min(cap, TrustLevel.PLANNING)
        # Walk down until we have enough samples for the level we are claiming.
        level = cap
        while level > TrustLevel.UNTRUSTED and self.streak < PROMOTION_SAMPLES.get(level, 1):
            level = TrustLevel(level - 1)
        return level


@dataclass
class FidelityLedger:
    """All the evidence, and the gate that reads it."""

    records: dict[str, FidelityRecord] = field(default_factory=dict)
    gamedata_version: str = ""

    # -- evidence --------------------------------------------------------

    def observe(
        self, div: Divergence, *, scope: str | None = None, kind: str = "stage",
        grounded: bool = False, gamedata_version: str = "",
    ) -> FidelityRecord:
        key = scope or div.stage_id
        rec = self.records.get(key)
        version = gamedata_version or self.gamedata_version
        if rec is None or (version and rec.gamedata_version and rec.gamedata_version != version):
            # A balance patch invalidates what we knew. Start again rather than
            # carrying stale confidence forward.
            rec = FidelityRecord(scope=key, kind=kind, gamedata_version=version)
            self.records[key] = rec
        rec.gamedata_version = version or rec.gamedata_version
        rec.grounded = rec.grounded or grounded

        seen = int(div.usable)
        if seen < rec.worst:
            # One bad observation drops the scope immediately and resets the
            # streak: trust is bounded by the worst thing we have seen.
            rec.worst = seen
            rec.streak = 1
        else:
            rec.streak += 1
        rec.mean_score = (rec.mean_score * rec.samples + div.score()) / (rec.samples + 1)
        rec.samples += 1
        rec.last_summary = div.summary()
        return rec

    def reset(self, scope: str) -> None:
        self.records.pop(scope, None)

    # -- the gate --------------------------------------------------------

    def trust(self, scope: str) -> TrustLevel:
        rec = self.records.get(scope)
        return rec.trust if rec else TrustLevel.UNKNOWN

    def allows_training(self, stage_id: str) -> bool:
        """May a battle on this stage produce a reward signal?

        This is the question INVARIANT I-5 exists for. An unchecked stage
        answers no, and that is not a bug to work around.
        """
        return self.trust(stage_id).allows_training

    def allows_planning(self, stage_id: str) -> bool:
        return self.trust(stage_id).allows_planning

    def allows_imitation(self, stage_id: str) -> bool:
        return self.trust(stage_id).allows_imitation

    def trainable_stages(self) -> tuple[str, ...]:
        return tuple(
            k for k, r in self.records.items()
            if r.kind == "stage" and r.trust.allows_training
        )

    def needs_grounding(self) -> tuple[str, ...]:
        """Stages whose only evidence is simulator-versus-simulator.

        These are the ones to point a device or a video at next: they are the
        cheapest way to expand what the system may legitimately train on.
        """
        return tuple(k for k, r in self.records.items() if r.samples and not r.grounded)

    # -- persistence -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "gamedata_version": self.gamedata_version,
                "records": {k: asdict(v) for k, v in self.records.items()},
            },
            ensure_ascii=False, indent=1, sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> FidelityLedger:
        raw = json.loads(text)
        return FidelityLedger(
            gamedata_version=raw.get("gamedata_version", ""),
            records={k: FidelityRecord(**v) for k, v in (raw.get("records") or {}).items()},
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), "utf-8")

    @staticmethod
    def load(path: Path) -> FidelityLedger:
        p = Path(path)
        return FidelityLedger.from_json(p.read_text("utf-8")) if p.exists() else FidelityLedger()

    def report(self) -> str:
        if not self.records:
            return "fidelity ledger is empty: the simulator has not been checked against anything"
        lines = [f"{'scope':28s} {'trust':10s} {'n':>3s} {'score':>6s}  grounded  last"]
        for k in sorted(self.records, key=lambda k: (-self.records[k].trust, k)):
            r = self.records[k]
            lines.append(
                f"{k[:28]:28s} {r.trust.name:10s} {r.samples:3d} {r.mean_score:6.3f}  "
                f"{'yes' if r.grounded else 'NO ':8s}  {r.last_summary[:60]}"
            )
        return "\n".join(lines)
