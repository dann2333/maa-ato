"""Turning two battle traces into concrete error numbers.

The thresholds here are the ones in ``docs/SIM_SPEC.md`` section D, and they are
deliberately tiered by *what the simulator is being asked to do*. Predicting
whether a stage clears is a much weaker requirement than predicting when each
enemy dies, and a simulator can be good enough for the first while being useless
for the second. Collapsing that into a single "is the sim good" boolean is how
projects end up training on a model that was only ever validated for something
easier.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ato.fidelity.trace import BattleTrace


class Usable(enum.IntEnum):
    """What a given level of agreement licenses. Strictly increasing demands."""

    NOTHING = 0
    #: Good enough to rank two plans against each other, not to trust the details.
    TRIAGE = 1
    #: Good enough to imitate: outcomes agree, so a cleared plan really clears.
    IMITATION = 2
    #: Good enough to plan with: event timings line up within a second.
    PLANNING = 3
    #: Good enough to generate reward from: timings line up within half a second.
    TRAINING = 4


#: Tier thresholds. Every one of these is an engineering judgement rather than a
#: measured constant, and each is stated as the question it answers.
CLEAR_AGREEMENT_FOR_IMITATION = True     # did both runs reach the same verdict
KILL_L1_FOR_PLANNING = 1.0               # seconds of mean kill-time error
KILL_L1_FOR_TRAINING = 0.5
FIRST_LEAK_ERROR_FOR_TRAINING = 0.5      # seconds
#: Beyond this fraction, the two runs did not even see the same battle.
KILL_COUNT_MISMATCH_FOR_TRIAGE = 0.25


@dataclass
class Divergence:
    """How far apart two traces are, and what that licenses."""

    stage_id: str
    clear_agreement: bool
    reference_cleared: bool
    candidate_cleared: bool
    kill_count_reference: int
    kill_count_candidate: int
    kill_time_l1: float | None
    first_leak_error: float | None
    life_lost_error: int
    duration_error: float
    dp_l1: float | None
    usable: Usable = Usable.NOTHING
    notes: list[str] = field(default_factory=list)

    @property
    def kill_count_mismatch(self) -> float:
        ref = max(self.kill_count_reference, 1)
        return abs(self.kill_count_candidate - self.kill_count_reference) / ref

    def score(self) -> float:
        """A single number in [0, 1], for ranking and for the ledger.

        Blunt on purpose: the tiered :attr:`usable` verdict is what decisions
        should key off. This exists so that trends over many comparisons are
        summarisable.
        """
        s = 1.0
        if not self.clear_agreement:
            s *= 0.3
        s *= max(0.0, 1.0 - self.kill_count_mismatch)
        if self.kill_time_l1 is not None:
            s *= 1.0 / (1.0 + self.kill_time_l1)
        if self.first_leak_error is not None:
            s *= 1.0 / (1.0 + 0.5 * self.first_leak_error)
        return max(0.0, min(1.0, s))

    def summary(self) -> str:
        parts = [
            f"stage={self.stage_id}",
            f"verdict={'agree' if self.clear_agreement else 'DISAGREE'}"
            f"({self.reference_cleared}->{self.candidate_cleared})",
            f"kills={self.kill_count_reference}/{self.kill_count_candidate}",
        ]
        if self.kill_time_l1 is not None:
            parts.append(f"kill_L1={self.kill_time_l1:.2f}s")
        if self.first_leak_error is not None:
            parts.append(f"first_leak_err={self.first_leak_error:.2f}s")
        if self.dp_l1 is not None:
            parts.append(f"dp_L1={self.dp_l1:.2f}")
        parts.append(f"score={self.score():.3f}")
        parts.append(f"usable={self.usable.name}")
        return "  ".join(parts)


def _kill_time_l1(ref: list[float], cand: list[float]) -> float | None:
    """Mean absolute error between the two kill-time sequences.

    Aligned by index rather than by nearest neighbour: the i-th kill in one run
    is meant to be the i-th kill in the other, and nearest-neighbour matching
    would hide exactly the drift we are trying to measure. Compared over the
    common prefix; the count difference is reported separately so a run that
    stops early cannot look accurate by being short.
    """
    n = min(len(ref), len(cand))
    if n == 0:
        return None
    return sum(abs(a - b) for a, b in zip(ref[:n], cand[:n])) / n


def _dp_l1(ref: BattleTrace, cand: BattleTrace) -> float | None:
    """Mean absolute DP difference over the overlapping time range.

    DP is the sharpest alignment signal available: it is deterministic given the
    stage and the deployments, so a DP curve that drifts means the two runs
    deployed differently or the cost rules differ -- both worth knowing before
    any timing metric is believed.
    """
    if not ref.dp or not cand.dp:
        return None
    end = min(ref.dp[-1][0], cand.dp[-1][0])
    if end <= 0:
        return None
    total = 0.0
    n = 0
    for t, v in ref.dp:
        if t > end:
            break
        other = cand.dp_at(t)
        if other is None:
            continue
        total += abs(v - other)
        n += 1
    return total / n if n else None


def classify(d: Divergence) -> Usable:
    """Map measured error onto what the simulator may be used for."""
    if d.kill_count_mismatch > KILL_COUNT_MISMATCH_FOR_TRIAGE:
        return Usable.NOTHING
    if not d.clear_agreement:
        return Usable.TRIAGE
    if d.kill_time_l1 is None:
        # Both runs agree on the verdict but nothing died in either. That is a
        # real agreement, just a weak one -- it says nothing about timing.
        return Usable.IMITATION
    if (
        d.kill_time_l1 <= KILL_L1_FOR_TRAINING
        and (d.first_leak_error is None or d.first_leak_error <= FIRST_LEAK_ERROR_FOR_TRAINING)
    ):
        return Usable.TRAINING
    if d.kill_time_l1 <= KILL_L1_FOR_PLANNING:
        return Usable.PLANNING
    return Usable.IMITATION


def compare(reference: BattleTrace, candidate: BattleTrace) -> Divergence:
    """Measure ``candidate`` against ``reference``.

    ``reference`` should be the more trustworthy trace: a device recording when
    validating the simulator, or the un-perturbed run when probing a plan's
    sensitivity to calibration.
    """
    notes: list[str] = []
    if reference.stage_id != candidate.stage_id:
        notes.append(f"stage mismatch: {reference.stage_id!r} vs {candidate.stage_id!r}")
    if len(reference.deploys) != len(candidate.deploys):
        notes.append(
            f"different action counts ({len(reference.deploys)} vs {len(candidate.deploys)}): "
            "timing metrics compare two different battles, not two models of one"
        )

    rl, cl = reference.first_leak, candidate.first_leak
    first_leak_error = None if (rl is None or cl is None) else abs(rl - cl)
    if (rl is None) != (cl is None):
        notes.append("one run leaked and the other did not")

    d = Divergence(
        stage_id=reference.stage_id,
        clear_agreement=reference.cleared == candidate.cleared,
        reference_cleared=reference.cleared,
        candidate_cleared=candidate.cleared,
        kill_count_reference=len(reference.kills),
        kill_count_candidate=len(candidate.kills),
        kill_time_l1=_kill_time_l1(reference.kills, candidate.kills),
        first_leak_error=first_leak_error,
        life_lost_error=abs(reference.life_lost - candidate.life_lost),
        duration_error=abs(reference.duration - candidate.duration),
        dp_l1=_dp_l1(reference, candidate),
        notes=notes,
    )
    d.usable = classify(d)
    return d
