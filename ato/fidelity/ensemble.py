"""Asking whether a plan depends on the simulator happening to be right.

This is a different question from "is the simulator accurate", and a more
answerable one. We cannot know the true value of every fitted constant, but we
can ask whether a plan survives the whole range we consider plausible. A plan
that clears at the point estimate and fails when enemy movement speed shifts by
10% was not a plan -- it was an exploit of a number nobody has measured yet.

Cheap enough to run on every plan before it is trusted, which is the intent:
robustness is a property to be checked routinely, not audited occasionally.
"""

from __future__ import annotations

import enum
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from ato.agent.actions import Plan
from ato.fidelity.divergence import Divergence, compare
from ato.fidelity.trace import BattleTrace
from ato.sim.engine import BattleEngine, EngineCalibration
from ato.train.randomize import sample_calibration


class Robustness(enum.StrEnum):
    ROBUST = "robust"                 # works across the plausible range
    FRAGILE = "fragile"               # works usually, fails at the edges
    SIM_DEPENDENT = "sim_dependent"   # works only near the point estimate
    BROKEN = "broken"                 # does not work even at the point estimate


@dataclass
class EnsembleVerdict:
    """What an ensemble of calibrations says about one plan."""

    stage_id: str
    reference_cleared: bool
    clear_rate: float
    n: int
    robustness: Robustness
    kill_l1_median: float | None
    kill_l1_worst: float | None
    life_lost_mean: float
    life_lost_worst: int
    divergences: list[Divergence] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.stage_id}: {self.robustness.value} "
            f"(reference {'cleared' if self.reference_cleared else 'FAILED'}, "
            f"{self.clear_rate:.0%} of {self.n} perturbed runs cleared, "
            f"kill_L1 median={self.kill_l1_median if self.kill_l1_median is None else round(self.kill_l1_median, 2)}s "
            f"worst={self.kill_l1_worst if self.kill_l1_worst is None else round(self.kill_l1_worst, 2)}s, "
            f"lives lost mean={self.life_lost_mean:.1f} worst={self.life_lost_worst})"
        )


#: A plan has to clear at least this share of the perturbed runs to count as
#: robust. Not 100%: the bands are deliberately wider than reality, so demanding
#: perfection would reject plans that are actually fine.
ROBUST_CLEAR_RATE = 0.85
FRAGILE_CLEAR_RATE = 0.5

EngineFactory = Callable[[EngineCalibration], BattleEngine]


def evaluate_plan(
    make_engine: EngineFactory,
    plan: Plan,
    *,
    n: int = 8,
    seed: int = 0,
    strength: float = 1.0,
    cards: list[str] | None = None,
) -> EnsembleVerdict:
    """Run ``plan`` once at the point estimate and ``n`` times under perturbation."""
    from ato.agent.execute import run_plan

    rng = random.Random(seed)
    base = EngineCalibration()

    ref_engine = make_engine(base)
    ref_trace, _ = run_plan(ref_engine, plan, cards=cards)

    divs: list[Divergence] = []
    traces: list[BattleTrace] = []
    for _ in range(n):
        cal = sample_calibration(rng, base, strength=strength)
        eng = make_engine(cal)
        trace, _ = run_plan(eng, plan, cards=cards)
        traces.append(trace)
        divs.append(compare(ref_trace, trace))

    cleared = [t.cleared for t in traces]
    rate = sum(cleared) / len(cleared) if cleared else 0.0
    l1s = [d.kill_time_l1 for d in divs if d.kill_time_l1 is not None]
    losses = [t.life_lost for t in traces]

    if not ref_trace.cleared:
        robustness = Robustness.BROKEN
    elif rate >= ROBUST_CLEAR_RATE:
        robustness = Robustness.ROBUST
    elif rate >= FRAGILE_CLEAR_RATE:
        robustness = Robustness.FRAGILE
    else:
        robustness = Robustness.SIM_DEPENDENT

    return EnsembleVerdict(
        stage_id=plan.stage_id or ref_trace.stage_id,
        reference_cleared=ref_trace.cleared,
        clear_rate=rate,
        n=len(traces),
        robustness=robustness,
        kill_l1_median=statistics.median(l1s) if l1s else None,
        kill_l1_worst=max(l1s) if l1s else None,
        life_lost_mean=statistics.fmean(losses) if losses else 0.0,
        life_lost_worst=max(losses) if losses else 0,
        divergences=divs,
    )


def sensitivity(
    make_engine: EngineFactory,
    plan: Plan,
    parameter: str,
    values: list[float],
    *,
    cards: list[str] | None = None,
) -> list[tuple[float, bool, int]]:
    """Sweep one calibration constant and report ``(value, cleared, lives lost)``.

    The ensemble says *whether* a plan is fragile; this says *to what*. When a
    plan fails only for low movement speed, that is a concrete instruction:
    measure that constant before believing anything else about the stage.
    """
    from dataclasses import replace

    from ato.agent.execute import run_plan

    out: list[tuple[float, bool, int]] = []
    for v in values:
        cal = replace(EngineCalibration(), **{parameter: v})
        trace, _ = run_plan(make_engine(cal), plan, cards=cards)
        out.append((v, trace.cleared, trace.life_lost))
    return out
