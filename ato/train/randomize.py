"""Domain randomisation: the mechanism behind three invariants at once.

* **I-2** — investment is randomised, so the same operator appears with wildly
  different stats across episodes. A network that memorised "deploy this face
  here" is wrong most of the time; one that reads the stat line is not.
* **I-5** — the simulator's fitted constants are randomised inside their
  uncertainty bands, so a policy cannot come to depend on a value we are not
  sure of. If performance collapses when ``move_tiles_per_second`` moves 10%,
  the policy was exploiting the simulator, and we want to find that out during
  training rather than on a real device.
* **I-6** — scenario perturbation is a *whitelist of named operators* with a
  solvability filter, not free rewriting. An unconstrained level generator
  drifts to unsolvable or trivial stages, and a policy trained on either learns
  nothing useful.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Callable

from ato.gamedata.models import OperatorSpec, resolve_operator, resolve_skill
from ato.gamedata.tables import GameData
from ato.sim.engine import EngineCalibration, RosterEntry
from ato.sim.scenario import Scenario, Wave, WaveAction, WaveFragment


@dataclass(frozen=True)
class InvestmentRange:
    """The span of account progress an agent must cope with.

    Defaults span "a new player's box" to "a fully built account", because that
    whole span is what a general agent has to handle. Narrowing it is a
    deliberate choice, not a default.
    """

    phases: tuple[int, ...] = (0, 1, 2)
    level_fraction: tuple[float, float] = (0.3, 1.0)
    trust: tuple[int, int] = (0, 200)
    potential: tuple[int, int] = (0, 5)
    mastery: tuple[int, int] = (0, 9)
    #: Probability of picking skill 1 / 2 / 3 rather than always the first.
    random_skill: bool = True


def sample_operator(
    gd: GameData, char_id: str, rng: random.Random, rng_range: InvestmentRange
) -> RosterEntry | None:
    """Resolve one operator at a randomly sampled investment level."""
    char = gd.characters.get(char_id)
    if char is None:
        return None
    phase = rng.choice(rng_range.phases)
    phases = char.get("phases") or []
    if not phases:
        return None
    phase = min(phase, len(phases) - 1)
    max_level = phases[phase].get("maxLevel", 1)
    frac = rng.uniform(*rng_range.level_fraction)
    level = max(1, min(int(round(max_level * frac)), max_level))
    spec: OperatorSpec = resolve_operator(
        char_id, char,
        phase=phase,
        level=level,
        trust=rng.randint(*rng_range.trust),
        potential=rng.randint(*rng_range.potential),
    )
    skill = None
    if spec.skill_ids:
        # Higher elite phases unlock later skills; picking one the operator
        # cannot have would train the policy on a unit that does not exist.
        usable = spec.skill_ids[: 1 + phase] or spec.skill_ids[:1]
        sid = rng.choice(usable) if rng_range.random_skill else usable[0]
        raw = gd.skills.get(sid)
        if raw:
            skill = resolve_skill(sid, raw, rng.randint(*rng_range.mastery))
    return RosterEntry(spec=spec, skill=skill)


def sample_roster(
    gd: GameData,
    pool: Sequence[str],
    rng: random.Random,
    *,
    size: int = 8,
    investment: InvestmentRange | None = None,
) -> list[RosterEntry]:
    """Draw a squad from a pool, each member at its own investment level."""
    inv = investment or InvestmentRange()
    picks = rng.sample(list(pool), k=min(size, len(pool)))
    out = [sample_operator(gd, cid, rng, inv) for cid in picks]
    return [e for e in out if e is not None]


#: Multiplicative bands around each calibration default, expressing how sure we
#: actually are. ``move_tiles_per_second`` is the widest because it is the least
#: constrained by evidence -- see docs/SIM_SPEC.md B-1.
CALIBRATION_BANDS: dict[str, tuple[float, float]] = {
    "move_tiles_per_second": (0.80, 1.25),
    "deploy_lock_seconds": (0.70, 1.30),
    "redeploy_multiplier": (0.95, 1.05),
    "sp_per_second": (0.95, 1.05),
    "sp_per_attack": (0.90, 1.10),
    "sp_per_hit_taken": (0.90, 1.10),
    # The floor is disputed between sources (10 vs 20) and the ceiling has no
    # source at all, so the band spans the disagreement rather than picking a
    # winner. Base 20 x [0.5, 1.0] covers both reported floors.
    "aspd_min": (0.50, 1.00),
    "aspd_max": (0.83, 1.17),
}

#: Calibration unknowns that are *discrete*, so they are sampled as one of a few
#: readings rather than smeared over a range. A policy that only works under one
#: reading of an unmeasured rule has learned the reading, not the game.
CALIBRATION_CHOICES: dict[str, tuple[object, ...]] = {
    # Whether a fixed-step client quantises each attack interval to whole
    # frames, and how it rounds. Worth ~4% DPS at baseAttackTime 0.78.
    "attack_quantization": ("none", "ceil", "round"),
    # How often the client rescans for targets. Reported as every third frame;
    # nothing measures it.
    "target_scan_period_ticks": (1, 2, 3),
}


def sample_calibration(
    rng: random.Random, base: EngineCalibration | None = None, *, strength: float = 1.0
) -> EngineCalibration:
    """Perturb the fitted constants within their uncertainty bands.

    ``strength`` scales the bands: 0 reproduces the point estimate (for
    evaluation), 1 uses the full band (for training).
    """
    cal = base or EngineCalibration()
    kw: dict[str, object] = {}
    for name, (lo, hi) in CALIBRATION_BANDS.items():
        cur = getattr(cal, name)
        lo_s = 1.0 + (lo - 1.0) * strength
        hi_s = 1.0 + (hi - 1.0) * strength
        kw[name] = cur * rng.uniform(lo_s, hi_s)
    if kw.get("aspd_min", cal.aspd_min) >= kw.get("aspd_max", cal.aspd_max):
        # Two independent draws can cross; a clamp with its ends swapped is not
        # a hypothesis about the game, it is a broken simulator.
        kw["aspd_min"], kw["aspd_max"] = cal.aspd_min, cal.aspd_max
    # The remaining unknowns are discrete, so they are sampled as one reading
    # rather than smeared: the policy should be robust to either, not tuned to
    # the midpoint of two things neither of which the game does.
    if strength > 0.0:
        if rng.random() < 0.25 * strength:
            kw["fragment_relative_delays"] = not cal.fragment_relative_delays
        for name, options in CALIBRATION_CHOICES.items():
            if rng.random() < strength:
                kw[name] = rng.choice(options)
    return replace(cal, **kw)


# ---------------------------------------------------------------------------
# Bounded scenario perturbation (INVARIANT I-6)
# ---------------------------------------------------------------------------

Perturbation = Callable[[Scenario, random.Random, float], Scenario]

#: Registered perturbation operators. Nothing else may touch a scenario. Adding
#: one is a deliberate act with a review, which is what keeps the auto-curriculum
#: from wandering into stages that teach the wrong thing.
PERTURBATIONS: dict[str, Perturbation] = {}


def perturbation(name: str) -> Callable[[Perturbation], Perturbation]:
    def deco(fn: Perturbation) -> Perturbation:
        PERTURBATIONS[name] = fn
        return fn

    return deco


def _map_actions(sc: Scenario, fn: Callable[[WaveAction], WaveAction]) -> Scenario:
    waves = tuple(
        Wave(
            pre_delay=w.pre_delay,
            post_delay=w.post_delay,
            max_wait_for_next=w.max_wait_for_next,
            fragments=tuple(
                WaveFragment(f.pre_delay, tuple(fn(a) for a in f.actions))
                for f in w.fragments
            ),
        )
        for w in sc.waves
    )
    return replace(sc, waves=waves)


@perturbation("wave_density")
def perturb_wave_density(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Tighten or loosen the interval between spawns in a wave."""
    factor = 1.0 + rng.uniform(-0.35, 0.35) * strength
    return _map_actions(
        sc, lambda a: replace(a, interval=max(a.interval * factor, 0.1)) if a.spawns else a
    )


@perturbation("enemy_count")
def perturb_enemy_count(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Add or remove enemies from spawn actions, never below one."""
    delta = rng.uniform(-0.25, 0.25) * strength
    return _map_actions(
        sc,
        lambda a: replace(a, count=max(1, int(round(a.count * (1.0 + delta)))))
        if a.spawns else a,
    )


@perturbation("initial_cost")
def perturb_initial_cost(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Shift the starting DP, which changes the whole opening."""
    opt = sc.options
    shift = int(round(rng.uniform(-6, 6) * strength))
    return replace(sc, options=replace(opt, initial_cost=max(0, opt.initial_cost + shift)))


@perturbation("life_points")
def perturb_life_points(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Change the margin for error."""
    opt = sc.options
    shift = int(round(rng.uniform(-2, 2) * strength))
    return replace(sc, options=replace(opt, max_life_point=max(1, opt.max_life_point + shift)))


@perturbation("squad_limit")
def perturb_squad_limit(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Restrict how many operators may be on the field."""
    opt = sc.options
    shift = int(round(rng.uniform(-2, 1) * strength))
    return replace(
        sc, options=replace(opt, character_limit=max(2, opt.character_limit + shift))
    )


@perturbation("cost_rate")
def perturb_cost_rate(sc: Scenario, rng: random.Random, strength: float) -> Scenario:
    """Speed up or slow down DP generation."""
    opt = sc.options
    factor = 1.0 + rng.uniform(-0.25, 0.25) * strength
    return replace(
        sc, options=replace(opt, cost_increase_time=max(0.2, opt.cost_increase_time * factor))
    )


def perturb(
    sc: Scenario,
    rng: random.Random,
    *,
    strength: float = 1.0,
    n: int = 2,
    allowed: Sequence[str] | None = None,
) -> tuple[Scenario, tuple[str, ...]]:
    """Apply ``n`` distinct registered perturbations. Returns the names applied."""
    names = list(allowed if allowed is not None else PERTURBATIONS)
    rng.shuffle(names)
    applied: list[str] = []
    out = sc
    for name in names[:n]:
        fn = PERTURBATIONS.get(name)
        if fn is None:
            continue
        out = fn(out, rng, strength)
        applied.append(name)
    return out, tuple(applied)


@dataclass(frozen=True)
class SolvabilityVerdict:
    solvable: bool
    trivial: bool
    score: float
    reason: str = ""


def check_solvable(
    make_engine: Callable[[Scenario], object],
    sc: Scenario,
    *,
    attempts: int = 2,
) -> SolvabilityVerdict:
    """Reject generated stages that teach nothing.

    A stage nobody can clear teaches the policy to give up; one that clears
    itself teaches nothing at all. Both are filtered out before the stage can
    enter the training distribution (INVARIANT I-6). The check runs the default
    rollout policy, so "trivial" means trivial *for a weak baseline* — exactly
    the bar a training stage should clear.
    """
    from ato.agent.search import MapHeuristics, PlacementIndex, rank_placements, rollout
    from ato.sim.types import BattleResult

    best = -1e18
    cleared = 0
    for _ in range(max(attempts, 1)):
        engine = make_engine(sc)
        heur = MapHeuristics.build(engine)  # type: ignore[arg-type]
        index = PlacementIndex(rank_placements(engine, heur))  # type: ignore[arg-type]
        probe = engine.clone()  # type: ignore[attr-defined]
        score = rollout(probe, heur, index)  # type: ignore[arg-type]
        best = max(best, score)
        after = probe.state.result if hasattr(probe, "state") else None
        if after is BattleResult.CLEARED:
            cleared += 1
    if cleared == 0:
        return SolvabilityVerdict(False, False, best, "baseline never cleared it")
    if cleared == max(attempts, 1):
        return SolvabilityVerdict(True, True, best, "baseline clears it every time")
    return SolvabilityVerdict(True, False, best, "")
