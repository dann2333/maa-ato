"""Reward. One file, on purpose (INVARIANT I-4).

The terminal reward is the game's own verdict: cleared, and how much of the
squad's life it cost. Everything else is *potential-based shaping* in the sense
of Ng, Harada & Russell (1999): the per-step bonus is exactly
``gamma * phi(s') - phi(s)`` for a state-only potential ``phi``.

That form is not decoration. Its sum along any trajectory telescopes to
``gamma^T * phi(s_T) - phi(s_0)`` — it depends only on the endpoints, never on
the route taken — so no policy can farm it, and the optimal policy is provably
unchanged. Any dense reward that is *not* of this form is an invitation to the
reward hacking the project is explicitly trying to avoid, which is why
:func:`assert_potential_based` exists and is exercised by the test suite.

The potential may read privileged simulator state: reward and critic are exactly
where privileged information belongs (INVARIANT I-3). The policy never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ato.sim.engine import BattleEngine
from ato.sim.types import BattleResult

#: Weights of the potential. These change how fast learning goes, never what it
#: converges to -- which is the entire point of using this form.
LIFE_WEIGHT = 1.0
KILL_WEIGHT = 0.05
THREAT_WEIGHT = 0.35

#: How quickly an approaching enemy becomes scary. An enemy five seconds from
#: the blue box is worth reacting to; one sixty seconds out is not.
URGENCY_TAU = 8.0


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.999
    clear_bonus: float = 10.0
    fail_penalty: float = -10.0
    #: Clearing without losing a single life point is the 3-star condition on
    #: most stages, and it is what separates "passed" from "played well".
    perfect_bonus: float = 5.0
    timeout_penalty: float = -10.0
    shaping: bool = True


def potential(engine: BattleEngine) -> float:
    """State-only potential. Higher is better.

    Three terms, each answering a question the agent actually faces: how much
    margin is left (life), how much of the job is done (kills), and how much
    danger is currently on the field weighted by how soon it arrives (threat).
    """
    st = engine.state
    threat = 0.0
    scale = (
        engine.sc.options.move_multiplier * engine.cal.move_tiles_per_second
    )
    goal = engine.sc.bmap.ends[0] if engine.sc.bmap.ends else None
    for e in st.active_enemies:
        speed = e.stats.move_speed * scale
        if goal is None or speed <= 0.0:
            seconds = 0.0
        else:
            # Straight-line distance is a deliberate approximation: the exact
            # remaining path is available but costs a walk of the route program
            # per enemy per step, and the potential only needs a monotone
            # ordering of urgency.
            seconds = ((goal.xy - e.position).length()) / speed
        urgency = URGENCY_TAU / (URGENCY_TAU + max(seconds, 0.0))
        threat += e.life_point_reduce * e.hp_ratio * urgency
    return (
        LIFE_WEIGHT * st.life_points
        + KILL_WEIGHT * st.kills
        - THREAT_WEIGHT * threat
    )


def terminal_reward(engine: BattleEngine, cfg: RewardConfig) -> float:
    st = engine.state
    if st.result is BattleResult.CLEARED:
        r = cfg.clear_bonus
        if st.life_points >= engine.sc.options.max_life_point:
            r += cfg.perfect_bonus
        return r
    if st.result is BattleResult.FAILED:
        return cfg.fail_penalty
    if st.result is BattleResult.TIMEOUT:
        return cfg.timeout_penalty
    return 0.0


def shaping_term(phi_before: float, phi_after: float, cfg: RewardConfig) -> float:
    """``gamma * phi(s') - phi(s)``. The only permitted dense reward."""
    return cfg.gamma * phi_after - phi_before


class RewardTracker:
    """Accumulates reward across a battle, one step at a time."""

    def __init__(self, engine: BattleEngine, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg or RewardConfig()
        self.engine = engine
        self.phi = potential(engine) if self.cfg.shaping else 0.0
        self.total = 0.0
        self.shaped_total = 0.0

    def step(self) -> float:
        """Reward for the transition that just happened."""
        r = 0.0
        if self.cfg.shaping:
            phi_now = potential(self.engine)
            s = shaping_term(self.phi, phi_now, self.cfg)
            self.phi = phi_now
            self.shaped_total += s
            r += s
        if self.engine.state.result is not BattleResult.RUNNING:
            r += terminal_reward(self.engine, self.cfg)
        self.total += r
        return r


def assert_potential_based(
    phis: list[float], rewards: list[float], cfg: RewardConfig, *, tol: float = 1e-6
) -> None:
    """Check that a recorded shaping sequence telescopes.

    Given the potentials visited and the shaping rewards emitted, the sum must
    equal ``gamma^T * phi_T - phi_0`` up to floating point. A regression that
    quietly adds a path-dependent bonus fails here rather than surfacing months
    later as a policy that farms an intermediate quantity.
    """
    if len(phis) < 2:
        return
    total = sum(rewards)
    expected = 0.0
    g = 1.0
    for a, b in zip(phis, phis[1:]):
        expected += g * (cfg.gamma * b - a)
        g *= cfg.gamma
    if abs(total - expected) > tol * max(1.0, abs(expected)):
        raise AssertionError(
            f"shaping is not potential-based: sum={total!r} expected={expected!r}"
        )
