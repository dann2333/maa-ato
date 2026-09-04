"""Rollout-based planning — the system's first teacher.

Before there is any learned policy there has to be something that plays well
enough to imitate. Search is that something: the simulator is fast enough to
try a placement, play the battle out, and keep whichever choice actually
survives. That is a far better teacher than hand-written heuristics because it
is grounded in the same outcome the policy is eventually scored on.

The planner is deliberately *greedy over decision points* rather than a full
tree search. A battle has only a handful of real decisions (roughly one per
deployment), each rollout costs ~100 ms, and a greedy pass with a good candidate
set already clears ordinary stages. MCTS becomes worthwhile once the engine's
hot loop moves to a compiled backend; the interfaces here do not change when it
does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ato.agent.actions import Action, ActionKind, Plan, PlanStep, Trigger, TriggerKind
from ato.sim.engine import BattleEngine
from ato.sim.pathing import route_waypoints
from ato.sim.types import BattleResult, Direction, Tile


@dataclass
class MapHeuristics:
    """Static, per-map quantities that make the candidate set small and sensible."""

    traffic: dict[Tile, float] = field(default_factory=dict)
    #: Rough "how late on their route" a tile is: high means enemies pass here
    #: shortly before leaking, so holding it is worth more.
    lateness: dict[Tile, float] = field(default_factory=dict)

    @classmethod
    def build(cls, engine: BattleEngine) -> MapHeuristics:
        traffic: dict[Tile, float] = {}
        lateness: dict[Tile, float] = {}
        bmap = engine.sc.bmap
        # Weight a route by how many enemies actually use it, so a decorative
        # route with no spawns does not attract the whole squad.
        weight: dict[int, int] = {}
        for wave in engine.sc.waves:
            for frag in wave.fragments:
                for act in frag.actions:
                    if act.spawns:
                        weight[act.route_index] = weight.get(act.route_index, 0) + act.count
        for idx, route in engine.sc.routes.items():
            w = weight.get(idx, 0)
            if w <= 0:
                continue
            pts = route_waypoints(bmap, route)
            n = max(len(pts) - 1, 1)
            for i, p in enumerate(pts):
                t = p.tile
                traffic[t] = traffic.get(t, 0.0) + w
                lateness[t] = max(lateness.get(t, 0.0), i / n)
        return cls(traffic, lateness)

    def tile_value(self, t: Tile) -> float:
        return self.traffic.get(t, 0.0) * (0.5 + self.lateness.get(t, 0.0))


@dataclass
class Candidate:
    card_index: int
    char_id: str
    tile: Tile
    direction: Direction
    prior: float

    def to_action(self) -> Action:
        return Action(ActionKind.DEPLOY, card=self.card_index, tile=self.tile,
                      direction=self.direction)


def _coverage(engine: BattleEngine, char_id: str, tile: Tile, direction: Direction,
              heur: MapHeuristics) -> float:
    """How much enemy traffic this placement's attack range covers."""
    entry = engine.roster[char_id]
    arange = entry.attack_range
    if arange is None:
        return 0.0
    total = 0.0
    for dr, dc in arange.rotated(int(direction)):
        total += heur.traffic.get(Tile(tile.row + dr, tile.col + dc), 0.0)
    return total


def rank_placements(engine: BattleEngine, heur: MapHeuristics) -> list[Candidate]:
    """Every (card, tile, facing) placement, ranked by a static prior.

    Computed once per scenario+squad, not per decision. The prior depends only on
    the map's traffic and the operator's own range and block count -- none of
    which change during a battle -- so recomputing it inside every rollout was
    pure waste, and it was the planner's dominant cost.
    """
    cards = list(engine.roster)
    out: list[Candidate] = []
    for i, cid in enumerate(cards):
        entry = engine.roster[cid]
        blocks = entry.spec.stats.block_cnt > 0
        for tile in engine.sc.bmap.deployable_tiles(entry.spec.position):
            for direction in Direction:
                cov = _coverage(engine, cid, tile, direction, heur)
                if cov <= 0.0:
                    continue
                # A blocker earns its keep by standing *on* the path; a ranged
                # unit by covering it. Scoring them the same way puts snipers in
                # the enemy's way, which is how squads die.
                prior = cov + (heur.tile_value(tile) * 2.0 if blocks else 0.0)
                out.append(Candidate(i, cid, tile, direction, prior))
    out.sort(key=lambda c: -c.prior)
    return out


def candidates(
    engine: BattleEngine,
    heur: MapHeuristics,
    *,
    top_k: int = 12,
    ranked: list[Candidate] | None = None,
    per_card: int = 3,
) -> list[Candidate]:
    """The legal subset of the ranked placements, best first.

    Pruning is by prior only, never by hard rules that could remove the only
    winning move. Keeping a few placements *per card* rather than the global best
    stops one dominant operator from crowding every slot out of the set.
    """
    pool = ranked if ranked is not None else rank_placements(engine, heur)
    seen: dict[int, int] = {}
    kept: list[Candidate] = []
    for c in pool:
        n = seen.get(c.card_index, 0)
        if n >= per_card:
            continue
        if not engine.can_deploy(c.char_id, c.tile)[0]:
            continue
        seen[c.card_index] = n + 1
        kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


class PlacementIndex:
    """Ranked placements, grouped by card so legality can be checked in bulk.

    Scanning all ~400 placements and calling ``can_deploy`` on each was costing
    more than the simulation itself: a rollout re-checks its options a few
    hundred times, which turned into six-figure legality checks per rollout.
    Almost every rejection is decided at the *card* level -- unaffordable, still
    on cooldown, already on the field -- so checking that once skips the whole
    group.
    """

    __slots__ = ("by_card", "order")

    def __init__(self, ranked: list[Candidate]) -> None:
        by_card: dict[int, list[Candidate]] = {}
        for c in ranked:
            by_card.setdefault(c.card_index, []).append(c)
        self.by_card = by_card
        # Cards in order of their best placement, so the scan still follows the
        # global ranking.
        self.order = sorted(by_card, key=lambda i: -by_card[i][0].prior)

    def _card_ready(self, engine: BattleEngine, char_id: str) -> bool:
        entry = engine.roster.get(char_id)
        if entry is None or engine.state.time < entry.available_at:
            return False
        if engine.state.cost < entry.cost:
            return False
        if char_id in engine.sc.excluded_chars:
            return False
        return not any(o.spec and o.spec.char_id == char_id for o in engine.state.deployed)

    def first_legal(self, engine: BattleEngine) -> Candidate | None:
        if len(engine.state.deployed) >= engine.sc.options.character_limit:
            return None
        occupied = {o.tile for o in engine.state.deployed}
        for ci in self.order:
            group = self.by_card[ci]
            if not self._card_ready(engine, group[0].char_id):
                continue
            for c in group:
                if c.tile not in occupied:
                    return c
        return None

    def legal(self, engine: BattleEngine, *, top_k: int, per_card: int = 3) -> list[Candidate]:
        if len(engine.state.deployed) >= engine.sc.options.character_limit:
            return []
        occupied = {o.tile for o in engine.state.deployed}
        kept: list[Candidate] = []
        for ci in self.order:
            group = self.by_card[ci]
            if not self._card_ready(engine, group[0].char_id):
                continue
            n = 0
            for c in group:
                if c.tile in occupied:
                    continue
                kept.append(c)
                n += 1
                if n >= per_card:
                    break
        kept.sort(key=lambda c: -c.prior)
        return kept[:top_k]


def first_legal(engine: BattleEngine, ranked: list[Candidate] | PlacementIndex) -> Candidate | None:
    """The highest-prior placement that is legal right now, or None."""
    index = ranked if isinstance(ranked, PlacementIndex) else PlacementIndex(ranked)
    return index.first_legal(engine)


def score_state(engine: BattleEngine) -> float:
    """Rank an outcome. Ordered by what the game actually rewards.

    Life points dominate because losing them is what fails a stage and what
    costs the 3-star rating; kills and time only break ties. Deliberately *not*
    a dense hand-tuned reward — this is a comparison key for search, and the
    learned agent's reward lives in one place with its potential-based shaping
    (INVARIANT I-4).
    """
    st = engine.state
    if st.result is BattleResult.CLEARED:
        base = 1_000_000.0
    elif st.result is BattleResult.FAILED:
        base = 0.0
    else:
        base = 500_000.0
    return (
        base
        + st.life_points * 10_000.0
        + st.kills * 100.0
        - st.leaked * 5_000.0
        - st.time * 0.1
    )


def rollout(
    engine: BattleEngine,
    heur: MapHeuristics,
    index: PlacementIndex,
    *,
    horizon: float | None = None,
    defer: float = 0.0,
) -> float:
    """Play the battle out with a cheap default policy and score the result.

    The default policy is "deploy the highest-prior legal placement as soon as DP
    allows". It is weak on purpose: a rollout policy only has to be *consistent*
    for the comparison between two candidate moves to carry information.
    """
    sim = engine.clone()
    limit = horizon if horizon is not None else sim.max_seconds
    hold_until = sim.state.time + defer
    next_try = 0.0
    while sim.state.result is BattleResult.RUNNING and sim.state.time < limit:
        sim.tick()
        # Deploying only every half second keeps the rollout cheap and matches
        # the granularity a human plays at.
        if sim.state.time >= next_try and sim.state.time >= hold_until:
            next_try = sim.state.time + 0.5
            best = index.first_legal(sim)
            if best is not None:
                sim.deploy(best.char_id, best.tile, best.direction)
    return score_state(sim)


@dataclass
class PlannerConfig:
    #: How many candidate placements to evaluate with a rollout at each decision.
    branch: int = 8
    #: Seconds between decision points.
    decision_interval: float = 0.5
    #: Also evaluate "do nothing yet" — waiting for DP is often correct.
    consider_waiting: bool = True
    #: Stop planning after this many rollouts, so a stage cannot run forever.
    rollout_budget: int = 400
    max_seconds: float = 900.0


@dataclass
class PlanResult:
    plan: Plan
    score: float
    result: BattleResult
    rollouts_used: int
    summary: dict[str, object]


def plan_stage(engine: BattleEngine, cfg: PlannerConfig | None = None) -> PlanResult:
    """Greedy rollout planning over one battle.

    Returns a condition-triggered :class:`Plan`, not a timed script: each chosen
    action records the DP and kill count at which it fired, so the plan replays
    on a real device where frame timing differs.
    """
    cfg = cfg or PlannerConfig()
    heur = MapHeuristics.build(engine)
    index = PlacementIndex(rank_placements(engine, heur))
    live = engine
    steps: list[PlanStep] = []
    used = 0
    next_decision = 0.0
    cards = list(engine.roster)

    while live.state.result is BattleResult.RUNNING and live.state.time < cfg.max_seconds:
        live.tick()
        if live.state.time < next_decision:
            continue
        next_decision = live.state.time + cfg.decision_interval

        cands = index.legal(live, top_k=cfg.branch)
        if not cands:
            continue

        if used >= cfg.rollout_budget:
            # Out of search budget: fall back to the default policy rather than
            # standing still. A planner that stops deploying when it stops
            # thinking loses stages it had already half-won.
            fallback = index.first_legal(live)
            if fallback is not None:
                # Capture DP *before* deploying: the trigger is the level at
                # which the action becomes possible, and deploy has already
                # spent it by the time the unit exists.
                dp = float(int(live.state.cost))
                if live.deploy(fallback.char_id, fallback.tile, fallback.direction) is not None:
                    steps.append(
                        PlanStep(
                            action=fallback.to_action(),
                            trigger=Trigger(TriggerKind.COST, dp),
                            note=f"greedy fallback t={live.state.time:.1f}s",
                        )
                    )
            continue

        best_score = -math.inf
        best: Candidate | None = None
        if cfg.consider_waiting:
            # The waiting branch has to actually wait. Rolling out with the
            # default policy would deploy immediately anyway, making the
            # baseline indistinguishable from acting -- which is exactly the bug
            # that made an earlier version of this planner deploy nothing at all
            # and lose a stage its own rollout policy could clear.
            best_score = rollout(live, heur, index, defer=cfg.decision_interval)
            used += 1

        for c in cands:
            if used >= cfg.rollout_budget:
                break
            trial = live.clone()
            if trial.deploy(c.char_id, c.tile, c.direction) is None:
                continue
            s = rollout(trial, heur, index)
            used += 1
            if s > best_score:
                best_score, best = s, c

        if best is not None:
            # DP is the most reliable trigger a screen-reading agent has: a
            # large, high-contrast integer that moves deterministically. Read it
            # before the deploy spends it.
            dp = float(int(live.state.cost))
            kills = live.state.kills
            if live.deploy(best.char_id, best.tile, best.direction) is not None:
                steps.append(
                    PlanStep(
                        action=best.to_action(),
                        trigger=Trigger(TriggerKind.COST, dp),
                        note=f"t={live.state.time:.1f}s kills={kills}",
                    )
                )

    plan = Plan(steps=steps, stage_id=engine.sc.level_id, squad=tuple(cards))
    return PlanResult(
        plan=plan,
        score=score_state(live),
        result=live.state.result,
        rollouts_used=used,
        summary=live.summary(),
    )
