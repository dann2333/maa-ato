"""Executing a :class:`~ato.agent.actions.Plan`.

One executor, two backends. In the simulator it calls engine methods; on a
device it issues gestures. The trigger logic -- "fire when DP reaches 37" -- is
shared, which is the point: a plan that was validated in the simulator replays
on hardware without being rewritten, and any disagreement is a fidelity问题
rather than a difference between two implementations of the same idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ato.agent.actions import ActionKind, Plan, PlanStep, TriggerKind
from ato.sim.engine import BattleEngine


@dataclass
class ExecutionState:
    """Where the executor is in the plan, and why it is waiting."""

    index: int = 0
    fired: list[tuple[float, int]] = field(default_factory=list)
    #: Steps whose trigger fired but whose action was rejected as illegal.
    blocked: list[tuple[float, int, str]] = field(default_factory=list)
    #: DP at the previous observation, so COST_DROP triggers can be detected.
    last_cost: float | None = None
    #: Set once the trigger fires, so ``pre_delay`` can be honoured afterwards.
    armed_at: float | None = None

    @property
    def done(self) -> bool:
        return self.index < 0


class PlanExecutor:
    """Fires plan steps against a simulated battle as their triggers come true.

    Strictly in order. Copilot plans and search output are both sequences whose
    later steps assume the earlier ones happened, so skipping ahead when a step
    is momentarily illegal would silently execute a different plan. A step that
    cannot run is retried until it can.
    """

    def __init__(self, plan: Plan, *, cards: list[str] | None = None) -> None:
        self.plan = plan
        self.cards = cards if cards is not None else list(plan.squad)
        self.state = ExecutionState()

    def _current(self) -> PlanStep | None:
        i = self.state.index
        return self.plan.steps[i] if 0 <= i < len(self.plan.steps) else None

    def step(self, engine: BattleEngine, recorder=None) -> bool:
        """Advance the executor one tick's worth. Returns whether it acted."""
        st = self.state
        item = self._current()
        if item is None:
            return False

        cost = engine.state.cost
        drop = 0.0 if st.last_cost is None else max(st.last_cost - cost, 0.0)
        st.last_cost = cost

        if st.armed_at is None:
            if not item.trigger.satisfied(
                kills=engine.state.kills,
                cost=cost,
                elapsed=engine.state.time,
                cost_drop=drop,
            ):
                return False
            st.armed_at = engine.state.time
        if engine.state.time < st.armed_at + item.pre_delay:
            return False

        acted, why = self._apply(engine, item, recorder)
        if not acted:
            # Keep waiting rather than dropping the step: "not enough DP yet" and
            # "tile occupied for now" both resolve on their own, and a plan with
            # a hole in it is a different plan.
            st.blocked.append((engine.state.time, st.index, why))
            return False
        st.fired.append((engine.state.time, st.index))
        st.index += 1
        st.armed_at = None
        if st.index >= len(self.plan.steps):
            st.index = -1
        return True

    def _apply(self, engine: BattleEngine, item: PlanStep, recorder) -> tuple[bool, str]:
        a = item.action
        if a.kind is ActionKind.WAIT:
            return True, ""
        if a.kind is ActionKind.DEPLOY:
            if a.tile is None or not (0 <= a.card < len(self.cards)):
                return False, "malformed deploy step"
            char_id = self.cards[a.card]
            ok, why = engine.can_deploy(char_id, a.tile)
            if not ok:
                return False, why
            unit = engine.deploy(char_id, a.tile, a.direction)
            if unit is None:
                return False, "deploy rejected"
            if recorder is not None:
                recorder.note_deploy(char_id, a.tile.row, a.tile.col, int(a.direction))
            return True, ""
        if a.kind is ActionKind.RETREAT:
            unit = self._resolve_unit(engine, a.unit, a.card)
            if unit is None:
                return False, "no such deployed operator"
            if recorder is not None and unit.spec:
                recorder.note_retreat(unit.spec.char_id)
            return engine.retreat(unit.uid), "retreat rejected"
        if a.kind is ActionKind.SKILL:
            unit = self._resolve_unit(engine, a.unit, a.card)
            if unit is None:
                return False, "no such deployed operator"
            if not engine.use_skill(unit.uid):
                return False, "skill not ready"
            if recorder is not None and unit.spec:
                recorder.note_skill(unit.spec.char_id)
            return True, ""
        return True, ""

    def _resolve_unit(self, engine: BattleEngine, uid: int, card: int):
        """Plans refer to operators by card, not by the uid of a past deployment."""
        if uid >= 0:
            unit = engine._units.get(uid)
            return unit if unit is not None and unit in engine.state.deployed else None
        if 0 <= card < len(self.cards):
            want = self.cards[card]
            for op in engine.state.deployed:
                if op.spec and op.spec.char_id == want:
                    return op
        return None

    def as_callback(self):
        """Adapt to the ``plan_apply(engine, recorder)`` shape used by tracing."""

        def _cb(engine: BattleEngine, recorder=None) -> None:
            self.step(engine, recorder)

        return _cb


def run_plan(engine: BattleEngine, plan: Plan, *, cards: list[str] | None = None):
    """Run a plan to completion in the simulator, returning ``(trace, executor)``."""
    from ato.fidelity.trace import run_and_trace

    ex = PlanExecutor(plan, cards=cards if cards is not None else list(engine.roster))
    trace = run_and_trace(engine, ex.as_callback())
    return trace, ex
