"""The action space.

Shaped so that a policy generalises across squads and maps (INVARIANT I-2/I-7):

* No fixed "operator slot" head. A deployment is a *pointer* into the cards that
  are actually available now, crossed with a pointer into the tiles that are
  actually legal now. Swap the squad and the head does not change shape.
* Actions are emitted as *intents* with a trigger condition rather than as
  timestamps. A plan that says "deploy the defender once DP reaches 23" survives
  frame jitter, 2x speed and a paused game; a plan that says "at t=13.0s" does not.
  This is also the abstraction MAA copilot files use, which is not a coincidence —
  it is what makes a plan robust enough for a human to write down.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ato.sim.types import Direction, Tile

if TYPE_CHECKING:
    from ato.sim.engine import BattleEngine


class ActionKind(enum.IntEnum):
    WAIT = 0
    DEPLOY = 1
    RETREAT = 2
    SKILL = 3
    SPEED = 4


@dataclass(frozen=True, slots=True)
class Action:
    """One decision. ``WAIT`` is a first-class choice, not the absence of one."""

    kind: ActionKind = ActionKind.WAIT
    #: Index into the *currently available* card list, not a global operator id.
    card: int = -1
    tile: Tile | None = None
    direction: Direction = Direction.RIGHT
    #: Index into the *currently deployed* operator list, for RETREAT/SKILL.
    unit: int = -1
    speed: int = 1

    def describe(self, cards: list[str] | None = None) -> str:
        if self.kind is ActionKind.DEPLOY:
            who = cards[self.card] if cards and 0 <= self.card < len(cards) else f"card{self.card}"
            return f"deploy {who} at ({self.tile.row},{self.tile.col}) facing {self.direction.name}"
        if self.kind is ActionKind.RETREAT:
            return f"retreat unit#{self.unit}"
        if self.kind is ActionKind.SKILL:
            return f"skill unit#{self.unit}"
        if self.kind is ActionKind.SPEED:
            return f"speed x{self.speed}"
        return "wait"


class TriggerKind(enum.IntEnum):
    """What makes a planned action fire. Ordered by how robust it is on a real device."""

    KILLS = 0        # kill counter reached N -- read straight off the screen
    COST = 1         # DP reached N -- the most reliable clock available
    COST_DROP = 2    # DP fell by N (i.e. someone else deployed)
    ELAPSED = 3      # seconds since battle start -- least robust, use last
    IMMEDIATE = 4    # as soon as legal


@dataclass(frozen=True, slots=True)
class Trigger:
    kind: TriggerKind = TriggerKind.IMMEDIATE
    value: float = 0.0

    def satisfied(self, *, kills: int, cost: float, elapsed: float, cost_drop: float) -> bool:
        if self.kind is TriggerKind.IMMEDIATE:
            return True
        if self.kind is TriggerKind.KILLS:
            return kills >= self.value
        if self.kind is TriggerKind.COST:
            return cost >= self.value
        if self.kind is TriggerKind.COST_DROP:
            return cost_drop >= self.value
        return elapsed >= self.value


@dataclass(frozen=True, slots=True)
class PlanStep:
    """An action plus the condition that releases it."""

    action: Action
    trigger: Trigger = Trigger()
    #: Extra seconds to wait after the trigger fires. Copilot files use this to
    #: express "a beat after the kill", which is often what makes a plan work.
    pre_delay: float = 0.0
    note: str = ""


@dataclass
class Plan:
    """An ordered, condition-triggered script. The output of search and of the
    copilot importer alike, so both feed the same executor."""

    steps: list[PlanStep] = field(default_factory=list)
    stage_id: str = ""
    #: Operator char ids the plan assumes, in card order.
    squad: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> str:
        return "\n".join(
            f"{i:2d}. [{s.trigger.kind.name}>={s.trigger.value:g}] {s.action.describe()}"
            + (f"  # {s.note}" if s.note else "")
            for i, s in enumerate(self.steps)
        )


def legal_deployments(
    engine: BattleEngine, *, cards: list[str] | None = None
) -> list[tuple[int, str, Tile]]:
    """Enumerate (card index, char id, tile) triples that are legal right now.

    This is the pointer network's candidate set. It is recomputed every decision
    point, which is exactly why the policy never needs a fixed action head.
    """
    ids = cards if cards is not None else list(engine.roster)
    out: list[tuple[int, str, Tile]] = []
    for i, cid in enumerate(ids):
        entry = engine.roster.get(cid)
        if entry is None:
            continue
        for tile in engine.sc.bmap.deployable_tiles(entry.spec.position):
            if engine.can_deploy(cid, tile)[0]:
                out.append((i, cid, tile))
    return out


def legal_unit_actions(engine: BattleEngine) -> tuple[list[int], list[int]]:
    """Returns ``(retreatable uids, skill-ready uids)`` among deployed operators."""
    retreat: list[int] = []
    skill: list[int] = []
    for op in engine.state.deployed:
        retreat.append(op.uid)
        if op.skill is not None and not op.skill.is_auto and op.sp_ready() and not op.skill_active:
            skill.append(op.uid)
    return retreat, skill


def apply(engine: BattleEngine, action: Action, cards: list[str] | None = None) -> bool:
    """Execute one action against the engine. Returns whether it took effect."""
    if action.kind is ActionKind.WAIT:
        return True
    ids = cards if cards is not None else list(engine.roster)
    if action.kind is ActionKind.DEPLOY:
        if not (0 <= action.card < len(ids)) or action.tile is None:
            return False
        return engine.deploy(ids[action.card], action.tile, action.direction) is not None
    if action.kind is ActionKind.RETREAT:
        return engine.retreat(action.unit)
    if action.kind is ActionKind.SKILL:
        return engine.use_skill(action.unit)
    return True
