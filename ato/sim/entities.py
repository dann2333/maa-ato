"""Units on the battlefield: their state and the rules that are purely local.

Anything that needs to see the *world* (target selection, blocking assignment,
damage application) lives in :mod:`ato.sim.engine`; entities own only what they
can decide alone. Keeping that split makes both halves testable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from itertools import count

from ato.gamedata.models import AttackRange, EnemySpec, OperatorSpec, SkillSpec
from ato.sim.combat import Modifier, StatBlock, StatusKind
from ato.sim.pathing import DisappearStep, MoveStep, RouteStep, TeleportStep, WaitStep
from ato.sim.types import DamageType, Direction, Side, Tile, Vec2

_ids = count(1)


class UnitState(enum.IntEnum):
    #: Placed but still playing its deployment animation — cannot act or be hit.
    DEPLOYING = 0
    ACTIVE = 1
    #: Enemy has left the field via a DISAPPEAR checkpoint.
    HIDDEN = 2
    DEAD = 3
    #: Enemy reached the blue box.
    LEAKED = 4
    #: Operator retreated and is waiting out its redeploy timer.
    RETREATED = 5


@dataclass
class Unit:
    """Shared state. ``hp`` is authoritative; ``stats.max_hp`` may change under buffs."""

    spec_name: str
    side: Side
    stats: StatBlock
    hp: float
    position: Vec2
    state: UnitState = UnitState.ACTIVE
    uid: int = field(default_factory=lambda: next(_ids))
    #: When this unit may next attack, in simulation seconds.
    next_attack_at: float = 0.0
    target_uid: int | None = None

    @property
    def alive(self) -> bool:
        return self.state in (UnitState.ACTIVE, UnitState.DEPLOYING, UnitState.HIDDEN)

    @property
    def targetable(self) -> bool:
        return self.state is UnitState.ACTIVE

    @property
    def hp_ratio(self) -> float:
        mx = self.stats.max_hp
        return 0.0 if mx <= 0 else max(0.0, min(1.0, self.hp / mx))

    def heal(self, amount: float) -> float:
        before = self.hp
        self.hp = min(self.hp + amount, self.stats.max_hp)
        return self.hp - before

    def take_damage(self, amount: float) -> tuple[float, bool]:
        """Apply damage. Returns ``(dealt, lethal)``; overkill is not counted as dealt."""
        if not self.targetable or amount <= 0.0:
            return 0.0, False
        dealt = min(amount, self.hp)
        self.hp -= amount
        if self.hp <= 0.0:
            self.hp = 0.0
            return dealt, True
        return dealt, False


@dataclass
class EnemyUnit(Unit):
    """An enemy executing a route program."""

    spec: EnemySpec | None = None
    damage_type: DamageType = DamageType.PHYSICAL
    program: tuple[RouteStep, ...] = ()
    #: Index into ``program``; ``point_index`` walks the current MoveStep.
    step_index: int = 0
    point_index: int = 0
    #: While > current time, the unit is holding at a WaitStep.
    wait_until: float = 0.0
    blocked_by: int | None = None
    life_point_reduce: int = 1
    route_index: int = 0
    #: Distance already travelled, used for "closest to the goal" targeting.
    progress: float = 0.0
    spawn_time: float = 0.0

    @property
    def is_flying(self) -> bool:
        return bool(self.spec and self.spec.is_flying)

    @property
    def is_ranged(self) -> bool:
        return bool(self.spec and self.spec.apply_way.upper() in ("RANGED", "ALL"))

    @property
    def attack_radius(self) -> float:
        return float(self.spec.range_radius) if self.spec else 0.0

    @property
    def blocked(self) -> bool:
        return self.blocked_by is not None

    def current_step(self) -> RouteStep | None:
        return self.program[self.step_index] if self.step_index < len(self.program) else None

    def advance(self, dt: float, now: float, speed_tiles_per_sec: float) -> bool:
        """Move along the program for ``dt`` seconds.

        Returns True when the program is exhausted, i.e. the unit reached the
        blue box. A blocked, stunned or waiting unit does not move but still
        consumes the tick — the caller decides what else it does.
        """
        if self.blocked or not self.stats.can_move:
            return False
        budget = dt * speed_tiles_per_sec
        while budget > 0.0:
            step = self.current_step()
            if step is None:
                return True
            if isinstance(step, WaitStep):
                if step.seconds >= 0.0:
                    if self.wait_until == 0.0:
                        self.wait_until = now + step.seconds
                    if now < self.wait_until:
                        return False
                    self.wait_until = 0.0
                    self.step_index += 1
                    continue
                # Engine-released wait: held until something clears wait_until.
                if self.wait_until > 0.0:
                    return False
                self.step_index += 1
                continue
            if isinstance(step, TeleportStep):
                self.position = step.position
                self.state = UnitState.ACTIVE
                self.step_index += 1
                continue
            if isinstance(step, DisappearStep):
                self.state = UnitState.HIDDEN
                self.step_index += 1
                continue
            assert isinstance(step, MoveStep)
            if self.point_index >= len(step.points):
                self.step_index += 1
                self.point_index = 0
                continue
            target = step.points[self.point_index]
            delta = target - self.position
            dist = delta.length()
            if dist <= budget:
                self.position = target
                self.progress += dist
                budget -= dist
                self.point_index += 1
            else:
                self.position = self.position + delta.normalized() * budget
                self.progress += budget
                budget = 0.0
        return False


@dataclass
class OperatorUnit(Unit):
    """A deployed operator."""

    spec: OperatorSpec | None = None
    skill: SkillSpec | None = None
    attack_range: AttackRange | None = None
    tile: Tile = Tile(0, 0)
    direction: Direction = Direction.RIGHT
    damage_type: DamageType = DamageType.PHYSICAL
    #: Enemies currently blocked, in the order they were blocked. The game
    #: releases block slots in arrival order, so a list, not a set.
    blocking: list[int] = field(default_factory=list)
    sp: float = 0.0
    skill_active_until: float = 0.0
    #: Set while the skill is running so effect modifiers can be removed cleanly.
    skill_modifier_source: str = ""
    deploy_time: float = 0.0
    #: When a retreated operator becomes redeployable.
    redeploy_at: float = 0.0
    #: How many times this operator has been deployed (redeploy costs rise).
    deploy_count: int = 0
    #: Covered tiles cached from range + facing; invalidated on rotate.
    _covered: tuple[Tile, ...] = ()

    @property
    def is_melee(self) -> bool:
        return self.stats.block_capacity > 0

    @property
    def skill_active(self) -> bool:
        return self.skill_active_until > 0.0

    @property
    def block_free(self) -> int:
        return max(self.stats.block_capacity - len(self.blocking), 0)

    def covered_tiles(self) -> tuple[Tile, ...]:
        """Tiles this operator can attack, given its range and facing."""
        if not self._covered and self.attack_range is not None:
            self._covered = tuple(
                Tile(self.tile.row + dr, self.tile.col + dc)
                for dr, dc in self.attack_range.rotated(int(self.direction))
            )
        return self._covered

    def face(self, direction: Direction) -> None:
        if direction != self.direction:
            self.direction = direction
            self._covered = ()

    def place(self, tile: Tile, direction: Direction) -> None:
        self.tile = tile
        self.position = tile.xy
        self.direction = direction
        self._covered = ()

    # -- skill -----------------------------------------------------------

    def sp_ready(self) -> bool:
        return self.skill is not None and self.sp >= self.skill.sp_cost

    def charge_sp(self, amount: float) -> None:
        """SP does not accumulate past the cost, and is frozen while the skill runs."""
        if self.skill is None or self.skill_active or self.stats.silenced:
            return
        self.sp = min(self.sp + amount, self.skill.sp_cost)

    def start_skill(self, now: float) -> bool:
        if self.skill is None or not self.sp_ready() or self.skill_active:
            return False
        if self.stats.silenced or not self.stats.can_act:
            return False
        self.sp -= self.skill.sp_cost
        # A zero-duration skill is instantaneous (a one-shot effect); the engine
        # applies its effect and never sets an end time.
        self.skill_active_until = now + self.skill.duration if self.skill.duration > 0 else 0.0
        return True

    def end_skill(self) -> None:
        if self.skill_modifier_source:
            self.stats.remove_source(self.skill_modifier_source)
            self.skill_modifier_source = ""
        self.skill_active_until = 0.0

    def buff(self, source: str, attr: str, *, add: float = 0.0, mul: float = 0.0,
             until: float | None = None) -> None:
        self.stats.add(Modifier(source, attr, add=add, mul=mul, expires_at=until))


def unblock(op: OperatorUnit, enemy: EnemyUnit) -> None:
    """Release a block link from both sides."""
    if enemy.blocked_by == op.uid:
        enemy.blocked_by = None
    if enemy.uid in op.blocking:
        op.blocking.remove(enemy.uid)


def try_block(op: OperatorUnit, enemy: EnemyUnit) -> bool:
    """Block ``enemy`` if there is capacity and it is not already blocked.

    Flying enemies are never blocked; that is a property of the enemy's motion
    mode rather than of the operator, so it is checked here in one place.
    """
    if enemy.blocked or enemy.is_flying or op.block_free <= 0:
        return False
    if op.state is not UnitState.ACTIVE or not enemy.targetable:
        return False
    if StatusKind.STUN in op.stats.statuses:
        return False
    op.blocking.append(enemy.uid)
    enemy.blocked_by = op.uid
    return True
