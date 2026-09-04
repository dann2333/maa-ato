"""The battle engine: a deterministic, fixed-tick simulation of one Arknights stage.

Design notes
------------
*Determinism.* The engine takes an explicit seed and uses it only where the game
is genuinely random (spawn jitter). Everything else is a pure function of the
scenario and the action sequence, so a plan replays identically — which is what
makes search, regression testing and sim-vs-real comparison possible.

*Honesty over coverage.* Where the client's behaviour is not recoverable from
the data files, the engine implements a named, configurable hypothesis rather
than a silent guess. Each one is a field on :class:`EngineCalibration`, is listed
in ``docs/SIM_SPEC.md``, and is meant to be fitted against observed battles.
Training randomises over them so that a policy cannot come to depend on a value
that may be wrong (INVARIANT I-5).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from ato.gamedata.models import (
    AttackRange,
    OperatorSpec,
    SkillSpec,
    resolve_range,
)
from ato.gamedata.tables import GameData
from ato.sim.combat import (
    DamageEvent,
    DamageLog,
    Modifier,
    StatBlock,
    StatusKind,
    damage_after_defence,
)
from ato.sim.entities import (
    EnemyUnit,
    OperatorUnit,
    UnitState,
    try_block,
    unblock,
)
from ato.sim.pathing import build_route_program
from ato.sim.registry import MechanismKind, NoveltyLog
from ato.sim.scenario import Scenario, WaveAction
from ato.sim.types import BattleResult, DamageType, Direction, Side, Tile, Vec2


@dataclass(frozen=True)
class EngineCalibration:
    """Constants the data files do not give us. Every one is a fitted parameter.

    Defaults are the most defensible starting point, not established fact. See
    ``docs/SIM_SPEC.md`` for each one's status and the experiment that fits it.
    """

    #: Tiles per second for a unit with ``moveSpeed == 1.0``, before the level's
    #: ``options.moveMultiplier`` (which is 0.5 in every level sampled).
    #: The single most impactful unknown in the whole simulator: it sets how much
    #: time an operator has to kill anything.
    move_tiles_per_second: float = 1.0

    #: Seconds an operator spends deploying before it can act or be attacked.
    deploy_lock_seconds: float = 1.0

    #: Seconds after retreat/death before the card is redeployable, as a multiple
    #: of the operator's ``respawnTime``.
    redeploy_multiplier: float = 1.0

    #: Whether a fragment's actions are scheduled from the fragment's start
    #: (True) or from the wave's start (False). The two agree whenever a wave has
    #: one fragment, which is why the data alone does not settle it.
    fragment_relative_delays: bool = True

    #: SP gained per second for INCREASE_WITH_TIME skills, before modifiers.
    sp_per_second: float = 1.0

    #: SP gained per attack landed / per hit taken for the other charge types.
    sp_per_attack: float = 1.0
    sp_per_hit_taken: float = 1.0

    #: Whether an operator starts with its skill's ``initSp``.
    grant_initial_sp: bool = True


#: Profession -> damage type. Operators have no explicit damage-type field, so it
#: is inferred; sub-professions that break the rule are listed below and anything
#: unrecognised is reported as novelty rather than assumed physical.
_PROFESSION_DAMAGE = {
    "CASTER": DamageType.ARTS,
    "MEDIC": DamageType.HEAL,
    "SUPPORTER": DamageType.PHYSICAL,
    "SUPPORT": DamageType.PHYSICAL,
    "SNIPER": DamageType.PHYSICAL,
    "WARRIOR": DamageType.PHYSICAL,
    "GUARD": DamageType.PHYSICAL,
    "TANK": DamageType.PHYSICAL,
    "PIONEER": DamageType.PHYSICAL,
    "SPECIAL": DamageType.PHYSICAL,
}

#: Sub-professions whose damage type differs from their profession's default.
_SUBPROFESSION_DAMAGE = {
    "artsfghter": DamageType.ARTS,       # Arts Fighter guards
    "artsprotector": DamageType.ARTS,    # Arts Protector defenders
    "instructor": DamageType.PHYSICAL,
    "hammer": DamageType.ARTS,           # Crusher guards deal arts on their AoE
    "bard": DamageType.HEAL,
    "underminer": DamageType.ARTS,
    "blastcaster": DamageType.ARTS,
    "ringhealer": DamageType.HEAL,
    "healer": DamageType.HEAL,
    "physician": DamageType.HEAL,
    "wandermedic": DamageType.HEAL,
    "incantationmedic": DamageType.HEAL,
    "chainhealer": DamageType.HEAL,
}


def operator_damage_type(spec: OperatorSpec, novelty: NoveltyLog | None = None) -> DamageType:
    sub = (spec.sub_profession or "").lower()
    if sub in _SUBPROFESSION_DAMAGE:
        return _SUBPROFESSION_DAMAGE[sub]
    prof = (spec.profession or "").upper()
    if prof in _PROFESSION_DAMAGE:
        return _PROFESSION_DAMAGE[prof]
    if novelty is not None:
        novelty.report(MechanismKind.TRAIT, prof or sub or spec.char_id,
                       "cannot infer damage type")
    return DamageType.PHYSICAL


@dataclass
class RosterEntry:
    """One operator the player brought, with the skill they picked."""

    spec: OperatorSpec
    skill: SkillSpec | None = None
    attack_range: AttackRange | None = None
    damage_type: DamageType = DamageType.PHYSICAL
    #: Set once deployed and retreated; the card is unusable until then.
    available_at: float = 0.0
    deploys_used: int = 0

    @property
    def char_id(self) -> str:
        return self.spec.char_id

    @property
    def cost(self) -> int:
        # Redeploying an operator costs more each time, capped by the client at
        # +2. Registered as a calibration item.
        return self.spec.stats.cost + min(self.deploys_used, 2)


@dataclass
class BattleState:
    """The engine's public snapshot. Everything a *privileged* observer can see.

    The pixel-facing observation is a strict subset of this, produced by
    ``ato.agent.observation``; see INVARIANT I-3. Nothing here may be fed to a
    policy network directly.
    """

    time: float = 0.0
    cost: float = 0.0
    life_points: int = 0
    kills: int = 0
    leaked: int = 0
    total_enemies: int = 0
    spawned: int = 0
    wave_index: int = 0
    result: BattleResult = BattleResult.RUNNING
    operators: list[OperatorUnit] = field(default_factory=list)
    enemies: list[EnemyUnit] = field(default_factory=list)

    @property
    def active_enemies(self) -> list[EnemyUnit]:
        return [e for e in self.enemies if e.targetable]

    @property
    def deployed(self) -> list[OperatorUnit]:
        return [o for o in self.operators if o.state in (UnitState.ACTIVE, UnitState.DEPLOYING)]


@dataclass
class _PendingAction:
    """One wave action expanded into its individual emissions."""

    action: WaveAction
    fire_at: float
    index: int


class WaveScheduler:
    """Runs the wave programme.

    Fragments within a wave are sequential; waves advance once their fragments
    have finished emitting and the enemies they spawned are gone, or once
    ``maxTimeWaitingForNextWave`` elapses. The exact rule the client uses for
    fragment chaining is a calibration item — see
    :attr:`EngineCalibration.fragment_relative_delays`.
    """

    def __init__(self, scenario: Scenario, cal: EngineCalibration) -> None:
        self.sc = scenario
        self.cal = cal
        self.wave_index = 0
        self.fragment_index = 0
        self.fragment_start = 0.0
        self.wave_start = 0.0
        self.pending: list[_PendingAction] = []
        self._fragment_loaded = False
        #: Enemies spawned by the current wave that still block its completion.
        self.blocking_uids: set[int] = set()
        self.finished = False
        self._wait_started: float | None = None

    @property
    def current_wave(self):  # noqa: ANN201 - Wave | None, avoiding an import cycle
        ws = self.sc.waves
        return ws[self.wave_index] if self.wave_index < len(ws) else None

    def _load_fragment(self, now: float) -> None:
        wave = self.current_wave
        if wave is None or self.fragment_index >= len(wave.fragments):
            return
        frag = wave.fragments[self.fragment_index]
        base = (self.fragment_start if self.cal.fragment_relative_delays else self.wave_start)
        base += frag.pre_delay if not self.cal.fragment_relative_delays else 0.0
        self.pending = []
        for act in frag.actions:
            for i in range(max(act.count, 1)):
                self.pending.append(
                    _PendingAction(act, base + act.pre_delay + i * act.interval, i)
                )
        self.pending.sort(key=lambda p: p.fire_at)
        self._fragment_loaded = True

    def due(self, now: float) -> list[_PendingAction]:
        """Pop the actions whose time has come."""
        if self.finished:
            return []
        if not self._fragment_loaded:
            wave = self.current_wave
            if wave is None:
                self.finished = True
                return []
            if self.fragment_index == 0 and now < self.wave_start + wave.pre_delay:
                return []
            if self.fragment_index == 0:
                self.fragment_start = self.wave_start + wave.pre_delay
            self._load_fragment(now)
        out = [p for p in self.pending if p.fire_at <= now]
        if out:
            self.pending = [p for p in self.pending if p.fire_at > now]
        return out

    def advance(self, now: float, live_wave_enemies: int) -> None:
        """Move to the next fragment/wave when the current one is done."""
        wave = self.current_wave
        if wave is None or not self._fragment_loaded:
            return
        if self.pending:
            return
        frag = wave.fragments[self.fragment_index]
        if now < self.fragment_start + frag.busy_until:
            return
        if self.fragment_index + 1 < len(wave.fragments):
            self.fragment_index += 1
            nxt = wave.fragments[self.fragment_index]
            self.fragment_start = now + (nxt.pre_delay if self.cal.fragment_relative_delays else 0.0)
            self._fragment_loaded = False
            return
        # Wave finished emitting. Hold until its enemies are resolved, or until
        # maxTimeWaitingForNextWave elapses (a negative value means "no timeout").
        if self._wait_started is None:
            self._wait_started = now
        timed_out = (
            wave.max_wait_for_next >= 0.0 and now - self._wait_started >= wave.max_wait_for_next
        )
        if live_wave_enemies > 0 and not timed_out:
            return
        if now < self._wait_started + wave.post_delay:
            return
        self.wave_index += 1
        self.fragment_index = 0
        self._fragment_loaded = False
        self._wait_started = None
        self.wave_start = now
        self.blocking_uids.clear()
        if self.wave_index >= len(self.sc.waves):
            self.finished = True

    @property
    def emitting_done(self) -> bool:
        return self.finished


class BattleEngine:
    """Simulates one battle. Drive it with :meth:`tick` and the action methods."""

    def __init__(
        self,
        scenario: Scenario,
        gamedata: GameData,
        roster: Sequence[RosterEntry],
        *,
        calibration: EngineCalibration | None = None,
        ticks_per_second: int = 30,
        max_seconds: float = 900.0,
        seed: int = 0,
        strict: bool = True,
        log_damage: bool = False,
    ) -> None:
        self.sc = scenario
        self.gd = gamedata
        self.cal = calibration or EngineCalibration()
        self.tps = ticks_per_second
        self.dt = 1.0 / ticks_per_second
        self.max_seconds = max_seconds
        self.rng = random.Random(seed or scenario.random_seed or 0)
        self.novelty = NoveltyLog(strict=strict)
        self.damage_log = DamageLog(enabled=log_damage)

        self.roster: dict[str, RosterEntry] = {}
        for e in roster:
            self.roster[e.char_id] = self._prepare(e)

        self.state = BattleState(
            life_points=scenario.options.max_life_point,
            total_enemies=scenario.enemy_count,
        )
        self.state.cost = float(scenario.options.initial_cost)
        self._cost_accum = 0.0
        self._cost_rate_scale = 1.0
        self._units: dict[int, OperatorUnit | EnemyUnit] = {}
        self._programs: dict[int, tuple] = {}
        #: Per-operator record of when each enemy entered its range, so that the
        #: "attacks whatever entered range first" rule is reproducible.
        self._in_range_since: dict[int, dict[int, float]] = {}
        self._wave_enemy_uids: set[int] = set()
        self.scheduler = WaveScheduler(scenario, self.cal)
        self._apply_runes()

    # -- setup -----------------------------------------------------------

    def _prepare(self, entry: RosterEntry) -> RosterEntry:
        rng_id = entry.spec.range_id
        arange = entry.attack_range
        if arange is None:
            raw = self.gd.ranges.get(rng_id)
            arange = resolve_range(rng_id, raw) if raw else AttackRange(rng_id, ((0, 0),))
        dmg = entry.damage_type
        if dmg == DamageType.PHYSICAL:
            dmg = operator_damage_type(entry.spec, self.novelty)
        return replace(entry, attack_range=arange, damage_type=dmg)

    def _apply_runes(self) -> None:
        """Apply the level's difficulty runes.

        Only the handful whose semantics are unambiguous are applied here; the
        rest are already recorded as novelty by ``compile_scenario`` so that a
        stage whose difficulty modifiers we cannot reproduce never silently
        yields an easier battle than the real one.
        """
        for rune in self.sc.runes:
            bb = rune.blackboard
            if rune.key in ("gbuff_lifepoint", "global_lifepoint"):
                self.state.life_points += int(bb.get("value", 0))
            elif rune.key == "global_initial_cost_add":
                self.state.cost += bb.get("value", 0.0)
            elif rune.key in ("cbuff_cost_recovery", "global_cost_recovery_mul"):
                self._cost_rate_scale *= bb.get("scale", bb.get("value", 1.0)) or 1.0

    def enemy_stat_multipliers(self) -> dict[str, float]:
        """Rune-driven multipliers applied to every spawned enemy."""
        out = {"max_hp": 1.0, "atk": 1.0, "defense": 1.0, "res": 1.0}
        for rune in self.sc.runes:
            if rune.key in ("ebuff_attribute", "enemy_attribute_mul"):
                bb = rune.blackboard
                out["max_hp"] *= bb.get("max_hp", 1.0)
                out["atk"] *= bb.get("atk", 1.0)
                out["defense"] *= bb.get("def", 1.0)
                out["res"] *= bb.get("magic_resistance", 1.0)
        return out

    # -- player actions ---------------------------------------------------

    def can_deploy(self, char_id: str, tile: Tile) -> tuple[bool, str]:
        entry = self.roster.get(char_id)
        if entry is None:
            return False, "not in roster"
        if char_id in self.sc.excluded_chars:
            return False, "excluded by the stage"
        if self.state.time < entry.available_at:
            return False, "redeploy timer"
        if self.state.cost < entry.cost:
            return False, "not enough DP"
        if len(self.state.deployed) >= self.sc.options.character_limit:
            return False, "deployment limit"
        if any(o.tile == tile for o in self.state.deployed):
            return False, "tile occupied"
        if not self.sc.bmap.deployable(tile, entry.spec.position):
            return False, "tile not deployable for this operator"
        if any(o.spec and o.spec.char_id == char_id for o in self.state.deployed):
            return False, "already deployed"
        return True, ""

    def deploy(self, char_id: str, tile: Tile, direction: Direction) -> OperatorUnit | None:
        ok, _why = self.can_deploy(char_id, tile)
        if not ok:
            return None
        entry = self.roster[char_id]
        stats = StatBlock(entry.spec.stats)
        unit = OperatorUnit(
            spec_name=entry.spec.name,
            side=Side.PLAYER,
            stats=stats,
            hp=entry.spec.stats.max_hp,
            position=tile.xy,
            spec=entry.spec,
            skill=entry.skill,
            attack_range=entry.attack_range,
            damage_type=entry.damage_type,
            state=UnitState.DEPLOYING,
            deploy_time=self.state.time,
        )
        unit.place(tile, direction)
        if entry.skill is not None and self.cal.grant_initial_sp:
            unit.sp = min(entry.skill.init_sp, entry.skill.sp_cost)
        unit.next_attack_at = self.state.time + self.cal.deploy_lock_seconds
        self.state.cost -= entry.cost
        entry.deploys_used += 1
        self.state.operators.append(unit)
        self._units[unit.uid] = unit
        self._in_range_since[unit.uid] = {}
        return unit

    def retreat(self, uid: int) -> bool:
        unit = self._units.get(uid)
        if not isinstance(unit, OperatorUnit) or unit.state not in (
            UnitState.ACTIVE,
            UnitState.DEPLOYING,
        ):
            return False
        self._remove_operator(unit, UnitState.RETREATED)
        return True

    def use_skill(self, uid: int) -> bool:
        unit = self._units.get(uid)
        if not isinstance(unit, OperatorUnit) or unit.state is not UnitState.ACTIVE:
            return False
        if unit.skill is None or unit.skill.is_auto:
            return False
        return self._activate_skill(unit)

    # -- the loop ---------------------------------------------------------

    def tick(self) -> BattleResult:
        if self.state.result is not BattleResult.RUNNING:
            return self.state.result
        now = self.state.time = round(self.state.time + self.dt, 6)

        self._accrue_cost()
        self._expire(now)
        self._spawn_due(now)
        self._move_enemies(now)
        self._assign_blocks()
        self._charge_sp(now)
        self._act_operators(now)
        self._act_enemies(now)
        self._reap(now)
        self.scheduler.advance(now, len(self._wave_enemy_uids))
        return self._check_end(now)

    def run(self, until: float | None = None) -> BattleResult:
        """Run to completion (or to ``until`` seconds) with no further input."""
        limit = until if until is not None else self.max_seconds
        while self.state.result is BattleResult.RUNNING and self.state.time < limit:
            self.tick()
        if self.state.result is BattleResult.RUNNING:
            self.state.result = BattleResult.TIMEOUT
        return self.state.result

    # -- loop stages ------------------------------------------------------

    def _accrue_cost(self) -> None:
        opt = self.sc.options
        if opt.cost_increase_time <= 0:
            return
        self._cost_accum += self.dt * self._cost_rate_scale / opt.cost_increase_time
        if self._cost_accum >= 1.0:
            gained = int(self._cost_accum)
            self._cost_accum -= gained
            self.state.cost = min(self.state.cost + gained, float(opt.max_cost))

    def _expire(self, now: float) -> None:
        for u in self._units.values():
            u.stats.expire(now)
        for o in self.state.deployed:
            if o.state is UnitState.DEPLOYING and now >= o.deploy_time + self.cal.deploy_lock_seconds:
                o.state = UnitState.ACTIVE
            if o.skill_active and now >= o.skill_active_until:
                o.end_skill()

    def _spawn_due(self, now: float) -> None:
        mult = self.enemy_stat_multipliers()
        for pending in self.scheduler.due(now):
            act = pending.action
            if act.kind != "SPAWN":
                # Predefined-unit actions are recognised but not yet simulated;
                # they are logged so a stage that depends on them cannot quietly
                # produce a training signal.
                if act.kind in ("ACTIVATE_PREDEFINED", "TRIGGER_PREDEFINED", "WITHDRAW_PREDEFINED"):
                    self.novelty.report(
                        MechanismKind.WAVE_ACTION, act.kind,
                        f"{self.sc.level_id}: predefined-unit control not simulated",
                    )
                continue
            self._spawn(act, now, mult)

    def _spawn(self, act: WaveAction, now: float, mult: dict[str, float]) -> None:
        route = self.sc.routes.get(act.route_index)
        if route is None:
            self.novelty.report(
                MechanismKind.WAVE_ACTION, f"route[{act.route_index}]",
                f"{self.sc.level_id}: spawn references a missing route",
            )
            return
        spec = None
        for (key, lvl), candidate in self.sc.enemy_specs.items():
            if key == act.key:
                spec = candidate if spec is None or lvl > spec.level else spec
        if spec is None:
            self.novelty.report(MechanismKind.ENEMY_ABILITY, act.key,
                                f"{self.sc.level_id}: enemy not in scenario roster")
            return

        base = replace(
            spec.stats,
            max_hp=spec.stats.max_hp * mult["max_hp"],
            atk=spec.stats.atk * mult["atk"],
            defense=spec.stats.defense * mult["defense"],
            res=spec.stats.res * mult["res"],
        )
        pos = route.spawn_position
        if route.spawn_random_range.length() > 0.0:
            pos = Vec2(
                pos.x + self.rng.uniform(-route.spawn_random_range.x, route.spawn_random_range.x),
                pos.y + self.rng.uniform(-route.spawn_random_range.y, route.spawn_random_range.y),
            )
        prog = self._programs.get(act.route_index)
        if prog is None:
            prog = build_route_program(self.sc.bmap, route)
            self._programs[act.route_index] = prog

        unit = EnemyUnit(
            spec_name=spec.name,
            side=Side.ENEMY,
            stats=StatBlock(base),
            hp=base.max_hp,
            position=pos,
            spec=spec,
            damage_type=self._enemy_damage_type(spec.key),
            program=prog,
            life_point_reduce=spec.life_point_reduce,
            route_index=act.route_index,
            spawn_time=now,
        )
        self.state.enemies.append(unit)
        self._units[unit.uid] = unit
        self.state.spawned += 1
        if not act.dont_block_wave:
            self._wave_enemy_uids.add(unit.uid)

    def _enemy_damage_type(self, enemy_id: str) -> DamageType:
        hb = self.gd.enemy_handbook
        entry = (hb.get("enemyData") or {}).get(enemy_id) if hb else None
        kinds = (entry or {}).get("damageType") or []
        if "MAGIC" in kinds:
            return DamageType.ARTS
        if "HEAL" in kinds and "PHYSIC" not in kinds:
            return DamageType.HEAL
        if "NO_DAMAGE" in kinds and len(kinds) == 1:
            return DamageType.TRUE      # deals no damage; atk is 0 anyway
        return DamageType.PHYSICAL

    def _move_enemies(self, now: float) -> None:
        scale = self.sc.options.move_multiplier * self.cal.move_tiles_per_second
        for e in self.state.enemies:
            if e.state not in (UnitState.ACTIVE, UnitState.HIDDEN):
                continue
            if e.advance(self.dt, now, e.stats.move_speed * scale):
                self._leak(e)

    def _leak(self, e: EnemyUnit) -> None:
        e.state = UnitState.LEAKED
        self.state.leaked += 1
        self.state.life_points -= e.life_point_reduce
        self._detach(e)

    def _assign_blocks(self) -> None:
        """Hand each unblocked ground enemy to the operator standing on its tile.

        Indexed by tile rather than compared pairwise: the naive
        operators-x-enemies scan was the engine's dominant cost, and blocking is
        decided purely by tile identity anyway.
        """
        free: dict[Tile, OperatorUnit] = {}
        for op in self.state.operators:
            if op.state is UnitState.ACTIVE and op.block_free > 0:
                free[op.tile] = op
        if not free:
            return
        for e in self.state.enemies:
            if e.state is not UnitState.ACTIVE or e.blocked_by is not None or e.is_flying:
                continue
            op = free.get(e.position.tile)
            if op is not None and try_block(op, e) and op.block_free <= 0:
                free.pop(op.tile, None)
                if not free:
                    return

    def _charge_sp(self, now: float) -> None:
        for op in self.state.deployed:
            sk = op.skill
            if sk is None or op.state is not UnitState.ACTIVE:
                continue
            if sk.sp_type == "INCREASE_WITH_TIME":
                op.charge_sp(self.dt * self.cal.sp_per_second * op.stats.sp_recovery_per_sec)
            if sk.is_auto and op.sp_ready() and not op.skill_active:
                self._activate_skill(op)

    def _activate_skill(self, op: OperatorUnit) -> bool:
        if not op.start_skill(self.state.time):
            return False
        sk = op.skill
        assert sk is not None
        source = f"skill:{op.uid}"
        op.skill_modifier_source = source
        until = op.skill_active_until or None
        applied = False
        bb = sk.blackboard
        # A deliberately small set of generic effects. Anything else is reported
        # as novelty: a skill we cannot model must not silently do nothing while
        # the policy is rewarded as though it worked.
        if "atk_scale" in bb:
            op.stats.add(Modifier(source, "atk", mul=bb["atk_scale"] - 1.0, expires_at=until))
            applied = True
        if "attack@atk_scale" in bb:
            op.stats.add(Modifier(source, "atk", mul=bb["attack@atk_scale"] - 1.0, expires_at=until))
            applied = True
        if "def_scale" in bb:
            op.stats.add(Modifier(source, "defense", mul=bb["def_scale"] - 1.0, expires_at=until))
            applied = True
        if "attack_speed" in bb:
            op.stats.add(Modifier(source, "attack_speed", add=bb["attack_speed"], expires_at=until))
            applied = True
        if "max_hp" in bb:
            op.stats.add(Modifier(source, "max_hp", mul=bb["max_hp"] - 1.0, expires_at=until))
            applied = True
        if not applied and bb:
            self.novelty.report(
                MechanismKind.SKILL_KEY, sk.skill_id,
                f"no modelled effect among {sorted(bb)[:6]}",
            )
        return True

    # -- attacking --------------------------------------------------------

    def _act_operators(self, now: float) -> None:
        """Let every ready operator take its attack.

        Enemies are bucketed by tile once per tick so an operator only looks at
        the tiles its range actually covers, instead of filtering the whole
        enemy list per operator.
        """
        deployed = self.state.deployed
        if not deployed:
            return
        occupancy: dict[Tile, list[EnemyUnit]] = {}
        for e in self.state.enemies:
            if e.state is UnitState.ACTIVE:
                occupancy.setdefault(e.position.tile, []).append(e)

        for op in deployed:
            if op.state is not UnitState.ACTIVE or not op.stats.can_act:
                continue
            seen = self._in_range_since.setdefault(op.uid, {})
            in_range: list[EnemyUnit] = []
            for t in op.covered_tiles():
                bucket = occupancy.get(t)
                if bucket:
                    in_range.extend(bucket)
            if in_range:
                fresh = {e.uid for e in in_range}
                for e in in_range:
                    seen.setdefault(e.uid, now)
                if len(seen) > len(fresh):
                    for uid in [u for u in seen if u not in fresh]:
                        del seen[uid]
            elif seen:
                seen.clear()
            if now < op.next_attack_at or not in_range:
                continue
            target = self._pick_operator_target(op, in_range, seen)
            if target is None:
                continue
            self._strike(op, target, now)
            op.next_attack_at = now + op.stats.attack_interval
            if op.skill is not None and op.skill.sp_type == "INCREASE_WHEN_ATTACK":
                op.charge_sp(self.cal.sp_per_attack)

    def _pick_operator_target(
        self, op: OperatorUnit, in_range: list[EnemyUnit], seen: dict[int, float]
    ) -> EnemyUnit | None:
        """Blocked enemies first, then whichever entered the range earliest.

        The tie-break on uid keeps the choice deterministic, which matters for
        replay and for comparing two policies on identical rollouts.
        """
        blocked = [e for e in in_range if e.uid in op.blocking]
        pool = blocked or in_range
        return min(pool, key=lambda e: (seen.get(e.uid, 0.0), e.uid))

    def _act_enemies(self, now: float) -> None:
        for e in self.state.enemies:
            if e.state is not UnitState.ACTIVE or not e.stats.can_act:
                continue
            if e.stats.atk <= 0.0:
                continue
            target = self._pick_enemy_target(e)
            if target is None:
                continue
            if now < e.next_attack_at:
                continue
            self._strike(e, target, now)
            e.next_attack_at = now + e.stats.attack_interval

    def _pick_enemy_target(self, e: EnemyUnit) -> OperatorUnit | None:
        if e.blocked_by is not None:
            op = self._units.get(e.blocked_by)
            return op if isinstance(op, OperatorUnit) and op.state is UnitState.ACTIVE else None
        if not e.is_ranged or e.attack_radius <= 0.0:
            return None
        best: OperatorUnit | None = None
        best_d = float("inf")
        for op in self.state.deployed:
            if op.state is not UnitState.ACTIVE:
                continue
            d = (op.position - e.position).length()
            if d <= e.attack_radius and d < best_d:
                best, best_d = op, d
        return best

    def _strike(self, src, dst, now: float) -> None:
        dtype: DamageType = src.damage_type
        atk = src.stats.atk
        if dtype is DamageType.HEAL:
            healed = dst.heal(atk) if dst.side is src.side else 0.0
            self.damage_log.record(DamageEvent(now, src.uid, dst.uid, dtype, atk, healed))
            return
        raw = damage_after_defence(atk, dtype, dst.stats.defense, dst.stats.res)
        dealt, lethal = dst.take_damage(raw)
        self.damage_log.record(
            DamageEvent(now, src.uid, dst.uid, dtype, raw, dealt, max(raw - dealt, 0.0), lethal)
        )
        if isinstance(dst, OperatorUnit) and dst.skill is not None:
            if dst.skill.sp_type == "INCREASE_WHEN_TAKEN_DAMAGE":
                dst.charge_sp(self.cal.sp_per_hit_taken)
        if lethal:
            self._kill(dst, now)

    def _kill(self, unit, now: float) -> None:
        if isinstance(unit, EnemyUnit):
            unit.state = UnitState.DEAD
            self.state.kills += 1
            self._detach(unit)
        else:
            self._remove_operator(unit, UnitState.RETREATED)

    def _detach(self, e: EnemyUnit) -> None:
        if e.blocked_by is not None:
            op = self._units.get(e.blocked_by)
            if isinstance(op, OperatorUnit):
                unblock(op, e)
        self._wave_enemy_uids.discard(e.uid)
        for seen in self._in_range_since.values():
            seen.pop(e.uid, None)

    def _remove_operator(self, op: OperatorUnit, state: UnitState) -> None:
        for uid in list(op.blocking):
            e = self._units.get(uid)
            if isinstance(e, EnemyUnit):
                unblock(op, e)
        op.end_skill()
        op.state = state
        entry = self.roster.get(op.spec.char_id if op.spec else "")
        if entry is not None:
            entry.available_at = self.state.time + (
                entry.spec.stats.respawn_time * self.cal.redeploy_multiplier
            )
        self._in_range_since.pop(op.uid, None)

    def _reap(self, now: float) -> None:
        if len(self.state.enemies) > 256:
            self.state.enemies = [e for e in self.state.enemies if e.alive or e.targetable]

    def _check_end(self, now: float) -> BattleResult:
        st = self.state
        if st.life_points <= 0:
            st.result = BattleResult.FAILED
            return st.result
        if self.scheduler.emitting_done and not st.active_enemies:
            st.result = BattleResult.CLEARED
            return st.result
        if now >= self.max_seconds:
            st.result = BattleResult.TIMEOUT
        return st.result

    # -- branching --------------------------------------------------------

    def clone(self) -> BattleEngine:
        """A deep copy of the mutable battle state, sharing the immutable inputs.

        Search needs to try an action, look ahead, and roll back. Copying the
        whole object would drag the 90 MB game-data snapshot and the compiled
        scenario along with it, so both are pinned in the memo and shared. Route
        programs are frozen dataclasses and are shared for the same reason.

        Correctness rests on those three genuinely being immutable during a
        battle: the engine only ever reads them.
        """
        memo: dict[int, object] = {
            id(self.gd): self.gd,
            id(self.sc): self.sc,
            id(self.cal): self.cal,
        }
        for prog in self._programs.values():
            memo[id(prog)] = prog
        return copy.deepcopy(self, memo)

    # -- introspection ----------------------------------------------------

    def summary(self) -> dict[str, object]:
        st = self.state
        return {
            "level": self.sc.level_id,
            "result": st.result.name,
            "time": round(st.time, 2),
            "kills": st.kills,
            "leaked": st.leaked,
            "spawned": st.spawned,
            "total_enemies": st.total_enemies,
            "life_points": st.life_points,
            "cost": round(st.cost, 1),
            "deployed": len(st.deployed),
            "novelty": [str(e) for e in self.novelty.events],
        }


def make_roster(
    gd: GameData,
    char_ids: Iterable[str],
    *,
    phase: int = 2,
    level: int | None = None,
    trust: int = 200,
    potential: int = 0,
    mastery: int = 6,
    skill_index: int = 0,
) -> list[RosterEntry]:
    """Build a squad at a given investment level.

    The investment parameters are deliberately explicit: training randomises over
    them so the policy conditions on *stats*, not on operator identity
    (INVARIANT I-2).
    """
    from ato.gamedata.models import resolve_operator, resolve_skill

    out: list[RosterEntry] = []
    for cid in char_ids:
        char = gd.characters.get(cid)
        if char is None:
            continue
        spec = resolve_operator(cid, char, phase=phase, level=level, trust=trust,
                                potential=potential)
        skill = None
        if spec.skill_ids:
            sid = spec.skill_ids[min(skill_index, len(spec.skill_ids) - 1)]
            raw = gd.skills.get(sid)
            if raw:
                skill = resolve_skill(sid, raw, mastery)
        out.append(RosterEntry(spec=spec, skill=skill))
    return out
