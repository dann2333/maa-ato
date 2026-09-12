"""Tick-exact timing: the arithmetic that decides *when* anything happens.

Every test here pins a number that was measurably wrong before. They are cheap
and boring on purpose: a timer that is 3% slow produces a battle that still
looks entirely plausible, so nothing but an exact assertion catches it.
"""

from __future__ import annotations

import pytest
from conftest import WALK_ROUTE, make_level

from ato.gamedata.tables import GameData
from ato.sim.engine import BattleEngine, EngineCalibration
from ato.sim.scenario import BattleOptions, compile_scenario
from ato.sim.timing import PeriodicGrant, exact_ratio, ticks_for

TPS = 30


# ---------------------------------------------------------------------------
# PeriodicGrant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "scale", "first_tick", "in_300"),
    [
        (1.0, 1.0, 30, 10),        # the float version fired at tick 31 and gave 9
        (1.0, 2.0, 15, 20),        # Adverse doubles the rate, it does not halve it twice
        (0.7, 1.0, 21, 14),        # a period that is not a whole number of tenths
        (2.0, 1.0, 60, 5),
        (999999.0, 1.0, None, 0),  # a real level value: essentially no DP recovery
        (0.01, 1.0, 1, 1000),      # shorter than a tick: several units in one tick
    ],
)
def test_periodic_grant_lands_on_the_exact_tick(
    period: float, scale: float, first_tick: int | None, in_300: int
) -> None:
    grant = PeriodicGrant(period, TPS, scale)
    first, total = None, 0
    for tick in range(1, 301):
        n = grant.tick()
        if n and first is None:
            first = tick
        total += n
    assert (first, total) == (first_tick, in_300)


@pytest.mark.parametrize(("period", "scale"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_periodic_grant_never_fires_on_a_dead_schedule(period: float, scale: float) -> None:
    grant = PeriodicGrant(period, TPS, scale)
    assert not grant.active
    assert sum(grant.tick() for _ in range(1000)) == 0


def test_exact_ratio_reads_the_decimal_not_the_binary_expansion() -> None:
    assert exact_ratio(0.7) == (7, 10)
    assert exact_ratio(1.0) == (1, 1)
    assert exact_ratio(0.5) == (1, 2)
    assert exact_ratio(999999.0) == (999999, 1)


def test_ticks_for_does_not_round_a_whole_number_of_ticks_up() -> None:
    # 0.1 * 30 == 3.0000000000000004; a bare ceil() makes that a 4-tick delay.
    assert ticks_for(0.1, TPS) == 3
    assert ticks_for(1.0, TPS) == 30
    assert ticks_for(0.0, TPS) == 0
    assert ticks_for(1.0 / TPS + 1e-12, TPS) == 1


# ---------------------------------------------------------------------------
# DP accrual in a real engine
# ---------------------------------------------------------------------------


#: One enemy, far enough in the future that it never spawns during these tests.
#: A level with no waves is CLEARED on its first tick, and a finished battle
#: stops ticking -- which would make every assertion below vacuously pass.
IDLE_WAVE: dict[str, object] = {
    "preDelay": 0.0,
    "postDelay": 0.0,
    "maxTimeWaitingForNextWave": -1.0,
    "fragments": [
        {
            "preDelay": 600.0,
            "actions": [
                {"actionType": "SPAWN", "key": "enemy_1000_test", "count": 1,
                 "preDelay": 0.0, "interval": 0.0, "routeIndex": 0,
                 "hiddenGroup": None, "randomSpawnGroupKey": None, "weight": 0},
            ],
        }
    ],
}


def _engine(gamedata: GameData, **options: object) -> BattleEngine:
    level = make_level(
        waves=[IDLE_WAVE],
        routes=[WALK_ROUTE],
        enemyDbRefs=[{"useDb": True, "id": "enemy_1000_test", "level": 0,
                      "overwrittenData": None}],
    )
    level["options"].update(options)
    sc = compile_scenario(level, gamedata, level_id="timing", strict=False)
    return BattleEngine(sc, gamedata, [], ticks_per_second=TPS, strict=False)


def test_first_dp_arrives_on_tick_30_not_31(gamedata: GameData) -> None:
    eng = _engine(gamedata, initialCost=0, costIncreaseTime=1.0)
    assert eng.state.cost == 0.0
    for _ in range(29):
        eng.tick()
    assert eng.state.cost == 0.0
    eng.tick()
    assert eng.state.cost == 1.0


def test_ten_seconds_of_battle_yields_ten_dp(gamedata: GameData) -> None:
    """The old float accumulator produced nine, every time, on every stage."""
    eng = _engine(gamedata, initialCost=0, costIncreaseTime=1.0)
    for _ in range(300):
        eng.tick()
    assert eng.state.cost == 10.0


def test_dp_stops_at_the_cap_and_does_not_bank_the_overflow(gamedata: GameData) -> None:
    eng = _engine(gamedata, initialCost=0, costIncreaseTime=1.0, maxCost=3)
    for _ in range(300):
        eng.tick()
    assert eng.state.cost == 3.0


def test_the_clock_is_derived_from_the_tick_count(gamedata: GameData) -> None:
    """``time`` must be ``k / tps`` exactly, not a sum of ``1 / tps``.

    An accumulated clock drifts away from the tick index, so a deadline set at
    ``t + interval`` lands on a different tick than the one it was meant for.
    """
    eng = _engine(gamedata)
    for k in (1, 30, 137, 900):
        while eng.tick_index < k:
            eng.tick()
        assert eng.state.time == k / TPS


# ---------------------------------------------------------------------------
# SP
# ---------------------------------------------------------------------------


def test_sp_is_stored_in_tick_units_so_a_30_sp_skill_is_ready_at_30_seconds() -> None:
    """The same shortfall as DP: 30 SP took 31 s, and every skill in the game was late."""
    from ato.gamedata.models import SkillSpec, Stats
    from ato.sim.combat import StatBlock
    from ato.sim.entities import OperatorUnit
    from ato.sim.types import Side, Vec2

    skill = SkillSpec(skill_id="s", name="s", level=0, sp_type="INCREASE_WITH_TIME",
                      sp_cost=30.0, init_sp=0.0, increment=1.0, duration=10.0,
                      duration_type="AMMO", skill_type="MANUAL", range_id=None)
    op = OperatorUnit(spec_name="u", side=Side.PLAYER, stats=StatBlock(Stats()), hp=1.0,
                      position=Vec2(0.0, 0.0), skill=skill, tick_rate=TPS)
    for _ in range(30 * TPS - 1):
        op.charge_sp_ticks(1.0)
        assert not op.sp_ready()
    op.charge_sp_ticks(1.0)
    assert op.sp_ready()
    assert op.sp == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Attack interval quantisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # baseAttackTime 0.78 s is 23.4 ticks: the three policies disagree by a
        # tick per attack, which is ~4% DPS. Which one the client does is
        # unmeasured, hence a switch rather than a decision.
        ("ceil", [24, 24, 24, 24, 24]),
        ("round", [23, 23, 23, 23, 23]),
        ("none", [24, 23, 24, 23, 23]),   # carries the remainder: 5 attacks in 117 ticks
    ],
)
def test_attack_quantisation_policies(
    gamedata: GameData, mode: str, expected: list[int]
) -> None:
    from ato.gamedata.models import Stats
    from ato.sim.combat import StatBlock
    from ato.sim.entities import EnemyUnit
    from ato.sim.types import Side, Vec2

    eng = _engine(gamedata)
    eng.cal = EngineCalibration(attack_quantization=mode)
    unit = EnemyUnit(spec_name="u", side=Side.ENEMY, hp=1.0, position=Vec2(0.0, 0.0),
                     stats=StatBlock(Stats(base_attack_time=0.78, attack_speed=100.0)))
    gaps, previous = [], 0
    for _ in range(len(expected)):
        eng._schedule_next_attack(unit)
        gaps.append(unit.next_attack_tick - previous)
        previous = unit.next_attack_tick
        eng.tick_index = unit.next_attack_tick
    assert gaps == expected


def test_carrying_the_remainder_keeps_the_long_run_rate_exact(gamedata: GameData) -> None:
    """Over a hundred attacks, ``none`` must not have lost a single tick to rounding."""
    from ato.gamedata.models import Stats
    from ato.sim.combat import StatBlock
    from ato.sim.entities import EnemyUnit
    from ato.sim.types import Side, Vec2

    eng = _engine(gamedata)
    eng.cal = EngineCalibration(attack_quantization="none")
    unit = EnemyUnit(spec_name="u", side=Side.ENEMY, hp=1.0, position=Vec2(0.0, 0.0),
                     stats=StatBlock(Stats(base_attack_time=0.78, attack_speed=100.0)))
    for _ in range(100):
        eng._schedule_next_attack(unit)
        eng.tick_index = unit.next_attack_tick
    assert unit.next_attack_tick == round(100 * 0.78 * TPS)


# ---------------------------------------------------------------------------
# Zero-valued level options
# ---------------------------------------------------------------------------


def test_zero_initial_cost_is_not_ten_free_dp() -> None:
    """Three levels in a 66-level sample ship ``initialCost: 0``."""
    opts = BattleOptions.from_level({"options": {"initialCost": 0, "costIncreaseTime": 999999.0}})
    assert opts.initial_cost == 0
    assert opts.cost_increase_time == 999999.0


def test_missing_move_multiplier_falls_back_to_the_observed_value() -> None:
    """The dataclass default and the parser fallback used to disagree (0.5 vs 1.0),
    so a level that omitted the field ran its enemies at double speed."""
    assert BattleOptions.from_level({"options": {}}).move_multiplier == 0.5
    assert BattleOptions().move_multiplier == 0.5
