"""Compiling a level JSON into a Scenario: waves, runes, roster, difficulty."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import ARENA_ROWS, FLY_ROUTE, PLACEHOLDER_ROUTE, WALK_ROUTE, make_level, registered

from ato.gamedata.tables import GameData
from ato.sim.registry import RUNES, MechanismKind, UnknownMechanism
from ato.sim.scenario import Difficulty, compile_scenario
from ato.sim.types import Direction, MotionMode, Tile

RUNE_KEYS = ("ebuff_attribute", "gbuff_lifepoint", "cbuff_cost_recovery", "rune_always")

WAVES: list[dict[str, Any]] = [
    {
        "preDelay": 0.0,
        "postDelay": 1.0,
        "maxTimeWaitingForNextWave": -1.0,
        "fragments": [
            {
                "preDelay": 0.0,
                "actions": [
                    {"actionType": "STORY", "key": "obt/story/test", "count": 1,
                     "preDelay": 0.0, "interval": 1.0, "routeIndex": 0,
                     "hiddenGroup": None, "randomSpawnGroupKey": None, "weight": 0},
                ],
            },
            {
                "preDelay": 3.0,
                "actions": [
                    # actionType 0 == SPAWN in the int-serialised files.
                    {"actionType": 0, "key": "enemy_1000_test", "count": 2,
                     "preDelay": 3.0, "interval": 1.5, "routeIndex": 1,
                     "hiddenGroup": None, "randomSpawnGroupKey": None, "weight": 0,
                     "dontBlockWave": True, "blockFragment": False},
                ],
            },
        ],
    },
    {
        "preDelay": 5.0,
        "postDelay": 0.0,
        "maxTimeWaitingForNextWave": 20.0,
        "fragments": [
            {
                "preDelay": 0.0,
                "actions": [
                    {"actionType": "SPAWN", "key": "enemy_1100_test_fly", "count": 3,
                     "preDelay": 0.0, "interval": 2.0, "routeIndex": 2,
                     "hiddenGroup": "hidden_1", "randomSpawnGroupKey": "grp", "weight": 5},
                    # actionType 6 == ACTIVATE_PREDEFINED.
                    {"actionType": 6, "key": "trap_002_emp", "count": 1,
                     "preDelay": 1.0, "interval": 1.0, "routeIndex": 0,
                     "hiddenGroup": None, "randomSpawnGroupKey": None, "weight": 0},
                ],
            },
        ],
    },
]

RUNES_RAW: list[dict[str, Any]] = [
    {"difficultyMask": "FOUR_STAR", "key": "ebuff_attribute", "professionMask": 1023,
     "buildableMask": "ALL",
     "blackboard": [{"key": "atk", "value": 1.2, "valueStr": None},
                    {"key": "max_hp", "value": 1.2, "valueStr": None}]},
    # Older files write the mask as an int; 2 == FOUR_STAR.
    {"difficultyMask": 2, "key": "gbuff_lifepoint", "professionMask": 1023,
     "buildableMask": "ALL",
     "blackboard": [{"key": "value", "value": 1.0, "valueStr": None}]},
    {"difficultyMask": "ALL", "key": "cbuff_cost_recovery", "professionMask": 8,
     "buildableMask": "MELEE",
     "blackboard": [{"key": "scale", "value": 2.0, "valueStr": None},
                    {"key": "tag", "value": 0.0, "valueStr": "dp"}]},
    {"difficultyMask": "NONE", "key": "rune_always", "professionMask": 1023,
     "buildableMask": "ALL", "blackboard": None},
]

PREDEFINES: dict[str, Any] = {
    "characterInsts": [
        {"position": {"row": 0, "col": 4}, "direction": "RIGHT", "hidden": True,
         "alias": "支援干员", "skillIndex": 1,
         "inst": {"characterKey": "char_999_test", "level": 30, "phase": "PHASE_1"}},
    ],
    "tokenInsts": [
        {"position": {"row": 2, "col": 1}, "direction": "UP", "hidden": False,
         "alias": None, "skillIndex": 0,
         "inst": {"characterKey": "trap_002_emp", "level": 10, "phase": "PHASE_0"}},
    ],
}

ENEMY_REFS: list[dict[str, Any]] = [
    {"useDb": True, "id": "enemy_1000_test", "level": 0, "overwrittenData": None},
    {"useDb": True, "id": "enemy_1000_test", "level": 1, "overwrittenData": None},
    {"useDb": True, "id": "enemy_1100_test_fly", "level": 0, "overwrittenData": None},
]


def battle_level(**overrides: Any) -> dict[str, Any]:
    level = make_level(
        ARENA_ROWS,
        routes=[PLACEHOLDER_ROUTE, WALK_ROUTE, FLY_ROUTE],
        extraRoutes=[FLY_ROUTE],
        waves=WAVES,
        runes=RUNES_RAW,
        predefines=PREDEFINES,
        enemyDbRefs=ENEMY_REFS,
        excludeCharIdList=["char_002_amiya"],
        randomSeed=1234,
    )
    level.update(copy.deepcopy(overrides))
    return level


@pytest.fixture(autouse=True)
def known_runes() -> Iterator[None]:
    """No rune handlers exist yet, so every rune key would read as novelty.

    Registering the fixture's own keys keeps each test's novelty log about the
    one thing that test injects; the registry is restored afterwards.
    """
    with registered(RUNES, *RUNE_KEYS):
        yield


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_compile_is_clean_when_every_mechanic_is_known(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, level_id="level_test", strict=True)
    assert sc.novelty.clean
    assert sc.level_id == "level_test"
    assert sc.bmap.height == 5 and sc.bmap.width == 7
    assert sc.excluded_chars == frozenset({"char_002_amiya"})
    assert sc.random_seed == 1234


def test_level_id_falls_back_to_the_raw_field(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert sc.level_id == "Obt/Main/level_test_00-01"


def test_options_are_parsed(gamedata: GameData) -> None:
    opt = compile_scenario(battle_level(), gamedata, strict=False).options
    assert opt.character_limit == 6
    assert opt.max_life_point == 3
    assert opt.initial_cost == 15
    assert opt.max_cost == 99
    assert opt.cost_increase_time == 1.0
    assert opt.move_multiplier == 0.5
    assert opt.steering_enabled
    assert opt.max_play_time == -1.0
    assert opt.function_disable_mask == "NONE"


def test_routes_are_indexed_and_placeholders_dropped(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert set(sc.routes) == {1, 2, -1}          # slot 0 is padding
    assert sc.routes[1].motion is MotionMode.WALK
    assert sc.routes[2].motion is MotionMode.FLY
    assert sc.routes[-1].motion is MotionMode.FLY    # extraRoutes get negative keys
    assert sc.routes[1].checkpoints[0].position == Tile(3, 1)


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------


def test_waves_fragments_and_int_serialised_actions(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert len(sc.waves) == 2

    first, second = sc.waves
    assert (first.pre_delay, first.post_delay, first.max_wait_for_next) == (0.0, 1.0, -1.0)
    assert (second.pre_delay, second.max_wait_for_next) == (5.0, 20.0)

    story = first.fragments[0].actions[0]
    assert story.kind == "STORY" and not story.spawns

    spawn = first.fragments[1].actions[0]
    assert spawn.kind == "SPAWN"                  # normalised from actionType 0
    assert spawn.spawns
    assert spawn.key == "enemy_1000_test"
    assert spawn.count == 2
    assert spawn.route_index == 1
    assert spawn.dont_block_wave
    assert spawn.busy_until == pytest.approx(4.5)         # 3.0 + (2-1) * 1.5
    assert first.fragments[1].busy_until == pytest.approx(7.5)

    hidden, activate = second.fragments[0].actions
    assert hidden.hidden_group == "hidden_1"
    assert hidden.random_spawn_group == "grp"
    assert hidden.weight == 5
    assert activate.kind == "ACTIVATE_PREDEFINED"  # normalised from actionType 6
    assert not activate.spawns


def test_enemy_count_ignores_hidden_groups_and_non_spawns(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert sc.enemy_count == 2


def test_unknown_wave_action_is_reported(gamedata: GameData) -> None:
    waves = copy.deepcopy(WAVES)
    waves[0]["fragments"][0]["actions"][0]["actionType"] = "SUMMON_A_NEW_THING"
    level = battle_level(waves=waves)

    sc = compile_scenario(level, gamedata, level_id="lv", strict=False)
    assert sc.waves[0].fragments[0].actions == ()          # dropped, not guessed
    assert [(e.kind, e.key) for e in sc.novelty.events] == [
        (MechanismKind.WAVE_ACTION, "SUMMON_A_NEW_THING")
    ]
    with pytest.raises(UnknownMechanism):
        compile_scenario(level, gamedata, level_id="lv", strict=True)


# ---------------------------------------------------------------------------
# Runes and difficulty
# ---------------------------------------------------------------------------


def test_normal_difficulty_excludes_four_star_runes(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, difficulty=Difficulty.NORMAL, strict=True)
    keys = [r.key for r in sc.runes]
    assert "ebuff_attribute" not in keys
    assert "gbuff_lifepoint" not in keys
    # An ALL mask and an unmasked rune apply to every variant.
    assert keys == ["cbuff_cost_recovery", "rune_always"]


def test_four_star_difficulty_adds_the_adverse_runes(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, difficulty=Difficulty.FOUR_STAR, strict=True)
    keys = [r.key for r in sc.runes]
    assert keys == ["ebuff_attribute", "gbuff_lifepoint", "cbuff_cost_recovery", "rune_always"]

    ebuff = sc.runes[0]
    assert ebuff.difficulty_mask is Difficulty.FOUR_STAR
    assert ebuff.blackboard == {"atk": 1.2, "max_hp": 1.2}
    # The int-serialised mask must resolve to the same flag.
    assert sc.runes[1].difficulty_mask is Difficulty.FOUR_STAR


def test_rune_blackboard_keeps_string_values(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, difficulty="NORMAL", strict=True)
    cbuff = sc.runes[0]
    assert cbuff.key == "cbuff_cost_recovery"
    assert cbuff.blackboard == {"scale": 2.0, "tag": 0.0}
    assert cbuff.value_str == {"tag": "dp"}
    assert cbuff.profession_mask == 8
    assert cbuff.buildable_mask == "MELEE"


def test_unknown_rune_raises_in_strict_mode_and_logs_otherwise(gamedata: GameData) -> None:
    runes = copy.deepcopy(RUNES_RAW)
    runes.append({"difficultyMask": "NORMAL", "key": "rune_from_the_future",
                  "professionMask": 1023, "buildableMask": "ALL", "blackboard": None})
    level = battle_level(runes=runes)

    with pytest.raises(UnknownMechanism) as exc:
        compile_scenario(level, gamedata, level_id="lv", strict=True)
    assert exc.value.event.kind is MechanismKind.RUNE
    assert exc.value.event.key == "rune_from_the_future"

    sc = compile_scenario(level, gamedata, level_id="lv", strict=False)
    assert not sc.novelty.clean
    assert [e.key for e in sc.novelty.events] == ["rune_from_the_future"]
    # The rune is still carried so the caller can see what it will not model.
    assert "rune_from_the_future" in [r.key for r in sc.runes]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NORMAL", Difficulty.NORMAL),
        ("FOUR_STAR", Difficulty.FOUR_STAR),
        ("EASY", Difficulty.EASY),
        ("SIX_STAR", Difficulty.SIX_STAR),
        ("ALL", Difficulty.ALL),
        ("NONE", Difficulty.NONE),
        ("NORMAL|FOUR_STAR", Difficulty.NORMAL | Difficulty.FOUR_STAR),
        ("NORMAL, FOUR_STAR", Difficulty.NORMAL | Difficulty.FOUR_STAR),
        (0, Difficulty.NONE),
        (1, Difficulty.NORMAL),
        (2, Difficulty.FOUR_STAR),
        (3, Difficulty.NORMAL | Difficulty.FOUR_STAR),
        (Difficulty.EASY, Difficulty.EASY),
        ("SOMETHING_NEW", Difficulty.NONE),
    ],
)
def test_difficulty_parse(value: object, expected: Difficulty) -> None:
    assert Difficulty.parse(value) == expected


# ---------------------------------------------------------------------------
# Enemy roster
# ---------------------------------------------------------------------------


def test_enemy_roster_resolves_every_referenced_level(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert set(sc.enemy_specs) == {
        ("enemy_1000_test", 0),
        ("enemy_1000_test", 1),
        ("enemy_1100_test_fly", 0),
    }
    assert sc.enemy("enemy_1000_test", 0).stats.max_hp == 1000.0
    assert sc.enemy("enemy_1000_test", 1).stats.max_hp == 2500.0   # level-1 override
    assert sc.enemy("enemy_1000_test", 1).stats.atk == 100.0       # inherited
    assert sc.enemy("enemy_1100_test_fly", 0).is_flying


def test_enemy_lookup_reports_what_is_missing(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    with pytest.raises(KeyError, match="enemy_1000_test"):
        sc.enemy("enemy_1000_test", 4)


def test_enemy_absent_from_the_database_is_novelty(gamedata: GameData) -> None:
    refs = [*copy.deepcopy(ENEMY_REFS),
            {"useDb": True, "id": "enemy_9999_unseen", "level": 0, "overwrittenData": None}]
    level = battle_level(enemyDbRefs=refs)

    sc = compile_scenario(level, gamedata, level_id="lv", strict=False)
    assert ("enemy_9999_unseen", 0) not in sc.enemy_specs
    assert [(e.kind, e.key) for e in sc.novelty.events] == [
        (MechanismKind.ENEMY_ABILITY, "enemy_9999_unseen")
    ]
    with pytest.raises(UnknownMechanism):
        compile_scenario(level, gamedata, level_id="lv", strict=True)


@pytest.mark.xfail(
    strict=False,
    reason="BUG: enemyDbRefs[].overwrittenData is dropped. Real levels use it "
    "(act1bossrush_level_bossrush1_01 gives enemy_1001_bigbo maxHp 40000 and "
    "lifePointReduce 5), so those stages would simulate with database stats.",
)
def test_enemy_ref_overwritten_data_is_applied(gamedata: GameData) -> None:
    refs = copy.deepcopy(ENEMY_REFS)
    refs[0]["overwrittenData"] = {
        "name": {"m_defined": False, "m_value": None},
        "lifePointReduce": {"m_defined": True, "m_value": 5},
        "attributes": {
            "maxHp": {"m_defined": True, "m_value": 40000},
            "atk": {"m_defined": False, "m_value": 0},
        },
    }
    sc = compile_scenario(battle_level(enemyDbRefs=refs), gamedata, strict=False)
    spec = sc.enemy("enemy_1000_test", 0)
    assert spec.stats.max_hp == 40000.0
    assert spec.life_point_reduce == 5


# ---------------------------------------------------------------------------
# Predefined units
# ---------------------------------------------------------------------------


def test_predefined_units(gamedata: GameData) -> None:
    sc = compile_scenario(battle_level(), gamedata, strict=False)
    assert len(sc.predefined) == 2

    operator, token = sc.predefined
    assert operator.char_key == "char_999_test"
    assert operator.tile == Tile(0, 4)
    assert operator.direction is Direction.RIGHT
    assert operator.phase == 1
    assert operator.level == 30
    assert operator.skill_index == 1
    assert operator.hidden
    assert operator.alias == "支援干员"
    assert not operator.is_token

    assert token.char_key == "trap_002_emp"
    assert token.tile == Tile(2, 1)
    assert token.direction is Direction.UP
    assert token.is_token
    assert not token.hidden


@pytest.mark.xfail(
    strict=False,
    reason="BUG: predefines[].direction is serialised as an int in older level files "
    "(act2bossrush_level_bossrush2_ex03 uses 1). Direction.parse only accepts names, so "
    "1 raises KeyError('1') and 0 is swallowed by `or \"UP\"` and silently faces UP.",
)
@pytest.mark.parametrize(
    ("raw", "expected"), [(0, Direction.RIGHT), (1, Direction.DOWN)]
)
def test_predefined_direction_accepts_ints(
    gamedata: GameData, raw: int, expected: Direction
) -> None:
    predefines = copy.deepcopy(PREDEFINES)
    predefines["tokenInsts"][0]["direction"] = raw
    sc = compile_scenario(battle_level(predefines=predefines), gamedata, strict=False)
    assert sc.predefined[-1].direction is expected
