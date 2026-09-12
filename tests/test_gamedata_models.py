"""Typed views over the raw tables: enemies, operators, skills, ranges."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import make_flier_levels

from ato.gamedata.models import (
    ATTACK_SPEED_BOUNDS,
    AttackRange,
    Stats,
    field_or,
    _lerp_frames,
    resolve_enemy,
    resolve_operator,
    resolve_range,
    resolve_skill,
)
from ato.sim.types import Direction

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aspd", "expected"),
    [
        (100.0, 2.0),      # nominal
        (200.0, 1.0),      # twice as fast
        (50.0, 4.0),       # half as fast
        (20.0, 10.0),      # exactly on the floor
        (5.0, 10.0),       # below the floor: clamped, not extrapolated
        (0.0, 10.0),       # a total slow must not divide by zero
        (-100.0, 10.0),
        (600.0, 1.0 / 3),  # exactly on the ceiling
        (2000.0, 1.0 / 3),  # above it: clamped. Unbounded, this was 0.05 s.
    ],
)
def test_attack_interval_scales_with_attack_speed(aspd: float, expected: float) -> None:
    stats = Stats(base_attack_time=2.0, attack_speed=aspd)
    assert stats.attack_interval == pytest.approx(expected)


def test_attack_interval_bounds_are_caller_supplied() -> None:
    """The clamp is a calibration constant, so the simulator must be able to move it.

    The floor is disputed between sources (10 vs 20) and neither reads it out of
    the client, so nothing may hard-code one of them.
    """
    stats = Stats(base_attack_time=2.0, attack_speed=15.0)
    assert stats.attack_interval_within(10.0, 600.0) == pytest.approx(2.0 * 100 / 15)
    assert stats.attack_interval_within(20.0, 600.0) == pytest.approx(10.0)
    assert stats.attack_interval == stats.attack_interval_within(*ATTACK_SPEED_BOUNDS)


def test_zero_valued_fields_survive_parsing() -> None:
    """``or`` cannot tell ``0`` from a missing field, and both occur in this data.

    A stationary enemy (``moveSpeed: 0``) that inherits the 1.0 default walks to
    the blue box; a skill with ``spRecoveryPerSec: 0`` that inherits 1.0 charges
    itself. Neither raises anything, which is exactly why it needs a test.
    """
    assert field_or({"moveSpeed": 0.0}, "moveSpeed", 1.0) == 0.0
    assert field_or({"moveSpeed": None}, "moveSpeed", 1.0) == 1.0
    assert field_or({}, "moveSpeed", 1.0) == 1.0
    assert field_or({"applyWay": ""}, "applyWay", "MELEE") == ""


# ---------------------------------------------------------------------------
# Enemies
# ---------------------------------------------------------------------------


def test_resolve_enemy_level_zero(enemy_levels: list[dict[str, Any]]) -> None:
    spec = resolve_enemy(enemy_levels, 0)
    assert spec.key == "enemy_1000_test"
    assert spec.name == "测试猎犬"
    assert spec.level == 0
    assert spec.stats.max_hp == 1000.0
    assert spec.stats.atk == 100.0
    assert spec.stats.defense == 50.0
    assert spec.stats.res == 10.0
    assert spec.stats.move_speed == 1.5
    assert spec.stats.base_attack_time == 2.0
    assert spec.stats.attack_interval == pytest.approx(2.0)
    assert spec.life_point_reduce == 2
    assert spec.tags == ("infection",)
    assert spec.apply_way == "MELEE"
    assert not spec.is_flying


def test_resolve_enemy_level_one_inherits_undeclared_fields(
    enemy_levels: list[dict[str, Any]],
) -> None:
    spec = resolve_enemy(enemy_levels, 1)
    assert spec.level == 1
    assert spec.stats.max_hp == 2500.0        # the only field level 1 redeclares
    assert spec.stats.atk == 100.0            # inherited from level 0
    assert spec.stats.defense == 50.0
    assert spec.stats.move_speed == 1.5
    assert spec.stats.base_attack_time == 2.0
    assert spec.name == "测试猎犬"             # m_defined:false must not blank it
    assert spec.key == "enemy_1000_test"
    assert spec.life_point_reduce == 2


def test_resolve_enemy_clamps_above_the_highest_defined_level(
    enemy_levels: list[dict[str, Any]],
) -> None:
    top = resolve_enemy(enemy_levels, 1)
    for level in (2, 7, 99):
        spec = resolve_enemy(enemy_levels, level)
        assert spec.level == 1
        assert spec.stats.max_hp == top.stats.max_hp


def test_resolve_enemy_is_independent_of_entry_order(
    enemy_levels: list[dict[str, Any]],
) -> None:
    assert resolve_enemy(list(reversed(enemy_levels)), 1).stats.max_hp == 2500.0


def test_resolve_enemy_immunities(enemy_levels: list[dict[str, Any]]) -> None:
    spec = resolve_enemy(enemy_levels, 0)
    # stunImmune/sleepImmune are declared true; silenceImmune is declared false
    # and frozenImmune is undefined — neither may appear.
    assert spec.stats.immunities == frozenset({"stunImmune", "sleepImmune"})
    assert resolve_enemy(enemy_levels, 1).stats.immunities == spec.stats.immunities


def test_resolve_enemy_flier() -> None:
    spec = resolve_enemy(make_flier_levels(), 0)
    assert spec.is_flying
    assert spec.motion == "FLY"
    assert spec.apply_way == "RANGED"
    assert spec.stats.attack_interval == pytest.approx(0.8)   # 1.6s at 200% aspd


def test_resolve_enemy_keeps_the_merged_attributes_inspectable(
    enemy_levels: list[dict[str, Any]],
) -> None:
    spec = resolve_enemy(enemy_levels, 1)
    assert spec.raw["attributes"]["maxHp"] == 2500
    assert spec.raw["attributes"]["atk"] == 100
    assert spec.raw["prefabKey"] == "enemy_1000_test"


def test_resolve_enemy_rejects_an_empty_level_list() -> None:
    with pytest.raises(ValueError, match="no level entries"):
        resolve_enemy([], 0)


@pytest.mark.xfail(
    strict=False,
    reason="BUG: `float(attrs.get('moveSpeed') or 1.0)` turns an explicitly declared "
    "moveSpeed of 0 (a stationary enemy) into 1.0. Same `or`-fallback pattern hides a "
    "declared 0 for baseAttackTime and maxDeployCount.",
)
def test_resolve_enemy_keeps_a_declared_zero_move_speed() -> None:
    levels = [
        {
            "level": 0,
            "enemyData": {
                "name": {"m_defined": True, "m_value": "静止"},
                "prefabKey": {"m_defined": True, "m_value": "enemy_static"},
                "attributes": {
                    "maxHp": {"m_defined": True, "m_value": 100},
                    "moveSpeed": {"m_defined": True, "m_value": 0.0},
                },
            },
        }
    ]
    assert resolve_enemy(levels, 0).stats.move_speed == 0.0


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_resolve_operator_at_level_one(character: dict[str, Any]) -> None:
    op = resolve_operator("char_999_test", character, phase=0, level=1, trust=0)
    assert op.char_id == "char_999_test"
    assert op.name == "测试狙击"
    assert op.phase == 0
    assert op.level == 1
    assert op.stats.max_hp == 1000.0
    assert op.stats.atk == 200.0
    assert op.stats.defense == 100.0
    assert op.stats.cost == 10
    assert op.stats.block_cnt == 1
    assert op.stats.sp_recovery_per_sec == 1.0
    assert op.range_id == "3-1"
    assert op.position == "RANGED"
    assert op.profession == "SNIPER"
    assert op.sub_profession == "fastshot"
    assert op.skill_ids == ("skchr_test_1", "skchr_test_2")   # the null id is dropped


def test_resolve_operator_level_none_means_max_level(character: dict[str, Any]) -> None:
    op = resolve_operator("char_999_test", character, phase=0, level=None, trust=0)
    assert op.level == 50
    assert op.stats.max_hp == 1490.0
    assert op.stats.atk == 298.0
    assert op.stats.defense == 149.0


def test_resolve_operator_interpolates_the_midpoint(character: dict[str, Any]) -> None:
    # Level 25 sits exactly halfway between the level-1 and level-50 frames.
    op = resolve_operator("char_999_test", character, phase=0, level=25, trust=0)
    assert op.stats.max_hp == 1240.0
    assert op.stats.atk == 248.0
    assert op.stats.defense == 124.0


@pytest.mark.parametrize(("level", "expected"), [(0, 1), (-5, 1), (51, 50), (999, 50)])
def test_resolve_operator_clamps_level(
    character: dict[str, Any], level: int, expected: int
) -> None:
    op = resolve_operator("char_999_test", character, phase=0, level=level, trust=0)
    assert op.level == expected


@pytest.mark.parametrize(("phase", "expected"), [(-1, 0), (0, 0), (1, 1), (2, 1), (9, 1)])
def test_resolve_operator_clamps_phase(
    character: dict[str, Any], phase: int, expected: int
) -> None:
    op = resolve_operator("char_999_test", character, phase=phase, trust=0)
    assert op.phase == expected


def test_resolve_operator_uses_the_phase_range_and_max_level(
    character: dict[str, Any],
) -> None:
    # The fixture has no E2, so the default phase=2 must fall back to E1.
    op = resolve_operator("char_999_test", character, trust=0)
    assert (op.phase, op.level, op.range_id) == (1, 60, "3-6")
    assert op.stats.max_hp == 2200.0
    assert op.stats.cost == 12


def test_resolve_operator_trust_endpoints(character: dict[str, Any]) -> None:
    bare = resolve_operator("char_999_test", character, phase=0, level=50, trust=0)
    full = resolve_operator("char_999_test", character, phase=0, level=50, trust=200)
    assert bare.stats.max_hp == 1490.0
    assert bare.stats.atk == 298.0
    assert full.stats.max_hp == 1690.0        # +200 maxHp at full trust
    assert full.stats.atk == 368.0            # +70 atk
    assert full.stats.cost == bare.stats.cost  # zero-valued favour fields are skipped


@pytest.mark.xfail(
    strict=False,
    reason="BUG: `trust` is documented as the 0-200 trust scale but is compared against "
    "the favour *key-frame* level, which caps at 50 in the real table. Every trust >= 50 "
    "therefore yields the full bonus; half trust should give half of it.",
)
def test_resolve_operator_trust_scales_linearly(character: dict[str, Any]) -> None:
    half = resolve_operator("char_999_test", character, phase=0, level=50, trust=100)
    assert half.stats.max_hp == 1590.0
    assert half.stats.atk == 333.0


def test_resolve_operator_potential_modifiers(character: dict[str, Any]) -> None:
    base = resolve_operator("char_999_test", character, phase=0, level=50, trust=0, potential=0)
    one = resolve_operator("char_999_test", character, phase=0, level=50, trust=0, potential=1)
    two = resolve_operator("char_999_test", character, phase=0, level=50, trust=0, potential=2)
    assert (base.stats.max_hp, base.stats.cost) == (1490.0, 10)
    assert (one.stats.max_hp, one.stats.cost) == (1690.0, 10)
    assert (two.stats.max_hp, two.stats.cost) == (1690.0, 9)


def test_resolve_operator_ignores_non_attribute_potential_ranks(
    character: dict[str, Any],
) -> None:
    # Rank 3 of the fixture is a CUSTOM (talent) rank with no attribute buff.
    two = resolve_operator("char_999_test", character, phase=0, level=50, trust=0, potential=2)
    three = resolve_operator("char_999_test", character, phase=0, level=50, trust=0, potential=3)
    assert three.stats.max_hp == two.stats.max_hp
    assert three.stats.cost == two.stats.cost
    # Asking for more potentials than exist must not raise.
    assert resolve_operator(
        "char_999_test", character, phase=0, level=50, trust=0, potential=9
    ).stats.cost == 9


def test_resolve_operator_immunities(character: dict[str, Any]) -> None:
    for frame in character["phases"][0]["attributesKeyFrames"]:
        frame["data"]["stunImmune"] = True
    op = resolve_operator("char_999_test", character, phase=0, level=25, trust=0)
    assert op.stats.immunities == frozenset({"stunImmune"})


def test_lerp_frames_keeps_integral_attributes_integral(character: dict[str, Any]) -> None:
    frames = character["phases"][0]["attributesKeyFrames"]
    data = _lerp_frames(frames, 25)
    for key in ("maxHp", "atk", "def", "cost", "blockCnt", "respawnTime"):
        assert isinstance(data[key], int), f"{key} became {type(data[key]).__name__}"
    assert isinstance(data["moveSpeed"], float)
    assert data["stunImmune"] is False        # bools must not be interpolated


def test_lerp_frames_with_a_single_key_frame(character: dict[str, Any]) -> None:
    frames = character["phases"][0]["attributesKeyFrames"][:1]
    assert _lerp_frames(frames, 40)["maxHp"] == 1000


@pytest.mark.xfail(
    strict=False,
    reason="BUG: character_table stores rarity as 'TIER_5'; resolve_operator only accepts "
    "an int, so every operator silently resolves to rarity 0 (a 1-star).",
)
def test_resolve_operator_rarity(character: dict[str, Any]) -> None:
    op = resolve_operator("char_999_test", character, trust=0)
    assert op.rarity in (4, 5)     # 5-star, in either the old or the new encoding


def test_operator_with_stats_replaces_only_what_it_is_given(
    character: dict[str, Any],
) -> None:
    op = resolve_operator("char_999_test", character, phase=0, level=1, trust=0)
    buffed = op.with_stats(atk=999.0)
    assert buffed.stats.atk == 999.0
    assert buffed.stats.max_hp == op.stats.max_hp
    assert op.stats.atk == 200.0               # the original is untouched


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _skill() -> dict[str, Any]:
    def level(idx: int) -> dict[str, Any]:
        return {
            "name": "测试技能",
            "rangeId": None if idx < 2 else "x-1",
            "description": f"level {idx}",
            "skillType": "MANUAL",
            "durationType": "NONE",
            "duration": 10.0 + idx,
            "spData": {
                "spType": "INCREASE_WHEN_ATTACK",
                "spCost": 30 - idx,
                "initSp": 10,
                "increment": 1.0,
            },
            "blackboard": [
                {"key": "atk_scale", "value": 1.5 + idx, "valueStr": None},
                {"key": None, "value": 0.0},
            ],
        }

    return {"levels": [level(i) for i in range(3)]}


def test_resolve_skill_selects_the_mastery_level() -> None:
    spec = resolve_skill("skchr_test_1", _skill(), mastery=1)
    assert spec.level == 1
    assert spec.sp_cost == 29.0
    assert spec.init_sp == 10.0
    assert spec.duration == 11.0
    assert spec.sp_type == "INCREASE_WHEN_ATTACK"
    assert spec.skill_type == "MANUAL"
    assert spec.blackboard == {"atk_scale": 2.5}   # the null-keyed entry is dropped
    assert spec.range_id is None
    assert not spec.is_passive and not spec.is_auto


@pytest.mark.parametrize(("mastery", "expected"), [(-3, 0), (0, 0), (2, 2), (6, 2)])
def test_resolve_skill_clamps_mastery(mastery: int, expected: int) -> None:
    assert resolve_skill("skchr_test_1", _skill(), mastery).level == expected


# ---------------------------------------------------------------------------
# Attack ranges
# ---------------------------------------------------------------------------

#: The shape range_table uses, taken from the real ``1-2`` entry.
RANGE_1_2 = {
    "id": "1-2",
    "direction": 1,
    "grids": [
        {"row": 1, "col": 0},
        {"row": 0, "col": 0},
        {"row": 0, "col": 1},
        {"row": -1, "col": 0},
    ],
}


def test_resolve_range() -> None:
    rng = resolve_range("1-2", RANGE_1_2)
    assert rng.range_id == "1-2"
    assert rng.grids == ((1, 0), (0, 0), (0, 1), (-1, 0))
    assert resolve_range("empty", {}).grids == ()


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (Direction.RIGHT, ((1, 0), (0, 0), (0, 1), (-1, 0))),
        (Direction.DOWN, ((0, 1), (0, 0), (-1, 0), (0, -1))),
        (Direction.LEFT, ((-1, 0), (0, 0), (0, -1), (1, 0))),
        (Direction.UP, ((0, -1), (0, 0), (1, 0), (0, 1))),
    ],
)
def test_attack_range_rotated(
    direction: Direction, expected: tuple[tuple[int, int], ...]
) -> None:
    assert resolve_range("1-2", RANGE_1_2).rotated(direction) == expected


def test_attack_range_rotation_matches_the_direction_deltas() -> None:
    # The single tile "one step in front" must land on the facing's own delta.
    front = AttackRange("front", ((0, 1),))
    for direction in Direction:
        assert front.rotated(direction) == (direction.delta,)


def test_attack_range_four_rotations_are_the_identity() -> None:
    rng = resolve_range("1-2", RANGE_1_2)
    grids = rng.grids
    for _ in range(4):
        grids = AttackRange("1-2", grids).rotated(Direction.DOWN)
    assert grids == rng.grids


def test_attack_range_rotation_wraps() -> None:
    rng = resolve_range("1-2", RANGE_1_2)
    assert rng.rotated(4) == rng.grids
    assert rng.rotated(-1) == rng.rotated(3)
    assert rng.rotated(5) == rng.rotated(1)
