"""The mechanism registry and the int/string enum normalisers."""

from __future__ import annotations

import pytest

from ato.sim.registry import (
    ALL_REGISTRIES,
    MechanismKind,
    NoveltyEvent,
    NoveltyLog,
    Registry,
    UnknownMechanism,
    checkpoint_type,
    motion_mode,
    wave_action_type,
)

# Declaration order recovered from int-serialised level files; these indices are
# the contract every older level file depends on.
CHECKPOINT_INTS = [
    (0, "MOVE"),
    (1, "WAIT_FOR_SECONDS"),
    (2, "WAIT_FOR_PLAY_TIME"),
    (3, "WAIT_CURRENT_FRAGMENT_TIME"),
    (4, "WAIT_CURRENT_WAVE_TIME"),
    (5, "DISAPPEAR"),
    (6, "APPEAR_AT_POS"),
    (7, "ALERT"),
    (8, "PATROL_MOVE"),
]

WAVE_ACTION_INTS = [
    (0, "SPAWN"),
    (1, "PREVIEW_CURSOR"),
    (2, "STORY"),
    (3, "TUTORIAL"),
    (4, "PLAY_OPERA"),
    (5, "TRIGGER_PREDEFINED"),
    (6, "ACTIVATE_PREDEFINED"),
    (7, "BATTLE_EVENTS"),
    (8, "DISPLAY_ENEMY_INFO"),
    (9, "WITHDRAW_PREDEFINED"),
]


@pytest.mark.parametrize(("value", "expected"), CHECKPOINT_INTS)
def test_checkpoint_type_from_int(value: int, expected: str) -> None:
    assert checkpoint_type(value) == expected


@pytest.mark.parametrize(("value", "expected"), CHECKPOINT_INTS)
def test_checkpoint_type_from_string_is_idempotent(value: int, expected: str) -> None:
    assert checkpoint_type(expected) == expected
    assert checkpoint_type(expected.lower()) == expected


def test_checkpoint_type_unknown_int_is_flagged_not_guessed() -> None:
    assert checkpoint_type(99) == "UNKNOWN_99"
    assert checkpoint_type(-1) == "UNKNOWN_-1"


def test_checkpoint_type_passes_unknown_strings_through() -> None:
    # WAIT_BOSSRUSH_WAVE only ever appears as a string; it must survive.
    assert checkpoint_type("WAIT_BOSSRUSH_WAVE") == "WAIT_BOSSRUSH_WAVE"


def test_bools_are_not_treated_as_enum_indices() -> None:
    # bool is a subclass of int; True must not silently become index 1.
    assert checkpoint_type(True) == "True"
    assert wave_action_type(False) == "False"


@pytest.mark.parametrize(("value", "expected"), WAVE_ACTION_INTS)
def test_wave_action_type_from_int(value: int, expected: str) -> None:
    assert wave_action_type(value) == expected


@pytest.mark.parametrize(("value", "expected"), WAVE_ACTION_INTS)
def test_wave_action_type_from_string(value: int, expected: str) -> None:
    assert wave_action_type(expected) == expected


def test_wave_action_type_unknown() -> None:
    assert wave_action_type(42) == "UNKNOWN_42"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "WALK"),
        (1, "FLY"),
        ("WALK", "WALK"),
        ("FLY", "FLY"),
        ("fly", "FLY"),
        ("E_NUM", "WALK"),      # padding slots
        (-1, "WALK"),           # UNKNOWN_-1 folds back to the safe default
        (None, "NONE"),
    ],
)
def test_motion_mode(value: object, expected: str) -> None:
    assert motion_mode(value) == expected


def test_registry_register_and_lookup() -> None:
    reg = Registry(MechanismKind.TRAIT)
    assert not reg.known("trait_x")

    @reg.register("trait_x", "trait_y")
    def handler() -> str:
        return "hit"

    assert reg.known("trait_x") and reg.known("trait_y")
    assert reg.get("trait_x") is handler
    assert reg.get("trait_x")() == "hit"
    assert reg.get("nope") is None
    assert reg.keys() == frozenset({"trait_x", "trait_y"})


def test_registry_ignore_is_a_decision_not_an_oversight() -> None:
    reg = Registry(MechanismKind.TRAIT)
    reg.ignore("trait_cosmetic", "visual only")
    assert reg.known("trait_cosmetic")
    # An ignored key is known but has no handler — the caller must not call it.
    assert reg.get("trait_cosmetic") is None
    reg.ignore_all({"a": "why a", "b": "why b"})
    assert reg.keys() == frozenset({"trait_cosmetic", "a", "b"})


def test_all_registries_are_indexed_by_their_own_kind() -> None:
    for kind, reg in ALL_REGISTRIES.items():
        assert reg.kind is kind


def test_novelty_log_records_and_dedups() -> None:
    log = NoveltyLog(strict=False)
    reg = Registry(MechanismKind.RUNE)

    assert log.check(reg, "rune_unknown", "level x") is False
    assert log.check(reg, "rune_unknown", "level x") is False
    assert log.events == [NoveltyEvent(MechanismKind.RUNE, "rune_unknown", "level x")]
    assert not log.clean

    # A different context is a different sighting and must be kept.
    log.check(reg, "rune_unknown", "level y")
    assert len(log.events) == 2


def test_novelty_log_known_keys_are_silent() -> None:
    log = NoveltyLog(strict=True)
    reg = Registry(MechanismKind.RUNE)
    reg.register("rune_known")(lambda: None)
    reg.ignore("rune_cosmetic", "no battle effect")

    assert log.check(reg, "rune_known") is True
    assert log.check(reg, "rune_cosmetic") is True
    assert log.clean


def test_novelty_log_strict_raises_with_the_event_attached() -> None:
    log = NoveltyLog(strict=True)
    reg = Registry(MechanismKind.RUNE)
    with pytest.raises(UnknownMechanism) as exc:
        log.check(reg, "rune_brand_new", "level z")
    assert exc.value.event == NoveltyEvent(MechanismKind.RUNE, "rune_brand_new", "level z")
    # The event is recorded before raising, so the continual-learning pipeline
    # still sees what was missing.
    assert log.events == [exc.value.event]
    assert "rune_brand_new" in str(exc.value)


def test_novelty_log_normalises_non_string_keys() -> None:
    log = NoveltyLog(strict=False)
    reg = Registry(MechanismKind.CHECKPOINT)
    log.check(reg, 17, "int key")
    assert log.events[0].key == "17"
