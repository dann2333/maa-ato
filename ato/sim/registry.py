"""The mechanism registry — how the simulator stays honest about what it models.

Arknights ships mechanics as *string keys*: tile kinds (``tile_telin``),
difficulty runes (``enemy_attribute_mul``), route checkpoint types
(``PATROL_MOVE``), wave actions (``ACTIVATE_PREDEFINED``), skill blackboard
entries (``attack@atk_scale``). New content adds new keys.

Rather than silently ignoring a key it does not implement — which produces a
simulator that is confidently wrong — the engine looks every key up here. An
unknown key raises in strict mode and is always recorded as a
:class:`NoveltyEvent`. Those events are the input to the continual-learning
pipeline: they say precisely which mechanic the system has never seen.

Older level files serialise these enums as integers, so parsers accept both.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar


class MechanismKind(enum.StrEnum):
    TILE = "tile"
    RUNE = "rune"
    CHECKPOINT = "checkpoint"
    WAVE_ACTION = "wave_action"
    SKILL_KEY = "skill_key"
    ENEMY_ABILITY = "enemy_ability"
    TRAIT = "trait"


@dataclass(frozen=True)
class NoveltyEvent:
    """A mechanic the simulator met but does not model."""

    kind: MechanismKind
    key: str
    context: str = ""

    def __str__(self) -> str:
        return f"unmodelled {self.kind}:{self.key}" + (f" ({self.context})" if self.context else "")


class UnknownMechanism(RuntimeError):
    """Raised in strict mode when the simulator meets a mechanic it cannot model."""

    def __init__(self, event: NoveltyEvent) -> None:
        super().__init__(str(event))
        self.event = event


T = TypeVar("T")


class Registry:
    """A namespace of known keys and their handlers."""

    def __init__(self, kind: MechanismKind) -> None:
        self.kind = kind
        self._handlers: dict[str, Callable[..., Any]] = {}
        #: Keys we know exist and deliberately treat as no-ops (cosmetic, story,
        #: preview cursors). Distinguishing these from genuinely unknown keys is
        #: the whole point — a no-op must be a *decision*, not an oversight.
        self._ignored: dict[str, str] = {}

    def register(self, *keys: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            for k in keys:
                self._handlers[k] = fn
            return fn

        return deco

    def ignore(self, key: str, why: str) -> None:
        self._ignored[key] = why

    def ignore_all(self, mapping: dict[str, str]) -> None:
        self._ignored.update(mapping)

    def known(self, key: str) -> bool:
        return key in self._handlers or key in self._ignored

    def get(self, key: str) -> Callable[..., Any] | None:
        return self._handlers.get(key)

    def keys(self) -> frozenset[str]:
        return frozenset(self._handlers) | frozenset(self._ignored)


@dataclass
class NoveltyLog:
    """Collects novelty across a scenario compile or a battle."""

    strict: bool = True
    events: list[NoveltyEvent] = field(default_factory=list)

    def report(self, kind: MechanismKind, key: str, context: str = "") -> None:
        ev = NoveltyEvent(kind, str(key), context)
        if ev not in self.events:
            self.events.append(ev)
        if self.strict:
            raise UnknownMechanism(ev)

    def check(self, reg: Registry, key: Any, context: str = "") -> bool:
        """Return True if ``key`` is modelled; otherwise record (and maybe raise)."""
        k = str(key)
        if reg.known(k):
            return True
        self.report(reg.kind, k, context)
        return False

    @property
    def clean(self) -> bool:
        return not self.events


# ---------------------------------------------------------------------------
# The registries themselves. Handlers are attached by the modules that own them
# (grid effects, rune application, pathing, wave scheduling, skill effects), so
# adding a mechanic is a local change plus one registration.
# ---------------------------------------------------------------------------

TILES = Registry(MechanismKind.TILE)
RUNES = Registry(MechanismKind.RUNE)
CHECKPOINTS = Registry(MechanismKind.CHECKPOINT)
WAVE_ACTIONS = Registry(MechanismKind.WAVE_ACTION)
SKILL_KEYS = Registry(MechanismKind.SKILL_KEY)

ALL_REGISTRIES = {
    MechanismKind.TILE: TILES,
    MechanismKind.RUNE: RUNES,
    MechanismKind.CHECKPOINT: CHECKPOINTS,
    MechanismKind.WAVE_ACTION: WAVE_ACTIONS,
    MechanismKind.SKILL_KEY: SKILL_KEYS,
}


# -- enum normalisation -----------------------------------------------------
#
# Level files written by older client versions store these enums as integers.
# The orderings below are the client's declaration order, recovered by matching
# integer-valued files against string-valued files for the same mechanic.

_CHECKPOINT_ORDER = (
    "MOVE",
    "WAIT_FOR_SECONDS",
    "WAIT_FOR_PLAY_TIME",
    "WAIT_CURRENT_FRAGMENT_TIME",
    "WAIT_CURRENT_WAVE_TIME",
    "DISAPPEAR",
    "APPEAR_AT_POS",
    "ALERT",
    "PATROL_MOVE",
)

_WAVE_ACTION_ORDER = (
    "SPAWN",
    "PREVIEW_CURSOR",
    "STORY",
    "TUTORIAL",
    "PLAY_OPERA",
    "TRIGGER_PREDEFINED",
    "ACTIVATE_PREDEFINED",
    "BATTLE_EVENTS",
    "DISPLAY_ENEMY_INFO",
    "WITHDRAW_PREDEFINED",
)

_MOTION_ORDER = ("WALK", "FLY")


def _norm(value: Any, order: tuple[str, ...]) -> str:
    """Normalise an enum that may arrive as an int or a string."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return order[value] if 0 <= value < len(order) else f"UNKNOWN_{value}"
    return str(value).upper()


def checkpoint_type(value: Any) -> str:
    return _norm(value, _CHECKPOINT_ORDER)


def wave_action_type(value: Any) -> str:
    return _norm(value, _WAVE_ACTION_ORDER)


def motion_mode(value: Any) -> str:
    v = _norm(value, _MOTION_ORDER)
    return "WALK" if v in ("E_NUM", "UNKNOWN_-1") else v
