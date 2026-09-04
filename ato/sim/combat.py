"""Damage resolution, stat modifiers and status effects.

Every formula here is a *hypothesis about the client*, not a fact read out of
the data files, so each one is registered in ``docs/SIM_SPEC.md`` with its
calibration status. The named constants exist so that system identification can
fit them against observed battles rather than someone re-deriving them from
memory (see INVARIANT I-5).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field

from ato.gamedata.models import Stats
from ato.sim.types import DamageType

#: A hit always lands for at least this fraction of the attacker's ATK, no
#: matter how much DEF or RES the target has. Widely observed as 5%.
MIN_DAMAGE_RATIO = 0.05

#: Effective attack speed is floored: buffs cannot drive the interval to zero
#: and slows cannot stop a unit outright.
MIN_ATTACK_SPEED = 10.0

#: RES is a percentage; it is clamped before use so that an over-shred does not
#: turn arts damage into healing.
MAX_RES = 100.0
MIN_RES = 0.0


class StatusKind(enum.StrEnum):
    """Control effects. The names match the immunity flags in the game data,
    which is how immunity checks stay honest instead of being hand-mapped."""

    STUN = "stun"
    SILENCE = "silence"
    SLEEP = "sleep"
    FROZEN = "frozen"
    LEVITATE = "levitate"
    DISARMED_COMBAT = "disarmedCombat"
    FEARED = "feared"
    PALSY = "palsy"
    ATTRACT = "attract"
    ROOT = "root"          # bound in place but still able to attack

    @property
    def immunity_flag(self) -> str:
        return f"{self.value}Immune"

    @property
    def blocks_action(self) -> bool:
        """Whether the unit is prevented from attacking/using skills."""
        return self in (
            StatusKind.STUN,
            StatusKind.SLEEP,
            StatusKind.FROZEN,
            StatusKind.LEVITATE,
            StatusKind.DISARMED_COMBAT,
            StatusKind.PALSY,
        )

    @property
    def blocks_movement(self) -> bool:
        return self in (
            StatusKind.STUN,
            StatusKind.SLEEP,
            StatusKind.FROZEN,
            StatusKind.LEVITATE,
            StatusKind.ROOT,
        )


def damage_after_defence(atk: float, dtype: DamageType, defence: float, res: float) -> float:
    """The single place damage is computed. All hits go through here."""
    if atk <= 0.0:
        return 0.0
    if dtype is DamageType.TRUE:
        return atk
    if dtype is DamageType.PHYSICAL:
        return max(atk - defence, atk * MIN_DAMAGE_RATIO)
    if dtype is DamageType.ARTS:
        r = min(max(res, MIN_RES), MAX_RES)
        return max(atk * (1.0 - r / 100.0), atk * MIN_DAMAGE_RATIO)
    # HEAL and ELEMENTAL do not interact with DEF/RES.
    return atk


@dataclass(slots=True)
class Modifier:
    """One additive/multiplicative adjustment to a stat.

    Composition rule: ``value = (base + sum(add)) * (1 + sum(mul))``.

    That ordering — flats first, then the sum of fractional buffs — is the
    community-consensus reading of the client and is a registered calibration
    item. It matters: multiplicative-stacking instead of additive would
    materially overrate stacked attack buffs, and a policy trained against the
    wrong rule would over-commit to buff-stacking compositions.
    """

    source: str
    attr: str
    add: float = 0.0
    mul: float = 0.0
    expires_at: float | None = None      # simulation time, seconds

    def active(self, now: float) -> bool:
        return self.expires_at is None or now < self.expires_at


@dataclass(slots=True)
class Status:
    kind: StatusKind
    expires_at: float
    source: str = ""


class StatBlock:
    """Base stats plus the modifiers and statuses currently acting on a unit."""

    __slots__ = ("base", "_mods", "_status", "_cache", "_dirty")

    def __init__(self, base: Stats) -> None:
        self.base = base
        self._mods: list[Modifier] = []
        self._status: dict[StatusKind, Status] = {}
        self._cache: dict[str, float] = {}
        self._dirty = True

    # -- modifiers -------------------------------------------------------

    def add(self, mod: Modifier) -> None:
        self._mods.append(mod)
        self._dirty = True

    def remove_source(self, source: str) -> None:
        n = len(self._mods)
        self._mods = [m for m in self._mods if m.source != source]
        if len(self._mods) != n:
            self._dirty = True

    def expire(self, now: float) -> None:
        """Drop timed-out modifiers and statuses. Called once per tick."""
        if any(not m.active(now) for m in self._mods):
            self._mods = [m for m in self._mods if m.active(now)]
            self._dirty = True
        if self._status:
            dead = [k for k, s in self._status.items() if now >= s.expires_at]
            for k in dead:
                del self._status[k]

    def modifiers(self) -> Iterable[Modifier]:
        return tuple(self._mods)

    # -- statuses --------------------------------------------------------

    def apply_status(self, kind: StatusKind, until: float, source: str = "") -> bool:
        """Apply a control effect unless the unit is immune. Returns whether it landed.

        Refreshing takes the *longer* of the two durations, which is how the
        client behaves for overlapping applications of the same effect.
        """
        if kind.immunity_flag in self.base.immunities:
            return False
        cur = self._status.get(kind)
        if cur is None or until > cur.expires_at:
            self._status[kind] = Status(kind, until, source)
        return True

    def has(self, kind: StatusKind) -> bool:
        return kind in self._status

    @property
    def statuses(self) -> tuple[StatusKind, ...]:
        return tuple(self._status)

    @property
    def can_act(self) -> bool:
        return not any(k.blocks_action for k in self._status)

    @property
    def can_move(self) -> bool:
        return not any(k.blocks_movement for k in self._status)

    @property
    def silenced(self) -> bool:
        return StatusKind.SILENCE in self._status

    # -- resolved values --------------------------------------------------

    def _resolve(self, attr: str, base: float) -> float:
        add = 0.0
        mul = 0.0
        for m in self._mods:
            if m.attr == attr:
                add += m.add
                mul += m.mul
        return (base + add) * (1.0 + mul)

    def _rebuild(self) -> None:
        b = self.base
        self._cache = {
            "atk": max(self._resolve("atk", b.atk), 0.0),
            "defense": max(self._resolve("defense", b.defense), 0.0),
            "res": min(max(self._resolve("res", b.res), MIN_RES), MAX_RES),
            "max_hp": max(self._resolve("max_hp", b.max_hp), 1.0),
            "attack_speed": max(self._resolve("attack_speed", b.attack_speed), MIN_ATTACK_SPEED),
            "move_speed": max(self._resolve("move_speed", b.move_speed), 0.0),
            "block_cnt": max(self._resolve("block_cnt", float(b.block_cnt)), 0.0),
            "taunt_level": self._resolve("taunt_level", float(b.taunt_level)),
            "hp_recovery_per_sec": self._resolve("hp_recovery_per_sec", b.hp_recovery_per_sec),
            "sp_recovery_per_sec": self._resolve("sp_recovery_per_sec", b.sp_recovery_per_sec),
        }
        self._dirty = False

    def __getattr__(self, name: str) -> float:
        # Only reached for names not in __slots__, i.e. the resolved stats.
        if self._dirty:
            self._rebuild()
        try:
            return self._cache[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def attack_interval(self) -> float:
        """Seconds between attacks, after attack-speed modifiers."""
        return self.base.base_attack_time * 100.0 / self.attack_speed

    @property
    def block_capacity(self) -> int:
        return int(self.block_cnt)


@dataclass(slots=True)
class DamageEvent:
    """A single resolved hit, kept for replay, debugging and calibration."""

    time: float
    source_id: int
    target_id: int
    dtype: DamageType
    raw: float
    dealt: float
    overkill: float = 0.0
    lethal: bool = False


@dataclass
class DamageLog:
    """Per-battle damage record. Also the substrate for sim-vs-real comparison."""

    events: list[DamageEvent] = field(default_factory=list)
    enabled: bool = False

    def record(self, ev: DamageEvent) -> None:
        if self.enabled:
            self.events.append(ev)

    def dealt_by(self, unit_id: int) -> float:
        return sum(e.dealt for e in self.events if e.source_id == unit_id)
