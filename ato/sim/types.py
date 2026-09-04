"""Core value types shared by the whole simulator.

Coordinate convention
---------------------
We keep the *game's own* convention rather than inventing one, because every
input (routes, checkpoints, predefined tokens, copilot files) speaks it:

* a tile is ``(row, col)``; ``row`` counts from the **bottom** of the map,
  ``col`` from the left. ``mapData.map`` is stored top-row-first, so the display
  index of a row is ``height - 1 - row``. (Verified empirically across several
  maps: route start/end positions only land on start/end tiles under this
  convention.)
* continuous positions are ``(x, y)`` with ``x = col``, ``y = row``, so a tile's
  centre is at integer coordinates and one unit is one tile.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class Direction(enum.IntEnum):
    """Facing. Values match the rotation order used by :meth:`AttackRange.rotated`."""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def delta(self) -> tuple[int, int]:
        """(d_row, d_col) one step in this direction."""
        return {
            Direction.RIGHT: (0, 1),
            Direction.DOWN: (-1, 0),
            Direction.LEFT: (0, -1),
            Direction.UP: (1, 0),
        }[self]

    @staticmethod
    def parse(name: str) -> Direction:
        """Parse the game's names (``UP``/``DOWN``/``LEFT``/``RIGHT``)."""
        return {
            "RIGHT": Direction.RIGHT,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "UP": Direction.UP,
            "EAST": Direction.RIGHT,
            "SOUTH": Direction.DOWN,
            "WEST": Direction.LEFT,
            "NORTH": Direction.UP,
        }[name.upper()]


@dataclass(frozen=True, slots=True)
class Tile:
    """A discrete tile coordinate."""

    row: int
    col: int

    def __add__(self, other: tuple[int, int]) -> Tile:
        return Tile(self.row + other[0], self.col + other[1])

    @property
    def xy(self) -> Vec2:
        return Vec2(float(self.col), float(self.row))


@dataclass(frozen=True, slots=True)
class Vec2:
    """A continuous position in tile units."""

    x: float
    y: float

    def __add__(self, o: Vec2) -> Vec2:
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: Vec2) -> Vec2:
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.y * k)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        n = self.length()
        return Vec2(0.0, 0.0) if n == 0.0 else Vec2(self.x / n, self.y / n)

    @property
    def tile(self) -> Tile:
        """The tile whose centre this position is nearest to."""
        return Tile(int(round(self.y)), int(round(self.x)))


class Side(enum.IntEnum):
    PLAYER = 0
    ENEMY = 1


class DamageType(enum.IntEnum):
    PHYSICAL = 0
    ARTS = 1
    TRUE = 2
    HEAL = 3
    #: Elemental build-up (erosion/burning/necrosis...). Modelled separately
    #: because it accumulates against ``epResistance`` rather than dealing HP damage.
    ELEMENTAL = 4


class MotionMode(enum.IntEnum):
    WALK = 0
    FLY = 1

    @staticmethod
    def parse(name: str) -> MotionMode:
        return MotionMode.FLY if str(name).upper() in ("FLY", "E_FLY") else MotionMode.WALK


class BattleResult(enum.IntEnum):
    RUNNING = 0
    CLEARED = 1      # all waves resolved with life points remaining
    FAILED = 2       # life points exhausted
    TIMEOUT = 3      # hit max_battle_seconds — treated as a failure, but flagged
