"""The battle map: tiles, buildability, passability and the movement graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ato.sim.registry import MechanismKind, NoveltyLog
from ato.sim.types import MotionMode, Tile

#: ``buildableType`` values as they appear in level JSON.
BUILD_NONE, BUILD_MELEE, BUILD_RANGED, BUILD_ALL = "NONE", "MELEE", "RANGED", "ALL"

#: A minority of level files serialise the tile enums as integers rather than
#: names. Parsed as strings they compare equal to nothing, so such a stage yields
#: no deployable tiles and no walkable tiles -- it is silently unplayable, with
#: no error anywhere.
#:
#: The mappings below were recovered empirically rather than assumed: for every
#: tile kind appearing in an integer-encoded file, the (height, buildable,
#: passable) triple was compared against the same tile kind's string encoding
#: elsewhere in the corpus. All eight kinds agreed. Values marked inferred were
#: not observed and follow the enum's shape; meeting one produces a NoveltyEvent
#: rather than a guess.
_HEIGHT_BY_INT = {0: "LOWLAND", 1: "HIGHLAND"}
_BUILDABLE_BY_INT = {0: "NONE", 1: "MELEE", 2: "RANGED", 3: "ALL"}  # 3 inferred
_PASSABLE_BY_INT = {0: "NONE", 1: "WALK", 2: "FLY_ONLY", 3: "ALL"}  # 0,1 inferred
_SIDE_BY_INT = {0: "NONE", 1: "PLAYER", 2: "ENEMY", 3: "ALL"}       # inferred


def _enum(value: object, table: dict[int, str], default: str, field: str,
          novelty: "NoveltyLog | None" = None) -> str:
    """Read a tile enum that may arrive as a name or as an integer."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        name = table.get(value)
        if name is None:
            if novelty is not None:
                novelty.report(MechanismKind.TILE, f"{field}={value}",
                               "integer tile enum outside the recovered mapping")
            return default
        return name
    return str(value)


#: Diagonal-inclusive neighbourhood, used when a route allows diagonal movement.
_NEIGHBOURS_8 = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
_NEIGHBOURS_4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


@dataclass(frozen=True, slots=True)
class TileInfo:
    """One tile's static properties."""

    key: str                # tile_road / tile_wall / tile_forbidden / tile_start / tile_end / ...
    height: str             # HIGHLAND | LOWLAND
    buildable: str          # NONE | MELEE | RANGED | ALL
    passable_mask: str      # ALL | FLY_ONLY | NONE
    player_side_mask: str = "ALL"
    blackboard: tuple[tuple[str, float], ...] = ()

    @property
    def is_highland(self) -> bool:
        return self.height == "HIGHLAND"

    def passable_for(self, motion: MotionMode) -> bool:
        if self.passable_mask == "ALL":
            return True
        if self.passable_mask == "FLY_ONLY":
            return motion is MotionMode.FLY
        return False

    def can_deploy(self, position: str) -> bool:
        """``position`` is the operator's deployment restriction: MELEE / RANGED / ALL."""
        if self.buildable == BUILD_NONE:
            return False
        if self.buildable == BUILD_ALL or position == "ALL":
            return True
        return self.buildable == position


class BattleMap:
    """Immutable static map, built from ``level.mapData``."""

    __slots__ = ("height", "width", "_tiles", "_index", "starts", "ends", "_disallowed")

    def __init__(
        self,
        height: int,
        width: int,
        tiles: list[list[TileInfo]],
        disallowed: Iterable[Tile] = (),
    ) -> None:
        self.height = height
        self.width = width
        self._tiles = tiles                      # indexed [row][col], row from bottom
        self._disallowed = frozenset(disallowed)
        self.starts = tuple(t for t in self.all_tiles() if self[t].key.endswith("start"))
        self.ends = tuple(t for t in self.all_tiles() if self[t].key == "tile_end")

    # -- construction ----------------------------------------------------

    @classmethod
    def from_level(cls, level: dict[str, Any], novelty: "NoveltyLog | None" = None) -> BattleMap:
        md = level["mapData"]
        grid = md["map"]
        raw_tiles = md["tiles"]
        h, w = len(grid), len(grid[0])
        infos: list[list[TileInfo]] = []
        for row in range(h):
            display = h - 1 - row                # row counts from the bottom
            line = []
            for col in range(w):
                t = raw_tiles[grid[display][col]]
                bb = tuple(
                    (b["key"], float(b.get("value") or 0.0))
                    for b in (t.get("blackboard") or [])
                )
                line.append(
                    TileInfo(
                        key=t["tileKey"],
                        height=_enum(t.get("heightType"), _HEIGHT_BY_INT, "LOWLAND",
                                     "heightType", novelty),
                        buildable=_enum(t.get("buildableType"), _BUILDABLE_BY_INT, "NONE",
                                        "buildableType", novelty),
                        passable_mask=_enum(t.get("passableMask"), _PASSABLE_BY_INT, "NONE",
                                            "passableMask", novelty),
                        player_side_mask=_enum(t.get("playerSideMask"), _SIDE_BY_INT, "ALL",
                                               "playerSideMask", novelty),
                        blackboard=bb,
                    )
                )
            infos.append(line)
        disallowed = [
            Tile(p["row"], p["col"]) for p in (level.get("tilesDisallowToLocate") or [])
        ]
        return cls(h, w, infos, disallowed)

    # -- access ----------------------------------------------------------

    def __getitem__(self, t: Tile) -> TileInfo:
        return self._tiles[t.row][t.col]

    def in_bounds(self, t: Tile) -> bool:
        return 0 <= t.row < self.height and 0 <= t.col < self.width

    def all_tiles(self) -> Iterable[Tile]:
        for r in range(self.height):
            for c in range(self.width):
                yield Tile(r, c)

    def passable(self, t: Tile, motion: MotionMode) -> bool:
        return self.in_bounds(t) and self[t].passable_for(motion)

    def deployable(self, t: Tile, position: str) -> bool:
        """Can an operator with this deployment restriction stand here?"""
        return (
            self.in_bounds(t)
            and t not in self._disallowed
            and self[t].can_deploy(position)
        )

    def deployable_tiles(self, position: str) -> tuple[Tile, ...]:
        return tuple(t for t in self.all_tiles() if self.deployable(t, position))

    def neighbours(self, t: Tile, motion: MotionMode, diagonal: bool = True) -> Iterable[Tile]:
        deltas = _NEIGHBOURS_8 if diagonal else _NEIGHBOURS_4
        for dr, dc in deltas:
            n = Tile(t.row + dr, t.col + dc)
            if not self.passable(n, motion):
                continue
            if dr and dc:
                # No corner-cutting through a pair of impassable orthogonals.
                if not (
                    self.passable(Tile(t.row + dr, t.col), motion)
                    or self.passable(Tile(t.row, t.col + dc), motion)
                ):
                    continue
            yield n

    def ascii(self) -> str:
        """Debug rendering, top row first (i.e. as the player sees it)."""
        sym = {
            "tile_road": ".", "tile_wall": "#", "tile_forbidden": " ",
            "tile_start": "S", "tile_flystart": "s", "tile_end": "E",
            "tile_floor": "_", "tile_hole": "o",
        }
        lines = []
        for row in range(self.height - 1, -1, -1):
            line = []
            for col in range(self.width):
                info = self._tiles[row][col]
                ch = sym.get(info.key, "?")
                if ch == "." and info.buildable == BUILD_MELEE:
                    ch = "m"
                elif info.buildable == BUILD_RANGED:
                    ch = "r"
                line.append(ch)
            lines.append(f"{row:2d} " + "".join(line))
        lines.append("   " + "".join(str(c % 10) for c in range(self.width)))
        return "\n".join(lines)
