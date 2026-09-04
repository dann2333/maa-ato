"""BattleMap construction, the row convention, buildability and the move graph."""

from __future__ import annotations

import pytest
from conftest import ARENA_ROWS, TILE_LEGEND, make_level

from ato.sim.grid import BattleMap, TileInfo
from ato.sim.types import MotionMode, Tile

WALK, FLY = MotionMode.WALK, MotionMode.FLY


# ---------------------------------------------------------------------------
# The row convention — the single most expensive thing to get wrong
# ---------------------------------------------------------------------------


def test_tile_row_zero_is_the_bottom_display_row(arena: BattleMap) -> None:
    # ARENA_ROWS is written top row first, exactly like mapData.map. The bottom
    # line "S..#..E" must therefore be row 0 and the top line "#######" row 4.
    assert (arena.height, arena.width) == (5, 7)
    assert arena[Tile(0, 0)].key == "tile_start"
    assert arena[Tile(0, 6)].key == "tile_end"
    assert arena[Tile(4, 0)].key == "tile_wall"
    assert arena[Tile(4, 6)].key == "tile_wall"
    # Reading the file top-first would put the wall row at row 0.
    assert arena[Tile(0, 0)].key != arena[Tile(4, 0)].key


def test_row_convention_holds_for_every_row(arena: BattleMap) -> None:
    for row in range(arena.height):
        line = ARENA_ROWS[arena.height - 1 - row]
        for col, ch in enumerate(line):
            assert arena[Tile(row, col)].key == TILE_LEGEND[ch]["tileKey"]


def test_starts_and_ends(arena: BattleMap) -> None:
    # Both tile_start and tile_flystart count as spawn tiles.
    assert arena.starts == (Tile(0, 0), Tile(3, 0))
    assert arena.ends == (Tile(0, 6),)


def test_ascii_renders_top_row_first(arena: BattleMap) -> None:
    lines = arena.ascii().splitlines()
    # Rows are printed high-to-low and annotated with their tile row, so the
    # first line is the top of the map and the last is the column ruler.
    assert lines[0] == " 4 rrrrrrr"      # highland wall row: ranged-only
    assert lines[4] == " 0 SmmrmmE"      # bottom row: start, melee road, end
    assert lines[-1] == "   0123456"


def test_tile_blackboard_is_parsed() -> None:
    bmap = BattleMap.from_level(make_level(("B.", "..")))
    assert bmap[Tile(1, 0)].key == "tile_telin"
    assert bmap[Tile(1, 0)].blackboard == (("to_key", 3.0),)
    assert bmap[Tile(0, 0)].blackboard == ()


# ---------------------------------------------------------------------------
# Passability
# ---------------------------------------------------------------------------


def test_passability_walk_vs_fly(arena: BattleMap) -> None:
    road, wall, forbidden = Tile(0, 1), Tile(4, 3), Tile(1, 6)
    assert arena[wall].key == "tile_wall" and arena[wall].is_highland
    assert arena[forbidden].key == "tile_forbidden"

    assert arena.passable(road, WALK) and arena.passable(road, FLY)
    assert not arena.passable(wall, WALK)
    assert arena.passable(wall, FLY)          # FLY_ONLY
    assert not arena.passable(forbidden, WALK)
    assert not arena.passable(forbidden, FLY)  # NONE blocks everything


def test_passable_is_false_out_of_bounds(arena: BattleMap) -> None:
    for tile in (Tile(-1, 0), Tile(0, -1), Tile(5, 0), Tile(0, 7)):
        assert not arena.in_bounds(tile)
        assert not arena.passable(tile, FLY)


def test_tile_info_passable_for() -> None:
    assert TileInfo("t", "LOWLAND", "NONE", "ALL").passable_for(WALK)
    assert not TileInfo("t", "HIGHLAND", "NONE", "FLY_ONLY").passable_for(WALK)
    assert TileInfo("t", "HIGHLAND", "NONE", "FLY_ONLY").passable_for(FLY)
    assert not TileInfo("t", "HIGHLAND", "NONE", "NONE").passable_for(FLY)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tile", "melee", "ranged"),
    [
        (Tile(0, 1), True, False),    # tile_road   -> MELEE
        (Tile(4, 2), False, True),    # tile_wall   -> RANGED highland
        (Tile(2, 3), True, True),     # tile_floor  -> ALL
        (Tile(0, 0), False, False),   # tile_start  -> NONE
        (Tile(0, 6), False, False),   # tile_end    -> NONE
        (Tile(1, 6), False, False),   # tile_forbidden
    ],
)
def test_deployable_respects_buildable_type(
    arena: BattleMap, tile: Tile, melee: bool, ranged: bool
) -> None:
    assert arena.deployable(tile, "MELEE") is melee
    assert arena.deployable(tile, "RANGED") is ranged
    # An operator with no restriction can stand anywhere that is buildable.
    assert arena.deployable(tile, "ALL") is (melee or ranged)


def test_deployable_tiles(arena: BattleMap) -> None:
    ranged = arena.deployable_tiles("RANGED")
    # Every highland tile: the top row and the wall column, plus the door,
    # which is buildable by anyone.
    assert set(ranged) == {Tile(4, c) for c in range(7)} | {Tile(r, 3) for r in range(4)}
    assert len(arena.deployable_tiles("MELEE")) == 21


def test_tiles_disallowed_to_locate_are_excluded() -> None:
    level = make_level(ARENA_ROWS, tilesDisallowToLocate=[{"row": 0, "col": 1}])
    bmap = BattleMap.from_level(level)
    assert bmap[Tile(0, 1)].can_deploy("MELEE")      # buildable...
    assert not bmap.deployable(Tile(0, 1), "MELEE")  # ...but fenced off anyway
    assert len(bmap.deployable_tiles("MELEE")) == 20


def test_deployable_is_false_out_of_bounds(arena: BattleMap) -> None:
    assert not arena.deployable(Tile(9, 9), "ALL")


# ---------------------------------------------------------------------------
# The movement graph
# ---------------------------------------------------------------------------


def test_neighbours_refuse_to_cut_a_corner(corner: BattleMap) -> None:
    # (0,0)'s orthogonals are a wall (1,0) and a forbidden tile (0,1); the
    # diagonal (1,1) is open but squeezing between the two is not allowed.
    assert set(corner.neighbours(Tile(0, 0), WALK)) == set()
    # For a flier the wall *is* passable, so the same diagonal becomes legal.
    assert set(corner.neighbours(Tile(0, 0), FLY)) == {Tile(1, 0), Tile(1, 1)}


def test_neighbours_allow_a_diagonal_with_one_open_orthogonal(corner: BattleMap) -> None:
    # (0,2): (0,1) is forbidden but (1,2) is open, so (1,1) is reachable.
    assert set(corner.neighbours(Tile(0, 2), WALK)) == {Tile(1, 2), Tile(1, 1)}


def test_neighbours_orthogonal_only(corner: BattleMap) -> None:
    assert set(corner.neighbours(Tile(0, 2), WALK, diagonal=False)) == {Tile(1, 2)}
    assert set(corner.neighbours(Tile(1, 1), WALK, diagonal=False)) == {
        Tile(1, 2), Tile(2, 1),
    }


def test_neighbours_stay_in_bounds(arena: BattleMap) -> None:
    for tile in arena.neighbours(Tile(0, 0), FLY):
        assert arena.in_bounds(tile)
    assert set(arena.neighbours(Tile(0, 0), WALK)) == {Tile(0, 1), Tile(1, 0), Tile(1, 1)}
