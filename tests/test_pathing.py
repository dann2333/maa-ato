"""Route parsing, A* over the tile graph, and route expansion."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from conftest import (
    FLY_ROUTE,
    PLACEHOLDER_ROUTE,
    SQRT2,
    WALK_ROUTE,
    assert_valid_path,
    path_cost,
)

from ato.sim.grid import BattleMap
from ato.sim.pathing import (
    DisappearStep,
    MoveStep,
    TeleportStep,
    WaitStep,
    build_route_program,
    parse_route,
    route_waypoints,
    shortest_path,
)
from ato.sim.registry import NoveltyLog, UnknownMechanism
from ato.sim.types import MotionMode, Tile, Vec2

WALK, FLY = MotionMode.WALK, MotionMode.FLY


@pytest.fixture
def log() -> NoveltyLog:
    return NoveltyLog(strict=False)


# ---------------------------------------------------------------------------
# parse_route
# ---------------------------------------------------------------------------


def test_parse_route_placeholder_slot_is_not_an_error(log: NoveltyLog) -> None:
    assert parse_route(0, PLACEHOLDER_ROUTE, log) is None
    assert parse_route(0, None, log) is None
    assert log.clean


def test_parse_route_normalises_int_enums(log: NoveltyLog) -> None:
    route = parse_route(1, WALK_ROUTE, log)
    assert route is not None
    assert route.index == 1
    assert route.motion is WALK                       # motionMode 0
    assert route.start == Tile(0, 0)
    assert route.end == Tile(0, 6)
    assert [c.kind for c in route.checkpoints] == ["MOVE", "WAIT_FOR_SECONDS", "MOVE"]
    assert route.checkpoints[0].position == Tile(3, 1)
    assert route.checkpoints[0].is_move
    assert route.checkpoints[1].is_wait
    assert route.checkpoints[1].time == 2.0
    assert route.checkpoints[2].position == Tile(2, 3)
    assert route.allow_diagonal
    assert log.clean


def test_parse_route_spawn_offset(log: NoveltyLog) -> None:
    route = parse_route(1, WALK_ROUTE, log)
    assert route is not None
    assert route.spawn_offset == Vec2(0.0, -0.5)
    assert route.spawn_position == Vec2(0.0, -0.5)    # start (0,0) shifted by the offset


def test_parse_route_flying(log: NoveltyLog) -> None:
    route = parse_route(2, FLY_ROUTE, log)
    assert route is not None
    assert route.motion is FLY                        # motionMode 1
    assert route.checkpoints == ()
    assert route.start == Tile(3, 0)


def test_parse_route_unknown_checkpoint_is_dropped_and_reported(log: NoveltyLog) -> None:
    raw = copy.deepcopy(WALK_ROUTE)
    raw["checkpoints"][1]["type"] = 99                # a mechanic we have never seen
    route = parse_route(1, raw, log)
    assert route is not None
    assert [c.kind for c in route.checkpoints] == ["MOVE", "MOVE"]
    assert [e.key for e in log.events] == ["UNKNOWN_99"]
    assert "route 1" in log.events[0].context


def test_parse_route_unknown_checkpoint_raises_in_strict_mode() -> None:
    raw = copy.deepcopy(WALK_ROUTE)
    raw["checkpoints"][1]["type"] = 99
    with pytest.raises(UnknownMechanism):
        parse_route(1, raw, NoveltyLog(strict=True))


def test_parse_route_keeps_allow_diagonal_false(log: NoveltyLog) -> None:
    raw: dict[str, Any] = copy.deepcopy(WALK_ROUTE)
    raw["allowDiagonalMove"] = False
    route = parse_route(1, raw, log)
    assert route is not None
    assert not route.allow_diagonal


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------


def test_shortest_path_to_self(arena: BattleMap) -> None:
    assert shortest_path(arena, Tile(0, 0), Tile(0, 0), WALK) == (Tile(0, 0),)


def test_shortest_path_takes_the_obvious_diagonal_route(arena: BattleMap) -> None:
    src, dst = Tile(0, 0), Tile(3, 2)
    path = shortest_path(arena, src, dst, WALK)
    assert_valid_path(arena, path, src, dst, WALK)
    assert len(path) == 4                             # three steps: two diagonal, one straight
    assert path_cost(path) == pytest.approx(2 * SQRT2 + 1)


def test_shortest_path_threads_the_only_door(arena: BattleMap) -> None:
    src, dst = Tile(0, 0), Tile(0, 6)
    path = shortest_path(arena, src, dst, WALK)
    assert_valid_path(arena, path, src, dst, WALK)
    assert Tile(2, 3) in path                         # the single gap in the wall column
    assert len(path) == 7
    assert path_cost(path) == pytest.approx(4 * SQRT2 + 2)


def test_shortest_path_returns_empty_when_walled_off(arena: BattleMap) -> None:
    # Sealing the door disconnects the two halves for anything that walks.
    assert shortest_path(arena, Tile(0, 0), Tile(0, 6), WALK,
                         blocked=frozenset({Tile(2, 3)})) == ()


def test_shortest_path_returns_empty_for_an_impassable_destination(arena: BattleMap) -> None:
    assert shortest_path(arena, Tile(0, 0), Tile(1, 6), WALK) == ()   # tile_forbidden
    assert shortest_path(arena, Tile(0, 0), Tile(1, 6), FLY) == ()
    assert shortest_path(arena, Tile(0, 0), Tile(4, 0), WALK) == ()   # highland


def test_shortest_path_respects_blocked_tiles(arena: BattleMap) -> None:
    src, dst = Tile(0, 0), Tile(0, 2)
    straight = shortest_path(arena, src, dst, WALK)
    assert straight == (Tile(0, 0), Tile(0, 1), Tile(0, 2))

    detour = shortest_path(arena, src, dst, WALK, blocked=frozenset({Tile(0, 1)}))
    assert_valid_path(arena, detour, src, dst, WALK)
    assert Tile(0, 1) not in detour
    assert len(detour) == 3                           # up-right then down-right

    wall = frozenset({Tile(r, 1) for r in range(4)})
    assert shortest_path(arena, src, dst, WALK, blocked=wall) == ()


def test_shortest_path_flies_over_the_wall(arena: BattleMap) -> None:
    src, dst = Tile(0, 0), Tile(0, 6)
    path = shortest_path(arena, src, dst, FLY)
    assert_valid_path(arena, path, src, dst, FLY)
    assert tuple(t.col for t in path) == (0, 1, 2, 3, 4, 5, 6)
    assert {t.row for t in path} == {0}               # straight along the bottom row
    assert Tile(0, 3) in path                         # the wall column itself


def test_shortest_path_orthogonal_is_longer_than_diagonal(arena: BattleMap) -> None:
    src, dst = Tile(0, 0), Tile(3, 2)
    ortho = shortest_path(arena, src, dst, WALK, diagonal=False)
    assert_valid_path(arena, ortho, src, dst, WALK)
    assert len(ortho) == 6                            # 5 orthogonal steps
    assert all(a.row == b.row or a.col == b.col
               for a, b in zip(ortho, ortho[1:], strict=False))
    assert len(shortest_path(arena, src, dst, WALK, diagonal=True)) == 4


def test_shortest_path_orthogonal_through_the_door(arena: BattleMap) -> None:
    path = shortest_path(arena, Tile(0, 0), Tile(0, 6), WALK, diagonal=False)
    assert len(path) == 11
    assert Tile(2, 3) in path


# ---------------------------------------------------------------------------
# route_waypoints
# ---------------------------------------------------------------------------


def test_route_waypoints_threads_checkpoints_in_order(
    arena: BattleMap, log: NoveltyLog
) -> None:
    route = parse_route(1, WALK_ROUTE, log)
    assert route is not None
    pts = route_waypoints(arena, route)

    assert pts[0] == route.spawn_position
    assert pts[-1] == Vec2(6.0, 0.0)                  # the end tile
    first_cp, door = Vec2(1.0, 3.0), Vec2(3.0, 2.0)   # x=col, y=row
    assert pts.index(first_cp) < pts.index(door) < len(pts) - 1
    # A WAIT checkpoint carries position (0,0); dropping it must not send the
    # unit back to the spawn tile.
    assert Vec2(0.0, 0.0) not in pts
    assert len(pts) == 9

    for a, b in zip(pts[1:], pts[2:], strict=False):
        assert max(abs(a.x - b.x), abs(a.y - b.y)) == 1.0


def test_route_waypoints_fly_straight_through_the_wall(
    arena: BattleMap, log: NoveltyLog
) -> None:
    route = parse_route(2, FLY_ROUTE, log)
    assert route is not None
    # No tile graph for fliers: spawn, then the destination.
    assert route_waypoints(arena, route) == (Vec2(0.0, 3.0), Vec2(6.0, 0.0))


def test_route_waypoints_falls_back_to_a_direct_hop_when_disconnected(
    arena: BattleMap, log: NoveltyLog
) -> None:
    route = parse_route(1, WALK_ROUTE, log)
    assert route is not None
    pts = route_waypoints(arena, route, blocked=frozenset({Tile(2, 3)}))
    # The door is sealed, so the last two segments degenerate into hops. This is
    # documented behaviour, not a path — the caller's calibration must notice.
    assert pts[-1] == Vec2(6.0, 0.0)
    assert Vec2(3.0, 2.0) in pts


# ---------------------------------------------------------------------------
# build_route_program
# ---------------------------------------------------------------------------


def test_build_route_program_keeps_waits_between_moves(
    arena: BattleMap, log: NoveltyLog
) -> None:
    route = parse_route(1, WALK_ROUTE, log)
    assert route is not None
    steps = build_route_program(arena, route)
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["MoveStep", "WaitStep", "MoveStep", "MoveStep"]
    wait = steps[1]
    assert isinstance(wait, WaitStep)
    assert (wait.seconds, wait.kind) == (2.0, "WAIT_FOR_SECONDS")
    first, last = steps[0], steps[-1]
    assert isinstance(first, MoveStep) and isinstance(last, MoveStep)
    assert first.points[-1] == Vec2(1.0, 3.0)         # stopped on the checkpoint
    assert last.points[-1] == Vec2(6.0, 0.0)


def test_build_route_program_teleports_and_disappears(
    arena: BattleMap, log: NoveltyLog
) -> None:
    raw = copy.deepcopy(WALK_ROUTE)
    raw["checkpoints"] = [
        {"type": 5, "time": 0.0, "position": {"row": 0, "col": 0},
         "reachOffset": {"x": 0.0, "y": 0.0}, "reachDistance": 0.0},        # DISAPPEAR
        {"type": 6, "time": 0.0, "position": {"row": 3, "col": 5},
         "reachOffset": {"x": 0.5, "y": 0.0}, "reachDistance": 0.0},        # APPEAR_AT_POS
    ]
    route = parse_route(1, raw, log)
    assert route is not None
    steps = build_route_program(arena, route)
    assert isinstance(steps[0], DisappearStep)
    assert isinstance(steps[1], TeleportStep)
    assert steps[1].position == Vec2(5.5, 3.0)        # position + reachOffset
    # After the teleport the unit walks on from where it reappeared.
    assert isinstance(steps[2], MoveStep)
    assert steps[2].points[-1] == Vec2(6.0, 0.0)
