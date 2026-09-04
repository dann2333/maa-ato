"""Turning a route definition into a concrete path of tile waypoints.

A route is a start tile, an end tile and an ordered list of checkpoints. Only
``MOVE``/``PATROL_MOVE`` checkpoints are positions to walk to; the rest are
timing or visibility instructions that the mover executes in order. Between two
successive positions the enemy walks the shortest tile path, which is what the
client does — it does not walk in a straight line through walls.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable

from ato.sim.grid import BattleMap
from ato.sim.registry import CHECKPOINTS, MechanismKind, NoveltyLog, checkpoint_type
from ato.sim.types import MotionMode, Tile, Vec2

#: Checkpoint kinds that name a destination the unit walks to.
MOVE_KINDS = frozenset({"MOVE", "PATROL_MOVE"})
#: Checkpoint kinds that pause the unit for a duration or until a condition.
WAIT_KINDS = frozenset({
    "WAIT_FOR_SECONDS",
    "WAIT_FOR_PLAY_TIME",
    "WAIT_CURRENT_FRAGMENT_TIME",
    "WAIT_CURRENT_WAVE_TIME",
    "WAIT_BOSSRUSH_WAVE",
})
#: Checkpoint kinds that teleport or hide the unit.
JUMP_KINDS = frozenset({"APPEAR_AT_POS", "DISAPPEAR"})

for _k in MOVE_KINDS | WAIT_KINDS | JUMP_KINDS:
    CHECKPOINTS.register(_k)(lambda *_a, **_k2: None)
CHECKPOINTS.ignore("ALERT", "purely a visual/audio cue in the client")


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One step of a route, normalised."""

    kind: str
    position: Tile
    time: float = 0.0
    reach_offset: Vec2 = Vec2(0.0, 0.0)
    reach_distance: float = 0.0

    @property
    def is_move(self) -> bool:
        return self.kind in MOVE_KINDS

    @property
    def is_wait(self) -> bool:
        return self.kind in WAIT_KINDS


@dataclass(frozen=True)
class RouteSpec:
    """A normalised movement route."""

    index: int
    motion: MotionMode
    start: Tile
    end: Tile
    checkpoints: tuple[Checkpoint, ...]
    allow_diagonal: bool = True
    spawn_offset: Vec2 = Vec2(0.0, 0.0)
    spawn_random_range: Vec2 = Vec2(0.0, 0.0)
    visit_every_tile_center: bool = False

    @property
    def spawn_position(self) -> Vec2:
        return self.start.xy + self.spawn_offset


def parse_route(index: int, raw: dict[str, Any], novelty: NoveltyLog) -> RouteSpec | None:
    """Normalise one ``level.routes`` entry. Returns ``None`` for placeholder slots."""
    if raw is None:
        return None
    from ato.sim.registry import motion_mode as _motion

    mode = _motion(raw.get("motionMode"))
    if raw.get("motionMode") in (None, "E_NUM") and not raw.get("checkpoints"):
        # Placeholder slot: the client pads the routes array. Not an error.
        if raw.get("startPosition") == raw.get("endPosition") == {"row": 0, "col": 0}:
            return None

    cps: list[Checkpoint] = []
    for c in raw.get("checkpoints") or ():
        if not c:
            continue
        kind = checkpoint_type(c.get("type"))
        if not novelty.check(CHECKPOINTS, kind, f"route {index}"):
            continue
        pos = c.get("position") or {"row": 0, "col": 0}
        off = c.get("reachOffset") or {"x": 0.0, "y": 0.0}
        cps.append(
            Checkpoint(
                kind=kind,
                position=Tile(pos["row"], pos["col"]),
                time=float(c.get("time") or 0.0),
                reach_offset=Vec2(float(off.get("x") or 0.0), float(off.get("y") or 0.0)),
                reach_distance=float(c.get("reachDistance") or 0.0),
            )
        )

    sp = raw.get("startPosition") or {"row": 0, "col": 0}
    ep = raw.get("endPosition") or {"row": 0, "col": 0}
    so = raw.get("spawnOffset") or {"x": 0.0, "y": 0.0}
    sr = raw.get("spawnRandomRange") or {"x": 0.0, "y": 0.0}
    return RouteSpec(
        index=index,
        motion=MotionMode.FLY if mode == "FLY" else MotionMode.WALK,
        start=Tile(sp["row"], sp["col"]),
        end=Tile(ep["row"], ep["col"]),
        checkpoints=tuple(cps),
        allow_diagonal=bool(raw.get("allowDiagonalMove", True)),
        spawn_offset=Vec2(float(so.get("x") or 0.0), float(so.get("y") or 0.0)),
        spawn_random_range=Vec2(float(sr.get("x") or 0.0), float(sr.get("y") or 0.0)),
        visit_every_tile_center=bool(raw.get("visitEveryTileCenter", False)),
    )


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

_SQRT2 = 2 ** 0.5


def shortest_path(
    bmap: BattleMap,
    src: Tile,
    dst: Tile,
    motion: MotionMode,
    *,
    diagonal: bool = True,
    blocked: frozenset[Tile] = frozenset(),
) -> tuple[Tile, ...]:
    """A* over the tile graph, returning tiles from ``src`` to ``dst`` inclusive.

    ``blocked`` lets the caller exclude tiles (used for mechanics that seal a
    route). Returns an empty tuple when no path exists — callers must handle
    that rather than assume connectivity, because some maps genuinely fence off
    regions until an event opens them.
    """
    if src == dst:
        return (src,)
    if not bmap.passable(dst, motion) or dst in blocked:
        return ()

    def h(t: Tile) -> float:
        dr, dc = abs(t.row - dst.row), abs(t.col - dst.col)
        if diagonal:
            lo, hi = min(dr, dc), max(dr, dc)
            return lo * _SQRT2 + (hi - lo)
        return dr + dc

    open_heap: list[tuple[float, int, Tile]] = [(h(src), 0, src)]
    came: dict[Tile, Tile] = {}
    g: dict[Tile, float] = {src: 0.0}
    counter = 1
    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == dst:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return tuple(reversed(path))
        for nb in bmap.neighbours(cur, motion, diagonal):
            if nb in blocked:
                continue
            step = _SQRT2 if (nb.row != cur.row and nb.col != cur.col) else 1.0
            ng = g[cur] + step
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), counter, nb))
                counter += 1
    return ()


def straight_line(src: Tile, dst: Tile) -> tuple[Tile, ...]:
    """Flying units ignore the tile graph and go straight."""
    return (src, dst)


def route_waypoints(
    bmap: BattleMap,
    route: RouteSpec,
    *,
    blocked: frozenset[Tile] = frozenset(),
) -> tuple[Vec2, ...]:
    """Expand a route into the polyline a unit actually walks.

    Waits and teleports are dropped here — this is the *geometric* path only;
    :class:`~ato.sim.entities.EnemyUnit` replays the checkpoint list itself so
    that timing checkpoints keep their meaning.
    """
    stops: list[Tile] = [route.start]
    for cp in route.checkpoints:
        if cp.is_move or cp.kind == "APPEAR_AT_POS":
            stops.append(cp.position)
    stops.append(route.end)

    pts: list[Vec2] = [route.spawn_position]
    for a, b in zip(stops, stops[1:]):
        if a == b:
            continue
        if route.motion is MotionMode.FLY:
            seg: Iterable[Tile] = straight_line(a, b)[1:]
        else:
            path = shortest_path(bmap, a, b, route.motion, diagonal=route.allow_diagonal,
                                 blocked=blocked)
            seg = path[1:] if path else (b,)
        for t in seg:
            pts.append(t.xy)
    return tuple(pts)
