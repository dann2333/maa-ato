"""Screen <-> tile mapping: the piece that turns a plan into a tap.

Two routes to the same thing, because they fail differently:

* :class:`Homography` — fit a 3x3 projective map from >=4 (tile, screen point)
  correspondences. The battlefield is a plane, so a plane-to-plane projective map
  is *exact* for any pinhole camera; no camera parameters needed. This is what
  calibration from a screenshot produces.
* :class:`PerspectiveProjector` — the analytic path: build the same mapping from
  camera position/rotation/fov. Useful when the camera is known (the client uses
  a fixed rig per map) and as a generator of synthetic correspondences.

Tiles use the game's convention: ``(row, col)`` with ``row`` counted from the
bottom of the map. Screen points are ``(x, y)`` pixels with ``y`` down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ato.sim.types import Tile, Vec2

TileLike = Tile | tuple[float, float] | Vec2
Point = tuple[float, float]


class TileHit(NamedTuple):
    """Result of un-projecting a screen point.

    ``exact`` is the continuous tile position (x=col, y=row) and ``residual`` is
    how far that is from the centre of the nearest tile, in tiles: > 0.5 means the
    point is nearer a neighbour, > ~0.7 means it is not really on a tile at all.
    """

    tile: Tile
    exact: Vec2
    residual: float


def _as_colrow(tile: TileLike) -> tuple[float, float]:
    """Everything internally works in (u=col, v=row) so it matches screen x/y order."""
    if isinstance(tile, Tile):
        return (float(tile.col), float(tile.row))
    if isinstance(tile, Vec2):
        return (float(tile.x), float(tile.y))
    row, col = tile
    return (float(col), float(row))


def _hit(u: float, v: float) -> TileHit:
    tile = Tile(int(round(v)), int(round(u)))
    return TileHit(tile, Vec2(u, v), math.hypot(u - tile.col, v - tile.row))


@dataclass(frozen=True, eq=False)
class Homography:
    """A 3x3 projective map from tile coordinates to screen pixels."""

    matrix: np.ndarray
    #: RMS reprojection error of the fit, in pixels (0 for a matrix supplied directly).
    rms_error: float = 0.0
    inverse: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {m.shape}")
        if abs(m[2, 2]) > 1e-12:
            m = m / m[2, 2]  # fix the scale so matrices compare and print sanely
        if abs(np.linalg.det(m)) < 1e-12:
            raise ValueError("degenerate homography (are the calibration points collinear?)")
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "inverse", np.linalg.inv(m))

    # -- fitting ------------------------------------------------------------ #

    @classmethod
    def from_correspondences(
        cls,
        tiles: Sequence[TileLike],
        points: Sequence[Point],
    ) -> Homography:
        """Least-squares DLT fit. Needs >=4 correspondences, no 3 of them collinear."""
        if len(tiles) != len(points):
            raise ValueError(f"{len(tiles)} tiles vs {len(points)} points")
        if len(tiles) < 4:
            raise ValueError("a homography needs at least 4 correspondences")
        src = np.array([_as_colrow(t) for t in tiles], dtype=np.float64)
        dst = np.array([(float(x), float(y)) for x, y in points], dtype=np.float64)

        # Hartley normalisation: without it the 2n x 9 system is badly conditioned
        # (pixel coordinates are ~10^3, tile coordinates ~10^0) and the smallest
        # singular vector is dominated by rounding.
        t_src, src_n = _normalise(src)
        t_dst, dst_n = _normalise(dst)

        n = len(src_n)
        a = np.zeros((2 * n, 9), dtype=np.float64)
        for i, ((u, v), (x, y)) in enumerate(zip(src_n, dst_n, strict=True)):
            a[2 * i] = (-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x)
            a[2 * i + 1] = (0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y)
        # Smallest right singular vector = the null-space direction of A.
        _, _, vt = np.linalg.svd(a)
        h_norm = vt[-1].reshape(3, 3)
        matrix = np.linalg.inv(t_dst) @ h_norm @ t_src

        fit = cls(matrix)
        return cls(matrix, rms_error=fit.reprojection_error(tiles, points)[1])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Homography:
        return cls(np.asarray(matrix, dtype=np.float64))

    # -- use ---------------------------------------------------------------- #

    def tile_to_screen(self, tile: TileLike) -> Point:
        u, v = _as_colrow(tile)
        x, y, w = self.matrix @ np.array((u, v, 1.0))
        if abs(w) < 1e-12:
            raise ValueError(f"tile {tile} projects to infinity (behind the camera?)")
        return (float(x / w), float(y / w))

    def screen_to_tile(self, x: float, y: float) -> TileHit:
        u, v, w = self.inverse @ np.array((float(x), float(y), 1.0))
        if abs(w) < 1e-12:
            raise ValueError(f"screen point ({x}, {y}) does not map onto the map plane")
        return _hit(float(u / w), float(v / w))

    def tiles_to_screen(self, tiles: Sequence[TileLike]) -> np.ndarray:
        """Vectorised projection, ``(n, 2)`` pixels — the agent projects whole plans."""
        src = np.array([(*_as_colrow(t), 1.0) for t in tiles], dtype=np.float64)
        out = src @ self.matrix.T
        return out[:, :2] / out[:, 2:3]

    def reprojection_error(
        self,
        tiles: Sequence[TileLike],
        points: Sequence[Point],
    ) -> tuple[float, float]:
        """``(max, rms)`` pixel error — the number that says a calibration is good."""
        pred = self.tiles_to_screen(tiles)
        err = np.linalg.norm(pred - np.asarray(points, dtype=np.float64), axis=1)
        return (float(err.max()), float(np.sqrt((err**2).mean())))


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity transform putting the centroid at 0 and mean radius at sqrt(2)."""
    centre = pts.mean(axis=0)
    shifted = pts - centre
    mean_dist = float(np.sqrt((shifted**2).sum(axis=1)).mean())
    scale = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    t = np.array(
        [[scale, 0.0, -scale * centre[0]], [0.0, scale, -scale * centre[1]], [0.0, 0.0, 1.0]]
    )
    return t, shifted * scale


def fit_homography_from_corners(
    bottom_left: Point,
    bottom_right: Point,
    top_left: Point,
    top_right: Point,
    *,
    map_height: int,
    map_width: int,
) -> Homography:
    """Fit from the four corner *tile centres* picked off a screenshot.

    ``bottom_left`` is the screen position of tile ``(0, 0)`` — row 0 is the
    bottom row of the map, as everywhere else in ATO.
    """
    if map_height < 2 or map_width < 2:
        raise ValueError("need a map at least 2x2 to fit from corners")
    tiles = [
        Tile(0, 0),
        Tile(0, map_width - 1),
        Tile(map_height - 1, 0),
        Tile(map_height - 1, map_width - 1),
    ]
    return Homography.from_correspondences(tiles, [bottom_left, bottom_right, top_left, top_right])


# --------------------------------------------------------------------------- #
# analytic camera
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraParams:
    """The battle camera.

    These are **calibration inputs**: the client uses a fixed rig whose height,
    pitch and fov depend on the map size and the viewport aspect ratio, and the
    player can pan/zoom. Fit them per map/aspect (or skip this class and fit a
    :class:`Homography` straight off a screenshot, which absorbs all of it).
    """

    #: Camera position in world units (1 unit = 1 tile), y up. The default frames
    #: a ~13x9 map at 16:9 with the usual downward tilt; it is a starting point
    #: for a fit, not a measurement of the client.
    position: tuple[float, float, float] = (0.0, 13.5, 19.2)
    #: Euler angles in degrees: pitch (negative looks down), yaw, roll.
    rotation: tuple[float, float, float] = (-35.0, 0.0, 0.0)
    #: Vertical field of view in degrees.
    fov_y: float = 20.0
    viewport: tuple[int, int] = (1920, 1080)
    near: float = 0.3
    far: float = 1000.0

    @property
    def aspect(self) -> float:
        return self.viewport[0] / self.viewport[1]


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Camera-to-world rotation, applied yaw * pitch * roll (Unity's order)."""
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    return ry @ rx @ rz


class PerspectiveProjector:
    """Project tile centres through an explicit view + projection matrix.

    World layout: the map lies on the ``y = 0`` plane, centred on the origin, with
    ``x`` = column (right) and ``z`` = -row, so a higher row is further from a
    camera that sits on +z and looks down -z — i.e. higher rows are higher up the
    screen, matching the client.
    """

    __slots__ = ("camera", "map_height", "map_width", "view", "projection", "_vp")

    def __init__(self, camera: CameraParams, map_size: tuple[int, int]) -> None:
        self.camera = camera
        self.map_height, self.map_width = map_size
        self.view = self._view_matrix()
        self.projection = self._projection_matrix()
        self._vp = self.projection @ self.view

    # -- matrices ----------------------------------------------------------- #

    def _view_matrix(self) -> np.ndarray:
        r = _rotation_matrix(*self.camera.rotation)
        eye = np.asarray(self.camera.position, dtype=np.float64)
        view = np.eye(4)
        view[:3, :3] = r.T            # world -> camera is the inverse rotation
        view[:3, 3] = -r.T @ eye
        return view

    def _projection_matrix(self) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.camera.fov_y) / 2.0)
        n, fa = self.camera.near, self.camera.far
        proj = np.zeros((4, 4))
        proj[0, 0] = f / self.camera.aspect
        proj[1, 1] = f
        proj[2, 2] = (fa + n) / (n - fa)
        proj[2, 3] = (2 * fa * n) / (n - fa)
        proj[3, 2] = -1.0
        return proj

    # -- world <-> tile ----------------------------------------------------- #

    def tile_to_world(self, tile: TileLike) -> np.ndarray:
        u, v = _as_colrow(tile)
        return np.array(
            [u - (self.map_width - 1) / 2.0, 0.0, -(v - (self.map_height - 1) / 2.0)],
            dtype=np.float64,
        )

    def world_to_tile(self, world: np.ndarray) -> tuple[float, float]:
        u = float(world[0]) + (self.map_width - 1) / 2.0
        v = -float(world[2]) + (self.map_height - 1) / 2.0
        return (u, v)

    # -- projection --------------------------------------------------------- #

    def project(self, world: Sequence[float]) -> Point:
        w, h = self.camera.viewport
        clip = self._vp @ np.array([world[0], world[1], world[2], 1.0])
        if clip[3] <= 1e-9:
            raise ValueError(f"point {tuple(world)} is behind the camera")
        ndc = clip[:3] / clip[3]
        return (float((ndc[0] + 1.0) * 0.5 * w), float((1.0 - ndc[1]) * 0.5 * h))

    def tile_to_screen(self, tile: TileLike) -> Point:
        return self.project(self.tile_to_world(tile))

    def ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Camera-space ray through a pixel, as ``(origin, unit direction)``."""
        w, h = self.camera.viewport
        ndc_x = 2.0 * x / w - 1.0
        ndc_y = 1.0 - 2.0 * y / h
        inv = np.linalg.inv(self._vp)
        near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        d = far - near
        return near, d / np.linalg.norm(d)

    def screen_to_tile(self, x: float, y: float) -> TileHit:
        origin, direction = self.ray(x, y)
        if abs(direction[1]) < 1e-9:
            raise ValueError("ray is parallel to the ground plane")
        t = -origin[1] / direction[1]
        if t <= 0.0:
            raise ValueError(f"screen point ({x}, {y}) looks away from the battlefield")
        u, v = self.world_to_tile(origin + direction * t)
        return _hit(u, v)

    # -- bridge ------------------------------------------------------------- #

    def to_homography(self) -> Homography:
        """The equivalent plane-to-plane map.

        Exact, not an approximation: a pinhole camera restricted to one plane *is*
        a homography. Fitting it from the corners lets the rest of the agent use a
        single 3x3 matrix regardless of where the mapping came from.
        """
        tiles = [
            Tile(0, 0),
            Tile(0, self.map_width - 1),
            Tile(self.map_height - 1, 0),
            Tile(self.map_height - 1, self.map_width - 1),
        ]
        return Homography.from_correspondences(tiles, [self.tile_to_screen(t) for t in tiles])


if __name__ == "__main__":
    # A homography fitted to a synthetic perspective view must reproduce that view
    # everywhere, not just at the four corners it saw.
    cam = CameraParams(
        position=(0.7, 9.5, 8.5), rotation=(-38.0, 5.0, 1.5), fov_y=22.0, viewport=(1920, 1080)
    )
    proj = PerspectiveProjector(cam, map_size=(9, 13))
    grid = [Tile(r, c) for r in range(9) for c in range(13)]
    truth = [proj.tile_to_screen(t) for t in grid]

    h = fit_homography_from_corners(
        proj.tile_to_screen(Tile(0, 0)),
        proj.tile_to_screen(Tile(0, 12)),
        proj.tile_to_screen(Tile(8, 0)),
        proj.tile_to_screen(Tile(8, 12)),
        map_height=9,
        map_width=13,
    )
    max_px, rms_px = h.reprojection_error(grid, truth)

    worst = 0.0
    for tile, point in zip(grid, truth, strict=True):
        hit = h.screen_to_tile(*point)
        worst = max(worst, math.hypot(hit.exact.x - tile.col, hit.exact.y - tile.row))
        assert hit.tile == tile, f"{point} -> {hit.tile}, expected {tile}"
        analytic = proj.screen_to_tile(*point)
        assert analytic.tile == tile, f"analytic un-projection gave {analytic.tile} for {tile}"

    assert worst < 0.05, f"round-trip error {worst:.4f} tiles"
    print(f"corner fit:   max {max_px:.4f} px, rms {rms_px:.4f} px")
    print(f"round-trip:   max {worst:.6f} tiles (limit 0.05)")

    # Over-determined fit with hand-clicked-quality noise: this is the path real
    # calibration takes, and it is where least squares has to earn its keep.
    rng = np.random.default_rng(20250904)
    jitter = rng.normal(0.0, 1.5, (len(truth), 2))
    noisy = [(x + n[0], y + n[1]) for (x, y), n in zip(truth, jitter, strict=True)]
    hn = Homography.from_correspondences(grid, noisy)
    worst_noisy = 0.0
    for tile, point in zip(grid, truth, strict=True):
        hit = hn.screen_to_tile(*point)
        worst_noisy = max(worst_noisy, math.hypot(hit.exact.x - tile.col, hit.exact.y - tile.row))
    assert worst_noisy < 0.05, f"noisy round-trip error {worst_noisy:.4f} tiles"
    print(f"noisy fit:    rms {hn.rms_error:.3f} px, round-trip max {worst_noisy:.4f} tiles")
    print("projection self-test OK")
