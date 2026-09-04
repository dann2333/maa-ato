"""Arknights input choreography, in screen coordinates.

Every gesture is *built* as a list of primitives before anything is sent, so a
plan can be inspected, unit-tested and replayed with no device attached. The
device only ever sees taps, swipes and long presses (see :mod:`ato.control.base`).

The delays and offsets below are **calibration parameters**, not physics: they
were chosen to match the client's deploy/menu animations at 1x speed and want
re-measuring per emulator, per game version and per battle speed. They live here
as named constants precisely so calibration is a one-file change.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ato.control.base import DeviceController
from ato.sim.types import Direction

# --- calibration: timings (milliseconds) ----------------------------------- #
#: Dwell on the card before dragging; a drag that starts too fast is read as a tap.
CARD_HOLD_MS = 120.0
#: Card -> tile drag. Long enough for the client to sample intermediate positions.
DEPLOY_DRAG_MS = 320.0
#: After releasing on the tile the direction selector animates in.
DEPLOY_SETTLE_MS = 220.0
#: The direction flick itself.
FACING_DRAG_MS = 160.0
#: Let the deploy animation finish before the next gesture aims at the same area.
POST_DEPLOY_MS = 260.0
#: Radial menu (retreat / skill) fade-in after tapping a deployed operator.
MENU_SETTLE_MS = 180.0
#: Slack after a plain UI tap before the screen is worth reading again.
TAP_ACK_MS = 90.0
#: Pause / resume dialog animation.
PAUSE_SETTLE_MS = 320.0
#: Gap between consecutive taps on the speed toggle so both register.
SPEED_TAP_GAP_MS = 140.0

# --- calibration: geometry (fractions of the screen) ------------------------ #
#: Distance of the direction flick, as a fraction of screen height.
FACING_DRAG_REL = 0.10
#: Radial menu button offsets from the operator, as fractions of screen height.
RETREAT_OFFSET_REL = (-0.105, 0.105)
SKILL_OFFSET_REL = (0.105, 0.105)
#: Bottom-right deploy card bar: centre of card 0 and the pitch towards the left.
CARD0_CENTRE_REL = (0.945, 0.855)
CARD_PITCH_REL = 0.083
#: In-battle top bar: pause button, speed toggle.
PAUSE_BUTTON_REL = (0.955, 0.055)
SPEED_BUTTON_REL = (0.895, 0.055)
#: "Continue" in the pause dialog.
RESUME_BUTTON_REL = (0.635, 0.735)
#: Speeds the toggle cycles through, in order.
SPEED_CYCLE: tuple[int, ...] = (1, 2)

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Action:
    """A primitive the device can execute. ``kind`` selects the device method."""

    kind: str                       # tap | swipe | long_press | wait
    points: tuple[Point, ...] = ()
    ms: float = 0.0

    def apply(self, device: DeviceController) -> None:
        if self.kind == "wait":
            time.sleep(self.ms / 1000.0)
        elif self.kind == "tap":
            device.tap(*self.points[0])
        elif self.kind == "long_press":
            device.long_press(*self.points[0], self.ms)
        elif self.kind == "swipe":
            device.swipe(self.points, self.ms)
        else:
            raise ValueError(f"unknown primitive {self.kind!r}")

    def __str__(self) -> str:
        pts = " ".join(f"({x:.0f},{y:.0f})" for x, y in self.points)
        return f"{self.kind}{' ' + pts if pts else ''}{f' {self.ms:.0f}ms' if self.ms else ''}"


def tap(x: float, y: float) -> Action:
    return Action("tap", ((x, y),))


def wait(ms: float) -> Action:
    return Action("wait", (), ms)


def long_press(x: float, y: float, ms: float) -> Action:
    return Action("long_press", ((x, y),), ms)


def swipe(points: Sequence[Point], ms: float) -> Action:
    return Action("swipe", tuple((float(x), float(y)) for x, y in points), ms)


@dataclass(frozen=True)
class Gesture(Sequence[Action]):
    """A named sequence of primitives. Sequence so it can be asserted on directly."""

    name: str
    actions: tuple[Action, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> Action:  # type: ignore[override]
        return self.actions[index]

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    @property
    def duration_ms(self) -> float:
        """Wall-clock the gesture occupies — the agent's next decision cannot land
        before this has elapsed."""
        return sum(a.ms for a in self.actions)

    def run(self, device: DeviceController) -> None:
        for action in self.actions:
            action.apply(device)

    def describe(self) -> str:
        return f"{self.name}: " + " -> ".join(str(a) for a in self.actions)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


#: Map facing -> screen direction under the default battle camera: +col is right
#: on screen and +row (which counts from the bottom of the map) is *up*, i.e. -y.
_FACING_SCREEN = {
    Direction.RIGHT: (1.0, 0.0),
    Direction.LEFT: (-1.0, 0.0),
    Direction.UP: (0.0, -1.0),
    Direction.DOWN: (0.0, 1.0),
}


@dataclass(frozen=True)
class Gestures:
    """Builds gestures for one screen size. Nothing here touches a device."""

    width: int
    height: int

    @classmethod
    def for_device(cls, device: DeviceController) -> Gestures:
        w, h = device.resolution
        return cls(w, h)

    # -- geometry ----------------------------------------------------------- #

    def rel(self, rx: float, ry: float) -> Point:
        return self.clamp((rx * self.width, ry * self.height))

    def clamp(self, p: Point) -> Point:
        return (
            min(max(p[0], 0.0), float(self.width - 1)),
            min(max(p[1], 0.0), float(self.height - 1)),
        )

    def card_point(self, index: int) -> Point:
        """Centre of deploy card ``index``, counting leftwards from the right edge.

        A fallback only: perception detects the actual card boxes, because the bar
        scrolls and re-packs as operators are deployed.
        """
        if index < 0:
            raise ValueError(f"card index must be >= 0, got {index}")
        rx = CARD0_CENTRE_REL[0] - index * CARD_PITCH_REL
        return self.rel(rx, CARD0_CENTRE_REL[1])

    def retreat_button(self, unit: Point) -> Point:
        dx, dy = RETREAT_OFFSET_REL
        return self.clamp((unit[0] + dx * self.height, unit[1] + dy * self.height))

    def skill_button(self, unit: Point) -> Point:
        dx, dy = SKILL_OFFSET_REL
        return self.clamp((unit[0] + dx * self.height, unit[1] + dy * self.height))

    def facing_end(self, target: Point, facing: Direction | str | Point) -> Point:
        """Where the direction flick ends, given a facing."""
        if isinstance(facing, tuple):
            vx, vy = facing
            norm = (vx * vx + vy * vy) ** 0.5 or 1.0
            vx, vy = vx / norm, vy / norm
        else:
            direction = facing if isinstance(facing, Direction) else Direction.parse(facing)
            vx, vy = _FACING_SCREEN[direction]
        reach = FACING_DRAG_REL * self.height
        return self.clamp((target[0] + vx * reach, target[1] + vy * reach))

    # -- gestures ----------------------------------------------------------- #

    def deploy(
        self,
        card: int | Point,
        target: Point,
        facing: Direction | str | Point,
    ) -> Gesture:
        """Press an operator card, drag it onto a tile, release, then flick to face.

        Two strokes, exactly as a player does it: the client places a ghost when
        the first stroke is released and only commits the operator when the second
        one sets the direction.
        """
        origin = self.card_point(card) if isinstance(card, int) else self.clamp(card)
        goal = self.clamp(target)
        # The doubled first point is the hold: the swipe writer spends one segment
        # of its duration going nowhere, which is what the card press needs.
        path = (origin, origin, _lerp(origin, goal, 0.4), _lerp(origin, goal, 0.75), goal)
        label = getattr(facing, "name", facing)
        return Gesture(
            f"deploy(card={card}, target=({goal[0]:.0f},{goal[1]:.0f}), facing={label})",
            (
                swipe(path, CARD_HOLD_MS + DEPLOY_DRAG_MS),
                wait(DEPLOY_SETTLE_MS),
                swipe((goal, self.facing_end(goal, facing)), FACING_DRAG_MS),
                wait(POST_DEPLOY_MS),
            ),
        )

    def retreat(self, unit: Point) -> Gesture:
        u = self.clamp(unit)
        return Gesture(
            f"retreat({u[0]:.0f},{u[1]:.0f})",
            (tap(*u), wait(MENU_SETTLE_MS), tap(*self.retreat_button(u)), wait(TAP_ACK_MS)),
        )

    def use_skill(self, unit: Point) -> Gesture:
        u = self.clamp(unit)
        return Gesture(
            f"use_skill({u[0]:.0f},{u[1]:.0f})",
            (tap(*u), wait(MENU_SETTLE_MS), tap(*self.skill_button(u)), wait(TAP_ACK_MS)),
        )

    def pause(self) -> Gesture:
        return Gesture("pause", (tap(*self.rel(*PAUSE_BUTTON_REL)), wait(PAUSE_SETTLE_MS)))

    def resume(self) -> Gesture:
        return Gesture("resume", (tap(*self.rel(*RESUME_BUTTON_REL)), wait(PAUSE_SETTLE_MS)))

    def set_speed(self, target: int, current: int = 1) -> Gesture:
        """Toggle battle speed to ``target``.

        The toggle is a cycle with no readable state, so the caller passes what
        perception last saw — the controller must not guess, and must not read it
        from anywhere but the screen.
        """
        if target not in SPEED_CYCLE:
            raise ValueError(f"speed must be one of {SPEED_CYCLE}, got {target}")
        if current not in SPEED_CYCLE:
            raise ValueError(f"current speed must be one of {SPEED_CYCLE}, got {current}")
        steps = (SPEED_CYCLE.index(target) - SPEED_CYCLE.index(current)) % len(SPEED_CYCLE)
        button = self.rel(*SPEED_BUTTON_REL)
        actions: list[Action] = []
        for _ in range(steps):
            actions += [tap(*button), wait(SPEED_TAP_GAP_MS)]
        return Gesture(f"set_speed({current}->{target})", tuple(actions))
