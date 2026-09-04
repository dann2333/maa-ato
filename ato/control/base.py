"""The device contract: screen frames in, touch events out. Nothing else.

Vision-and-touch invariant
--------------------------
:class:`DeviceController` is deliberately tiny. It exposes exactly one input
channel (``screencap``) and three output channels (``tap``, ``swipe``,
``long_press``), plus two static facts about the target (``resolution``,
``connected``).

Adding a capability that reads game state by any means other than the screen —
memory reads, hooks, packet capture, an instrumented client, an accessibility
service walking the view tree — is a violation of the project's invariant, not a
feature. Such a method would let the agent silently learn from a channel it will
not have at evaluation time, which invalidates every measurement made with it.
The offline simulator (``ato.sim``) is the only other source of state, and it is
built from public static data, never from a running game.
"""

from __future__ import annotations

import abc
import contextlib
import math
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np


class DeviceUnavailable(RuntimeError):
    """No usable device behind the controller (not connected, adb missing, gone away).

    Every controller method raises this rather than a backend-specific error, so
    the agent has exactly one failure mode to handle for "the emulator is gone".
    """


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured screen.

    ``image`` is HxWx3 uint8 in **BGR** order (the convention OpenCV and the rest
    of ``ato.perception`` use). ``timestamp`` is ``time.monotonic()`` sampled as
    close to the actual capture as the backend allows — it is monotonic, not wall
    clock, because it is only ever used for differences (staleness, frame pacing).
    ``source_size`` is the device-side ``(width, height)`` before any downscale,
    so a consumer can map coordinates back to real touch coordinates.
    """

    image: np.ndarray
    timestamp: float
    source_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 BGR, got shape {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {self.image.dtype}")

    @classmethod
    def from_bgr(
        cls,
        image: np.ndarray,
        *,
        timestamp: float | None = None,
        source_size: tuple[int, int] | None = None,
    ) -> Frame:
        h, w = image.shape[:2]
        return cls(
            image=image,
            timestamp=time.monotonic() if timestamp is None else timestamp,
            source_size=(w, h) if source_size is None else source_size,
        )

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the pixels actually held."""
        return (self.width, self.height)

    @property
    def scale(self) -> float:
        """``image width / device width``. 1.0 unless the backend downscaled."""
        return self.width / self.source_size[0] if self.source_size[0] else 1.0

    @property
    def age(self) -> float:
        """Seconds since capture. The agent must not act on a stale frame."""
        return time.monotonic() - self.timestamp

    def crop(self, x: int, y: int, w: int, h: int) -> Frame:
        """A view of a sub-rectangle, keeping the capture timestamp."""
        return Frame(self.image[y : y + h, x : x + w], self.timestamp, self.source_size)

    def to_device(self, x: float, y: float) -> tuple[float, float]:
        """Map a point in *this frame's* pixels back to device touch coordinates."""
        s = self.scale
        return (x / s, y / s) if s else (x, y)


class DeviceController(abc.ABC):
    """A screen to look at and a finger to poke it with.

    Implementations must not add state-reading capabilities beyond ``screencap``
    (see the module docstring): the agent is only allowed to know what a human
    looking at the screen could know.
    """

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """True when a device is actually reachable right now. Never raises."""

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Device screen ``(width, height)`` in touch coordinates."""

    @abc.abstractmethod
    def screencap(self) -> Frame:
        """Grab one frame. The only permitted input channel."""

    @abc.abstractmethod
    def tap(self, x: float, y: float) -> None:
        """Touch down and up at a point."""

    @abc.abstractmethod
    def swipe(self, points: Sequence[tuple[float, float]], duration_ms: float) -> None:
        """Touch down at ``points[0]``, trace the polyline, release at the last point."""

    @abc.abstractmethod
    def long_press(self, x: float, y: float, ms: float) -> None:
        """Touch down, hold for ``ms``, release."""


@runtime_checkable
class DeviceProtocol(Protocol):
    """Structural mirror of :class:`DeviceController`, for code that does not want
    to inherit (test doubles, replay shims). Same five capabilities, no more."""

    @property
    def connected(self) -> bool: ...

    @property
    def resolution(self) -> tuple[int, int]: ...

    def screencap(self) -> Frame: ...

    def tap(self, x: float, y: float) -> None: ...

    def swipe(self, points: Sequence[tuple[float, float]], duration_ms: float) -> None: ...

    def long_press(self, x: float, y: float, ms: float) -> None: ...


Channel = Literal["capture", "decode", "act"]
#: Measured stages of one perceive-decide-act cycle. ``capture`` is the device
#: round trip, ``decode`` turning bytes into pixels, ``act`` issuing input.
CHANNELS: tuple[Channel, ...] = ("capture", "decode", "act")


@dataclass
class LatencyBudget:
    """Rolling latency measurements, in milliseconds.

    The agent uses this to decide whether it can keep up: Arknights runs at 30/60
    fps and a deploy that lands 300 ms late is a different move. Percentiles are
    over a sliding window because emulator latency drifts (thermal throttling,
    host load), so a lifetime average would hide the drift.
    """

    window: int = 120
    #: Decision rate the agent is trying to sustain.
    target_hz: float = 10.0
    samples: dict[str, deque[float]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = {ch: deque(maxlen=self.window) for ch in CHANNELS}

    def record(self, channel: Channel, ms: float) -> None:
        self.samples[channel].append(float(ms))

    @contextlib.contextmanager
    def measure(self, channel: Channel) -> Iterator[None]:
        """Time a block into ``channel``. Records even when the block raises, so a
        timeout shows up as the latency it cost rather than vanishing."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(channel, (time.perf_counter() - t0) * 1000.0)

    def count(self, channel: Channel) -> int:
        return len(self.samples[channel])

    def percentile(self, channel: Channel, q: float) -> float:
        """Nearest-rank percentile; 0.0 when there is nothing measured yet."""
        data = sorted(self.samples[channel])
        if not data:
            return 0.0
        k = max(0, min(len(data) - 1, math.ceil(q * len(data)) - 1))
        return data[k]

    def p50(self, channel: Channel) -> float:
        return self.percentile(channel, 0.50)

    def p95(self, channel: Channel) -> float:
        return self.percentile(channel, 0.95)

    @property
    def frame_ms(self) -> float:
        """Typical cost of getting one usable frame (capture + decode)."""
        return self.p50("capture") + self.p50("decode")

    @property
    def fps(self) -> float:
        """Frames per second the capture path sustains at p50."""
        return 1000.0 / self.frame_ms if self.frame_ms > 0.0 else 0.0

    @property
    def cycle_ms(self) -> float:
        """Worst-case (p95) perceive-and-act cycle."""
        return self.p95("capture") + self.p95("decode") + self.p95("act")

    def keeps_up(self, hz: float | None = None) -> bool:
        """Whether the worst-case cycle fits in one decision period."""
        rate = self.target_hz if hz is None else hz
        return self.cycle_ms > 0.0 and self.cycle_ms <= 1000.0 / rate

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ch in CHANNELS:
            out[f"{ch}_p50_ms"] = self.p50(ch)
            out[f"{ch}_p95_ms"] = self.p95(ch)
            out[f"{ch}_n"] = float(self.count(ch))
        out["fps"] = self.fps
        out["cycle_p95_ms"] = self.cycle_ms
        return out

    def reset(self) -> None:
        for q in self.samples.values():
            q.clear()

    def __str__(self) -> str:
        parts = [f"{ch} p50={self.p50(ch):.1f} p95={self.p95(ch):.1f}" for ch in CHANNELS]
        return f"LatencyBudget({'; '.join(parts)}; {self.fps:.1f} fps)"


class RecordingDevice(DeviceController):
    """A controller that records what it was asked to do and returns blank frames.

    A test double, not a device: it exists so gesture choreography can be dry-run
    and asserted on without an emulator attached.
    """

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)) -> None:
        self._resolution = resolution
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    @property
    def connected(self) -> bool:
        return True

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def screencap(self) -> Frame:
        w, h = self._resolution
        self.calls.append(("screencap", ()))
        return Frame(np.zeros((h, w, 3), dtype=np.uint8), time.monotonic(), self._resolution)

    def tap(self, x: float, y: float) -> None:
        self.calls.append(("tap", (x, y)))

    def swipe(self, points: Sequence[tuple[float, float]], duration_ms: float) -> None:
        self.calls.append(("swipe", (tuple(points), duration_ms)))

    def long_press(self, x: float, y: float, ms: float) -> None:
        self.calls.append(("long_press", (x, y, ms)))
