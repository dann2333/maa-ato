"""Driving the game: screen frames in, touch events out.

The live agent's only input is :meth:`~ato.control.base.DeviceController.screencap`
and its only outputs are ``tap`` / ``swipe`` / ``long_press``. That is the whole
device surface, by design — see :mod:`ato.control.base` for why widening it would
invalidate every measurement the project makes.

Layers:
    base        the contract (``Frame``, ``DeviceController``, ``LatencyBudget``)
    adb         a real Android emulator behind adb, with selectable capture and
                touch backends
    gestures    Arknights input choreography, built as inspectable primitives
    projection  screen <-> tile mapping (fitted homography, or an explicit camera)
"""

from __future__ import annotations

from ato.control.adb import (
    AdbDevice,
    AdbError,
    BenchmarkResult,
    CaptureStrategy,
    TouchBackend,
    TouchPipe,
    decode_png,
    decode_raw,
)
from ato.control.base import (
    CHANNELS,
    DeviceController,
    DeviceProtocol,
    DeviceUnavailable,
    Frame,
    LatencyBudget,
    RecordingDevice,
)
from ato.control.gestures import Action, Gesture, Gestures
from ato.control.projection import (
    CameraParams,
    Homography,
    PerspectiveProjector,
    TileHit,
    fit_homography_from_corners,
)

__all__ = [
    "CHANNELS",
    "Action",
    "AdbDevice",
    "AdbError",
    "BenchmarkResult",
    "CameraParams",
    "CaptureStrategy",
    "DeviceController",
    "DeviceProtocol",
    "DeviceUnavailable",
    "Frame",
    "Gesture",
    "Gestures",
    "Homography",
    "LatencyBudget",
    "PerspectiveProjector",
    "RecordingDevice",
    "TileHit",
    "TouchBackend",
    "TouchPipe",
    "decode_png",
    "decode_raw",
    "fit_homography_from_corners",
]
