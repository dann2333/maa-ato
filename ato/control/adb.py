"""ADB-backed device controller for an Android emulator.

Throughput is the binding constraint, so capture and input are both pluggable:

* capture: ``exec-out screencap -p`` (PNG), ``exec-out screencap`` (raw
  framebuffer), or the raw framebuffer gzipped on-device. Raw is fastest to
  decode (a reshape) but biggest on the wire; gzip trades emulator CPU for USB /
  loopback bandwidth. Which one wins depends on the host, so measure with
  :meth:`AdbDevice.benchmark`.
* input: ``input tap``/``input swipe`` always works but pays a process spawn
  (~80-200 ms) per event and cannot express a multi-point drag. MaaTouch /
  minitouch keep one ``adb shell`` open and take a text protocol on stdin, which
  is what makes a real deploy drag (down, many moves, up) possible.

Nothing here reads game memory or traffic: pixels in, touch events out.
"""

from __future__ import annotations

import enum
import gzip
import re
import shutil
import struct
import subprocess
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ato.control.base import DeviceController, DeviceUnavailable, Frame, LatencyBudget

#: Android ``PixelFormat`` codes that ``screencap`` can emit, and their stride.
_BYTES_PER_PIXEL = {1: 4, 2: 4, 3: 3, 4: 2, 5: 4}
_FORMAT_NAMES = {1: "RGBA_8888", 2: "RGBX_8888", 3: "RGB_888", 4: "RGB_565", 5: "BGRA_8888"}

#: ``screencap``'s raw header is w/h/format (12 bytes); Android 9 appended a
#: colorspace word (16 bytes). Which one is in front of us is recovered from the
#: payload length, not from a version probe — emulators lie about their version.
_HEADER_SIZES = (16, 12)

#: Where the touch helper is expected to have been pushed.
MAATOUCH_PATH = "/data/local/tmp/maatouch"
MINITOUCH_PATH = "/data/local/tmp/minitouch"


class AdbError(RuntimeError):
    """An adb invocation failed (non-zero exit, or output we cannot parse)."""


class CaptureStrategy(enum.StrEnum):
    PNG = "png"
    RAW = "raw"
    RAW_GZIP = "raw_gzip"


class TouchBackend(enum.StrEnum):
    INPUT = "input"
    MAATOUCH = "maatouch"
    AUTO = "auto"


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


def decode_raw(payload: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode a raw ``screencap`` framebuffer into HxWx3 BGR.

    Returns the image and the device-reported ``(width, height)``.
    """
    for header in _HEADER_SIZES:
        if len(payload) <= header:
            continue
        w, h, fmt = struct.unpack_from("<III", payload, 0)
        bpp = _BYTES_PER_PIXEL.get(fmt)
        if bpp is None or not (0 < w <= 16384 and 0 < h <= 16384):
            continue
        if w * h * bpp != len(payload) - header:
            continue
        return _pixels_to_bgr(payload[header:], w, h, fmt), (w, h)
    head = struct.unpack_from("<IIII", payload, 0) if len(payload) >= 16 else ()
    raise AdbError(
        f"unrecognised screencap framebuffer: {len(payload)} bytes, header words {head}. "
        "Neither the 12- nor the 16-byte header explains the payload length."
    )


def _pixels_to_bgr(buf: bytes, w: int, h: int, fmt: int) -> np.ndarray:
    if fmt == 4:  # RGB_565, two bytes per pixel
        v = np.frombuffer(buf, dtype="<u2").reshape(h, w).astype(np.uint16)
        r = ((v >> 11) & 0x1F).astype(np.uint16)
        g = ((v >> 5) & 0x3F).astype(np.uint16)
        b = (v & 0x1F).astype(np.uint16)
        # Replicate high bits into the low ones so full-scale stays full-scale.
        out = np.empty((h, w, 3), dtype=np.uint8)
        out[..., 0] = ((b << 3) | (b >> 2)).astype(np.uint8)
        out[..., 1] = ((g << 2) | (g >> 4)).astype(np.uint8)
        out[..., 2] = ((r << 3) | (r >> 2)).astype(np.uint8)
        return out
    bpp = _BYTES_PER_PIXEL[fmt]
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, bpp)
    if fmt == 5:  # BGRA_8888 is already in our order
        return np.ascontiguousarray(arr[..., :3])
    return np.ascontiguousarray(arr[..., 2::-1])  # RGB(A|X) -> BGR


def decode_png(data: bytes) -> np.ndarray:
    """Decode a PNG to HxWx3 BGR, preferring OpenCV and falling back to stdlib.

    The fallback is a minimal reader for exactly what ``screencap -p`` emits
    (8-bit, non-interlaced, gray/RGB/RGBA). It unfilters in Python, which costs
    tens of milliseconds per megapixel — a reason to prefer the raw strategy.
    """
    try:
        import cv2
    except ImportError:
        return _png_to_bgr_stdlib(data)
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AdbError("cv2 failed to decode the screencap PNG")
    return img


def _png_to_bgr_stdlib(data: bytes) -> np.ndarray:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AdbError("not a PNG (bad signature) — is the device translating CRLF?")
    pos, idat, header = 8, bytearray(), None
    while pos + 8 <= len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # length + type + body + crc
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
    if header is None:
        raise AdbError("PNG has no IHDR chunk")
    w, h, depth, colour, compression, filt, interlace = header
    if depth != 8 or interlace != 0 or compression != 0 or filt != 0:
        raise AdbError(
            f"unsupported PNG (depth={depth} interlace={interlace}); "
            "install opencv-python-headless (extra 'device') or use the raw capture strategy"
        )
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(colour)
    if channels is None:
        raise AdbError(f"unsupported PNG colour type {colour} (palette images are not handled)")
    planes = _unfilter(zlib.decompress(bytes(idat)), w, h, channels)
    if channels == 1:
        return np.repeat(planes, 3, axis=2)
    if channels == 2:
        return np.repeat(planes[..., :1], 3, axis=2)
    return np.ascontiguousarray(planes[..., 2::-1])


def _unfilter(raw: bytes, w: int, h: int, channels: int) -> np.ndarray:
    """Reverse the per-scanline PNG filters. Filters 1/3/4 chain on the pixel to
    the left, so those rows cannot be vectorised and run byte by byte."""
    stride = w * channels
    if len(raw) < h * (stride + 1):
        raise AdbError("truncated PNG image data")
    out = np.empty((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(h):
        ftype = raw[pos]
        pos += 1
        cur = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos).astype(np.int32)
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 2:
            cur = (cur + prev) & 0xFF
        elif ftype == 1:
            for i in range(channels, stride):
                cur[i] = (cur[i] + cur[i - channels]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = cur[i - channels] if i >= channels else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = cur[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pred) & 0xFF
        else:
            raise AdbError(f"unknown PNG filter type {ftype} on row {y}")
        out[y] = cur.astype(np.uint8)
        prev = cur
    return out.reshape(h, w, channels)


# --------------------------------------------------------------------------- #
# minitouch / MaaTouch protocol
# --------------------------------------------------------------------------- #


class TouchPipe:
    """A persistent ``adb shell`` speaking the minitouch text protocol.

    Protocol (MaaTouch is protocol-compatible with minitouch)::

        <- v <version>
        <- ^ <max_contacts> <max_x> <max_y> <max_pressure>
        <- $ <pid>
        -> d <contact> <x> <y> <pressure>    touch down
        -> m <contact> <x> <y> <pressure>    move
        -> u <contact>                       lift
        -> w <ms>                            wait, on the device, between commits
        -> c                                 commit the batch written so far
        -> r                                 reset all contacts

    Because ``w`` and ``c`` are executed device-side, a whole drag is one write:
    the host round trip is paid once instead of once per sample, which is the
    entire point of using this over ``input swipe``.
    """

    def __init__(self, args: Sequence[str], *, screen: tuple[int, int], timeout: float = 5.0):
        self.args = list(args)
        self.screen = screen
        self.timeout = timeout
        self.proc: subprocess.Popen[bytes] | None = None
        self.max_contacts = 10
        self.max_x = screen[0] - 1
        self.max_y = screen[1] - 1
        self.max_pressure = 100
        self.version = 0
        self.pid = 0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(  # noqa: S603 - args are built by us, not user input
                self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            raise DeviceUnavailable(f"cannot start touch helper {self.args!r}: {exc}") from exc
        self._read_banner()

    def _read_banner(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise DeviceUnavailable(
                    f"touch helper {self.args[-1]!r} exited before announcing itself"
                )
            text = line.decode("utf-8", "replace").strip()
            if text.startswith("v "):
                self.version = int(text[2:].split()[0])
            elif text.startswith("^"):
                parts = text.split()
                self.max_contacts = int(parts[1])
                self.max_x = int(parts[2])
                self.max_y = int(parts[3])
                self.max_pressure = int(parts[4])
            elif text.startswith("$"):
                self.pid = int(text.split()[1])
                return
        raise DeviceUnavailable("touch helper did not send its banner in time")

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.alive:
                self._write("r\n")
                assert self.proc.stdin is not None
                self.proc.stdin.close()
                self.proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired, DeviceUnavailable):
            self.proc.kill()
        finally:
            self.proc = None

    # -- protocol ----------------------------------------------------------- #

    def _write(self, payload: str, queued_ms: float = 0.0) -> None:
        """Send a batch, then wait out the time the device will spend on it.

        The helper protocol's ``w`` command is executed *on the device*, while
        this write returns the moment the bytes are flushed. Without the wait the
        two backends disagree about time: a caller that asks for a 1080 ms
        gesture gets one that returns in roughly half that, and any settle delay
        it queued is swallowed -- on a deployment that means the facing flick
        fires before the direction selector has appeared, so the operator faces
        the wrong way. The ``input`` fallback blocks naturally, so this is what
        makes the two agree.
        """
        if not self.alive or self.proc is None or self.proc.stdin is None:
            raise DeviceUnavailable("touch helper pipe is closed")
        try:
            self.proc.stdin.write(payload.encode("ascii"))
            self.proc.stdin.flush()
        except OSError as exc:
            raise DeviceUnavailable(f"touch helper pipe broke: {exc}") from exc
        if queued_ms > 0.0:
            time.sleep(queued_ms / 1000.0)

    def _scale(self, x: float, y: float) -> tuple[int, int]:
        """Screen pixels -> touch-device units. They differ on real hardware; on
        emulators they are usually identical and this is the identity."""
        sw, sh = self.screen
        tx = 0 if sw <= 1 else round(x * self.max_x / (sw - 1))
        ty = 0 if sh <= 1 else round(y * self.max_y / (sh - 1))
        return (max(0, min(self.max_x, tx)), max(0, min(self.max_y, ty)))

    def _pressure(self) -> int:
        return max(1, min(self.max_pressure, 50))

    def down(self, contact: int, x: float, y: float) -> None:
        tx, ty = self._scale(x, y)
        self._write(f"d {contact} {tx} {ty} {self._pressure()}\nc\n")

    def move(self, contact: int, x: float, y: float) -> None:
        tx, ty = self._scale(x, y)
        self._write(f"m {contact} {tx} {ty} {self._pressure()}\nc\n")

    def up(self, contact: int) -> None:
        self._write(f"u {contact}\nc\n")

    def reset(self) -> None:
        self._write("r\n")

    def tap(self, x: float, y: float, *, hold_ms: float = 60.0, contact: int = 0) -> None:
        tx, ty = self._scale(x, y)
        p = self._pressure()
        self._write(
            f"d {contact} {tx} {ty} {p}\nc\nw {int(hold_ms)}\nu {contact}\nc\n",
            queued_ms=hold_ms,
        )

    def swipe(
        self,
        points: Sequence[tuple[float, float]],
        duration_ms: float,
        *,
        contact: int = 0,
        steps_per_segment: int = 10,
        settle_ms: float = 40.0,
    ) -> None:
        if len(points) < 2:
            raise ValueError("a swipe needs at least two points")
        segments = len(points) - 1
        per_step = max(1, int(duration_ms / max(1, segments * steps_per_segment)))
        p = self._pressure()
        x0, y0 = self._scale(*points[0])
        out = [f"d {contact} {x0} {y0} {p}", "c"]
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:], strict=True):
            for s in range(1, steps_per_segment + 1):
                t = s / steps_per_segment
                mx, my = self._scale(ax + (bx - ax) * t, ay + (by - ay) * t)
                out += [f"w {per_step}", f"m {contact} {mx} {my} {p}", "c"]
        # The client samples the last position before the lift; without this dwell
        # a fast drag can be read as a flick past the target.
        out += [f"w {int(settle_ms)}", f"u {contact}", "c"]
        queued = per_step * segments * steps_per_segment + settle_ms
        self._write("\n".join(out) + "\n", queued_ms=queued)

    def long_press(self, x: float, y: float, ms: float, *, contact: int = 0) -> None:
        self.tap(x, y, hold_ms=ms, contact=contact)


# --------------------------------------------------------------------------- #
# the controller
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    """What :meth:`AdbDevice.benchmark` measured."""

    strategy: str
    touch: str
    resolution: tuple[int, int]
    frames: int
    capture_p50_ms: float
    capture_p95_ms: float
    decode_p50_ms: float
    tap_p50_ms: float
    fps: float
    bytes_per_frame: int = 0

    def __str__(self) -> str:
        return (
            f"{self.strategy}/{self.touch} {self.resolution[0]}x{self.resolution[1]}: "
            f"{self.fps:.1f} fps (capture p50 {self.capture_p50_ms:.1f} ms, "
            f"p95 {self.capture_p95_ms:.1f} ms, decode p50 {self.decode_p50_ms:.1f} ms, "
            f"tap {self.tap_p50_ms:.1f} ms)"
        )


@dataclass
class AdbDevice(DeviceController):
    """Drive an emulator over adb. Constructing this never touches the device."""

    serial: str | None = None
    #: ``host:port`` to ``adb connect`` before use (emulators: 127.0.0.1:5555, 16384, ...).
    address: str | None = None
    adb_path: str = "adb"
    strategy: CaptureStrategy = CaptureStrategy.RAW
    touch: TouchBackend = TouchBackend.AUTO
    timeout: float = 20.0
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    #: Command used for the gzipped raw capture; ``toybox`` on modern Android.
    gzip_command: str = "screencap | toybox gzip -1"

    _resolution: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _probe: tuple[float, bool] = field(default=(0.0, False), init=False, repr=False)
    _pipe: TouchPipe | None = field(default=None, init=False, repr=False)
    _touch_note: str = field(default="", init=False, repr=False)
    _touch_gave_up: bool = field(default=False, init=False, repr=False)
    _last_bytes: int = field(default=0, init=False, repr=False)

    # -- plumbing ----------------------------------------------------------- #

    def _args(self, *args: str, targeted: bool = True) -> list[str]:
        head = [self.adb_path]
        if targeted and self.serial:
            head += ["-s", self.serial]
        return head + list(args)

    def _run(
        self,
        args: Sequence[str],
        *,
        binary: bool = False,
        timeout: float | None = None,
        check: bool = True,
    ) -> Any:
        try:
            proc = subprocess.run(  # noqa: S603 - argv is constructed here, never a shell string
                list(args),
                capture_output=True,
                timeout=self.timeout if timeout is None else timeout,
            )
        except FileNotFoundError as exc:
            raise DeviceUnavailable(
                f"adb executable {self.adb_path!r} not found on PATH "
                "(install platform-tools, or pass adb_path=...)"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DeviceUnavailable(f"adb timed out: {' '.join(args)}") from exc
        if check and proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace").strip()
            raise AdbError(f"{' '.join(args)} failed ({proc.returncode}): {err}")
        return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")

    def adb_available(self) -> bool:
        return shutil.which(self.adb_path) is not None or "/" in self.adb_path

    def devices(self) -> list[tuple[str, str]]:
        """``[(serial, state)]`` from ``adb devices``; empty when adb is missing."""
        if not self.adb_available():
            return []
        try:
            out = self._run(self._args("devices", targeted=False), check=False)
        except DeviceUnavailable:
            return []
        found: list[tuple[str, str]] = []
        for line in str(out).splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                found.append((parts[0], parts[1]))
        return found

    def connect(self) -> bool:
        """``adb connect`` the configured address. True when a device answers."""
        if self.address and self.adb_available():
            out = str(self._run(self._args("connect", self.address, targeted=False), check=False))
            # adb usually names a network device by its address, but emulators
            # started locally answer as ``emulator-5554``; only pin the serial when
            # the address really is the serial, else let resolution pick it up.
            if "connected to" in out and self.serial is None:
                online = [s for s, state in self.devices() if state == "device"]
                if self.address in online:
                    self.serial = self.address
        # The 500 ms probe cache was almost certainly filled with False by
        # whoever asked "are we connected?" immediately before calling this.
        # Answering from it would report failure for a device that just came up.
        self._probe = (0.0, False)
        return self.connected

    def _resolve_serial(self) -> str:
        online = [s for s, state in self.devices() if state == "device"]
        if self.serial:
            if self.serial in online:
                return self.serial
            raise DeviceUnavailable(
                f"device {self.serial!r} is not online (adb devices: {online or 'none'})"
            )
        if not online:
            raise DeviceUnavailable(
                "no adb device is online"
                + (f"; tried to connect {self.address}" if self.address else "")
            )
        if len(online) > 1:
            raise DeviceUnavailable(f"several devices online {online}; set serial= to pick one")
        self.serial = online[0]
        return self.serial

    @property
    def connected(self) -> bool:
        """Cached for 500 ms: ``adb devices`` costs a process spawn and the agent
        asks constantly."""
        now = time.monotonic()
        when, value = self._probe
        if now - when < 0.5:
            return value
        try:
            self._resolve_serial()
            value = True
        except DeviceUnavailable:
            value = False
        self._probe = (now, value)
        return value

    def _require(self) -> str:
        """Resolve the serial for an action, paying at most one `adb devices`.

        ``connected`` already resolves as a side effect and caches the answer;
        calling ``_resolve_serial`` again afterwards spawned a second process for
        every frame and every touch, which is exactly what the cache exists to
        avoid.
        """
        if self.connected and self.serial:
            return self.serial
        if self.address:
            self.connect()
        return self._resolve_serial()

    # -- static facts ------------------------------------------------------- #

    @property
    def resolution(self) -> tuple[int, int]:
        if self._resolution is None:
            self._require()
            out = str(self._run(self._args("shell", "wm", "size")))
            # "Physical size: 1920x1080" plus, when the display is overridden,
            # "Override size: 1280x720" — the override is what is actually drawn.
            sizes = re.findall(r"(\d+)x(\d+)", out)
            if not sizes:
                raise AdbError(f"cannot parse `wm size` output: {out!r}")
            w, h = sizes[-1]
            self._resolution = (int(w), int(h))
        return self._resolution

    # -- capture ------------------------------------------------------------ #

    def screencap(self) -> Frame:
        self._require()
        t0 = time.monotonic()
        with self.latency.measure("capture"):
            payload = self._capture_bytes()
        self._last_bytes = len(payload)
        with self.latency.measure("decode"):
            image, size = self._decode(payload)
        if self._resolution is None:
            self._resolution = size
        # The device samples the framebuffer at the start of the transfer, so t0
        # is the honest timestamp — using "now" would understate staleness.
        return Frame(image, t0, size)

    def _capture_bytes(self) -> bytes:
        if self.strategy is CaptureStrategy.PNG:
            return bytes(self._run(self._args("exec-out", "screencap", "-p"), binary=True))
        if self.strategy is CaptureStrategy.RAW_GZIP:
            return bytes(self._run(self._args("exec-out", self.gzip_command), binary=True))
        return bytes(self._run(self._args("exec-out", "screencap"), binary=True))

    def _decode(self, payload: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        if not payload:
            raise AdbError("screencap returned nothing (is the device screen off?)")
        if self.strategy is CaptureStrategy.PNG:
            image = decode_png(payload)
            return image, (image.shape[1], image.shape[0])
        if self.strategy is CaptureStrategy.RAW_GZIP:
            payload = gzip.decompress(payload)
        return decode_raw(payload)

    # -- input -------------------------------------------------------------- #

    @property
    def touch_backend(self) -> str:
        """Which injector is actually in use, after any fallback."""
        if self.touch is TouchBackend.INPUT:
            return TouchBackend.INPUT.value
        pipe = self._touch_pipe()
        return TouchBackend.MAATOUCH.value if pipe is not None else TouchBackend.INPUT.value

    @property
    def touch_note(self) -> str:
        """Why the touch backend is what it is (empty when nothing fell back)."""
        return self._touch_note

    def _helper_command(self) -> list[str] | None:
        """Pick the touch helper that is actually present on the device."""
        out = str(
            self._run(
                self._args("shell", f"ls {MAATOUCH_PATH} {MINITOUCH_PATH} 2>/dev/null"),
                check=False,
            )
        )
        if MAATOUCH_PATH in out:
            # MaaTouch is a jar: run it through app_process with CLASSPATH set.
            return self._args(
                "shell",
                f"CLASSPATH={MAATOUCH_PATH} app_process / com.shxyke.MaaTouch.App",
            )
        if MINITOUCH_PATH in out:
            return self._args("shell", f"{MINITOUCH_PATH} -i")  # -i: protocol on stdin
        return None

    def _touch_pipe(self) -> TouchPipe | None:
        if self.touch is TouchBackend.INPUT:
            return None
        if self._pipe is not None and self._pipe.alive:
            return self._pipe
        # Probing costs an `adb shell ls`; once we have fallen back, stay fallen
        # back until someone calls reset_touch() (e.g. after pushing the helper).
        if self._touch_gave_up:
            if self.touch is TouchBackend.MAATOUCH:
                raise DeviceUnavailable(self._touch_note or "MaaTouch requested but unavailable")
            return None
        self._pipe = None
        try:
            self._require()
            cmd = self._helper_command()
            if cmd is None:
                self._touch_note = (
                    f"neither {MAATOUCH_PATH} nor {MINITOUCH_PATH} on the device; "
                    "falling back to `input`"
                )
            else:
                pipe = TouchPipe(cmd, screen=self.resolution)
                pipe.start()
                self._pipe = pipe
                self._touch_note = ""
        except (DeviceUnavailable, AdbError) as exc:
            self._touch_note = f"touch helper unavailable ({exc}); falling back to `input`"
            self._pipe = None
        if self._pipe is None:
            self._touch_gave_up = True
            if self.touch is TouchBackend.MAATOUCH:
                raise DeviceUnavailable(self._touch_note or "MaaTouch requested but unavailable")
        return self._pipe

    def reset_touch(self) -> None:
        """Forget a touch-backend fallback and probe again on the next input."""
        self.close()
        self._touch_gave_up = False
        self._touch_note = ""

    def tap(self, x: float, y: float) -> None:
        self._require()
        with self.latency.measure("act"):
            pipe = self._touch_pipe()
            if pipe is not None:
                pipe.tap(x, y)
            else:
                self._run(self._args("shell", "input", "tap", str(int(x)), str(int(y))))

    def swipe(self, points: Sequence[tuple[float, float]], duration_ms: float) -> None:
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            raise ValueError("a swipe needs at least two points")
        self._require()
        with self.latency.measure("act"):
            pipe = self._touch_pipe()
            if pipe is not None:
                pipe.swipe(pts, duration_ms)
                return
            # `input swipe` presses down and lifts within a single call, so
            # emitting one call per segment would turn a deployment drag into
            # several disjoint strokes: the card gets released partway to the
            # tile and the deploy never lands. Collapse to one stroke instead and
            # accept that intermediate waypoints are lost -- a straight drag is
            # what the client needs, and it is what a player does anyway.
            (ax, ay), (bx, by) = pts[0], pts[-1]
            self._run(
                self._args(
                    "shell", "input", "swipe",
                    str(int(ax)), str(int(ay)), str(int(bx)), str(int(by)),
                    str(max(1, int(duration_ms))),
                )
            )

    def long_press(self, x: float, y: float, ms: float) -> None:
        self._require()
        with self.latency.measure("act"):
            pipe = self._touch_pipe()
            if pipe is not None:
                pipe.long_press(x, y, ms)
            else:
                # `input swipe` onto the same point is the standard long-press trick.
                xi, yi = str(int(x)), str(int(y))
                self._run(self._args("shell", "input", "swipe", xi, yi, xi, yi, str(int(ms))))

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None

    def __enter__(self) -> AdbDevice:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- measurement -------------------------------------------------------- #

    def benchmark(
        self, n: int = 20, *, tap_point: tuple[int, int] | None = (1, 1)
    ) -> BenchmarkResult:
        """Measure capture throughput and tap round trip, filling :attr:`latency`.

        ``tap_point`` defaults to the extreme corner because a benchmark must not
        press anything meaningful; pass ``None`` to skip input measurement.
        """
        self._require()
        self.latency.reset()
        for _ in range(max(1, n)):
            self.screencap()
        taps: list[float] = []
        if tap_point is not None:
            for _ in range(max(1, n // 4)):
                t0 = time.perf_counter()
                self.tap(*tap_point)
                taps.append((time.perf_counter() - t0) * 1000.0)
        return BenchmarkResult(
            strategy=self.strategy.value,
            touch=self.touch_backend,
            resolution=self.resolution,
            frames=self.latency.count("capture"),
            capture_p50_ms=self.latency.p50("capture"),
            capture_p95_ms=self.latency.p95("capture"),
            decode_p50_ms=self.latency.p50("decode"),
            tap_p50_ms=(sorted(taps)[len(taps) // 2] if taps else 0.0),
            fps=self.latency.fps,
            bytes_per_frame=self._last_bytes,
        )
