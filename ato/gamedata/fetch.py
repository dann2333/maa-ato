"""Fetch a version-stamped snapshot of the public game data.

The snapshot is the simulator's ground truth, so it must be *pinned*: a policy
trained against version 77.0.0 must be reproducible after the game updates.
Every fetch writes an immutable directory plus a manifest of content hashes;
:mod:`ato.gamedata.diff` compares two manifests to discover what changed, which
is what wakes the continual-learning pipeline.

No game data is committed to this repository — snapshots live in gitignored
``data/snapshots/`` and are re-fetched on demand.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ato.config import SNAPSHOT_DIR, ensure_dirs
from ato.gamedata.sources import CN_BASE, EN_BASE, TABLES, TableSpec, level_path

_USER_AGENT = "ato-gamedata-fetcher/0.1 (+https://github.com/dann2333/maa-ato)"
_TIMEOUT = 180
_RETRIES = 4


@dataclass
class FileRecord:
    name: str
    path: str
    sha256: str
    size: int


@dataclass
class Manifest:
    """What a snapshot contains and which game version it came from."""

    version: str                     # e.g. "77.0.0"
    stream: str                      # raw first line of data_version.txt
    server: str                      # "cn" | "en"
    files: dict[str, FileRecord] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "stream": self.stream,
                "server": self.server,
                "files": {k: asdict(v) for k, v in self.files.items()},
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> Manifest:
        raw = json.loads(text)
        return Manifest(
            version=raw["version"],
            stream=raw["stream"],
            server=raw["server"],
            files={k: FileRecord(**v) for k, v in raw["files"].items()},
        )


def _get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
            last = exc
            if attempt < _RETRIES - 1:
                # 2s, 4s, 8s — matches the repo's push/fetch retry convention.
                import time

                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _base(server: str) -> str:
    return {"cn": CN_BASE, "en": EN_BASE}[server]


def parse_version(data_version_txt: str) -> tuple[str, str]:
    """Return ``(version, stream)`` from the contents of ``data_version.txt``.

    The file is a few ``Key:Value`` lines; ``VersionControl`` carries the number
    we pin on. Falls back to the first line if the format ever changes, so a
    format change degrades to "unknown version" rather than crashing a fetch.
    """
    stream = data_version_txt.strip().splitlines()[0].strip() if data_version_txt.strip() else ""
    m = re.search(r"VersionControl:\s*([0-9][0-9.]*)", data_version_txt)
    return (m.group(1) if m else "unknown", stream)


def fetch_snapshot(
    server: str = "cn",
    *,
    tables: tuple[TableSpec, ...] = TABLES,
    dest_root: Path | None = None,
    force: bool = False,
    workers: int = 4,
) -> tuple[Path, Manifest]:
    """Download the excel/enemy tables and write ``data/snapshots/<server>-<version>/``.

    Level files are *not* fetched here — there are >3000 of them and the
    simulator pulls them lazily via :func:`fetch_level`.
    """
    ensure_dirs()
    root = dest_root or SNAPSHOT_DIR
    base = _base(server)

    version_txt = _get(f"{base}/excel/data_version.txt").decode("utf-8", "replace")
    version, stream = parse_version(version_txt)
    snap = root / f"{server}-{version}"
    manifest_path = snap / "manifest.json"

    if manifest_path.exists() and not force:
        return snap, Manifest.from_json(manifest_path.read_text("utf-8"))

    tmp = snap.with_suffix(".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    manifest = Manifest(version=version, stream=stream, server=server)

    def one(spec: TableSpec) -> tuple[TableSpec, bytes | None]:
        try:
            return spec, _get(f"{base}/{spec.path}")
        except RuntimeError:
            if spec.required:
                raise
            return spec, None

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for spec, blob in pool.map(one, tables):
            if blob is None:
                continue
            out = tmp / f"{spec.name}{Path(spec.path).suffix}"
            out.write_bytes(blob)
            manifest.files[spec.name] = FileRecord(
                name=spec.name,
                path=spec.path,
                sha256=hashlib.sha256(blob).hexdigest(),
                size=len(blob),
            )

    (tmp / "manifest.json").write_text(manifest.to_json(), "utf-8")
    if snap.exists():
        shutil.rmtree(snap)
    tmp.rename(snap)
    return snap, manifest


def fetch_level(level_id: str, snapshot: Path, *, server: str = "cn") -> Path:
    """Fetch one level file into ``<snapshot>/levels/`` and return its path.

    Cached: a level already present is returned untouched, so a training run
    that replays the same stage thousands of times hits the network once.
    """
    rel = level_path(level_id)
    out = snapshot / "levels" / rel
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = _get(f"{_base(server)}/levels/{rel}")
    out.write_bytes(blob)
    return out


def latest_snapshot(root: Path | None = None, server: str = "cn") -> Path | None:
    """Most recent locally available snapshot for a server, by version order."""
    root = root or SNAPSHOT_DIR
    if not root.exists():
        return None

    def key(p: Path) -> tuple[int, ...]:
        v = p.name.split("-", 1)[-1]
        try:
            return tuple(int(x) for x in v.split("."))
        except ValueError:
            return (-1,)

    cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{server}-")]
    return max(cands, key=key) if cands else None
