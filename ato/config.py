"""Global paths and tunables. Everything else imports from here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("ATO_DATA_ROOT", REPO_ROOT / "data"))

GAMEDATA_DIR = DATA_ROOT / "gamedata"      # working snapshot (symlink-ish: latest fetched)
SNAPSHOT_DIR = DATA_ROOT / "snapshots"     # immutable, version-stamped snapshots
REPLAY_DIR = DATA_ROOT / "replays"
RUN_DIR = DATA_ROOT / "runs"

#: Battle logic runs on a fixed tick. Arknights quantises combat to frames; the
#: simulator uses the same discretisation so that attack intervals, skill
#: durations and spawn delays land on identical boundaries.
TICKS_PER_SECOND = 30
TICK_SECONDS = 1.0 / TICKS_PER_SECOND


@dataclass(frozen=True)
class SimConfig:
    """Knobs that change simulator *behaviour* (not performance)."""

    ticks_per_second: int = TICKS_PER_SECOND
    #: Hard cap so a stuck battle cannot hang a training worker.
    max_battle_seconds: float = 900.0
    #: Enemies spawn with a small random offset in the real game
    #: (``route.spawnRandomRange``). Disable for reproducible search.
    spawn_jitter: bool = False
    #: Fail fast when the scenario references content the simulator cannot model,
    #: instead of silently producing a wrong result. Training uses strict=True.
    strict: bool = True


def ensure_dirs() -> None:
    for p in (DATA_ROOT, GAMEDATA_DIR, SNAPSHOT_DIR, REPLAY_DIR, RUN_DIR):
        p.mkdir(parents=True, exist_ok=True)
