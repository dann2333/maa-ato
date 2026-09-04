"""A battle recorded in the only terms sim and reality share: what is on screen.

This constraint is the whole design. A trace may not contain an enemy's exact
HP, its route progress, or the wave programme, because a camera cannot produce
those and a trace containing them could never be compared against a real
battle. What it may contain is what a player reads off the UI: the DP number,
the kill counter, the life counter, the clock, and the actions we ourselves
issued (which we know because we issued them).

That makes a trace directly comparable whether it came from the simulator, from
a device under ADB, or from a video someone uploaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ato.sim.engine import BattleEngine
from ato.sim.types import BattleResult

Source = Literal["sim", "device", "video"]


@dataclass(frozen=True, slots=True)
class DeployRecord:
    """An action we issued. Known exactly, from either side."""

    t: float
    char_id: str
    row: int
    col: int
    direction: int


@dataclass
class BattleTrace:
    """One battle, in screen-observable quantities."""

    stage_id: str
    source: Source = "sim"
    #: Timestamps at which the kill counter incremented.
    kills: list[float] = field(default_factory=list)
    #: ``(timestamp, life points lost)`` each time the life counter dropped.
    leaks: list[tuple[float, int]] = field(default_factory=list)
    #: ``(timestamp, dp)`` sampled on a fixed grid. DP is the best clock
    #: available: a large high-contrast integer that moves deterministically.
    dp: list[tuple[float, float]] = field(default_factory=list)
    deploys: list[DeployRecord] = field(default_factory=list)
    retreats: list[tuple[float, str]] = field(default_factory=list)
    skills: list[tuple[float, str]] = field(default_factory=list)
    result: str = "RUNNING"
    duration: float = 0.0
    #: Free-form provenance: calibration used, game version, device, video id.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cleared(self) -> bool:
        return self.result == BattleResult.CLEARED.name

    @property
    def life_lost(self) -> int:
        return sum(n for _, n in self.leaks)

    @property
    def first_leak(self) -> float | None:
        return self.leaks[0][0] if self.leaks else None

    def dp_at(self, t: float) -> float | None:
        """DP at a time, by nearest earlier sample."""
        best: float | None = None
        for ts, v in self.dp:
            if ts <= t:
                best = v
            else:
                break
        return best

    # -- persistence -----------------------------------------------------

    def to_json(self) -> str:
        d = asdict(self)
        d["deploys"] = [asdict(x) for x in self.deploys]
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> BattleTrace:
        raw = json.loads(text)
        deploys = [DeployRecord(**x) for x in raw.pop("deploys", [])]
        raw["leaks"] = [tuple(x) for x in raw.get("leaks", [])]
        raw["dp"] = [tuple(x) for x in raw.get("dp", [])]
        raw["retreats"] = [tuple(x) for x in raw.get("retreats", [])]
        raw["skills"] = [tuple(x) for x in raw.get("skills", [])]
        return BattleTrace(deploys=deploys, **raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), "utf-8")

    @staticmethod
    def load(path: Path) -> BattleTrace:
        return BattleTrace.from_json(Path(path).read_text("utf-8"))


class TraceRecorder:
    """Watches an engine and produces a :class:`BattleTrace`.

    Deliberately polls the same counters a screen reader would rather than
    hooking engine internals, so that a bug in the recorder shows up as a
    disagreement with reality instead of hiding behind privileged access.
    """

    def __init__(self, engine: BattleEngine, *, dp_interval: float = 1.0) -> None:
        self.engine = engine
        self.dp_interval = dp_interval
        self.trace = BattleTrace(
            stage_id=engine.sc.level_id,
            source="sim",
            meta={
                "calibration": {
                    k: getattr(engine.cal, k)
                    for k in (
                        "move_tiles_per_second", "deploy_lock_seconds",
                        "redeploy_multiplier", "fragment_relative_delays",
                        "sp_per_second",
                    )
                },
                "gamedata_version": engine.gd.version,
                "seed": getattr(engine.rng, "_seed", None),
            },
        )
        self._kills = engine.state.kills
        self._life = engine.state.life_points
        self._next_dp = 0.0

    def observe(self) -> None:
        """Call once per tick, after :meth:`BattleEngine.tick`."""
        st = self.engine.state
        t = st.time
        if st.kills > self._kills:
            # The counter can jump by more than one within a tick; record each
            # increment at the same timestamp rather than losing the count.
            self.trace.kills.extend([t] * (st.kills - self._kills))
            self._kills = st.kills
        if st.life_points < self._life:
            self.trace.leaks.append((t, self._life - st.life_points))
            self._life = st.life_points
        if t >= self._next_dp:
            self.trace.dp.append((round(t, 3), float(int(st.cost))))
            self._next_dp = t + self.dp_interval

    def note_deploy(self, char_id: str, row: int, col: int, direction: int) -> None:
        self.trace.deploys.append(
            DeployRecord(self.engine.state.time, char_id, row, col, direction)
        )

    def note_retreat(self, char_id: str) -> None:
        self.trace.retreats.append((self.engine.state.time, char_id))

    def note_skill(self, char_id: str) -> None:
        self.trace.skills.append((self.engine.state.time, char_id))

    def finish(self) -> BattleTrace:
        self.trace.result = self.engine.state.result.name
        self.trace.duration = self.engine.state.time
        return self.trace


def run_and_trace(
    engine: BattleEngine,
    plan_apply=None,
    *,
    dp_interval: float = 1.0,
    max_seconds: float | None = None,
) -> BattleTrace:
    """Run a battle to completion, recording a trace.

    ``plan_apply(engine, recorder)`` is called after every tick so a plan or a
    policy can act; it is where deployments get recorded.
    """
    rec = TraceRecorder(engine, dp_interval=dp_interval)
    limit = max_seconds if max_seconds is not None else engine.max_seconds
    while engine.state.result is BattleResult.RUNNING and engine.state.time < limit:
        engine.tick()
        rec.observe()
        if plan_apply is not None:
            plan_apply(engine, rec)
    return rec.finish()
