"""Lazy, cached access to a game-data snapshot.

A snapshot is ~90 MB of JSON; nothing is parsed until something asks for it,
and each table is parsed at most once per process.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

from ato.gamedata.fetch import Manifest, fetch_level, fetch_snapshot, latest_snapshot


class GameData:
    """A pinned snapshot of the game's static data."""

    def __init__(self, root: Path, server: str = "cn") -> None:
        self.root = Path(root)
        self.server = server
        self._cache: dict[str, Any] = {}
        manifest_path = self.root / "manifest.json"
        self.manifest: Manifest | None = (
            Manifest.from_json(manifest_path.read_text("utf-8")) if manifest_path.exists() else None
        )

    # -- construction ----------------------------------------------------

    @classmethod
    def open(cls, server: str = "cn", *, download: bool = True) -> GameData:
        """Open the newest local snapshot, fetching one if there is none."""
        snap = latest_snapshot(server=server)
        if snap is None:
            if not download:
                raise FileNotFoundError(
                    "no local game-data snapshot; run `ato gamedata fetch` or pass download=True"
                )
            snap, _ = fetch_snapshot(server)
        return cls(snap, server)

    @property
    def version(self) -> str:
        return self.manifest.version if self.manifest else self.root.name.split("-", 1)[-1]

    # -- raw tables ------------------------------------------------------

    def table(self, name: str) -> Any:
        if name not in self._cache:
            path = self.root / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"{name} missing from snapshot {self.root}")
            self._cache[name] = json.loads(path.read_text("utf-8"))
        return self._cache[name]

    @cached_property
    def characters(self) -> dict[str, Any]:
        return self.table("character_table")

    @cached_property
    def skills(self) -> dict[str, Any]:
        return self.table("skill_table")

    @cached_property
    def ranges(self) -> dict[str, Any]:
        return self.table("range_table")

    @cached_property
    def stages(self) -> dict[str, Any]:
        return self.table("stage_table")["stages"]

    @cached_property
    def enemies(self) -> dict[str, list[dict[str, Any]]]:
        """``enemy_database.json`` is a list of ``{Key, Value}``; index it by key."""
        raw = self.table("enemy_database")["enemies"]
        return {e["Key"]: e["Value"] for e in raw}

    @cached_property
    def enemy_handbook(self) -> dict[str, Any]:
        try:
            return self.table("enemy_handbook_table")
        except FileNotFoundError:
            return {}

    # -- levels ----------------------------------------------------------

    def stage_by_code(self, code: str) -> dict[str, Any] | None:
        """Look a stage up by its human-facing code, e.g. ``1-7`` or ``CE-6``."""
        for st in self.stages.values():
            if st.get("code") == code:
                return st
        return None

    def level_json(self, level_id: str, *, download: bool = True) -> dict[str, Any]:
        """Load one level file, fetching it into the snapshot if absent."""
        key = f"__level__{level_id.lower()}"
        if key in self._cache:
            return self._cache[key]
        from ato.gamedata.sources import level_path

        path = self.root / "levels" / level_path(level_id)
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"level {level_id} not in snapshot {self.root}")
            path = fetch_level(level_id, self.root, server=self.server)
        data = json.loads(path.read_text("utf-8"))
        self._cache[key] = data
        return data

    def level_json_for_stage(self, stage_id: str, **kw: Any) -> dict[str, Any]:
        st = self.stages.get(stage_id)
        if st is None:
            raise KeyError(f"unknown stage id {stage_id!r}")
        level_id = st.get("levelId")
        if not level_id:
            raise ValueError(f"stage {stage_id!r} has no levelId (story-only stage?)")
        return self.level_json(level_id, **kw)
