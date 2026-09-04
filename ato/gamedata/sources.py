"""Where the public game data lives, and which files we depend on.

Only *data* is fetched — no game assets are vendored into this repository.
Snapshots land under ``data/snapshots/<version>/`` which is gitignored.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Community mirror of the CN client's shipped gamedata. Verified reachable and
#: current (data_version.txt reported VersionControl 77.0.0 at time of writing).
CN_BASE = "https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata"
#: Global (YoStar) servers live in a separate repository, same layout.
EN_BASE = "https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData_YoStar/main/en_US/gamedata"


@dataclass(frozen=True)
class TableSpec:
    """One JSON table we depend on."""

    name: str
    path: str
    #: False for tables that are nice-to-have; a missing one is a warning, not an error.
    required: bool = True
    #: Roughly how large, so the fetcher can order and report sensibly.
    approx_mb: float = 1.0
    why: str = ""


#: The simulator's hard dependencies come first; meta-layer tables follow.
TABLES: tuple[TableSpec, ...] = (
    TableSpec("data_version", "excel/data_version.txt", True, 0.001,
              "version stamp — the trigger for continual-learning ingestion"),
    TableSpec("character_table", "excel/character_table.json", True, 15.0,
              "operator stats per phase/level, talents, skill ids, ranges"),
    TableSpec("skill_table", "excel/skill_table.json", True, 11.0,
              "skill sp cost/type, duration, effect blackboards"),
    TableSpec("range_table", "excel/range_table.json", True, 0.06,
              "attack ranges as grid offsets"),
    TableSpec("enemy_database", "levels/enemydata/enemy_database.json", True, 15.0,
              "per-level enemy attributes, immunities, skills"),
    TableSpec("stage_table", "excel/stage_table.json", True, 20.0,
              "stage -> levelId mapping, ap cost, stage type"),
    TableSpec("enemy_handbook_table", "excel/enemy_handbook_table.json", False, 1.8,
              "enemy display names and prose descriptions (LLM-readable)"),
    TableSpec("uniequip_table", "excel/uniequip_table.json", False, 3.4,
              "modules: which operators have them and their unlock conditions"),
    TableSpec("battle_equip_table", "excel/battle_equip_table.json", False, 5.7,
              "module combat effects (stat deltas and talent overrides)"),
    TableSpec("gamedata_const", "excel/gamedata_const.json", False, 0.06,
              "global constants (e.g. favor/trust curves)"),
    TableSpec("char_patch_table", "excel/char_patch_table.json", False, 0.05,
              "Amiya's alternate forms"),
    TableSpec("building_data", "excel/building_data.json", False, 5.2,
              "RIIC base rooms and operator base skills (meta layer)"),
    TableSpec("roguelike_topic_table", "excel/roguelike_topic_table.json", False, 18.0,
              "Integrated Strategies structure, relics, squads (meta layer)"),
    TableSpec("gacha_table", "excel/gacha_table.json", False, 0.45,
              "recruitment tag pools (meta layer)"),
)

TABLES_BY_NAME = {t.name: t for t in TABLES}


def level_path(level_id: str) -> str:
    """``Obt/Main/level_main_01-07`` -> ``levels/obt/main/level_main_01-07.json``.

    stage_table stores mixed-case level ids; the files on disk are lower-cased.
    """
    return f"{level_id.lower()}.json"
