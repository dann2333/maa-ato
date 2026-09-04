"""Hand-built fixtures for the offline regression suite.

The suite must run without a game-data snapshot, so every structure here is
synthetic. The shapes mirror the real tables closely enough to exercise the
parsers that consume them: ``m_defined`` wrappers in the enemy database, two
attribute key frames per elite phase, ``mapData.map`` stored top-row-first, and
enums serialised both as strings and as the integers older level files use.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

# The package is used from a checkout rather than installed, and pytest only
# puts the test directory itself on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ato.gamedata.tables import GameData  # noqa: E402
from ato.sim.grid import BattleMap  # noqa: E402
from ato.sim.registry import Registry  # noqa: E402
from ato.sim.types import MotionMode, Tile  # noqa: E402

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def _tile(
    key: str,
    height: str,
    buildable: str,
    mask: str,
    blackboard: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "tileKey": key,
        "heightType": height,
        "buildableType": buildable,
        "passableMask": mask,
        "playerSideMask": "ALL",
        "blackboard": blackboard,
        "effects": None,
    }


#: ASCII legend for :func:`make_map_data`.
TILE_LEGEND: dict[str, dict[str, Any]] = {
    ".": _tile("tile_road", "LOWLAND", "MELEE", "ALL"),
    "_": _tile("tile_floor", "LOWLAND", "ALL", "ALL"),
    "#": _tile("tile_wall", "HIGHLAND", "RANGED", "FLY_ONLY"),
    "X": _tile("tile_forbidden", "HIGHLAND", "NONE", "NONE"),
    "S": _tile("tile_start", "LOWLAND", "NONE", "ALL"),
    "s": _tile("tile_flystart", "LOWLAND", "NONE", "ALL"),
    "E": _tile("tile_end", "LOWLAND", "NONE", "ALL"),
    "B": _tile("tile_telin", "LOWLAND", "NONE", "ALL", [{"key": "to_key", "value": 3.0}]),
}

#: 5x7, deliberately asymmetric top-to-bottom so a row-order mistake shows up.
#: A highland row on top, two start tiles on the left, the single end tile in
#: the bottom-right, and a wall column with exactly one door at ``(2, 3)``.
ARENA_ROWS: tuple[str, ...] = (
    "#######",   # display 0 -> Tile row 4
    "s..#...",   # display 1 -> Tile row 3
    "..._...",   # display 2 -> Tile row 2   ('_' is the door through the wall)
    "...#..X",   # display 3 -> Tile row 1
    "S..#..E",   # display 4 -> Tile row 0
)

#: Two impassable orthogonals meeting at a corner, for the corner-cutting rule.
CORNER_ROWS: tuple[str, ...] = (
    "...",       # Tile row 2
    "#..",       # Tile row 1 — wall: blocks walkers, fliers pass
    ".X.",       # Tile row 0 — forbidden: blocks everything
)


def make_map_data(rows: Sequence[str]) -> dict[str, Any]:
    """Build ``level.mapData`` from display-ordered rows (top row first)."""
    tiles: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    grid: list[list[int]] = []
    for line in rows:
        cells: list[int] = []
        for ch in line:
            if ch not in index:
                index[ch] = len(tiles)
                tiles.append(copy.deepcopy(TILE_LEGEND[ch]))
            cells.append(index[ch])
        grid.append(cells)
    return {"map": grid, "tiles": tiles, "width": len(rows[0]), "height": len(rows)}


DEFAULT_OPTIONS: dict[str, Any] = {
    "characterLimit": 6,
    "maxLifePoint": 3,
    "initialCost": 15,
    "maxCost": 99,
    "costIncreaseTime": 1.0,
    "moveMultiplier": 0.5,
    "steeringEnabled": True,
    "maxPlayTime": -1.0,
    "functionDisableMask": "NONE",
    "configBlackBoard": None,
}


def make_level(rows: Sequence[str] = ARENA_ROWS, **overrides: Any) -> dict[str, Any]:
    """A minimal but structurally faithful level JSON."""
    level: dict[str, Any] = {
        "levelId": "Obt/Main/level_test_00-01",
        "mapId": "map_test",
        "mapData": make_map_data(rows),
        "tilesDisallowToLocate": [],
        "runes": None,
        "optionalRunes": None,
        "routes": [],
        "extraRoutes": [],
        "enemies": [],
        "enemyDbRefs": [],
        "waves": [],
        "branches": None,
        "predefines": {"characterInsts": [], "tokenInsts": []},
        "excludeCharIdList": [],
        "randomSeed": 0,
        "options": dict(DEFAULT_OPTIONS),
    }
    level.update(copy.deepcopy(overrides))
    return level


@pytest.fixture
def arena_level() -> dict[str, Any]:
    return make_level(ARENA_ROWS)


@pytest.fixture
def arena(arena_level: dict[str, Any]) -> BattleMap:
    return BattleMap.from_level(arena_level)


@pytest.fixture
def corner() -> BattleMap:
    return BattleMap.from_level(make_level(CORNER_ROWS))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

#: A padding slot exactly as the client writes it — must parse to ``None``.
PLACEHOLDER_ROUTE: dict[str, Any] = {
    "motionMode": "E_NUM",
    "startPosition": {"row": 0, "col": 0},
    "endPosition": {"row": 0, "col": 0},
    "spawnRandomRange": {"x": 0.0, "y": 0.0},
    "spawnOffset": {"x": 0.0, "y": 0.0},
    "checkpoints": None,
    "allowDiagonalMove": True,
    "visitEveryTileCenter": False,
}

#: Ground route with int-serialised enums (0 == WALK, 0 == MOVE, 1 == WAIT).
WALK_ROUTE: dict[str, Any] = {
    "motionMode": 0,
    "startPosition": {"row": 0, "col": 0},
    "endPosition": {"row": 0, "col": 6},
    "spawnRandomRange": {"x": 0.0, "y": 0.0},
    "spawnOffset": {"x": 0.0, "y": -0.5},
    "checkpoints": [
        {"type": 0, "time": 0.0, "position": {"row": 3, "col": 1},
         "reachOffset": {"x": 0.0, "y": 0.0}, "reachDistance": 0.0},
        {"type": 1, "time": 2.0, "position": {"row": 0, "col": 0},
         "reachOffset": {"x": 0.0, "y": 0.0}, "reachDistance": 0.0},
        {"type": 0, "time": 0.0, "position": {"row": 2, "col": 3},
         "reachOffset": {"x": 0.0, "y": 0.0}, "reachDistance": 0.0},
    ],
    "allowDiagonalMove": True,
    "visitEveryTileCenter": False,
}

#: Flier: crosses the wall column that no walker can cross.
FLY_ROUTE: dict[str, Any] = {
    "motionMode": 1,
    "startPosition": {"row": 3, "col": 0},
    "endPosition": {"row": 0, "col": 6},
    "spawnRandomRange": {"x": 0.0, "y": 0.0},
    "spawnOffset": {"x": 0.0, "y": 0.0},
    "checkpoints": None,
    "allowDiagonalMove": True,
    "visitEveryTileCenter": False,
}


# ---------------------------------------------------------------------------
# Enemy database
# ---------------------------------------------------------------------------


def _d(value: Any) -> dict[str, Any]:
    return {"m_defined": True, "m_value": value}


def _u(value: Any = None) -> dict[str, Any]:
    """Undefined field: the value is still serialised but must be ignored."""
    return {"m_defined": False, "m_value": value}


def make_enemy_levels() -> list[dict[str, Any]]:
    """Two levels; level 1 redeclares only ``maxHp``."""
    return [
        {
            "level": 0,
            "enemyData": {
                "name": _d("测试猎犬"),
                "description": _d("fixture"),
                "prefabKey": _d("enemy_1000_test"),
                "applyWay": _d("MELEE"),
                "motion": _d("WALK"),
                "enemyTags": _d(["infection"]),
                "lifePointReduce": _d(2),
                "levelType": _u("NORMAL"),
                "rangeRadius": _u(0.0),
                "talentBlackboard": None,
                "skills": None,
                "attributes": {
                    "maxHp": _d(1000),
                    "atk": _d(100),
                    "def": _d(50),
                    "magicResistance": _d(10.0),
                    "cost": _u(0),
                    "blockCnt": _d(1),
                    "moveSpeed": _d(1.5),
                    "attackSpeed": _d(100.0),
                    "baseAttackTime": _d(2.0),
                    "respawnTime": _u(0),
                    "hpRecoveryPerSec": _d(0.0),
                    "spRecoveryPerSec": _d(0.0),
                    "maxDeployCount": _u(0),
                    "massLevel": _d(0),
                    "tauntLevel": _u(0),
                    "stunImmune": _d(True),
                    "silenceImmune": _d(False),
                    "sleepImmune": _d(True),
                    "frozenImmune": _u(False),
                },
            },
        },
        {
            "level": 1,
            "enemyData": {
                "name": _u(),
                "description": _u(),
                "prefabKey": _u(),
                "applyWay": _u(),
                "motion": _u(),
                "enemyTags": _u(),
                "lifePointReduce": _u(0),
                "attributes": {
                    "maxHp": _d(2500),
                    "atk": _u(0),
                    "def": _u(0),
                    "magicResistance": _u(0.0),
                    "moveSpeed": _u(0.0),
                    "baseAttackTime": _u(0.0),
                    "stunImmune": _u(False),
                },
            },
        },
    ]


def make_flier_levels() -> list[dict[str, Any]]:
    return [
        {
            "level": 0,
            "enemyData": {
                "name": _d("测试飞行"),
                "prefabKey": _d("enemy_1100_test_fly"),
                "applyWay": _d("RANGED"),
                "motion": _d("FLY"),
                "lifePointReduce": _d(1),
                "attributes": {
                    "maxHp": _d(600),
                    "atk": _d(80),
                    "def": _d(0),
                    "magicResistance": _d(0.0),
                    "attackSpeed": _d(200.0),
                    "baseAttackTime": _d(1.6),
                    "moveSpeed": _d(1.0),
                },
            },
        }
    ]


def make_enemy_db() -> dict[str, Any]:
    return {
        "enemies": [
            {"Key": "enemy_1000_test", "Value": make_enemy_levels()},
            {"Key": "enemy_1100_test_fly", "Value": make_flier_levels()},
        ]
    }


@pytest.fixture
def enemy_levels() -> list[dict[str, Any]]:
    return make_enemy_levels()


@pytest.fixture
def gamedata(tmp_path: Path) -> GameData:
    """A real :class:`GameData` over a throwaway snapshot — never downloads."""
    (tmp_path / "enemy_database.json").write_text(
        json.dumps(make_enemy_db(), ensure_ascii=False), encoding="utf-8"
    )
    return GameData(tmp_path)


# ---------------------------------------------------------------------------
# Character table
# ---------------------------------------------------------------------------

_IMMUNE_FLAGS = {
    "stunImmune": False,
    "silenceImmune": False,
    "sleepImmune": False,
    "frozenImmune": False,
    "levitateImmune": False,
    "disarmedCombatImmune": False,
    "fearedImmune": False,
    "palsyImmune": False,
    "attractImmune": False,
    "teleportImmune": False,
    "groundBoundImmune": False,
}


def _frame(level: int, **data: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "maxHp": 1000,
        "atk": 200,
        "def": 100,
        "magicResistance": 0.0,
        "cost": 10,
        "blockCnt": 1,
        "moveSpeed": 1.0,
        "attackSpeed": 100.0,
        "baseAttackTime": 1.0,
        "respawnTime": 70,
        "hpRecoveryPerSec": 0.0,
        "spRecoveryPerSec": 1.0,
        "maxDeployCount": 1,
        "maxDeckStackCnt": 0,
        "tauntLevel": 0,
        "massLevel": 0,
        "baseForceLevel": 0,
        **_IMMUNE_FLAGS,
    }
    base.update(data)
    return {"level": level, "data": base}


def make_character() -> dict[str, Any]:
    """Two elite phases, real key-frame/favour/potential shapes.

    Phase 0 deltas are multiples of 49 so that level 25 — the exact midpoint of
    the 1..50 key frames — lands on integers with no rounding ambiguity.
    """
    return {
        "name": "测试狙击",
        "position": "RANGED",
        "tagList": ["输出"],
        "rarity": "TIER_5",
        "profession": "SNIPER",
        "subProfessionId": "fastshot",
        "phases": [
            {
                "characterPrefabKey": "char_999_test",
                "rangeId": "3-1",
                "maxLevel": 50,
                "attributesKeyFrames": [
                    _frame(1),
                    _frame(50, maxHp=1490, atk=298, **{"def": 149}),
                ],
            },
            {
                "characterPrefabKey": "char_999_test",
                "rangeId": "3-6",
                "maxLevel": 60,
                "attributesKeyFrames": [
                    _frame(1, maxHp=1600, atk=320, cost=12, **{"def": 150}),
                    _frame(60, maxHp=2200, atk=440, cost=12, **{"def": 210}),
                ],
            },
        ],
        # Favour frames cap at level 50 in the real table even though callers
        # think in 0..200 trust points.
        "favorKeyFrames": [
            _frame(0, maxHp=0, atk=0, cost=0, blockCnt=0, moveSpeed=0.0, attackSpeed=0.0,
                   baseAttackTime=0.0, respawnTime=0, spRecoveryPerSec=0.0, maxDeployCount=0,
                   **{"def": 0}),
            _frame(50, maxHp=200, atk=70, cost=0, blockCnt=0, moveSpeed=0.0, attackSpeed=0.0,
                   baseAttackTime=0.0, respawnTime=0, spRecoveryPerSec=0.0, maxDeployCount=0,
                   **{"def": 0}),
        ],
        "potentialRanks": [
            {
                "type": "BUFF",
                "description": "生命上限+200",
                "buff": {"attributes": {"attributeModifiers": [
                    {"attributeType": "MAX_HP", "formulaItem": "ADDITION", "value": 200.0},
                ]}},
            },
            {
                "type": "BUFF",
                "description": "部署费用-1",
                "buff": {"attributes": {"attributeModifiers": [
                    {"attributeType": "COST", "formulaItem": "ADDITION", "value": -1.0},
                ]}},
            },
            {"type": "CUSTOM", "description": "第一天赋效果增强", "buff": None},
        ],
        "skills": [
            {"skillId": "skchr_test_1"},
            {"skillId": "skchr_test_2"},
            {"skillId": None},
        ],
        "talents": [],
    }


@pytest.fixture
def character() -> dict[str, Any]:
    return make_character()


# ---------------------------------------------------------------------------
# Registry sandboxing
# ---------------------------------------------------------------------------


@contextmanager
def registered(reg: Registry, *keys: str) -> Iterator[Registry]:
    """Temporarily teach a global registry some keys.

    The registries are process-global and have no public unregister, so the
    handler tables are snapshotted and restored to keep tests order-independent.
    """
    handlers = dict(reg._handlers)
    ignored = dict(reg._ignored)
    reg.register(*keys)(lambda *_a, **_k: None)
    try:
        yield reg
    finally:
        reg._handlers.clear()
        reg._handlers.update(handlers)
        reg._ignored.clear()
        reg._ignored.update(ignored)


# ---------------------------------------------------------------------------
# Assertions shared by the pathing tests
# ---------------------------------------------------------------------------


def path_cost(path: Sequence[Tile]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:], strict=False):
        total += SQRT2 if (a.row != b.row and a.col != b.col) else 1.0
    return total


def assert_valid_path(
    bmap: BattleMap,
    path: Sequence[Tile],
    src: Tile,
    dst: Tile,
    motion: MotionMode,
) -> None:
    """A returned path must start at src, end at dst and be legal for that motion."""
    assert path[0] == src
    assert path[-1] == dst
    for a, b in zip(path, path[1:], strict=False):
        assert bmap.passable(b, motion)
        assert max(abs(a.row - b.row), abs(a.col - b.col)) == 1
        if a.row != b.row and a.col != b.col:
            assert bmap.passable(Tile(b.row, a.col), motion) or bmap.passable(
                Tile(a.row, b.col), motion
            ), f"corner cut between {a} and {b}"
