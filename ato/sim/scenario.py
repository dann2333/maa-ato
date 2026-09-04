"""Compile a level JSON into everything the engine needs to run it.

A :class:`Scenario` is pure data: map, routes, the wave programme, difficulty
runes and the resolved enemy roster. It is derived only from the *public static
game data*. It is never fed by anything read out of a running game — the live
agent perceives the game through pixels alone (see ``docs/ARCHITECTURE.md``,
"Vision-and-touch invariant").
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ato.gamedata.models import EnemySpec, resolve_enemy
from ato.gamedata.tables import GameData
from ato.sim.grid import BattleMap
from ato.sim.pathing import RouteSpec, parse_route
from ato.sim.registry import (
    RUNES,
    WAVE_ACTIONS,
    MechanismKind,
    NoveltyLog,
    wave_action_type,
)
from ato.sim.types import Direction, Tile

# Wave actions that put an enemy on the field.
SPAWNING_ACTIONS = frozenset({"SPAWN"})
# Wave actions that manipulate operators/tokens the level itself placed.
PREDEFINED_ACTIONS = frozenset({
    "ACTIVATE_PREDEFINED", "TRIGGER_PREDEFINED", "WITHDRAW_PREDEFINED",
})
# Actions with no simulation effect: purely presentational.
COSMETIC_ACTIONS = {
    "STORY": "story beat, no battle effect",
    "PREVIEW_CURSOR": "camera hint shown to the player",
    "DISPLAY_ENEMY_INFO": "opens the enemy info card",
    "PLAY_OPERA": "cutscene",
    "TUTORIAL": "tutorial overlay",
    "BATTLE_EVENTS": "presentation trigger",
    "EMPTY": "explicit no-op used for timing padding",
}

for _a in SPAWNING_ACTIONS | PREDEFINED_ACTIONS:
    WAVE_ACTIONS.register(_a)(lambda *_a2, **_k: None)
WAVE_ACTIONS.ignore_all(COSMETIC_ACTIONS)


@dataclass(frozen=True)
class BattleOptions:
    """``level.options`` — the rules of this particular battle."""

    character_limit: int = 8
    max_life_point: int = 3
    initial_cost: int = 10
    max_cost: int = 99
    cost_increase_time: float = 1.0     # seconds per DP
    move_multiplier: float = 0.5        # global enemy speed scale (see SIM_SPEC)
    steering_enabled: bool = True
    max_play_time: float = -1.0
    function_disable_mask: str = "NONE"

    @classmethod
    def from_level(cls, level: dict[str, Any]) -> BattleOptions:
        o = level.get("options") or {}
        return cls(
            character_limit=int(o.get("characterLimit") or 8),
            max_life_point=int(o.get("maxLifePoint") or 3),
            initial_cost=int(o.get("initialCost") or 10),
            max_cost=int(o.get("maxCost") or 99),
            cost_increase_time=float(o.get("costIncreaseTime") or 1.0),
            move_multiplier=float(o.get("moveMultiplier") or 1.0),
            steering_enabled=bool(o.get("steeringEnabled", True)),
            max_play_time=float(o.get("maxPlayTime") if o.get("maxPlayTime") is not None else -1.0),
            function_disable_mask=str(o.get("functionDisableMask") or "NONE"),
        )


@dataclass(frozen=True)
class WaveAction:
    """One scheduled action inside a wave fragment."""

    kind: str
    key: str
    count: int
    pre_delay: float
    interval: float
    route_index: int
    dont_block_wave: bool = False
    block_fragment: bool = False
    hidden_group: str | None = None
    random_spawn_group: str | None = None
    weight: int = 0

    @property
    def spawns(self) -> bool:
        return self.kind in SPAWNING_ACTIONS

    @property
    def busy_until(self) -> float:
        """When this action has finished emitting, relative to fragment start."""
        return self.pre_delay + max(self.count - 1, 0) * self.interval


@dataclass(frozen=True)
class WaveFragment:
    pre_delay: float
    actions: tuple[WaveAction, ...]

    @property
    def busy_until(self) -> float:
        return self.pre_delay + max((a.busy_until for a in self.actions), default=0.0)


@dataclass(frozen=True)
class Wave:
    pre_delay: float
    post_delay: float
    max_wait_for_next: float
    fragments: tuple[WaveFragment, ...]


class Difficulty(enum.IntFlag):
    """Which variant of a stage is being played.

    A stage and its Adverse (突袭) variant share one level file — ``stage_table``
    gives ``main_XX-YY`` difficulty ``NORMAL`` and ``main_XX-YY#f#`` difficulty
    ``FOUR_STAR``, both pointing at the same ``levelId``. The difference is
    entirely in which runes apply, so the simulator supports Adverse mode for
    free. ``SIX_STAR`` is the later six-star variant.

    The numeric flag values are inferred: some level files serialise
    ``difficultyMask`` as an integer, and 1/2 co-occur with the same rune keys
    that appear as ``NORMAL``/``FOUR_STAR`` elsewhere. Listed in
    ``docs/SIM_SPEC.md`` as a calibration item.
    """

    NONE = 0
    NORMAL = 1
    FOUR_STAR = 2
    EASY = 4
    SIX_STAR = 8
    ALL = NORMAL | FOUR_STAR | EASY | SIX_STAR

    @staticmethod
    def parse(value: Any) -> "Difficulty":
        if isinstance(value, Difficulty):
            return value
        if isinstance(value, int):
            return Difficulty(value & Difficulty.ALL) if value else Difficulty.NONE
        out = Difficulty.NONE
        for part in str(value).replace("|", " ").replace(",", " ").split():
            part = part.strip().upper()
            if part == "ALL":
                out |= Difficulty.ALL
            elif part in Difficulty.__members__:
                out |= Difficulty[part]
        return out


@dataclass(frozen=True)
class Rune:
    """A difficulty modifier (hard mode, Contingency Contract, event rules)."""

    key: str
    difficulty_mask: Difficulty
    blackboard: dict[str, float]
    value_str: dict[str, str] = field(default_factory=dict)
    profession_mask: int = 1023
    buildable_mask: str = "ALL"


@dataclass(frozen=True)
class PredefinedUnit:
    """An operator/trap the level places itself (traps, plot allies, boss adds)."""

    char_key: str
    tile: Tile
    direction: Direction
    level: int = 1
    phase: int = 0
    skill_index: int = 0
    hidden: bool = False
    is_token: bool = True
    alias: str | None = None


@dataclass
class Scenario:
    """Everything needed to run one battle."""

    level_id: str
    bmap: BattleMap
    routes: dict[int, RouteSpec]
    waves: tuple[Wave, ...]
    options: BattleOptions
    runes: tuple[Rune, ...]
    predefined: tuple[PredefinedUnit, ...]
    enemy_specs: dict[tuple[str, int], EnemySpec]
    excluded_chars: frozenset[str] = frozenset()
    random_seed: int = 0
    novelty: NoveltyLog = field(default_factory=lambda: NoveltyLog(strict=False))
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def enemy_count(self) -> int:
        """Total enemies the wave programme will emit (ignoring hidden groups)."""
        n = 0
        for w in self.waves:
            for f in w.fragments:
                for a in f.actions:
                    if a.spawns and not a.hidden_group:
                        n += a.count
        return n

    def enemy(self, key: str, level: int) -> EnemySpec:
        spec = self.enemy_specs.get((key, level))
        if spec is None:
            raise KeyError(f"enemy {key!r} level {level} not in scenario roster")
        return spec


def _blackboard(entries: Any) -> tuple[dict[str, float], dict[str, str]]:
    nums: dict[str, float] = {}
    strs: dict[str, str] = {}
    for b in entries or ():
        if not b or b.get("key") is None:
            continue
        if b.get("valueStr") is not None:
            strs[b["key"]] = b["valueStr"]
        nums[b["key"]] = float(b.get("value") or 0.0)
    return nums, strs


def compile_scenario(
    level: dict[str, Any],
    gd: GameData,
    *,
    level_id: str = "",
    difficulty: Difficulty | str = Difficulty.NORMAL,
    strict: bool = True,
) -> Scenario:
    """Turn a raw level JSON into a :class:`Scenario`.

    ``difficulty`` selects which runes apply. Passing ``Difficulty.FOUR_STAR``
    turns the same level file into its Adverse (突袭) variant.
    Unknown mechanics are reported through the scenario's :class:`NoveltyLog`;
    with ``strict=True`` the first one raises, which is what training wants —
    a scenario the simulator cannot model must never silently produce a reward.
    """
    novelty = NoveltyLog(strict=strict)
    want = Difficulty.parse(difficulty)
    bmap = BattleMap.from_level(level)

    routes: dict[int, RouteSpec] = {}
    for i, raw in enumerate(level.get("routes") or ()):
        r = parse_route(i, raw, novelty)
        if r is not None:
            routes[i] = r
    for i, raw in enumerate(level.get("extraRoutes") or ()):
        r = parse_route(-(i + 1), raw, novelty)
        if r is not None:
            routes[-(i + 1)] = r

    # -- waves ------------------------------------------------------------
    waves: list[Wave] = []
    for w in level.get("waves") or ():
        if not w:
            continue
        frags: list[WaveFragment] = []
        for f in w.get("fragments") or ():
            if not f:
                continue
            acts: list[WaveAction] = []
            for a in f.get("actions") or ():
                if not a:
                    continue
                kind = wave_action_type(a.get("actionType"))
                if not novelty.check(WAVE_ACTIONS, kind, f"level {level_id}"):
                    continue
                acts.append(
                    WaveAction(
                        kind=kind,
                        key=str(a.get("key") or ""),
                        count=int(a.get("count") or 1),
                        pre_delay=float(a.get("preDelay") or 0.0),
                        interval=float(a.get("interval") or 0.0),
                        route_index=int(a.get("routeIndex") if a.get("routeIndex") is not None else 0),
                        dont_block_wave=bool(a.get("dontBlockWave", False)),
                        block_fragment=bool(a.get("blockFragment", False)),
                        hidden_group=a.get("hiddenGroup"),
                        random_spawn_group=a.get("randomSpawnGroupKey"),
                        weight=int(a.get("weight") or 0),
                    )
                )
            frags.append(WaveFragment(float(f.get("preDelay") or 0.0), tuple(acts)))
        waves.append(
            Wave(
                pre_delay=float(w.get("preDelay") or 0.0),
                post_delay=float(w.get("postDelay") or 0.0),
                max_wait_for_next=float(
                    w.get("maxTimeWaitingForNextWave")
                    if w.get("maxTimeWaitingForNextWave") is not None
                    else -1.0
                ),
                fragments=tuple(frags),
            )
        )

    # -- runes ------------------------------------------------------------
    runes: list[Rune] = []
    for r in (level.get("runes") or ()):
        if not r:
            continue
        mask = Difficulty.parse(r.get("difficultyMask"))
        if mask is not Difficulty.NONE and not (mask & want):
            continue
        key = str(r.get("key") or "")
        novelty.check(RUNES, key, f"level {level_id}")
        nums, strs = _blackboard(r.get("blackboard"))
        runes.append(
            Rune(
                key=key,
                difficulty_mask=mask,
                blackboard=nums,
                value_str=strs,
                profession_mask=int(r.get("professionMask") or 1023),
                buildable_mask=str(r.get("buildableMask") or "ALL"),
            )
        )

    # -- enemy roster -----------------------------------------------------
    refs: dict[str, int] = {}
    for ref in level.get("enemyDbRefs") or ():
        if ref and ref.get("id"):
            refs[ref["id"]] = max(refs.get(ref["id"], 0), int(ref.get("level") or 0))
    specs: dict[tuple[str, int], EnemySpec] = {}
    for enemy_id in refs:
        levels = gd.enemies.get(enemy_id)
        if levels is None:
            novelty.report(
                MechanismKind.ENEMY_ABILITY, enemy_id, "enemy id absent from enemy_database"
            )
            continue
        for ref in level.get("enemyDbRefs") or ():
            if ref and ref.get("id") == enemy_id:
                lv = int(ref.get("level") or 0)
                specs[(enemy_id, lv)] = resolve_enemy(levels, lv)

    # -- predefined units --------------------------------------------------
    pre: list[PredefinedUnit] = []
    pd = level.get("predefines") or {}
    for group, is_token in (("characterInsts", False), ("tokenInsts", True)):
        for u in pd.get(group) or ():
            if not u:
                continue
            inst = u.get("inst") or {}
            pos = u.get("position") or {"row": 0, "col": 0}
            pre.append(
                PredefinedUnit(
                    char_key=str(inst.get("characterKey") or ""),
                    tile=Tile(pos["row"], pos["col"]),
                    direction=Direction.parse(str(u.get("direction") or "UP")),
                    level=int(inst.get("level") or 1),
                    phase={"PHASE_0": 0, "PHASE_1": 1, "PHASE_2": 2}.get(
                        str(inst.get("phase") or "PHASE_0"), 0
                    ),
                    skill_index=int(u.get("skillIndex") or 0),
                    hidden=bool(u.get("hidden", False)),
                    is_token=is_token,
                    alias=u.get("alias"),
                )
            )

    return Scenario(
        level_id=level_id or str(level.get("levelId") or ""),
        bmap=bmap,
        routes=routes,
        waves=tuple(waves),
        options=BattleOptions.from_level(level),
        runes=tuple(runes),
        predefined=tuple(pre),
        enemy_specs=specs,
        excluded_chars=frozenset(level.get("excludeCharIdList") or ()),
        random_seed=int(level.get("randomSeed") or 0),
        novelty=novelty,
        raw=level,
    )
