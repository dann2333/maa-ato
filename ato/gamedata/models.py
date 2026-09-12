"""Typed views over the raw tables.

The raw JSON is awkward in two specific ways, and both are handled here so that
nothing downstream has to know about them:

1. ``enemy_database`` wraps every field as ``{"m_defined": bool, "m_value": x}``
   and higher enemy *levels* only re-declare the fields they override, so a
   level-2 enemy inherits everything else from level 0.
2. ``character_table`` stores stats as two key frames per elite phase (level 1
   and max level); intermediate levels are interpolated, then trust, potential
   and module bonuses are added on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# Fields of an enemy/operator that the simulator actually consumes.
_NUMERIC_ATTRS = (
    "maxHp", "atk", "def", "magicResistance", "cost", "blockCnt", "moveSpeed",
    "attackSpeed", "baseAttackTime", "respawnTime", "hpRecoveryPerSec",
    "spRecoveryPerSec", "maxDeployCount", "massLevel", "baseForceLevel",
    "tauntLevel", "epDamageResistance", "epResistance",
)
_BOOL_ATTRS = (
    "stunImmune", "silenceImmune", "sleepImmune", "frozenImmune", "levitateImmune",
    "disarmedCombatImmune", "fearedImmune", "palsyImmune", "attractImmune",
    "teleportImmune", "groundBoundImmune",
)

#: Almost no enemy declares ``blockCnt``, so the implicit default decides how
#: many enemies a defender actually holds -- one of the highest-leverage unknowns
#: in the simulator. Registered as a calibration item in docs/SIM_SPEC.md rather
#: than buried as a literal.
DEFAULT_ENEMY_BLOCK_CNT = 1

#: Clamp on effective attack speed, in percent. Community sources disagree on
#: the floor (10 vs 20) and none of them reads it out of the binary, so both
#: ends are calibration parameters the simulator may override; the missing
#: ceiling is what let a buffed unit reach a 0.05 s attack interval. SIM_SPEC B-13.
ATTACK_SPEED_BOUNDS = (20.0, 600.0)


@dataclass
class Stats:
    """Resolved combat attributes, shared by operators and enemies."""

    max_hp: float = 0.0
    atk: float = 0.0
    defense: float = 0.0
    res: float = 0.0                 # magicResistance, 0-100
    cost: int = 0
    block_cnt: int = 0
    move_speed: float = 1.0
    attack_speed: float = 100.0      # percent; 100 == baseAttackTime unchanged
    base_attack_time: float = 1.0    # seconds between attacks at 100 aspd
    respawn_time: float = 0.0
    hp_recovery_per_sec: float = 0.0
    sp_recovery_per_sec: float = 1.0
    max_deploy_count: int = 1
    taunt_level: int = 0
    mass_level: int = 0
    immunities: frozenset[str] = frozenset()

    def attack_interval_within(self, low: float, high: float) -> float:
        """Seconds between attacks after attack-speed scaling, clamped to ``[low, high]``.

        Arknights scales the interval by ``100 / attackSpeed`` where attackSpeed
        is a percentage. The engine clamps that percentage at both ends; the
        bounds are calibration constants, so the simulator passes its own rather
        than letting this module decide (see ``EngineCalibration.aspd_min``).
        """
        return self.base_attack_time * 100.0 / min(max(self.attack_speed, low), high)

    @property
    def attack_interval(self) -> float:
        """The interval under the default bounds. Prefer :meth:`attack_interval_within`."""
        return self.attack_interval_within(*ATTACK_SPEED_BOUNDS)


def _unwrap(node: Any) -> Any:
    """``{"m_defined": true, "m_value": 5}`` -> ``5``; undefined -> ``None``."""
    if isinstance(node, dict) and "m_defined" in node:
        return node["m_value"] if node["m_defined"] else None
    return node


def _merge_defined(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Overlay only the fields the override actually declares."""
    out = dict(base)
    for k, v in override.items():
        uv = _unwrap(v)
        if uv is not None:
            out[k] = uv
    return out


@dataclass
class EnemySpec:
    """One enemy type at one level, fully resolved."""

    key: str
    level: int
    name: str
    stats: Stats
    apply_way: str = "MELEE"          # MELEE / RANGED / ALL / NONE
    motion: str = "WALK"              # WALK / FLY
    life_point_reduce: int = 1
    tags: tuple[str, ...] = ()
    range_radius: float = 0.0
    skills: tuple[dict[str, Any], ...] = ()
    talent_blackboard: tuple[dict[str, Any], ...] = ()
    #: Raw resolved dict, kept so unmodelled fields stay inspectable.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_flying(self) -> bool:
        return self.motion.upper() == "FLY"


def _merge_enemy_levels(
    levels: list[dict[str, Any]], level: int
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Collapse the level chain into ``(fields, attributes, effective level)``.

    Levels above the highest defined one clamp, matching the game's behaviour of
    reusing the top-most defined tier.
    """
    if not levels:
        raise ValueError("enemy has no level entries")
    ordered = sorted(levels, key=lambda e: e["level"])
    target = min(level, ordered[-1]["level"])
    merged: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    for entry in ordered:
        if entry["level"] > target:
            break
        data = entry["enemyData"]
        merged = _merge_defined(merged, {k: v for k, v in data.items() if k != "attributes"})
        attrs = _merge_defined(attrs, data.get("attributes") or {})
    return merged, attrs, target


def field_or(node: dict[str, Any], key: str, default: Any) -> Any:
    """``node[key]`` unless it is absent or null — deliberately not ``or``.

    Zero is a real value in this data and ``or`` cannot tell it from a missing
    field. ``moveSpeed: 0`` marks an enemy that never moves, and ``or 1.0``
    sent it walking to the blue box; ``spRecoveryPerSec: 0`` marks a skill that
    never charges on its own, and ``or 1.0`` charged it anyway. Neither error
    raises anything — the battle just is not the one the game would play.
    """
    value = node.get(key)
    return default if value is None else value


def _stats_from(attrs: dict[str, Any]) -> Stats:
    immunities = frozenset(k for k in _BOOL_ATTRS if attrs.get(k))
    return Stats(
        max_hp=float(attrs.get("maxHp") or 0),
        atk=float(attrs.get("atk") or 0),
        defense=float(attrs.get("def") or 0),
        res=float(attrs.get("magicResistance") or 0),
        cost=int(attrs.get("cost") or 0),
        block_cnt=int(field_or(attrs, "blockCnt", DEFAULT_ENEMY_BLOCK_CNT)),
        move_speed=float(field_or(attrs, "moveSpeed", 1.0)),
        attack_speed=float(field_or(attrs, "attackSpeed", 100.0)),
        base_attack_time=float(field_or(attrs, "baseAttackTime", 1.0)),
        respawn_time=float(attrs.get("respawnTime") or 0),
        hp_recovery_per_sec=float(attrs.get("hpRecoveryPerSec") or 0.0),
        sp_recovery_per_sec=float(attrs.get("spRecoveryPerSec") or 0.0),
        max_deploy_count=int(field_or(attrs, "maxDeployCount", 1)),
        taunt_level=int(attrs.get("tauntLevel") or 0),
        mass_level=int(attrs.get("massLevel") or 0),
        immunities=immunities,
    )


def _spec_from(merged: dict[str, Any], attrs: dict[str, Any], level: int,
               fallback_key: str = "") -> EnemySpec:
    return EnemySpec(
        key=merged.get("prefabKey") or fallback_key,
        level=level,
        name=merged.get("name") or "",
        stats=_stats_from(attrs),
        apply_way=str(field_or(merged, "applyWay", "MELEE")),
        motion=str(field_or(merged, "motion", "WALK")),
        life_point_reduce=int(field_or(merged, "lifePointReduce", 1)),
        tags=tuple(merged.get("enemyTags") or ()),
        range_radius=float(merged.get("rangeRadius") or 0.0),
        skills=tuple(merged.get("skills") or ()),
        talent_blackboard=tuple(merged.get("talentBlackboard") or ()),
        raw={**merged, "attributes": attrs},
    )


def resolve_enemy(levels: list[dict[str, Any]], level: int) -> EnemySpec:
    """Collapse ``enemy_database`` level entries down to a single spec."""
    merged, attrs, target = _merge_enemy_levels(levels, level)
    return _spec_from(merged, attrs, target)


def resolve_enemy_ref(
    levels: list[dict[str, Any]] | None,
    level: int,
    overwritten: dict[str, Any] | None = None,
    *,
    use_db: bool = True,
    enemy_id: str = "",
) -> EnemySpec:
    """Resolve one ``level.enemyDbRefs`` entry, honouring the level's overrides.

    A stage may reshape an enemy for its own purposes, and a meaningful minority
    of references do: the ``overwrittenData`` block is a partial ``enemyData`` in
    the same ``{m_defined, m_value}`` form, layered on top of the database entry.
    Ignoring it -- which this code did until it was measured -- silently
    simulates a different enemy than the stage actually spawns, and the error is
    invisible because the result still looks like a plausible battle.

    ``use_db=False`` means the stage defines the enemy outright and the database
    is not consulted at all.
    """
    if use_db and levels:
        merged, attrs, target = _merge_enemy_levels(levels, level)
    elif not use_db and overwritten:
        merged, attrs, target = {}, {}, level
    else:
        raise KeyError(f"enemy {enemy_id!r} not in enemy_database and no inline definition")

    if overwritten:
        merged = _merge_defined(merged, {k: v for k, v in overwritten.items()
                                         if k != "attributes"})
        attrs = _merge_defined(attrs, overwritten.get("attributes") or {})
    return _spec_from(merged, attrs, target, fallback_key=enemy_id)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

PHASE_INDEX = {"PHASE_0": 0, "PHASE_1": 1, "PHASE_2": 2, "PHASE_3": 3}


def _lerp_frames(frames: list[dict[str, Any]], level: int) -> dict[str, Any]:
    """Interpolate between the level-1 and max-level key frames.

    NOTE (calibration): the client interpolates linearly and rounds to int for
    integral attributes. Rounding mode is a known unknown — it is registered as
    a calibration parameter in ``docs/SIM_SPEC.md`` and cross-checked against
    observed in-game values rather than assumed correct.
    """
    if len(frames) == 1:
        return dict(frames[0]["data"])
    lo, hi = frames[0], frames[-1]
    l0, l1 = lo["level"], hi["level"]
    t = 0.0 if l1 == l0 else (min(max(level, l0), l1) - l0) / (l1 - l0)
    out: dict[str, Any] = {}
    for k, v0 in lo["data"].items():
        v1 = hi["data"].get(k, v0)
        if isinstance(v0, bool) or not isinstance(v0, (int, float)):
            out[k] = v1 if t >= 1.0 else v0
        elif isinstance(v0, int) and isinstance(v1, int):
            out[k] = int(round(v0 + (v1 - v0) * t))
        else:
            out[k] = v0 + (v1 - v0) * t
    return out


def _apply_favor(data: dict[str, Any], favor_frames: list[dict[str, Any]], trust: int) -> None:
    """Add the trust bonus, which scales linearly to its value at trust 200."""
    if not favor_frames:
        return
    top = max(favor_frames, key=lambda f: f["level"])
    scale = min(max(trust, 0), top["level"] or 200) / (top["level"] or 200)
    for k, v in (top.get("data") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
            data[k] = data.get(k, 0) + (int(v * scale) if isinstance(v, int) else v * scale)


def _apply_potential(data: dict[str, Any], potential_ranks: list[dict[str, Any]], pot: int) -> None:
    """Apply attribute modifiers from potential ranks 2..pot (rank 1 is the base)."""
    for rank in potential_ranks[: max(pot, 0)]:
        buff = (rank.get("buff") or {}).get("attributes") or {}
        for mod in buff.get("attributeModifiers") or ():
            key = _ATTR_ALIASES.get(mod["attributeType"], mod["attributeType"])
            data[key] = data.get(key, 0) + mod["value"]


#: potentialRanks name attributes with UPPER_SNAKE ids; map them onto key-frame keys.
_ATTR_ALIASES = {
    "MAX_HP": "maxHp", "ATK": "atk", "DEF": "def", "MAGIC_RESISTANCE": "magicResistance",
    "COST": "cost", "BLOCK_CNT": "blockCnt", "MOVE_SPEED": "moveSpeed",
    "ATTACK_SPEED": "attackSpeed", "RESPAWN_TIME": "respawnTime",
    "HP_RECOVERY_PER_SEC": "hpRecoveryPerSec", "SP_RECOVERY_PER_SEC": "spRecoveryPerSec",
}


@dataclass
class OperatorSpec:
    """One operator at a specific elite phase / level / trust / potential."""

    char_id: str
    name: str
    profession: str
    sub_profession: str
    rarity: int
    phase: int
    level: int
    stats: Stats
    range_id: str
    skill_ids: tuple[str, ...] = ()
    talents: tuple[dict[str, Any], ...] = ()
    position: str = "MELEE"          # deployment restriction: MELEE / RANGED / ALL
    tags: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def with_stats(self, **kw: Any) -> OperatorSpec:
        return replace(self, stats=replace(self.stats, **kw))


def resolve_operator(
    char_id: str,
    char: dict[str, Any],
    *,
    phase: int = 2,
    level: int | None = None,
    trust: int = 200,
    potential: int = 0,
) -> OperatorSpec:
    """Resolve an operator's combat stats at a given investment level.

    ``phase`` clamps to what the operator actually has (a 3-star cannot E2), and
    ``level=None`` means "max level for that phase" — the common training case.
    """
    phases = char["phases"]
    p = min(max(phase, 0), len(phases) - 1)
    ph = phases[p]
    lvl = ph["maxLevel"] if level is None else min(max(level, 1), ph["maxLevel"])

    data = _lerp_frames(ph["attributesKeyFrames"], lvl)
    _apply_favor(data, char.get("favorKeyFrames") or [], trust)
    _apply_potential(data, char.get("potentialRanks") or [], potential)

    immunities = frozenset(k for k in _BOOL_ATTRS if data.get(k))
    stats = Stats(
        max_hp=float(data.get("maxHp") or 0),
        atk=float(data.get("atk") or 0),
        defense=float(data.get("def") or 0),
        res=float(data.get("magicResistance") or 0),
        cost=int(data.get("cost") or 0),
        block_cnt=int(data.get("blockCnt") or 0),
        move_speed=float(field_or(data, "moveSpeed", 1.0)),
        attack_speed=float(field_or(data, "attackSpeed", 100.0)),
        base_attack_time=float(field_or(data, "baseAttackTime", 1.0)),
        respawn_time=float(data.get("respawnTime") or 0),
        hp_recovery_per_sec=float(data.get("hpRecoveryPerSec") or 0.0),
        sp_recovery_per_sec=float(field_or(data, "spRecoveryPerSec", 1.0)),
        max_deploy_count=int(field_or(data, "maxDeployCount", 1)),
        taunt_level=int(data.get("tauntLevel") or 0),
        mass_level=int(data.get("massLevel") or 0),
        immunities=immunities,
    )
    return OperatorSpec(
        char_id=char_id,
        name=char.get("name") or char_id,
        profession=char.get("profession") or "",
        sub_profession=char.get("subProfessionId") or "",
        rarity=int(char.get("rarity") or 0) if isinstance(char.get("rarity"), int) else 0,
        phase=p,
        level=lvl,
        stats=stats,
        range_id=ph.get("rangeId") or "0-1",
        skill_ids=tuple(s["skillId"] for s in (char.get("skills") or []) if s.get("skillId")),
        talents=tuple(char.get("talents") or ()),
        position=char.get("position") or "MELEE",
        tags=tuple(char.get("tagList") or ()),
        raw=char,
    )


@dataclass(frozen=True)
class SkillSpec:
    """One skill at one mastery level."""

    skill_id: str
    name: str
    level: int
    sp_type: str          # INCREASE_WITH_TIME / INCREASE_WHEN_ATTACK / INCREASE_WHEN_TAKEN_DAMAGE
    sp_cost: float
    init_sp: float
    increment: float
    duration: float
    duration_type: str    # NONE / AMMO / PASSIVE
    skill_type: str       # AUTO / MANUAL / PASSIVE
    range_id: str | None
    blackboard: dict[str, float] = field(default_factory=dict)
    description: str = ""

    @property
    def is_passive(self) -> bool:
        return self.skill_type.upper() == "PASSIVE"

    @property
    def is_auto(self) -> bool:
        return self.skill_type.upper() == "AUTO"


def resolve_skill(skill_id: str, skill: dict[str, Any], mastery: int = 6) -> SkillSpec:
    """Resolve a skill at a mastery level (0-based index; 6 == M3 for 7-level skills)."""
    levels = skill["levels"]
    idx = min(max(mastery, 0), len(levels) - 1)
    lv = levels[idx]
    sp = lv.get("spData") or {}
    bb = {b["key"]: b["value"] for b in (lv.get("blackboard") or []) if b.get("key") is not None}
    return SkillSpec(
        skill_id=skill_id,
        name=lv.get("name") or skill_id,
        level=idx,
        sp_type=str(field_or(sp, "spType", "INCREASE_WITH_TIME")),
        sp_cost=float(sp.get("spCost") or 0),
        init_sp=float(sp.get("initSp") or 0),
        increment=float(field_or(sp, "increment", 1.0)),
        duration=float(lv.get("duration") or 0.0),
        duration_type=str(field_or(lv, "durationType", "NONE")),
        skill_type=str(field_or(lv, "skillType", "MANUAL")),
        range_id=lv.get("rangeId"),
        blackboard=bb,
        description=lv.get("description") or "",
    )


@dataclass(frozen=True)
class AttackRange:
    """An attack range: tile offsets relative to the unit, facing +col."""

    range_id: str
    grids: tuple[tuple[int, int], ...]   # (row, col) offsets

    def rotated(self, direction: int) -> tuple[tuple[int, int], ...]:
        """Rotate the range for a facing. 0=right(+col), 1=down, 2=left, 3=up."""
        out = []
        for r, c in self.grids:
            for _ in range(direction % 4):
                r, c = -c, r
            out.append((r, c))
        return tuple(out)


def resolve_range(range_id: str, entry: dict[str, Any]) -> AttackRange:
    grids = tuple((g["row"], g["col"]) for g in entry.get("grids") or ())
    return AttackRange(range_id=range_id, grids=grids)
