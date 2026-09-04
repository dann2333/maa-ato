"""Turning a :class:`~ato.sim.engine.BattleState` into arrays a network can read.

The same encoder produces both views. The teacher gets every column; the student
gets the pixel-realizable subset plus perception noise. They share one code path
on purpose — two encoders would drift, and a drift between teacher and student
features is exactly the bug INVARIANT I-3 exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ato.agent.features import (
    OBSERVATION,
    FeatureGroup,
    ObservationSpec,
    droppable_slices,
    noise_vector,
)
from ato.sim.engine import BattleEngine
from ato.sim.entities import EnemyUnit, OperatorUnit
from ato.sim.types import DamageType, Direction, Tile

#: Range masks are rendered into a fixed 9x9 window centred on the unit. That
#: covers every range in the game with room to spare, and it generalises: a range
#: the network has never seen is still just a mask, not an unknown id.
RANGE_R = 4
RANGE_W = 2 * RANGE_R + 1

_PROFESSIONS = (
    "PIONEER", "WARRIOR", "TANK", "SNIPER", "CASTER", "MEDIC", "SUPPORT",
    "SPECIAL", "TOKEN", "TRAP",
)

_SP_TYPES = ("INCREASE_WITH_TIME", "INCREASE_WHEN_ATTACK", "INCREASE_WHEN_TAKEN_DAMAGE", "PASSIVE")

#: Blackboard keys the engine actually models, in a fixed order. A skill whose
#: effect we cannot model leaves this block zero, which is the honest encoding:
#: the policy learns that the skill's effect is unknown rather than being told a
#: wrong number.
_SKILL_EFFECT_KEYS = (
    "atk_scale", "attack@atk_scale", "def_scale", "attack_speed",
    "max_hp", "attack@times", "sp_recovery", "cost",
)


@dataclass
class Observation:
    """One encoded frame. Entity arrays are padded; ``*_mask`` marks real rows."""

    globals_: np.ndarray
    operators: np.ndarray
    operator_mask: np.ndarray
    enemies: np.ndarray
    enemy_mask: np.ndarray
    cards: np.ndarray
    card_mask: np.ndarray
    tiles: np.ndarray
    map_shape: tuple[int, int]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "globals": self.globals_,
            "operators": self.operators,
            "operator_mask": self.operator_mask,
            "enemies": self.enemies,
            "enemy_mask": self.enemy_mask,
            "cards": self.cards,
            "card_mask": self.card_mask,
            "tiles": self.tiles,
        }


def _onehot(value: object, options: tuple[str, ...]) -> list[float]:
    out = [0.0] * len(options)
    try:
        out[options.index(str(value).upper())] = 1.0
    except ValueError:
        pass
    return out


def _damage_onehot(d: DamageType) -> list[float]:
    out = [0.0, 0.0, 0.0, 0.0]
    idx = {DamageType.PHYSICAL: 0, DamageType.ARTS: 1, DamageType.TRUE: 2, DamageType.HEAL: 3}
    out[idx.get(d, 0)] = 1.0
    return out


def _range_mask(grids: tuple[tuple[int, int], ...], direction: Direction) -> list[float]:
    """Render tile offsets into the fixed window, rotated by facing."""
    m = np.zeros((RANGE_W, RANGE_W), dtype=np.float32)
    for dr, dc in grids:
        r, c = dr, dc
        for _ in range(int(direction) % 4):
            r, c = -c, r
        rr, cc = r + RANGE_R, c + RANGE_R
        if 0 <= rr < RANGE_W and 0 <= cc < RANGE_W:
            m[rr, cc] = 1.0
    return m.reshape(-1).tolist()


def _hash_identity(key: str, dim: int = 16) -> list[float]:
    """A stable, collision-tolerant identity code.

    Hashed rather than looked up in a learned table so that an operator the
    network has never seen still gets *a* code instead of an out-of-range index.
    It is droppable regardless (INVARIANT I-2) — this exists so the network can
    tell two units apart within an episode, not so it can memorise who they are.
    """
    h = abs(hash(key))
    return [float((h >> i) & 1) for i in range(dim)]


class Encoder:
    """Encodes battles against one scenario. Reused across episodes."""

    def __init__(
        self,
        engine: BattleEngine,
        *,
        spec: ObservationSpec = OBSERVATION,
        privileged: bool = False,
        max_operators: int = 16,
        max_enemies: int = 48,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spec = spec if privileged else spec.for_policy()
        self.privileged = privileged
        self.max_operators = max_operators
        self.max_enemies = max_enemies
        self.rng = rng or np.random.default_rng(0)
        self.bmap = engine.sc.bmap
        self._cards = list(engine.roster)
        self._noise = {
            g.name: np.asarray(noise_vector(g), dtype=np.float32) for g in self.spec.groups
        }
        self._drop = {g.name: droppable_slices(g) for g in self.spec.groups}
        self._keep = {g.name: self._keep_index(spec.group(g.name), g) for g in self.spec.groups}
        self._tiles = self._encode_tiles(engine)

    @staticmethod
    def _keep_index(full: FeatureGroup, kept: FeatureGroup) -> np.ndarray:
        """Column indices of ``kept`` inside ``full``, so one builder serves both views."""
        offs = full.offsets()
        cols: list[int] = []
        for s in kept.specs:
            a, b = offs[s.name]
            cols.extend(range(a, b))
        return np.asarray(cols, dtype=np.int64)

    # -- static map ------------------------------------------------------

    def _encode_tiles(self, engine: BattleEngine) -> np.ndarray:
        bmap = self.bmap
        rows: list[list[float]] = []
        goal = bmap.ends[0] if bmap.ends else Tile(0, 0)
        traffic = self._route_traffic(engine)
        for t in bmap.all_tiles():
            info = bmap[t]
            special = _hash_identity(info.key, 8) if info.key not in (
                "tile_road", "tile_wall", "tile_forbidden", "tile_floor",
                "tile_start", "tile_end",
            ) else [0.0] * 8
            rows.append([
                1.0 if info.can_deploy("MELEE") else 0.0,
                1.0 if info.can_deploy("RANGED") else 0.0,
                1.0 if info.passable_mask == "ALL" else 0.0,
                1.0 if info.passable_mask in ("ALL", "FLY_ONLY") else 0.0,
                1.0 if info.is_highland else 0.0,
                1.0 if info.key.endswith("start") else 0.0,
                1.0 if info.key == "tile_end" else 0.0,
                0.0,                                    # occupied, filled per frame
                *special,
                float(traffic.get(t, 0)),               # privileged
                float(abs(t.row - goal.row) + abs(t.col - goal.col)),  # privileged
            ])
        full = np.asarray(rows, dtype=np.float32)
        return full[:, self._keep["tile"]]

    def _route_traffic(self, engine: BattleEngine) -> dict[Tile, int]:
        from ato.sim.pathing import route_waypoints

        out: dict[Tile, int] = {}
        for route in engine.sc.routes.values():
            for p in route_waypoints(self.bmap, route):
                out[p.tile] = out.get(p.tile, 0) + 1
        return out

    # -- per frame -------------------------------------------------------

    def encode(self, engine: BattleEngine) -> Observation:
        st = engine.state
        g_full = np.asarray([
            st.cost,
            self._cost_rate(engine),
            float(st.life_points),
            float(st.kills),
            float(st.total_enemies),
            float(len(st.deployed)),
            float(engine.sc.options.character_limit),
            st.time,
            0.0,                                  # 2x speed: the simulator runs at 1x
            float(engine.scheduler.wave_index),   # privileged
            self._time_to_next_spawn(engine),     # privileged
            float(max(st.total_enemies - st.spawned, 0)),  # privileged
        ], dtype=np.float32)[self._keep["global"]]

        ops = np.zeros((self.max_operators, self.spec.group("operator").dim), dtype=np.float32)
        omask = np.zeros(self.max_operators, dtype=np.float32)
        for i, op in enumerate(st.deployed[: self.max_operators]):
            ops[i] = self._encode_operator(op, st.time)
            omask[i] = 1.0

        ens = np.zeros((self.max_enemies, self.spec.group("enemy").dim), dtype=np.float32)
        emask = np.zeros(self.max_enemies, dtype=np.float32)
        for i, e in enumerate(st.active_enemies[: self.max_enemies]):
            ens[i] = self._encode_enemy(e, engine)
            emask[i] = 1.0

        cards = np.zeros((len(self._cards), self.spec.group("card").dim), dtype=np.float32)
        cmask = np.zeros(len(self._cards), dtype=np.float32)
        for i, cid in enumerate(self._cards):
            cards[i] = self._encode_card(engine, cid)
            cmask[i] = 1.0

        tiles = self._tiles.copy()
        occ_col = self._occupied_column()
        if occ_col is not None:
            for op in st.deployed:
                idx = op.tile.row * self.bmap.width + op.tile.col
                if 0 <= idx < tiles.shape[0]:
                    tiles[idx, occ_col] = 1.0

        obs = Observation(
            globals_=g_full, operators=ops, operator_mask=omask,
            enemies=ens, enemy_mask=emask, cards=cards, card_mask=cmask,
            tiles=tiles, map_shape=(self.bmap.height, self.bmap.width),
        )
        if not self.privileged:
            self._corrupt(obs)
        return obs

    def _occupied_column(self) -> int | None:
        offs = self.spec.group("tile").offsets()
        return offs["occupied"][0] if "occupied" in offs else None

    def _cost_rate(self, engine: BattleEngine) -> float:
        o = engine.sc.options
        return 0.0 if o.cost_increase_time <= 0 else engine._cost_rate_scale / o.cost_increase_time

    def _time_to_next_spawn(self, engine: BattleEngine) -> float:
        pend = engine.scheduler.pending
        return min((p.fire_at for p in pend), default=-1.0) - engine.state.time if pend else -1.0

    def _encode_operator(self, op: OperatorUnit, now: float) -> np.ndarray:
        spec = op.spec
        sk = op.skill
        grids = op.attack_range.grids if op.attack_range else ()
        row = [
            float(op.tile.row), float(op.tile.col),
            *[1.0 if int(op.direction) == d else 0.0 for d in range(4)],
            op.hp_ratio,
            0.0 if sk is None or sk.sp_cost <= 0 else min(op.sp / sk.sp_cost, 1.0),
            1.0 if op.skill_active else 0.0,
            float(len(op.blocking)),
            op.stats.atk, op.stats.defense, op.stats.res, op.stats.max_hp,
            op.stats.attack_interval, float(op.stats.block_capacity),
            float(spec.stats.cost if spec else 0),
            *_damage_onehot(op.damage_type),
            *_range_mask(grids, op.direction),
            float(sk.sp_cost if sk else 0.0), float(sk.duration if sk else 0.0),
            *_onehot(sk.sp_type if sk else "", _SP_TYPES),
            1.0 if (sk is not None and not sk.is_auto) else 0.0,
            now - op.deploy_time,
            *_hash_identity(spec.char_id if spec else ""),
            op.hp,                                              # privileged
            op.sp,                                              # privileged
            max(op.skill_active_until - now, 0.0),              # privileged
        ]
        return np.asarray(row, dtype=np.float32)[self._keep["operator"]]

    def _encode_enemy(self, e: EnemyUnit, engine: BattleEngine) -> np.ndarray:
        spec = e.spec
        level = getattr(spec, "raw", {}).get("levelType") if spec else None
        hb = engine.gd.enemy_handbook.get("enemyData", {}) if engine.gd.enemy_handbook else {}
        tier = (hb.get(spec.key, {}) if spec else {}).get("enemyLevel", "NORMAL")
        remaining = self._route_remaining(e, engine)
        row = [
            e.position.x, e.position.y, e.hp_ratio,
            1.0 if e.is_flying else 0.0,
            1.0 if e.blocked else 0.0,
            1.0 if tier == "ELITE" else 0.0,
            1.0 if tier == "BOSS" else 0.0,
            e.stats.atk, e.stats.defense, e.stats.res, e.stats.max_hp,
            e.stats.move_speed, e.stats.attack_interval,
            *_damage_onehot(e.damage_type),
            float(e.life_point_reduce),
            *_hash_identity(spec.key if spec else ""),
            e.hp,                                               # privileged
            e.progress,                                         # privileged
            remaining,                                          # privileged
            *self._future_path(e),                              # privileged
        ]
        del level
        return np.asarray(row, dtype=np.float32)[self._keep["enemy"]]

    def _route_remaining(self, e: EnemyUnit, engine: BattleEngine) -> float:
        """Seconds to the blue box at current speed, ignoring blocking.

        The single most informative teacher signal: it converts "where is this
        enemy" into "how long do I have", which is the quantity a plan is
        actually about.
        """
        speed = e.stats.move_speed * engine.sc.options.move_multiplier * engine.cal.move_tiles_per_second
        if speed <= 0:
            return -1.0
        dist = 0.0
        pos = e.position
        for si in range(e.step_index, len(e.program)):
            step = e.program[si]
            pts = getattr(step, "points", None)
            if pts is None:
                continue
            start = e.point_index if si == e.step_index else 0
            for p in pts[start:]:
                dist += (p - pos).length()
                pos = p
        return dist / speed

    def _future_path(self, e: EnemyUnit, n: int = 16) -> list[float]:
        out: list[float] = []
        for si in range(e.step_index, len(e.program)):
            pts = getattr(e.program[si], "points", None)
            if pts is None:
                continue
            start = e.point_index if si == e.step_index else 0
            for p in pts[start:]:
                out.extend((p.x, p.y))
                if len(out) >= 2 * n:
                    return out[: 2 * n]
        return out + [0.0] * (2 * n - len(out))

    def _encode_card(self, engine: BattleEngine, char_id: str) -> np.ndarray:
        entry = engine.roster[char_id]
        spec = entry.spec
        sk = entry.skill
        s = spec.stats
        grids = entry.attack_range.grids if entry.attack_range else ()
        effects = [float(sk.blackboard.get(k, 0.0)) if sk else 0.0 for k in _SKILL_EFFECT_KEYS]
        row = [
            float(entry.cost),
            1.0 if engine.state.cost >= entry.cost else 0.0,
            0.0 if engine.state.time >= entry.available_at else min(
                (entry.available_at - engine.state.time) / max(s.respawn_time, 1.0), 1.0
            ),
            1.0 if any(o.spec and o.spec.char_id == char_id for o in engine.state.deployed) else 0.0,
            1.0 if spec.position in ("MELEE", "ALL") else 0.0,
            1.0 if spec.position in ("RANGED", "ALL") else 0.0,
            s.atk, s.defense, s.res, s.max_hp, s.attack_interval,
            float(s.block_cnt), float(s.respawn_time),
            *_damage_onehot(entry.damage_type),
            *_onehot(spec.profession, _PROFESSIONS),
            *_range_mask(grids, Direction.RIGHT),
            float(sk.sp_cost if sk else 0.0), float(sk.init_sp if sk else 0.0),
            float(sk.duration if sk else 0.0),
            *_onehot(sk.sp_type if sk else "", _SP_TYPES),
            1.0 if (sk is not None and not sk.is_auto) else 0.0,
            *effects,
            *_hash_identity(char_id),
        ]
        return np.asarray(row, dtype=np.float32)[self._keep["card"]]

    # -- student-side corruption -----------------------------------------

    def _corrupt(self, obs: Observation) -> None:
        """Inject the perception error the real pipeline will have, and drop
        identity blocks, so the student cannot learn to rely on either."""
        for name, arr in (("global", obs.globals_), ("operator", obs.operators),
                          ("enemy", obs.enemies), ("card", obs.cards)):
            noise = self._noise[name]
            if noise.any():
                arr += self.rng.normal(0.0, 1.0, arr.shape).astype(np.float32) * noise
            for a, b in self._drop[name]:
                if self.rng.random() < 0.5:
                    arr[..., a:b] = 0.0
