"""Feature schema, and the place INVARIANT I-3 is enforced mechanically.

Two things go wrong in projects like this, and both are invisible until the
policy meets the real device:

1. The policy quietly learns operator *identity* instead of operator *stats*,
   so it collapses the moment the account's roster or investment level differs
   from training (INVARIANT I-2).
2. A feature that only the simulator can produce leaks into the policy's input,
   so offline metrics look excellent and the live agent fails (INVARIANT I-3).

Both are prevented here rather than by discipline. Every feature declares
whether perception can produce it and which component does; a policy built from
a spec containing an unrealizable feature raises at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One scalar (or fixed-width block) of the observation."""

    name: str
    dim: int
    #: Can a vision-and-touch agent obtain this from the screen at all?
    realizable_from_pixels: bool
    #: Which perception component produces it. "" for privileged features.
    source: str = ""
    #: Perception is imperfect. This is the expected noise the teacher should
    #: inject when training the student, so the student never learns to rely on
    #: precision the real pipeline cannot deliver.
    noise_std: float = 0.0
    note: str = ""
    #: Identity-like features must be droppable: training zeroes them with some
    #: probability so the network cannot come to depend on recognising a face.
    droppable: bool = False


@dataclass(frozen=True)
class FeatureGroup:
    """A set of features describing one kind of entity."""

    name: str
    specs: tuple[FeatureSpec, ...]

    @property
    def dim(self) -> int:
        return sum(s.dim for s in self.specs)

    def offsets(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        i = 0
        for s in self.specs:
            out[s.name] = (i, i + s.dim)
            i += s.dim
        return out

    def filter(self, *, pixels_only: bool) -> FeatureGroup:
        if not pixels_only:
            return self
        return FeatureGroup(self.name, tuple(s for s in self.specs if s.realizable_from_pixels))

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.specs)


class PrivilegedFeatureLeak(RuntimeError):
    """Raised when a policy is built from a spec containing privileged features."""


@dataclass(frozen=True)
class ObservationSpec:
    """The full schema. ``for_policy()`` is the only thing a student may consume."""

    groups: tuple[FeatureGroup, ...]

    def group(self, name: str) -> FeatureGroup:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(name)

    def for_policy(self) -> ObservationSpec:
        """The pixel-realizable subset — what the deployed agent actually sees."""
        return ObservationSpec(tuple(g.filter(pixels_only=True) for g in self.groups))

    def privileged_names(self) -> tuple[str, ...]:
        return tuple(
            f"{g.name}.{s.name}"
            for g in self.groups
            for s in g.specs
            if not s.realizable_from_pixels
        )

    def assert_policy_safe(self) -> None:
        """Fail loudly if any privileged feature survived into a policy input."""
        leaked = self.privileged_names()
        if leaked:
            raise PrivilegedFeatureLeak(
                "privileged features would reach the policy: "
                + ", ".join(leaked)
                + " -- route them to the critic instead (see docs/INVARIANTS.md I-3)"
            )

    def dims(self) -> dict[str, int]:
        return {g.name: g.dim for g in self.groups}


# ---------------------------------------------------------------------------
# The schema itself
# ---------------------------------------------------------------------------

def _f(name: str, dim: int = 1, *, px: bool = True, src: str = "", noise: float = 0.0,
       note: str = "", drop: bool = False) -> FeatureSpec:
    return FeatureSpec(name, dim, px, src, noise, note, drop)


#: Battle-wide scalars. All of these are on screen; DP and the kill counter are
#: the two that also serve as clocks for sim-vs-real alignment.
GLOBAL = FeatureGroup("global", (
    _f("dp", src="ocr:dp_counter", noise=0.0, note="integer, large and high-contrast"),
    _f("dp_rate", src="derived", note="DP per second, from consecutive readings"),
    _f("life_points", src="ocr:life_counter"),
    _f("kills", src="ocr:kill_counter"),
    _f("kills_total", src="ocr:kill_counter", note="the denominator is drawn next to it"),
    _f("deployed_count", src="tracker"),
    _f("deploy_limit", src="ocr:kill_counter_area"),
    _f("elapsed", src="clock", note="wall clock since battle start"),
    _f("speed_2x", src="template:speed_button"),
    # Privileged: the simulator knows the wave programme; a player does not.
    _f("wave_index", px=False, note="teacher/critic only"),
    _f("time_to_next_spawn", px=False, note="teacher/critic only"),
    _f("enemies_remaining_unspawned", px=False, note="teacher/critic only"),
))

#: Per deployed operator. Deliberately attribute-only: no identity embedding is
#: required to act well, and the one we do provide is droppable.
OPERATOR = FeatureGroup("operator", (
    _f("tile_row"), _f("tile_col"),
    _f("facing_onehot", 4, src="tracker", note="known from the deploy gesture we issued"),
    _f("hp_ratio", src="bar:hp", noise=0.03),
    _f("sp_ratio", src="bar:sp", noise=0.03),
    _f("skill_active", src="template:skill_glow"),
    _f("blocking_count", src="tracker"),
    # Stats come from the roster we brought, so they are known without vision.
    _f("atk", src="roster"), _f("def", src="roster"), _f("res", src="roster"),
    _f("max_hp", src="roster"), _f("attack_interval", src="roster"),
    _f("block_capacity", src="roster"), _f("cost", src="roster"),
    _f("damage_type_onehot", 4, src="roster"),
    _f("range_mask", 81, src="roster", note="9x9 tile mask centred on the unit, facing-rotated"),
    _f("skill_sp_cost", src="roster"), _f("skill_duration", src="roster"),
    _f("skill_sp_type_onehot", 4, src="roster"),
    _f("skill_is_manual", src="roster"),
    _f("seconds_deployed", src="tracker"),
    _f("identity", 16, src="roster", drop=True,
       note="hashed char id; zeroed with probability p during training (I-2)"),
    # Privileged: exact internal values the screen only approximates.
    _f("hp_exact", px=False), _f("sp_exact", px=False),
    _f("skill_remaining", px=False),
))

#: Per enemy on the field. The hard part of perception, and the reason the
#: student is trained with the teacher's noise model rather than clean values.
ENEMY = FeatureGroup("enemy", (
    _f("x", src="detector:enemy", noise=0.15),
    _f("y", src="detector:enemy", noise=0.15),
    _f("hp_ratio", src="bar:enemy_hp", noise=0.05),
    _f("is_flying", src="detector:enemy", note="from the sprite class"),
    _f("is_blocked", src="tracker"),
    _f("is_elite", src="detector:enemy"),
    _f("is_boss", src="detector:enemy"),
    # Type stats are lookups once the sprite is classified, so they are
    # realizable -- but only as well as the classifier is.
    _f("atk", src="classifier:enemy_type", noise=0.0),
    _f("def", src="classifier:enemy_type"),
    _f("res", src="classifier:enemy_type"),
    _f("max_hp", src="classifier:enemy_type"),
    _f("move_speed", src="classifier:enemy_type"),
    _f("attack_interval", src="classifier:enemy_type"),
    _f("damage_type_onehot", 4, src="classifier:enemy_type"),
    _f("life_point_reduce", src="classifier:enemy_type"),
    _f("identity", 16, src="classifier:enemy_type", drop=True),
    # Privileged.
    _f("hp_exact", px=False),
    _f("route_progress", px=False, note="how far along its route -- teacher only"),
    _f("seconds_to_goal", px=False, note="the single most useful teacher signal"),
    _f("future_path", 32, px=False, note="next 16 waypoints -- teacher only"),
))

#: Per available card in the bottom bar. This is what makes the policy adapt to
#: whatever roster and investment level the account actually has.
CARD = FeatureGroup("card", (
    _f("cost", src="ocr:card_cost"),
    _f("affordable", src="derived"),
    _f("cooldown_ratio", src="template:card_cooldown", noise=0.05),
    _f("deployed_already", src="tracker"),
    _f("position_melee", src="roster"), _f("position_ranged", src="roster"),
    _f("atk", src="roster"), _f("def", src="roster"), _f("res", src="roster"),
    _f("max_hp", src="roster"), _f("attack_interval", src="roster"),
    _f("block_capacity", src="roster"), _f("respawn_time", src="roster"),
    _f("damage_type_onehot", 4, src="roster"),
    _f("profession_onehot", 10, src="roster"),
    _f("range_mask", 81, src="roster"),
    _f("skill_sp_cost", src="roster"), _f("skill_init_sp", src="roster"),
    _f("skill_duration", src="roster"), _f("skill_sp_type_onehot", 4, src="roster"),
    _f("skill_is_manual", src="roster"),
    _f("skill_effect", 8, src="roster",
       note="modelled blackboard effects; unknown skills leave this zero, which "
            "is honest -- the policy learns the skill is unpredictable"),
    _f("identity", 16, src="roster", drop=True),
))

#: Per tile. The map is static and fully visible, so all of it is realizable.
TILE = FeatureGroup("tile", (
    _f("buildable_melee"), _f("buildable_ranged"),
    _f("passable_ground"), _f("passable_air"),
    _f("is_highland"), _f("is_start"), _f("is_end"),
    _f("occupied", src="tracker"),
    _f("special_kind", 8, src="map", note="hashed tileKey for non-standard tiles"),
    # Privileged: derived from the route graph, not visible in a frame.
    _f("enemy_traffic", px=False, note="how many routes cross this tile"),
    _f("distance_to_goal", px=False),
))

OBSERVATION = ObservationSpec((GLOBAL, OPERATOR, ENEMY, CARD, TILE))


def policy_spec() -> ObservationSpec:
    """The spec a student policy is allowed to consume."""
    spec = OBSERVATION.for_policy()
    spec.assert_policy_safe()
    return spec


def teacher_spec() -> ObservationSpec:
    """The full spec, for the critic and for privileged teachers only."""
    return OBSERVATION


def droppable_slices(group: FeatureGroup) -> tuple[tuple[int, int], ...]:
    """Column ranges training should randomly zero, per INVARIANT I-2."""
    offs = group.offsets()
    return tuple(offs[s.name] for s in group.specs if s.droppable)


def noise_vector(group: FeatureGroup) -> tuple[float, ...]:
    """Per-column perception noise, for training the student under realistic error."""
    out: list[float] = []
    for s in group.specs:
        out.extend([s.noise_std] * s.dim)
    return tuple(out)


def with_extra(group: FeatureGroup, *specs: FeatureSpec) -> FeatureGroup:
    """Extend a group. New content adds features here rather than editing the
    schema in place, so a spec version stays comparable across checkpoints."""
    return replace(group, specs=group.specs + specs)
