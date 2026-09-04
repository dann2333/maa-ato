"""The MAA copilot document format, as dataclasses.

Field names and defaults follow MAA's own parser
(``src/MaaCore/Config/Miscellaneous/CopilotConfig.cpp``) rather than its prose
documentation; where the two disagree the parser wins, because that is what
actually runs against the community corpus.  See :mod:`ato.copilot` for the
full list of sources.

Parsing is deliberately tolerant.  The corpus spans a decade of MAA versions
and two independent editors, so a strict reader would reject a large fraction
of it for no benefit.  Unknown keys and unknown action types are kept verbatim
in ``extras`` / ``raw_type`` and reported through
:class:`~ato.sim.registry.NoveltyLog` — the same channel the simulator uses for
unmodelled game mechanics, so "the corpus contains something we do not
understand" surfaces in exactly one place.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ato.sim.registry import MechanismKind, NoveltyLog, Registry

#: Fields we carry but whose meaning we could not confirm from any source.
UNCONFIRMED: dict[str, str] = {
    "version": (
        "integer, on ~5/6 of documents served by prts.maa.plus, absent from MAA's "
        "parser and docs; presumably the web editor's document revision. Preserved."
    ),
    "video_url": "string, occurs on published documents; not read by MAA. Preserved.",
    "elapsed_time": (
        "the protocol doc calls this field 'time_elapsed', CopilotConfig.cpp reads "
        "'elapsed_time'. Both spellings are accepted; the parser's is written back."
    ),
    "timeout": "declared 'reserved, not yet implemented' by MAA; default differs between revisions.",
    "requirements": "declared 'reserved interface, not yet implemented' by MAA.",
    "role": "oper/action class hint; only newer parser revisions read it.",
    "difficulty": "0 unset / 1 normal / 2 challenge / 3 both, per the protocol doc only.",
}


class ActionType(enum.StrEnum):
    """Canonical action names. Values are MAA's English spellings."""

    DEPLOY = "Deploy"
    SKILL = "Skill"
    RETREAT = "Retreat"
    SKILL_USAGE = "SkillUsage"
    SPEED_UP = "SpeedUp"
    BULLET_TIME = "BulletTime"
    OUTPUT = "Output"
    SKILL_DAEMON = "SkillDaemon"
    MOVE_CAMERA = "MoveCamera"
    DRAW_CARD = "DrawCard"
    CHECK_IF_START_OVER = "CheckIfStartOver"
    RESET_STOPWATCH = "ResetStopwatch"
    #: Not a MAA type: what an unrecognised ``type`` string parses to.
    UNKNOWN = "Unknown"


# Transcribed from CopilotConfig.cpp's ActionTypeMapping. Case variants are
# folded below instead of listed, but the Chinese spellings must be literal.
_ACTION_ALIAS_SEED: dict[ActionType, tuple[str, ...]] = {
    ActionType.DEPLOY: ("Deploy", "部署"),
    ActionType.SKILL: ("Skill", "技能"),
    ActionType.RETREAT: ("Retreat", "撤退"),
    ActionType.SPEED_UP: ("SpeedUp", "二倍速"),
    ActionType.BULLET_TIME: ("BulletTime", "子弹时间"),
    ActionType.SKILL_USAGE: ("SkillUsage", "技能用法"),
    ActionType.OUTPUT: ("Output", "输出", "打印"),
    ActionType.SKILL_DAEMON: ("SkillDaemon", "DoNothing", "摆完挂机", "开摆"),
    ActionType.MOVE_CAMERA: ("MoveCamera", "移动镜头", "移动相机"),
    ActionType.DRAW_CARD: ("DrawCard", "抽卡", "抽牌", "调配", "调配干员"),
    ActionType.CHECK_IF_START_OVER: ("CheckIfStartOver", "检查重开"),
    ActionType.RESET_STOPWATCH: ("ResetStopwatch", "重置全局计时器"),
}

ACTION_ALIASES: dict[str, ActionType] = {
    alias.lower(): kind for kind, aliases in _ACTION_ALIAS_SEED.items() for alias in aliases
}


class SkillUsage(enum.IntEnum):
    """``skill_usage`` — when MAA fires an operator's skill (AsstBattleDef.h)."""

    NOT_USE = 0  # only when an explicit Skill action says so
    POSSIBLY = 1  # whenever it is ready
    TIMES = 2  # ``skill_times`` times, then stop
    IN_TIME = 3  # "decide automatically" — declared a placeholder by MAA


class DeployDirection(enum.IntEnum):
    """``direction``. Values match :class:`ato.sim.types.Direction` for 0-3."""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    NONE = 4  # drones and devices, which have no facing


_DIRECTION_ALIAS_SEED: dict[DeployDirection, tuple[str, ...]] = {
    DeployDirection.RIGHT: ("Right", "右"),
    DeployDirection.DOWN: ("Down", "下"),
    DeployDirection.LEFT: ("Left", "左"),
    DeployDirection.UP: ("Up", "上"),
    DeployDirection.NONE: ("None", "无"),
}

DIRECTION_ALIASES: dict[str, DeployDirection] = {
    alias.lower(): d for d, aliases in _DIRECTION_ALIAS_SEED.items() for alias in aliases
}

# Keys each object is known to carry. Anything else lands in ``extras`` and is
# reported once, so a new MAA field shows up as novelty instead of silent loss.
_DOC_KEYS = frozenset(
    {"stage_name", "minimum_required", "doc", "opers", "groups", "actions", "difficulty"}
)
_DOC_EXTRA_KEYS = frozenset({"version", "video_url", "type"})
_META_KEYS = frozenset({"title", "title_color", "details", "details_color"})
_OPER_KEYS = frozenset({"name", "skill", "skill_usage", "skill_times", "requirements", "role"})
_REQ_KEYS = frozenset(
    {"elite", "level", "skill_level", "module", "module_level", "potentiality", "potential"}
)
_GROUP_KEYS = frozenset({"name", "opers"})
_ACTION_KEYS = frozenset(
    {
        "type",
        "name",
        "location",
        "direction",
        "kills",
        "costs",
        "cost_changes",
        "cooling",
        "skill_usage",
        "skill_times",
        "pre_delay",
        "post_delay",
        "rear_delay",
        "timeout",
        "elapsed_time",
        "time_elapsed",
        "distance",
        "tool_men",
        "role",
        "skip_if_not_ready",
        "doc",
        "doc_color",
    }
)


def _log(novelty: NoveltyLog | None) -> NoveltyLog:
    # Copilot parsing defaults to non-strict: the corpus is other people's data
    # and we would rather record what we do not understand than refuse to read.
    return novelty if novelty is not None else NoveltyLog(strict=False)


def _unknown_fields(
    raw: Mapping[str, Any], known: frozenset[str], scope: str, where: str, novelty: NoveltyLog
) -> dict[str, Any]:
    extras = {k: v for k, v in raw.items() if k not in known}
    for k in extras:
        novelty.check(_TRAITS, f"copilot.{scope}.field:{k}", where)
    return extras


#: Copilot keys are not simulator mechanics, so they get their own registry
#: rather than polluting ``ato.sim.registry.ALL_REGISTRIES``. It starts empty:
#: every key we meet here is by definition one the format documentation did not
#: cover, which is exactly what we want reported.
_TRAITS = Registry(MechanismKind.TRAIT)


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


@dataclass(frozen=True)
class OperatorRequirements:
    """``opers[].requirements`` — investment the plan assumes. Reserved by MAA."""

    elite: int = 0
    level: int = 0
    skill_level: int = 0
    module: int = -1
    module_level: int = 0
    potentiality: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self == OperatorRequirements()

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any], *, where: str = "", novelty: NoveltyLog | None = None
    ) -> OperatorRequirements:
        log = _log(novelty)
        return cls(
            elite=_as_int(raw.get("elite"), 0),
            level=_as_int(raw.get("level"), 0),
            skill_level=_as_int(raw.get("skill_level"), 0),
            module=_as_int(raw.get("module"), -1),
            module_level=_as_int(raw.get("module_level"), 0),
            # the doc spells it "potential", the editor and corpus "potentiality"
            potentiality=_as_int(raw.get("potentiality", raw.get("potential")), 0),
            extras=_unknown_fields(raw, _REQ_KEYS, "requirements", where, log),
        )

    def to_json_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.extras)
        for key, default in (
            ("elite", 0),
            ("level", 0),
            ("skill_level", 0),
            ("module", -1),
            ("module_level", 0),
            ("potentiality", 0),
        ):
            value = getattr(self, key)
            if value != default:
                out[key] = value
        return out


@dataclass(frozen=True)
class CopilotOper:
    """One entry of ``opers`` (or of a group's ``opers``)."""

    name: str
    skill: int = 0
    skill_usage: SkillUsage = SkillUsage.NOT_USE
    skill_times: int = 1
    requirements: OperatorRequirements = field(default_factory=OperatorRequirements)
    role: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any], *, where: str = "", novelty: NoveltyLog | None = None
    ) -> CopilotOper:
        log = _log(novelty)
        req_raw = raw.get("requirements")
        usage = _as_int(raw.get("skill_usage"), 0)
        if usage not in tuple(SkillUsage):
            log.check(_TRAITS, f"copilot.skill_usage:{usage}", where)
            usage = 0
        return cls(
            name=str(raw.get("name", "")),
            skill=_as_int(raw.get("skill"), 0),
            skill_usage=SkillUsage(usage),
            skill_times=_as_int(raw.get("skill_times"), 1),
            requirements=(
                OperatorRequirements.parse(req_raw, where=where, novelty=log)
                if isinstance(req_raw, Mapping)
                else OperatorRequirements()
            ),
            role=str(raw.get("role", "")),
            extras=_unknown_fields(raw, _OPER_KEYS, "oper", where, log),
        )

    def to_json_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        out.update(self.extras)
        if self.skill:
            out["skill"] = self.skill
        if self.skill_usage != SkillUsage.NOT_USE:
            out["skill_usage"] = int(self.skill_usage)
        if self.skill_times != 1:
            out["skill_times"] = self.skill_times
        if not self.requirements.is_empty:
            out["requirements"] = self.requirements.to_json_obj()
        if self.role:
            out["role"] = self.role
        return out


@dataclass(frozen=True)
class CopilotGroup:
    """One entry of ``groups``: a name an action may deploy, plus candidates.

    Any one candidate satisfies the group, so a plan written against groups
    survives the reader not owning a specific operator.
    """

    name: str
    opers: tuple[CopilotOper, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any], *, where: str = "", novelty: NoveltyLog | None = None
    ) -> CopilotGroup:
        log = _log(novelty)
        raw_opers = raw.get("opers")
        opers = tuple(
            CopilotOper.parse(o, where=where, novelty=log)
            for o in (raw_opers if isinstance(raw_opers, Sequence) else ())
            if isinstance(o, Mapping)
        )
        return cls(
            name=str(raw.get("name", "")),
            opers=opers,
            extras=_unknown_fields(raw, _GROUP_KEYS, "group", where, log),
        )

    def to_json_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        out.update(self.extras)
        out["opers"] = [o.to_json_obj() for o in self.opers]
        return out


@dataclass(frozen=True)
class CopilotAction:
    """One entry of ``actions``.

    An action fires when *all* of its conditions hold (MAA ANDs them), then
    waits ``pre_delay`` ms, acts, and waits ``post_delay`` ms.  ``kills``,
    ``costs``, ``cost_changes`` and ``cooling`` are the conditions; their
    documented "no condition" values are 0, 0, 0 and -1.
    """

    type: ActionType = ActionType.DEPLOY
    #: The ``type`` string exactly as written, so re-serialising keeps the file's
    #: own language and so unknown types survive a load/save round trip.
    raw_type: str = "Deploy"
    name: str = ""
    #: ``[x, y]`` verbatim. See :func:`ato.copilot.convert.location_to_tile`.
    location: tuple[int, int] | None = None
    direction: DeployDirection = DeployDirection.RIGHT
    raw_direction: str = ""
    kills: int = 0
    costs: int = 0
    cost_changes: int = 0
    cooling: int = -1
    elapsed_time: int = 0
    skill_usage: SkillUsage = SkillUsage.NOT_USE
    skill_times: int = 1
    pre_delay: int = 0
    post_delay: int = 0
    timeout: int | None = None
    #: ``MoveCamera`` only: (dx, dy) in tiles, may be fractional.
    distance: tuple[float, float] | None = None
    #: ``CheckIfStartOver`` only: class name -> required count on field.
    tool_men: dict[str, int] = field(default_factory=dict)
    role: str = ""
    skip_if_not_ready: bool = False
    doc: str = ""
    doc_color: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def has_condition(self) -> bool:
        return bool(self.kills or self.costs or self.cost_changes or self.elapsed_time) or (
            self.cooling >= 0
        )

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any], *, where: str = "", novelty: NoveltyLog | None = None
    ) -> CopilotAction:
        log = _log(novelty)
        raw_type = str(raw.get("type", "Deploy"))
        kind = ACTION_ALIASES.get(raw_type.lower(), ActionType.UNKNOWN)
        if kind is ActionType.UNKNOWN:
            # MAA drops the action; we keep it so a corpus survey can count it.
            log.check(_TRAITS, f"copilot.action_type:{raw_type}", where)

        raw_dir = raw.get("direction")
        direction = DeployDirection.RIGHT  # MAA's fallback for anything unparseable
        if raw_dir is not None:
            key = str(raw_dir).lower()
            if key in DIRECTION_ALIASES:
                direction = DIRECTION_ALIASES[key]
            else:
                log.check(_TRAITS, f"copilot.direction:{raw_dir}", where)

        loc = raw.get("location")
        location: tuple[int, int] | None = None
        if isinstance(loc, Sequence) and not isinstance(loc, (str, bytes)) and len(loc) >= 2:
            location = (_as_int(loc[0], 0), _as_int(loc[1], 0))
        elif loc is not None:
            log.check(_TRAITS, f"copilot.location_shape:{type(loc).__name__}", where)

        dist = raw.get("distance")
        distance: tuple[float, float] | None = None
        if isinstance(dist, Sequence) and not isinstance(dist, (str, bytes)) and len(dist) >= 2:
            distance = (float(dist[0]), float(dist[1]))

        tool_men_raw = raw.get("tool_men")
        tool_men = (
            {str(k): _as_int(v, 0) for k, v in tool_men_raw.items()}
            if isinstance(tool_men_raw, Mapping)
            else {}
        )

        usage = _as_int(raw.get("skill_usage"), 0)
        if usage not in tuple(SkillUsage):
            log.check(_TRAITS, f"copilot.skill_usage:{usage}", where)
            usage = 0

        # "rear_delay" is the historical spelling; MAA still falls back to it.
        post = raw.get("post_delay")
        if post is None:
            post = raw.get("rear_delay")
        # The protocol doc says "time_elapsed", CopilotConfig.cpp reads "elapsed_time".
        elapsed = raw.get("elapsed_time")
        if elapsed is None:
            elapsed = raw.get("time_elapsed")

        return cls(
            type=kind,
            raw_type=raw_type,
            name=str(raw.get("name", "")),
            location=location,
            direction=direction,
            raw_direction=str(raw_dir) if raw_dir is not None else "",
            kills=_as_int(raw.get("kills"), 0),
            costs=_as_int(raw.get("costs"), 0),
            cost_changes=_as_int(raw.get("cost_changes"), 0),
            cooling=_as_int(raw.get("cooling"), -1),
            elapsed_time=_as_int(elapsed, 0),
            skill_usage=SkillUsage(usage),
            skill_times=_as_int(raw.get("skill_times"), 1),
            pre_delay=_as_int(raw.get("pre_delay"), 0),
            post_delay=_as_int(post, 0),
            timeout=None if raw.get("timeout") is None else _as_int(raw.get("timeout"), 0),
            distance=distance,
            tool_men=tool_men,
            role=str(raw.get("role", "")),
            skip_if_not_ready=bool(raw.get("skip_if_not_ready", False)),
            doc=str(raw.get("doc", "")),
            doc_color=str(raw.get("doc_color", "")),
            extras=_unknown_fields(raw, _ACTION_KEYS, "action", where, log),
        )

    def to_json_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.raw_type or str(self.type)}
        out.update(self.extras)
        if self.name:
            out["name"] = self.name
        if self.location is not None:
            out["location"] = [self.location[0], self.location[1]]
        if self.type is ActionType.DEPLOY or self.raw_direction:
            out["direction"] = self.raw_direction or self.direction.name.capitalize()
        for key, default in (
            ("kills", 0),
            ("costs", 0),
            ("cost_changes", 0),
            ("cooling", -1),
            ("elapsed_time", 0),
            ("pre_delay", 0),
            ("post_delay", 0),
        ):
            value = getattr(self, key)
            if value != default:
                out[key] = value
        if self.skill_usage != SkillUsage.NOT_USE or self.type is ActionType.SKILL_USAGE:
            out["skill_usage"] = int(self.skill_usage)
        if self.skill_times != 1:
            out["skill_times"] = self.skill_times
        if self.timeout is not None:
            out["timeout"] = self.timeout
        if self.distance is not None:
            out["distance"] = [self.distance[0], self.distance[1]]
        if self.tool_men:
            out["tool_men"] = dict(self.tool_men)
        if self.role:
            out["role"] = self.role
        if self.skip_if_not_ready:
            out["skip_if_not_ready"] = True
        if self.doc:
            out["doc"] = self.doc
        if self.doc_color:
            out["doc_color"] = self.doc_color
        return out


@dataclass(frozen=True)
class CopilotDoc:
    """A whole copilot file."""

    stage_name: str
    minimum_required: str = "v4.0.0"
    title: str = ""
    title_color: str = ""
    details: str = ""
    details_color: str = ""
    opers: tuple[CopilotOper, ...] = ()
    groups: tuple[CopilotGroup, ...] = ()
    actions: tuple[CopilotAction, ...] = ()
    difficulty: int = 0
    #: ``"SSS"`` for Stationary Security Service plans, which are a different
    #: protocol sharing this file extension; empty for an ordinary plan.
    doc_type: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    #: The object as loaded, so anything this class drops is still recoverable.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sss(self) -> bool:
        return self.doc_type.upper() == "SSS"

    @property
    def operator_names(self) -> tuple[str, ...]:
        names = [o.name for o in self.opers]
        names += [o.name for g in self.groups for o in g.opers]
        return tuple(dict.fromkeys(n for n in names if n))

    @property
    def group_names(self) -> frozenset[str]:
        return frozenset(g.name for g in self.groups if g.name)

    def to_json_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.doc_type:
            out["type"] = self.doc_type
        out["stage_name"] = self.stage_name
        out["minimum_required"] = self.minimum_required
        for k, v in self.extras.items():
            out.setdefault(k, v)
        meta = {
            k: v
            for k, v in (
                ("title", self.title),
                ("title_color", self.title_color),
                ("details", self.details),
                ("details_color", self.details_color),
            )
            if v
        }
        out["doc"] = meta
        if self.opers:
            out["opers"] = [o.to_json_obj() for o in self.opers]
        if self.groups:
            out["groups"] = [g.to_json_obj() for g in self.groups]
        out["actions"] = [a.to_json_obj() for a in self.actions]
        if self.difficulty:
            out["difficulty"] = self.difficulty
        return out


def parse_action(
    raw: Mapping[str, Any], *, where: str = "", novelty: NoveltyLog | None = None
) -> CopilotAction:
    return CopilotAction.parse(raw, where=where, novelty=novelty)


def parse_doc(
    raw: Mapping[str, Any], *, source: str = "", novelty: NoveltyLog | None = None
) -> CopilotDoc:
    """Build a :class:`CopilotDoc` from a decoded JSON object.

    ``source`` is only used to give novelty events a locatable context.
    """
    if not isinstance(raw, Mapping):  # a bare list or string is not a copilot doc
        raise TypeError(f"copilot document must be a JSON object, got {type(raw).__name__}")
    log = _log(novelty)
    stage = str(raw.get("stage_name", ""))
    where = f"{source or stage}"

    doc_type = str(raw.get("type", ""))
    if doc_type:
        # SSS plans carry buff/equipment/drops/tool_men sections we do not model;
        # they stay in ``raw`` and the document is flagged rather than mangled.
        log.check(_TRAITS, f"copilot.doc_type:{doc_type}", where)

    meta = raw.get("doc")
    meta = meta if isinstance(meta, Mapping) else {}
    for k in meta:
        if k not in _META_KEYS:
            log.check(_TRAITS, f"copilot.doc_meta.field:{k}", where)

    def _seq(key: str) -> list[Any]:
        v = raw.get(key)
        return list(v) if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else []

    opers = tuple(
        CopilotOper.parse(o, where=where, novelty=log) for o in _seq("opers") if isinstance(o, Mapping)
    )
    groups = tuple(
        CopilotGroup.parse(g, where=where, novelty=log)
        for g in _seq("groups")
        if isinstance(g, Mapping)
    )
    actions = tuple(
        CopilotAction.parse(a, where=f"{where} action[{i}]", novelty=log)
        for i, a in enumerate(_seq("actions"))
        if isinstance(a, Mapping)
    )

    known = _DOC_KEYS | _DOC_EXTRA_KEYS
    extras = {k: v for k, v in raw.items() if k not in known}
    for k in extras:
        log.check(_TRAITS, f"copilot.doc.field:{k}", where)
    # version/video_url are known-but-unmodelled; keep them without crying novelty.
    for k in ("version", "video_url"):
        if k in raw:
            extras[k] = raw[k]

    return CopilotDoc(
        stage_name=stage,
        minimum_required=str(raw.get("minimum_required", "")),
        title=str(meta.get("title", "")),
        title_color=str(meta.get("title_color", "")),
        details=str(meta.get("details", "")),
        details_color=str(meta.get("details_color", "")),
        opers=opers,
        groups=groups,
        actions=actions,
        difficulty=_as_int(raw.get("difficulty"), 0),
        doc_type=doc_type,
        extras=extras,
        raw=dict(raw),
    )
