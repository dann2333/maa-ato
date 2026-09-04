"""Copilot documents <-> ATO's internal plan representation.

Why the plan is condition-based
-------------------------------
A copilot action does not say "at t=13.4s, deploy Thorns".  It says "when the
kill counter reaches 6, deploy Thorns".  That is not a stylistic choice, it is
what makes these plans survive contact with reality: emulator frame rate,
device speed, the 2x-speed toggle, spawn jitter and the player's own reaction
time all move wall-clock time around, while the kill counter and the DP bar are
*game state* that both the human and the agent can read off the screen and that
advance in the same order every run.

So ATO's plan keeps the same abstraction.  ``trigger_kills`` / ``trigger_cost``
/ ``trigger_cost_change`` are the preconditions; ``pre_delay`` / ``post_delay``
are only the small local pauses MAA needs to let an animation finish.  A
planner that emitted absolute timestamps would produce plans that work in the
simulator and desynchronise on hardware; a planner that emits conditions
produces plans that can be exported to MAA and replayed by anyone.

Coordinates
-----------
The one real hazard in this format is the ``location`` axis convention; see
:func:`location_to_tile`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ato.copilot.schema import (
    ActionType,
    CopilotAction,
    CopilotDoc,
    CopilotOper,
    DeployDirection,
    SkillUsage,
)
from ato.sim.registry import MechanismKind, NoveltyLog
from ato.sim.types import Direction, Tile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ato.gamedata.tables import GameData

#: MAA delays are milliseconds; ATO time is seconds.
_MS = 1000.0

#: Valid SkillUsage values, so a round trip cannot resurrect an out-of-range one.
_SKILL_USAGE_VALUES = frozenset(int(u) for u in SkillUsage)

#: Fixture behind :func:`location_to_tile`, kept in code so the convention can
#: be re-checked (or flipped) in one place.  Each row is
#: ``(level_id, map_height, location, expected_tile, why)`` and was verified on
#: 2026-09-04 against the level files and 60 published copilot documents.
MAA_LOCATION_FIXTURE: tuple[tuple[str, int, tuple[int, int], Tile, str], ...] = (
    (
        "Obt/Main/level_main_04-01",
        7,
        (8, 2),
        Tile(4, 8),
        "德克萨斯 (Texas, a melee Vanguard); mapData.map[2][8] is tile_road/MELEE "
        "while map[4][8] is tile_wall/RANGED",
    ),
    (
        "Obt/Main/level_main_04-01",
        7,
        (8, 4),
        Tile(2, 8),
        "圣聆初雪 (a ranged operator); mapData.map[4][8] is tile_wall/RANGED "
        "while map[2][8] is tile_road/MELEE",
    ),
    (
        "Obt/Main/level_sub_05-3-2",
        8,
        (5, 2),
        Tile(5, 5),
        "娜仁图亚 (a Sniper); mapData.map[2][5] is tile_wall/RANGED while "
        "map[5][5] is tile_forbidden/NONE, which nobody can stand on",
    ),
)


def location_to_tile(location: Sequence[int], map_height: int) -> Tile:
    """Convert a copilot ``location`` to an ATO :class:`~ato.sim.types.Tile`.

    ``location`` is ``[x, y]`` where ``x`` is the column from the left and ``y``
    indexes ``level.mapData.map`` directly — that is, ``y`` is the **display**
    row, counted from the *top*.  ATO's ``Tile.row`` counts from the *bottom*,
    so ``row = map_height - 1 - y``.

    Confirmed, not assumed:

    * ``TilePack.cpp`` builds MAA's tile table as
      ``for y in range(h): for x in range(w): loc = Point(x, y)`` over
      ``level.get_item(y, x)``, which indexes ``mapData.map`` top row first.
    * Empirically, over 220 Deploy actions in 60 published copilot documents:
      under this convention 220/220 land on a buildable tile and 201/202 match
      the deployed operator's melee/ranged restriction.  Under the opposite
      convention only 137/220 and 85/202 do.  On the 151 actions where the two
      readings disagree, this one is right every time.
    * :data:`MAA_LOCATION_FIXTURE` records three of those disagreeing cases.
    """
    x, y = int(location[0]), int(location[1])
    return Tile(map_height - 1 - y, x)


def location_to_display_tile(location: Sequence[int]) -> Tile:
    """Read a ``location`` as a tile **without** flipping the row.

    For callers that have no level file to hand, and therefore cannot know the
    map height.  The column is correct; ``row`` is the display row, so such a
    tile must be passed through :func:`reproject_plan` before it means anything
    to the simulator.
    """
    return Tile(int(location[1]), int(location[0]))


class _DisplayRows:
    """Sentinel: keep copilot display rows as they are, do not convert."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "DISPLAY_ROWS"


#: Pass this as ``map_height`` to state that the rows are already copilot display
#: rows and no conversion is wanted. Spelling it out is the point: copilot counts
#: rows from the top of the map and ATO counts from the bottom, so a silent
#: no-op writes a vertically mirrored plan that looks entirely reasonable and
#: deploys every operator on the wrong side of the map.
DISPLAY_ROWS = _DisplayRows()

RowBasis = int | _DisplayRows


def tile_to_location(tile: Tile, map_height: RowBasis) -> list[int]:
    """Inverse of :func:`location_to_tile`.

    ``map_height`` is mandatory. It used to default to "do not convert", which
    made the mirrored output the easiest thing to produce by accident -- an ATO
    plan always carries bottom-origin rows, so exporting one without the map
    height silently flipped it.
    """
    if isinstance(map_height, _DisplayRows):
        return [tile.col, tile.row]
    return [tile.col, map_height - 1 - tile.row]


@dataclass(frozen=True)
class PlannedAction:
    """One step of an ATO battle plan.

    The trigger is a *condition*, never a timestamp — see the module docstring.
    ``pre_delay``/``post_delay`` are seconds.
    """

    kind: str  # deploy | retreat | skill | speed | wait
    char_name: str  # display name as written in the copilot file
    tile: Tile | None
    direction: Direction | None
    # the trigger: copilot actions fire on a condition, not a wall-clock time
    trigger_kills: int | None
    trigger_cost: int | None
    trigger_cost_change: int | None
    pre_delay: float
    post_delay: float
    #: Which of the operator's three skills the plan brings, and how the plan
    #: wants it fired. Carried from ``opers[]`` rather than dropped: which skill
    #: an operator brought changes what the plan means, so a demonstration
    #: stripped of it teaches the wrong lesson.
    skill_index: int = 0
    skill_usage: int = 0
    skill_times: int = 1


#: The subset of MAA action types ATO's planner has a concept of. ``Output`` is
#: MAA's no-op action, which is exactly what "hold until this condition" is.
_KIND_OF: dict[ActionType, str] = {
    ActionType.DEPLOY: "deploy",
    ActionType.RETREAT: "retreat",
    ActionType.SKILL: "skill",
    ActionType.SPEED_UP: "speed",
    ActionType.OUTPUT: "wait",
}
_TYPE_OF: dict[str, ActionType] = {v: k for k, v in _KIND_OF.items()}

#: Everything else is a MAA execution detail (UI text, camera work, autopilot
#: hand-off, SSS card draws) with no offline meaning. Dropping them is a
#: decision, so :func:`to_plan` records each one as novelty.
DROPPED_TYPES: frozenset[ActionType] = frozenset(set(ActionType) - set(_KIND_OF))


def _direction_of(action: CopilotAction) -> Direction | None:
    if action.type is not ActionType.DEPLOY:
        return None
    if action.direction is DeployDirection.NONE:
        return None  # drones and devices have no facing
    return Direction(int(action.direction))


def to_plan(
    doc: CopilotDoc,
    *,
    map_height: RowBasis = DISPLAY_ROWS,
    novelty: NoveltyLog | None = None,
) -> tuple[PlannedAction, ...]:
    """Translate a copilot document into an ATO plan.

    Pass ``map_height`` (``len(level["mapData"]["map"])``, or
    ``BattleMap.height``) to get tiles in ATO's bottom-origin convention.
    Without it the columns are right and the rows are still the copilot's
    display rows; :func:`reproject_plan` fixes them once the map is known.
    Either way ``from_plan`` inverts ``to_plan`` exactly, as long as the same
    ``map_height`` is used for both.
    """
    log = novelty if novelty is not None else NoveltyLog(strict=False)
    # Skill choice lives on the roster entry, not on the action, so index it up
    # front and attach it to each step the operator takes.
    loadout: dict[str, tuple[int, int, int]] = {}
    for group in doc.groups:
        for oper in group.opers:
            loadout[oper.name] = (oper.skill, int(oper.skill_usage), oper.skill_times)
    for oper in doc.opers:
        loadout[oper.name] = (oper.skill, int(oper.skill_usage), oper.skill_times)
    out: list[PlannedAction] = []
    for i, action in enumerate(doc.actions):
        kind = _KIND_OF.get(action.type)
        if kind is None:
            _report_drop(log, doc, i, action)
            continue
        tile: Tile | None = None
        if action.location is not None:
            tile = (
                location_to_display_tile(action.location)
                if isinstance(map_height, _DisplayRows)
                else location_to_tile(action.location, map_height)
            )
        skill_index, skill_usage, skill_times = loadout.get(action.name, (0, 0, 1))
        out.append(
            PlannedAction(
                kind=kind,
                char_name=action.name,
                tile=tile,
                direction=_direction_of(action),
                # 0 / -1 are the format's "no condition" values, not real thresholds
                trigger_kills=action.kills or None,
                trigger_cost=action.costs or None,
                trigger_cost_change=action.cost_changes or None,
                pre_delay=action.pre_delay / _MS,
                post_delay=action.post_delay / _MS,
                skill_index=skill_index,
                skill_usage=skill_usage,
                skill_times=skill_times,
            )
        )
    return tuple(out)


def _report_drop(log: NoveltyLog, doc: CopilotDoc, index: int, action: CopilotAction) -> None:
    key = f"copilot.plan.dropped:{action.raw_type or action.type}"
    where = f"{doc.stage_name} action[{index}]"
    if log.strict:
        # A dropped action is a known limit of the plan vocabulary, not a
        # mechanic we failed to model, so it must never abort a corpus scan.
        NoveltyLog(strict=False, events=log.events).report(MechanismKind.TRAIT, key, where)
    else:
        log.report(MechanismKind.TRAIT, key, where)


def reproject_plan(plan: Iterable[PlannedAction], map_height: int) -> tuple[PlannedAction, ...]:
    """Flip display-convention rows to ATO rows, for a plan built without a map."""
    return tuple(
        a if a.tile is None else replace(a, tile=Tile(map_height - 1 - a.tile.row, a.tile.col))
        for a in plan
    )


def from_plan(
    actions: Iterable[PlannedAction],
    stage_name: str,
    *,
    map_height: RowBasis,
    title: str = "",
    details: str = "",
    minimum_required: str = "v4.0.0",
) -> CopilotDoc:
    """Render an ATO plan as a copilot document MAA can execute.

    ``opers`` is filled from the operators the plan deploys so MAA's
    auto-formation knows whom to bring; skills are left unset because the plan
    does not carry them.
    """
    plan = list(actions)
    built: list[CopilotAction] = []
    for step in plan:
        kind = _TYPE_OF.get(step.kind)
        if kind is None:
            raise ValueError(f"unknown planned action kind {step.kind!r}")
        direction = DeployDirection.NONE if step.direction is None else DeployDirection(int(step.direction))
        built.append(
            CopilotAction(
                type=kind,
                raw_type=str(kind),
                name=step.char_name,
                location=(
                    None
                    if step.tile is None
                    else tuple(tile_to_location(step.tile, map_height))  # type: ignore[arg-type]
                ),
                direction=direction,
                kills=step.trigger_kills or 0,
                costs=step.trigger_cost or 0,
                cost_changes=step.trigger_cost_change or 0,
                pre_delay=round(step.pre_delay * _MS),
                post_delay=round(step.post_delay * _MS),
            )
        )
    # Rebuild the roster from the plan, carrying each operator's skill choice
    # back out. A round trip that dropped it would quietly rewrite a plan that
    # depends on someone's second skill into one that brings their first.
    loadout: dict[str, tuple[int, int, int]] = {}
    for step in plan:
        if step.kind == "deploy" and step.char_name:
            loadout.setdefault(
                step.char_name, (step.skill_index, step.skill_usage, step.skill_times)
            )
    return CopilotDoc(
        stage_name=stage_name,
        minimum_required=minimum_required,
        title=title,
        details=details,
        opers=tuple(
            CopilotOper(
                name=n,
                skill=sk,
                skill_usage=SkillUsage(use) if use in _SKILL_USAGE_VALUES else SkillUsage.NOT_USE,
                skill_times=times,
            )
            for n, (sk, use, times) in loadout.items()
        ),
        actions=tuple(built),
    )


# ---------------------------------------------------------------------------
# name -> char_id
# ---------------------------------------------------------------------------

def _rank(char_id: str, entry: dict[str, Any]) -> int:
    """Preference order when one display name maps to several table entries.

    A stated rule, not a guess: copilot files name units a player can put on
    the field, so playable operators (0) outrank integrated-strategy-only
    variants (1), which outrank summons (2), which outrank the ``trap_*`` props
    that reuse operator names (3).
    """
    if char_id.startswith("char_"):
        return 0 if not entry.get("isNotObtainable") else 1
    if char_id.startswith("token_"):
        return 2
    return 3


def build_name_index(gamedata: GameData) -> dict[str, tuple[str, ...]]:
    """Map every display name in ``character_table`` to its candidate char ids.

    Candidates are ordered by :func:`_rank`; ties within the best rank are
    what :func:`resolve_names` reports as ambiguous.
    """
    index: dict[str, dict[str, int]] = {}
    for char_id, entry in gamedata.characters.items():
        if not isinstance(entry, dict):
            continue
        rank = _rank(char_id, entry)
        # name and appellation coincide for units with no Chinese name ("12F",
        # "Lancet-2"), so index by id to keep one candidate, not two.
        for key in (entry.get("name"), entry.get("appellation")):
            if isinstance(key, str) and key:
                index.setdefault(key, {})[char_id] = rank
    return {
        name: tuple(cid for cid, _ in sorted(cands.items(), key=lambda kv: (kv[1], kv[0])))
        for name, cands in index.items()
    }


@dataclass(frozen=True)
class NameResolution:
    """Outcome of matching copilot display names against ``character_table``."""

    mapping: dict[str, str]
    ambiguous: dict[str, tuple[str, ...]]
    missing: tuple[str, ...]
    groups: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.ambiguous and not self.missing

    def report(self) -> str:
        lines = [f"resolved {len(self.mapping)} name(s)"]
        if self.groups:
            lines.append(f"  group names (not operators): {', '.join(sorted(self.groups))}")
        for name, cands in sorted(self.ambiguous.items()):
            lines.append(f"  ambiguous {name}: {', '.join(cands)}")
        if self.missing:
            lines.append(f"  missing: {', '.join(sorted(self.missing))}")
        return "\n".join(lines)


def resolve_names(
    plan: Iterable[PlannedAction] | CopilotDoc,
    gamedata: GameData,
    *,
    known_groups: Iterable[str] = (),
) -> NameResolution:
    """Map the display names a plan uses to ``char_*`` ids.

    Names that a copilot file defines as *group* names are not operators; pass
    them in ``known_groups`` (or pass the whole :class:`CopilotDoc`, which knows
    its own) so they are reported separately instead of as missing.

    Ambiguity is reported, never guessed away: a name that survives the
    preference order with more than one candidate lands in ``ambiguous`` with
    all of them.
    """
    groups = set(known_groups)
    if isinstance(plan, CopilotDoc):
        groups |= set(plan.group_names)
        names = [a.name for a in plan.actions if a.name] + list(plan.operator_names)
    else:
        names = [s.char_name for s in plan if s.char_name]

    index = build_name_index(gamedata)
    mapping: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    seen_groups: list[str] = []
    for name in dict.fromkeys(names):
        if name in groups:
            seen_groups.append(name)
            continue
        cands = index.get(name)
        if not cands:
            missing.append(name)
            continue
        best = _rank_of(cands[0], gamedata)
        top = tuple(c for c in cands if _rank_of(c, gamedata) == best)
        if len(top) == 1:
            mapping[name] = top[0]
        else:
            ambiguous[name] = top
    return NameResolution(mapping, ambiguous, tuple(missing), tuple(seen_groups))


def _rank_of(char_id: str, gamedata: GameData) -> int:
    entry = gamedata.characters.get(char_id)
    return _rank(char_id, entry if isinstance(entry, dict) else {})
