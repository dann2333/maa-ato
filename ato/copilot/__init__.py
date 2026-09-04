"""Reading and writing MAA "copilot" (作业) battle plans.

MaaAssistantArknights scripts a stage clear as an ordered list of conditional
actions.  The community has published tens of thousands of these, which makes
them the cheapest human tactical demonstrations available to ATO: we read them
for imitation learning and write them so ATO's own plans can be replayed by a
tool people already trust.

A copilot file is *not* a runtime information channel into the game.  It is a
static text file describing what a human did; executing one still requires
seeing the screen and touching it.

Sources of truth
----------------
Everything in :mod:`ato.copilot.schema` was derived from these, fetched
2026-09-04 (raw.githubusercontent.com; api.github.com is unreachable here):

* ``https://raw.githubusercontent.com/MaaAssistantArknights/MaaAssistantArknights/dev/docs/zh-cn/protocol/copilot-schema.md``
* ``https://raw.githubusercontent.com/MaaAssistantArknights/MaaAssistantArknights/dev/docs/en-us/protocol/copilot-schema.md``
  — the protocol documentation, field by field.
* ``.../MaaAssistantArknights/dev/src/MaaCore/Config/Miscellaneous/CopilotConfig.cpp``
  — the parser MAA actually runs; the authority whenever it disagrees with the docs.
* ``.../MaaAssistantArknights/dev/src/MaaCore/Common/AsstBattleDef.h``
  — the ``ActionType`` / ``SkillUsage`` / ``DeployDirection`` / ``Role`` enums.
* ``.../MaaAssistantArknights/dev/src/MaaCore/Config/Miscellaneous/TilePack.cpp``
  — how MAA turns a level into tiles, which fixes the ``location`` axes.
* ``.../MaaAssistantArknights/dev/docs/en-us/protocol/sss-schema.md``
  — the separate ``"type": "SSS"`` protocol, detected but not modelled.
* ``https://raw.githubusercontent.com/MaaAssistantArknights/maa-copilot-frontend/main/src/models/copilot.schema.json``
* ``.../maa-copilot-frontend/main/src/models/types.ts``
  — the web editor's JSON Schema and action-type table.
* ``https://prts.maa.plus/copilot/get/<id>`` — 60 real published documents, used
  to check which fields actually occur and to verify the coordinate convention.
* ``https://raw.githubusercontent.com/MaaAssistantArknights/MaaAssistantArknights/master/resource/copilot/OF-1_credit_fight.json``
  — MAA's own bundled example.

Fields whose meaning could not be confirmed are listed in
:data:`ato.copilot.schema.UNCONFIRMED`.
"""

from __future__ import annotations

from ato.copilot.convert import (
    MAA_LOCATION_FIXTURE,
    NameResolution,
    PlannedAction,
    build_name_index,
    from_plan,
    location_to_display_tile,
    location_to_tile,
    reproject_plan,
    resolve_names,
    tile_to_location,
    to_plan,
)
from ato.copilot.io import (
    CopilotStats,
    ParseFailure,
    dumps,
    iter_corpus,
    iter_documents,
    load_doc,
    loads,
    save_doc,
    scan_corpus,
    unwrap_envelope,
    validate,
)
from ato.copilot.schema import (
    ACTION_ALIASES,
    UNCONFIRMED,
    ActionType,
    CopilotAction,
    CopilotDoc,
    CopilotGroup,
    CopilotOper,
    DeployDirection,
    OperatorRequirements,
    SkillUsage,
    parse_action,
    parse_doc,
)

__all__ = [
    "ACTION_ALIASES",
    "MAA_LOCATION_FIXTURE",
    "UNCONFIRMED",
    "ActionType",
    "CopilotAction",
    "CopilotDoc",
    "CopilotGroup",
    "CopilotOper",
    "CopilotStats",
    "DeployDirection",
    "NameResolution",
    "OperatorRequirements",
    "ParseFailure",
    "PlannedAction",
    "SkillUsage",
    "build_name_index",
    "dumps",
    "from_plan",
    "iter_corpus",
    "iter_documents",
    "load_doc",
    "loads",
    "location_to_display_tile",
    "location_to_tile",
    "parse_action",
    "parse_doc",
    "reproject_plan",
    "resolve_names",
    "save_doc",
    "scan_corpus",
    "tile_to_location",
    "to_plan",
    "unwrap_envelope",
    "validate",
]
