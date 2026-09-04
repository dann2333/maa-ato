"""The ``ato`` command line: inspect snapshots, stages, scenarios and operators.

Everything here is a read-only view over the *public static data* plus the
offline simulator's compilers. Nothing in this module talks to a device, and
nothing it prints is available to the live agent — the runtime channel into the
game is pixels in, touches out (see ``docs/INVARIANTS.md``, I-1).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from ato import __version__
from ato.config import SNAPSHOT_DIR
from ato.gamedata.fetch import Manifest, fetch_snapshot, latest_snapshot
from ato.gamedata.models import (
    PHASE_INDEX,
    OperatorSpec,
    resolve_operator,
    resolve_range,
    resolve_skill,
)
from ato.gamedata.sources import TABLES, level_path
from ato.gamedata.tables import GameData
from ato.sim.pathing import route_waypoints
from ato.sim.scenario import Difficulty, Scenario, compile_scenario

EXIT_OK = 0
EXIT_ERROR = 1
#: Reserved for "this feature is not built yet" so scripts can tell it apart
#: from a genuine failure. argparse uses 2 for usage errors as well.
EXIT_UNAVAILABLE = 2

#: character_table also holds traps and summons; they are findable by exact id
#: but must never win a fuzzy name match against a real operator.
_NON_PLAYABLE = frozenset({"TOKEN", "TRAP"})

_DIFFICULTY_BITS = ("NORMAL", "FOUR_STAR", "EASY", "SIX_STAR")

_SP_TYPE_SHORT = {
    "INCREASE_WITH_TIME": "time",
    "INCREASE_WHEN_ATTACK": "attack",
    "INCREASE_WHEN_TAKEN_DAMAGE": "damage",
}


class CliError(RuntimeError):
    """An error with a message meant for a human, not a traceback."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _display_width(text: str) -> int:
    """Terminal columns a string occupies; enemy/operator names are CJK."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _num(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf: print as-is
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _num(value)
    if isinstance(value, (list, tuple)):
        return ",".join(_cell(v) for v in value) if value else "-"
    return str(value)


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    align: str = "",
    indent: str = "  ",
    gap: str = "  ",
) -> list[str]:
    """Aligned plain-text columns. ``align`` is one of ``l``/``r`` per column."""
    head = [str(h) for h in headers]
    body = [[_cell(c) for c in row] for row in rows]
    widths = [_display_width(h) for h in head]
    for row in body:
        for i, cell in enumerate(row[: len(widths)]):
            widths[i] = max(widths[i], _display_width(cell))

    def line(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells[: len(widths)]):
            pad = " " * max(widths[i] - _display_width(cell), 0)
            out.append(pad + cell if (align[i : i + 1] or "l") == "r" else cell + pad)
        return (indent + gap.join(out)).rstrip()

    return [line(head), line(["-" * w for w in widths]), *(line(r) for r in body)]


def _rule(title: str, width: int = 78) -> str:
    return f"-- {title} " + "-" * max(width - _display_width(title) - 4, 3)


def _wrap_items(items: Sequence[str], *, indent: str = "    ", width: int = 96) -> list[str]:
    lines: list[str] = []
    cur = indent
    for item in items:
        if cur != indent and _display_width(cur) + 1 + _display_width(item) > width:
            lines.append(cur)
            cur = indent
        cur += item if cur == indent else " " + item
    if cur != indent:
        lines.append(cur)
    return lines


def _kv(pairs: Sequence[tuple[str, Any]], *, indent: str = "  ") -> list[str]:
    width = max((len(k) for k, _ in pairs), default=0)
    return [f"{indent}{k.ljust(width)}  {_cell(v)}" for k, v in pairs]


def _difficulty_name(mask: Difficulty) -> str:
    parts = [n for n in _DIFFICULTY_BITS if Difficulty[n] & mask]
    return "|".join(parts) if parts else "NONE"


def _dump_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def _emit(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def _open_gamedata(server: str) -> GameData:
    """Open the newest local snapshot. Never downloads: fetching is explicit."""
    try:
        return GameData.open(server, download=False)
    except FileNotFoundError as exc:
        raise CliError(
            f"no local game-data snapshot for server {server!r} under {SNAPSHOT_DIR}; "
            f"run: ato gamedata fetch --server {server}"
        ) from exc


def _resolve_snapshot(token: str) -> Path:
    """Accept a path, a directory name (``cn-77.0.0``) or a bare version."""
    candidates = [Path(token), SNAPSHOT_DIR / token]
    candidates += [SNAPSHOT_DIR / f"{s}-{token}" for s in ("cn", "en")]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise CliError(f"no snapshot matching {token!r}; see `ato gamedata info`")


def _manifest_of(snapshot: Path) -> Manifest:
    path = snapshot / "manifest.json"
    if not path.exists():
        raise CliError(f"{snapshot} has no manifest.json — re-fetch it")
    return Manifest.from_json(path.read_text("utf-8"))


def _gamedata_at(path: Path) -> GameData:
    server = path.name.split("-", 1)[0]
    if (path / "manifest.json").exists():
        server = _manifest_of(path).server or server
    return GameData(path, server if server in ("cn", "en") else "cn")


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _find_stage(gd: GameData, token: str) -> tuple[str, dict[str, Any]]:
    """Resolve a stage id, falling back to the human-facing code (``1-7``)."""
    stages = gd.stages
    if token in stages:
        return token, stages[token]
    by_code = gd.stage_by_code(token)
    if by_code is not None:
        return str(by_code.get("stageId") or token), by_code
    low = token.lower()
    for sid, st in stages.items():
        if sid.lower() == low:
            return sid, st
    raise CliError(f"unknown stage {token!r}; try: ato stage find {token}")


def _rarity(char: dict[str, Any]) -> int:
    """Stars. Modern tables say ``TIER_5``; older ones stored 0-based ints."""
    raw = char.get("rarity")
    if isinstance(raw, str) and raw.startswith("TIER_"):
        return int(raw[5:])
    return int(raw) + 1 if isinstance(raw, int) and not isinstance(raw, bool) else 0


def _find_operator(gd: GameData, token: str) -> tuple[str, dict[str, Any]]:
    chars = gd.characters
    if token in chars:
        return token, chars[token]
    low = token.lower()
    # `low in (a, b)` is an equality test against either; the `partial` list below
    # is the substring one.
    exact = [
        (k, v)
        for k, v in chars.items()
        if low in ((v.get("name") or "").lower(), (v.get("appellation") or "").lower())
    ]
    partial = [
        (k, v)
        for k, v in chars.items()
        if low in k.lower()
        or low in (v.get("name") or "").lower()
        or low in (v.get("appellation") or "").lower()
    ]
    for pool in (exact, partial):
        if not pool:
            continue
        playable = [kv for kv in pool if kv[1].get("profession") not in _NON_PLAYABLE]
        chosen = playable or pool
        if len(chosen) == 1:
            return chosen[0]
        listing = "\n".join(
            f"  {k:<28} {v.get('name') or ''} ({v.get('appellation') or ''})"
            for k, v in sorted(chosen)[:20]
        )
        more = "" if len(chosen) <= 20 else f"\n  ... and {len(chosen) - 20} more"
        raise CliError(f"{token!r} matches {len(chosen)} characters:\n{listing}{more}")
    raise CliError(f"no operator matches {token!r}")


# ---------------------------------------------------------------------------
# ato gamedata
# ---------------------------------------------------------------------------


def cmd_gamedata_fetch(args: argparse.Namespace) -> int:
    try:
        path, manifest = fetch_snapshot(args.server, force=args.force)
    except (RuntimeError, OSError, KeyError) as exc:
        raise CliError(f"fetch failed: {exc}") from exc

    total = sum(f.size for f in manifest.files.values())
    _emit(
        _kv(
            [
                ("path", path),
                ("server", manifest.server),
                ("version", manifest.version),
                ("stream", manifest.stream),
                ("files", len(manifest.files)),
                ("size", f"{total / 1e6:.1f} MB"),
            ]
        )
    )
    rows = [
        (name, rec.path, f"{rec.size / 1e6:.2f}", rec.sha256[:12])
        for name, rec in sorted(manifest.files.items())
    ]
    print()
    _emit(_table(("table", "source path", "MB", "sha256"), rows, align="llrl"))
    missing = [t.name for t in TABLES if t.name not in manifest.files]
    if missing:
        print()
        print("declared in ato.gamedata.sources but absent here: " + ", ".join(missing))
    return EXIT_OK


def _snapshot_row(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    # Directories are named ``<server>-<version>``; the manifest wins when readable.
    server, _, version = path.name.partition("-")
    stream, files = "", 0
    if manifest_path.exists():
        try:
            m = Manifest.from_json(manifest_path.read_text("utf-8"))
        except (ValueError, KeyError):
            stream = "unreadable manifest"
        else:
            version, server, stream, files = m.version, m.server, m.stream, len(m.files)
    levels = path / "levels"
    return {
        "path": str(path),
        "server": server,
        "version": version,
        "stream": stream,
        "tables": files,
        "levels": len(list(levels.rglob("*.json"))) if levels.is_dir() else 0,
        "bytes": _dir_bytes(path),
    }


def cmd_gamedata_info(args: argparse.Namespace) -> int:
    root = SNAPSHOT_DIR
    snaps = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    rows = [_snapshot_row(p) for p in snaps]
    servers = sorted({r["server"] for r in rows})
    newest = {s: latest_snapshot(server=s) for s in servers}
    latest_paths = {str(p) for p in newest.values() if p is not None}
    for r in rows:
        r["latest"] = r["path"] in latest_paths

    if args.as_json:
        _dump_json(
            {
                "root": str(root),
                "snapshots": rows,
                "latest": {s: str(p) if p else None for s, p in newest.items()},
            }
        )
        return EXIT_OK

    print(f"snapshot root  {root}")
    if not rows:
        print("no snapshots yet; run: ato gamedata fetch")
        return EXIT_OK
    _emit(
        _table(
            ("", "server", "version", "tables", "levels", "MB", "path"),
            [
                (
                    "*" if r["latest"] else "",
                    r["server"],
                    r["version"],
                    r["tables"],
                    r["levels"],
                    f"{r['bytes'] / 1e6:.1f}",
                    r["path"],
                )
                for r in rows
            ],
            align="lllrrrl",
        )
    )
    print("\n* = newest for that server (what `GameData.open` uses)")
    return EXIT_OK


def _brief(value: Any) -> str:
    """One-line preview of an unknown value, for rendering a foreign result."""
    if isinstance(value, dict):
        keys = ", ".join(str(k) for k in list(value)[:6])
        return f"{len(value)} entries" + (f": {keys}" if keys else "")
    if isinstance(value, (list, tuple, set, frozenset)):
        items = ", ".join(str(v) for v in list(value)[:6])
        return f"{len(value)} items" + (f": {items}" if items else "")
    return _cell(value)


def _summarise(obj: Any) -> list[str]:
    """Render whatever ``diff_snapshots`` returned without assuming its shape."""
    if isinstance(obj, str):
        return obj.splitlines()
    for attr in ("summary", "report", "describe"):
        value = getattr(obj, attr, None)
        if callable(value):
            try:
                text = value()
            except TypeError:
                continue
            if isinstance(text, str):
                return text.splitlines()
        elif isinstance(value, str):
            return value.splitlines()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _kv([(f.name, _brief(getattr(obj, f.name))) for f in dataclasses.fields(obj)])
    if isinstance(obj, dict):
        return _kv([(str(k), _brief(v)) for k, v in obj.items()])
    if isinstance(obj, (list, tuple)):
        return [f"  {item}" for item in obj]
    return [f"  {obj!r}"]


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (str, int, float, bool, type(None), list, dict)):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except TypeError:
            pass
    to_json = getattr(obj, "to_json", None)
    if callable(to_json):
        try:
            return json.loads(to_json())
        except (TypeError, ValueError):
            pass
    return getattr(obj, "__dict__", None) or str(obj)


def cmd_gamedata_diff(args: argparse.Namespace) -> int:
    # Imported lazily: this module is written independently of the rest of the
    # CLI, and its absence must not break any other command.
    try:
        from ato.gamedata import diff as diff_mod
    except ImportError as exc:
        print(f"ato gamedata diff: not available yet ({exc})", file=sys.stderr)
        return EXIT_UNAVAILABLE
    fn = getattr(diff_mod, "diff_snapshots", None)
    if not callable(fn):
        print(
            "ato gamedata diff: not available yet "
            "(ato.gamedata.diff defines no diff_snapshots)",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE

    old, new = _resolve_snapshot(args.old), _resolve_snapshot(args.new)
    params = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    # diff.py is owned by another module; adapt to whatever it asks for.
    annotations = " ".join(str(p.annotation) for p in params[:2])
    if "GameData" in annotations:
        adapt: Callable[[Path], Any] = _gamedata_at
    elif "Manifest" in annotations:
        adapt = _manifest_of
    else:
        adapt = Path  # it already is one; diff takes the directory itself
    pair = (adapt(old), adapt(new))
    try:
        result = fn(*pair)
    except (TypeError, AttributeError) as exc:
        print(
            f"ato gamedata diff: diff_snapshots did not accept "
            f"{type(pair[0]).__name__} arguments ({exc})",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE
    except (OSError, ValueError, KeyError) as exc:
        raise CliError(f"diff failed: {exc}") from exc

    if args.as_json:
        _dump_json({"old": str(old), "new": str(new), "diff": _jsonable(result)})
        return EXIT_OK
    print(f"old  {old}")
    print(f"new  {new}")
    print()
    _emit(_summarise(result))
    return EXIT_OK


# ---------------------------------------------------------------------------
# ato stage
# ---------------------------------------------------------------------------


def cmd_stage_find(args: argparse.Namespace) -> int:
    gd = _open_gamedata(args.server)
    query = args.query.strip()
    low = query.lower()
    hits: list[tuple[int, str, dict[str, Any]]] = []
    for sid, st in gd.stages.items():
        code = str(st.get("code") or "")
        name = str(st.get("name") or "")
        if not (low in sid.lower() or low in code.lower() or low in name.lower()):
            continue
        rank = 0 if low in (sid.lower(), code.lower()) else 1 if code.lower().startswith(low) else 2
        hits.append((rank, sid, st))
    hits.sort(key=lambda h: (h[0], str(h[2].get("code") or ""), h[1]))

    if not hits:
        raise CliError(f"no stage matches {query!r}")
    shown = hits if args.limit <= 0 else hits[: args.limit]
    records = [
        {
            "stageId": sid,
            "code": st.get("code"),
            "name": st.get("name"),
            "apCost": st.get("apCost"),
            "difficulty": st.get("difficulty"),
            "stageType": st.get("stageType"),
            "levelId": st.get("levelId"),
        }
        for _, sid, st in shown
    ]
    if args.as_json:
        _dump_json({"query": query, "matches": len(hits), "stages": records})
        return EXIT_OK

    _emit(
        _table(
            ("stageId", "code", "name", "ap", "difficulty", "type", "levelId"),
            [
                (
                    r["stageId"],
                    r["code"],
                    r["name"],
                    r["apCost"],
                    r["difficulty"],
                    r["stageType"],
                    r["levelId"],
                )
                for r in records
            ],
            align="lllrlll",
        )
    )
    if len(shown) < len(hits):
        print(f"\n{len(hits) - len(shown)} more matches (use --limit 0 for all)")
    return EXIT_OK


def _load_scenario(
    gd: GameData, token: str, difficulty: str | None
) -> tuple[str, dict[str, Any], Difficulty, Scenario]:
    stage_id, stage = _find_stage(gd, token)
    level_id = str(stage.get("levelId") or "")
    if not level_id:
        raise CliError(f"stage {stage_id} has no levelId (story-only stage, nothing to simulate)")
    # A stage and its Adverse variant share one level file, so the stage's own
    # difficulty is the right default; --difficulty forces the other reading.
    want = Difficulty.parse(difficulty or stage.get("difficulty") or "NORMAL")
    try:
        level = gd.level_json(level_id)
    except ValueError as exc:
        # A cached 404 body parses as garbage; say which file to delete.
        cached = gd.root / "levels" / level_path(level_id)
        raise CliError(f"level {level_id} is not valid JSON ({exc}); delete {cached}") from exc
    except (RuntimeError, OSError) as exc:
        raise CliError(f"could not load level {level_id}: {exc}") from exc
    # strict=False: an unmodelled mechanic must be reported, not raised, when a
    # human is only looking at the stage.
    scenario = compile_scenario(level, gd, level_id=level_id, difficulty=want, strict=False)
    return stage_id, stage, want, scenario


def _timeline(scenario: Scenario) -> list[dict[str, Any]]:
    """Earliest absolute time of every wave action.

    Waves and fragments run in sequence, so the earliest a fragment can start is
    when the previous one finished *emitting*. The engine may start later —
    ``blockFragment`` holds a fragment until its enemies die and
    ``maxTimeWaitingForNextWave`` caps a wait — so these are lower bounds.
    """
    rows: list[dict[str, Any]] = []
    clock = 0.0
    for wi, wave in enumerate(scenario.waves):
        clock += wave.pre_delay
        for fi, frag in enumerate(wave.fragments):
            origin = clock + frag.pre_delay
            for action in frag.actions:
                first = origin + action.pre_delay
                rows.append(
                    {
                        "wave": wi,
                        "fragment": fi,
                        "kind": action.kind,
                        "key": action.key,
                        "count": action.count,
                        "first": round(first, 3),
                        "last": round(first + max(action.count - 1, 0) * action.interval, 3),
                        "interval": action.interval,
                        "route": action.route_index,
                        "spawns": action.spawns,
                        "dont_block_wave": action.dont_block_wave,
                        "block_fragment": action.block_fragment,
                        "hidden_group": action.hidden_group,
                        "random_spawn_group": action.random_spawn_group,
                        "weight": action.weight,
                    }
                )
            blocking = [a.busy_until for a in frag.actions if not a.dont_block_wave]
            clock = origin + max(blocking, default=0.0)
        clock += wave.post_delay
    # A fragment may declare its actions out of order; a timeline reads by time.
    rows.sort(key=lambda r: (r["first"], r["wave"], r["fragment"]))
    return rows


def _enemy_rows(scenario: Scenario, gd: GameData) -> list[dict[str, Any]]:
    handbook = (gd.enemy_handbook or {}).get("enemyData") or {}
    rows = []
    for (key, level), spec in sorted(scenario.enemy_specs.items()):
        hb = handbook.get(key) or {}
        s = spec.stats
        rows.append(
            {
                "key": key,
                "level": level,
                "name": spec.name or hb.get("name") or key,
                "rank": hb.get("enemyLevel"),
                "damage": ",".join(hb.get("damageType") or ()) or None,
                "max_hp": s.max_hp,
                "atk": s.atk,
                "defense": s.defense,
                "res": s.res,
                "block": s.block_cnt,
                "move_speed": s.move_speed,
                "attack_interval": round(s.attack_interval, 3),
                "range_radius": spec.range_radius,
                "motion": spec.motion,
                "apply_way": spec.apply_way,
                "life_point_reduce": spec.life_point_reduce,
                "tags": list(spec.tags),
                "immunities": sorted(s.immunities),
            }
        )
    return rows


def _tile_summary(scenario: Scenario) -> dict[str, Any]:
    bmap = scenario.bmap
    kinds: dict[tuple[str, str, str], list[int]] = {}
    melee = ranged = blocked = 0
    for t in bmap.all_tiles():
        info = bmap[t]
        entry = kinds.setdefault((info.key, info.height, info.buildable), [0, 0])
        entry[0] += 1
        free = bmap.deployable(t, "ALL")
        entry[1] += int(free)
        melee += int(bmap.deployable(t, "MELEE"))
        ranged += int(bmap.deployable(t, "RANGED"))
        blocked += int(info.buildable != "NONE" and not free)
    rows = [
        {"key": k, "height": h, "buildable": b, "count": n, "deployable": d}
        for (k, h, b), (n, d) in sorted(kinds.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ]
    return {
        "height": bmap.height,
        "width": bmap.width,
        "starts": [{"row": t.row, "col": t.col} for t in bmap.starts],
        "ends": [{"row": t.row, "col": t.col} for t in bmap.ends],
        "by_kind": rows,
        "melee_deployable": melee,
        "ranged_deployable": ranged,
        "blocked_by_tilesDisallowToLocate": blocked,
    }


def _novelty_rows(scenario: Scenario) -> list[dict[str, str]]:
    return [
        {"kind": str(e.kind), "key": e.key, "context": e.context} for e in scenario.novelty.events
    ]


def _stage_payload(
    gd: GameData, stage_id: str, stage: dict[str, Any], want: Difficulty, sc: Scenario
) -> dict[str, Any]:
    """One description of the scenario, feeding both the JSON and the text
    renderer, so the two can never disagree about what the stage contains."""
    return {
        "stage": {
            "stageId": stage_id,
            "code": stage.get("code"),
            "name": stage.get("name"),
            "apCost": stage.get("apCost"),
            "stageType": stage.get("stageType"),
            "difficulty": stage.get("difficulty"),
            "levelId": stage.get("levelId"),
        },
        "gamedata_version": gd.version,
        "gamedata_server": gd.server,
        "difficulty_applied": _difficulty_name(want),
        "map": {**_tile_summary(sc), "ascii": sc.bmap.ascii()},
        "waves": _timeline(sc),
        "wave_count": len(sc.waves),
        "enemy_count": sc.enemy_count,
        "enemies": _enemy_rows(sc, gd),
        "runes": [
            {
                "key": r.key,
                "difficulty_mask": _difficulty_name(r.difficulty_mask),
                "blackboard": r.blackboard,
                "value_str": r.value_str,
                "profession_mask": r.profession_mask,
                "buildable_mask": r.buildable_mask,
            }
            for r in sc.runes
        ],
        "options": {f.name: getattr(sc.options, f.name) for f in dataclasses.fields(sc.options)},
        "predefined": [
            {
                "char_key": u.char_key,
                "tile": {"row": u.tile.row, "col": u.tile.col},
                "direction": u.direction.name,
                "level": u.level,
                "phase": u.phase,
                "skill_index": u.skill_index,
                "hidden": u.hidden,
                "is_token": u.is_token,
                "alias": u.alias,
            }
            for u in sc.predefined
        ],
        "excluded_chars": sorted(sc.excluded_chars),
        "random_seed": sc.random_seed,
        "novelty": _novelty_rows(sc),
    }


def _render_map(tiles: dict[str, Any]) -> None:
    print(_rule(f"map {tiles['width']}x{tiles['height']}"))
    print(tiles["ascii"])
    print("  legend  . road   m melee-buildable   r ranged-buildable   # wall   _ floor")
    print("          S start  s fly-start         E end                o hole   (blank) forbidden")
    print("          ? = a tile kind with no ascii symbol; see the table below")
    starts = " ".join(f"({t['row']},{t['col']})" for t in tiles["starts"]) or "-"
    ends = " ".join(f"({t['row']},{t['col']})" for t in tiles["ends"]) or "-"
    print(f"  starts  {starts}\n  ends    {ends}")


def _render_tiles(tiles: dict[str, Any], character_limit: int) -> None:
    print(_rule("deployable tiles"))
    _emit(
        _table(
            ("tile kind", "height", "buildable", "tiles", "deployable"),
            [
                (r["key"], r["height"], r["buildable"], r["count"], r["deployable"])
                for r in tiles["by_kind"]
            ],
            align="lllrr",
        )
    )
    _emit(
        _kv(
            [
                ("melee positions", tiles["melee_deployable"]),
                ("ranged positions", tiles["ranged_deployable"]),
                ("blocked by tilesDisallowToLocate", tiles["blocked_by_tilesDisallowToLocate"]),
                ("character limit", character_limit),
            ]
        )
    )


def _render_waves(payload: dict[str, Any]) -> None:
    timeline = payload["waves"]
    spawns = sum(1 for r in timeline if r["spawns"])
    print(
        _rule(
            f"waves: {payload['wave_count']}, {spawns} spawn actions, "
            f"{payload['enemy_count']} enemies"
        )
    )
    names = {e["key"]: e["name"] for e in payload["enemies"]}
    rows = []
    for r in timeline:
        flags = []
        if r["hidden_group"]:
            flags.append(f"hidden={r['hidden_group']}")
        if r["random_spawn_group"]:
            flags.append(f"random={r['random_spawn_group']}(w{r['weight']})")
        if r["dont_block_wave"]:
            flags.append("dontBlockWave")
        if r["block_fragment"]:
            flags.append("blockFragment")
        rows.append(
            (
                f"{r['first']:.1f}",
                f"{r['last']:.1f}" if r["last"] != r["first"] else "",
                r["wave"],
                r["fragment"],
                r["kind"],
                r["key"] or "-",
                names.get(r["key"], ""),
                r["count"],
                f"{r['interval']:.1f}" if r["count"] > 1 else "",
                r["route"],
                " ".join(flags),
            )
        )
    _emit(
        _table(
            ("t", "..last", "w", "f", "action", "key", "name", "n", "every", "route", "flags"),
            rows,
            align="rrrrlllrrrl",
        )
    )
    print("  times are the earliest possible; blockFragment / wave waits can push them later")


def _render_enemies(enemies: list[dict[str, Any]]) -> None:
    print(_rule(f"enemies: {len(enemies)} types"))
    _emit(
        _table(
            (
                "key", "lv", "name", "rank", "dmg", "hp", "atk", "def", "res",
                "blk", "spd", "interval", "range", "motion", "apply", "lp", "tags",
            ),
            [
                (
                    e["key"], e["level"], e["name"], e["rank"], e["damage"], e["max_hp"],
                    e["atk"], e["defense"], e["res"], e["block"], e["move_speed"],
                    e["attack_interval"], e["range_radius"], e["motion"], e["apply_way"],
                    e["life_point_reduce"], ",".join(e["tags"] + e["immunities"]) or "",
                )
                for e in enemies
            ],
            align="lrlllrrrrrrrrllrl",
        )
    )


def _render_runes(runes: list[dict[str, Any]], difficulty: str) -> None:
    print(_rule(f"runes active for {difficulty}: {len(runes)}"))
    if not runes:
        print("  none")
        return
    _emit(
        _table(
            ("key", "mask", "blackboard", "professionMask", "buildableMask"),
            [
                (
                    r["key"],
                    r["difficulty_mask"],
                    " ".join(f"{k}={_num(v)}" for k, v in sorted(r["blackboard"].items()))
                    + "".join(f" {k}={v!r}" for k, v in sorted(r["value_str"].items())),
                    r["profession_mask"],
                    r["buildable_mask"],
                )
                for r in runes
            ],
            align="lllrl",
        )
    )


def _render_predefined(units: list[dict[str, Any]]) -> None:
    print(_rule(f"predefined units: {len(units)}"))
    _emit(
        _table(
            ("charKey", "tile", "dir", "phase", "level", "skill", "token", "hidden", "alias"),
            [
                (
                    u["char_key"],
                    f"({u['tile']['row']},{u['tile']['col']})",
                    u["direction"],
                    u["phase"],
                    u["level"],
                    u["skill_index"],
                    u["is_token"],
                    u["hidden"],
                    u["alias"],
                )
                for u in units
            ],
            align="lllrrrlll",
        )
    )


def _render_novelty(novelty: list[dict[str, str]]) -> None:
    if not novelty:
        print(_rule("NOVELTY: none"))
        print("  every mechanic in this level is registered")
        return
    print(_rule(f"NOVELTY: {len(novelty)} unmodelled mechanics"))
    _emit(_table(("kind", "key", "context"), [(n["kind"], n["key"], n["context"]) for n in novelty]))
    print("  strict=True would refuse this scenario; each line is a mechanic to implement")


def _render_stage(payload: dict[str, Any]) -> None:
    st = payload["stage"]
    _emit(
        _kv(
            [
                ("stage", f"{st['stageId']}  {st['code'] or ''}  {st['name'] or ''}"),
                ("levelId", st["levelId"]),
                ("type", f"{st['stageType']}  ap {st['apCost']}"),
                (
                    "difficulty",
                    f"{payload['difficulty_applied']}  (stage says {st['difficulty']})",
                ),
                ("gamedata", f"{payload['gamedata_version']} [{payload['gamedata_server']}]"),
                ("seed", payload["random_seed"]),
            ]
        )
    )
    print()
    _render_map(payload["map"])
    print()
    _render_tiles(payload["map"], payload["options"]["character_limit"])
    print()
    _render_waves(payload)
    print()
    _render_enemies(payload["enemies"])
    print()
    _render_runes(payload["runes"], payload["difficulty_applied"])
    print()
    print(_rule("battle options"))
    _emit(_kv(sorted(payload["options"].items())))
    if payload["excluded_chars"]:
        print("  excluded chars  " + ", ".join(payload["excluded_chars"]))
    if payload["predefined"]:
        print()
        _render_predefined(payload["predefined"])
    print()
    _render_novelty(payload["novelty"])


def cmd_stage_show(args: argparse.Namespace) -> int:
    gd = _open_gamedata(args.server)
    stage_id, stage, want, sc = _load_scenario(gd, args.stage, args.difficulty)
    payload = _stage_payload(gd, stage_id, stage, want, sc)
    if args.as_json:
        _dump_json(payload)
        return EXIT_OK
    _render_stage(payload)
    return EXIT_OK


def cmd_stage_routes(args: argparse.Namespace) -> int:
    gd = _open_gamedata(args.server)
    stage_id, stage, want, sc = _load_scenario(gd, args.stage, None)

    records = []
    for index in sorted(sc.routes):
        route = sc.routes[index]
        points = route_waypoints(sc.bmap, route)
        records.append(
            {
                "index": index,
                "motion": route.motion.name,
                "start": {"row": route.start.row, "col": route.start.col},
                "end": {"row": route.end.row, "col": route.end.col},
                "allow_diagonal": route.allow_diagonal,
                "spawn_offset": [route.spawn_offset.x, route.spawn_offset.y],
                "spawn_random_range": [route.spawn_random_range.x, route.spawn_random_range.y],
                "visit_every_tile_center": route.visit_every_tile_center,
                "checkpoints": [
                    {
                        "kind": c.kind,
                        "position": {"row": c.position.row, "col": c.position.col},
                        "time": c.time,
                        "reach_offset": [c.reach_offset.x, c.reach_offset.y],
                        "reach_distance": c.reach_distance,
                    }
                    for c in route.checkpoints
                ],
                "waypoints": [[p.x, p.y] for p in points],
            }
        )

    if args.as_json:
        _dump_json({"stageId": stage_id, "levelId": stage.get("levelId"), "routes": records})
        return EXIT_OK

    print(f"{stage_id}  {stage.get('code') or ''}  {stage.get('name') or ''}"
          f"   {_difficulty_name(want)}   routes: {len(records)}")
    for r in records:
        start, end = r["start"], r["end"]
        print()
        print(
            f"  #{r['index']:<4} {r['motion']:<5} "
            f"({start['row']},{start['col']}) -> ({end['row']},{end['col']})   "
            f"diagonal={'yes' if r['allow_diagonal'] else 'no'}   "
            f"spawnOffset=({_num(r['spawn_offset'][0])},{_num(r['spawn_offset'][1])})"
        )
        if r["checkpoints"]:
            marks = [
                f"{c['kind']}"
                + (
                    f"({c['position']['row']},{c['position']['col']})"
                    if c["kind"] not in ("WAIT_FOR_SECONDS",)
                    else ""
                )
                + (f"[{_num(c['time'])}s]" if c["time"] else "")
                for c in r["checkpoints"]
            ]
            print("    checkpoints:")
            _emit(_wrap_items(marks, indent="      "))
        else:
            print("    checkpoints: none")
        pts = [f"({_num(x)},{_num(y)})" for x, y in r["waypoints"]]
        print(f"    waypoints ({len(pts)}), as (x=col,y=row):")
        _emit(_wrap_items(pts, indent="      "))
    return EXIT_OK


# ---------------------------------------------------------------------------
# ato op
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"</?[^<>]{0,64}>")
_PLACEHOLDER_RE = re.compile(r"\{(-?)([^:{}]+)(?::([^{}]*))?\}")


def _describe(text: str, blackboard: dict[str, float]) -> str:
    """Render a skill/talent description: strip rich-text tags, fill in values.

    Descriptions carry ``{key}`` / ``{key:0%}`` placeholders resolved against the
    skill's own blackboard; an unresolvable one is left verbatim, not guessed.
    """
    lower = {k.lower(): v for k, v in blackboard.items()}

    def sub(m: re.Match[str]) -> str:
        key, fmt = m.group(2).strip(), m.group(3) or ""
        value = lower.get(key.lower())
        if value is None:
            return m.group(0)
        if m.group(1):
            value = -value
        if fmt.endswith("%"):
            digits = len(fmt.split(".")[-1].rstrip("%")) if "." in fmt else 0
            return f"{value * 100:.{digits}f}%"
        digits = len(fmt.split(".")[-1]) if "." in fmt else 0
        return f"{value:.{digits}f}" if digits else _num(value)

    plain = _TAG_RE.sub("", text.replace("\\n", "\n"))
    return _PLACEHOLDER_RE.sub(sub, plain).strip()


def _talent_candidate(
    talent: dict[str, Any], phase: int, level: int, potential: int
) -> dict[str, Any] | None:
    """The candidate a given investment unlocks; candidates ascend by requirement."""
    best: dict[str, Any] | None = None
    for cand in talent.get("candidates") or ():
        cond = cand.get("unlockCondition") or {}
        need = (
            PHASE_INDEX.get(str(cond.get("phase") or "PHASE_0"), 0),
            int(cond.get("level") or 1),
        )
        if need <= (phase, level) and int(cand.get("requiredPotentialRank") or 0) <= potential:
            best = cand
    return best


def _range_ascii(grids: Sequence[tuple[int, int]]) -> list[str]:
    """Render an attack range facing right; ``@`` is the operator's own tile."""
    cells = set(grids) | {(0, 0)}
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    out = []
    for row in range(max(rows), min(rows) - 1, -1):
        line = []
        for col in range(min(cols), max(cols) + 1):
            covered = "#" if (row, col) in cells else "."
            line.append("@" if (row, col) == (0, 0) else covered)
        out.append("  " + " ".join(line))
    return out


def _operator_payload(
    gd: GameData, spec: OperatorSpec, mastery: int, potential: int
) -> dict[str, Any]:
    char = spec.raw
    stats = {f.name: getattr(spec.stats, f.name) for f in dataclasses.fields(spec.stats)}
    stats["immunities"] = sorted(spec.stats.immunities)
    stats["attack_interval"] = round(spec.stats.attack_interval, 4)

    range_entry = gd.ranges.get(spec.range_id)
    grids: tuple[tuple[int, int], ...] = ()
    if range_entry is not None:
        grids = resolve_range(spec.range_id, range_entry).grids

    skills = []
    for i, skill_id in enumerate(spec.skill_ids):
        entry = gd.skills.get(skill_id)
        if entry is None:
            skills.append({"index": i + 1, "skill_id": skill_id, "missing": True})
            continue
        sk = resolve_skill(skill_id, entry, mastery)
        skills.append(
            {
                "index": i + 1,
                "skill_id": skill_id,
                "name": sk.name,
                "level": sk.level,
                "skill_type": sk.skill_type,
                "sp_type": sk.sp_type,
                "sp_cost": sk.sp_cost,
                "init_sp": sk.init_sp,
                "increment": sk.increment,
                "duration": sk.duration,
                "duration_type": sk.duration_type,
                "range_id": sk.range_id,
                "blackboard": sk.blackboard,
                "description": _describe(sk.description, sk.blackboard),
            }
        )

    talents = []
    for talent in spec.talents:
        cand = _talent_candidate(talent, spec.phase, spec.level, potential)
        if cand is None:
            continue
        cond = cand.get("unlockCondition") or {}
        talents.append(
            {
                "name": cand.get("name") or "",
                "description": _describe(
                    cand.get("description") or "",
                    {
                        b["key"]: float(b.get("value") or 0.0)
                        for b in (cand.get("blackboard") or [])
                        if b.get("key")
                    },
                ),
                "phase": PHASE_INDEX.get(str(cond.get("phase") or "PHASE_0"), 0),
                "level": int(cond.get("level") or 1),
                "potential": int(cand.get("requiredPotentialRank") or 0),
            }
        )

    return {
        "char_id": spec.char_id,
        "name": spec.name,
        "appellation": char.get("appellation"),
        "rarity": _rarity(char),
        "profession": spec.profession,
        "sub_profession": spec.sub_profession,
        "position": spec.position,
        "tags": list(spec.tags),
        "phase": spec.phase,
        "level": spec.level,
        "max_level": char["phases"][spec.phase].get("maxLevel"),
        "stats": stats,
        "range": {"range_id": spec.range_id, "grids": [list(g) for g in grids]},
        "talents": talents,
        "skills": skills,
    }


def cmd_op_show(args: argparse.Namespace) -> int:
    gd = _open_gamedata(args.server)
    char_id, char = _find_operator(gd, args.query)
    spec = resolve_operator(
        char_id,
        char,
        phase=args.phase,
        level=args.level,
        trust=args.trust,
        potential=args.potential,
    )
    payload = _operator_payload(gd, spec, args.mastery, args.potential)
    payload["trust"] = args.trust
    payload["potential"] = args.potential

    if args.as_json:
        _dump_json(payload)
        return EXIT_OK

    s = payload["stats"]
    _emit(
        _kv(
            [
                ("operator", f"{payload['char_id']}  {payload['name']} "
                             f"({payload['appellation'] or '-'})"),
                ("rarity", "*" * payload["rarity"] + f"  ({payload['rarity']})"),
                ("class", f"{payload['profession']} / {payload['sub_profession']}"),
                ("deploys on", payload["position"]),
                ("tags", ", ".join(payload["tags"]) or "-"),
                (
                    "investment",
                    f"E{payload['phase']} lv{payload['level']}/{payload['max_level']}  "
                    f"trust {payload['trust']}  potential {payload['potential']}",
                ),
            ]
        )
    )

    print()
    print(_rule("stats"))
    _emit(
        _table(
            ("hp", "atk", "def", "res", "cost", "block", "interval", "aspd", "respawn"),
            [
                (
                    s["max_hp"], s["atk"], s["defense"], s["res"], s["cost"], s["block_cnt"],
                    s["attack_interval"], s["attack_speed"], s["respawn_time"],
                )
            ],
            align="rrrrrrrrr",
        )
    )
    _emit(
        _kv(
            [
                ("hp regen /s", s["hp_recovery_per_sec"]),
                ("sp regen /s", s["sp_recovery_per_sec"]),
                ("base attack time", s["base_attack_time"]),
                ("max deploy count", s["max_deploy_count"]),
                ("taunt / mass level", f"{s['taunt_level']} / {s['mass_level']}"),
                ("immunities", ", ".join(s["immunities"]) or "-"),
            ]
        )
    )

    print()
    print(_rule(f"range {payload['range']['range_id']}"))
    grids = [(g[0], g[1]) for g in payload["range"]["grids"]]
    if grids:
        _emit(_range_ascii(grids))
        print("  facing right (+col); @ = the operator's tile, # = covered")
    else:
        print("  range id not present in range_table")

    if payload["talents"]:
        print()
        print(_rule("talents"))
        for t in payload["talents"]:
            print(f"  {t['name'] or '-'}  [E{t['phase']} lv{t['level']} pot{t['potential']}]")
            for para in t["description"].splitlines():
                _emit(_wrap_items(para.split(), indent="      "))

    print()
    print(_rule(f"skills (mastery {args.mastery})"))
    for sk in payload["skills"]:
        if sk.get("missing"):
            print(f"  {sk['index']}. {sk['skill_id']}  (absent from skill_table)")
            continue
        print(
            f"  {sk['index']}. {sk['name']}  [{sk['skill_id']}  lv index {sk['level']}]"
        )
        _emit(
            _kv(
                [
                    ("trigger", f"{sk['skill_type']} / sp by "
                                f"{_SP_TYPE_SHORT.get(sk['sp_type'], sk['sp_type'])}"),
                    ("sp", f"cost {_num(sk['sp_cost'])}  initial {_num(sk['init_sp'])}  "
                           f"+{_num(sk['increment'])}/tick"),
                    ("duration", f"{_num(sk['duration'])}s  ({sk['duration_type']})"),
                    ("range", sk["range_id"] or "inherits the operator's range"),
                ],
                indent="     ",
            )
        )
        for para in sk["description"].splitlines():
            _emit(_wrap_items(para.split(), indent="     "))
        print()
    return EXIT_OK


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--server", choices=("cn", "en"), default="cn", help="which server's data (default: cn)"
    )
    jsonable = argparse.ArgumentParser(add_help=False)
    jsonable.add_argument(
        "--json", dest="as_json", action="store_true", help="emit JSON instead of a table"
    )

    parser = argparse.ArgumentParser(
        prog="ato",
        description="ATO — inspect the game-data snapshots, stages and operators "
        "that the offline simulator is built from.",
    )
    parser.add_argument("--version", action="version", version=f"ato {__version__}")
    groups = parser.add_subparsers(dest="group", metavar="GROUP")

    # -- gamedata ---------------------------------------------------------
    gamedata = groups.add_parser("gamedata", help="snapshots of the public game data")
    gamedata.set_defaults(usage_parser=gamedata)
    gd_sub = gamedata.add_subparsers(dest="command", metavar="COMMAND")

    p = gd_sub.add_parser(
        "fetch", parents=[common], help="download a version-stamped snapshot (needs network)"
    )
    p.add_argument("--force", action="store_true", help="re-download even if the version exists")
    p.set_defaults(func=cmd_gamedata_fetch)

    p = gd_sub.add_parser("info", parents=[jsonable], help="list local snapshots")
    p.set_defaults(func=cmd_gamedata_info)

    p = gd_sub.add_parser(
        "diff", parents=[jsonable], help="compare two snapshots (ato.gamedata.diff)"
    )
    p.add_argument("old", help="path, directory name (cn-77.0.0) or bare version")
    p.add_argument("new", help="path, directory name or bare version")
    p.set_defaults(func=cmd_gamedata_diff)

    # -- stage ------------------------------------------------------------
    stage = groups.add_parser("stage", help="stages and their compiled scenarios")
    stage.set_defaults(usage_parser=stage)
    st_sub = stage.add_subparsers(dest="command", metavar="COMMAND")

    p = st_sub.add_parser(
        "find", parents=[common, jsonable], help="search stages by code, stage id or name"
    )
    p.add_argument("query", help="substring of a stage code, stage id or name")
    p.add_argument("--limit", type=int, default=50, help="max rows, 0 for all (default: 50)")
    p.set_defaults(func=cmd_stage_find)

    difficulty = ("NORMAL", "FOUR_STAR", "EASY", "SIX_STAR")
    p = st_sub.add_parser(
        "show",
        parents=[common, jsonable],
        help="compile a stage and print map, waves, enemies, runes and novelty",
    )
    p.add_argument("stage", help="stage id (main_01-07) or code (1-7)")
    p.add_argument(
        "--difficulty",
        choices=difficulty,
        default=None,
        help="which rune set applies (default: the stage's own difficulty)",
    )
    p.set_defaults(func=cmd_stage_show)

    p = st_sub.add_parser(
        "routes", parents=[common, jsonable], help="print every route and its waypoint polyline"
    )
    p.add_argument("stage", help="stage id or code")
    p.set_defaults(func=cmd_stage_routes)

    # -- op ---------------------------------------------------------------
    op = groups.add_parser("op", help="operators")
    op.set_defaults(usage_parser=op)
    op_sub = op.add_subparsers(dest="command", metavar="COMMAND")

    p = op_sub.add_parser(
        "show", parents=[common, jsonable], help="resolve an operator's stats, range and skills"
    )
    p.add_argument("query", help="char id, name or appellation")
    p.add_argument("--phase", type=int, default=2, help="elite phase, clamped (default: 2)")
    p.add_argument("--level", type=int, default=None, help="level (default: max for the phase)")
    p.add_argument("--trust", type=int, default=200, help="trust 0-200 (default: 200)")
    p.add_argument("--potential", type=int, default=0, help="potential ranks applied (default: 0)")
    p.add_argument("--mastery", type=int, default=6, help="skill level index, clamped (default: 6)")
    p.set_defaults(func=cmd_op_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    func = getattr(args, "func", None)
    if func is None:
        # A bare group (`ato stage`) is a usage error: show that group's help.
        usage: argparse.ArgumentParser = getattr(args, "usage_parser", parser)
        usage.print_help(sys.stderr)
        return EXIT_ERROR
    try:
        return int(func(args))
    except CliError as exc:
        print(f"ato: {exc}", file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"ato: {exc}\n     the snapshot looks incomplete; run: ato gamedata fetch --force",
              file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("ato: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
