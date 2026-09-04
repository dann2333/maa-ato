"""Loading, saving and surveying copilot files.

The published corpus is tens of thousands of small files of wildly varying
quality, so everything here is a generator over one file at a time and a parse
failure is data (counted, sampled, attributed) rather than an exception that
kills the scan.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ato.copilot.schema import ActionType, CopilotDoc, parse_doc
from ato.sim.registry import NoveltyLog


class ParseFailure(ValueError):
    """A file that is not a usable copilot document."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


def unwrap_envelope(obj: Any) -> Any:
    """Unwrap the response envelope used by the copilot server.

    ``https://prts.maa.plus/copilot/get/<id>`` returns
    ``{"status_code": 200, "data": {..., "content": "<the document, as a JSON
    string>"}}``, and mirrors of the corpus are usually saved in that shape.
    A bare document passes through untouched.
    """
    if isinstance(obj, Mapping) and "data" in obj and "stage_name" not in obj:
        data = obj["data"]
        if isinstance(data, Mapping) and isinstance(data.get("content"), str):
            return json.loads(data["content"])
        if isinstance(data, Mapping):
            return data
    return obj


def loads(text: str, *, source: str = "<string>", novelty: NoveltyLog | None = None) -> CopilotDoc:
    """Parse one copilot document from JSON text."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseFailure(source, f"invalid JSON: {exc}") from exc
    try:
        return parse_doc(unwrap_envelope(obj), source=source, novelty=novelty)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        raise ParseFailure(source, f"{type(exc).__name__}: {exc}") from exc


def dumps(doc: CopilotDoc, *, indent: int | None = 2) -> str:
    """Serialise a document. ``ensure_ascii`` is off: these files are Chinese."""
    return json.dumps(doc.to_json_obj(), ensure_ascii=False, indent=indent)


def load_doc(path: str | Path, *, novelty: NoveltyLog | None = None) -> CopilotDoc:
    p = Path(path)
    try:
        # utf-8-sig: a noticeable slice of the corpus is BOM-prefixed
        text = p.read_text("utf-8-sig")
    except OSError as exc:
        raise ParseFailure(str(p), f"unreadable: {exc}") from exc
    return loads(text, source=str(p), novelty=novelty)


def save_doc(doc: CopilotDoc, path: str | Path, *, indent: int | None = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(doc, indent=indent), "utf-8")
    return p


def validate(doc: CopilotDoc) -> list[str]:
    """Return the reasons MAA would struggle with this document. Empty is good.

    This is what MAA's parser and the editor's schema jointly require, not a
    guess at what plays well: a stage to run on, at least one action, a target
    for every action that needs one, and a position for every deployment.
    """
    problems: list[str] = []
    if not doc.stage_name:
        problems.append("stage_name is empty")
    if not doc.actions:
        problems.append("no actions")
    if doc.is_sss:
        problems.append(f"document type {doc.doc_type!r} is a different protocol")
    known = doc.group_names | {o.name for o in doc.opers if o.name}
    for i, action in enumerate(doc.actions):
        if action.type is ActionType.UNKNOWN:
            problems.append(f"action[{i}]: unknown type {action.raw_type!r}")
            continue
        if action.type is ActionType.DEPLOY:
            if not action.name:
                problems.append(f"action[{i}]: Deploy without a name")
            elif known and action.name not in known:
                # not fatal for MAA (it can pick an operator off the field), but
                # it means auto-formation has nothing to bring
                problems.append(f"action[{i}]: deploys {action.name!r}, absent from opers/groups")
            if action.location is None:
                problems.append(f"action[{i}]: Deploy without a location")
            elif min(action.location) < 0:
                problems.append(f"action[{i}]: negative location {list(action.location)}")
        if action.type in (ActionType.SKILL, ActionType.RETREAT) and not (
            action.name or action.location
        ):
            problems.append(f"action[{i}]: {action.type} needs a name or a location")
        if action.type is ActionType.MOVE_CAMERA and action.distance is None:
            problems.append(f"action[{i}]: MoveCamera without a distance")
        if action.pre_delay < 0 or action.post_delay < 0:
            problems.append(f"action[{i}]: negative delay")
    return problems


def iter_documents(
    paths: Iterable[str | Path],
    *,
    novelty: NoveltyLog | None = None,
    on_error: Callable[[ParseFailure], None] | None = None,
) -> Iterator[tuple[Path, CopilotDoc]]:
    """Parse each path in turn, yielding only what parsed.

    ``on_error(ParseFailure)`` is called for the rest; the default swallows
    them, because a scan of somebody else's corpus should not stop at the first
    truncated download.
    """
    for path in paths:
        p = Path(path)
        try:
            yield p, load_doc(p, novelty=novelty)
        except ParseFailure as exc:
            if on_error is not None:
                on_error(exc)


def iter_corpus(
    root: str | Path,
    *,
    pattern: str = "**/*.json",
    sort: bool = True,
    novelty: NoveltyLog | None = None,
    on_error: Callable[[ParseFailure], None] | None = None,
) -> Iterator[tuple[Path, CopilotDoc]]:
    """Walk a directory of copilot files lazily.

    Only one document is ever held in memory. ``sort=True`` costs a list of
    paths (cheap next to the files themselves) and buys a reproducible order;
    ``sort=False`` streams straight off ``Path.glob``.
    """
    found = Path(root).glob(pattern)
    return iter_documents(sorted(found) if sort else found, novelty=novelty, on_error=on_error)


@dataclass
class CopilotStats:
    """A survey of a corpus. Counters only — never the documents themselves."""

    files: int = 0
    parsed: int = 0
    failed: int = 0
    invalid: int = 0
    actions: int = 0
    action_types: Counter[str] = field(default_factory=Counter)
    unknown_action_types: Counter[str] = field(default_factory=Counter)
    operators: Counter[str] = field(default_factory=Counter)
    stages: Counter[str] = field(default_factory=Counter)
    minimum_required: Counter[str] = field(default_factory=Counter)
    #: Bounded samples, so a corpus of 50k broken files cannot exhaust memory.
    failures: list[tuple[str, str]] = field(default_factory=list)
    problems: list[tuple[str, str]] = field(default_factory=list)
    max_samples: int = 50
    novelty: NoveltyLog = field(default_factory=lambda: NoveltyLog(strict=False))

    def add(self, source: str, doc: CopilotDoc) -> None:
        self.files += 1
        self.parsed += 1
        self.stages[doc.stage_name] += 1
        self.minimum_required[doc.minimum_required or "(absent)"] += 1
        for name in doc.operator_names:
            self.operators[name] += 1
        for action in doc.actions:
            self.actions += 1
            self.action_types[str(action.type)] += 1
            if action.type is ActionType.UNKNOWN:
                self.unknown_action_types[action.raw_type] += 1
        problems = validate(doc)
        if problems:
            self.invalid += 1
            if len(self.problems) < self.max_samples:
                self.problems.append((source, problems[0]))

    def add_failure(self, failure: ParseFailure) -> None:
        self.files += 1
        self.failed += 1
        if len(self.failures) < self.max_samples:
            self.failures.append((failure.source, failure.reason))

    def report(self, top: int = 10) -> str:
        lines = [
            f"{self.files} file(s): {self.parsed} parsed, {self.failed} failed, "
            f"{self.invalid} parsed-but-invalid",
            f"{self.actions} action(s) across {len(self.stages)} distinct stage name(s), "
            f"{len(self.operators)} distinct operator name(s)",
            "  by type: "
            + ", ".join(f"{k}={v}" for k, v in self.action_types.most_common()),
        ]
        if self.unknown_action_types:
            lines.append(
                "  unknown types: "
                + ", ".join(f"{k!r}={v}" for k, v in self.unknown_action_types.most_common(top))
            )
        if self.operators:
            lines.append(
                "  top operators: "
                + ", ".join(f"{k}={v}" for k, v in self.operators.most_common(top))
            )
        for source, reason in self.failures[:top]:
            lines.append(f"  FAILED {source}: {reason}")
        for source, reason in self.problems[:top]:
            lines.append(f"  INVALID {source}: {reason}")
        if self.novelty.events:
            lines.append(f"  novelty: {len(self.novelty.events)} distinct event(s)")
            for ev in self.novelty.events[:top]:
                lines.append(f"    {ev}")
        return "\n".join(lines)


def scan_corpus(
    root: str | Path,
    *,
    pattern: str = "**/*.json",
    limit: int | None = None,
    max_samples: int = 50,
) -> CopilotStats:
    """Survey a corpus without ever holding more than one document in memory."""
    stats = CopilotStats(max_samples=max_samples)
    docs = iter_corpus(root, pattern=pattern, novelty=stats.novelty, on_error=stats.add_failure)
    for n, (path, doc) in enumerate(docs, 1):
        stats.add(str(path), doc)
        if limit is not None and n >= limit:
            break
    return stats
