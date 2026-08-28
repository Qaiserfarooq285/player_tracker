"""ADR-19 -- manual-annotation sidecar parser (CLAUDE.md §14.3).

Pure, no I/O beyond reading the sidecar text file itself. Tolerant of the client's own natural
phrasing (comments, a bare source-URL line, an optional "Min " time prefix); never crashes on a
malformed line and never silently drops one either -- anything that looks like content but doesn't
match the grammar is returned in `problems` (CLAUDE.md §10 "log everything dropped. Silent
filtering = hidden bugs.").

Grammar (one event per line): ``[<time_prefix>]<TIME> player #<N> in <COLOUR> <ACTION PHRASE>``
- ``TIME`` = ``MM:SS`` or ``H:MM:SS``/``HH:MM:SS`` (`configs/annotations.yaml: time_pattern`),
  converted to a float seconds-from-video-start offset.
- ``#<N>`` = the target's jersey number (integer), exactly as the client wrote it.
- ``COLOUR`` = a single free word -- this parser records it verbatim; a LATER stage (Stage 4's
  manual-mode wiring) is the one that matches it against an ADR-12 team-colour cluster (Golden
  Rule 5: this module only records what was written, never resolves it against tracking data).
- ``ACTION PHRASE`` = free text, mapped to an `EventType` via `configs/annotations.yaml:
  phrase_map` (first matching entry wins, case-insensitive substring match) or
  `unmapped_event_type` when no entry matches.

No new dependency (ADR-19): stdlib `re` only.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from src.common.types import Annotation, Event, EventType

_URL_PREFIXES = ("http://", "https://")


def _strip_time_prefix(line: str, time_prefixes: list[str]) -> str:
    """Strip the FIRST matching literal prefix (e.g. ``"Min "``) off `line`, if present -- the
    grammar treats the prefix as optional, so a line without it is passed through unchanged."""
    for prefix in time_prefixes:
        if line.startswith(prefix):
            return line[len(prefix) :]
    return line


def _parse_time(time_str: str) -> float:
    """``MM:SS`` or ``H:MM:SS``/``HH:MM:SS`` -> float seconds. The caller has already matched this
    string against `configs/annotations.yaml: time_pattern`, so it is always well-formed here."""
    parts = [int(p) for p in time_str.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = parts
    return float(hours * 3600 + minutes * 60 + seconds)


def _line_regex(cfg: dict) -> re.Pattern[str]:
    """Build the whole-line grammar regex around `configs/annotations.yaml: time_pattern` -- the
    time sub-pattern is config-owned (CLAUDE.md §10: no magic numbers/patterns in code) while the
    surrounding ``player #<N> in <colour> <phrase>`` structure is the parser's own fixed grammar
    (CLAUDE.md §14.3), not something a client would ever need to retune."""
    time_pattern = cfg["time_pattern"]
    return re.compile(
        rf"^(?P<time>{time_pattern})\s+player\s+#(?P<jersey>\d+)\s+in\s+(?P<colour>\S+)\s+"
        rf"(?P<phrase>.+)$",
        re.IGNORECASE,
    )


def _event_type_for_phrase(phrase: str, cfg: dict) -> EventType:
    """First `phrase_map` entry (checked top-to-bottom, per the config's own documented ordering
    rule) whose any `patterns` substring appears (case-insensitive) in `phrase`; `EventType`
    values in the YAML are written UPPERCASE for readability, matching `EventType`'s member NAMES,
    not its (lowercase) string VALUES -- converted here, once, rather than duplicating the mapping.
    """
    lowered = phrase.lower()
    for entry in cfg["phrase_map"]:
        if any(pattern in lowered for pattern in entry["patterns"]):
            return EventType[entry["event_type"]]
    return EventType[cfg["unmapped_event_type"]]


def parse_annotations(path: str | Path, cfg: dict) -> tuple[list[Annotation], list[str]]:
    """Parse one sidecar file into ``(annotations, problems)``.

    `annotations` is every line that matched the grammar, in file order. `problems` is every
    non-blank, non-comment, non-URL line that did NOT match the grammar, prefixed with its 1-based
    line number for traceability (Golden Rule 5) -- never raised as an exception, never silently
    dropped (CLAUDE.md §10). A comment (``#...``) or a bare source-URL line is deliberately
    excluded from `problems` -- the client's own example sidecar (§14.3) includes exactly such a
    line and it is not "malformed", it is intentionally-ignored context.
    """
    text = Path(path).read_text(encoding="utf-8")
    pattern = _line_regex(cfg)
    time_prefixes = cfg["time_prefixes"]

    annotations: list[Annotation] = []
    problems: list[str] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(_URL_PREFIXES):
            continue

        candidate = _strip_time_prefix(line, time_prefixes)
        match = pattern.match(candidate)
        if match is None:
            problems.append(f"line {lineno}: {raw_line}")
            continue

        phrase = match.group("phrase").strip()
        annotations.append(
            Annotation(
                t=_parse_time(match.group("time")),
                jersey_number=int(match.group("jersey")),
                team_colour=match.group("colour").lower(),
                action_phrase=phrase,
                event_type=_event_type_for_phrase(phrase, cfg),
                raw_line=raw_line,
            )
        )

    return annotations, problems


def annotation_to_event(annotation: Annotation, cfg: dict, take_id: int | None) -> Event:
    """One `Annotation` -> one `Event(source="manual_annotation")` (ADR-19).

    ``t_end = t_start + event_window_seconds`` (`configs/annotations.yaml`) since an annotation
    records a single instant the client watched, not a measured `[t_start, t_end]` interval the
    way an auto-detector heuristic produces one (see that config key's own comment).
    `player_track_id` is left `None` here -- Stage 4's manual-mode wiring is responsible for
    associating this event to a real track by colour/timing where possible; this function only
    ever converts what the client actually wrote, never invents a track association.
    """
    window = cfg["event_window_seconds"]
    return Event(
        id=str(uuid.uuid4()),
        type=annotation.event_type,
        t_start=annotation.t,
        t_end=annotation.t + window,
        player_track_id=None,
        take_id=take_id,
        confidence=cfg["default_confidence"],
        source="manual_annotation",
        evidence={
            "jersey_number": annotation.jersey_number,
            "team_colour": annotation.team_colour,
            "action_phrase": annotation.action_phrase,
            "raw_line": annotation.raw_line,
        },
    )
