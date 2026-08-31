"""ADR-19 -- manual-annotation sidecar parser (CLAUDE.md §14.3).

Pure, no I/O beyond reading the sidecar text file itself. Tolerant of the client's own natural
phrasing (comments, a bare source-URL line, an optional "Min " time prefix); never crashes on a
malformed line and never silently drops one either -- anything that looks like content but doesn't
match the grammar is returned in `problems` (CLAUDE.md §10 "log everything dropped. Silent
filtering = hidden bugs.").

Grammar (one event per line): ``[<time_prefix>]<TIME>[<separator>][player ][<colour clause>]
#<N>[<colour clause>] <ACTION PHRASE>`` where ``<colour clause>`` is ``in <COLOUR>`` and may
appear either immediately before or immediately after ``#<N>`` (or not at all).
- ``TIME`` = ``MM:SS`` or ``H:MM:SS``/``HH:MM:SS`` (`configs/annotations.yaml: time_pattern`),
  converted to a float seconds-from-video-start offset.
- an optional literal separator between the time and the rest of the content
  (`configs/annotations.yaml: line_separators`, e.g. an em-dash) beyond a plain space.
- the literal word "player" (any case) is optional -- structural grammar, not client-tunable data,
  same treatment as the literal "#"/"in" connectors below.
- ``#<N>`` = the target's jersey number (integer), exactly as the client wrote it. REQUIRED --
  a line with no jersey token is not parseable and lands in `problems`.
- ``COLOUR`` = one entry from the known-colour vocabulary (`configs/annotations.yaml:
  colour_reference_lab` keys -- reused as-is rather than duplicated in a second list, CLAUDE.md
  §10), one OR two words (e.g. "white", "light blue"), matched longest-first so "light blue" is
  never shadowed by a bare "blue". Optional -- a line naming no colour still parses ("Player #11
  scores goal"). This parser only records the word verbatim; a LATER stage (Stage 4's manual-mode
  wiring) is the one that matches it against an ADR-12 team-colour cluster (Golden Rule 5: this
  module only records what was written, never resolves it against tracking data).
- ``ACTION PHRASE`` = whatever free text remains after the time/separator/player-word/jersey/
  colour tokens above are removed, mapped to an `EventType` via `configs/annotations.yaml:
  phrase_map` (first matching entry wins, case-insensitive substring match) or
  `unmapped_event_type` when no entry matches.

Token-extraction approach (CLAUDE.md §14.3, extended 2026-08-31 for real client phrasing that
doesn't fit one fixed-order regex -- see `input/video1/instruction`: an em-dash separator, a
colour clause that can appear either before OR after the jersey number within the SAME file, an
optional "Player" word, and multi-word colours): find and strip the time, then find and strip the
jersey token and the colour clause independently of each other's position, then whatever remains
is the action phrase. This is deliberately NOT a single anchored regex -- a fixed field order
cannot express "colour before OR after jersey number, both occurring in the same sidecar".

No new dependency (ADR-19): stdlib `re` only.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from src.common.types import Annotation, Event, EventType

_URL_PREFIXES = ("http://", "https://")

_PLAYER_WORD_RE = re.compile(r"^\s*player\b\s*", re.IGNORECASE)
_JERSEY_RE = re.compile(r"#\s*(\d+)")


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


def _time_regex(cfg: dict) -> re.Pattern[str]:
    """Anchor just the TIME token at the head of the (prefix-stripped) line -- everything after it
    is handled by the token-extraction helpers below, since jersey/colour ordering varies (see
    module docstring)."""
    return re.compile(rf"^(?P<time>{cfg['time_pattern']})(?P<rest>.*)$", re.IGNORECASE | re.DOTALL)


def _strip_separator(rest: str, separators: list[str]) -> str:
    """Strip one optional literal separator (`configs/annotations.yaml: line_separators`, e.g. an
    em-dash) between the TIME token and the rest of the content, beyond the plain space that is
    always implicitly accepted. Checked longest-first so a longer separator can never be shadowed
    by a shorter one that happens to be its own prefix."""
    stripped = rest.lstrip()
    for sep in sorted(separators, key=len, reverse=True):
        if stripped.startswith(sep):
            return stripped[len(sep) :].lstrip()
    return stripped


def _colour_pattern(colour_words: list[str]) -> re.Pattern[str] | None:
    """One alternation over the known colour vocabulary, longest phrase first (so "light blue"
    wins over a bare "blue"), each optionally preceded by the connector "in " -- both the "in" and
    the colour phrase itself are held to whole-word boundaries (`\\b`) so e.g. "win" never matches
    the connector and "blues" never matches the colour "blue". `None` when the vocabulary is
    empty (defensive; `colour_reference_lab` is never actually empty in practice)."""
    if not colour_words:
        return None
    alternatives = sorted((re.escape(c) for c in colour_words), key=len, reverse=True)
    return re.compile(r"\b(?:in\s+)?(" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def _extract_colour(text: str, colour_words: list[str]) -> tuple[str | None, str]:
    """Find a known colour word/phrase anywhere in `text` -- independent of whether it sits before
    or after the jersey token (CLAUDE.md §14.3's grammar note) -- and return
    ``(colour_lowercased, text_with_the_clause_removed)``. ``(None, text)`` unchanged when no known
    colour word appears."""
    pattern = _colour_pattern(colour_words)
    if pattern is None:
        return None, text
    match = pattern.search(text)
    if match is None:
        return None, text
    colour = match.group(1).lower()
    remaining = text[: match.start()] + text[match.end() :]
    return colour, remaining


def _extract_jersey(text: str) -> tuple[int | None, str]:
    """Find ``#<N>`` anywhere in `text` and return ``(jersey_number, text_with_token_removed)``.
    ``(None, text)`` unchanged when no jersey token appears at all -- the caller treats that as a
    malformed line (a jersey number is the one truly required piece of this grammar)."""
    match = _JERSEY_RE.search(text)
    if match is None:
        return None, text
    remaining = text[: match.start()] + text[match.end() :]
    return int(match.group(1)), remaining


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
    time_pattern = _time_regex(cfg)
    time_prefixes = cfg["time_prefixes"]
    separators = cfg["line_separators"]
    colour_words = list(cfg["colour_reference_lab"].keys())

    annotations: list[Annotation] = []
    problems: list[str] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(_URL_PREFIXES):
            continue

        candidate = _strip_time_prefix(line, time_prefixes)
        time_match = time_pattern.match(candidate)
        if time_match is None:
            problems.append(f"line {lineno}: {raw_line}")
            continue

        content = _strip_separator(time_match.group("rest"), separators)
        content = _PLAYER_WORD_RE.sub("", content, count=1)
        colour, content = _extract_colour(content, colour_words)
        if colour is None:
            # Grammar note (CLAUDE.md §14.3): the colour clause is optional -- a line naming no
            # colour at all ("Player #11 scores goal", input/video1/instruction) still parses.
            # `Annotation.team_colour` is a required `str`, so a config-owned sentinel (never a
            # real colour word, so it can never spuriously match a `colour_reference_lab` entry
            # downstream) records "no colour was given" rather than fabricating one.
            colour = cfg["unspecified_colour"]
        jersey, content = _extract_jersey(content)
        if jersey is None:
            # The one truly required token (a jersey number) never appeared -- not parseable.
            problems.append(f"line {lineno}: {raw_line}")
            continue

        phrase = re.sub(r"\s+", " ", content).strip()
        annotations.append(
            Annotation(
                t=_parse_time(time_match.group("time")),
                jersey_number=jersey,
                team_colour=colour,
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
