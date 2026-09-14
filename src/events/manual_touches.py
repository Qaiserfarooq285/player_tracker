"""Plan Stage 2 ("streamed-gathering-treehouse", 2026-09-14) -- optional client-typed ball-touch
times, submitted alongside the multi-anchor click flow (Stage 1) rather than through the
manual-annotation sidecar (ADR-19): a click already routes a run to the auto-detect pipeline and
explicitly skips the sidecar (`apps/api/main.py::_run_pipeline_job`'s own "sidecar, if any, is not
used" log line) -- so touch times need their own field that works INSIDE the click branch, not a
second annotation-sidecar mechanism.

Owner's chosen merge rule: "their times win, auto-detection fills the gaps" -- a client-typed touch
time is always kept; an AUTO-detected `TOUCH` event within `configs/events.yaml:
manual_touch.merge_window_s` seconds of one is treated as the SAME touch (dropped, never double-
counted -- CLAUDE.md §10 "log everything dropped. Silent filtering = hidden bugs.").

Reuses `src.annotations.parse._parse_time` (CLAUDE.md §10: do not write a second time parser) --
this module's OWN `configs/events.yaml: manual_touch.time_pattern` key mirrors (does not cross-
import) `configs/annotations.yaml: time_pattern`'s value, per this file's own "independently
runnable stage configs" convention (see e.g. `configs/events.yaml: touch.match_tolerance_s`'s own
comment for the identical duplication precedent).
"""

from __future__ import annotations

import re
import uuid

from src.annotations.parse import _parse_time
from src.common.types import Event, EventType


def parse_touch_times(raw: str, cfg: dict) -> tuple[list[float], list[str]]:
    """Parse a comma- or newline-separated list of `M:SS`/`H:MM:SS` touch times (the owner's own
    example format, e.g. ``"12:34, 45:10"`` or one entry per line) into ``(times, problems)``.

    An entry that doesn't match ``cfg["time_pattern"]`` lands in `problems` VERBATIM (CLAUDE.md
    §10: never silently dropped) rather than raising or being skipped outright -- the caller is
    responsible for surfacing `problems` in the run report, same discipline as
    `src.annotations.parse.parse_annotations`'s own `problems` list.
    """
    pattern = re.compile(rf"^(?P<time>{cfg['time_pattern']})$", re.IGNORECASE)
    times: list[float] = []
    problems: list[str] = []
    for entry in re.split(r"[,\n]", raw or ""):
        stripped = entry.strip()
        if not stripped:
            continue
        match = pattern.match(stripped)
        if match is None:
            problems.append(stripped)
            continue
        times.append(_parse_time(match.group("time")))
    return times, problems


def manual_touch_events(times: list[float], cfg: dict) -> list[Event]:
    """One `Event(type=TOUCH, source="manual_touch")` per client-typed time.

    `player_track_id` is left `None` here -- same reasoning as
    `src.annotations.parse.annotation_to_event`: this function only ever converts what the client
    actually typed, never invents a track association. `t_end = t + cfg["event_window_seconds"]`
    since a typed touch time records a single INSTANT the client watched, not a measured
    `[t_start, t_end]` interval the way an auto-detector heuristic produces one.
    """
    window = cfg["event_window_seconds"]
    confidence = cfg["confidence"]
    events: list[Event] = []
    for t in times:
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.TOUCH,
                t_start=t,
                t_end=t + window,
                player_track_id=None,
                take_id=None,
                confidence=confidence,
                source="manual_touch",
                evidence={"raw_time": t, "provenance": "user_supplied"},
            )
        )
    return events


def merge_manual_touches(
    auto_events: list[Event], manual_events: list[Event], cfg: dict
) -> tuple[list[Event], int]:
    """Owner's merge rule: "their times win, auto-detection fills the gaps".

    Keeps every manual touch unconditionally; drops an AUTO `EventType.TOUCH` event whose own
    `t_start` sits within `cfg["merge_window_s"]` seconds of ANY manual touch's `t_start` (the same
    real touch, not a second one -- never double-counted). Every non-TOUCH event (sprints, shots,
    passes, ...) passes through completely untouched -- this function only ever compares TOUCH
    against TOUCH. Returns `(merged_events, n_suppressed)` so the caller can log the drop count
    (CLAUDE.md §10 "log everything dropped").
    """
    window = cfg["merge_window_s"]
    manual_times = [ev.t_start for ev in manual_events]
    kept: list[Event] = []
    n_suppressed = 0
    for ev in auto_events:
        if ev.type == EventType.TOUCH and any(
            abs(ev.t_start - mt) <= window for mt in manual_times
        ):
            n_suppressed += 1
            continue
        kept.append(ev)
    return kept + list(manual_events), n_suppressed
