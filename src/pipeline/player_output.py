"""ADR-15 — per-verified-player output writer (CLAUDE.md §13.2/§13.4/§13.5).

One call per DISTINCT verified jersey number found across a filename-less video's takes, writing
`output/<slug>/players/player_<N>/{statcard.md, highlights/*.mp4, events/event_timeline.json}`.
The owner's exact `statcard.md` template (§13.2, revised 2026-08-27 — supersedes the older
plain-text template `src/stats/stats.py` still writes for the 5 filename-based clips, which is
untouched) is reproduced verbatim; category highlight reels reuse the existing, tested Stage 6
cutting/reel machinery (`src.highlights.cutting.cut_clips`, `src.highlights.reel.build_reel`).
"""

from __future__ import annotations

from pathlib import Path

from src.common.io import save_json
from src.common.logging import DropCounter, get_logger
from src.common.types import Event, EventType, Take
from src.highlights.cutting import cut_clips
from src.highlights.reel import build_reel
from src.pipeline.statcard_pdf import render_statcard_pdf

logger = get_logger(__name__)

# CLAUDE.md §13.2's Event Timeline table lists exactly these row types. POSSESSION is
# deliberately NOT among them -- it feeds the "Possession Time" summary stat instead, and the
# owner's template has no per-possession timeline row; followed literally, not padded out.
_TIMELINE_EVENT_TYPES = {
    EventType.TOUCH: "Touch",
    EventType.PASS: "Pass",
    EventType.TURNOVER: "Turnover",  # ADR-20: a possession loss to an opposing colour cluster --
    # its own template row, distinct from (and never counted toward) "Pass".
    EventType.SPRINT: "Sprint",
    EventType.DRIBBLE: "Dribble",
    EventType.SHOT: "Shot",
    EventType.TACKLE: "Tackle",
    EventType.SAVE: "Save",  # owner (2026-08-31): "every save (for goalkeepers)" is an explicit
    # timeline row in their category list (CLAUDE.md §13.2/§13.3) -- the detector
    # (src/events/saves.py, ADR-13) already existed and was already counted in the summary line
    # ("Saves: N"), it just had no path into the timeline table below it. This was the one real
    # gap Stage 6 found; every other listed category already had both a summary line and a
    # timeline row.
    EventType.GOAL: "Goal",
    EventType.ASSIST: "Assist",  # ADR-17: real detector now exists (src/events/goals.py) --
    # previously omitted since EventType.ASSIST didn't exist at all.
    EventType.OUT_OF_BOUNDS: "Out of bounds",  # ADR-19: annotation-only -- there is no auto
    # detector for this (CLAUDE.md §14.3), it only ever appears via a parsed manual annotation.
    # EventType.KEY_MOMENT splits into "Celebration"/"Other Key Moment" via _key_moment_label —
    # the owner's template distinguishes the two but this project's single Gemini prompt (ADR-14)
    # classifies both together; the split below is a light, honestly-documented text heuristic on
    # Gemini's own reasoning sentence, not a second detector.
}

# CLAUDE.md §13.4's highlight-reel categories and which EventType feeds each. "goals"/"assists"
# are expected to be empty/absent on THIS footage (§3.2(3): no scoreboard anywhere to verify a
# goal against) -- but that is now a measured data ceiling (ADR-17's own real detector genuinely
# finds nothing here), not a structural absence of the EventType/detector the way it was before
# ASSIST existed at all.
_HIGHLIGHT_CATEGORIES: dict[str, list[EventType]] = {
    "ball_possession": [EventType.POSSESSION],
    "passes": [EventType.PASS],  # ADR-20/CLAUDE.md §13.4: the owner's explicitly requested reel.
    "turnovers": [EventType.TURNOVER],  # ADR-20: kept separate from "passes" -- a turnover is
    # never a completed pass, so it never contributes to that category (Golden Rule 5).
    "dribbles": [EventType.DRIBBLE],
    "assists": [EventType.ASSIST],
    "goals": [EventType.GOAL],
    "key_moments": [EventType.KEY_MOMENT],
}


def _format_timestamp(seconds: float) -> str:
    """`M:SS.s` rendering, identical convention to `src.stats.stats._format_timestamp`."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:04.1f}"


def _key_moment_label(event: Event) -> str:
    """ "Celebration" vs "Other Key Moment" — a light, documented text heuristic on either
    Gemini's own one-sentence reasoning (`Event.evidence['gemini_raw_response']`,
    `src/events/key_moments.py`, ADR-14) or, for a manual-annotation-sourced KEY_MOMENT (ADR-19,
    `src.annotations.parse.annotation_to_event`), the client's own `evidence['action_phrase']` --
    the SAME "celebrat" substring test either way (`configs/annotations.yaml`'s own phrase map
    deliberately maps "celebrat" to KEY_MOMENT rather than a second, competing EventType, exactly
    so this one heuristic covers both routes). Defaults to "Other Key Moment" when neither text
    clearly says so."""
    raw = str(
        event.evidence.get("gemini_raw_response") or event.evidence.get("action_phrase") or ""
    ).lower()
    return "Celebration" if "celebrat" in raw else "Other Key Moment"


def build_event_timeline_rows(events: list[Event]) -> list[dict]:
    """Chronological `{t_start, label, confidence}` rows for the statcard timeline table + the
    machine-readable `event_timeline.json` — pure, no I/O, so it's unit-testable directly."""
    rows = []
    for ev in events:
        if ev.type == EventType.KEY_MOMENT:
            label = _key_moment_label(ev)
        elif ev.type in _TIMELINE_EVENT_TYPES:
            label = _TIMELINE_EVENT_TYPES[ev.type]
        else:
            continue  # POSSESSION: not a template timeline row, see module docstring above
        rows.append(
            {
                "t_start": ev.t_start,
                "t_end": ev.t_end,
                "type": ev.type.value,
                "label": label,
                "confidence": ev.confidence,
                "take_id": ev.take_id,
                "source": ev.source,
            }
        )
    rows.sort(key=lambda r: r["t_start"])
    return rows


def render_statcard_markdown(
    jersey_number: int | None,
    counts: dict[str, int],
    possession_seconds: float | None,
    distance_result: dict | None,
    timeline_rows: list[dict],
    goal_reason: str | None = None,
    identity_status: str = "Verified",
) -> str:
    """The owner's EXACT `statcard.md` template (CLAUDE.md §13.2, revised 2026-08-27), filled with
    real computed values. `Goals`/`Assists` are REAL computed counts (ADR-17) when the scoreboard
    detector actually found something for this run; when the count is genuinely zero,
    `goal_reason` decides what that zero MEANS: a non-`None` `goal_reason` (auto mode's
    `GoalDetectionResult.reason` VERBATIM -- always a complete "not available (...)" sentence, see
    `src/events/goals.py`) means the zero is genuinely UNCERTAIN (no scoreboard/goal-region source
    ran at all), so the reason text is shown instead of a bare number that would overclaim
    certainty. `goal_reason=None` (bug fix 2026-08-31 -- the manual-events caller,
    `src.pipeline.manual_events`) means there is nothing uncertain about the zero: the event source
    for this run (the client's own sidecar) is authoritative and complete, so a real `0` is shown,
    exactly like every other stat (Touches/Passes/Sprints/...) already does when its own count is
    zero. Previously `manual_events.py` forced a "not available" string even when its own event
    list was authoritative and simply had no GOAL line -- this was backwards (Golden Rule 4: a
    human directly watching the footage and reporting no goal IS the answer "0", not "unknown").

    `goal_reason` (when it IS supplied and the count is zero) is used as-is here rather than
    re-wrapped in a second "not available (...)" layer. On this project's own auto-mode footage
    `goal_reason` measurably explains "not available" every time (§3.2(3): no scoreboard anywhere)
    -- a data ceiling, not a missing feature.

    `identity_status` (ADR-19) is `"Verified"` by default (the existing ADR-15 auto path --
    `TakeIdentityResult.status` only has two literal values, `verified`/`unverified`, and an
    unverified take never reaches this function at all, see `run_extended_pipeline_for_video`) or
    `"Human-provided (manual annotation)"` when the caller is Stage 4's manual-events mode (the
    jersey number/colour came straight from the client's own sidecar, never OCR/VLM-verified).

    `possession_seconds`/`distance_result` are `None` (rendered as `uncertain`, the owner's own
    word, CLAUDE.md §13.2) rather than a required float/dict, since Stage 4's manual-events mode
    has no ball-proximity possession heuristic run over annotation-only events to derive either
    value from (Golden Rule 5: never print a number -- not even a fabricated 0.0 -- for a value
    genuinely not computed this run).
    """
    goals_count = counts.get("goal", 0)
    assists_count = counts.get("assist", 0)
    # goal_reason is None => the event source for this run is authoritative (bug fix 2026-08-31,
    # see docstring) => a zero count is the real, honest answer "0", not "not available".
    goals_line = (
        f"**Goals:** {goals_count}"
        if goals_count > 0 or goal_reason is None
        else f"**Goals:** {goal_reason}"
    )
    assists_line = (
        f"**Assists:** {assists_count}"
        if assists_count > 0 or goal_reason is None
        else f"**Assists:** {goal_reason}"
    )
    lines = [
        "# Player Statistics",
        "",
        # `## Player #10` whenever a number is known (CLAUDE.md §13.2's exact template,
        # unchanged). A target the human clicked whose number was never legible is named for what
        # it IS rather than printed as `#None` -- Golden Rule 5, reachable since 2026-09-15 (see
        # `src.pipeline.annotated_video.target_name`).
        (
            f"## Player #{jersey_number}"
            if jersey_number is not None
            else "## Target player (jersey number not readable)"
        ),
        "",
        f"**Identity Status:** {identity_status}",
        "",
        f"**Touches:** {counts.get('touch', 0)}",
        f"**Passes:** {counts.get('pass', 0)}",
        f"**Turnovers:** {counts.get('turnover', 0)}",
        f"**Sprints/Runs:** {counts.get('sprint', 0)}",
        goals_line,
        assists_line,
        f"**Shots:** {counts.get('shot', 0)}",
        f"**Tackles:** {counts.get('tackle', 0)}",
        f"**Saves:** {counts.get('save', 0)}",
        f"**Dribbles:** {counts.get('dribble', 0)}",
        (
            f"**Possession Time:** {possession_seconds:.1f}s"
            if possession_seconds is not None
            else "**Possession Time:** uncertain"
        ),
        (
            f"**Distance Covered:** {distance_result['distance']:.2f} {distance_result['unit']} "
            "(uncalibrated)"
            if distance_result is not None
            else "**Distance Covered:** uncertain"
        ),
        "",
        "## Event Timeline",
        "",
        "| Time | Event | Confidence |",
        "|------|-------|------------|",
    ]
    if timeline_rows:
        for row in timeline_rows:
            lines.append(
                f"| {_format_timestamp(row['t_start'])} | {row['label']} | "
                f"{row['confidence']:.2f} |"
            )
    else:
        lines.append("| _(no events attributed to this player)_ | | |")
    return "\n".join(lines) + "\n"


def _cut_category_reel(
    category: str,
    category_events: list[Event],
    takes_by_id: dict[int, Take],
    video_path,
    player_dir: Path,
    highlights_cfg: dict,
    drops: DropCounter,
) -> None:
    """§13.4: one `highlights/<category>.mp4` per non-empty category — an EMPTY/ABSENT file
    (never written at all here), not a fabricated one, when `category_events` is empty."""
    if not category_events:
        logger.info(
            "player output: category=%s has zero events -- no highlight file written", category
        )
        return
    ranked = sorted(
        ((ev, ev.confidence) for ev in category_events), key=lambda pair: pair[1], reverse=True
    )
    clips_dir = player_dir / "highlights" / f"_{category}_clips"
    clips = cut_clips(ranked, takes_by_id, video_path, clips_dir, highlights_cfg, drops)
    if not clips:
        logger.info(
            "player output: category=%s had %d event(s) but zero survived clip-cutting -- no "
            "highlight file written",
            category,
            len(category_events),
        )
        return
    out_path = player_dir / "highlights" / f"{category}.mp4"
    build_reel(clips, out_path)


def write_player_output(
    player_dir: Path,
    jersey_number: int | None,
    events: list[Event],
    possession_seconds: float | None,
    distance_result: dict | None,
    takes_by_id: dict[int, Take],
    video_path,
    highlights_cfg: dict,
    goal_reason: str | None = None,
    identity_status: str = "Verified",
) -> None:
    """Write one verified player's complete `output/<slug>/players/player_<N>/` tree
    (CLAUDE.md §13.5). `goal_reason` (ADR-17) is the whole-video
    `src.events.goals.GoalDetectionResult.reason` -- a per-video property (scoreboard
    availability), threaded through so `statcard.md`'s Goals/Assists lines carry the SPECIFIC
    reason (Golden Rule 5) instead of a bare "not available" whenever this player's own counts
    are genuinely zero. `identity_status` (ADR-19) is `"Verified"` by default (the existing
    ADR-15 auto path) or `"Human-provided (manual annotation)"` from Stage 4's manual-events mode.
    Also writes `statcard.pdf` alongside `statcard.md` (Plan Stage 3, 2026-09-14,
    `src.pipeline.statcard_pdf.render_statcard_pdf`) from the exact same inputs -- best-effort,
    never fatal if `reportlab` isn't installed (see the try/except around that call below).
    """
    player_dir.mkdir(parents=True, exist_ok=True)
    (player_dir / "events").mkdir(parents=True, exist_ok=True)
    (player_dir / "highlights").mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.type.value] = counts.get(ev.type.value, 0) + 1

    timeline_rows = build_event_timeline_rows(events)
    save_json(timeline_rows, player_dir / "events" / "event_timeline.json")

    markdown = render_statcard_markdown(
        jersey_number,
        counts,
        possession_seconds,
        distance_result,
        timeline_rows,
        goal_reason,
        identity_status,
    )
    (player_dir / "statcard.md").write_text(markdown)

    # Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14): a real, downloadable PDF next to
    # the markdown, built from the SAME inputs (no re-typed/approximated numbers, unlike the old
    # frontend exporter this replaces -- see statcard_pdf.py's own module docstring). `reportlab`
    # lives in the OPTIONAL `api` extra (CLAUDE.md §7) -- never let its absence break the pipeline
    # or the authoritative statcard.md this function already wrote above. Fail-soft, same pattern
    # as `src.identity.jersey_models.load_optional_jersey_stack`: log a warning and move on, never
    # raise, never crash the run over a missing/broken optional artifact.
    try:
        render_statcard_pdf(
            jersey_number,
            counts,
            possession_seconds,
            distance_result,
            timeline_rows,
            goal_reason,
            identity_status,
            output_path=player_dir / "statcard.pdf",
        )
    except ImportError:
        logger.warning(
            "reportlab is not installed (`api` extra, CLAUDE.md §7: `uv pip install -e '.[api]'`) "
            "-- skipping statcard.pdf for player_%d; statcard.md was written normally",
            jersey_number,
        )
    except Exception:
        logger.warning(
            "statcard.pdf rendering failed for player_%d -- statcard.md was written normally, "
            "this player's PDF is simply missing this run",
            jersey_number,
            exc_info=True,
        )

    # Label the counter by whatever identity this target actually has -- `player_None_highlights`
    # in a log line is just noise (see `_player_dir_name` in `src.pipeline.run` for the same rule
    # applied to the on-disk folder).
    label = f"player_{jersey_number}" if jersey_number is not None else "target"
    drops = DropCounter(f"{label}_highlights")
    events_by_type: dict[EventType, list[Event]] = {}
    for ev in events:
        events_by_type.setdefault(ev.type, []).append(ev)

    for category, event_types in _HIGHLIGHT_CATEGORIES.items():
        category_events = [ev for et in event_types for ev in events_by_type.get(et, [])]
        _cut_category_reel(
            category, category_events, takes_by_id, video_path, player_dir, highlights_cfg, drops
        )

    dropped = drops.report()
    if dropped:
        logger.info("player_%s highlight cutting drops: %s", jersey_number, dropped)
