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

logger = get_logger(__name__)

# CLAUDE.md §13.2's Event Timeline table lists exactly these row types (Save/Possession are
# deliberately NOT among them — Possession feeds the "Possession Time" summary stat instead, and
# the owner's template simply has no "Save" row; followed literally, not padded out).
_TIMELINE_EVENT_TYPES = {
    EventType.TOUCH: "Touch",
    EventType.PASS: "Pass",
    EventType.SPRINT: "Sprint",
    EventType.DRIBBLE: "Dribble",
    EventType.SHOT: "Shot",
    EventType.TACKLE: "Tackle",
    EventType.GOAL: "Goal",
    # EventType.KEY_MOMENT splits into "Celebration"/"Other Key Moment" via _key_moment_label —
    # the owner's template distinguishes the two but this project's single Gemini prompt (ADR-14)
    # classifies both together; the split below is a light, honestly-documented text heuristic on
    # Gemini's own reasoning sentence, not a second detector.
}

# CLAUDE.md §13.4's highlight-reel categories and which EventType feeds each. "assists"/"goals"
# are included for completeness of the directory layout but their EventType lists are always
# empty on this project (no ASSIST EventType exists at all -- structurally impossible per ADR-13;
# GOAL is never emitted, §3.2(3)) -- both therefore ALWAYS come out absent, exactly as CLAUDE.md
# §13.4 expects ("expect assists.mp4 and goals.mp4 to always be empty on this footage").
_HIGHLIGHT_CATEGORIES: dict[str, list[EventType]] = {
    "ball_possession": [EventType.POSSESSION],
    "dribbles": [EventType.DRIBBLE],
    "assists": [],  # no ASSIST EventType exists in this codebase -- see module docstring
    "goals": [EventType.GOAL],
    "key_moments": [EventType.KEY_MOMENT],
}


def _format_timestamp(seconds: float) -> str:
    """`M:SS.s` rendering, identical convention to `src.stats.stats._format_timestamp`."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:04.1f}"


def _key_moment_label(event: Event) -> str:
    """ "Celebration" vs "Other Key Moment" — a light, documented text heuristic on Gemini's own
    one-sentence reasoning (`Event.evidence['gemini_raw_response']`, `src/events/key_moments.py`),
    since the single ADR-14 prompt classifies both together and the owner's template distinguishes
    them. Defaults to "Other Key Moment" when the reasoning text doesn't clearly say so."""
    raw = str(event.evidence.get("gemini_raw_response", "")).lower()
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
            continue  # SAVE/POSSESSION: not a template timeline row, see module docstring
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
    jersey_number: int,
    counts: dict[str, int],
    possession_seconds: float,
    distance_result: dict,
    timeline_rows: list[dict],
) -> str:
    """The owner's EXACT `statcard.md` template (CLAUDE.md §13.2, revised 2026-08-27), filled with
    real computed values. `Goals`/`Assists` are always `not available` on this footage (§3.2(3):
    no scoreboard anywhere to verify against) — a data ceiling, not a missing feature.
    """
    lines = [
        "# Player Statistics",
        "",
        f"## Player #{jersey_number}",
        "",
        "**Identity Status:** Verified",
        "",
        f"**Touches:** {counts.get('touch', 0)}",
        f"**Passes:** {counts.get('pass', 0)}",
        f"**Sprints/Runs:** {counts.get('sprint', 0)}",
        "**Goals:** not available",
        "**Assists:** not available",
        f"**Shots:** {counts.get('shot', 0)}",
        f"**Tackles:** {counts.get('tackle', 0)}",
        f"**Saves:** {counts.get('save', 0)}",
        f"**Dribbles:** {counts.get('dribble', 0)}",
        f"**Possession Time:** {possession_seconds:.1f}s",
        f"**Distance Covered:** {distance_result['distance']:.2f} {distance_result['unit']} "
        "(uncalibrated)",
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
    jersey_number: int,
    events: list[Event],
    possession_seconds: float,
    distance_result: dict,
    takes_by_id: dict[int, Take],
    video_path,
    highlights_cfg: dict,
) -> None:
    """Write one verified player's complete `output/<slug>/players/player_<N>/` tree
    (CLAUDE.md §13.5)."""
    player_dir.mkdir(parents=True, exist_ok=True)
    (player_dir / "events").mkdir(parents=True, exist_ok=True)
    (player_dir / "highlights").mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.type.value] = counts.get(ev.type.value, 0) + 1

    timeline_rows = build_event_timeline_rows(events)
    save_json(timeline_rows, player_dir / "events" / "event_timeline.json")

    markdown = render_statcard_markdown(
        jersey_number, counts, possession_seconds, distance_result, timeline_rows
    )
    (player_dir / "statcard.md").write_text(markdown)

    drops = DropCounter(f"player_{jersey_number}_highlights")
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
        logger.info("player_%d highlight cutting drops: %s", jersey_number, dropped)
