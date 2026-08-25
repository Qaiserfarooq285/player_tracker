"""Stage 6 — player stat card (CLAUDE.md §5 Stage 6, Golden Rule 5: every stat traceable + a
confidence; §3.2(3)/ADR-6: goals report "not available", speed/distance are always labelled
"uncalibrated"). Writes both a machine-readable `stat_card.json` and a human-readable
`stat_card.md` to `output/<slug>/`.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from src.common.io import save_json
from src.common.types import Clip, Event, EventType, PlayerStats

SPEED_UNIT = "bbox_heights_per_second"
DISTANCE_UNIT = "bbox_heights"
UNCALIBRATED_NOTE = (
    "UNCALIBRATED -- no metric pitch homography is available for this footage (ADR-6, "
    "CLAUDE.md §3.2(4)); every speed/distance figure below is in a self-defined, self-scaling "
    "pixel unit, never metres or km/h."
)


def build_player_stats(player_ref: str, track_ids: list[int], events: list[Event]) -> PlayerStats:
    """Aggregate `events` (already filtered to the target timeline by
    `src.highlights.ranking.rank_events`) into a `PlayerStats` — counts and mean confidence per
    `EventType`, plus every contributing event id (click-to-clip traceability lives in
    `write_stat_card`'s `events` list, which joins these ids back to their cut `Clip` paths).
    """
    counts: dict[str, int] = {}
    conf_lists: dict[str, list[float]] = {}
    for ev in events:
        counts[ev.type.value] = counts.get(ev.type.value, 0) + 1
        conf_lists.setdefault(ev.type.value, []).append(ev.confidence)
    confidences = {t: mean(v) for t, v in conf_lists.items()}
    return PlayerStats(
        player_ref=player_ref,
        track_ids=track_ids,
        counts=counts,
        confidences=confidences,
        events=[ev.id for ev in events],
    )


def total_sprint_distance(events: list[Event]) -> float:
    """Sum of `mean_speed * duration` across every SPRINT event -- a re-derivation of numbers
    `src/events/sprints.py` already recorded in `Event.evidence` (never a newly fabricated
    figure), in the SAME normalised-pixel unit (`bbox-heights`, ADR-6) as everywhere else."""
    total = 0.0
    for ev in events:
        if ev.type != EventType.SPRINT:
            continue
        t0, t1 = ev.evidence.get("t_range", [ev.t_start, ev.t_end])
        total += ev.evidence.get("mean_speed", 0.0) * (t1 - t0)
    return total


def peak_sprint_speed(events: list[Event]) -> float:
    """The single highest `peak_speed` recorded across every SPRINT event, `0.0` if there are
    none -- also a re-derivation of already-recorded evidence, ADR-6 unit."""
    peaks = [ev.evidence.get("peak_speed", 0.0) for ev in events if ev.type == EventType.SPRINT]
    return max(peaks) if peaks else 0.0


def write_stat_card(
    stats: PlayerStats,
    events: list[Event],
    clips_by_event_id: dict[str, Clip],
    out_dir: str | Path,
    target_jersey: int | None,
    selection_summary: dict,
    goal_reason: str,
) -> tuple[Path, Path]:
    """Write `stat_card.json` + `stat_card.md` to `out_dir`. Returns their paths.

    `clips_by_event_id` makes click-to-clip traceability REAL, not decorative (Golden Rule 5):
    every event listed carries the actual cut clip's file path when one exists (an event can be
    ranked-out/deduped by Stage 6 and therefore never cut -- still listed in `counts`/
    `confidences` since it genuinely happened, just without a `clip_path`).
    """
    out_dir = Path(out_dir)
    events_by_id = {ev.id: ev for ev in events}

    event_rows = []
    for event_id in stats.events:
        ev = events_by_id.get(event_id)
        if ev is None:
            continue
        clip = clips_by_event_id.get(event_id)
        event_rows.append(
            {
                "event_id": ev.id,
                "type": ev.type.value,
                "t_start": ev.t_start,
                "t_end": ev.t_end,
                "confidence": ev.confidence,
                "source": ev.source,
                "calibrated": ev.evidence.get("calibrated"),
                "clip_path": str(clip.path) if clip and clip.path else None,
            }
        )
    # best-first by rank_score when a clip exists (uncut events sort after, oldest-first)
    event_rows.sort(
        key=lambda r: (
            -(
                clips_by_event_id[r["event_id"]].rank_score
                if r["event_id"] in clips_by_event_id
                else -1
            ),
            r["t_start"],
        )
    )

    card = {
        "player_ref": stats.player_ref,
        "target_jersey": target_jersey,
        "selection": selection_summary,
        "track_ids": stats.track_ids,
        "counts": stats.counts,
        "mean_confidence_by_type": stats.confidences,
        "goals": goal_reason,
        "speed": {
            "unit": SPEED_UNIT,
            "distance_unit": DISTANCE_UNIT,
            "calibrated": False,
            "note": UNCALIBRATED_NOTE,
            "total_sprint_distance": total_sprint_distance(events_for(stats, events_by_id)),
            "peak_sprint_speed": peak_sprint_speed(events_for(stats, events_by_id)),
        },
        "events": event_rows,
    }

    json_path = out_dir / "stat_card.json"
    save_json(card, json_path)

    md_lines = [
        f"# Player stat card -- {stats.player_ref}",
        "",
        f"Target jersey: **{target_jersey if target_jersey is not None else 'unknown'}**  ",
        f"Selection method: **{selection_summary.get('method', 'n/a')}** "
        f"(confidence {selection_summary.get('confidence', 0.0):.2f}, "
        f"needs human confirmation: {selection_summary.get('needs_human_confirmation')})  ",
        f"Track id(s) in this timeline: {stats.track_ids}",
        "",
        "## Event counts",
        "",
        "| Type | Count | Mean confidence |",
        "|---|---|---|",
    ]
    for event_type, count in sorted(stats.counts.items()):
        md_lines.append(
            f"| {event_type} | {count} | {stats.confidences.get(event_type, 0.0):.2f} |"
        )
    md_lines += [
        "",
        "## Goals",
        "",
        goal_reason,
        "",
        "## Speed / distance -- ⚠️ UNCALIBRATED",
        "",
        UNCALIBRATED_NOTE,
        "",
        f"- Total sprint distance: **{card['speed']['total_sprint_distance']:.2f} "
        f"{DISTANCE_UNIT}** (UNCALIBRATED)",
        f"- Peak sprint speed: **{card['speed']['peak_sprint_speed']:.2f} "
        f"{SPEED_UNIT}** (UNCALIBRATED)",
        "",
        "## Events (click-to-clip)",
        "",
        "| Type | t_start | t_end | Confidence | Calibrated | Clip |",
        "|---|---|---|---|---|---|",
    ]
    for row in event_rows:
        clip_cell = row["clip_path"] or "_(not cut -- ranked out / deduped)_"
        md_lines.append(
            f"| {row['type']} | {row['t_start']:.2f}s | {row['t_end']:.2f}s | "
            f"{row['confidence']:.2f} | {row['calibrated']} | {clip_cell} |"
        )

    md_path = out_dir / "stat_card.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")

    return json_path, md_path


def events_for(stats: PlayerStats, events_by_id: dict[str, Event]) -> list[Event]:
    """The actual `Event` objects behind `stats.events` (helper to avoid re-filtering twice)."""
    return [events_by_id[eid] for eid in stats.events if eid in events_by_id]
