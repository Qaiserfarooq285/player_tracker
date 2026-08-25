"""Stage 6 — clip cutting (CLAUDE.md §5 Stage 6). Adds pre/post-roll padding, snaps to the
enclosing take's own boundaries (Golden Rule 3: a clip must NEVER cross a camera cut), dedupes
time-overlapping candidates, then cuts survivors via `src.common.video.extract_clip` (NVENC with
an automatic libx264 fallback — reused, not reimplemented).
"""

from __future__ import annotations

from pathlib import Path

from src.common.logging import DropCounter, get_logger
from src.common.types import Clip, Event, Take
from src.common.video import extract_clip

logger = get_logger(__name__)


def clip_range_for_event(event: Event, take: Take | None, clip_cfg: dict) -> tuple[float, float]:
    """`[event.t_start - pre_roll_s, event.t_end + post_roll_s]`, clamped into `take`'s own
    `[t_start, t_end]` when `clip_cfg['snap_to_take_boundaries']` is set and `take` is known
    (Golden Rule 3). Every Phase-1 event already carries the `take_id` it was derived within, so
    `take is None` should not occur in practice; when it does, the range is left unclamped rather
    than silently guessing which take it belongs to.
    """
    start = event.t_start - clip_cfg["pre_roll_s"]
    end = event.t_end + clip_cfg["post_roll_s"]
    if clip_cfg["snap_to_take_boundaries"] and take is not None:
        start = max(start, take.t_start)
        end = min(end, take.t_end)
    return start, end


def time_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection-over-union of two `[start, end)` time ranges (`0.0` for non-overlapping or
    degenerate ranges)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def dedupe_clip_ranges(
    ranked_events: list[Event],
    ranges: dict[str, tuple[float, float]],
    overlap_threshold: float,
    drops: DropCounter | None = None,
) -> list[Event]:
    """`ranked_events` must already be sorted BEST-FIRST. Drops any later event whose own clip
    range (`ranges[event.id]`) overlaps (`time_iou`) an already-kept, higher-ranked event's range
    by MORE than `overlap_threshold` — only the higher-ranked one survives.
    """
    kept: list[Event] = []
    kept_ranges: list[tuple[float, float]] = []
    for ev in ranked_events:
        rng = ranges[ev.id]
        is_dup = any(time_iou(rng, kr) > overlap_threshold for kr in kept_ranges)
        if is_dup:
            if drops is not None:
                drops.drop("deduped_overlapping_clip")
            continue
        kept.append(ev)
        kept_ranges.append(rng)
    return kept


def cut_clips(
    ranked_events: list[tuple[Event, float]],
    takes_by_id: dict[int, Take],
    video_path: str | Path,
    out_dir: str | Path,
    highlights_cfg: dict,
    drops: DropCounter | None = None,
) -> list[Clip]:
    """Snap, dedupe, and cut `ranked_events` (best-first `(event, rank_score)` pairs from
    `src.highlights.ranking.rank_events`) into individual clip files under `out_dir`.

    Returns `Clip`s in the SAME best-first order as the surviving events, each carrying its own
    `rank_score` (Golden Rule 5 traceability: every `Clip.event_id` maps back to exactly one
    `Event`).
    """
    clip_cfg = highlights_cfg["clip"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_by_id = {ev.id: score for ev, score in ranked_events}
    events_sorted = [ev for ev, _ in ranked_events]  # already best-first per rank_events' contract

    ranges: dict[str, tuple[float, float]] = {}
    for ev in events_sorted:
        take = takes_by_id.get(ev.take_id) if ev.take_id is not None else None
        rng = clip_range_for_event(ev, take, clip_cfg)
        if rng[1] <= rng[0]:
            logger.warning(
                "event %s produced a degenerate/empty clip range %s -- dropping", ev.id, rng
            )
            if drops is not None:
                drops.drop("degenerate_clip_range")
            continue
        ranges[ev.id] = rng

    survivors = [ev for ev in events_sorted if ev.id in ranges]
    deduped = dedupe_clip_ranges(
        survivors, ranges, highlights_cfg["dedupe"]["iou_over_time_threshold"], drops
    )

    clips: list[Clip] = []
    for rank_index, ev in enumerate(deduped):
        t_start, t_end = ranges[ev.id]
        path = out_dir / f"clip_{rank_index:03d}_{ev.type.value}_{ev.id[:8]}.mp4"
        extract_clip(video_path, path, t_start, t_end)
        clips.append(
            Clip(
                event_id=ev.id,
                t_start=t_start,
                t_end=t_end,
                take_id=ev.take_id,
                rank_score=score_by_id.get(ev.id, 0.0),
                path=path,
            )
        )
    logger.info(
        "cut %d clip(s) into %s (from %d ranked event(s))", len(clips), out_dir, len(ranked_events)
    )
    return clips
