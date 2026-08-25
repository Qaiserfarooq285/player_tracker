"""Stage 6 — event ranking (CLAUDE.md §5 Stage 6). Attributes events to the SELECTED target
timeline (`src/highlights/selection.py`) and scores/orders the ones that qualify.

Phase 1 only ever produces `sprint`/`shot` events (`goal` is always `[]` on this footage, see
`src/events/goals.py`) — the remaining CLAUDE.md §5 priority order (assist/dribble/key_pass/pass)
is reserved `configs/highlights.yaml: rank.weights` keys for when Phase 2+ adds those event types;
they simply never get looked up here yet.
"""

from __future__ import annotations

from src.common.types import BallDetection, Event, EventType, Track
from src.highlights.selection import SelectionResult


def event_rank_score(event: Event, weights: dict[str, float], confidence_weight: float) -> float:
    """`weight[type] * (1 + confidence_weight * event.confidence)`.

    Confidence only NUDGES the ordering within/across types — it can never let a low-priority
    type outrank a high-priority one on confidence alone, since CLAUDE.md §5's type order (goal >
    assist > shot > dribble > key_pass > sprint > pass) is fixed by the weight table itself (a
    0.99-confidence sprint scores `20 * 1.5 = 30`, still well under a 0.3-confidence shot's
    `60 * 1.15 = 69`).
    """
    base = weights.get(event.type.value, 0.0)
    return base * (1.0 + confidence_weight * event.confidence)


def timeline_track_ids(selection: SelectionResult) -> dict[int, set[int]]:
    """`{take_id: {track_ids stitched into that take's timeline}}` from a `SelectionResult`."""
    return {t.take_id: set(t.track_ids) for t in selection.takes}


def _nearest_position(
    tracks: list[Track], t: float, max_gap_s: float
) -> tuple[float, float, float] | None:
    """The nearest-in-time `(cx, cy, bbox_height)` among `tracks`' own boxes to timestamp `t`,
    within `max_gap_s` seconds; `None` if nothing is close enough."""
    best = None
    best_dt = max_gap_s
    for tr in tracks:
        for box in tr.boxes:
            dt = abs(box.t - t)
            if dt <= best_dt:
                best_dt = dt
                best = (box.bbox.cx, box.bbox.cy, box.bbox.height)
    return best


def attribute_shot(
    event: Event,
    timeline_tracks: list[Track],
    balls: list[BallDetection],
    attribution_cfg: dict,
) -> bool:
    """Whether a ball-only `SHOT` event (`player_track_id is None`) belongs to the selected
    timeline: the timeline's own nearest-in-time position at `event.t_start` must be within
    `shot_proximity_bbox_heights` (bbox-heights, same normalised unit as everywhere else in Stage
    4/5/6) of the ball's own position at that instant. Never guessed when either position is
    unknown within `max_time_gap_s` (Golden Rule 5) — an unattributable shot is simply excluded
    from this player's reel/stats, not defaulted to "belongs to them".
    """
    max_gap_s = attribution_cfg["max_time_gap_s"]
    timeline_pos = _nearest_position(timeline_tracks, event.t_start, max_gap_s)
    if timeline_pos is None:
        return False

    ball_candidates = [b for b in balls if abs(b.t - event.t_start) <= max_gap_s]
    if not ball_candidates:
        return False
    ball = min(ball_candidates, key=lambda b: abs(b.t - event.t_start))

    tx, ty, theight = timeline_pos
    dx = tx - ball.bbox.cx
    dy = ty - ball.bbox.cy
    dist = (dx * dx + dy * dy) ** 0.5
    if theight <= 0:
        return False
    return (dist / theight) <= attribution_cfg["shot_proximity_bbox_heights"]


def rank_events(
    events: list[Event],
    selection: SelectionResult,
    tracks_by_take: dict[int, list[Track]],
    balls_by_take: dict[int, list[BallDetection]],
    highlights_cfg: dict,
) -> list[tuple[Event, float]]:
    """Filter `events` to only those attributed to `selection`'s target timeline, score them, and
    return best-first `(event, rank_score)` pairs.

    Attribution: an event carrying a `player_track_id` (sprints) is attributed when that id was
    stitched into the timeline for its own `take_id`; a `SHOT` event (no `player_track_id`) is
    attributed via `attribute_shot`'s spatial-proximity check instead.
    """
    ids_by_take = timeline_track_ids(selection)
    attribution_cfg = highlights_cfg["attribution"]
    weights = highlights_cfg["rank"]["weights"]
    confidence_weight = highlights_cfg["rank"]["confidence_weight"]

    attributed: list[Event] = []
    for ev in events:
        if ev.take_id is None:
            continue
        timeline_ids = ids_by_take.get(ev.take_id, set())
        if ev.player_track_id is not None:
            if ev.player_track_id in timeline_ids:
                attributed.append(ev)
            continue
        if ev.type == EventType.SHOT:
            timeline_tracks = [
                tr for tr in tracks_by_take.get(ev.take_id, []) if tr.id in timeline_ids
            ]
            balls = balls_by_take.get(ev.take_id, [])
            if attribute_shot(ev, timeline_tracks, balls, attribution_cfg):
                attributed.append(ev)

    scored = [(ev, event_rank_score(ev, weights, confidence_weight)) for ev in attributed]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
