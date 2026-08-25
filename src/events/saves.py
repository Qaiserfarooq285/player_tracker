"""Best-effort goalkeeper-save detection (ADR-13/14; CLAUDE.md §13.3 "Save (GK)" row).

⚠️ Owner-authorized exception to Golden Rule 7 (CLAUDE.md §12/ADR-13): arithmetic on cached
`Track`/`BallDetection` artifacts, no training/new model (Golden Rule 2). Structurally the
WEAKEST heuristic in this file (`configs/events.yaml: save.max_confidence` is the lowest ceiling
of every category added by ADR-13/14): there is no goal-position model or pitch homography on this
footage (ADR-3), so "near the goalkeeper" only ever means "near whatever bbox a GOALKEEPER-class
track happens to have" — there is no notion of "the goal line", "on target", or "kept out" at all,
only "the ball moved fast toward this bbox, then sharply slowed or reversed".

Only evaluated for a take with at least one `dominant_class == GOALKEEPER` track — a take with
none produces zero save events, never a forced/guessed one (CLAUDE.md task spec, Golden Rule 5).

⚠️ Honest limitation, not a caveat buried in a config comment: as of this task, NEITHER
`work/clip2_77` NOR `work/clip4_77` contains a single GOALKEEPER-class track (checked 2026-08-25 —
`src/team/`'s dominant-class majority vote never produced one on either clip's tracked players).
This module has therefore never been exercised against real footage, only the synthetic cases in
`tests/test_saves.py` — see the run report in the final task summary for what that means in
practice (zero events on both available clips, which is the correct behaviour given the input, not
a bug to chase).
"""

from __future__ import annotations

import uuid

from src.common.types import BallDetection, DetectionClass, Event, EventType, Track
from src.events.segments import find_threshold_segments
from src.events.shots import ball_speed_series
from src.events.touches import nearest_box_in_time


def goalkeeper_tracks(take_tracks: list[Track]) -> list[Track]:
    """Every `dominant_class == GOALKEEPER` track in one take's own track list."""
    return [tr for tr in take_tracks if tr.dominant_class == DetectionClass.GOALKEEPER]


def _nearest_goalkeeper_within(
    bx: float,
    by: float,
    t: float,
    gk_tracks: list[Track],
    proximity_px: float,
    match_tolerance_s: float,
) -> Track | None:
    """The goalkeeper track whose own box (nearest in time to `t`, within `match_tolerance_s`) is
    closest to `(bx, by)`, if within `proximity_px` — `None` when no goalkeeper qualifies (either
    no box at all near `t`, or the nearest one is still too far)."""
    best: Track | None = None
    best_dist: float | None = None
    for gk in gk_tracks:
        box = nearest_box_in_time(gk, t, match_tolerance_s)
        if box is None:
            continue
        dist = ((box.bbox.cx - bx) ** 2 + (box.bbox.cy - by) ** 2) ** 0.5
        if dist <= proximity_px and (best_dist is None or dist < best_dist):
            best, best_dist = gk, dist
    return best


def save_confidence(
    peak_speed_before: float, peak_speed_after: float, reversed_direction: bool, save_cfg: dict
) -> float:
    """Blends how far `peak_speed_after` fell below `save_deceleration_ratio * peak_speed_before`
    with a flat bonus for a net direction reversal, hard-clamped to `save`'s own (lowest-of-all-
    categories) `[min_confidence, max_confidence]`. Either signal alone can max the (very low)
    ceiling — this heuristic has no way to tell a genuine save apart from an unrelated deceleration
    near a goalkeeper who wasn't actually involved, so it never claims more than the config ceiling
    regardless of how "clean" the deceleration/reversal looks (Golden Rule 5).
    """
    ratio_cfg = save_cfg["save_deceleration_ratio"]
    if peak_speed_before > 0 and ratio_cfg > 0:
        actual_ratio = peak_speed_after / peak_speed_before
        deceleration_strength = max(0.0, 1.0 - actual_ratio / ratio_cfg)
    else:
        deceleration_strength = 0.0
    signal = max(deceleration_strength, 1.0 if reversed_direction else 0.0)
    signal = min(1.0, signal)
    span = save_cfg["max_confidence"] - save_cfg["min_confidence"]
    raw = save_cfg["min_confidence"] + signal * span
    return min(save_cfg["max_confidence"], max(save_cfg["min_confidence"], raw))


def detect_saves(
    balls: list[BallDetection],
    take_tracks: list[Track],
    take_id: int | None,
    frame_width: float,
    events_cfg: dict,
    match_tolerance_s: float = 0.15,
    drops=None,
) -> list[Event]:
    """Detect save candidates within one take's own ball detections + tracks (`take_id` stamped
    directly, same convention as every other `detect_*` — the caller must have already bucketed
    both into a single take, Golden Rule 3).

    A candidate = a fast, sustained ball-speed segment (`save.ball_speed_threshold`/
    `min_duration_s`/`merge_gap_s`, reusing `src/events/shots.py::ball_speed_series` — the same
    underlying signal `shot` uses) whose END sits within `save_gk_proximity_px` of some goalkeeper
    track, followed within `save_max_window_s` by either a sharp deceleration
    (`save_deceleration_ratio`) or a net horizontal-direction reversal.

    `match_tolerance_s` defaults to the same value this project's other geometry lookups use
    (`configs/highlights.yaml: selection.arrow_match_tolerance_s` /
    `configs/events.yaml: touch.match_tolerance_s`) — kept as a parameter rather than a second
    config key purely because `save` has no natural "own" tolerance distinct from theirs; callers
    that already have `events_cfg["possession"]["match_tolerance_s"]` in hand should pass it.

    Returns `[]` immediately (never a forced/guessed save) when `take_tracks` contains no
    `dominant_class == GOALKEEPER` track at all.

    Asserted defensively, same reasoning as `src/events/{touches,possession,tackles}.py`:
    `take_tracks` must be one take's own.
    """
    take_ids = {tr.take_id for tr in take_tracks}
    assert len(take_ids) <= 1, "detect_saves must only ever see one take's tracks"

    save_cfg = events_cfg["save"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]

    gk_tracks = goalkeeper_tracks(take_tracks)
    if not gk_tracks:
        if drops is not None:
            drops.drop("save_no_goalkeeper_track_in_take")
        return []

    times, speeds, dxs = ball_speed_series(
        balls, frame_width, save_cfg["smoothing_window_frames"], save_cfg["min_ball_conf"]
    )
    approach_segments = find_threshold_segments(
        times,
        speeds,
        save_cfg["ball_speed_threshold"],
        save_cfg["min_duration_s"],
        save_cfg["merge_gap_s"],
    )
    if not approach_segments:
        if drops is not None:
            drops.drop("save_no_fast_ball_approach")
        return []

    min_ball_conf = save_cfg["min_ball_conf"]
    kept_balls = sorted((b for b in balls if b.conf >= min_ball_conf), key=lambda b: b.t)

    events: list[Event] = []
    for t_start, t_end in approach_segments:
        idxs_in = [i for i, t in enumerate(times) if t_start <= t <= t_end]
        if not idxs_in:
            continue
        end_idx = idxs_in[-1]
        end_ball = kept_balls[end_idx]

        goalkeeper = _nearest_goalkeeper_within(
            end_ball.bbox.cx,
            end_ball.bbox.cy,
            end_ball.t,
            gk_tracks,
            save_cfg["save_gk_proximity_px"],
            match_tolerance_s,
        )
        if goalkeeper is None:
            if drops is not None:
                drops.drop("save_approach_not_near_goalkeeper")
            continue

        max_window_s = save_cfg["save_max_window_s"]
        after_idxs = [i for i in range(end_idx + 1, len(times)) if times[i] - t_end <= max_window_s]
        if not after_idxs:
            if drops is not None:
                drops.drop("save_no_reaction_window")
            continue

        peak_before = max(speeds[i] for i in idxs_in)
        peak_after = max(speeds[i] for i in after_idxs)
        net_before = sum(dxs[i] for i in idxs_in)
        net_after = sum(dxs[i] for i in after_idxs)

        decelerated = (
            peak_before > 0 and peak_after <= save_cfg["save_deceleration_ratio"] * peak_before
        )
        reversed_direction = (
            net_before != 0 and net_after != 0 and (net_before > 0) != (net_after > 0)
        )
        if not (decelerated or reversed_direction):
            if drops is not None:
                drops.drop("save_no_deceleration_or_reversal")
            continue

        confidence = save_confidence(peak_before, peak_after, reversed_direction, save_cfg)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("save_below_min_emit_confidence")
            continue

        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.SAVE,
                t_start=t_start,
                t_end=times[after_idxs[-1]],
                player_track_id=goalkeeper.id,
                take_id=take_id,
                confidence=confidence,
                source="ball_deceleration_near_goalkeeper_heuristic",
                evidence={
                    "goalkeeper_track_id": goalkeeper.id,
                    "peak_speed_before": peak_before,
                    "peak_speed_after": peak_after,
                    "decelerated": decelerated,
                    "reversed_direction": reversed_direction,
                    "save_gk_proximity_px": save_cfg["save_gk_proximity_px"],
                    "unit": "frame_widths_per_second",
                    "calibrated": False,
                    "t_range": [t_start, t_end],
                },
            )
        )
    return events
