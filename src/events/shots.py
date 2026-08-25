"""Stage 4 — shot events: heuristic ONLY, from ball motion (CLAUDE.md §5 Stage 5 shots row).

There is no goal-position model or pitch homography on this footage (ADR-3 — the pitch lines are
too sparse/non-standard for the assisted manual calibration to run at all on several clips,
CLAUDE.md §3.2(4)). "Toward a goal" is therefore approximated as: the ball moves fast, AND that
fast motion is sustained in one consistent horizontal direction for a minimum duration. This is a
deliberately coarse stand-in — it has no idea where either goal actually is — and will also fire
on a hard clearance, a long cross-field pass, or a fast solo dribble toward either touchline end.

**Honest expectation, not just a caveat**: on this footage (a wide, high, mostly-static sideline
camera; the ball is frequently small/occluded/interpolated — CLAUDE.md §3.2), expect this
heuristic to surface FEW candidates, and for those it does surface to be a mix of real strikes,
clearances, and hopeful long balls. `configs/events.yaml: shot.max_confidence` hard-caps every
shot event at a low ceiling for exactly this reason (Golden Rule 5) — the correct response to a
weak signal is fewer/no events at the configured threshold, never a lowered threshold to manufacture
some. See the Stage 4 run report for what this actually produced on `clip2`/`clip4`.
"""

from __future__ import annotations

import uuid

from src.common.logging import DropCounter
from src.common.types import BallDetection, Event, EventType
from src.events.segments import find_threshold_segments, moving_average


def ball_speed_series(
    balls: list[BallDetection], frame_width: float, window: int, min_conf: float
) -> tuple[list[float], list[float], list[float]]:
    """Normalised ball speed: pixel displacement / dt / `frame_width` ("frame-widths per second"
    — self-scaling across clips decoded at different widths; ADR-6: still never a metric unit).

    Detections below `min_conf` are dropped from the series before differencing entirely — this
    also softly deweights long interpolated stretches, since an interpolated `BallDetection`'s own
    `conf` is already penalised by `configs/detect.yaml:
    ball_interpolation.interpolated_conf_penalty`.

    Returns `(times, speeds, dxs)` — `dxs[i]` is the raw (unsigned-by-nothing, i.e. signed) x
    displacement between sample `i-1` and `i`, needed by `_direction_consistency` below; `speeds`
    and `dxs` are aligned 1:1 with the filtered+sorted `balls`, with `speeds[0] == dxs[0] == 0.0`
    (no prior sample to diff against).
    """
    kept = sorted((b for b in balls if b.conf >= min_conf), key=lambda b: b.t)
    if len(kept) < 2:
        return [b.t for b in kept], [0.0] * len(kept), [0.0] * len(kept)

    smoothed_cx = moving_average([b.bbox.cx for b in kept], window)
    smoothed_cy = moving_average([b.bbox.cy for b in kept], window)
    times = [b.t for b in kept]

    speeds = [0.0]
    dxs = [0.0]
    for i in range(1, len(kept)):
        dt = times[i] - times[i - 1]
        dx = smoothed_cx[i] - smoothed_cx[i - 1]
        dy = smoothed_cy[i] - smoothed_cy[i - 1]
        dist = (dx * dx + dy * dy) ** 0.5
        speeds.append(0.0 if dt <= 0 or frame_width <= 0 else (dist / dt) / frame_width)
        dxs.append(dx)
    return times, speeds, dxs


def direction_consistency(dxs: list[float]) -> float:
    """Fraction of nonzero x-steps in `dxs` that share the sign of their own sum.

    `0.0` when every step is exactly zero (no horizontal motion at all — cannot be "sustained in
    one direction" by definition). Pure function, unit-testable without a `BallDetection`.
    """
    nonzero = [d for d in dxs if d != 0.0]
    if not nonzero:
        return 0.0
    net_positive = sum(nonzero) >= 0
    matching = sum(1 for d in nonzero if (d >= 0) == net_positive)
    return matching / len(nonzero)


def shot_confidence(peak_speed: float, threshold: float, shot_cfg: dict) -> float:
    """Confidence scales linearly with how far `peak_speed` clears `threshold`, hard-clamped to
    `[min_confidence, max_confidence]` (Golden Rule 5: this heuristic must never read as more than
    "worth a human glance" — see module docstring for why).
    """
    if threshold <= 0:
        return shot_cfg["min_confidence"]
    excess = max(0.0, (peak_speed / threshold) - 1.0)
    raw = shot_cfg["min_confidence"] + excess * 0.1
    return min(shot_cfg["max_confidence"], max(shot_cfg["min_confidence"], raw))


def detect_shots(
    balls: list[BallDetection],
    take_id: int | None,
    frame_width: float,
    events_cfg: dict,
    drops: DropCounter | None = None,
) -> list[Event]:
    """Detect shot candidates within one take's own ball detections (`take_id` is stamped
    directly onto every emitted `Event` — the caller is responsible for having already bucketed
    `balls` into a single take, so a shot can never span a cut, Golden Rule 3).

    `player_track_id` is deliberately left `None` — this is a ball-only signal with no player
    attribution; `src/highlights/ranking.py` decides whether a shot is close enough to the
    selected target's own timeline to credit it to that player (see `configs/highlights.yaml:
    attribution`), rather than this module guessing.
    """
    shot_cfg = events_cfg["shot"]
    times, speeds, dxs = ball_speed_series(
        balls, frame_width, shot_cfg["smoothing_window_frames"], shot_cfg["min_ball_conf"]
    )
    segments = find_threshold_segments(
        times,
        speeds,
        shot_cfg["ball_speed_threshold"],
        shot_cfg["min_duration_s"],
        shot_cfg["merge_gap_s"],
    )

    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    events: list[Event] = []
    for t_start, t_end in segments:
        idxs_in = [i for i, t in enumerate(times) if t_start <= t <= t_end]
        window_speeds = [speeds[i] for i in idxs_in]
        window_dxs = [dxs[i] for i in idxs_in]
        consistency = direction_consistency(window_dxs)
        if consistency < shot_cfg["direction_consistency_ratio"]:
            # fast but not travelling in one sustained direction (contested/bouncing ball) — not
            # a plausible single directed strike, per this heuristic's own stated proxy.
            if drops is not None:
                drops.drop("shot_direction_inconsistent")
            continue

        peak_speed = max(window_speeds) if window_speeds else 0.0
        confidence = shot_confidence(peak_speed, shot_cfg["ball_speed_threshold"], shot_cfg)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("shot_below_min_emit_confidence")
            continue

        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.SHOT,
                t_start=t_start,
                t_end=t_end,
                player_track_id=None,
                take_id=take_id,
                confidence=confidence,
                source="heuristic_ball_motion",
                evidence={
                    "peak_speed": peak_speed,
                    "unit": "frame_widths_per_second",
                    "calibrated": False,
                    "direction_consistency": consistency,
                    "t_range": [t_start, t_end],
                },
            )
        )
    return events
