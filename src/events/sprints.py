"""Stage 4 — sprint events (CLAUDE.md §5 Stage 5 sprints row; ADR-6).

⚠️ ADR-6: there is no metric pitch homography on this footage (`src/pitch/` is an unimplemented
seam — CLAUDE.md §3.2(4)). Speed is therefore measured in a NORMALISED PIXEL unit we define
ourselves — "bbox-heights per second" — never fabricated m/s (Golden Rule 5). See
`configs/events.yaml: sprint` for the full justification of the unit and every threshold below.

A `Track` never crosses a take boundary (Golden Rule 3 — IDs reset at cuts), so a sprint detected
within one track's own box sequence can never cross a cut either; no extra guard is needed here.
"""

from __future__ import annotations

import uuid

from src.common.logging import DropCounter
from src.common.types import Event, EventType, Track
from src.events.segments import find_threshold_segments, moving_average


def track_speed_series(track: Track, window: int) -> tuple[list[float], list[float]]:
    """Per-track normalised speed series, aligned 1:1 with `track.boxes`.

    `speeds[i]` is the smoothed-centroid displacement between box `i-1` and box `i`, divided by
    `dt` and by the mean bbox height of the two boxes (see module docstring for the unit
    rationale). `speeds[0]` is always `0.0` by construction (no prior box to diff against) — this
    is harmless: a genuine sprint must sustain `sprint_min_duration_s`, so one leading zero-sample
    never wrongly starts or ends a segment on its own.
    """
    boxes = track.boxes
    if len(boxes) < 2:
        return [b.t for b in boxes], [0.0] * len(boxes)

    smoothed_cx = moving_average([b.bbox.cx for b in boxes], window)
    smoothed_cy = moving_average([b.bbox.cy for b in boxes], window)
    heights = [b.bbox.height for b in boxes]
    times = [b.t for b in boxes]

    speeds = [0.0]
    for i in range(1, len(boxes)):
        dt = times[i] - times[i - 1]
        mean_h = (heights[i] + heights[i - 1]) / 2.0
        if dt <= 0 or mean_h <= 0:
            speeds.append(0.0)
            continue
        dx = smoothed_cx[i] - smoothed_cx[i - 1]
        dy = smoothed_cy[i] - smoothed_cy[i - 1]
        dist = (dx * dx + dy * dy) ** 0.5
        speeds.append((dist / dt) / mean_h)
    return times, speeds


def segment_confidence(
    track: Track, t_start: float, t_end: float, events_cfg: dict, fps_sample: float
) -> tuple[float, float, int]:
    """Sprint confidence: blend track CONTINUITY and mean DETECTION confidence over the sprint
    window, then apply ADR-6's uncalibrated penalty (`configs/events.yaml:
    sprint.uncalibrated_confidence_penalty`) — a pixel-unit speed must never read as confidently
    as a calibrated one.

    `continuity` = (# of the track's own boxes actually falling in `[t_start, t_end]`) / (# boxes
    EXPECTED over that duration at `fps_sample`), capped at 1.0 — a track with detection gaps
    mid-sprint (occlusion, a missed frame) scores lower than one with a dense, unbroken run.

    Returns `(confidence, mean_detection_confidence, n_boxes_in_window)` so the caller can carry
    the last two into `Event.evidence` for traceability (Golden Rule 5).
    """
    boxes_in = [b for b in track.boxes if t_start - 1e-9 <= b.t <= t_end + 1e-9]
    n = len(boxes_in)
    if n == 0:
        return 0.0, 0.0, 0
    mean_det_conf = sum(b.conf for b in boxes_in) / n
    duration = max(t_end - t_start, 1e-6)
    expected = max(1, round(duration * fps_sample) + 1)
    continuity = min(1.0, n / expected)

    weight = events_cfg["confidence"]["continuity_weight"]
    base = weight * continuity + (1.0 - weight) * mean_det_conf
    penalty = events_cfg["sprint"]["uncalibrated_confidence_penalty"]
    return base * penalty, mean_det_conf, n


def detect_sprints(
    track: Track, events_cfg: dict, fps_sample: float, drops: DropCounter | None = None
) -> list[Event]:
    """Detect sprint events within one track's own box sequence (never crosses a take — Golden
    Rule 3, since a `Track` belongs to exactly one `take_id`).
    """
    sprint_cfg = events_cfg["sprint"]
    if len(track.boxes) < sprint_cfg["min_track_boxes"]:
        if drops is not None:
            drops.drop("sprint_track_too_short")
        return []

    times, speeds = track_speed_series(track, sprint_cfg["smoothing_window_frames"])
    segments = find_threshold_segments(
        times,
        speeds,
        sprint_cfg["sprint_speed_threshold"],
        sprint_cfg["sprint_min_duration_s"],
        sprint_cfg["sprint_merge_gap_s"],
    )

    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    events: list[Event] = []
    for t_start, t_end in segments:
        window_speeds = [s for t, s in zip(times, speeds, strict=True) if t_start <= t <= t_end]
        confidence, mean_det_conf, n_boxes = segment_confidence(
            track, t_start, t_end, events_cfg, fps_sample
        )
        if confidence < min_emit:
            if drops is not None:
                drops.drop("sprint_below_min_emit_confidence")
            continue
        peak_speed = max(window_speeds) if window_speeds else 0.0
        mean_speed = sum(window_speeds) / len(window_speeds) if window_speeds else 0.0
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.SPRINT,
                t_start=t_start,
                t_end=t_end,
                player_track_id=track.id,
                take_id=track.take_id,
                confidence=confidence,
                source="speed_heuristic_normalized_pixel",
                evidence={
                    "track_id": track.id,
                    "peak_speed": peak_speed,
                    "mean_speed": mean_speed,
                    "unit": "bbox_heights_per_second",
                    "calibrated": False,
                    "t_range": [t_start, t_end],
                    "mean_detection_confidence": mean_det_conf,
                    "n_boxes_in_window": n_boxes,
                },
            )
        )
    return events
