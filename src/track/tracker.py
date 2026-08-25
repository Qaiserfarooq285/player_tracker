"""Stage 3 — within-take multi-object tracking (CLAUDE.md §5 Stage 3, Golden Rule 3).

Uses `supervision`'s `ByteTrack` (MIT) — never BoxMOT (AGPL-3.0, Golden Rule 6). A **fresh**
`ByteTrack` instance is constructed per take (`track_take`) so track IDs never continue across a
camera cut (ADR-7: `clip4` has 3 real cuts despite being single-camera source footage; Golden
Rule 3: "track only within a take; reset IDs at every cut").

Note on `supervision` 0.30.0: it marks `ByteTrack` deprecated in favour of `ByteTrackTracker` from
a separate `trackers` package (not installed in this project's environment, and not in its
dependency table) — the CLAUDE.md task spec is explicit ("Tracker MUST be `supervision`'s
ByteTrack"), so this module uses it as specified; the library's `FutureWarning` is suppressed at
the one call site that constructs it, since it is a known, accepted, spec'd choice rather than an
accidental use of soon-to-be-removed API. Revisit if/when `trackers` is added to the project.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict

import numpy as np
import supervision as sv

from src.common.logging import get_logger
from src.common.types import BBox, Detection, DetectionClass, Take, Track, TrackBox

logger = get_logger(__name__)

# Fixed, order-stable mapping between our DetectionClass enum and supervision's integer class_id
# (supervision's Detections/ByteTrack machinery needs an int id, not our str enum).
_CLASS_LIST = list(DetectionClass)
_CLASS_ID_BY_DETECTION_CLASS = {c: i for i, c in enumerate(_CLASS_LIST)}
_DETECTION_CLASS_BY_ID = dict(enumerate(_CLASS_LIST))


def _build_bytetrack(track_cfg: dict, fps_sample: float) -> sv.ByteTrack:
    """Construct one fresh `sv.ByteTrack` from `configs/track.yaml: bytetrack`."""
    bt_cfg = track_cfg["bytetrack"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return sv.ByteTrack(
            track_activation_threshold=bt_cfg["track_activation_threshold"],
            lost_track_buffer=bt_cfg["lost_track_buffer"],
            minimum_matching_threshold=bt_cfg["minimum_matching_threshold"],
            frame_rate=fps_sample,
        )


def _detections_to_sv(detections: list[Detection]) -> sv.Detections:
    """Convert our `Detection` list (one frame's worth) to `supervision.Detections`."""
    if not detections:
        return sv.Detections.empty()
    xyxy = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float32)
    confidence = np.array([d.conf for d in detections], dtype=np.float32)
    class_id = np.array([_CLASS_ID_BY_DETECTION_CLASS[d.cls] for d in detections], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def track_take(
    detections_by_frame: list[tuple[int, float, list[Detection]]],
    take: Take,
    track_cfg: dict,
    fps_sample: float,
) -> list[Track]:
    """Track one take's worth of detections with a fresh `ByteTrack` instance.

    `detections_by_frame` must already be sorted ascending by frame time — `ByteTrack` is a
    stateful, streaming (Kalman-filter-based) tracker; feeding frames out of order silently
    corrupts its internal state rather than raising. Only classes listed in
    `track_cfg["tracked_classes"]` are fed to the tracker at all (ball is excluded by convention —
    see the module docstring / `configs/track.yaml`). Each returned `Track.dominant_class` is the
    majority-vote `DetectionClass` across that track's own matched detections.
    """
    tracker = _build_bytetrack(track_cfg, fps_sample)
    tracked_classes = {DetectionClass(c) for c in track_cfg["tracked_classes"]}

    boxes_by_id: dict[int, list[TrackBox]] = defaultdict(list)
    classes_by_id: dict[int, Counter[DetectionClass]] = defaultdict(Counter)

    for frame_index, t, dets in detections_by_frame:
        dets = [d for d in dets if d.cls in tracked_classes]
        sv_dets = _detections_to_sv(dets)
        tracked = tracker.update_with_detections(sv_dets)
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            boxes_by_id[tid].append(
                TrackBox(
                    frame_index=frame_index, t=t, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), conf=conf
                )
            )
            classes_by_id[tid][_DETECTION_CLASS_BY_ID[int(tracked.class_id[i])]] += 1

    tracks: list[Track] = []
    for tid, boxes in boxes_by_id.items():
        boxes.sort(key=lambda b: b.frame_index)
        dominant_class = classes_by_id[tid].most_common(1)[0][0] if classes_by_id[tid] else None
        tracks.append(Track(id=tid, take_id=take.id, boxes=boxes, dominant_class=dominant_class))
    tracks.sort(key=lambda tr: tr.id)
    return tracks


def assign_take_id(t: float, takes: list[Take]) -> int | None:
    """Return the `Take.id` whose `[t_start, t_end)` contains timestamp `t`, or `None`.

    Pure helper shared by `src/track/run.py` to bucket Stage 2's (take-agnostic) sampled
    detections into takes before calling `track_take` once per take. `takes` need not be sorted;
    the last take's interval is treated as closed (`t <= t_end`) so a detection exactly at a
    take's own final timestamp isn't dropped.
    """
    for take in takes:
        if take.t_start <= t < take.t_end:
            return take.id
    # closed-interval fallback for a timestamp landing exactly on the very last take's t_end
    for take in takes:
        if take.t_start <= t <= take.t_end:
            return take.id
    return None
