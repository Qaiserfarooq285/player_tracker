"""Per-take camera-motion estimation, so fragment stitching can measure how far a PLAYER actually
moved rather than how far the CAMERA swung.

Owner-reported bug this exists for (2026-09-01): "00:23 to 00:25 when the camera move fast its
lose the player detection and the detection box is removed". Measured on
`chelsea_burnley_target10` take 0 -- camera translation there spikes from a 2.01 px/frame median
to **12.84 px/frame** (max 16.95) across 24-26s, and ten raw tracks die/spawn inside that one
second as ByteTrack's IoU matching collapses.

The damage is not just to ByteTrack. `stitch_timeline`'s own re-acquisition thresholds
(`stitch_max_dist`, and the `stitch_max_speed_bbox_heights_per_s` gate) are measured in IMAGE
space, where a fast pan makes a stationary player look like a sprinter. Measured, for the real
target (Chelsea #10) across that exact pan:

| join (verified by reading the shirt) | raw | camera-compensated |
|---|---|---|
| track 1 -> 48, #10 -> #10 (CORRECT)  | 3.45 bbox-h/s -> rejected | **0.24** -> accepted |
| track 1 -> 47, #10 -> #21 (WRONG)    | 3.07 bbox-h/s -> rejected | **6.41** -> still rejected |

Raw, the correct and the wrong continuation are indistinguishable (3.45 vs 3.07 -- the wrong one
even looks *slower*). Compensated, they separate by ~27x. That is the whole point of this module:
it does not loosen any threshold, it removes the camera term that was making the thresholds
meaningless.

Reuses `src.goal.detect._fit_affine_transform` verbatim (the same
goodFeaturesToTrack/calcOpticalFlowPyrLK/estimateAffinePartial2D pattern and the same
`configs/profile.yaml: motion` keys Stage 0.5 already uses, ADR-11) rather than introducing a
second, drifting motion estimator.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.types import Take
from src.common.video import decode_frames

logger = get_logger(__name__)


class TakeCameraMotion(BaseModel):
    """Cumulative camera transform per sampled instant of ONE take.

    `cumulative[i]` is the flattened 2x3 affine mapping the take's FIRST sampled frame into the
    frame at `times[i]`. Mapping a point the other way (into that shared reference frame) is what
    makes two positions at different instants directly comparable -- see `to_reference`.
    """

    take_id: int
    times: list[float]
    cumulative: list[list[float]]

    def _nearest_index(self, t: float) -> int | None:
        if not self.times:
            return None
        idx = int(np.argmin([abs(x - t) for x in self.times]))
        # a sample further away than the sampling period itself is not evidence about this instant
        if len(self.times) > 1:
            period = abs(self.times[1] - self.times[0])
            if abs(self.times[idx] - t) > max(period * 2.0, 0.1):
                return None
        return idx

    def to_reference(self, t: float, x: float, y: float) -> tuple[float, float] | None:
        """Map image point `(x, y)` observed at time `t` into the take's own reference frame
        (its first sampled frame). `None` when this instant has no usable motion estimate --
        callers must then fall back to the raw, uncompensated comparison rather than guess.
        """
        idx = self._nearest_index(t)
        if idx is None:
            return None
        mat = np.array(self.cumulative[idx], dtype=np.float64).reshape(2, 3)
        full = np.vstack([mat, [0.0, 0.0, 1.0]])
        try:
            inv = np.linalg.inv(full)
        except np.linalg.LinAlgError:
            return None
        px, py, _ = inv @ np.array([x, y, 1.0])
        return float(px), float(py)


class CameraMotionReport(BaseModel):
    """Cache payload: one `TakeCameraMotion` per take that could be estimated."""

    video: str
    takes: list[TakeCameraMotion]

    def by_take(self) -> dict[int, TakeCameraMotion]:
        return {t.take_id: t for t in self.takes}


def estimate_take_camera_motion(
    video_path: str,
    take: Take,
    motion_cfg: dict,
    fps: float,
    scale_width: int | None,
    use_nvdec: bool = True,
) -> TakeCameraMotion:
    """Accumulate frame-to-frame camera transforms across one take at `fps`.

    A pair whose transform can't be fit (too few trackable features -- `_fit_affine_transform`
    returns `None`, its own honest-failure convention) carries the previous cumulative transform
    forward unchanged: that says "no measured camera motion here", which is the neutral,
    non-fabricating choice, and `to_reference` still degrades to the raw comparison for any
    instant it cannot cover.
    """
    from src.goal.detect import _fit_affine_transform

    times: list[float] = []
    cumulative: list[list[float]] = []
    running = np.eye(3, dtype=np.float64)
    prev_gray = None
    n_failed = 0

    for _idx, t, frame in decode_frames(
        video_path,
        fps=fps,
        start=take.t_start,
        end=take.t_end,
        scale_width=scale_width,
        use_nvdec=use_nvdec,
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mat = _fit_affine_transform(prev_gray, gray, motion_cfg)
            if mat is None:
                n_failed += 1
            else:
                running = np.vstack([mat, [0.0, 0.0, 1.0]]) @ running
        prev_gray = gray
        times.append(t)
        cumulative.append([float(v) for v in running[:2].reshape(-1)])

    if n_failed:
        logger.info(
            "camera_motion: take=%d %d/%d frame pair(s) had too few trackable features "
            "(carried the previous transform forward -- an honest 'no measured motion here')",
            take.id,
            n_failed,
            max(len(times) - 1, 0),
        )
    return TakeCameraMotion(take_id=take.id, times=times, cumulative=cumulative)


def estimate_camera_motion_for_video(
    video_path: str,
    takes: list[Take],
    motion_cfg: dict,
    fps: float,
    scale_width: int | None,
    use_nvdec: bool = True,
) -> CameraMotionReport:
    """`estimate_take_camera_motion` for every take, as one cacheable report."""
    results = []
    for take in takes:
        results.append(
            estimate_take_camera_motion(
                video_path, take, motion_cfg, fps, scale_width, use_nvdec=use_nvdec
            )
        )
    return CameraMotionReport(video=str(video_path), takes=results)


def compensated_jump(
    a_t: float,
    a_cx: float,
    a_cy: float,
    b_t: float,
    b_cx: float,
    b_cy: float,
    mean_height: float,
    motion: TakeCameraMotion | None,
) -> tuple[float, bool]:
    """`(jump_in_bbox_heights, was_compensated)` between two box centres at different instants.

    Falls back to the raw image-space distance (and `was_compensated=False`) whenever there is no
    usable motion estimate for either instant -- never silently pretends a compensation happened.
    """
    if mean_height <= 0:
        return float("inf"), False
    if motion is not None:
        pa = motion.to_reference(a_t, a_cx, a_cy)
        pb = motion.to_reference(b_t, b_cx, b_cy)
        if pa is not None and pb is not None:
            dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            return dist / mean_height, True
    dist = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
    return dist / mean_height, False
