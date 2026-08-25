"""Stage 2 — SAHI-tiled ball detection + short-gap interpolation (CLAUDE.md §5: "ball is the
weak spot"; §3.1: SAHI tiling restricted to the ball class).

The ball is small relative to a 1920-wide detection frame and gets missed by a single full-frame
pass. `detect_ball_sahi` slices the frame into overlapping tiles (`sahi.slicing.get_slice_bboxes`
— reusing SAHI's own tiling geometry rather than reimplementing it), runs the already-loaded
Stage 2 detector on the whole batch of tiles in one `predict()` call (measured: for a 1920x1080
frame at `tile_size=640`/`overlap_ratio=0.2` this is 8 *uniformly-sized* tiles — no edge-clipping
special case needed), translates tile-local boxes back to full-frame pixel coordinates, and merges
cross-tile duplicates (a ball straddling a tile border gets detected in both) with NMS. Only ever
returns the single highest-confidence ball in the frame (there is one ball).

`interpolate_ball_gaps` then fills short frame-to-frame gaps between *observed* detections by
linear interpolation — flagged `interpolated=True` (never presented as observed, Golden Rule 5) —
and leaves gaps longer than `max_gap_frames` unfilled rather than guessing where the ball went.
"""

from __future__ import annotations

from bisect import bisect_left

import cv2
import numpy as np
import supervision as sv
import torch
from sahi.slicing import get_slice_bboxes

from src.common.logging import get_logger
from src.common.types import BallDetection, BBox, DetectionClass
from src.detect.detector import DetectorHandle, _snap_to_block

logger = get_logger(__name__)


def _ball_class_name(handle: DetectorHandle) -> str:
    """The detector's own class-name string that maps to `DetectionClass.BALL`."""
    for name, cls in handle.class_map.items():
        if cls == DetectionClass.BALL:
            return name
    raise ValueError("detector class_map has no BALL entry")


def detect_ball_sahi(
    handle: DetectorHandle,
    frame_bgr: np.ndarray,
    frame_index: int,
    detect_cfg: dict,
    hardware_cfg: dict,
    t: float = 0.0,
) -> BallDetection | None:
    """Tiled ball detection over one frame. Returns `None` if no tile detects a ball."""
    sahi_cfg = hardware_cfg["stages"]["ball"]["sahi"]
    tile_conf = detect_cfg["sahi"]["tile_conf_threshold"]
    tile_size = sahi_cfg["tile_size"]
    height, width = frame_bgr.shape[:2]

    tile_boxes = get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=sahi_cfg["overlap_ratio"],
        overlap_width_ratio=sahi_cfg["overlap_ratio"],
    )

    tiles_rgb: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in tile_boxes:
        tile = frame_bgr[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        tiles_rgb.append(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        offsets.append((x1, y1))
    if not tiles_rgb:
        return None

    # A single common `shape` (derived from the configured tile_size, not each crop's actual
    # pixel size) lets every tile in the batch share one forward pass regardless of edge-tile
    # clipping — `predict()` resizes each image to `shape` internally either way.
    shape = (_snap_to_block(tile_size, handle.block_size),) * 2
    ball_name = _ball_class_name(handle)

    if handle.use_fp16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            results = handle.model.predict(tiles_rgb, threshold=tile_conf, shape=shape)
    else:
        results = handle.model.predict(tiles_rgb, threshold=tile_conf, shape=shape)
    if not isinstance(results, list):
        results = [results]

    all_xyxy: list[list[float]] = []
    all_conf: list[float] = []
    for (ox, oy), dets in zip(offsets, results, strict=True):
        names = dets.data.get("class_name", [])
        for i in range(len(dets)):
            if str(names[i]) != ball_name:
                continue
            bx1, by1, bx2, by2 = (float(c) for c in dets.xyxy[i])
            all_xyxy.append([bx1 + ox, by1 + oy, bx2 + ox, by2 + oy])
            all_conf.append(float(dets.confidence[i]))

    if not all_xyxy:
        return None

    merged = sv.Detections(
        xyxy=np.array(all_xyxy, dtype=np.float32),
        confidence=np.array(all_conf, dtype=np.float32),
        class_id=np.zeros(len(all_xyxy), dtype=int),
    )
    if len(merged) > 1:
        merged = merged.with_nms(
            threshold=detect_cfg["sahi"]["tile_nms_iou_threshold"], class_agnostic=True
        )

    best_i = int(np.argmax(merged.confidence))
    x1, y1, x2, y2 = (float(c) for c in merged.xyxy[best_i])
    return BallDetection(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        conf=float(merged.confidence[best_i]),
        frame_index=frame_index,
        t=t,
        interpolated=False,
    )


def interpolate_ball_gaps(
    detections: list[BallDetection],
    sampled_frames: list[tuple[int, float]],
    ball_cfg: dict,
) -> list[BallDetection]:
    """Fill short gaps between observed `BallDetection`s by linear interpolation.

    `sampled_frames` is the full ordered list of `(frame_index, t)` the ball stage sampled (some
    with an observed detection, some without) — gaps are measured in *sampled-frame* counts
    (`configs/detect.yaml: ball_interpolation.max_gap_frames`), not raw video frames. A gap longer
    than `max_gap_frames`, or a missing frame before the first / after the last observation, is
    left unfilled — Golden Rule 5 forbids fabricating an unbounded guess at where the ball went.
    Interpolated rows get `interpolated=True` and a confidence penalised via
    `interpolated_conf_penalty` (CLAUDE.md task spec: "never present interpolated as observed").
    """
    by_frame = {d.frame_index: d for d in detections}
    if not by_frame:
        return []
    observed = sorted(by_frame.keys())
    max_gap = ball_cfg["max_gap_frames"]
    penalty = ball_cfg["interpolated_conf_penalty"]

    out: list[BallDetection] = []
    for frame_index, t in sampled_frames:
        if frame_index in by_frame:
            out.append(by_frame[frame_index])
            continue

        pos = bisect_left(observed, frame_index)
        if pos == 0 or pos == len(observed):
            continue  # before the first or after the last observation — do not extrapolate
        prev_idx, next_idx = observed[pos - 1], observed[pos]
        gap = next_idx - prev_idx
        if gap > max_gap:
            continue

        prev_det, next_det = by_frame[prev_idx], by_frame[next_idx]
        frac = (frame_index - prev_idx) / gap
        bbox = BBox(
            x1=prev_det.bbox.x1 + frac * (next_det.bbox.x1 - prev_det.bbox.x1),
            y1=prev_det.bbox.y1 + frac * (next_det.bbox.y1 - prev_det.bbox.y1),
            x2=prev_det.bbox.x2 + frac * (next_det.bbox.x2 - prev_det.bbox.x2),
            y2=prev_det.bbox.y2 + frac * (next_det.bbox.y2 - prev_det.bbox.y2),
        )
        conf = (prev_det.conf + next_det.conf) / 2.0 * penalty
        out.append(
            BallDetection(bbox=bbox, conf=conf, frame_index=frame_index, t=t, interpolated=True)
        )
    return out
