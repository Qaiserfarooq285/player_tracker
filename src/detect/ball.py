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

from src.common.logging import DropCounter, get_logger
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


def filter_implausible_ball_jumps(
    observed: list[BallDetection],
    frame_width: float,
    plausibility_cfg: dict,
    drops: DropCounter | None = None,
) -> list[BallDetection]:
    """Reject an observed `BallDetection` that implies a physically impossible jump from the last
    *accepted* observation (Stage 1 plausibility filter, CLAUDE.md §5/§10).

    `detect_ball_sahi` keeps only the single highest-confidence candidate per frame and has no
    access to neighbouring frames at all (see its own docstring/module docstring above) — so
    nothing upstream of this function can consider continuity. This is the first point in the
    pipeline that does. A rejected observation simply becomes a gap; `interpolate_ball_gaps`
    (the very next step in `src/detect/run.py`) already bridges a gap correctly — it just cannot
    currently tell a real miss from a garbage detection, because nothing feeds it that distinction.

    Speed is measured in the SAME "frame-widths per second" unit `src/events/shots.py::
    ball_speed_series` already uses (centroid displacement / dt / `frame_width` — self-scaling
    across clips decoded at different widths, still never a metric unit, ADR-6).

    MEASURED 2026-08-31 on `input/clip1 43.mp4`'s cached raw `ball_detections.parquet` (334 SAHI
    observations pre-interpolation, at the pipeline's 1920px decode width): frame-to-frame speed
    has NO clean bimodal gap the way e.g. `configs/detect.yaml: arrow_min_area_px` does — it is a
    continuous tail from 0.002 up to 24.21 fw/s. Inspecting the raw `(t, cx, cy)` series explains
    why: the detector alternates — sometimes every single sampled frame — between the real ball
    (drifting smoothly around cx~850-900px, cy~580px early in the clip) and a SEPARATE, ALSO
    smoothly-drifting false positive ~235px higher in frame (cy~343-347px), almost certainly
    clip1's painted American-football yard-line numbers (CLAUDE.md §3.2(4)) sliding with this
    clip's own camera pan (§3.2's highest-motion clip, 12.70 px/frame). E.g. index 0->1 jumps
    cx 893.0->1751.7 in one 33ms sample (24.2 fw/s — clearly a detector swap, not ball motion),
    then 1->2 continues smoothly WITHIN that false cluster (1751.7->1740.2, 0.18 fw/s — internally
    plausible, just the wrong object), before index 3 jumps straight back to the real cluster
    (883.8, only 0.15 fw/s from the true anchor two samples back). `configs/events.yaml: shot.`
    already documents its own real accepted shot candidates peaking at 4.46-8.30 fw/s (measured on
    clip2). `max_plausible_speed_fw_per_s` is set to 10.0: comfortably above every real ball motion
    this project has ever measured as plausible, comfortably below the 11.55-24.21 fw/s band
    clip1's detector-swap jumps occupy (the smallest of clip1's top-40 frame-to-frame jumps is
    11.55 fw/s) — there is no single clean threshold that perfectly separates the two on this
    clip's continuous distribution, so 10.0 is a margin-based choice between two independently
    measured reference points, not a bimodal cliff; documented as such rather than overclaimed.

    Self-correcting re-seed (Golden Rule 5 — never let one bad early sample permanently blind the
    filter): a run of `reseed_run_length` consecutive REJECTED candidates that are each plausible
    relative to the PREVIOUS rejected candidate (not the stale anchor) is treated as evidence that
    the anchor — not these points — was the error, and the whole run is promoted to `accepted`,
    with tracking continuing from its last point. A run that never reaches that length, followed
    by a candidate that resumes plausibly from the ORIGINAL anchor, is simply dropped as noise (the
    anchor was right all along). This specific failure mode — a bad first anchor that would
    otherwise reject every real subsequent observation forever — is exercised directly in
    `tests/test_ball.py::test_filter_implausible_ball_jumps_reseeds_after_bad_anchor`. On real
    clip1/clip2 data at these thresholds this path never actually triggers (0 reseeds either
    clip — the true ball reasserts itself well inside `reseed_run_length` samples every time); it
    exists as a safety net for inputs where the true trajectory takes longer to reassert itself.

    Honest limitation: with no appearance model, this cannot always tell WHICH of two
    simultaneously-plausible, internally-smooth tracks is the real ball (see clip1's yard-number
    false cluster above, which is itself locally smooth) — it only guarantees a genuinely sustained
    trajectory is not permanently rejected just because it disagrees with one earlier anchor point.
    """
    if frame_width <= 0 or len(observed) < 2:
        return list(observed)

    max_speed = plausibility_cfg["max_plausible_speed_fw_per_s"]
    reseed_run_length = plausibility_cfg["reseed_run_length"]

    ordered = sorted(observed, key=lambda d: d.t)

    def _speed(a: BallDetection, b: BallDetection) -> float:
        dt = b.t - a.t
        if dt <= 0:
            return 0.0 if (a.bbox.cx, a.bbox.cy) == (b.bbox.cx, b.bbox.cy) else float("inf")
        dx = b.bbox.cx - a.bbox.cx
        dy = b.bbox.cy - a.bbox.cy
        return ((dx * dx + dy * dy) ** 0.5) / dt / frame_width

    accepted: list[BallDetection] = [ordered[0]]
    seed_confirmed = False  # True once `accepted[0]` has been directly corroborated by at least
    # one later plausible sample -- see the bad-anchor case below.
    pending: list[BallDetection] = []  # a run of consecutive rejections, checked for mutual
    # self-consistency below — see "Self-correcting re-seed" in the docstring above.

    for det in ordered[1:]:
        if _speed(accepted[-1], det) <= max_speed:
            if pending:
                if drops is not None:
                    drops.drop("ball_implausible_jump", len(pending))
                pending = []
            accepted.append(det)
            seed_confirmed = True
            continue

        if pending and _speed(pending[-1], det) <= max_speed:
            pending.append(det)
        else:
            if pending:
                if drops is not None:
                    drops.drop("ball_implausible_jump", len(pending))
            pending = [det]

        if len(pending) >= reseed_run_length:
            if not seed_confirmed and len(accepted) == 1:
                # The very first observation was NEVER corroborated by anything before this
                # sustained, mutually-consistent run showed up -- that is evidence the seed
                # itself was the error (e.g. the video opens on a stray false positive), not
                # these `reseed_run_length` points. Discard the seed rather than keeping a
                # one-sample "trajectory" the rest of the clip never agreed with.
                if drops is not None:
                    drops.drop("ball_implausible_jump", 1)
                accepted = list(pending)
            else:
                accepted.extend(pending)
            seed_confirmed = True
            pending = []

    if pending:
        if drops is not None:
            drops.drop("ball_implausible_jump", len(pending))

    return accepted


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
