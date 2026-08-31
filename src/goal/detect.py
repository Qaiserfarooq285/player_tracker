"""Stage C — general-purpose goal-structure detection ("streamed-gathering-treehouse" plan).

Finds the real soccer-goal structure (crossbar + posts) in a take, so `src/events/goals.py`'s
"ball into the white box = goal" rule (ADR-20) can key off an ACTUALLY-DETECTED goal region
instead of only a human-hand-drawn one (`configs/goal_region.yaml`). No training, no new
dependency — OpenCV (already a dependency, CLAUDE.md §7) + the existing Gemini VLM client
(`src/common/gemini.py`). Every threshold used here lives in `configs/goal_structure.yaml`,
carrying its own measured-vs-reasoned provenance comment (read that file in full before touching
this module) — no magic numbers in this file, CLAUDE.md §10/§12.

**Two-stage design, both measured necessary** (`configs/goal_structure.yaml: vlm`'s own comment):
full-frame classical CV alone FAILS on this project's own footage (locks onto painted
American-football yard lines / background buildings; a temporal-median-background variant also
fails under real camera motion, smearing thin posts). So:
  1. **Localize** (`_localize_goal_vlm`) — Gemini roughly locates the goal in a sampled frame
     (generalizes across venue/angle/lighting where classical CV alone cannot).
  2. **Refine** (`_refine_goal_topology_native`) — classical CV, AT NATIVE RESOLUTION (not the
     decode/detect stage's own downscaled frames — measured necessary, thin goal structures need
     full pixel detail), finds the exact crossbar/post topology inside the VLM's proposed region,
     with RF-DETR player/goalkeeper boxes structurally excluded from the mask BEFORE line-fitting
     (guarantees a goalkeeper standing in the goal mouth can never become the "goal" box,
     independent of any shape-heuristic tuning).
  3. **Aggregate** (`_aggregate_topologies`) across `vlm.frames_per_take` sampled frames, rejecting
     outliers by IoU against the running median — Golden Rule 5: a single-frame coincidence never
     becomes a take's goal-structure estimate; `aggregation.min_corroborating_frames` sets the
     floor, and a take that never corroborates emits NOTHING (`localize_goal_structure_for_take`
     returns `None`), never a hallucinated box.
  4. **Track across the take** (`goal_bbox_at`) — for a near-static camera one anchor covers the
     whole take; when this take's own MEASURED motion score (not the coarse static/panning label,
     ADR-11) exceeds `tracking.reestimate_motion_score_threshold`, the anchor's bbox is propagated
     to a later query instant via a single `cv2.estimateAffinePartial2D` hop (the SAME tool/pattern
     `src/pipeline/profiler.py::_anchor_translation_magnitude` uses for camera-motion estimation) —
     cheap (pure CV, no VLM cost) and bounded (one hop per query, not a per-frame tracking loop).

Cached per video at `work/<slug>/goal/goal_structures.json` (`StageCache`, same convention as
`work/<slug>/identity.json` / `work/<slug>/profile.json`). `GoalStructure`/`GoalStructureReport`
are declared LOCALLY (not `src/common/types.py`) — the same stage-local-artifact precedent as
`src/detect/overlay_mask.py::ArrowHint`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.gemini import call_gemini_vision
from src.common.io import StageCache, load_json, save_json, work_dir_for
from src.common.logging import get_logger
from src.common.types import BBox, Take, Track
from src.common.video import decode_frames, probe
from src.pipeline.profiler import compute_motion_score

logger = get_logger(__name__)

__all__ = [
    "GoalStructure",
    "GoalStructureReport",
    "detect_goal_structures_for_video",
    "goal_bbox_at",
    "localize_goal_structure_for_take",
    "parse_vlm_goal_response",
    "take_motion_score",
]


# ---------------------------------------------------------------------------
# stage-local artifact models (ArrowHint precedent — not src/common/types.py)
# ---------------------------------------------------------------------------


class GoalStructure(BaseModel):
    """One take's tracked goal-frame structure estimate.

    `bbox`/`crossbar`/`posts` are all NATIVE-resolution, FULL-FRAME pixel coordinates (matching
    ADR-15's own native-decode convention for anything requiring fine pixel detail). `confidence`
    is the plain corroboration ratio (`n_corroborating_frames / n_frames_sampled`) — simple and
    directly auditable, deliberately not blended with an extra unmeasured "base rate" constant.
    """

    take_id: int
    t_anchor: float  # timestamp (seconds, video-timeline) this estimate is anchored to
    bbox: BBox
    crossbar: tuple[float, float, float, float]  # x1, y1, x2, y2
    posts: list[tuple[float, float, float, float]]
    confidence: float
    n_corroborating_frames: int
    n_frames_sampled: int
    source: str = "vlm_localize_cv_refine"


class GoalStructureReport(BaseModel):
    """Stage C's whole-video output (`work/<slug>/goal/goal_structures.json`). A take with no
    confident structure simply has no entry in `takes` — Golden Rule 5: absence is honest, never a
    fabricated placeholder."""

    video: str
    takes: list[GoalStructure] = []


# ---------------------------------------------------------------------------
# VLM localization
# ---------------------------------------------------------------------------

_GOAL_BOX_PATTERN = re.compile(r"GOAL\s*=\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")


def parse_vlm_goal_response(
    text: str | None, none_literal: str
) -> tuple[float, float, float, float] | None:
    """Parse a `GOAL=x1,y1,x2,y2` VLM response into a `[0, 1]`-fraction box, or `None` for an
    honest abstention (`GOAL=<none_literal>`) or a malformed/unparseable response — never a guess
    (Golden Rule 5).

    **MEASURED coordinate convention** (`configs/goal_structure.yaml: vlm.coordinate_convention`):
    despite an earlier prompt asking for a `[0, 1]` fraction, Gemini consistently returned
    coordinates on a 0-1000 integer scale regardless of prompt wording — this function always
    divides by 1000.0, never trusts a raw `[0, 1]` assumption.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.upper() == f"GOAL={none_literal}".upper():
        return None
    match = _GOAL_BOX_PATTERN.search(stripped)
    if not match:
        return None
    x1, y1, x2, y2 = (int(g) for g in match.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    # MEASURED: 0-1000 convention, not 0-1 -- see this function's own docstring.
    return x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0


def _localize_goal_vlm(
    frame_bgr: np.ndarray, api_key: str, vlm_cfg: dict
) -> tuple[float, float, float, float] | None:
    """One Gemini call attempting to localize the goal in `frame_bgr`. `None` on any failure
    (network/quota exhaustion/malformed response) or an honest "no goal visible" abstention —
    callers must never treat a `None` here as evidence of anything (Golden Rule 5)."""
    text, error, model_used = call_gemini_vision(vlm_cfg["prompt"], frame_bgr, api_key, vlm_cfg)
    if error is not None:
        logger.warning("goal-structure VLM localization call failed: %s", error)
        return None
    box = parse_vlm_goal_response(text, vlm_cfg["none_response"])
    if box is None:
        logger.info("goal-structure VLM (model=%s): no goal visible in this frame", model_used)
    return box


# ---------------------------------------------------------------------------
# classical CV refinement at native resolution
# ---------------------------------------------------------------------------


def _hsv_white_mask(frame_bgr: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    """Low-saturation/high-value ("white paint") binary mask — same HSV-threshold-then-shape-filter
    pattern as `src/detect/overlay_mask.py::find_arrow`, MEASURED thresholds
    (`configs/goal_structure.yaml: cv_refine.hsv_s_max`/`hsv_v_min`)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return (((s < s_max) & (v > v_min)).astype(np.uint8)) * 255


def _exclude_player_boxes(
    mask: np.ndarray, boxes: list[BBox], roi_x1: float, roi_y1: float
) -> np.ndarray:
    """Zero out `mask` pixels that fall inside any of `boxes` (native-resolution, FULL-FRAME
    coordinates, offset into the ROI's own local coordinate space by `roi_x1`/`roi_y1`) — this is
    what guarantees a goalkeeper standing in the goal mouth can never become the "goal" box,
    independent of any shape-heuristic tuning (structural exclusion, not a shape heuristic)."""
    if not boxes:
        return mask
    out = mask.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1 = int(max(0, box.x1 - roi_x1))
        y1 = int(max(0, box.y1 - roi_y1))
        x2 = int(min(w, box.x2 - roi_x1))
        y2 = int(min(h, box.y2 - roi_y1))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 0
    return out


def _segment_angle_from_horizontal_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle of the line `(x1,y1)-(x2,y2)` from horizontal, folded into `[0, 90]` (a line and its
    exact reverse/vertical-flip all read the same "how tilted is this" answer)."""
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return 180.0 - angle if angle > 90.0 else angle


def _fit_goal_topology(mask: np.ndarray, cv_cfg: dict) -> dict[str, Any] | None:
    """Fit a crossbar+post topology to a binary white-paint `mask` (ROI-local pixel coordinates,
    players already excluded) via `cv2.createLineSegmentDetector` — MEASURED thresholds
    (`configs/goal_structure.yaml: cv_refine`, from `clip1 43.mp4` @ t=7.0s, native 4K).

    Returns `{"crossbar": (x1,y1,x2,y2), "posts": [(x1,y1,x2,y2), ...]}` in the SAME ROI-local
    coordinates the caller passed in, or `None` when no plausible topology is found (no horizontal
    candidate, no post attaching to it, or an implausible aspect ratio) — Golden Rule 5, never a
    guessed shape.
    """
    lsd = cv2.createLineSegmentDetector(0)
    detected = lsd.detect(mask)
    lines = detected[0] if detected is not None else None
    if lines is None or len(lines) == 0:
        return None

    roi_height = mask.shape[0]
    min_len = cv_cfg["min_segment_length_frac_of_roi_height"] * roi_height

    horiz_candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    vert_candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for line in lines:
        # cv2.createLineSegmentDetector().detect() has returned shape (N, 4) directly (verified
        # empirically this session) rather than the (N, 1, 4) shape other OpenCV line detectors
        # (e.g. HoughLinesP) use -- `.reshape(-1)` is robust to either, rather than assuming one.
        x1, y1, x2, y2 = (float(v) for v in np.asarray(line).reshape(-1)[:4])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        angle = _segment_angle_from_horizontal_deg(x1, y1, x2, y2)
        if angle <= cv_cfg["horizontal_max_angle_deg"]:
            horiz_candidates.append((length, (x1, y1, x2, y2)))
        elif angle >= cv_cfg["vertical_min_angle_deg"]:
            vert_candidates.append((length, (x1, y1, x2, y2)))

    if not horiz_candidates:
        return None
    horiz_candidates.sort(key=lambda c: c[0], reverse=True)
    crossbar_len, (cx1, cy1, cx2, cy2) = horiz_candidates[0]
    if cx1 > cx2:  # normalise left-to-right
        cx1, cy1, cx2, cy2 = cx2, cy2, cx1, cy1

    x_tol = cv_cfg["post_endpoint_x_tolerance_frac_of_crossbar_len"] * crossbar_len
    y_tol = cv_cfg["post_top_y_tolerance_frac_of_crossbar_len"] * crossbar_len

    posts: list[tuple[float, float, float, float]] = []
    for _length, (vx1, vy1, vx2, vy2) in sorted(vert_candidates, key=lambda c: c[0], reverse=True):
        top_x, top_y = (vx1, vy1) if vy1 <= vy2 else (vx2, vy2)  # smaller y = higher on screen
        near_left = abs(top_x - cx1) <= x_tol and abs(top_y - cy1) <= y_tol
        near_right = abs(top_x - cx2) <= x_tol and abs(top_y - cy2) <= y_tol
        if near_left or near_right:
            posts.append((vx1, vy1, vx2, vy2))
        if len(posts) >= 2:
            break

    if not posts:
        return None

    max_post_height = max(abs(y2 - y1) for _x1, y1, _x2, y2 in posts)
    if max_post_height <= 0:
        return None
    aspect = crossbar_len / max_post_height
    if not (cv_cfg["aspect_ratio_min"] <= aspect <= cv_cfg["aspect_ratio_max"]):
        return None

    return {"crossbar": (cx1, cy1, cx2, cy2), "posts": posts}


def _refine_goal_topology_native(
    native_frame: np.ndarray,
    vlm_box_frac: tuple[float, float, float, float],
    player_boxes_native: list[BBox],
    cv_cfg: dict,
) -> dict[str, Any] | None:
    """Refine the VLM's rough `[0, 1]`-fraction box into an exact crossbar+post topology at NATIVE
    resolution — MEASURED necessary (`configs/goal_structure.yaml: vlm`'s own "full-frame
    classical CV fails" note: thin goal structures need full pixel detail, not the decode/detect
    stage's downscaled frames).

    Crops a padded ROI around the VLM's box (`cv_cfg['roi_pad_frac']`), excludes player/goalkeeper
    pixels, tries the LOOSE HSV mask first and falls back to the TIGHT mask only if loose finds no
    plausible topology (`configs/goal_structure.yaml: cv_refine`'s own measured loose/tight
    comparison — loose is the adopted default, tight is the explicit fallback for excess
    glare/noise). Returns crossbar/posts in FULL-FRAME native pixel coordinates, or `None`.
    """
    h, w = native_frame.shape[:2]
    x1f, y1f, x2f, y2f = vlm_box_frac
    pad_x = (x2f - x1f) * cv_cfg["roi_pad_frac"]
    pad_y = (y2f - y1f) * cv_cfg["roi_pad_frac"]
    rx1 = int(max(0.0, x1f - pad_x) * w)
    ry1 = int(max(0.0, y1f - pad_y) * h)
    rx2 = int(min(1.0, x2f + pad_x) * w)
    ry2 = int(min(1.0, y2f + pad_y) * h)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None
    roi = native_frame[ry1:ry2, rx1:rx2]

    for s_key, v_key in (("hsv_s_max", "hsv_v_min"), ("hsv_s_max_tight", "hsv_v_min_tight")):
        mask = _hsv_white_mask(roi, cv_cfg[s_key], cv_cfg[v_key])
        mask = _exclude_player_boxes(mask, player_boxes_native, rx1, ry1)
        topology = _fit_goal_topology(mask, cv_cfg)
        if topology is not None:
            cbx1, cby1, cbx2, cby2 = topology["crossbar"]
            crossbar = (cbx1 + rx1, cby1 + ry1, cbx2 + rx1, cby2 + ry1)
            posts = [
                (px1 + rx1, py1 + ry1, px2 + rx1, py2 + ry1)
                for px1, py1, px2, py2 in topology["posts"]
            ]
            return {"crossbar": crossbar, "posts": posts}
    return None


def _topology_bbox(topology: dict[str, Any]) -> BBox:
    xs = [topology["crossbar"][0], topology["crossbar"][2]]
    ys = [topology["crossbar"][1], topology["crossbar"][3]]
    for x1, y1, x2, y2 in topology["posts"]:
        xs += [x1, x2]
        ys += [y1, y2]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _aggregate_topologies(
    candidates: list[tuple[float, dict[str, Any]]], agg_cfg: dict
) -> dict[str, Any] | None:
    """Combine per-frame `(t, topology)` candidates into ONE take-level estimate, rejecting
    outliers by IoU against the running median bbox before requiring
    `agg_cfg['min_corroborating_frames']` inliers to remain — Golden Rule 5: a single-frame
    coincidence is not enough to track a goal for a whole take; `None` when the bar isn't cleared.
    """
    if not candidates:
        return None
    scored = [(_topology_bbox(topo), t, topo) for t, topo in candidates]
    ref = BBox(
        x1=float(np.median([b.x1 for b, _t, _topo in scored])),
        y1=float(np.median([b.y1 for b, _t, _topo in scored])),
        x2=float(np.median([b.x2 for b, _t, _topo in scored])),
        y2=float(np.median([b.y2 for b, _t, _topo in scored])),
    )
    inliers = [
        item for item in scored if _iou(item[0], ref) >= agg_cfg["outlier_reject_iou_threshold"]
    ]
    if len(inliers) < agg_cfg["min_corroborating_frames"]:
        return None

    final_bbox = BBox(
        x1=float(np.median([b.x1 for b, _t, _topo in inliers])),
        y1=float(np.median([b.y1 for b, _t, _topo in inliers])),
        x2=float(np.median([b.x2 for b, _t, _topo in inliers])),
        y2=float(np.median([b.y2 for b, _t, _topo in inliers])),
    )
    # Crossbar/post ENDPOINTS come from whichever real inlier sits closest to the final median
    # bbox -- a component-wise median of endpoints across DIFFERENT detections could produce a
    # geometrically inconsistent shape (a "crossbar" and "posts" that never belonged to the same
    # frame's own detection), so the representative topology is always a REAL one, not a synthetic
    # blend.
    representative = max(inliers, key=lambda item: _iou(item[0], final_bbox))
    _rep_bbox, rep_t, rep_topology = representative
    return {
        "bbox": final_bbox,
        "crossbar": rep_topology["crossbar"],
        "posts": rep_topology["posts"],
        "t_anchor": rep_t,
        "n_corroborating_frames": len(inliers),
        "n_frames_sampled": len(candidates),
    }


# ---------------------------------------------------------------------------
# per-take orchestration
# ---------------------------------------------------------------------------


def _scale_bbox_to_native(bbox: BBox, scale_x: float, scale_y: float) -> BBox:
    return BBox(
        x1=bbox.x1 * scale_x, y1=bbox.y1 * scale_y, x2=bbox.x2 * scale_x, y2=bbox.y2 * scale_y
    )


def _player_boxes_near(
    take_tracks: list[Track], t: float, tolerance_s: float, scale_x: float, scale_y: float
) -> list[BBox]:
    boxes: list[BBox] = []
    for tr in take_tracks:
        box = next((b for b in tr.boxes if abs(b.t - t) <= tolerance_s), None)
        if box is not None:
            boxes.append(_scale_bbox_to_native(box.bbox, scale_x, scale_y))
    return boxes


def localize_goal_structure_for_take(
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    detect_frame_width: float,
    detect_frame_height: float,
    goal_structure_cfg: dict,
    gemini_api_key: str | None,
    use_nvdec: bool = True,
) -> GoalStructure | None:
    """Full Stage C pipeline for ONE take: sample `vlm.frames_per_take` frames (evenly spread,
    decoded at NATIVE resolution in one pass), localize+refine+exclude-players on each (bounded to
    `vlm.max_escalations_per_take` Gemini calls total), aggregate. Returns `None` — never a
    hallucinated box — when no `GEMINI_API_KEY` is available (the VLM step is required; full-frame
    classical CV alone is MEASURED to fail on this project's footage, see module docstring) or
    when aggregation doesn't clear `aggregation.min_corroborating_frames`.
    """
    vlm_cfg = goal_structure_cfg["vlm"]
    cv_cfg = goal_structure_cfg["cv_refine"]
    agg_cfg = goal_structure_cfg["aggregation"]

    if not gemini_api_key:
        logger.warning(
            "goal-structure detection for take=%d: no GEMINI_API_KEY -- skipped entirely "
            "(VLM localization is required; full-frame classical CV alone is measured to fail "
            "on this footage, see configs/goal_structure.yaml: vlm's own note)",
            take.id,
        )
        return None

    duration = max(take.t_end - take.t_start, 1e-3)
    n_frames = max(1, vlm_cfg["frames_per_take"])
    fps_for_sampling = max(n_frames / duration, 0.01)

    frames: list[tuple[float, np.ndarray]] = []
    for _idx, t, frame in decode_frames(
        video_path,
        fps=fps_for_sampling,
        start=take.t_start,
        end=take.t_end,
        scale_width=None,  # native resolution -- measured necessary, see module docstring
        use_nvdec=use_nvdec,
    ):
        frames.append((t, frame.copy()))
    frames = frames[:n_frames]

    if not frames:
        return None

    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]
    scale_x = native_w / detect_frame_width if detect_frame_width else 1.0
    scale_y = native_h / detect_frame_height if detect_frame_height else 1.0
    tolerance_s = cv_cfg["player_box_match_tolerance_s"]

    max_calls = vlm_cfg["max_escalations_per_take"]
    n_calls = 0
    candidates: list[tuple[float, dict[str, Any]]] = []
    for t, frame in frames:
        if n_calls >= max_calls:
            break
        n_calls += 1
        vlm_box = _localize_goal_vlm(frame, gemini_api_key, vlm_cfg)
        if vlm_box is None:
            continue
        player_boxes = _player_boxes_near(take_tracks, t, tolerance_s, scale_x, scale_y)
        topology = _refine_goal_topology_native(frame, vlm_box, player_boxes, cv_cfg)
        if topology is not None:
            candidates.append((t, topology))

    aggregated = _aggregate_topologies(candidates, agg_cfg)
    if aggregated is None:
        logger.info(
            "goal-structure detection for take=%d: %d/%d sampled frame(s) corroborated -- below "
            "min_corroborating_frames=%d, emitting nothing (Golden Rule 5)",
            take.id,
            len(candidates),
            len(frames),
            agg_cfg["min_corroborating_frames"],
        )
        return None

    confidence = aggregated["n_corroborating_frames"] / max(aggregated["n_frames_sampled"], 1)
    return GoalStructure(
        take_id=take.id,
        t_anchor=aggregated["t_anchor"],
        bbox=aggregated["bbox"],
        crossbar=aggregated["crossbar"],
        posts=aggregated["posts"],
        confidence=confidence,
        n_corroborating_frames=aggregated["n_corroborating_frames"],
        n_frames_sampled=aggregated["n_frames_sampled"],
    )


# ---------------------------------------------------------------------------
# top-level orchestration + cache
# ---------------------------------------------------------------------------


def detect_goal_structures_for_video(
    video_path: str | Path,
    takes: list[Take],
    tracks_by_take: dict[int, list[Track]],
    detect_frame_width: float,
    detect_frame_height: float,
    goal_structure_cfg: dict,
    gemini_api_key: str | None,
    work_root: str | Path = "work",
    use_nvdec: bool = True,
) -> GoalStructureReport:
    """Run Stage C for every take of `video_path`, cached at
    `work/<slug>/goal/goal_structures.json` (`StageCache`, same convention as
    `work/<slug>/identity.json` / `work/<slug>/profile.json`)."""
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    goal_dir = work_dir / "goal"
    out_path = goal_dir / "goal_structures.json"

    cache_config = {
        "goal_structure_cfg": goal_structure_cfg,
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
        "n_takes": len(takes),
        "have_gemini_key": bool(gemini_api_key),
    }
    cache = StageCache(out_path, cache_config, stage="goal_structure")
    if cache.hit():
        return GoalStructureReport.model_validate(load_json(out_path))

    results: list[GoalStructure] = []
    for take in takes:
        structure = localize_goal_structure_for_take(
            video_path,
            take,
            tracks_by_take.get(take.id, []),
            detect_frame_width,
            detect_frame_height,
            goal_structure_cfg,
            gemini_api_key,
            use_nvdec=use_nvdec,
        )
        if structure is not None:
            results.append(structure)
            logger.info(
                "goal-structure take=%d: bbox=%s confidence=%.2f (%d/%d frames)",
                take.id,
                structure.bbox,
                structure.confidence,
                structure.n_corroborating_frames,
                structure.n_frames_sampled,
            )
        else:
            logger.info("goal-structure take=%d: not available", take.id)

    report = GoalStructureReport(video=str(video_path), takes=results)
    save_json(report, out_path)
    cache.write_meta()
    return report


# ---------------------------------------------------------------------------
# tracking across the take (single affine hop, motion-score-gated)
# ---------------------------------------------------------------------------


def take_motion_score(
    video_path: str | Path,
    take: Take,
    hardware_cfg: dict,
    profile_cfg: dict,
    use_nvdec: bool = True,
) -> float:
    """This TAKE's own measured camera-motion score (px/native-frame) -- a thin wrapper around
    `src/pipeline/profiler.py::compute_motion_score`'s `start`/`end` windowing, added for exactly
    this caller (ADR-11: `goal_bbox_at`'s re-localization cadence decision must consume the take's
    own measured signal, never the whole-video aggregate `RunProfile.motion_score` a multi-take
    video like `clip4`/`jordan_thomas_highlight_video` would otherwise blend across venues)."""
    return compute_motion_score(
        video_path,
        hardware_cfg,
        profile_cfg,
        use_nvdec=use_nvdec,
        start=take.t_start,
        end=take.t_end,
    )


def _fit_affine_transform(prev_gray: np.ndarray, curr_gray: np.ndarray, motion_cfg: dict) -> Any:
    """Sparse-feature RANSAC-affine transform between two frames -- the SAME
    goodFeaturesToTrack/calcOpticalFlowPyrLK/estimateAffinePartial2D pattern and `motion_cfg` keys
    as `src/pipeline/profiler.py::_anchor_translation_magnitude`, kept as a separate function here
    (not a shared refactor of that already-tested one) because this caller needs the full 2x3
    transform MATRIX to warp box corners, not just its translation magnitude. Returns `None` when
    too few features are found/tracked/agree on a model -- same honest-failure convention as its
    profiler.py counterpart."""
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=motion_cfg["max_corners"],
        qualityLevel=motion_cfg["quality_level"],
        minDistance=motion_cfg["min_distance"],
        blockSize=motion_cfg["block_size"],
    )
    if pts is None or len(pts) < motion_cfg["min_tracked_points"]:
        return None
    win = motion_cfg["lk_win_size"]
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, winSize=(win, win), maxLevel=motion_cfg["lk_max_level"]
    )
    ok = status.reshape(-1).astype(bool)
    p0, p1 = pts[ok], next_pts[ok]
    if len(p0) < motion_cfg["min_tracked_points"]:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=motion_cfg["ransac_reproj_threshold"]
    )
    return matrix


def _warp_bbox(bbox: BBox, matrix: np.ndarray) -> BBox:
    corners = np.array(
        [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]],
        dtype=np.float64,
    )
    ones = np.ones((4, 1))
    homogeneous = np.hstack([corners, ones])
    transformed = homogeneous @ matrix.T
    xs, ys = transformed[:, 0], transformed[:, 1]
    return BBox(x1=float(xs.min()), y1=float(ys.min()), x2=float(xs.max()), y2=float(ys.max()))


def _grab_frame_at(
    video_path: str | Path, t: float, take: Take, window_s: float, use_nvdec: bool
) -> np.ndarray | None:
    start = max(take.t_start, t - window_s)
    end = min(take.t_end, t + window_s)
    best: tuple[float, np.ndarray] | None = None
    for _idx, frame_t, frame in decode_frames(
        video_path, fps=2, start=start, end=end, scale_width=None, use_nvdec=use_nvdec
    ):
        dt = abs(frame_t - t)
        if best is None or dt < best[0]:
            best = (dt, frame.copy())
    return best[1] if best is not None else None


def goal_bbox_at(
    structure: GoalStructure,
    video_path: str | Path,
    t_query: float,
    take: Take,
    motion_score: float,
    tracking_cfg: dict,
    motion_cfg: dict,
    use_nvdec: bool = True,
) -> BBox:
    """The tracked goal bbox at `t_query`. Below `tracking_cfg
    ['reestimate_motion_score_threshold']` (or when `t_query == structure.t_anchor`), the anchor's
    own bbox is returned unchanged -- a near-static camera doesn't move the goal's on-screen
    position enough to matter (ADR-11's own lesson: consume the MEASURED signal, never a fixed
    interval or the coarse static/panning label). Above the threshold, the anchor bbox is
    propagated to `t_query` via a SINGLE `cv2.estimateAffinePartial2D` hop between the anchor frame
    and the query frame.

    When the affine fit itself fails (too few tracked points -- e.g. a hard cut inside the take, or
    genuinely featureless frames), the anchor bbox is returned UNCHANGED rather than `None`: a real
    goal WAS found earlier in this take, and reporting its last-known position is a smaller, more
    honest error than reporting no goal at all for a take known to have one -- a documented
    approximation, never a silent one (Golden Rule 5).
    """
    if abs(t_query - structure.t_anchor) < 1e-6:
        return structure.bbox
    if motion_score <= tracking_cfg["reestimate_motion_score_threshold"]:
        return structure.bbox

    window_s = tracking_cfg["frame_grab_window_s"]
    anchor_frame = _grab_frame_at(video_path, structure.t_anchor, take, window_s, use_nvdec)
    query_frame = _grab_frame_at(video_path, t_query, take, window_s, use_nvdec)
    if anchor_frame is None or query_frame is None:
        return structure.bbox

    anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
    matrix = _fit_affine_transform(anchor_gray, query_gray, motion_cfg)
    if matrix is None:
        return structure.bbox
    return _warp_bbox(structure.bbox, matrix)
