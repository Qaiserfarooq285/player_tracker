"""Stage 2 — burned-in red-arrow + "veo" watermark masking (CLAUDE.md §3.2 consequence 1).

The owner's Veo footage carries two editorial graphics baked into the pixels (not real objects):
a red arrow pointing at the target player for the first few seconds of `clip2`/`clip4`, and a
"veo" watermark in the bottom-right corner of every clip. Both (a) can trigger spurious detector
output and (b) the arrow specifically occludes/deforms the box of the real player underneath it.
This module finds the arrow (`find_arrow`) and filters detections that fall inside either graphic
(`filter_arrow_overlap`, `filter_watermark`) — every drop is counted via `DropCounter`
(CLAUDE.md §10: "log everything dropped"). The arrow's tip is *also* kept (not just used to
reject) as a free, jersey-number-free prior for manual player selection later (Stage 5+) — it
never reads a jersey number, so keeping it does not violate Golden Rule 7.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.logging import DropCounter, get_logger
from src.common.types import BBox, Detection

logger = get_logger(__name__)


class ArrowHint(BaseModel):
    """One frame's burned-in red-arrow graphic, if any (CLAUDE.md §3.2 consequence 1).

    `frame_index` is the *sampled*-sequence index (0, 1, 2, ... in decode order at Stage 2's own
    `fps_sample`), not a native-video frame number — see `Detection.frame_index`'s docstring for
    the same distinction; `t` (seconds) is what any cross-artifact timestamp comparison must use.
    `tip_x`/`tip_y` is the arrowhead location — the end pointing at the player, not the tail
    (see `_find_tip`'s docstring for why); `bbox`/`area_px` describe the whole connected
    component (arrowhead + curved shaft) and are what Stage 2's own detection-overlap rejection
    uses.
    """

    frame_index: int
    t: float
    bbox: BBox
    area_px: int
    tip_x: float
    tip_y: float


def _find_tip(
    pts: np.ndarray, component_mask: np.ndarray, probe_radius_px: float, tie_margin: float
) -> tuple[float, float]:
    """Return the arrow component's tip: whichever end of its principal axis is the ARROWHEAD.

    **Corrected 2026-08-24** (previously picked whichever end sat furthest from the component's
    centroid — measured wrong on `clip2` @ t=0.9s: it landed on the tail at the top of the curve,
    ~300px from the actual player, not the arrowhead pointing at them; the distance-from-centroid
    relationship between shaft and head is shape/frame-dependent, not reliably one-directional).

    The two ends of the component's principal axis are still found the same way (PCA projection
    extremes), but disambiguated by **local pixel density**, not distance: an arrowhead is a
    short, solid, filled triangular blob, while the shaft/tail is a thin curved stroke — a small
    disc of radius `probe_radius_px` centred on the arrowhead end contains substantially more
    foreground pixels than the same-size disc centred on the tail end. Ties within `tie_margin`
    (relative) fall back to *density* (mass / the disc's actually-in-bounds area) rather than raw
    count, so a disc partially clipped by the frame border doesn't get mistaken for "thin" purely
    from running out of frame.

    `pts` is an `(N, 2)` array of the component's own `(x, y)` pixel coordinates; `component_mask`
    is the full-frame boolean mask of just this connected component (same source `pts` was drawn
    from, via `np.where`).
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ principal
    i_max, i_min = int(np.argmax(proj)), int(np.argmin(proj))
    candidates = (pts[i_max], pts[i_min])

    def local_mass(point: np.ndarray) -> tuple[int, int]:
        cx, cy = point
        height, width = component_mask.shape
        x0 = max(0, int(cx - probe_radius_px))
        x1 = min(width, int(cx + probe_radius_px) + 1)
        y0 = max(0, int(cy - probe_radius_px))
        y1 = min(height, int(cy + probe_radius_px) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= probe_radius_px**2
        region = component_mask[y0:y1, x0:x1]
        count = int(np.count_nonzero(region & disc))
        valid_area = int(np.count_nonzero(disc))
        return count, valid_area

    counts_areas = [local_mass(c) for c in candidates]
    counts = [c for c, _a in counts_areas]
    densities = [c / a if a > 0 else 0.0 for c, a in counts_areas]

    hi, lo = (0, 1) if counts[0] >= counts[1] else (1, 0)
    if counts[hi] > 0 and counts[lo] >= counts[hi] * (1.0 - tie_margin):
        winner = hi if densities[hi] >= densities[lo] else lo
    else:
        winner = hi

    tip = candidates[winner]
    return float(tip[0]), float(tip[1])


def find_arrow(
    frame_bgr: np.ndarray, masking_cfg: dict, frame_index: int = 0, t: float = 0.0
) -> ArrowHint | None:
    """Find the burned-in red-arrow graphic in one BGR frame, if present.

    Thresholds are the ones measured in CLAUDE.md §3.2(1) / `configs/detect.yaml: masking`:
    HSV `H<=h_low_max OR H>=h_high_min` (hue wraps at 0/180 in OpenCV's 8-bit HSV), `S>s_min`,
    `V>v_min`, then the single largest 8-connected component between `arrow_min_area_px` and
    `arrow_max_area_px`. Returns `None` when no component clears the area window — either because
    there's no red at all (most frames — the arrow is only on-screen for the opening few seconds
    of a clip), or because the only red present is a bulk surface, not the arrow (measured on
    `clip4`: a running-track/infield venue's red-orange surface is ~18x larger than any genuine
    arrow — `arrow_max_area_px` exists specifically to reject it, see `configs/detect.yaml`).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_cfg = masking_cfg["arrow_hsv"]
    mask = (
        ((h <= hsv_cfg["h_low_max"]) | (h >= hsv_cfg["h_high_min"]))
        & (s > hsv_cfg["s_min"])
        & (v > hsv_cfg["v_min"])
    ).astype(np.uint8)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:  # label 0 is background; no foreground components at all
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    area = int(stats[best_idx, cv2.CC_STAT_AREA])
    if area < masking_cfg["arrow_min_area_px"]:
        return None
    if area > masking_cfg["arrow_max_area_px"]:
        # Some other bulk red surface (measured: a running-track/infield surface on clip4's
        # t~50-110s venues), not a bigger/closer arrow — see configs/detect.yaml's
        # arrow_max_area_px comment for the measured false-positive this guards against.
        return None

    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    h_box = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    if h_box < masking_cfg["arrow_min_height_width_ratio"] * w:
        # Some other wider-than-tall red surface (measured: a distant red-brick house on clip4's
        # t~90-110s venue — small enough to clear arrow_max_area_px, but a completely different
        # shape). Every genuine arrow measured so far (clip2 + clip4) is a mostly-vertical curved
        # stroke, height:width ~1.4-2.0; both measured false positives (running track, house) are
        # WIDER than tall (~0.09 and ~0.62) — see configs/detect.yaml for the numbers.
        return None

    component_mask = labels == best_idx
    ys, xs = np.where(component_mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    probe_radius_px = masking_cfg["arrow_head_probe_radius_frac"] * max(w, h_box)
    tip_x, tip_y = _find_tip(
        pts, component_mask, probe_radius_px, masking_cfg["arrow_head_tie_margin"]
    )

    return ArrowHint(
        frame_index=frame_index,
        t=t,
        bbox=BBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h_box)),
        area_px=area,
        tip_x=tip_x,
        tip_y=tip_y,
    )


def _overlap_fraction(bbox: BBox, other: BBox) -> float:
    """Fraction of `bbox`'s own area covered by `other`.

    Deliberately asymmetric rather than IoU: a small player box mostly swallowed by the (much
    larger) arrow bbox should be rejected outright even though a *symmetric* IoU would read small
    just because the arrow's own bbox area dominates the union.
    """
    ix1, iy1 = max(bbox.x1, other.x1), max(bbox.y1, other.y1)
    ix2, iy2 = min(bbox.x2, other.x2), min(bbox.y2, other.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if bbox.area <= 0:
        return 0.0
    return inter / bbox.area


def filter_arrow_overlap(
    detections: list[Detection],
    arrow: ArrowHint | None,
    masking_cfg: dict,
    drops: DropCounter | None = None,
) -> list[Detection]:
    """Drop detections whose bbox overlaps `arrow`'s bbox above `arrow_overlap_reject`.

    Counted in `drops` as `dropped_arrow_overlap` when a `DropCounter` is supplied. A no-op
    (returns `detections` unchanged) when `arrow` is `None`.
    """
    if arrow is None or not detections:
        return detections
    threshold = masking_cfg["arrow_overlap_reject"]
    kept, n_dropped = [], 0
    for d in detections:
        if _overlap_fraction(d.bbox, arrow.bbox) > threshold:
            n_dropped += 1
        else:
            kept.append(d)
    if n_dropped and drops is not None:
        drops.drop("dropped_arrow_overlap", n_dropped)
    return kept


def is_in_watermark_roi(bbox: BBox, frame_shape: tuple[int, int], masking_cfg: dict) -> bool:
    """True if `bbox`'s centre falls inside the configured relative watermark ROI.

    `frame_shape` is `(height, width, ...)` as returned by `ndarray.shape` for a decoded frame.
    """
    height, width = frame_shape[0], frame_shape[1]
    x1r, y1r, x2r, y2r = masking_cfg["watermark_roi_relative"]
    x1, y1, x2, y2 = x1r * width, y1r * height, x2r * width, y2r * height
    return x1 <= bbox.cx <= x2 and y1 <= bbox.cy <= y2


def filter_watermark(
    detections: list[Detection],
    frame_shape: tuple[int, int],
    masking_cfg: dict,
    drops: DropCounter | None = None,
) -> list[Detection]:
    """Drop detections whose bbox centre falls inside the "veo" watermark ROI.

    Counted in `drops` as `dropped_watermark` when a `DropCounter` is supplied.
    """
    if not detections:
        return detections
    kept, n_dropped = [], 0
    for d in detections:
        if is_in_watermark_roi(d.bbox, frame_shape, masking_cfg):
            n_dropped += 1
        else:
            kept.append(d)
    if n_dropped and drops is not None:
        drops.drop("dropped_watermark", n_dropped)
    return kept
