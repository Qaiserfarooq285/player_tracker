"""Pure-logic unit tests for Stage 2 (CLAUDE.md task spec: no GPU in unit tests).

Covers: arrow-component selection + tip geometry on synthetic images (`src/detect/overlay_mask.py`),
arrow/watermark overlap-rejection math, and ball gap-interpolation incl. `interpolated` flagging
and gaps longer than `max_gap_frames` (`src/detect/ball.py`). Thresholds are read from the real
`configs/detect.yaml` (matching `tests/test_profiler.py`'s pattern) so these tests exercise the
actual configured masking behaviour, not a hand-copied duplicate of it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, Detection, DetectionClass
from src.detect.ball import interpolate_ball_gaps
from src.detect.overlay_mask import (
    ArrowHint,
    _find_tip,
    filter_arrow_overlap,
    filter_watermark,
    find_arrow,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def detect_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "detect.yaml")


@pytest.fixture(scope="module")
def masking_cfg(detect_config) -> dict:
    return detect_config["masking"]


@pytest.fixture(scope="module")
def ball_cfg(detect_config) -> dict:
    return detect_config["ball_interpolation"]


def _pure_red_frame(height: int, width: int) -> np.ndarray:
    """A frame that is red enough everywhere to satisfy the configured HSV threshold (BGR)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (20, 20, 220)  # BGR: strong red, low blue/green -> high S, high V, hue ~0
    return frame


# ---------------------------------------------------------------------------
# find_arrow: component selection
# ---------------------------------------------------------------------------


def test_find_arrow_returns_none_with_no_red(masking_cfg):
    frame = np.full((200, 200, 3), (30, 90, 30), dtype=np.uint8)  # green, not red
    assert find_arrow(frame, masking_cfg) is None


def test_find_arrow_returns_none_below_area_threshold(masking_cfg):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    small_side = max(1, int((masking_cfg["arrow_min_area_px"] * 0.5) ** 0.5))
    frame[10 : 10 + small_side, 10 : 10 + small_side] = (20, 20, 220)
    assert find_arrow(frame, masking_cfg) is None


def test_find_arrow_returns_none_above_max_area(masking_cfg):
    """Regression test (found running Stage 2 on clip4): a large bulk red surface (measured: a
    running-track/infield surface, ~128k px @ 1920w) must NOT be mistaken for a bigger arrow.
    """
    max_area = masking_cfg["arrow_max_area_px"]
    side = int((max_area * 2) ** 0.5)
    frame = np.zeros((side + 20, side + 20, 3), dtype=np.uint8)
    frame[10 : 10 + side, 10 : 10 + side] = (20, 20, 220)
    assert find_arrow(frame, masking_cfg) is None


def test_find_arrow_returns_none_for_wider_than_tall_component(masking_cfg):
    """Regression test (found running Stage 2 on clip4 a second time): a distant red-brick house
    in the background (~126x78px, height:width~0.62, small enough to clear arrow_max_area_px)
    must NOT be mistaken for the arrow — every genuine arrow measured is taller than wide.
    """
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Width 200, height 60 -> height:width = 0.3, well under arrow_min_height_width_ratio (1.0),
    # and area (12000) comfortably inside [arrow_min_area_px, arrow_max_area_px].
    frame[10:70, 10:210] = (20, 20, 220)
    assert find_arrow(frame, masking_cfg) is None


def test_find_arrow_accepts_tall_component_within_ratio(masking_cfg):
    frame = np.zeros((300, 100, 3), dtype=np.uint8)
    # Width 60, height 120 -> height:width = 2.0, comfortably above the 1.0 minimum.
    frame[10:130, 10:70] = (20, 20, 220)
    hint = find_arrow(frame, masking_cfg)
    assert hint is not None


def test_find_arrow_picks_the_largest_component(masking_cfg):
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    # A small red speck, well below the area threshold.
    frame[5:8, 5:8] = (20, 20, 220)
    # A big red blob, comfortably above the area threshold.
    big_side = int((masking_cfg["arrow_min_area_px"] * 4) ** 0.5)
    frame[100 : 100 + big_side, 100 : 100 + big_side] = (20, 20, 220)

    hint = find_arrow(frame, masking_cfg, frame_index=7, t=1.5)
    assert hint is not None
    assert hint.frame_index == 7
    assert hint.t == 1.5
    assert hint.area_px >= masking_cfg["arrow_min_area_px"]
    # the speck (top-left) must not have been selected
    assert hint.bbox.x1 >= 50


# ---------------------------------------------------------------------------
# _find_tip: PCA extreme points, disambiguated by LOCAL PIXEL DENSITY (not distance from
# centroid — that heuristic was tried first and measured WRONG on a real frame, see below).
# ---------------------------------------------------------------------------

_PROBE_RADIUS_PX = 12.0
_TIE_MARGIN = 0.15


def _synthetic_arrow_mask(
    height: int,
    width: int,
    tail_pt: tuple[int, int],
    head_pt: tuple[int, int],
    shaft_thickness: int = 3,
    head_radius: int = 15,
) -> np.ndarray:
    """A synthetic arrow-like boolean mask: a thin shaft between the two points, plus a solid
    filled-circle "head" at `head_pt` — isolates `_find_tip`'s local-density disambiguation from
    HSV thresholding/connected-components (tested separately above).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(mask, tail_pt, head_pt, 1, shaft_thickness)
    cv2.circle(mask, head_pt, head_radius, 1, -1)
    return mask.astype(bool)


def _pts_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def test_find_tip_picks_the_solid_head_when_head_is_at_the_bottom():
    mask = _synthetic_arrow_mask(220, 60, tail_pt=(30, 10), head_pt=(30, 190))
    tip_x, tip_y = _find_tip(_pts_from_mask(mask), mask, _PROBE_RADIUS_PX, _TIE_MARGIN)
    assert tip_y > 150  # near the solid head at the bottom, not the thin tail at the top


def test_find_tip_picks_the_solid_head_when_head_is_at_the_top():
    mask = _synthetic_arrow_mask(220, 60, tail_pt=(30, 190), head_pt=(30, 10))
    tip_x, tip_y = _find_tip(_pts_from_mask(mask), mask, _PROBE_RADIUS_PX, _TIE_MARGIN)
    assert tip_y < 70  # near the solid head at the top, not the thin tail at the bottom


def test_find_tip_uniform_thickness_line_picks_one_of_the_two_endpoints():
    # No density difference along a uniform-thickness straight line -> still a deterministic,
    # valid endpoint (not some interior point).
    mask = np.zeros((20, 220), dtype=np.uint8)
    cv2.line(mask, (10, 10), (200, 10), 1, 3)
    mask = mask.astype(bool)
    tip_x, _tip_y = _find_tip(_pts_from_mask(mask), mask, _PROBE_RADIUS_PX, _TIE_MARGIN)
    assert tip_x <= 13 or tip_x >= 197


def test_find_tip_regression_real_clip2_frame_t0_9s(masking_cfg):
    """Locks in the 2026-08-24 fix: `find_arrow` on the actual `clip2 77.mp4` frame at t=0.9s
    used to return a tip at the TAIL (top of the bbox, ~300px from the player) instead of the
    arrowhead (bottom-left of the bbox, right next to player #77) — verified visually. The tip
    must now fall in the lower half of the component's own bbox.
    """
    from src.common.video import decode_frames

    video_path = REPO_ROOT / "input" / "clip2 77.mp4"
    if not video_path.exists():
        pytest.skip("input/clip2 77.mp4 not present in this environment")

    frame = None
    for _idx, _t, f in decode_frames(
        video_path, fps=None, start=0.9, end=0.95, scale_width=1920, use_nvdec=True
    ):
        frame = f
        break
    assert frame is not None

    hint = find_arrow(frame, masking_cfg, frame_index=9, t=0.9)
    assert hint is not None
    bbox_mid_y = (hint.bbox.y1 + hint.bbox.y2) / 2
    assert hint.tip_y > bbox_mid_y  # arrowhead is in the lower half, not the upper (tail) half


# ---------------------------------------------------------------------------
# filter_arrow_overlap / filter_watermark
# ---------------------------------------------------------------------------


def _det(x1, y1, x2, y2, t=0.0, frame_index=0) -> Detection:
    return Detection(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        cls=DetectionClass.PLAYER,
        conf=0.8,
        frame_index=frame_index,
        t=t,
    )


def test_filter_arrow_overlap_drops_heavily_overlapping_detection(masking_cfg):
    arrow = ArrowHint(
        frame_index=0,
        t=0.0,
        bbox=BBox(x1=0, y1=0, x2=100, y2=100),
        area_px=10000,
        tip_x=90,
        tip_y=90,
    )
    # fully inside the arrow bbox -> 100% overlap of its own area
    swallowed = _det(10, 10, 20, 20)
    # far away, no overlap at all
    clear = _det(500, 500, 520, 520)

    kept = filter_arrow_overlap([swallowed, clear], arrow, masking_cfg)
    assert kept == [clear]


def test_filter_arrow_overlap_noop_when_no_arrow(masking_cfg):
    dets = [_det(0, 0, 10, 10)]
    assert filter_arrow_overlap(dets, None, masking_cfg) == dets


def test_filter_watermark_drops_centre_inside_roi(masking_cfg):
    frame_shape = (1000, 2000, 3)  # height, width
    x1r, y1r, x2r, y2r = masking_cfg["watermark_roi_relative"]
    # a detection whose centre sits in the middle of the configured ROI
    cx = (x1r + x2r) / 2 * frame_shape[1]
    cy = (y1r + y2r) / 2 * frame_shape[0]
    inside = _det(cx - 5, cy - 5, cx + 5, cy + 5)
    outside = _det(0, 0, 10, 10)

    kept = filter_watermark([inside, outside], frame_shape, masking_cfg)
    assert kept == [outside]


# ---------------------------------------------------------------------------
# interpolate_ball_gaps
# ---------------------------------------------------------------------------


def _ball(frame_index, t, x, conf=0.5) -> BallDetection:
    return BallDetection(
        bbox=BBox(x1=x, y1=x, x2=x + 5, y2=x + 5), conf=conf, frame_index=frame_index, t=t
    )


def test_interpolate_fills_short_gap_and_flags_interpolated(ball_cfg):
    observed = [_ball(0, 0.0, 0.0, conf=0.8), _ball(5, 0.5, 50.0, conf=0.6)]
    sampled = [(i, i * 0.1) for i in range(6)]

    result = interpolate_ball_gaps(observed, sampled, ball_cfg)
    by_frame = {d.frame_index: d for d in result}

    assert by_frame[0].interpolated is False
    assert by_frame[5].interpolated is False
    for i in range(1, 5):
        assert by_frame[i].interpolated is True
        # linear interpolation between x=0 (frame 0) and x=50 (frame 5)
        expected_x = 50.0 * (i / 5)
        assert by_frame[i].bbox.x1 == pytest.approx(expected_x)
        # confidence penalised below both bracketing observed confidences
        assert by_frame[i].conf < min(0.8, 0.6)


def test_interpolate_does_not_fill_gap_longer_than_max(ball_cfg):
    max_gap = ball_cfg["max_gap_frames"]
    far = max_gap + 5
    observed = [_ball(0, 0.0, 0.0), _ball(far, far * 0.1, 100.0)]
    sampled = [(i, i * 0.1) for i in range(far + 1)]

    result = interpolate_ball_gaps(observed, sampled, ball_cfg)
    by_frame = {d.frame_index: d for d in result}

    assert set(by_frame.keys()) == {0, far}  # nothing in between got fabricated


def test_interpolate_does_not_extrapolate_before_first_or_after_last(ball_cfg):
    observed = [_ball(3, 0.3, 10.0)]
    sampled = [(i, i * 0.1) for i in range(7)]

    result = interpolate_ball_gaps(observed, sampled, ball_cfg)
    by_frame = {d.frame_index: d for d in result}

    assert set(by_frame.keys()) == {3}  # frames 0-2 and 4-6 are unfillable (no bracketing pair)


def test_interpolate_empty_detections_returns_empty(ball_cfg):
    assert interpolate_ball_gaps([], [(0, 0.0), (1, 0.1)], ball_cfg) == []
