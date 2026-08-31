"""Pure-logic unit tests for Stage C (`src/goal/detect.py`, "streamed-gathering-treehouse" plan)
-- no GPU/live network required. Covers: the VLM response parser (0-1000 convention, malformed/
NONE input), goal-topology fitting on synthetic frames (real `cv2.createLineSegmentDetector`,
mocked nothing -- these are genuine CV calls on synthetic images), player-box exclusion, IoU-based
aggregation with outlier rejection, and the motion-score-gated single-affine-hop tracking query.
"""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np

import src.goal.detect as goal_mod
from src.common.io import load_yaml
from src.common.types import BBox, Take
from src.goal.detect import (
    GoalStructure,
    goal_bbox_at,
    parse_vlm_goal_response,
)

REPO_ROOT_CFG = "configs/goal_structure.yaml"


def _cv_cfg() -> dict:
    return load_yaml(REPO_ROOT_CFG)["cv_refine"]


def _agg_cfg() -> dict:
    return load_yaml(REPO_ROOT_CFG)["aggregation"]


def _tracking_cfg() -> dict:
    return load_yaml(REPO_ROOT_CFG)["tracking"]


# ---------------------------------------------------------------------------
# parse_vlm_goal_response -- 0-1000 convention, malformed/NONE input (Golden Rule 5)
# ---------------------------------------------------------------------------


def test_parse_vlm_goal_response_divides_by_1000_not_1():
    # MEASURED convention (configs/goal_structure.yaml: vlm.coordinate_convention): Gemini returns
    # integers on a 0-1000 scale, e.g. "GOAL=580,317,659,372" on a real 1600x900-equivalent frame.
    box = parse_vlm_goal_response("GOAL=580,317,659,372", "NONE")
    assert box == (0.580, 0.317, 0.659, 0.372)


def test_parse_vlm_goal_response_none_literal_is_honest_abstention():
    assert parse_vlm_goal_response("GOAL=NONE", "NONE") is None
    assert parse_vlm_goal_response("goal=none", "NONE") is None  # case-insensitive


def test_parse_vlm_goal_response_malformed_text_returns_none():
    assert parse_vlm_goal_response("I can see a goal in the image", "NONE") is None
    assert parse_vlm_goal_response("", "NONE") is None
    assert parse_vlm_goal_response(None, "NONE") is None


def test_parse_vlm_goal_response_degenerate_box_rejected():
    # x2 <= x1 -- a genuinely malformed/degenerate box must never be accepted as real geometry.
    assert parse_vlm_goal_response("GOAL=500,300,500,400", "NONE") is None
    assert parse_vlm_goal_response("GOAL=600,300,500,400", "NONE") is None


def test_parse_vlm_goal_response_tolerates_extra_whitespace():
    box = parse_vlm_goal_response("GOAL = 100 , 200 , 300 , 400", "NONE")
    assert box == (0.1, 0.2, 0.3, 0.4)


# ---------------------------------------------------------------------------
# _fit_goal_topology -- real cv2.createLineSegmentDetector on synthetic frames
# ---------------------------------------------------------------------------


def _draw_goal_mask(size: int = 300) -> np.ndarray:
    """A synthetic crossbar+two-posts topology: crossbar (50,50)-(250,50) [len 200, angle 0], left
    post (50,50)-(50,150) and right post (250,50)-(250,150) [len 100 each, angle 90] -- same
    proportions (aspect ~2.0) as the real MEASURED clip1 sample (347.1/108.8 = 3.19), comfortably
    inside `cv_refine.aspect_ratio_min/max` (0.8-6.0)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.line(mask, (50, 50), (250, 50), 255, 3)
    cv2.line(mask, (50, 50), (50, 150), 255, 3)
    cv2.line(mask, (250, 50), (250, 150), 255, 3)
    return mask


def test_fit_goal_topology_finds_crossbar_and_both_posts_on_synthetic_goal():
    mask = _draw_goal_mask()
    result = goal_mod._fit_goal_topology(mask, _cv_cfg())
    assert result is not None
    cx1, cy1, cx2, cy2 = result["crossbar"]
    assert abs(cy1 - 50) < 5 and abs(cy2 - 50) < 5  # near-horizontal, at y~50
    assert abs((cx2 - cx1) - 200) < 10  # crossbar length ~200px
    assert len(result["posts"]) == 2


def test_fit_goal_topology_none_on_empty_mask():
    empty = np.zeros((300, 300), dtype=np.uint8)
    assert goal_mod._fit_goal_topology(empty, _cv_cfg()) is None


def test_fit_goal_topology_none_when_only_crossbar_no_posts():
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.line(mask, (50, 50), (250, 50), 255, 3)
    assert goal_mod._fit_goal_topology(mask, _cv_cfg()) is None


def test_fit_goal_topology_rejects_implausible_aspect_ratio():
    # a crossbar 200px wide with a post only 10px tall -- aspect ratio 20.0, way outside
    # aspect_ratio_max=6.0 -- must be rejected even though a horizontal + a "vertical" line exist.
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.line(mask, (50, 50), (250, 50), 255, 3)
    cv2.line(mask, (50, 50), (50, 60), 255, 3)
    cfg = dict(_cv_cfg())
    cfg["min_segment_length_frac_of_roi_height"] = 0.01  # let the short post clear the size gate
    assert goal_mod._fit_goal_topology(mask, cfg) is None


# ---------------------------------------------------------------------------
# _exclude_player_boxes -- structural exclusion (never a shape-heuristic-only guarantee)
# ---------------------------------------------------------------------------


def test_exclude_player_boxes_zeroes_the_covered_region():
    mask = np.full((100, 100), 255, dtype=np.uint8)
    out = goal_mod._exclude_player_boxes(
        mask, [BBox(x1=10, y1=10, x2=50, y2=50)], roi_x1=0, roi_y1=0
    )
    assert np.count_nonzero(out[10:50, 10:50]) == 0
    assert np.count_nonzero(out[60:90, 60:90]) > 0  # untouched elsewhere


def test_exclude_player_boxes_offsets_by_roi_origin():
    # a player box in FULL-FRAME coordinates must be correctly re-based into ROI-local coordinates.
    mask = np.full((50, 50), 255, dtype=np.uint8)
    # full-frame box at (110,110)-(130,130); ROI itself starts at full-frame (100,100) -- so the
    # box should land at ROI-local (10,10)-(30,30).
    out = goal_mod._exclude_player_boxes(
        mask, [BBox(x1=110, y1=110, x2=130, y2=130)], roi_x1=100, roi_y1=100
    )
    assert np.count_nonzero(out[10:30, 10:30]) == 0
    assert np.count_nonzero(out[0:5, 0:5]) > 0


def test_exclude_player_boxes_no_boxes_is_a_no_op():
    mask = np.full((20, 20), 255, dtype=np.uint8)
    out = goal_mod._exclude_player_boxes(mask, [], roi_x1=0, roi_y1=0)
    assert np.array_equal(out, mask)


# ---------------------------------------------------------------------------
# _aggregate_topologies -- corroboration + IoU outlier rejection (Golden Rule 5)
# ---------------------------------------------------------------------------


def _topo(x1, y1, x2, y2) -> dict:
    return {"crossbar": (x1, y1, x2, y2), "posts": [(x1, y1, x1, y2), (x2, y1, x2, y2)]}


def test_aggregate_topologies_none_below_min_corroborating_frames():
    # only 1 candidate; default min_corroborating_frames=2 -- must emit nothing.
    candidates = [(1.0, _topo(0, 0, 100, 50))]
    assert goal_mod._aggregate_topologies(candidates, _agg_cfg()) is None


def test_aggregate_topologies_corroborates_two_agreeing_frames():
    candidates = [(1.0, _topo(0, 0, 100, 50)), (2.0, _topo(2, 1, 102, 51))]
    result = goal_mod._aggregate_topologies(candidates, _agg_cfg())
    assert result is not None
    assert result["n_corroborating_frames"] == 2
    assert result["n_frames_sampled"] == 2


def test_aggregate_topologies_rejects_a_genuine_outlier():
    # two agreeing frames + one wildly different box (a false CV detection elsewhere in the ROI)
    # -- the outlier must be excluded, and corroboration still passes on the two real ones.
    candidates = [
        (1.0, _topo(0, 0, 100, 50)),
        (2.0, _topo(2, 1, 102, 51)),
        (3.0, _topo(500, 500, 600, 550)),  # far away, near-zero IoU against the other two
    ]
    result = goal_mod._aggregate_topologies(candidates, _agg_cfg())
    assert result is not None
    assert result["n_corroborating_frames"] == 2
    assert result["n_frames_sampled"] == 3


def test_aggregate_topologies_empty_input_is_none():
    assert goal_mod._aggregate_topologies([], _agg_cfg()) is None


# ---------------------------------------------------------------------------
# goal_bbox_at -- motion-score-gated single affine hop (ADR-11: consume the measured signal)
# ---------------------------------------------------------------------------


def _structure(t_anchor: float = 5.0) -> GoalStructure:
    return GoalStructure(
        take_id=0,
        t_anchor=t_anchor,
        bbox=BBox(x1=100, y1=100, x2=200, y2=150),
        crossbar=(100, 100, 200, 100),
        posts=[(100, 100, 100, 150), (200, 100, 200, 150)],
        confidence=1.0,
        n_corroborating_frames=2,
        n_frames_sampled=2,
    )


def test_goal_bbox_at_returns_anchor_unchanged_at_the_anchor_instant():
    structure = _structure(t_anchor=5.0)
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    result = goal_bbox_at(
        structure,
        "fake.mp4",
        5.0,
        take,
        motion_score=999.0,
        tracking_cfg=_tracking_cfg(),
        motion_cfg={},
    )
    assert result == structure.bbox


def test_goal_bbox_at_returns_anchor_unchanged_below_motion_threshold():
    structure = _structure(t_anchor=5.0)
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    tracking_cfg = _tracking_cfg()
    below = tracking_cfg["reestimate_motion_score_threshold"] - 0.1
    result = goal_bbox_at(
        structure,
        "fake.mp4",
        8.0,
        take,
        motion_score=below,
        tracking_cfg=tracking_cfg,
        motion_cfg={},
    )
    assert result == structure.bbox


def test_goal_bbox_at_applies_affine_warp_above_motion_threshold():
    structure = _structure(t_anchor=5.0)
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    tracking_cfg = _tracking_cfg()
    above = tracking_cfg["reestimate_motion_score_threshold"] + 1.0

    # a pure +10px-x, +5px-y translation matrix
    shift_matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]])

    with (
        patch.object(
            goal_mod, "_grab_frame_at", return_value=np.zeros((50, 50, 3), dtype=np.uint8)
        ),
        patch.object(goal_mod, "_fit_affine_transform", return_value=shift_matrix),
    ):
        result = goal_bbox_at(
            structure,
            "fake.mp4",
            8.0,
            take,
            motion_score=above,
            tracking_cfg=tracking_cfg,
            motion_cfg={},
        )
    assert result.x1 == structure.bbox.x1 + 10
    assert result.y1 == structure.bbox.y1 + 5
    assert result.x2 == structure.bbox.x2 + 10
    assert result.y2 == structure.bbox.y2 + 5


def test_goal_bbox_at_falls_back_to_anchor_when_affine_fit_fails():
    structure = _structure(t_anchor=5.0)
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    tracking_cfg = _tracking_cfg()
    above = tracking_cfg["reestimate_motion_score_threshold"] + 1.0

    with (
        patch.object(
            goal_mod, "_grab_frame_at", return_value=np.zeros((50, 50, 3), dtype=np.uint8)
        ),
        patch.object(goal_mod, "_fit_affine_transform", return_value=None),
    ):
        result = goal_bbox_at(
            structure,
            "fake.mp4",
            8.0,
            take,
            motion_score=above,
            tracking_cfg=tracking_cfg,
            motion_cfg={},
        )
    # honest fallback: last-known anchor position, never a fabricated extrapolation.
    assert result == structure.bbox


def test_goal_bbox_at_falls_back_to_anchor_when_frame_grab_fails():
    structure = _structure(t_anchor=5.0)
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    tracking_cfg = _tracking_cfg()
    above = tracking_cfg["reestimate_motion_score_threshold"] + 1.0

    with patch.object(goal_mod, "_grab_frame_at", return_value=None):
        result = goal_bbox_at(
            structure,
            "fake.mp4",
            8.0,
            take,
            motion_score=above,
            tracking_cfg=tracking_cfg,
            motion_cfg={},
        )
    assert result == structure.bbox
