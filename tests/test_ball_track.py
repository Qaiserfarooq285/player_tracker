"""Pure-logic unit tests for `src/events/ball_track.py` (Stage 1 of the evidence-based event
redesign, owner spec 2026-09-02). No GPU/video decode -- synthetic `BallDetection`s only.
"""

from __future__ import annotations

from pathlib import Path

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, Track, TrackBox
from src.events import ball_track as bt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _events_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")


def _ball(t: float, cx: float, cy: float, conf: float = 0.9, interp: bool = False) -> BallDetection:
    return BallDetection(
        bbox=BBox(x1=cx - 5, y1=cy - 5, x2=cx + 5, y2=cy + 5),
        conf=conf,
        frame_index=int(round(t * 25)),
        t=t,
        interpolated=interp,
    )


def _track_with_height(height: float) -> Track:
    box = TrackBox(frame_index=0, t=0.0, bbox=BBox(x1=0, y1=0, x2=20, y2=height), conf=0.9)
    return Track(id=1, take_id=0, boxes=[box])


# ---------------------------------------------------------------------------
# reference_player_height
# ---------------------------------------------------------------------------


def test_reference_player_height_median_of_all_boxes():
    tracks = [_track_with_height(100.0), _track_with_height(80.0), _track_with_height(120.0)]
    assert bt.reference_player_height(tracks) == 100.0


def test_reference_player_height_empty_tracks_is_zero():
    assert bt.reference_player_height([]) == 0.0


# ---------------------------------------------------------------------------
# build_ball_state_series
# ---------------------------------------------------------------------------


def test_low_confidence_samples_are_dropped_not_marked_unknown():
    """A sample below min_conf_for_state is a HOLE in the series (absent), never a low-confidence
    BallState a careless caller could still read a position out of."""
    cfg = _events_config()["ball"]
    balls = [_ball(0.0, 100, 100, conf=0.9), _ball(0.1, 105, 100, conf=0.1)]  # second too low
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert len(series) == 1
    assert series[0].t == 0.0


def test_moving_ball_has_nonzero_speed_and_direction():
    cfg = dict(_events_config()["ball"])
    cfg["smoothing_window_frames"] = 1  # isolate the velocity maths from smoothing
    balls = [_ball(0.0, 0, 0), _ball(0.1, 10, 0)]  # 10px right in 0.1s
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert len(series) == 2
    assert series[0].vx == 0.0 and series[0].vy == 0.0  # no prior sample to diff against
    # 10px / 0.1s / 100px(height) = 1.0 bbox-height/s, moving in +x
    assert series[1].vx > 0
    assert series[1].speed > 0
    assert series[1].direction_deg is not None


def test_stationary_ball_has_no_direction():
    cfg = dict(_events_config()["ball"])
    cfg["smoothing_window_frames"] = 1
    balls = [_ball(0.0, 50, 50), _ball(0.1, 50, 50)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert series[1].speed == 0.0
    assert series[1].direction_deg is None


def test_gap_wider_than_max_bridge_marks_unknown_not_a_fabricated_velocity():
    """Owner: 'do NOT immediately create a touch/pass/shot/goal because the ball disappeared.'
    A gap Stage 2 itself declined to bridge must not produce a velocity reading here either."""
    cfg = dict(_events_config()["ball"])
    cfg["max_gap_bridge_s"] = 0.5
    balls = [_ball(0.0, 0, 0), _ball(2.0, 500, 500)]  # huge implied speed if naively diffed
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert len(series) == 2
    assert series[1].known is False
    assert series[1].vx == 0.0 and series[1].vy == 0.0  # no fabricated velocity


def test_gap_within_max_bridge_is_known():
    cfg = dict(_events_config()["ball"])
    cfg["max_gap_bridge_s"] = 0.5
    cfg["smoothing_window_frames"] = 1
    balls = [_ball(0.0, 0, 0), _ball(0.3, 30, 0)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert series[1].known is True


def test_interpolated_flag_passes_through():
    cfg = _events_config()["ball"]
    balls = [_ball(0.0, 0, 0, interp=False), _ball(0.1, 10, 0, interp=True)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert series[0].is_interpolated is False
    assert series[1].is_interpolated is True


def test_no_motion_model_falls_back_to_raw_uncompensated():
    cfg = _events_config()["ball"]
    balls = [_ball(0.0, 0, 0), _ball(0.1, 10, 0)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0, motion=None)
    assert all(not s.is_camera_compensated for s in series)


def test_zero_reference_height_falls_back_to_raw_pixel_scale():
    """A take with no measurable player height (reference_height_px <= 0) must not divide by
    zero or crash -- falls back to a scale of 1.0 (raw pixels/second), never fabricates a
    plausible-looking bbox-height number from nothing."""
    cfg = dict(_events_config()["ball"])
    cfg["smoothing_window_frames"] = 1
    balls = [_ball(0.0, 0, 0), _ball(0.1, 10, 0)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=0.0)
    assert series[1].vx == 100.0  # 10px / 0.1s / 1.0 scale


def test_empty_balls_returns_empty_series():
    cfg = _events_config()["ball"]
    assert bt.build_ball_state_series([], cfg, reference_height_px=100.0) == []


def test_single_sample_has_position_but_no_velocity():
    cfg = _events_config()["ball"]
    balls = [_ball(1.0, 50, 50)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert len(series) == 1
    assert series[0].vx == 0.0
    assert series[0].known is True  # a lone position is real evidence, just no velocity yet


# ---------------------------------------------------------------------------
# nearest_ball_state
# ---------------------------------------------------------------------------


def test_nearest_ball_state_within_tolerance():
    cfg = dict(_events_config()["ball"])
    cfg["smoothing_window_frames"] = 1
    balls = [_ball(1.0, 0, 0), _ball(1.2, 10, 0)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    found = bt.nearest_ball_state(series, t=1.05, tolerance_s=0.1)
    assert found is not None
    assert found.t == 1.0


def test_nearest_ball_state_outside_tolerance_is_none():
    cfg = _events_config()["ball"]
    balls = [_ball(1.0, 0, 0)]
    series = bt.build_ball_state_series(balls, cfg, reference_height_px=100.0)
    assert bt.nearest_ball_state(series, t=5.0, tolerance_s=0.1) is None
