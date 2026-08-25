"""Pure-logic unit tests for `src/events/touches.py` (ADR-13/14; no GPU/video decode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, DetectionClass, EventType, Track, TrackBox
from src.events import touches

REPO_ROOT = Path(__file__).resolve().parents[1]


def _events_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")


def _box(t: float, cx: float, cy: float, height: float = 100.0, conf: float = 0.8) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 10)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=conf,
    )


def _ball(t: float, cx: float, cy: float, conf: float = 0.9) -> BallDetection:
    return BallDetection(
        bbox=BBox(x1=cx - 5.0, y1=cy - 5.0, x2=cx + 5.0, y2=cy + 5.0),
        conf=conf,
        frame_index=int(round(t * 30)),
        t=t,
    )


# ---------------------------------------------------------------------------
# point_to_bbox_distance -- exact boundary behaviour
# ---------------------------------------------------------------------------


def test_point_to_bbox_distance_inside_is_zero():
    bbox = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    assert touches.point_to_bbox_distance(50.0, 50.0, bbox) == 0.0


def test_point_to_bbox_distance_on_boundary_is_zero():
    bbox = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    assert touches.point_to_bbox_distance(100.0, 50.0, bbox) == 0.0


def test_point_to_bbox_distance_outside_is_positive():
    bbox = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    # 30px to the right of the box's right edge, same y -- pure horizontal distance
    assert touches.point_to_bbox_distance(130.0, 50.0, bbox) == pytest.approx(30.0)


def test_point_to_bbox_distance_diagonal_corner():
    bbox = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    # 3-4-5 triangle from the bottom-right corner
    assert touches.point_to_bbox_distance(103.0, 104.0, bbox) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# nearest_player_edge_distance
# ---------------------------------------------------------------------------


def test_nearest_player_edge_distance_prefers_closer_track():
    near = Track(id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0, height=50.0)])
    far = Track(id=2, take_id=0, boxes=[_box(1.0, 900.0, 900.0, height=50.0)])
    ball = _ball(1.0, 100.0, 100.0)
    result = touches.nearest_player_edge_distance(ball, [near, far], match_tolerance_s=0.15)
    assert result == (1, pytest.approx(0.0))


def test_nearest_player_edge_distance_none_when_no_track_in_time_tolerance():
    track = Track(id=1, take_id=0, boxes=[_box(10.0, 100.0, 100.0)])
    ball = _ball(1.0, 100.0, 100.0)  # 9s away, way beyond tolerance
    assert touches.nearest_player_edge_distance(ball, [track], match_tolerance_s=0.15) is None


# ---------------------------------------------------------------------------
# touch_confidence
# ---------------------------------------------------------------------------


def test_touch_confidence_closest_and_confident_ball_is_near_ceiling():
    cfg = _events_config()["touch"]
    confidence = touches.touch_confidence(distance_px=0.0, ball_conf=1.0, touch_cfg=cfg)
    assert confidence == pytest.approx(cfg["max_confidence"])


def test_touch_confidence_at_exact_threshold_distance_is_floor():
    cfg = _events_config()["touch"]
    confidence = touches.touch_confidence(
        distance_px=cfg["touch_distance_px"], ball_conf=1.0, touch_cfg=cfg
    )
    assert confidence == pytest.approx(cfg["min_confidence"])


def test_touch_confidence_never_exceeds_max_confidence():
    cfg = _events_config()["touch"]
    confidence = touches.touch_confidence(distance_px=-100.0, ball_conf=2.0, touch_cfg=cfg)
    assert confidence <= cfg["max_confidence"]


def test_touch_confidence_never_below_min_confidence():
    cfg = _events_config()["touch"]
    confidence = touches.touch_confidence(distance_px=1_000_000.0, ball_conf=0.0, touch_cfg=cfg)
    assert confidence >= cfg["min_confidence"]


# ---------------------------------------------------------------------------
# detect_touches -- should-fire / should-not-fire + debounce + take stamping
# ---------------------------------------------------------------------------


def test_detect_touches_fires_when_ball_is_at_player_bbox():
    cfg = _events_config()
    track = Track(
        id=5, take_id=2, boxes=[_box(1.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    ball = _ball(1.0, 100.0, 100.0)
    events = touches.detect_touches([ball], [track], take_id=2, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.TOUCH
    assert ev.player_track_id == 5
    assert ev.take_id == 2
    assert ev.evidence["calibrated"] is False
    assert 0.0 <= ev.confidence <= cfg["touch"]["max_confidence"]


def test_detect_touches_does_not_fire_when_ball_is_far_from_every_player():
    cfg = _events_config()
    far_dist = cfg["touch"]["touch_distance_px"] + 1000.0
    track = Track(
        id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    ball = _ball(1.0, 100.0 + far_dist, 100.0)
    events = touches.detect_touches([ball], [track], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_touches_excludes_referees():
    cfg = _events_config()
    referee = Track(
        id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0)], dominant_class=DetectionClass.REFEREE
    )
    ball = _ball(1.0, 100.0, 100.0)
    events = touches.detect_touches([ball], [referee], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_touches_debounces_rapid_repeats_for_the_same_identity():
    cfg = _events_config()
    track = Track(
        id=1,
        take_id=0,
        boxes=[_box(t, 100.0, 100.0) for t in [0.0, 0.05, 0.1, 0.9]],
        dominant_class=DetectionClass.PLAYER,
    )
    balls = [_ball(t, 100.0, 100.0) for t in [0.0, 0.05, 0.1, 0.9]]
    events = touches.detect_touches(balls, [track], take_id=0, events_cfg=cfg)
    # min_gap_s=0.3: samples at 0.0/0.05/0.1 collapse to ONE touch, 0.9 is far enough for a second
    assert [e.t_start for e in events] == pytest.approx([0.0, 0.9])


def test_detect_touches_debounce_is_per_identity_not_global():
    cfg = _events_config()
    track_a = Track(
        id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    track_b = Track(
        id=2, take_id=0, boxes=[_box(0.01, 500.0, 500.0)], dominant_class=DetectionClass.PLAYER
    )
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.01, 500.0, 500.0)]
    events = touches.detect_touches(balls, [track_a, track_b], take_id=0, events_cfg=cfg)
    assert {e.player_track_id for e in events} == {1, 2}


def test_detect_touches_uses_identity_map_when_provided():
    cfg = _events_config()
    # two raw fragments (e.g. across a brief occlusion) mapped to the SAME stitched identity
    frag_a = Track(
        id=10, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    frag_b = Track(
        id=11, take_id=0, boxes=[_box(1.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 100.0, 100.0)]
    identity_of = {10: 10, 11: 10}
    events = touches.detect_touches(
        balls, [frag_a, frag_b], take_id=0, events_cfg=cfg, identity_of=identity_of
    )
    assert {e.player_track_id for e in events} == {10}
    assert all(e.evidence["raw_track_id"] in (10, 11) for e in events)


def test_detect_touches_rejects_mixed_take_ids():
    """The mandatory take-crossing rejection test: raw `Track.id`s reset per take (Golden Rule 3),
    so `detect_touches` must reject a `tracks` list spanning more than one `take_id` outright,
    never silently attribute a touch across a cut -- even when the two tracks are geometrically
    identical (same position, same instant)."""
    cfg = _events_config()
    same_take = Track(
        id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    other_take = Track(
        id=2, take_id=1, boxes=[_box(0.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    ball = _ball(0.0, 100.0, 100.0)
    with pytest.raises(AssertionError):
        touches.detect_touches([ball], [same_take, other_take], take_id=0, events_cfg=cfg)


def test_detect_touches_ignores_low_confidence_ball_detections():
    cfg = _events_config()
    track = Track(
        id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], dominant_class=DetectionClass.PLAYER
    )
    low_conf_ball = _ball(0.0, 100.0, 100.0, conf=cfg["touch"]["min_ball_conf"] - 0.01)
    events = touches.detect_touches([low_conf_ball], [track], take_id=0, events_cfg=cfg)
    assert events == []
