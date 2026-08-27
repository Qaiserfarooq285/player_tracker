"""Pure-logic unit tests for src/events/aggregate.py's target-attribution helpers (ADR-15) — no
GPU/network calls."""

from __future__ import annotations

from src.common.types import BBox, DetectionClass, Event, EventType, Track, TrackBox
from src.events.aggregate import attribute_events_to_target, target_identity_id

ATTRIBUTION_CFG = {"shot_proximity_bbox_heights": 4.0, "max_time_gap_s": 1.0}


def _event(etype, t, player_track_id=None, take_id=0) -> Event:
    return Event(
        id=f"{etype.value}-{t}",
        type=etype,
        t_start=t,
        t_end=t,
        player_track_id=player_track_id,
        take_id=take_id,
        confidence=0.5,
        source="test",
        evidence={},
    )


def _track(tid: int, cx: float = 0.0, cy: float = 0.0, height: float = 100.0) -> Track:
    box = TrackBox(
        frame_index=0,
        t=0.0,
        bbox=BBox(x1=cx - 10, y1=cy - height / 2, x2=cx + 10, y2=cy + height / 2),
        conf=0.8,
    )
    return Track(id=tid, take_id=0, boxes=[box], dominant_class=DetectionClass.PLAYER)


# ---------------------------------------------------------------------------
# target_identity_id
# ---------------------------------------------------------------------------


def test_target_identity_id_majority_vote():
    identity_of = {10: 100, 11: 100, 12: 200}
    assert target_identity_id([10, 11, 12], identity_of) == 100


def test_target_identity_id_empty_mapping_returns_none():
    assert target_identity_id([10, 11], {}) is None


def test_target_identity_id_no_location_ids_returns_none():
    assert target_identity_id([], {10: 100}) is None


# ---------------------------------------------------------------------------
# attribute_events_to_target
# ---------------------------------------------------------------------------


def test_attribute_raw_track_id_event_belongs_when_in_location_ids():
    # sprint-style event: player_track_id is a RAW track id
    ev = _event(EventType.SPRINT, 1.0, player_track_id=10)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10, 11], identity_of={}, attribution_cfg=ATTRIBUTION_CFG
    )
    assert result == [ev]


def test_attribute_raw_track_id_event_excluded_when_not_in_location_ids():
    ev = _event(EventType.SPRINT, 1.0, player_track_id=99)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10, 11], identity_of={}, attribution_cfg=ATTRIBUTION_CFG
    )
    assert result == []


def test_attribute_identity_id_event_belongs_via_majority_mapping():
    # touch/possession-style event: player_track_id is an IDENTITY id from build_take_identities
    identity_of = {10: 500, 11: 500}
    ev = _event(EventType.TOUCH, 1.0, player_track_id=500)
    result = attribute_events_to_target(
        [ev],
        [],
        [],
        location_track_ids=[10, 11],
        identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == [ev]


def test_attribute_identity_id_event_excluded_for_other_identity():
    identity_of = {10: 500, 20: 600}
    ev = _event(EventType.TOUCH, 1.0, player_track_id=600)
    result = attribute_events_to_target(
        [ev],
        [],
        [],
        location_track_ids=[10],
        identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == []


def test_attribute_shot_uses_spatial_proximity():
    from src.common.types import BallDetection

    target_track = _track(10, cx=0.0, cy=0.0, height=100.0)
    ball = BallDetection(
        bbox=BBox(x1=-5, y1=-5, x2=5, y2=5), conf=0.9, frame_index=0, t=1.0, interpolated=False
    )
    shot_event = _event(EventType.SHOT, 1.0, player_track_id=None)
    result = attribute_events_to_target(
        [shot_event],
        take_tracks=[target_track],
        take_balls=[ball],
        location_track_ids=[10],
        identity_of={},
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == [shot_event]


def test_attribute_shot_excluded_when_ball_far_from_target():
    from src.common.types import BallDetection

    target_track = _track(10, cx=0.0, cy=0.0, height=100.0)
    far_ball = BallDetection(
        bbox=BBox(x1=995, y1=995, x2=1005, y2=1005),
        conf=0.9,
        frame_index=0,
        t=1.0,
        interpolated=False,
    )
    shot_event = _event(EventType.SHOT, 1.0, player_track_id=None)
    result = attribute_events_to_target(
        [shot_event],
        take_tracks=[target_track],
        take_balls=[far_ball],
        location_track_ids=[10],
        identity_of={},
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == []
