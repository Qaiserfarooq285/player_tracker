"""Pure-logic unit tests for src/events/aggregate.py's target-attribution helpers (ADR-15) — no
GPU/network calls."""

from __future__ import annotations

from src.common.types import BBox, DetectionClass, Event, EventType, Track, TrackBox
from src.events.aggregate import (
    attribute_events_to_target,
    suppress_turnovers_covered_by_tackles,
    target_identity_id,
)

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


def _turnover(losing_raw: int, receiving_raw: int, t_end: float) -> Event:
    return Event(
        id=f"turnover-{losing_raw}-{receiving_raw}-{t_end}",
        type=EventType.TURNOVER,
        t_start=t_end - 0.1,
        t_end=t_end,
        player_track_id=losing_raw,
        take_id=0,
        confidence=0.2,
        source="possession_change_opposing_colour",
        evidence={"losing_raw_track_id": losing_raw, "receiving_raw_track_id": receiving_raw},
    )


def _tackle(tackler_raw: int, tackled_raw: int, t_end: float) -> Event:
    return Event(
        id=f"tackle-{tackler_raw}-{tackled_raw}-{t_end}",
        type=EventType.TACKLE,
        t_start=t_end - 0.5,
        t_end=t_end,
        player_track_id=tackler_raw,
        take_id=0,
        confidence=0.25,
        source="possession_change_closing_speed_heuristic",
        evidence={"tackler_raw_track_id": tackler_raw, "tackled_raw_track_id": tackled_raw},
    )


# ---------------------------------------------------------------------------
# suppress_turnovers_covered_by_tackles (ADR-20 §3, tackle/turnover double-count guard)
# ---------------------------------------------------------------------------

TURNOVER_CFG = {"suppress_if_tackle_within_s": 0.5}


def test_suppress_turnovers_covered_by_tackles_removes_the_matching_one():
    turnover = _turnover(losing_raw=1, receiving_raw=2, t_end=10.0)
    tackle = _tackle(tackler_raw=2, tackled_raw=1, t_end=10.05)  # same transition, near-identical t
    kept = suppress_turnovers_covered_by_tackles([turnover, tackle], TURNOVER_CFG)
    assert kept == [tackle]


def test_suppress_turnovers_covered_by_tackles_logs_the_drop():
    from src.common.logging import DropCounter

    turnover = _turnover(losing_raw=1, receiving_raw=2, t_end=10.0)
    tackle = _tackle(tackler_raw=2, tackled_raw=1, t_end=10.0)
    drops = DropCounter("test")
    suppress_turnovers_covered_by_tackles([turnover, tackle], TURNOVER_CFG, drops)
    assert drops.as_dict() == {"turnover_suppressed_by_tackle": 1}


def test_suppress_turnovers_covered_by_tackles_keeps_unrelated_turnover():
    # a tackle for a COMPLETELY different track pair must never suppress this turnover
    turnover = _turnover(losing_raw=1, receiving_raw=2, t_end=10.0)
    unrelated_tackle = _tackle(tackler_raw=5, tackled_raw=6, t_end=10.0)
    kept = suppress_turnovers_covered_by_tackles([turnover, unrelated_tackle], TURNOVER_CFG)
    assert turnover in kept


def test_suppress_turnovers_covered_by_tackles_keeps_when_outside_time_window():
    # same track pair, but the tackle happened well outside the configured window
    turnover = _turnover(losing_raw=1, receiving_raw=2, t_end=10.0)
    far_tackle = _tackle(tackler_raw=2, tackled_raw=1, t_end=20.0)
    kept = suppress_turnovers_covered_by_tackles([turnover, far_tackle], TURNOVER_CFG)
    assert turnover in kept


def test_suppress_turnovers_covered_by_tackles_leaves_non_turnover_events_untouched():
    touch = Event(
        id="touch-1",
        type=EventType.TOUCH,
        t_start=1.0,
        t_end=1.0,
        player_track_id=1,
        take_id=0,
        confidence=0.3,
        source="test",
        evidence={},
    )
    kept = suppress_turnovers_covered_by_tackles([touch], TURNOVER_CFG)
    assert kept == [touch]


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


# ---------------------------------------------------------------------------
# attribute_events_to_target -- 2026-09-19, "not counting the passes correctly"
# ---------------------------------------------------------------------------


def _pass(passer_raw: int, receiver_raw: int, identity: int, t: float = 3.0) -> Event:
    return Event(
        id=f"pass-{t}",
        type=EventType.PASS,
        t_start=t,
        t_end=t + 0.4,
        player_track_id=identity,
        take_id=0,
        confidence=0.3,
        source="test",
        evidence={"passer_raw_track_id": passer_raw, "receiver_raw_track_id": receiver_raw},
    )


def test_attribute_pass_kept_when_target_split_across_two_identities():
    """The target's verified fragments {10, 11, 25} were split by `build_take_identities` into
    identity 500 (10, 11 -- the majority) and identity 25 (25 + a stranger 40). A pass the target
    made while tracked as raw 25 used to be dropped by the majority vote; its evidence names raw
    25, which IS a verified target fragment, so it belongs."""
    identity_of = {10: 500, 11: 500, 25: 25, 40: 25}
    ev = _pass(passer_raw=25, receiver_raw=7, identity=25)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10, 11, 25], identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == [ev]


def test_attribute_pass_by_stranger_stitched_into_target_identity_is_excluded():
    """The mirror case: the partition joined stranger raw 40 into the target's own majority chain
    500. A pass raw 40 made carries identity 500 but its passer raw id is NOT a verified target
    fragment -- the human-verified location set wins over the greedy stitch."""
    identity_of = {10: 500, 11: 500, 40: 500}
    ev = _pass(passer_raw=40, receiver_raw=7, identity=500)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10, 11], identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == []


def test_attribute_pass_received_by_target_is_not_the_targets_pass():
    identity_of = {10: 500, 7: 7}
    ev = _pass(passer_raw=7, receiver_raw=10, identity=7)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10], identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == []


def test_attribute_identity_only_event_belongs_via_any_mapped_identity_not_just_majority():
    """An event with no raw-id evidence (the generic `_event` helper) whose identity is the
    MINORITY chain of the target's own fragments still belongs."""
    identity_of = {10: 500, 11: 500, 25: 25}
    ev = _event(EventType.TOUCH, 1.0, player_track_id=25)
    result = attribute_events_to_target(
        [ev], [], [], location_track_ids=[10, 11, 25], identity_of=identity_of,
        attribution_cfg=ATTRIBUTION_CFG,
    )
    assert result == [ev]
