"""Pure-logic unit tests for src/pipeline/annotated_video.py (ADR-15, CLAUDE.md §13.1) — no
GPU/video I/O."""

from __future__ import annotations

import src.pipeline.annotated_video as annotated_video_mod
from src.common.types import BBox, Event, EventType, Take, Track, TrackBox
from src.goal.detect import GoalStructure
from src.identity.verify import TakeIdentityResult
from src.pipeline.annotated_video import (
    _ActiveEventIndex,
    _bbox_to_polygon,
    _event_caption_label,
    _format_caption_timestamp,
    _nearest_by_time,
    _nearest_tracked_bbox,
    _NumberProgress,
    _SortedTimeIndex,
    _target_panel_header,
    _tracked_goal_bboxes_for_take,
    red_box_track_ids,
)


def _event(etype: EventType, t: float) -> Event:
    return Event(
        id=f"{etype.value}-{t}",
        type=etype,
        t_start=t,
        t_end=t,
        player_track_id=1,
        take_id=0,
        confidence=0.5,
        source="test",
        evidence={},
    )


def test_number_progress_accumulates_forward_only():
    events = [
        _event(EventType.TOUCH, 1.0),
        _event(EventType.TOUCH, 3.0),
        _event(EventType.PASS, 5.0),
    ]
    progress = _NumberProgress(events)
    assert progress.advance_to(0.5) == {}
    assert progress.advance_to(2.0) == {"touch": 1}
    assert progress.advance_to(4.0) == {"touch": 2}
    assert progress.advance_to(10.0) == {"touch": 2, "pass": 1}


def test_number_progress_never_double_counts_on_repeated_calls_at_same_time():
    events = [_event(EventType.SHOT, 1.0)]
    progress = _NumberProgress(events)
    progress.advance_to(1.0)
    progress.advance_to(1.0)
    assert progress.counts == {"shot": 1}


def test_nearest_by_time_within_tolerance():
    items = [{"t": 0.0}, {"t": 1.0}, {"t": 2.0}]
    found = _nearest_by_time(items, 1.05, tolerance=0.2, key=lambda i: i["t"])
    assert found == {"t": 1.0}


def test_nearest_by_time_outside_tolerance_returns_none():
    items = [{"t": 0.0}, {"t": 5.0}]
    assert _nearest_by_time(items, 2.5, tolerance=0.5, key=lambda i: i["t"]) is None


def test_nearest_by_time_empty_list_returns_none():
    assert _nearest_by_time([], 1.0, tolerance=1.0, key=lambda i: i) is None


# ---------------------------------------------------------------------------
# red_box_track_ids -- the verified-identity render gate (CLAUDE.md §13.1)
# ---------------------------------------------------------------------------


def _identity(
    status: str,
    jersey_number: int | None,
    location_track_ids: list[int],
    association_confirmed_by_jersey: bool = True,
) -> TakeIdentityResult:
    return TakeIdentityResult(
        take_id=0,
        jersey_number=jersey_number,
        status=status,
        confidence=0.8,
        evidence_frames=[1, 2],
        location_method="arrow_vote",
        location_track_ids=location_track_ids,
        association_confirmed_by_jersey=association_confirmed_by_jersey,
    )


def test_red_box_track_ids_verified_take_returns_its_location_ids():
    identity = _identity("verified", 7, [10, 11, 12])
    assert red_box_track_ids(identity) == {10, 11, 12}


def test_red_box_track_ids_unverified_take_returns_empty_never_a_fallback():
    identity = _identity("unverified", None, [10, 11])
    assert red_box_track_ids(identity) == set()


def test_red_box_track_ids_none_identity_returns_empty():
    assert red_box_track_ids(None) == set()


# ---------------------------------------------------------------------------
# _target_panel_header -- owner-reported bug fix, 2026-08-31: never display "VERIFIED" when the
# drawn box was only a colour pick with no real digit confirmation behind it (Golden Rule 5).
# ---------------------------------------------------------------------------


def test_target_panel_header_says_verified_when_jersey_confirmed():
    identity = _identity("verified", 10, [208], association_confirmed_by_jersey=True)
    assert _target_panel_header(identity) == "TARGET #10 -- VERIFIED"


def test_target_panel_header_says_colour_match_when_jersey_unconfirmed():
    identity = _identity("verified", 10, [208], association_confirmed_by_jersey=False)
    assert _target_panel_header(identity) == "TARGET #10 -- COLOUR MATCH (jersey unconfirmed)"


# ---------------------------------------------------------------------------
# regression: raw Track.id resets per take (Golden Rule 3) -- a box-lookup index MUST be keyed
# by (take_id, track_id), never track_id alone, or two different takes' same-numbered tracks
# silently collide (caught for real: a full render produced ZERO green boxes anywhere because
# every lookup was querying whichever take's same-id track happened to be indexed last).
# ---------------------------------------------------------------------------


def _track_box(t: float, cx: float) -> TrackBox:
    return TrackBox(
        frame_index=int(t * 10), t=t, bbox=BBox(x1=cx - 10, y1=0, x2=cx + 10, y2=100), conf=0.9
    )


def test_box_index_keyed_by_take_and_track_id_avoids_cross_take_collision():
    # two DIFFERENT takes each have their own track id=1, at completely different timestamps --
    # exactly the real-world shape (IDs reset per take).
    take0_track1 = Track(id=1, take_id=0, boxes=[_track_box(1.0, 100.0)])
    take5_track1 = Track(id=1, take_id=5, boxes=[_track_box(50.0, 900.0)])

    index_by_key = {
        (tr.take_id, tr.id): _SortedTimeIndex(tr.boxes, key=lambda b: b.t)
        for tr in (take0_track1, take5_track1)
    }

    # a query for take 0's track 1 at t=1.0 must find TAKE 0's box, never take 5's
    found = index_by_key[(0, 1)].nearest(1.0, 0.3)
    assert found is not None
    assert found.bbox.cx == 100.0

    found_other = index_by_key[(5, 1)].nearest(50.0, 0.3)
    assert found_other is not None
    assert found_other.bbox.cx == 900.0

    # a naive flat {track_id: index} dict would have collided these -- confirm the fix actually
    # keeps them distinct (this is the assertion that would have caught the real bug)
    assert index_by_key[(0, 1)] is not index_by_key[(5, 1)]


# ---------------------------------------------------------------------------
# _ActiveEventIndex / _event_caption_label / _format_caption_timestamp (ADR-19/20 event captions,
# CLAUDE.md §13.1) -- pure, no I/O; the render loop itself stays untested, as today.
# ---------------------------------------------------------------------------


def test_format_caption_timestamp():
    assert _format_caption_timestamp(65.0) == "1:05"
    assert _format_caption_timestamp(5.0) == "0:05"
    assert _format_caption_timestamp(0.0) == "0:00"


def test_event_caption_label_uses_the_same_timeline_label_as_the_statcard():
    touch = _event(EventType.TOUCH, 1.0)
    assert _event_caption_label(touch) == "Touch"
    turnover = _event(EventType.TURNOVER, 1.0)
    assert _event_caption_label(turnover) == "Turnover"
    out_of_bounds = _event(EventType.OUT_OF_BOUNDS, 1.0)
    assert _event_caption_label(out_of_bounds) == "Out of bounds"


def test_event_caption_label_key_moment_reads_action_phrase():
    ev = Event(
        id="km-1",
        type=EventType.KEY_MOMENT,
        t_start=1.0,
        t_end=1.0,
        player_track_id=1,
        take_id=0,
        confidence=0.5,
        source="manual_annotation",
        evidence={"action_phrase": "celebrates with the fans"},
    )
    assert _event_caption_label(ev) == "Celebration"


def test_active_event_index_active_within_caption_window():
    events_by_number = {2: [_event(EventType.TOUCH, 10.0)]}
    index = _ActiveEventIndex(events_by_number, caption_duration_s=2.0)
    assert index.active_at(10.0) == [(2, events_by_number[2][0])]
    assert index.active_at(11.5) == [(2, events_by_number[2][0])]  # still within the window
    assert index.active_at(9.9) == []  # before the event even starts
    assert index.active_at(12.1) == []  # past the caption duration


def test_active_event_index_multiple_simultaneous_events_across_numbers():
    ev_a = _event(EventType.TOUCH, 10.0)
    ev_b = _event(EventType.PASS, 10.0)
    events_by_number = {2: [ev_a], 7: [ev_b]}
    index = _ActiveEventIndex(events_by_number, caption_duration_s=1.0)
    active = index.active_at(10.0)
    assert sorted(active, key=lambda pair: pair[0]) == [(2, ev_a), (7, ev_b)]


def test_active_event_index_empty_when_no_events():
    index = _ActiveEventIndex({}, caption_duration_s=2.0)
    assert index.active_at(0.0) == []


# ---------------------------------------------------------------------------
# Stage E ("streamed-gathering-treehouse" plan) -- _tracked_goal_bboxes_for_take /
# _nearest_tracked_bbox / _bbox_to_polygon. take_motion_score/goal_bbox_at are mocked (no real
# decode/CV); these tests verify the anchor-generation + nearest-lookup + polygon-conversion
# logic itself.
# ---------------------------------------------------------------------------


def _goal_structure() -> GoalStructure:
    return GoalStructure(
        take_id=0,
        t_anchor=5.0,
        bbox=BBox(x1=100.0, y1=100.0, x2=200.0, y2=150.0),
        crossbar=(100.0, 100.0, 200.0, 100.0),
        posts=[(100.0, 100.0, 100.0, 150.0), (200.0, 100.0, 200.0, 150.0)],
        confidence=1.0,
        n_corroborating_frames=2,
        n_frames_sampled=2,
    )


def _goal_structure_cfg() -> dict:
    return {
        "tracking": {
            "reestimate_motion_score_threshold": 3.75,
            "min_reestimate_interval_s": 5.0,
            "max_reestimate_interval_s": 30.0,
            "frame_grab_window_s": 1.0,
        }
    }


def test_tracked_goal_bboxes_for_take_one_anchor_when_static(monkeypatch):
    monkeypatch.setattr(annotated_video_mod, "take_motion_score", lambda *a, **k: 0.0)
    fixed_bbox = BBox(x1=100.0, y1=100.0, x2=200.0, y2=150.0)
    monkeypatch.setattr(annotated_video_mod, "goal_bbox_at", lambda *a, **k: fixed_bbox)

    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    anchors = _tracked_goal_bboxes_for_take(
        _goal_structure(),
        "fake.mp4",
        take,
        hardware_cfg={"stages": {"profile": {"fps_sample": 2}}},
        profile_cfg={"motion": {}},
        goal_structure_cfg=_goal_structure_cfg(),
        use_nvdec=False,
    )
    assert len(anchors) == 1
    assert anchors[0][1] == fixed_bbox


def test_tracked_goal_bboxes_for_take_more_anchors_under_high_motion(monkeypatch):
    tracking_cfg = _goal_structure_cfg()["tracking"]
    above = tracking_cfg["reestimate_motion_score_threshold"] + 1.0
    monkeypatch.setattr(annotated_video_mod, "take_motion_score", lambda *a, **k: above)
    monkeypatch.setattr(annotated_video_mod, "goal_bbox_at", lambda *a, **k: _goal_structure().bbox)

    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    anchors = _tracked_goal_bboxes_for_take(
        _goal_structure(),
        "fake.mp4",
        take,
        hardware_cfg={"stages": {"profile": {"fps_sample": 2}}},
        profile_cfg={"motion": {}},
        goal_structure_cfg=_goal_structure_cfg(),
        use_nvdec=False,
    )
    assert len(anchors) == 2  # duration=10, min_reestimate_interval_s=5 -> ceil(10/5)=2


def test_nearest_tracked_bbox_picks_closest_in_time():
    box_a = BBox(x1=0, y1=0, x2=10, y2=10)
    box_b = BBox(x1=100, y1=100, x2=110, y2=110)
    anchors = [(1.0, box_a), (9.0, box_b)]
    assert _nearest_tracked_bbox(anchors, 2.0) == box_a
    assert _nearest_tracked_bbox(anchors, 8.5) == box_b


def test_nearest_tracked_bbox_empty_list_is_none():
    assert _nearest_tracked_bbox([], 5.0) is None


def test_bbox_to_polygon_is_a_four_corner_rectangle():
    bbox = BBox(x1=10.4, y1=20.6, x2=100.2, y2=200.9)
    poly = _bbox_to_polygon(bbox)
    assert poly.shape == (4, 1, 2)
    xs = sorted({int(poly[i][0][0]) for i in range(4)})
    ys = sorted({int(poly[i][0][1]) for i in range(4)})
    assert xs == [10, 100]
    assert ys == [21, 201]
