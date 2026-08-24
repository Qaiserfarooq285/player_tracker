"""Round-trip + property tests for the pydantic data contracts in src/common/types.py
(CLAUDE.md §4).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.types import (
    BallDetection,
    BBox,
    Clip,
    Detection,
    DetectionClass,
    Event,
    EventType,
    Frame,
    PitchHomography,
    PlayerStats,
    RunProfile,
    RunReport,
    SourceProfile,
    StageTiming,
    Take,
    Track,
    TrackBox,
)


def _round_trip(model):
    """Construct -> model_dump_json -> validate back; return the reconstructed instance."""
    cls = type(model)
    dumped = model.model_dump_json()
    restored = cls.model_validate_json(dumped)
    assert restored == model
    return restored


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


def test_bbox_property_math():
    b = BBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
    assert b.width == 10.0
    assert b.height == 20.0
    assert b.area == 200.0
    assert b.cx == 5.0
    assert b.cy == 10.0
    assert b.to_xyxy() == (0.0, 0.0, 10.0, 20.0)


def test_bbox_round_trip():
    _round_trip(BBox(x1=1.5, y1=2.5, x2=11.5, y2=22.5))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_source_profile_values():
    assert SourceProfile.BROADCAST == "broadcast"
    assert SourceProfile.SINGLE_STATIC == "single-static"
    assert SourceProfile.SINGLE_PANNING == "single-panning"


def test_detection_class_values():
    assert DetectionClass.PLAYER == "player"
    assert DetectionClass.GOALKEEPER == "goalkeeper"
    assert DetectionClass.REFEREE == "referee"
    assert DetectionClass.BALL == "ball"


def test_event_type_values():
    assert EventType.GOAL == "goal"
    assert EventType.SHOT == "shot"
    assert EventType.SPRINT == "sprint"
    assert EventType.PASS == "pass"
    assert EventType.TOUCH == "touch"
    assert EventType.TACKLE == "tackle"
    assert EventType.DRIBBLE == "dribble"
    assert EventType.KEY_MOMENT == "key_moment"


# ---------------------------------------------------------------------------
# Frame / Detection / BallDetection
# ---------------------------------------------------------------------------


def test_frame_round_trip():
    _round_trip(Frame(index=0, t=0.0, path=None))
    _round_trip(Frame(index=5, t=0.166, path="work/clip1_43/frames/000005.png"))


def test_detection_round_trip():
    d = Detection(
        bbox=BBox(x1=0, y1=0, x2=50, y2=100),
        cls=DetectionClass.PLAYER,
        conf=0.91,
        frame_index=12,
    )
    restored = _round_trip(d)
    assert restored.cls == DetectionClass.PLAYER


def test_ball_detection_round_trip():
    bd = BallDetection(
        bbox=BBox(x1=100, y1=100, x2=110, y2=110),
        conf=0.4,
        frame_index=30,
        interpolated=True,
    )
    _round_trip(bd)


# ---------------------------------------------------------------------------
# PitchHomography
# ---------------------------------------------------------------------------


def test_pitch_homography_round_trip_and_helpers():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    h = PitchHomography(
        matrix=identity,
        keypoints=[(0.0, 0.0), (1.0, 1.0)],
        conf=0.85,
        frame_index=0,
        method="manual_4pt",
    )
    _round_trip(h)

    arr = h.to_numpy()
    assert arr.shape == (3, 3)

    pts = h.pixel_to_pitch([(1.0, 2.0), (3.0, 4.0)])
    assert len(pts) == 2
    assert pts[0] == pytest.approx((1.0, 2.0), abs=1e-6)


def test_pitch_homography_rejects_non_3x3():
    with pytest.raises(ValueError):
        PitchHomography(
            matrix=[[1.0, 0.0], [0.0, 1.0]],
            conf=0.5,
            method="manual_4pt",
        )


# ---------------------------------------------------------------------------
# Take
# ---------------------------------------------------------------------------


def test_take_round_trip_default_kind():
    t = Take(id=0, t_start=0.0, t_end=5.0, frame_start=0, frame_end=150)
    assert t.kind == "unknown"
    assert t.dropped_reason is None
    _round_trip(t)


def test_take_round_trip_replay_kind():
    t = Take(
        id=1,
        t_start=5.0,
        t_end=6.0,
        frame_start=150,
        frame_end=180,
        kind="replay",
        dropped_reason="replay_sting_detected",
    )
    _round_trip(t)


# ---------------------------------------------------------------------------
# TrackBox / Track
# ---------------------------------------------------------------------------


def test_trackbox_round_trip():
    tb = TrackBox(frame_index=10, t=0.33, bbox=BBox(x1=0, y1=0, x2=10, y2=10), conf=0.7)
    _round_trip(tb)


def _make_track(n_boxes: int, take_id: int = 0) -> Track:
    boxes = [
        TrackBox(
            frame_index=i,
            t=float(i),
            bbox=BBox(x1=float(i), y1=0.0, x2=float(i) + 10.0, y2=10.0),
            conf=0.9,
        )
        for i in range(n_boxes)
    ]
    return Track(id=1, take_id=take_id, boxes=boxes)


def test_track_duration_and_n_boxes():
    track = _make_track(4)  # t = 0,1,2,3
    assert track.n_boxes == 4
    assert track.t_start == 0.0
    assert track.t_end == 3.0
    assert track.duration == 3.0


def test_track_empty_boxes_defaults():
    track = Track(id=2, take_id=0, boxes=[])
    assert track.n_boxes == 0
    assert track.t_start == 0.0
    assert track.t_end == 0.0
    assert track.duration == 0.0


def test_track_round_trip():
    track = _make_track(3)
    restored = _round_trip(track)
    assert restored.n_boxes == 3


# ---------------------------------------------------------------------------
# Event / Clip / PlayerStats
# ---------------------------------------------------------------------------


def test_event_round_trip():
    e = Event(
        id="evt-1",
        type=EventType.SPRINT,
        t_start=10.0,
        t_end=12.0,
        player_track_id=7,
        take_id=0,
        confidence=0.66,
        source="speed_heuristic",
        evidence={"peak_speed_mps": 7.2, "frame_indices": [300, 301, 302]},
    )
    restored = _round_trip(e)
    assert restored.evidence["peak_speed_mps"] == 7.2


def test_event_optional_player_track_id_none():
    e = Event(
        id="evt-2",
        type=EventType.GOAL,
        t_start=100.0,
        t_end=103.0,
        player_track_id=None,
        take_id=None,
        confidence=0.5,
        source="scoreboard_delta",
    )
    _round_trip(e)


def test_clip_round_trip():
    c = Clip(event_id="evt-1", t_start=7.0, t_end=15.0, take_id=0, rank_score=42.0, path=None)
    _round_trip(c)


def test_player_stats_round_trip():
    ps = PlayerStats(
        player_ref="jersey_43",
        track_ids=[1, 4, 9],
        counts={"sprint": 3, "shot": 1},
        confidences={"sprint": 0.7, "shot": 0.4},
        events=["evt-1", "evt-2"],
    )
    _round_trip(ps)


# ---------------------------------------------------------------------------
# RunProfile / StageTiming / RunReport
# ---------------------------------------------------------------------------


def test_run_profile_round_trip():
    rp = RunProfile(
        profile=SourceProfile.SINGLE_STATIC,
        n_cuts=0,
        cuts_per_min=0.0,
        motion_score=0.8,
        width=3840,
        height=2160,
        fps=30.0,
        duration=11.1,
        quality_flag="ok",
    )
    restored = _round_trip(rp)
    assert restored.profile == SourceProfile.SINGLE_STATIC


def test_stage_timing_round_trip():
    st = StageTiming(stage="detect", wall_seconds=12.3, vram_peak_mb=3200.0, notes="batch=2")
    _round_trip(st)


def test_run_report_round_trip():
    report = RunReport(
        run_id="run-20260824-1",
        video="input/clip1 43.mp4",
        config_hash="abc123def456",
        profile=RunProfile(
            profile=SourceProfile.SINGLE_STATIC,
            n_cuts=0,
            cuts_per_min=0.0,
            motion_score=0.8,
            width=3840,
            height=2160,
            fps=30.0,
            duration=11.1,
            quality_flag="ok",
        ),
        timings=[StageTiming(stage="ingest", wall_seconds=1.2, vram_peak_mb=None)],
        dropped={"replay": 0, "low_confidence": 5},
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    restored = _round_trip(report)
    assert restored.dropped["low_confidence"] == 5
