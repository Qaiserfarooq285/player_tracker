"""Pure-logic unit tests for Stage 4 (CLAUDE.md task spec: no GPU/video decode).

Covers: the shared threshold-segmentation helper (`src/events/segments.py`) at its exact
merge-gap/min-duration boundaries, per-track/per-ball normalised speed computation, sprint/shot
event emission (confidence, evidence, ADR-6 "never calibrated" contract), and the goal detector's
explicit "not available" result on this footage (CLAUDE.md §3.2(3), Golden Rule 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BBox, EventType, Track, TrackBox
from src.events import shots, sprints
from src.events.goals import check_goal_availability
from src.events.segments import find_threshold_segments, moving_average

REPO_ROOT = Path(__file__).resolve().parents[1]


def _events_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------


def test_moving_average_window_one_is_identity():
    assert moving_average([1.0, 5.0, 3.0], 1) == [1.0, 5.0, 3.0]


def test_moving_average_empty_is_empty():
    assert moving_average([], 5) == []


def test_moving_average_ramps_up_before_full_window():
    # window=3: out[0]=avg(1)=1, out[1]=avg(1,2)=1.5, out[2]=avg(1,2,3)=2, out[3]=avg(2,3,4)=3
    out = moving_average([1.0, 2.0, 3.0, 4.0], 3)
    assert out == pytest.approx([1.0, 1.5, 2.0, 3.0])


def test_moving_average_constant_series_unchanged():
    assert moving_average([7.0] * 10, 4) == pytest.approx([7.0] * 10)


# ---------------------------------------------------------------------------
# find_threshold_segments -- exact boundary semantics
# ---------------------------------------------------------------------------


def test_find_threshold_segments_basic_run():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = [0.0, 5.0, 5.0, 5.0, 0.0]
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=0.5, merge_gap_s=0.3
    )
    assert segments == [(0.5, 1.5)]


def test_find_threshold_segments_value_exactly_at_threshold_qualifies():
    times = [0.0, 1.0]
    values = [4.0, 4.0]
    segments = find_threshold_segments(
        times, values, threshold=4.0, min_duration_s=1.0, merge_gap_s=0.1
    )
    assert segments == [(0.0, 1.0)]


def test_find_threshold_segments_duration_exactly_at_min_duration_is_kept():
    # "at least" min_duration_s -- exactly equal must qualify (inclusive boundary).
    times = [0.0, 1.0]
    values = [10.0, 10.0]
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=1.0, merge_gap_s=0.1
    )
    assert segments == [(0.0, 1.0)]


def test_find_threshold_segments_duration_just_under_min_duration_is_dropped():
    times = [0.0, 0.99]
    values = [10.0, 10.0]
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=1.0, merge_gap_s=0.1
    )
    assert segments == []


def test_find_threshold_segments_gap_exactly_at_merge_gap_is_not_merged():
    # two 1s-long runs with a gap of EXACTLY merge_gap_s=1.0 apart -- "STRICTLY LESS than" the
    # spec's own boundary semantics means this must NOT merge.
    times = [0.0, 1.0, 2.0, 3.0]
    values = [10.0, 10.0, 10.0, 10.0]
    # run 1 = [0.0, 1.0], gap to run 2 start (2.0) is exactly 1.0 -- simulate by removing the
    # in-between sample so there's a genuine gap between two separate runs:
    times = [0.0, 1.0, 2.0, 3.0]
    values = [10.0, 10.0, 1.0, 10.0]  # dip below threshold at t=2.0 splits the runs
    # run A = [0.0, 1.0], run B = [3.0, 3.0] (single point) -- gap = 3.0 - 1.0 = 2.0
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=0.0, merge_gap_s=2.0
    )
    # gap (2.0) is NOT strictly less than merge_gap_s (2.0) -> stays two segments
    assert segments == [(0.0, 1.0), (3.0, 3.0)]


def test_find_threshold_segments_gap_just_under_merge_gap_is_merged():
    times = [0.0, 1.0, 2.0, 3.0]
    values = [10.0, 10.0, 1.0, 10.0]
    # same runs as above; gap = 2.0, merge_gap_s slightly above 2.0 -> merges into one segment
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=0.0, merge_gap_s=2.01
    )
    assert segments == [(0.0, 3.0)]


def test_find_threshold_segments_chained_merges():
    # three isolated single-sample runs at t=0.0, t=1.0, t=3.0: gap(run1->run2)=1.0,
    # gap(run2->run3)=2.0. merge_gap_s=1.5 merges the first pair (1.0 < 1.5) but not the second
    # (2.0 is NOT < 1.5) -- also proves the merge correctly extends from the MERGED end (1.0),
    # not the original first run's end, when computing the next gap.
    times = [0.0, 0.5, 1.0, 1.5, 3.0, 3.5]
    values = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    segments = find_threshold_segments(
        times, values, threshold=5.0, min_duration_s=0.0, merge_gap_s=1.5
    )
    assert segments == [(0.0, 1.0), (3.0, 3.0)]


def test_find_threshold_segments_empty_input():
    assert find_threshold_segments([], [], threshold=1.0, min_duration_s=0.1, merge_gap_s=0.1) == []


def test_find_threshold_segments_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        find_threshold_segments(
            [0.0, 1.0], [0.0], threshold=1.0, min_duration_s=0.1, merge_gap_s=0.1
        )


# ---------------------------------------------------------------------------
# sprints
# ---------------------------------------------------------------------------


def _box(t: float, cx: float, cy: float, height: float = 100.0, conf: float = 0.8) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 10)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=conf,
    )


def test_track_speed_series_stationary_track_has_zero_speed():
    boxes = [_box(t=i * 0.1, cx=100.0, cy=100.0) for i in range(10)]
    track = Track(id=1, take_id=0, boxes=boxes)
    times, speeds = sprints.track_speed_series(track, window=1)
    assert all(s == pytest.approx(0.0, abs=1e-9) for s in speeds)


def test_track_speed_series_moving_track_has_positive_speed():
    # bbox height 100px; moving 50px per 0.1s step -> raw speed 500 px/s / 100px height = 5.0
    boxes = [_box(t=i * 0.1, cx=100.0 + i * 50.0, cy=100.0) for i in range(10)]
    track = Track(id=1, take_id=0, boxes=boxes)
    times, speeds = sprints.track_speed_series(track, window=1)
    assert speeds[0] == pytest.approx(0.0)
    for s in speeds[1:]:
        assert s == pytest.approx(5.0, rel=1e-6)


def test_track_speed_series_short_track_no_crash():
    track = Track(id=1, take_id=0, boxes=[_box(t=0.0, cx=0.0, cy=0.0)])
    times, speeds = sprints.track_speed_series(track, window=3)
    assert times == [0.0]
    assert speeds == [0.0]


def test_detect_sprints_emits_event_for_sustained_fast_run():
    cfg = _events_config()
    # 2 seconds of a fast, sustained run (speed ~10 bbox-heights/s, threshold default 4.0)
    boxes = [_box(t=i * 0.1, cx=100.0 + i * 100.0, cy=100.0, conf=0.9) for i in range(20)]
    track = Track(id=7, take_id=2, boxes=boxes)

    events = sprints.detect_sprints(track, cfg, fps_sample=10.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.SPRINT
    assert ev.player_track_id == 7
    assert ev.take_id == 2  # never invents/loses the take id
    assert ev.evidence["calibrated"] is False
    assert ev.evidence["unit"] == "bbox_heights_per_second"
    assert ev.evidence["peak_speed"] > cfg["sprint"]["sprint_speed_threshold"]
    assert 0.0 <= ev.confidence <= 1.0


def test_detect_sprints_no_event_for_slow_track():
    cfg = _events_config()
    boxes = [_box(t=i * 0.1, cx=100.0 + i * 0.5, cy=100.0) for i in range(20)]  # barely moving
    track = Track(id=1, take_id=0, boxes=boxes)
    events = sprints.detect_sprints(track, cfg, fps_sample=10.0)
    assert events == []


def test_detect_sprints_too_short_track_skipped():
    cfg = _events_config()
    boxes = [_box(t=i * 0.1, cx=100.0 + i * 100.0, cy=100.0) for i in range(2)]
    track = Track(id=1, take_id=0, boxes=boxes)
    assert sprints.detect_sprints(track, cfg, fps_sample=10.0) == []


def test_segment_confidence_never_exceeds_uncalibrated_penalty():
    """Even a perfect track (full continuity, conf=1.0) must be penalised below 1.0 by ADR-6's
    uncalibrated_confidence_penalty -- a pixel-unit speed must never read as fully confident."""
    cfg = _events_config()
    boxes = [_box(t=i * 0.1, cx=float(i), cy=0.0, conf=1.0) for i in range(11)]
    track = Track(id=1, take_id=0, boxes=boxes)
    confidence, mean_conf, n = sprints.segment_confidence(track, 0.0, 1.0, cfg, fps_sample=10.0)
    assert mean_conf == pytest.approx(1.0)
    assert confidence <= cfg["sprint"]["uncalibrated_confidence_penalty"] + 1e-9


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------


def test_direction_consistency_all_same_sign_is_one():
    assert shots.direction_consistency([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_direction_consistency_mixed_signs_is_fractional():
    assert shots.direction_consistency([1.0, -1.0, 1.0, 1.0]) == pytest.approx(0.75)


def test_direction_consistency_all_zero_is_zero():
    assert shots.direction_consistency([0.0, 0.0]) == 0.0


def test_shot_confidence_hard_capped_at_max_confidence():
    cfg = _events_config()["shot"]
    # wildly exceeding the threshold must still never clear max_confidence
    confidence = shots.shot_confidence(
        peak_speed=1000.0, threshold=cfg["ball_speed_threshold"], shot_cfg=cfg
    )
    assert confidence == pytest.approx(cfg["max_confidence"])


def test_shot_confidence_floors_at_min_confidence():
    cfg = _events_config()["shot"]
    confidence = shots.shot_confidence(
        peak_speed=cfg["ball_speed_threshold"], threshold=cfg["ball_speed_threshold"], shot_cfg=cfg
    )
    assert confidence == pytest.approx(cfg["min_confidence"])


# ---------------------------------------------------------------------------
# goals -- must be an explicit "not available", never guessed (Golden Rule 5)
# ---------------------------------------------------------------------------


def test_goal_availability_is_explicitly_not_available_on_this_footage():
    cfg = _events_config()["goal"]
    result = check_goal_availability(profile=None, goal_cfg=cfg)
    assert result.available is False
    assert result.reason.startswith("not available")
    assert result.events == []


def test_detect_goals_scoreboard_delta_is_dead_code():
    from src.events.goals import detect_goals_scoreboard_delta

    with pytest.raises(NotImplementedError):
        detect_goals_scoreboard_delta([], [], _events_config()["goal"])
