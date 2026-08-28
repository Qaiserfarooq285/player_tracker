"""Pure-logic unit tests for src/annotations/associate.py (ADR-19 Stage 4) -- no GPU/video I/O.

`associate_annotation_to_track` itself does a real decode and is exercised only indirectly (via
monkeypatching) by the manual-events pipeline's own smoke test -- consistent with how other
I/O-heavy helpers in this codebase (e.g. `src.team.classifier.collect_track_crops`) are treated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.annotations.associate import candidate_tracks_near_time, nearest_track_by_colour
from src.common.io import load_yaml
from src.common.types import BBox, Track, TrackBox

REPO_ROOT = Path(__file__).resolve().parents[1]


def _annotations_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "annotations.yaml")


def _box(t: float) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=0, y1=0, x2=20, y2=100), conf=0.9)


def _track(tid: int, times: list[float]) -> Track:
    return Track(id=tid, take_id=0, boxes=[_box(t) for t in times])


# ---------------------------------------------------------------------------
# candidate_tracks_near_time
# ---------------------------------------------------------------------------


def test_candidate_tracks_near_time_includes_track_within_tolerance():
    track = _track(1, [10.0])
    assert candidate_tracks_near_time([track], t=10.1, tolerance_s=0.2) == [track]


def test_candidate_tracks_near_time_excludes_track_outside_tolerance():
    track = _track(1, [10.0])
    assert candidate_tracks_near_time([track], t=15.0, tolerance_s=0.2) == []


def test_candidate_tracks_near_time_multiple_candidates():
    a, b, c = _track(1, [10.0]), _track(2, [10.05]), _track(3, [50.0])
    result = candidate_tracks_near_time([a, b, c], t=10.0, tolerance_s=0.2)
    assert result == [a, b]


# ---------------------------------------------------------------------------
# nearest_track_by_colour
# ---------------------------------------------------------------------------


def test_nearest_track_by_colour_picks_the_closest_match():
    cfg = _annotations_config()
    lab_by_track_id = {
        1: np.array([99.0, 0.5, -0.5]),  # close to "white" [100, 0, 0]
        2: np.array([1.0, 0.0, 0.0]),  # close to "black" [0, 0, 0]
    }
    track_id, distance = nearest_track_by_colour(lab_by_track_id, "white", cfg)
    assert track_id == 1
    assert distance is not None and distance < cfg["colour_match_max_distance"]


def test_nearest_track_by_colour_unknown_colour_word_returns_none_none():
    cfg = _annotations_config()
    lab_by_track_id = {1: np.array([99.0, 0.0, 0.0])}
    assert nearest_track_by_colour(lab_by_track_id, "chartreuse", cfg) == (None, None)


def test_nearest_track_by_colour_no_candidates_returns_none_none():
    cfg = _annotations_config()
    assert nearest_track_by_colour({}, "white", cfg) == (None, None)


def test_nearest_track_by_colour_too_far_returns_none_with_distance():
    cfg = _annotations_config()
    # nothing here is anywhere near "white" [100, 0, 0]
    lab_by_track_id = {1: np.array([0.0, 0.0, 0.0])}
    track_id, distance = nearest_track_by_colour(lab_by_track_id, "white", cfg)
    assert track_id is None
    assert distance is not None and distance > cfg["colour_match_max_distance"]


def test_nearest_track_by_colour_case_insensitive():
    cfg = _annotations_config()
    lab_by_track_id = {1: np.array([99.0, 0.0, 0.0])}
    track_id, _distance = nearest_track_by_colour(lab_by_track_id, "WHITE", cfg)
    assert track_id == 1
