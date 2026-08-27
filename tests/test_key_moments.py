"""Pure-logic unit tests for ADR-14's celebration/key-moment pre-filter + response parsing
(CLAUDE.md §13.3) — no GPU/network calls.
"""

from __future__ import annotations

from src.common.types import BBox, Track, TrackBox
from src.events.key_moments import _parse_verdict, find_candidate_windows

PREFILTER_CFG = {
    "smoothing_window_frames": 1,
    "speed_threshold": 1.0,
    "min_duration_s": 0.5,
    "merge_gap_s": 0.5,
    "max_candidates_per_take": 6,
}


def _track_with_burst(burst_start: float, burst_end: float, step: float = 0.1) -> Track:
    """A track that sits still, then moves fast for [burst_start, burst_end], then sits still
    again -- same construction style as tests/test_events.py's sprint tests."""
    boxes = []
    t = 0.0
    x = 0.0
    while t <= burst_end + 1.0:
        moving = burst_start <= t <= burst_end
        if moving:
            x += 20.0  # fast displacement -> high speed
        boxes.append(
            TrackBox(
                frame_index=int(round(t * 10)),
                t=t,
                bbox=BBox(x1=x, y1=0, x2=x + 20, y2=100),
                conf=0.9,
            )
        )
        t = round(t + step, 2)
    return Track(id=1, take_id=0, boxes=boxes)


def test_find_candidate_windows_detects_a_speed_burst():
    track = _track_with_burst(1.0, 2.0)
    candidates = find_candidate_windows(track, PREFILTER_CFG, existing_windows=[])
    assert len(candidates) == 1
    t0, t1, peak = candidates[0]
    assert 0.9 <= t0 <= 1.2
    assert peak > 0


def test_find_candidate_windows_excludes_already_explained_window():
    track = _track_with_burst(1.0, 2.0)
    # the whole burst is already covered by an existing (e.g. sprint) event window
    candidates = find_candidate_windows(track, PREFILTER_CFG, existing_windows=[(0.9, 2.1)])
    assert candidates == []


def test_find_candidate_windows_caps_at_max_candidates():
    track = _track_with_burst(1.0, 1.6)
    cfg = {**PREFILTER_CFG, "max_candidates_per_take": 0}
    assert find_candidate_windows(track, cfg, existing_windows=[]) == []


def test_find_candidate_windows_no_motion_no_candidates():
    boxes = [
        TrackBox(frame_index=i, t=i * 0.1, bbox=BBox(x1=0, y1=0, x2=20, y2=100), conf=0.9)
        for i in range(20)
    ]
    track = Track(id=1, take_id=0, boxes=boxes)
    assert find_candidate_windows(track, PREFILTER_CFG, existing_windows=[]) == []


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_yes():
    is_yes, conf, raw = _parse_verdict("VERDICT=YES; players celebrating a goal together")
    assert is_yes
    assert conf > 0.0


def test_parse_verdict_no():
    is_yes, conf, raw = _parse_verdict("VERDICT=NO; this is ordinary run of play")
    assert not is_yes
    assert conf == 0.0


def test_parse_verdict_unparseable_defaults_to_no_never_fabricated_yes():
    is_yes, conf, raw = _parse_verdict("I'm not sure what you mean.")
    assert not is_yes
    assert conf == 0.0
    assert raw.startswith("UNPARSEABLE")
