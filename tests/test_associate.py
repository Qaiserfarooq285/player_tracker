"""Pure-logic unit tests for src/annotations/associate.py (ADR-19 Stage 4) -- no GPU/video I/O.

`associate_annotation_to_track` itself does a real decode and is exercised only indirectly (via
monkeypatching) by the manual-events pipeline's own smoke test -- consistent with how other
I/O-heavy helpers in this codebase (e.g. `src.team.classifier.collect_track_crops`) are treated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.annotations.associate as associate_mod
from src.annotations.associate import (
    associate_annotation_to_track,
    candidate_tracks_near_time,
    nearest_track_by_arrow_hint,
    nearest_track_by_colour,
)
from src.common.io import load_yaml
from src.common.types import Annotation, BBox, EventType, Take, Track, TrackBox
from src.detect.overlay_mask import ArrowHint

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


# ---------------------------------------------------------------------------
# nearest_track_by_arrow_hint (owner request 2026-08-31: the burned-in-arrow corroborating
# signal, generic to every manual-mode video -- reuses the exact same tip-vs-box geometry test
# src.highlights.selection.vote_seed_tracks already trusts for auto seed selection, scoped to one
# annotation instant). Pure logic -- no video decode.
# ---------------------------------------------------------------------------


def _hint(t: float, tip_x: float, tip_y: float) -> ArrowHint:
    return ArrowHint(
        frame_index=int(t * 10),
        t=t,
        bbox=BBox(x1=tip_x - 5, y1=tip_y - 5, x2=tip_x + 5, y2=tip_y + 5),
        area_px=100,
        tip_x=tip_x,
        tip_y=tip_y,
    )


def _box_at(t: float, x1: float, y1: float, x2: float, y2: float) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), conf=0.9)


def test_arrow_hint_containing_box_wins_outright():
    """The tip lands INSIDE track 1's own box -- picked regardless of any other nearby track."""
    track1 = Track(id=1, take_id=0, boxes=[_box_at(5.0, 0, 0, 50, 100)])
    track2 = Track(id=2, take_id=0, boxes=[_box_at(5.0, 200, 200, 250, 300)])
    hints = [_hint(t=5.0, tip_x=25, tip_y=10)]  # inside track1's box

    track_id, distance = nearest_track_by_arrow_hint(
        hints, [track1, track2], t=5.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=200
    )
    assert track_id == 1
    assert distance == 0.0


def test_arrow_hint_nearest_by_centroid_when_no_box_contains_tip():
    """Tip is outside every box -- falls back to nearest centroid, within `max_dist_px`."""
    track1 = Track(id=1, take_id=0, boxes=[_box_at(5.0, 0, 0, 20, 20)])  # centroid (10, 10)
    track2 = Track(id=2, take_id=0, boxes=[_box_at(5.0, 100, 100, 120, 120)])  # centroid (110,110)
    hints = [_hint(t=5.0, tip_x=15, tip_y=15)]  # near track1's centroid, inside neither box...

    track_id, distance = nearest_track_by_arrow_hint(
        hints, [track1, track2], t=5.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=200
    )
    assert track_id == 1
    assert distance is not None and distance < 200


def test_arrow_hint_beyond_max_dist_returns_none():
    track1 = Track(id=1, take_id=0, boxes=[_box_at(5.0, 0, 0, 20, 20)])
    hints = [_hint(t=5.0, tip_x=1000, tip_y=1000)]  # nowhere near track1

    track_id, distance = nearest_track_by_arrow_hint(
        hints, [track1], t=5.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=50
    )
    assert track_id is None
    assert distance is None


def test_arrow_hint_outside_time_window_returns_none():
    track1 = Track(id=1, take_id=0, boxes=[_box_at(5.0, 0, 0, 20, 20)])
    hints = [_hint(t=5.0, tip_x=10, tip_y=10)]  # would match, but annotation is far in time

    track_id, distance = nearest_track_by_arrow_hint(
        hints, [track1], t=50.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=200
    )
    assert track_id is None
    assert distance is None


def test_arrow_hint_no_hints_at_all_returns_none():
    track1 = Track(id=1, take_id=0, boxes=[_box_at(5.0, 0, 0, 20, 20)])
    track_id, distance = nearest_track_by_arrow_hint(
        [], [track1], t=5.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=200
    )
    assert (track_id, distance) == (None, None)


def test_arrow_hint_no_candidates_returns_none():
    hints = [_hint(t=5.0, tip_x=10, tip_y=10)]
    track_id, distance = nearest_track_by_arrow_hint(
        hints, [], t=5.0, hint_window_s=1.0, match_tolerance_s=0.2, max_dist_px=200
    )
    assert (track_id, distance) == (None, None)


# ---------------------------------------------------------------------------
# associate_annotation_to_track's colour + arrow COMBINATION rule (colour stays primary, arrow
# corroborates/falls back, never silently overrides -- module docstring). `decode_frames` is
# monkeypatched to an empty generator (no real video I/O) and the two sub-signals
# (`nearest_track_by_colour`/`nearest_track_by_arrow_hint`) are monkeypatched directly so each
# combination branch can be exercised deterministically without a live decode.
# ---------------------------------------------------------------------------


def _ann(t: float = 5.0) -> Annotation:
    return Annotation(
        t=t,
        jersey_number=9,
        team_colour="white",
        action_phrase="takes a touch",
        event_type=EventType.TOUCH,
        raw_line="Min 0:05 player #9 in white takes a touch",
    )


def _setup_combination_test(monkeypatch, colour_result, arrow_result):
    monkeypatch.setattr(associate_mod, "decode_frames", lambda *a, **k: iter([]))
    monkeypatch.setattr(associate_mod, "nearest_track_by_colour", lambda *a, **k: colour_result)
    monkeypatch.setattr(
        associate_mod,
        "nearest_track_by_arrow_hint",
        lambda *a, **k: arrow_result,
    )
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    track = _track(1, [5.0])
    return take, [track]


def _run(monkeypatch, colour_result, arrow_result, arrow_hints):
    take, tracks = _setup_combination_test(monkeypatch, colour_result, arrow_result)
    return associate_annotation_to_track(
        "fake.mp4",
        take,
        tracks,
        _ann(),
        colour_cfg=_annotations_config(),
        decode_cfg={},
        sampling_cfg=_annotations_config()["track_association"],
        use_nvdec=False,
        arrow_hints=arrow_hints,
        selection_cfg={"arrow_match_tolerance_s": 0.15, "arrow_match_max_dist_px": 200},
    )


def test_combination_colour_only_when_no_arrow_hints_supplied(monkeypatch):
    """No arrow_hints at all (e.g. every existing caller before this feature) -> pure colour
    behaviour, unchanged."""
    track_id, distance, evidence = _run(monkeypatch, (1, 5.0), (1, 5.0), arrow_hints=None)
    assert (track_id, distance) == (1, 5.0)
    assert evidence["agreement"] == "colour_only"
    assert evidence["arrow_track_id"] is None


def test_combination_corroborated_when_colour_and_arrow_agree(monkeypatch):
    track_id, distance, evidence = _run(monkeypatch, (1, 5.0), (1, 42.0), arrow_hints=[object()])
    assert track_id == 1
    assert distance == 5.0  # colour's own distance is kept, not overwritten by the arrow's
    assert evidence["agreement"] == "corroborated"
    assert evidence["arrow_track_id"] == 1


def test_combination_disagreement_keeps_colour_pick_but_logs_it(monkeypatch):
    """Colour and arrow point at DIFFERENT tracks -- colour (the primary test) wins, but the
    disagreement is recorded, never silently dropped (Golden Rule 5)."""
    track_id, distance, evidence = _run(monkeypatch, (1, 5.0), (2, 10.0), arrow_hints=[object()])
    assert track_id == 1
    assert distance == 5.0
    assert evidence["agreement"] == "disagreement"
    assert evidence["arrow_track_id"] == 2


def test_combination_arrow_fallback_when_colour_finds_nothing(monkeypatch):
    """Colour has no confident match at all -- the arrow's own pick is used as an honest
    fallback, not left as a caption-only no-op."""
    track_id, distance, evidence = _run(
        monkeypatch, (None, 55.0), (2, 10.0), arrow_hints=[object()]
    )
    assert track_id == 2
    assert distance == 10.0
    assert evidence["agreement"] == "arrow_fallback"


def test_combination_neither_signal_confident_stays_caption_only(monkeypatch):
    track_id, distance, evidence = _run(
        monkeypatch, (None, 55.0), (None, None), arrow_hints=[object()]
    )
    assert (track_id, distance) == (None, None)
    assert evidence["agreement"] == "no_confident_signal"
