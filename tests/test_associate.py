"""Pure-logic unit tests for src/annotations/associate.py (ADR-19 Stage 4) -- no GPU/video I/O.

`associate_annotation_to_track` itself does a real decode and is exercised only indirectly (via
monkeypatching) by the manual-events pipeline's own smoke test -- consistent with how other
I/O-heavy helpers in this codebase (e.g. `src.team.classifier.collect_track_crops`) are treated.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import src.annotations.associate as associate_mod
from src.annotations.associate import (
    associate_annotation_to_track,
    candidate_tracks_near_time,
    expanded_bbox_region,
    nearest_track_by_arrow_hint,
    nearest_track_by_colour,
    opencv_lab_to_cielab,
    read_jersey_number_for_candidates,
)
from src.common.io import load_yaml
from src.common.types import Annotation, BBox, EventType, Take, Track, TrackBox
from src.detect.overlay_mask import ArrowHint
from src.identity.jersey_ocr import OcrRead
from src.identity.jersey_parseq import ParseqRead
from src.identity.legibility import LegibilityRead

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
# opencv_lab_to_cielab -- bug fix 2026-08-31 (CLAUDE.md's own "no fabricated stats" rule caught a
# real unit mismatch: crop_lab_mean/per_track_lab_median return OpenCV's 8-bit-scaled Lab, while
# colour_reference_lab is written in TRUE CIELAB units -- see the function's own docstring).
# ---------------------------------------------------------------------------


def test_opencv_lab_to_cielab_white_round_trips_to_true_white():
    """Pure white through OpenCV's own BGR2LAB is exactly [255, 128, 128] -- must convert to true
    CIELAB [100, 0, 0], matching `colour_reference_lab: white`."""
    raw = cv2.cvtColor(np.uint8([[[255, 255, 255]]]), cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
    true_lab = opencv_lab_to_cielab(raw)
    assert np.allclose(true_lab, [100.0, 0.0, 0.0], atol=0.5)


def test_opencv_lab_to_cielab_black_round_trips_to_true_black():
    raw = cv2.cvtColor(np.uint8([[[0, 0, 0]]]), cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
    true_lab = opencv_lab_to_cielab(raw)
    assert np.allclose(true_lab, [0.0, 0.0, 0.0], atol=0.5)


def test_opencv_lab_to_cielab_matches_the_documented_scaling_formula():
    """L_true = L_cv*100/255, a_true = a_cv-128, b_true = b_cv-128 -- verified directly against
    an arbitrary raw value, not just the two degenerate white/black cases above."""
    raw = np.array([130.0, 150.0, 90.0])
    true_lab = opencv_lab_to_cielab(raw)
    assert np.allclose(true_lab, [130.0 * 100.0 / 255.0, 150.0 - 128.0, 90.0 - 128.0])


def test_opencv_lab_to_cielab_rejects_impossible_raw_values_as_the_smoking_gun():
    """A raw OpenCV L of 130-146 (measured on real cached data, see the config's own comment) is
    IMPOSSIBLE for true CIELAB (which caps at 100) -- this is exactly the tell that caught the
    original bug. After conversion it must land back in the valid [0, 100] range."""
    raw = np.array([140.0, 120.0, 135.0])  # measured-shape value, would be invalid as true CIELAB
    true_lab = opencv_lab_to_cielab(raw)
    assert 0.0 <= true_lab[0] <= 100.0


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


# ---------------------------------------------------------------------------
# Regression test for the unit-mismatch bug (2026-08-31): runs the REAL colour pipeline end to
# end (per_track_lab_median -> opencv_lab_to_cielab -> nearest_track_by_colour), only mocking
# `decode_frames` with a synthetic single-colour frame -- no live video I/O, but nothing about the
# Lab computation itself is mocked away. Before the fix this test fails (a raw, unconverted white
# crop sits ~232 units from the true-CIELAB white reference, far past any sane
# colour_match_max_distance); after the fix it must pass.
# ---------------------------------------------------------------------------


def test_associate_annotation_to_track_converts_units_before_colour_match_real_pipeline(
    monkeypatch,
):
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)  # solid white BGR
    monkeypatch.setattr(associate_mod, "decode_frames", lambda *a, **k: iter([(0, 5.0, frame)]))

    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")
    box = TrackBox(
        frame_index=50, t=5.0, bbox=BBox(x1=50, y1=50, x2=150, y2=190), conf=0.9
    )  # height=140 >= min_box_height_px=25
    track = Track(id=42, take_id=0, boxes=[box])
    ann = Annotation(
        t=5.0,
        jersey_number=1,
        team_colour="white",
        action_phrase="takes a touch",
        event_type=EventType.TOUCH,
        raw_line="Min 0:05 player #1 in white takes a touch",
    )
    cfg = _annotations_config()

    track_id, distance, evidence = associate_annotation_to_track(
        "fake.mp4",
        take,
        [track],
        ann,
        colour_cfg=cfg,
        decode_cfg={},
        sampling_cfg=cfg["track_association"],
        use_nvdec=False,
    )

    assert track_id == 42
    assert distance is not None and distance < cfg["colour_match_max_distance"]
    assert evidence["agreement"] == "colour_only"


# ---------------------------------------------------------------------------
# expanded_bbox_region (pure geometry)
# ---------------------------------------------------------------------------


def test_expanded_bbox_region_grows_by_fraction_on_each_side():
    bbox = BBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)  # width=100, height=200
    region = expanded_bbox_region(bbox, expand_frac=0.1)
    assert region.x1 == 90.0  # 100 - 0.1*100
    assert region.x2 == 210.0  # 200 + 0.1*100
    assert region.y1 == 80.0  # 100 - 0.1*200
    assert region.y2 == 320.0  # 300 + 0.1*200


def test_expanded_bbox_region_zero_fraction_is_a_no_op():
    bbox = BBox(x1=10.0, y1=20.0, x2=30.0, y2=90.0)
    region = expanded_bbox_region(bbox, expand_frac=0.0)
    assert (region.x1, region.y1, region.x2, region.y2) == (10.0, 20.0, 30.0, 90.0)


# ---------------------------------------------------------------------------
# read_jersey_number_for_candidates (Stage B, "streamed-gathering-treehouse" plan) -- mocked
# OCR/VLM and `decode_frames`, no GPU/live network. Two non-overlapping candidate boxes are
# stamped with distinct marker pixel values in a synthetic frame (`expand_frac=0.0` so the crop
# geometry is exact); a fake `read_jersey_digits`/`classify_jersey_number` reads the marker back
# out of `crop[0, 0, 0]` to decide which "digits" that candidate produces -- deterministic, no
# real OCR/VLM model involved.
# ---------------------------------------------------------------------------


def _jersey_reid_identity_cfg() -> dict:
    return {
        "crop": {
            "torso_crop_expand": 0.0,
            "min_crop_height_frac": 0.05,
            "crop_upscale_factor": 1.0,
        },
        "ocr": {},
        "vlm": {},
        "aggregation": {"min_agreeing_frames": 2, "min_verified_confidence": 0.0},
    }


def _jersey_reid_cfg(max_vlm: int = 6, max_frames: int = 2) -> dict:
    return {
        "enabled": True,
        "max_vlm_escalations_per_call": max_vlm,
        "max_frames_per_candidate": max_frames,
    }


def _marker_frame(height: int = 100, width: int = 300) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _stamp_box(frame: np.ndarray, box: BBox, marker: int) -> None:
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    frame[y1:y2, x1:x2, 0] = marker


def _candidate(track_id: int, times: list[float], box: BBox) -> Track:
    """A candidate with a box at the SAME location at each of `times` -- multiple lifespan
    instants to sample from (Stage B's revised multi-frame design), same crop content each time
    (the marker frame below stamps every candidate's box region identically regardless of t)."""
    boxes = [TrackBox(frame_index=int(round(tt * 10)), t=tt, bbox=box, conf=0.9) for tt in times]
    return Track(id=track_id, take_id=0, boxes=boxes)


def _take_for_reid() -> Take:
    return Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100, kind="main")


# fps_sample=10 (0.1s grid) + match_tolerance_s=0.15 -- fine enough that any real box time is
# always covered by a nearby decoded frame regardless of exactly where decode_start's own
# fractional offset lands (see _candidate_lifespan's own bug-fix docstring for why the decode
# window's own bounds matter here).
_SAMPLING_CFG = {"match_tolerance_s": 0.15, "window_s": 0.0, "fps_sample": 10}
_DECODE_CFG: dict = {}


def _fake_decode_over_range(frame: np.ndarray):
    """A fake `decode_frames` that marches through `[start, end]` at `fps` yielding the SAME
    (already-stamped) frame at each grid point -- mirrors real `decode_frames`' own
    `(frame_index, t, frame)` yield shape without any real ffmpeg/video I/O."""

    def _fake(video_path, fps=None, start=None, end=None, scale_width=None, use_nvdec=True):
        step = 1.0 / fps
        n_steps = int(round((end - start) / step)) + 1
        for i in range(max(1, n_steps)):
            tt = start + i * step
            if tt > end + 1e-6:
                break
            yield i, tt, frame.copy()

    return _fake


def _fake_ocr_by_marker(marker_to_digits: dict[int, str | None]):
    def _fake(reader, crop, ocr_cfg):
        marker = int(crop[0, 0, 0])
        digits = marker_to_digits.get(marker)
        return OcrRead(
            digits=digits, confidence=0.9 if digits else 0.0, is_confident=digits is not None
        )

    return _fake


def test_read_jersey_number_for_candidates_no_candidates_short_circuits(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("decode_frames must not be called with zero candidates")

    monkeypatch.setattr(associate_mod, "decode_frames", _boom)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=None,
        gemini_api_key=None,
    )
    assert (track_id, source) == (None, None)
    assert evidence["outcome"] == "no_candidates"


def test_read_jersey_number_for_candidates_matches_on_two_agreeing_ocr_frames(monkeypatch):
    """The core revised behaviour: a candidate is only confirmed once >= 2 INDEPENDENT frames
    (across its own lifespan, not one snapshot) agree on the claimed digit."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)  # height 60
    box_b = BBox(x1=100, y1=10, x2=130, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)  # reads "7" every time -- matches claimed
    _stamp_box(frame, box_b, marker=2)  # reads "3" every time -- doesn't match
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(associate_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "7", 2: "3"}))

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_b = _candidate(2, [4.0, 6.0], box_b)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a, track_b],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
    )
    assert (track_id, source) == (1, "ocr")
    assert evidence["outcome"] == "matched"
    assert evidence["n_ocr_confident"] == 4  # 2 frames x 2 candidates
    assert evidence["n_vlm_calls"] == 0


def test_read_jersey_number_for_candidates_single_agreeing_frame_is_not_enough(monkeypatch):
    """A candidate whose OWN reads only agree on the claimed number ONCE (not twice) must NOT be
    confirmed -- mirrors aggregate_take_identity's own "never verify off one frame" rule."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    # first read matches claimed (7), second read (a different, later instant) disagrees --
    # inconsistent evidence, never verified off a single agreeing frame.
    calls = {"n": 0}

    def _flaky_ocr(reader, crop, ocr_cfg):
        calls["n"] += 1
        digits = "7" if calls["n"] == 1 else "9"
        return OcrRead(digits=digits, confidence=0.9, is_confident=True)

    monkeypatch.setattr(associate_mod, "read_jersey_digits", _flaky_ocr)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
    )
    assert (track_id, source) == (None, None)
    assert evidence["outcome"] == "no_match"


def test_read_jersey_number_for_candidates_ambiguous_when_two_candidates_match(monkeypatch):
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    box_b = BBox(x1=100, y1=10, x2=130, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    _stamp_box(frame, box_b, marker=2)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(associate_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "7", 2: "7"}))

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_b = _candidate(2, [4.0, 6.0], box_b)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a, track_b],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
    )
    assert (track_id, source) == (None, None)
    assert evidence["outcome"] == "ambiguous_multiple_candidates_matched"
    assert evidence["ambiguous_track_ids"] == [1, 2]


def test_read_jersey_number_for_candidates_too_small_crop_is_skipped(monkeypatch):
    # height 60*0.05 gate would need frame_height*0.05 <= box height; make the box itself tiny.
    box_a = BBox(x1=10, y1=10, x2=15, y2=12)  # height=2, far below any real gate
    frame = _marker_frame()
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))

    def _boom(*a, **k):
        raise AssertionError("OCR must never be called on a crop that fails the size gate")

    monkeypatch.setattr(associate_mod, "read_jersey_digits", _boom)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
    )
    assert (track_id, source) == (None, None)
    assert evidence["n_too_small"] == 2  # both of its 2 sampled instants failed the size gate
    assert evidence["n_crops_considered"] == 0


def test_read_jersey_number_for_candidates_escalates_ambiguous_ocr_to_vlm(monkeypatch):
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    box_b = BBox(x1=100, y1=10, x2=130, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)  # VLM will read "7" every time -- matches
    _stamp_box(frame, box_b, marker=2)  # VLM will read "3" every time -- no match
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    # OCR is always ambiguous here, forcing VLM escalation for every sampled crop.
    monkeypatch.setattr(
        associate_mod,
        "read_jersey_digits",
        lambda reader, crop, cfg: OcrRead(digits=None, confidence=0.0, is_confident=False),
    )

    def _fake_vlm(crop, api_key, vlm_cfg):
        marker = int(crop[0, 0, 0])
        digits = {1: "7", 2: "3"}[marker]
        return digits, 0.75, f"[model=fake] NUMBER={digits}"

    monkeypatch.setattr(associate_mod, "classify_jersey_number", _fake_vlm)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_b = _candidate(2, [4.0, 6.0], box_b)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a, track_b],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(max_vlm=6),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key="fake-key",
    )
    assert (track_id, source) == (1, "vlm")
    assert evidence["outcome"] == "matched"
    assert evidence["n_vlm_calls"] == 4  # 2 frames x 2 candidates


def test_read_jersey_number_for_candidates_vlm_budget_bounds_real_cost(monkeypatch):
    """Cost-bound proof: with the VLM escalation cap set to 2 (enough for exactly ONE candidate's
    own 2 sampled frames, checked first in deterministic track-id order), the SECOND candidate --
    the one that would actually have matched -- never gets checked at all. The honest result is
    "no_match", not a guess -- the documented, accepted trade-off of bounding cost (CLAUDE.md
    Golden Rule 5: an honest miss, never a fabricated hit)."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)  # track 1 -- checked first, does NOT match
    box_b = BBox(x1=100, y1=10, x2=130, y2=70)  # track 2 -- would match, budget runs out first
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    _stamp_box(frame, box_b, marker=2)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(
        associate_mod,
        "read_jersey_digits",
        lambda reader, crop, cfg: OcrRead(digits=None, confidence=0.0, is_confident=False),
    )

    def _fake_vlm(crop, api_key, vlm_cfg):
        marker = int(crop[0, 0, 0])
        digits = {1: "3", 2: "7"}[marker]
        return digits, 0.75, f"[model=fake] NUMBER={digits}"

    monkeypatch.setattr(associate_mod, "classify_jersey_number", _fake_vlm)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_b = _candidate(2, [4.0, 6.0], box_b)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a, track_b],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(max_vlm=2),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key="fake-key",
    )
    assert (track_id, source) == (None, None)
    assert evidence["outcome"] == "no_match"
    assert evidence["n_vlm_calls"] == 2  # budget spent entirely on track 1, track 2 never checked


def test_read_jersey_number_for_candidates_priority_track_id_gets_first_claim_on_vlm_budget(
    monkeypatch,
):
    """Real bug fix, 2026-08-31 (`chelsea_burnley_target10`, 15 near-time candidates, VLM budget
    of 6 -- the budget was exhausted on two unrelated candidates before ever reaching the one
    colour itself picked). SAME tight-budget setup as the test above (track 2 is the one that
    would actually match, track 1 would not) -- but this time `priority_track_id=2` is given
    (colour's own pick), so track 2 is checked FIRST and gets its fair chance despite the
    identical budget that starved it in the un-prioritized test above."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)  # track 1 -- NOT the priority pick
    box_b = BBox(x1=100, y1=10, x2=130, y2=70)  # track 2 -- the priority pick, matches
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    _stamp_box(frame, box_b, marker=2)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(
        associate_mod,
        "read_jersey_digits",
        lambda reader, crop, cfg: OcrRead(digits=None, confidence=0.0, is_confident=False),
    )

    def _fake_vlm(crop, api_key, vlm_cfg):
        marker = int(crop[0, 0, 0])
        digits = {1: "3", 2: "7"}[marker]
        return digits, 0.75, f"[model=fake] NUMBER={digits}"

    monkeypatch.setattr(associate_mod, "classify_jersey_number", _fake_vlm)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_b = _candidate(2, [4.0, 6.0], box_b)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a, track_b],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(max_vlm=2),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key="fake-key",
        priority_track_id=2,
    )
    assert (track_id, source) == (2, "vlm")
    assert evidence["outcome"] == "matched"
    assert evidence["n_vlm_calls"] == 2  # both spent on the prioritized track 2, which matched


# ---------------------------------------------------------------------------
# Chain-priority logic (owner-authorized 2026-08-31, CLAUDE.md §7): legibility gate ->
# PARSeq-SoccerNet -> EasyOCR -> VLM, mirroring src/identity/verify.py's own
# `_collect_reads_for_take` chain, now reused by manual-mode reacquisition too. All three model
# stages are mocked -- no real checkpoint/GPU/network involved.
# ---------------------------------------------------------------------------


def _fake_legible_by_marker(legible_markers: set[int]):
    def _fake(crop, model, legibility_cfg):
        marker = int(crop[0, 0, 0])
        is_legible_ = marker in legible_markers
        return LegibilityRead(confidence=0.9 if is_legible_ else 0.1, is_legible=is_legible_)

    return _fake


def _fake_parseq_by_marker(marker_to_digits: dict[int, str | None], confident_markers: set[int]):
    def _fake(crop, model, transform, parseq_cfg):
        marker = int(crop[0, 0, 0])
        digits = marker_to_digits.get(marker)
        is_confident = marker in confident_markers and digits is not None
        return ParseqRead(
            digits=digits,
            confidence=0.9 if is_confident else 0.2,
            is_confident=is_confident,
            raw_text=digits,
        )

    return _fake


def test_read_jersey_number_for_candidates_confident_legible_parseq_wins_outright(monkeypatch):
    """A legible crop that PARSeq reads confidently and correctly must win WITHOUT ever falling
    through to OCR -- the chain's own cost/priority ordering, not just its correctness."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(associate_mod, "is_legible", _fake_legible_by_marker({1}))
    monkeypatch.setattr(
        associate_mod, "read_jersey_number_parseq", _fake_parseq_by_marker({1: "7"}, {1})
    )

    def _boom_ocr(*a, **k):
        raise AssertionError("OCR must not run once PARSeq already confidently read the digit")

    monkeypatch.setattr(associate_mod, "read_jersey_digits", _boom_ocr)

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert (track_id, source) == (1, "parseq_soccernet")
    assert evidence["outcome"] == "matched"
    assert evidence["n_legible"] == 2
    assert evidence["n_parseq_confident"] == 2
    assert evidence["n_ocr_confident"] == 0


def test_read_jersey_number_for_candidates_illegible_crop_falls_back_to_ocr(monkeypatch):
    """A crop the legibility gate rejects must never reach PARSeq at all -- OCR is the honest
    fallback, exactly the prior (pre-PARSeq) behaviour."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(associate_mod, "is_legible", _fake_legible_by_marker(set()))  # illegible

    def _boom_parseq(*a, **k):
        raise AssertionError("PARSeq must not run on a crop the legibility gate rejected")

    monkeypatch.setattr(associate_mod, "read_jersey_number_parseq", _boom_parseq)
    monkeypatch.setattr(associate_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "7"}))

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert (track_id, source) == (1, "ocr")
    assert evidence["n_legible"] == 0
    assert evidence["n_parseq_confident"] == 0
    assert evidence["n_ocr_confident"] == 2


def test_read_jersey_number_for_candidates_legible_but_unconfident_parseq_falls_back_to_ocr(
    monkeypatch,
):
    """A legible crop where PARSeq's OWN read is unconfident must still fall through to OCR --
    legibility alone does not short-circuit the chain, only a CONFIDENT PARSeq read does."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))
    monkeypatch.setattr(associate_mod, "is_legible", _fake_legible_by_marker({1}))
    # PARSeq "reads" a wrong/low-confidence digit -- confident_markers is empty, so it never wins.
    monkeypatch.setattr(
        associate_mod, "read_jersey_number_parseq", _fake_parseq_by_marker({1: "3"}, set())
    )
    monkeypatch.setattr(associate_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "7"}))

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert (track_id, source) == (1, "ocr")
    assert evidence["n_legible"] == 2
    assert evidence["n_parseq_confident"] == 0
    assert evidence["n_ocr_confident"] == 2


def test_read_jersey_number_for_candidates_none_models_preserve_prior_ocr_only_behaviour(
    monkeypatch,
):
    """Default `legibility_model=None, parseq_model=None` (every pre-existing caller/test) must
    behave EXACTLY as before this chain existed: legibility/PARSeq never even get referenced."""
    box_a = BBox(x1=10, y1=10, x2=40, y2=70)
    frame = _marker_frame()
    _stamp_box(frame, box_a, marker=1)
    monkeypatch.setattr(associate_mod, "decode_frames", _fake_decode_over_range(frame))

    def _boom(*a, **k):
        raise AssertionError("legibility/PARSeq must not be referenced when both models are None")

    monkeypatch.setattr(associate_mod, "is_legible", _boom)
    monkeypatch.setattr(associate_mod, "read_jersey_number_parseq", _boom)
    monkeypatch.setattr(associate_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "7"}))

    track_a = _candidate(1, [4.0, 6.0], box_a)
    track_id, source, evidence = read_jersey_number_for_candidates(
        "fake.mp4",
        _take_for_reid(),
        [track_a],
        t=5.0,
        claimed_jersey_number=7,
        identity_cfg=_jersey_reid_identity_cfg(),
        jersey_reid_cfg=_jersey_reid_cfg(),
        decode_cfg=_DECODE_CFG,
        sampling_cfg=_SAMPLING_CFG,
        reader=object(),
        gemini_api_key=None,
    )
    assert (track_id, source) == (1, "ocr")
    assert evidence["n_legible"] == 0
    assert evidence["n_parseq_confident"] == 0
