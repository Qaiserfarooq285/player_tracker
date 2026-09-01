"""Chain-priority unit tests for `src.identity.verify._collect_reads_for_take` (CLAUDE.md §7,
owner-authorized 2026-08-31): legibility gate -> PARSeq-SoccerNet -> EasyOCR -> Gemini VLM.

No real checkpoint/GPU/network is used -- `is_legible`/`read_jersey_number_parseq`/
`read_jersey_digits`/`classify_jersey_number` are monkeypatched on the `src.identity.verify`
module itself, mirroring the existing marker-frame convention in `tests/test_associate.py`'s own
Stage-B jersey-reid tests (a fake "digit reader" looks up the crop's own stamped marker pixel
rather than running a real model).
"""

from __future__ import annotations

import numpy as np

import src.identity.verify as verify_mod
from src.common.types import BBox, Track, TrackBox
from src.identity.jersey_ocr import OcrRead
from src.identity.jersey_parseq import ParseqRead
from src.identity.legibility import LegibilityRead

FRAME_H, FRAME_W = 100, 300


def _identity_cfg() -> dict:
    return {
        "crop": {"min_crop_height_frac": 0.05, "torso_crop_expand": 0.0, "crop_upscale_factor": 1.0},
        "ocr": {},
        "vlm": {"max_escalations_per_take": 6},
        "legibility": {},
        "parseq_soccernet": {},
    }


def _marker_frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _track_with_box(marker: int) -> tuple[Track, np.ndarray]:
    box = BBox(x1=10, y1=10, x2=40, y2=70)  # height 60 >= min_height (0.05 * 100 = 5)
    frame = _marker_frame()
    frame[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2), 0] = marker
    track = Track(id=1, take_id=0, boxes=[TrackBox(frame_index=0, t=1.0, bbox=box, conf=0.9)])
    return track, frame


def _fake_legible_by_marker(legible_markers: set[int]):
    def _fake(crop, model, legibility_cfg):
        marker = int(crop[0, 0, 0])
        ok = marker in legible_markers
        return LegibilityRead(confidence=0.9 if ok else 0.1, is_legible=ok)

    return _fake


def _fake_parseq_by_marker(marker_to_digits: dict[int, str | None], confident_markers: set[int]):
    def _fake(crop, model, transform, parseq_cfg):
        marker = int(crop[0, 0, 0])
        digits = marker_to_digits.get(marker)
        ok = marker in confident_markers and digits is not None
        return ParseqRead(
            digits=digits, confidence=0.9 if ok else 0.2, is_confident=ok, raw_text=digits
        )

    return _fake


def _fake_ocr_by_marker(marker_to_digits: dict[int, str | None]):
    def _fake(reader, crop, ocr_cfg):
        marker = int(crop[0, 0, 0])
        digits = marker_to_digits.get(marker)
        return OcrRead(
            digits=digits, confidence=0.9 if digits else 0.0, is_confident=digits is not None
        )

    return _fake


def _run(track, frame, monkeypatch, gemini_api_key=None, **model_kwargs):
    frames_by_take = {0: [(0, 1.0, frame)]}
    return verify_mod._collect_reads_for_take(
        frames_by_take,
        [track],
        scale_x=1.0,
        scale_y=1.0,
        native_w=FRAME_W,
        native_h=FRAME_H,
        identity_cfg=_identity_cfg(),
        reader=object(),
        gemini_api_key=gemini_api_key,
        identity_fps_sample=10.0,
        **model_kwargs,
    )


def test_legible_confident_parseq_wins_outright_never_calls_ocr(monkeypatch):
    track, frame = _track_with_box(marker=1)
    monkeypatch.setattr(verify_mod, "is_legible", _fake_legible_by_marker({1}))
    monkeypatch.setattr(
        verify_mod, "read_jersey_number_parseq", _fake_parseq_by_marker({1: "7"}, {1})
    )

    def _boom(*a, **k):
        raise AssertionError("OCR must not run once PARSeq confidently read the digit")

    monkeypatch.setattr(verify_mod, "read_jersey_digits", _boom)

    reads, counters = _run(
        track,
        frame,
        monkeypatch,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert len(reads) == 1
    assert reads[0].source == "parseq_soccernet"
    assert reads[0].digits == "7"
    assert counters["n_legible"] == 1
    assert counters["n_parseq_confident"] == 1
    assert counters["n_ocr_confident"] == 0


def test_illegible_crop_falls_back_to_ocr(monkeypatch):
    track, frame = _track_with_box(marker=1)
    monkeypatch.setattr(verify_mod, "is_legible", _fake_legible_by_marker(set()))  # illegible

    def _boom(*a, **k):
        raise AssertionError("PARSeq must not run on a crop the legibility gate rejected")

    monkeypatch.setattr(verify_mod, "read_jersey_number_parseq", _boom)
    monkeypatch.setattr(verify_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "9"}))

    reads, counters = _run(
        track,
        frame,
        monkeypatch,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert len(reads) == 1
    assert reads[0].source == "ocr"
    assert reads[0].digits == "9"
    assert counters["n_legible"] == 0
    assert counters["n_parseq_confident"] == 0
    assert counters["n_ocr_confident"] == 1


def test_legible_but_unconfident_parseq_falls_back_to_ocr(monkeypatch):
    track, frame = _track_with_box(marker=1)
    monkeypatch.setattr(verify_mod, "is_legible", _fake_legible_by_marker({1}))
    # PARSeq reads SOMETHING but never clears its own confidence threshold.
    monkeypatch.setattr(
        verify_mod, "read_jersey_number_parseq", _fake_parseq_by_marker({1: "3"}, set())
    )
    monkeypatch.setattr(verify_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "9"}))

    reads, counters = _run(
        track,
        frame,
        monkeypatch,
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert len(reads) == 1
    assert reads[0].source == "ocr"
    assert reads[0].digits == "9"
    assert counters["n_legible"] == 1
    assert counters["n_parseq_confident"] == 0
    assert counters["n_ocr_confident"] == 1


def test_both_legibility_stack_and_ocr_fail_escalates_to_vlm(monkeypatch):
    track, frame = _track_with_box(marker=1)
    monkeypatch.setattr(verify_mod, "is_legible", _fake_legible_by_marker({1}))
    monkeypatch.setattr(
        verify_mod, "read_jersey_number_parseq", _fake_parseq_by_marker({}, set())
    )  # PARSeq finds nothing
    monkeypatch.setattr(verify_mod, "read_jersey_digits", _fake_ocr_by_marker({}))  # OCR silent

    def _fake_vlm(crop, api_key, vlm_cfg):
        return "5", 0.8, "raw"

    monkeypatch.setattr(verify_mod, "classify_jersey_number", _fake_vlm)

    reads, counters = _run(
        track,
        frame,
        monkeypatch,
        gemini_api_key="fake-key",
        legibility_model=object(),
        parseq_model=object(),
        parseq_transform=object(),
    )
    assert len(reads) == 1
    assert reads[0].source == "vlm"
    assert reads[0].digits == "5"
    assert counters["n_vlm_calls"] == 1


def test_none_models_preserve_prior_ocr_only_behaviour(monkeypatch):
    """Default `legibility_model=None, parseq_model=None` (every pre-existing caller) must behave
    EXACTLY as before this chain existed -- legibility/PARSeq are never even referenced."""
    track, frame = _track_with_box(marker=1)

    def _boom(*a, **k):
        raise AssertionError("legibility/PARSeq must not run when both models are None")

    monkeypatch.setattr(verify_mod, "is_legible", _boom)
    monkeypatch.setattr(verify_mod, "read_jersey_number_parseq", _boom)
    monkeypatch.setattr(verify_mod, "read_jersey_digits", _fake_ocr_by_marker({1: "9"}))

    reads, counters = _run(track, frame, monkeypatch)
    assert len(reads) == 1
    assert reads[0].source == "ocr"
    assert counters["n_legible"] == 0
    assert counters["n_parseq_confident"] == 0
