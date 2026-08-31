"""Pure-logic unit tests for Stage 5.5 (ADR-15, CLAUDE.md §3.3/§5.1) — no GPU/network calls.

Covers: `aggregate_take_identity`'s temporal-agreement rule (including the "never break a tie"
rule), `read_jersey_digits`'s confident-vs-ambiguous classification (mocked EasyOCR reader, no
real model load), and `jersey_vlm`'s response parsing/retry behaviour (mocked `requests`, no real
network call).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src.identity.jersey_ocr import read_jersey_digits
from src.identity.jersey_vlm import _parse_response, classify_jersey_number
from src.identity.verify import FrameRead, aggregate_take_identity

AGG_CFG = {"min_agreeing_frames": 2, "min_verified_confidence": 0.3}


def _read(t: float, idx: int, source: str, digits: str | None, conf: float) -> FrameRead:
    return FrameRead(t=t, frame_index=idx, source=source, digits=digits, confidence=conf)


# ---------------------------------------------------------------------------
# Stage A -- resolution-aware jersey-crop size gate (configs/identity.yaml: crop.
# min_crop_height_frac). Pure arithmetic + the one real config value, no GPU/network.
# ---------------------------------------------------------------------------


def _load_min_crop_height_frac() -> float:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "identity.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg["crop"]["min_crop_height_frac"]


def test_min_crop_height_frac_reproduces_original_4k_calibration():
    # The original absolute gate (through 2026-08-27) was 120px on native 3840x2160 decode.
    # The fraction must reproduce that exact pixel threshold at the resolution it was calibrated
    # on, so the already-verified 4K behaviour is unchanged by expressing it as a fraction.
    frac = _load_min_crop_height_frac()
    native_h_4k = 2160
    assert round(frac * native_h_4k) == 120


def test_min_crop_height_frac_is_permissive_enough_for_measured_720p_boxes():
    # Measured 2026-08-31 on real 720p broadcast footage (input/video2, video3): mean player
    # bbox height 71.9px, median 70.7px. The old fixed 120px gate let through only 0.63% of
    # boxes; the resolution-aware fraction must clear comfortably below that measured mean/median
    # so the OCR/VLM re-identification signal (Stage B) is not silently starved on 720p input.
    frac = _load_min_crop_height_frac()
    native_h_720p = 720
    threshold_px = frac * native_h_720p
    assert threshold_px < 70.7  # below the measured median box height
    assert threshold_px > 0  # still a real, non-trivial gate -- not a no-op


# ---------------------------------------------------------------------------
# aggregate_take_identity
# ---------------------------------------------------------------------------


def test_aggregate_verifies_on_two_agreeing_frames():
    reads = [_read(0.0, 0, "ocr", "7", 0.8), _read(0.5, 1, "ocr", "7", 0.9)]
    result = aggregate_take_identity(reads, AGG_CFG)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 7
    assert result["evidence_frames"] == [0, 1]


def test_aggregate_unverified_on_single_frame_only():
    reads = [_read(0.0, 0, "ocr", "7", 0.95)]
    result = aggregate_take_identity(reads, AGG_CFG)
    assert result["status"] == "unverified"
    assert result["jersey_number"] is None


def test_aggregate_never_breaks_a_tie():
    # 2 reads of "7" and 2 reads of "9" -- an exact tie for the top count must NEVER be resolved
    # arbitrarily (owner's explicit rule).
    reads = [
        _read(0.0, 0, "ocr", "7", 0.8),
        _read(0.5, 1, "ocr", "7", 0.8),
        _read(1.0, 2, "ocr", "9", 0.8),
        _read(1.5, 3, "ocr", "9", 0.8),
    ]
    result = aggregate_take_identity(reads, AGG_CFG)
    assert result["status"] == "unverified"
    assert result["jersey_number"] is None


def test_aggregate_picks_the_strict_majority_over_a_minority_read():
    reads = [
        _read(0.0, 0, "ocr", "7", 0.8),
        _read(0.5, 1, "ocr", "7", 0.8),
        _read(1.0, 2, "vlm", "7", 0.75),
        _read(1.5, 3, "ocr", "3", 0.8),
    ]
    result = aggregate_take_identity(reads, AGG_CFG)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 7
    assert result["evidence_frames"] == [0, 1, 2]


def test_aggregate_empty_reads_is_unverified():
    assert aggregate_take_identity([], AGG_CFG)["status"] == "unverified"


def test_aggregate_no_digit_reads_at_all_is_unverified():
    reads = [_read(0.0, 0, "ocr", None, 0.0), _read(0.5, 1, "vlm", None, 0.15)]
    assert aggregate_take_identity(reads, AGG_CFG)["status"] == "unverified"


def test_aggregate_low_confidence_agreement_still_gated_by_min_verified_confidence():
    cfg = {"min_agreeing_frames": 2, "min_verified_confidence": 0.9}  # unreachably high floor
    reads = [_read(0.0, 0, "ocr", "7", 0.5), _read(0.5, 1, "ocr", "7", 0.5)]
    result = aggregate_take_identity(reads, cfg)
    assert result["status"] == "unverified"
    assert result["jersey_number"] is None


# ---------------------------------------------------------------------------
# aggregate_take_identity -- ADR-18 (1): human-confirmed jersey override (Golden Rule 4)
# ---------------------------------------------------------------------------


def test_aggregate_human_override_verifies_on_a_single_supporting_read():
    # normally a single frame is NOT enough (see test_aggregate_unverified_on_single_frame_only
    # above) -- a human-confirmed number needs only ONE supporting read (Golden Rule 4).
    reads = [_read(0.0, 0, "ocr", "9", 0.6)]
    result = aggregate_take_identity(reads, AGG_CFG, human_confirmed_jersey=9)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 9
    assert result["evidence_frames"] == [0]
    assert "accepted" in result["human_override_note"]
    assert "DESPITE" not in result["human_override_note"]


def test_aggregate_human_override_wins_despite_a_contradicting_majority_but_flags_it():
    # 3 independent reads all say "7" -- a real automated majority -- but the human watched the
    # footage and confirmed "9", which only has ONE supporting read. Golden Rule 4: human
    # authority wins regardless; Golden Rule 5: the disagreement must be visible, never hidden.
    reads = [
        _read(0.0, 0, "ocr", "7", 0.9),
        _read(0.5, 1, "ocr", "7", 0.9),
        _read(1.0, 2, "vlm", "7", 0.9),
        _read(1.5, 3, "ocr", "9", 0.6),
    ]
    result = aggregate_take_identity(reads, AGG_CFG, human_confirmed_jersey=9)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 9
    assert "DESPITE" in result["human_override_note"]
    assert "7" in result["human_override_note"]


def test_aggregate_human_override_with_zero_supporting_reads_falls_back_to_automated_vote():
    # the human's number was never actually read anywhere in this take -- the override cannot
    # fabricate evidence, so it falls back to the normal automated vote (which here verifies "7").
    reads = [_read(0.0, 0, "ocr", "7", 0.9), _read(0.5, 1, "ocr", "7", 0.9)]
    result = aggregate_take_identity(reads, AGG_CFG, human_confirmed_jersey=42)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 7  # NOT 42 -- never fabricated
    assert "ZERO" in result["human_override_note"]


def test_aggregate_human_override_none_is_a_pure_no_op():
    # human_confirmed_jersey=None (the default) must behave IDENTICALLY to calling without it.
    reads = [_read(0.0, 0, "ocr", "7", 0.8), _read(0.5, 1, "ocr", "7", 0.9)]
    result = aggregate_take_identity(reads, AGG_CFG, human_confirmed_jersey=None)
    assert result["status"] == "verified"
    assert result["jersey_number"] == 7
    assert result["human_override_note"] is None


# ---------------------------------------------------------------------------
# jersey_ocr.read_jersey_digits (mocked EasyOCR reader)
# ---------------------------------------------------------------------------

OCR_CFG = {"min_easyocr_confidence": 0.4, "max_digits": 2}


def _fake_reader(results: list[tuple[str, float]]) -> MagicMock:
    reader = MagicMock()
    reader.readtext.return_value = [((0, 0, 0, 0), text, conf) for text, conf in results]
    return reader


def test_read_jersey_digits_confident_single_read():
    import numpy as np

    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    reader = _fake_reader([("7", 0.9)])
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert result.is_confident
    assert result.digits == "7"


def test_read_jersey_digits_ambiguous_two_disagreeing_strong_reads():
    import numpy as np

    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    reader = _fake_reader([("7", 0.9), ("17", 0.85)])
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert not result.is_confident


def test_read_jersey_digits_silent_no_candidates():
    import numpy as np

    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    reader = _fake_reader([])
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert result.digits is None
    assert not result.is_confident


def test_read_jersey_digits_discards_three_digit_noise():
    import numpy as np

    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    reader = _fake_reader([("123", 0.99)])  # not a plausible jersey number, max_digits=2
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert result.digits is None
    assert not result.is_confident


def test_read_jersey_digits_below_threshold_is_ambiguous_not_confident():
    import numpy as np

    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    reader = _fake_reader([("7", 0.2)])  # below min_easyocr_confidence
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert not result.is_confident


def test_read_jersey_digits_empty_crop_never_crashes():
    import numpy as np

    crop = np.zeros((0, 0, 3), dtype=np.uint8)
    reader = _fake_reader([("7", 0.9)])
    result = read_jersey_digits(reader, crop, OCR_CFG)
    assert result.digits is None
    reader.readtext.assert_not_called()


# ---------------------------------------------------------------------------
# jersey_vlm response parsing + retry (mocked requests, no real network call)
# ---------------------------------------------------------------------------


def test_parse_response_confident_number():
    digits, conf, raw = _parse_response("NUMBER=7; clearly visible on the back of the shirt")
    assert digits == "7"
    assert 0.0 < conf < 1.0


def test_parse_response_honest_unknown_abstention():
    digits, conf, raw = _parse_response("NUMBER=UNKNOWN; player is facing away from camera")
    assert digits is None
    assert conf > 0.0  # a real abstention, NOT a failed call
    assert not raw.startswith("CALL_FAILED")


def test_parse_response_unparseable_is_tagged_call_failed():
    digits, conf, raw = _parse_response("I cannot help with that request.")
    assert digits is None
    assert conf == 0.0
    assert raw.startswith("CALL_FAILED")


VLM_CFG = {
    "api_url_template": "https://example.invalid/{model}:generateContent",
    "models": ["model-a", "model-b"],
    "timeout_s": 5,
    "max_retries": 3,
    "retry_backoff_base_s": 0.0,  # no real sleeping in tests
    "temperature": 0.0,
    "jpeg_quality": 90,
}


def _ok_response(text: str) -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return resp


def _quota_exhausted_response() -> MagicMock:
    resp = MagicMock(status_code=429, text='{"error": {"status": "RESOURCE_EXHAUSTED"}}')
    resp.json.return_value = {"error": {"status": "RESOURCE_EXHAUSTED", "code": 429}}
    return resp


def test_classify_jersey_number_retries_then_succeeds():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    ok_response = _ok_response("NUMBER=9; visible")
    fail_response = MagicMock(status_code=503, text="temporarily unavailable")

    with patch("src.common.gemini.requests.post", side_effect=[fail_response, ok_response]):
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits == "9"
    assert "[model=model-a]" in raw  # succeeded on the FIRST model after one retry


def test_classify_jersey_number_exhausts_all_models_and_reports_call_failed():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    fail_response = MagicMock(status_code=503, text="still unavailable")

    with patch("src.common.gemini.requests.post", return_value=fail_response):
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits is None
    assert conf == 0.0
    assert raw.startswith("CALL_FAILED")


def test_classify_jersey_number_non_retryable_error_advances_to_next_model():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    bad_key_response = MagicMock(status_code=401, text="invalid api key")

    with patch("src.common.gemini.requests.post", return_value=bad_key_response) as mock_post:
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits is None
    assert raw.startswith("CALL_FAILED")
    # ONE call per model (never retried a non-retryable 401 on the SAME model), not one total
    assert mock_post.call_count == len(VLM_CFG["models"])


def test_classify_jersey_number_daily_quota_exhausted_falls_back_to_next_model():
    """The exact scenario a real run hit 2026-08-27: model-a's daily quota is gone (429
    RESOURCE_EXHAUSTED) -- must NOT retry model-a (no point until tomorrow), must fall back to
    model-b immediately, and model-b's real answer must be trusted, not discarded."""
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    quota_response = _quota_exhausted_response()
    ok_response = _ok_response("NUMBER=14; clearly visible")

    with patch(
        "src.common.gemini.requests.post", side_effect=[quota_response, ok_response]
    ) as mock_post:
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits == "14"
    assert "[model=model-b]" in raw
    # exactly 2 calls: ONE attempt on model-a (no retry burned on a quota exhaustion), then
    # model-b succeeds on its first try
    assert mock_post.call_count == 2


def test_classify_jersey_number_all_models_quota_exhausted_is_call_failed_never_fabricated():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    with patch(
        "src.common.gemini.requests.post", return_value=_quota_exhausted_response()
    ) as mock_post:
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits is None
    assert conf == 0.0
    assert raw.startswith("CALL_FAILED")
    assert mock_post.call_count == len(VLM_CFG["models"])  # one attempt per model, no retries
