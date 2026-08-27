"""Pure-logic unit tests for Stage 5.5 (ADR-15, CLAUDE.md §3.3/§5.1) — no GPU/network calls.

Covers: `aggregate_take_identity`'s temporal-agreement rule (including the "never break a tie"
rule), `read_jersey_digits`'s confident-vs-ambiguous classification (mocked EasyOCR reader, no
real model load), and `jersey_vlm`'s response parsing/retry behaviour (mocked `requests`, no real
network call).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.identity.jersey_ocr import read_jersey_digits
from src.identity.jersey_vlm import _parse_response, classify_jersey_number
from src.identity.verify import FrameRead, aggregate_take_identity

AGG_CFG = {"min_agreeing_frames": 2, "min_verified_confidence": 0.3}


def _read(t: float, idx: int, source: str, digits: str | None, conf: float) -> FrameRead:
    return FrameRead(t=t, frame_index=idx, source=source, digits=digits, confidence=conf)


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
    "model": "gemini-3.6-flash",
    "timeout_s": 5,
    "max_retries": 3,
    "retry_backoff_base_s": 0.0,  # no real sleeping in tests
    "temperature": 0.0,
    "jpeg_quality": 90,
}


def test_classify_jersey_number_retries_then_succeeds():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    ok_response = MagicMock(status_code=200)
    ok_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "NUMBER=9; visible"}]}}]
    }
    fail_response = MagicMock(status_code=503, text="temporarily unavailable")

    with patch("src.common.gemini.requests.post", side_effect=[fail_response, ok_response]):
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits == "9"


def test_classify_jersey_number_exhausts_retries_and_reports_call_failed():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    fail_response = MagicMock(status_code=503, text="still unavailable")

    with patch("src.common.gemini.requests.post", return_value=fail_response):
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits is None
    assert conf == 0.0
    assert raw.startswith("CALL_FAILED")


def test_classify_jersey_number_non_retryable_error_fails_fast():
    import numpy as np

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    bad_key_response = MagicMock(status_code=401, text="invalid api key")

    with patch("src.common.gemini.requests.post", return_value=bad_key_response) as mock_post:
        digits, conf, raw = classify_jersey_number(crop, "fake-key", VLM_CFG)
    assert digits is None
    assert raw.startswith("CALL_FAILED")
    mock_post.assert_called_once()  # never retried a non-retryable 401
