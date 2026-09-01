"""Pure decode-logic unit tests for PARSeq-SoccerNet jersey reading (CLAUDE.md §7, ADR-15).

The real model/checkpoint is never loaded here -- `model`/`transform` are mocked so only
`read_jersey_number_parseq`'s own post-decode filtering logic (pure-digit acceptance, max_digits,
confidence threshold) is under test, matching this module's own documented Golden-Rule-5 stance:
never salvage a forced/truncated digit string out of a non-numeric decode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.identity.jersey_parseq import read_jersey_number_parseq  # noqa: E402

PARSEQ_CFG = {"min_confidence": 0.5, "max_digits": 2}


def _make_mock_model(raw_text: str, mean_conf: float, logits_shape=(1, 5, 40)):
    """A MagicMock standing in for a loaded `strhub` PARSeq model: callable (returns real logits
    so `.softmax(-1)` behaves like a real tensor), with a `tokenizer.decode` returning a fixed
    `(labels, confs)` pair -- exactly the two things `read_jersey_number_parseq` calls."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])  # -> device = cpu
    model.return_value = torch.randn(*logits_shape)
    model.tokenizer.decode.return_value = ([raw_text], [torch.full((3,), mean_conf)])
    return model


def _identity_transform(pil_img):
    return torch.zeros(3, 32, 128)


def _crop() -> np.ndarray:
    return np.zeros((40, 20, 3), dtype=np.uint8)


def test_confident_pure_digit_read_is_accepted():
    model = _make_mock_model("77", mean_conf=0.9)
    result = read_jersey_number_parseq(_crop(), model, _identity_transform, PARSEQ_CFG)
    assert result.digits == "77"
    assert result.is_confident is True
    assert result.source == "parseq_soccernet"
    assert result.confidence == pytest.approx(0.9)


def test_non_digit_decode_is_an_honest_non_match_never_salvaged():
    model = _make_mock_model("A7", mean_conf=0.95)  # high confidence, but not pure-digit
    result = read_jersey_number_parseq(_crop(), model, _identity_transform, PARSEQ_CFG)
    assert result.digits is None
    assert result.is_confident is False
    assert result.raw_text == "A7"


def test_three_digit_decode_rejected_by_max_digits():
    model = _make_mock_model("123", mean_conf=0.95)
    result = read_jersey_number_parseq(_crop(), model, _identity_transform, PARSEQ_CFG)
    assert result.digits is None
    assert result.is_confident is False


def test_pure_digit_but_below_confidence_threshold_is_not_confident():
    model = _make_mock_model("9", mean_conf=0.2)
    result = read_jersey_number_parseq(_crop(), model, _identity_transform, PARSEQ_CFG)
    # Digits are still surfaced (an honest low-confidence read), just not trusted as confident.
    assert result.digits == "9"
    assert result.is_confident is False


def test_empty_crop_never_crashes():
    model = _make_mock_model("7", mean_conf=0.9)
    result = read_jersey_number_parseq(
        np.zeros((0, 0, 3), dtype=np.uint8), model, _identity_transform, PARSEQ_CFG
    )
    assert result.digits is None
    assert result.confidence == 0.0
    assert result.is_confident is False


# ---------------------------------------------------------------------------
# ADR-21 -- number_region: the crop PARSeq actually reads.
#
# Root cause it fixes (measured 2026-09-01 on video2 take 0, 720p broadcast, 5 tracks with
# visually-confirmed ground truth #8/#4/#30/#19/#37): PARSeq resizes input to 128x32 (4:1,
# text-line shaped), so a tall full-body player crop squashes the digits into illegibility and
# the model returns a confident-but-WRONG read that is CONSISTENT per track -- which temporal
# voting cannot filter, because agreement measures consistency, not correctness. That produced
# the owner-reported "player 8 detected as player 10". Full-body scored 0/5; this region 5/5.
# ---------------------------------------------------------------------------

from src.identity.jersey_parseq import number_region  # noqa: E402

NUMBER_CROP_CFG = {
    "left_inset_frac": 0.15,
    "right_inset_frac": 0.15,
    "top_inset_frac": 0.12,
    "bottom_frac": 0.45,
}


def test_number_region_narrows_to_upper_torso():
    """A normal player box narrows to the upper-torso band, inset on both sides."""
    x1, y1, x2, y2 = number_region(100, 200, 200, 400, NUMBER_CROP_CFG)  # 100w x 200h
    assert (x1, x2) == (115, 185)  # 15% inset each side
    assert y1 == 224  # 200 + 12% of 200
    assert y2 == 290  # 200 + 45% of 200
    # the result must stay strictly inside the input box -- never expand it
    assert 100 <= x1 < x2 <= 200
    assert 200 <= y1 < y2 <= 400


def test_number_region_shifts_aspect_ratio_toward_text_line_shape():
    """The mechanism behind the fix: PARSeq resizes to 128x32 (4:1, wide). A full-body box is
    tall+narrow, so the digits get crushed vertically. The number region does not become strictly
    wide (a real 100x250 box yields 70x82), but it moves the width:height ratio SUBSTANTIALLY
    toward the text-line shape, which is what recovers the digits."""
    box = (0, 0, 100, 250)  # typical player box: much taller than wide
    full_ratio = (box[2] - box[0]) / (box[3] - box[1])
    nx1, ny1, nx2, ny2 = number_region(*box, NUMBER_CROP_CFG)
    region_ratio = (nx2 - nx1) / (ny2 - ny1)
    assert full_ratio == pytest.approx(0.40)
    assert region_ratio == pytest.approx(70 / 82, rel=1e-3)
    assert region_ratio > 2 * full_ratio  # >2x less vertical squashing


def test_number_region_degenerate_small_box_returns_input_unchanged():
    """A box too small to inset meaningfully falls back to the original rect -- never an inverted
    or zero-area crop that would silently become an empty read downstream."""
    assert number_region(10, 10, 13, 13, NUMBER_CROP_CFG) == (10, 10, 13, 13)


def test_number_region_zero_area_box_returns_input_unchanged():
    assert number_region(50, 50, 50, 50, NUMBER_CROP_CFG) == (50, 50, 50, 50)
    assert number_region(50, 50, 40, 30, NUMBER_CROP_CFG) == (50, 50, 40, 30)
