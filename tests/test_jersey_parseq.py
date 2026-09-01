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
