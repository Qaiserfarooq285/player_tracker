"""Pure-logic/architecture unit tests for the legibility gate (CLAUDE.md §7 "mkoshkina/
jersey-number-pipeline CHECKPOINTS ONLY" row, ADR-15). No real checkpoint download and no
network calls -- the model is built with random (untrained) weights via `torchvision.models.
resnet34(weights=None)`, exactly what `_build_resnet34_legibility_net` does, so these tests only
check shape/range/wiring, never accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.identity.legibility import (  # noqa: E402
    LegibilityRead,
    _build_resnet34_legibility_net,
    _preprocess,
    is_legible,
)

LEGIBILITY_CFG = {"min_confidence": 0.5}


def test_build_resnet34_legibility_net_forward_pass_shape():
    model = _build_resnet34_legibility_net()
    model.eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (1, 1)


def test_build_resnet34_legibility_net_sigmoid_output_in_unit_range():
    model = _build_resnet34_legibility_net()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.sigmoid(logits)
    assert probs.shape == (2, 1)
    assert bool((probs >= 0.0).all())
    assert bool((probs <= 1.0).all())


def test_preprocess_output_shape_and_dtype():
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    tensor = _preprocess(crop)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32


def test_is_legible_empty_crop_never_crashes():
    model = _build_resnet34_legibility_net()
    model.eval()
    result = is_legible(np.zeros((0, 0, 3), dtype=np.uint8), model, LEGIBILITY_CFG)
    assert result == LegibilityRead(confidence=0.0, is_legible=False)


def test_is_legible_thresholds_a_fixed_logit_above_and_below_boundary():
    # A trivial 1-parameter module standing in for the real resnet34 -- forces a KNOWN logit so
    # the threshold comparison itself (not resnet34's own untrained randomness) is what's tested.
    class _FixedLogit(torch.nn.Module):
        def __init__(self, logit_value: float) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([logit_value]))

        def forward(self, x):
            return self.bias.unsqueeze(0)

    crop = np.zeros((80, 40, 3), dtype=np.uint8)

    confident_model = _FixedLogit(5.0)  # sigmoid(5.0) ~= 0.993 -- clearly legible
    confident_model.eval()
    confident = is_legible(crop, confident_model, LEGIBILITY_CFG)
    assert confident.is_legible is True
    assert confident.confidence > 0.9

    illegible_model = _FixedLogit(-5.0)  # sigmoid(-5.0) ~= 0.007 -- clearly illegible
    illegible_model.eval()
    illegible = is_legible(crop, illegible_model, LEGIBILITY_CFG)
    assert illegible.is_legible is False
    assert illegible.confidence < 0.1
