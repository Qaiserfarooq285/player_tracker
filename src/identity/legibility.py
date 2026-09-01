"""Legibility gate for jersey-number reading (CLAUDE.md §7 "mkoshkina/jersey-number-pipeline
CHECKPOINTS ONLY" row, owner-authorized 2026-08-31).

**License scoping (read before touching this file):** only the downloaded checkpoint file
`models/jersey-parseq-soccernet/legibility_resnet34_soccer_20240215.pth` is CC BY-NC 3.0
(non-commercial). The MODEL DEFINITION below is independently authored, ordinary project code — a
standard `torchvision.models.resnet34` backbone (Apache/BSD-licensed torchvision itself) with its
final `fc` layer replaced by a single `nn.Linear(num_ftrs, 1)`, exactly the architecture the
downloaded checkpoint's own `state_dict` keys and shapes confirm (verified 2026-08-31 by loading
the real checkpoint and inspecting `state_dict.keys()`/shapes directly, not assumed from a
secondhand description -- see the shapes below). This file deliberately contains NONE of
`mkoshkina/jersey-number-pipeline`'s own source (`networks.py`, `legibility_classifier.py`, ...),
which is itself under that repo's NC license, not just its weights.

**Verified checkpoint shape (2026-08-31):** `state_dict` has 218 keys, all prefixed `model_ft.`
(a submodule attribute name -- an ordinary Python naming choice, not copied code), a standard
resnet34 layer1..layer4 (BasicBlock counts 3/4/6/3, confirmed by the presence of
`model_ft.layer4.2.*` and absence of `model_ft.layer4.3.*`), and a final
`model_ft.fc.weight` of shape `(1, 512)` / `model_ft.fc.bias` of shape `(1,)` -- i.e. resnet34's
native 512-dim `fc.in_features` feeding a single logit, matching the brief's description exactly.
No shape mismatch was observed; `strict=True` state_dict loading succeeds.

Same "load once per video/take, free when done" convention as
`src.identity.jersey_ocr.load_easyocr_reader`/`free_easyocr_reader` (CLAUDE.md §11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger

logger = get_logger(__name__)

# ImageNet mean/std -- confirmed correct preprocessing for this checkpoint (2026-08-31): a
# torchvision resnet34 backbone loaded with pretrained ImageNet weights before its own
# soccer-jersey fine-tuning is always documented (torchvision's own model card) to expect
# ImageNet-normalized 224x224 RGB input; nothing in the checkpoint's own state_dict overrides this
# (no separate stored normalization buffers), so the standard values are the correct, honestly-
# sourced choice here, not a guess.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 256  # CLAUDE.md task spec's documented preprocessing size for this checkpoint.


class LegibilityRead(BaseModel):
    """One legibility-classifier pass over one crop.

    `confidence` is the raw sigmoid output (P(legible)); `is_legible` is that confidence
    thresholded against `identity.yaml: legibility.min_confidence` (Golden Rule 5: the caller gets
    both the boolean gate AND the underlying score, never a black-box yes/no).
    """

    confidence: float
    is_legible: bool


def _build_resnet34_legibility_net() -> Any:
    """Independently-authored model definition: `torchvision.models.resnet34` + a 1-unit linear
    head, matching the downloaded checkpoint's own verified `state_dict` shape (module docstring).
    Lazy `torch`/`torchvision` import (same convention as `jersey_ocr.load_easyocr_reader`) so
    importing this module never pulls torch for callers that only need pure logic.
    """
    import torch.nn as nn
    from torchvision.models import resnet34

    class _ResNet34Legibility(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Attribute name `model_ft` chosen ONLY because it matches the checkpoint's own
            # `state_dict` key prefix (`model_ft.*`) so `load_state_dict(strict=True)` succeeds --
            # an ordinary naming choice, not copied code.
            self.model_ft = resnet34(weights=None)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, 1)

        def forward(self, x):
            return self.model_ft(x)

    return _ResNet34Legibility()


def load_legibility_model(checkpoint_path: str | Path, device: str = "cuda") -> Any:
    """Load the legibility classifier ONCE per pipeline run (CLAUDE.md §11: never reload
    per-crop), same convention as `jersey_ocr.load_easyocr_reader`.

    Raises `RuntimeError` (never silently falls back) if the checkpoint's `state_dict` doesn't
    match this module's architecture -- per this task's own decision rule, a shape mismatch is
    important information to surface, not paper over.
    """
    import torch

    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model = _build_resnet34_legibility_net()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"legibility checkpoint {checkpoint_path!r} does not match the independently-authored "
            f"resnet34+linear(1) architecture (see src/identity/legibility.py's own module "
            f"docstring for the shapes this was verified against) -- refusing to force-load: {exc}"
        ) from exc
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()
    model = model.to("cuda" if use_cuda else "cpu")
    logger.info("loaded legibility classifier (device=%s)", "cuda" if use_cuda else "cpu")
    return model


def free_legibility_model(model: Any) -> None:
    """Explicitly free the legibility model's VRAM (CLAUDE.md §11)."""
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _preprocess(crop_bgr: np.ndarray) -> Any:
    """BGR crop -> a `(1, 3, 256, 256)` ImageNet-normalized float tensor on the model's device."""
    import torch

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def is_legible(crop_bgr: np.ndarray, model: Any, legibility_cfg: dict) -> LegibilityRead:
    """Run the legibility classifier over one BGR crop.

    A crop with zero pixels (defensive) reads as illegible with 0.0 confidence, never crashes --
    same defensive convention as `jersey_ocr.read_jersey_digits`.
    """
    import torch

    if crop_bgr.size == 0:
        return LegibilityRead(confidence=0.0, is_legible=False)

    device = next(model.parameters()).device
    tensor = _preprocess(crop_bgr).to(device)
    with torch.inference_mode():
        logit = model(tensor)
        confidence = torch.sigmoid(logit).item()

    threshold = legibility_cfg["min_confidence"]
    return LegibilityRead(confidence=confidence, is_legible=confidence >= threshold)
