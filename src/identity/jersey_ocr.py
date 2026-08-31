"""ADR-15 — EasyOCR digit reading for jersey-number verification (CLAUDE.md §3.3/§5.1).

Apache-2.0 (`ocr` extra, CLAUDE.md §7). Crops are expected at NATIVE decode resolution (the
sharpest pixels available — `src/identity/verify.py` decodes without downscaling, unlike Stage 2's
1920px-wide detection pass), restricted to a digit-only allowlist so EasyOCR never wastes a read on
a sponsor logo's letters or a background sign.

Lazy `import easyocr`/`import torch` (same convention as `src/team/classifier.py::load_siglip`/
`src/detect/detector.py::load_detector`) so importing this module for `read_jersey_digits`'s pure
result-parsing logic never pulls torch — see `tests/test_identity.py`.
"""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger

logger = get_logger(__name__)


def upscale_crop(crop_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Upscale a jersey-number crop before handing it to EasyOCR/Gemini -- owner-reported bug fix,
    2026-08-31: standard, well-established OCR practice for small/blurry text (the pixels are
    genuinely low-detail either way; upscaling does not fabricate information, it just gives both
    the classical OCR engine and the VLM a larger, smoother version of the same evidence to work
    from, matching how EasyOCR/most OCR pipelines are documented to be used on small source text).
    `factor <= 1.0` or a degenerate (zero-area) crop is a no-op, never a crash. `cv2.INTER_CUBIC`
    is used deliberately (not the default `INTER_LINEAR`) -- it produces smoother, less blocky
    edges on small upscaled text, which is what actually helps a downstream digit read.
    Reused identically by `src/annotations/associate.py` (Stage B re-identification) and
    `src/identity/verify.py` (ADR-15's own per-take verification) -- the same crop-legibility
    limitation applies to both, confirmed on real 720p broadcast footage where a locked track's
    own box height measured only ~70-97px across its whole lifespan (CLAUDE.md §3.2's own
    "resolution ceiling" territory, not something either caller should solve independently).
    """
    if factor <= 1.0 or crop_bgr.size == 0:
        return crop_bgr
    height, width = crop_bgr.shape[:2]
    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return cv2.resize(crop_bgr, new_size, interpolation=cv2.INTER_CUBIC)


class OcrRead(BaseModel):
    """One EasyOCR pass over one crop.

    `is_confident` is True only when EXACTLY ONE distinct digit string clears
    `ocr.min_easyocr_confidence` — two different strong-but-disagreeing reads (or zero) are left
    `is_confident=False` (ambiguous) rather than picking one arbitrarily; `src/identity/verify.py`
    escalates ambiguous/silent crops to the VLM. `candidates` keeps every raw
    `(digits, confidence)` EasyOCR returned for traceability (Golden Rule 5), not just the winner.
    """

    digits: str | None
    confidence: float
    is_confident: bool
    candidates: list[tuple[str, float]] = []
    source: Literal["ocr"] = "ocr"


def load_easyocr_reader(ocr_cfg: dict) -> Any:
    """Load an EasyOCR `Reader` once per pipeline run (CLAUDE.md §11: never reload per-crop)."""
    import easyocr

    gpu = bool(ocr_cfg.get("gpu", True))
    logger.info("loading EasyOCR reader (gpu=%s)", gpu)
    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def free_easyocr_reader(reader: Any) -> None:
    """Explicitly free EasyOCR's VRAM (CLAUDE.md §11: unload each model before the next)."""
    import torch

    del reader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jersey_digits(reader: Any, crop_bgr: np.ndarray, ocr_cfg: dict) -> OcrRead:
    """Run EasyOCR (digit allowlist) over one BGR crop and classify the result as confident or
    ambiguous (see :class:`OcrRead` docstring). A crop with zero pixels (defensive — a caller
    should never pass one) reads as silent, never crashes.
    """
    if crop_bgr.size == 0:
        return OcrRead(digits=None, confidence=0.0, is_confident=False, candidates=[])

    raw = reader.readtext(crop_bgr, allowlist="0123456789")
    max_digits = ocr_cfg["max_digits"]
    candidates = [
        (text, float(conf))
        for _bbox, text, conf in raw
        if text.isdigit() and 1 <= len(text) <= max_digits
    ]
    if not candidates:
        return OcrRead(digits=None, confidence=0.0, is_confident=False, candidates=[])

    min_conf = ocr_cfg["min_easyocr_confidence"]
    strong = [c for c in candidates if c[1] >= min_conf]
    distinct_strong_digits = {c[0] for c in strong}

    if len(distinct_strong_digits) == 1:
        digits = next(iter(distinct_strong_digits))
        best_conf = max(c[1] for c in strong if c[0] == digits)
        return OcrRead(
            digits=digits, confidence=best_conf, is_confident=True, candidates=candidates
        )

    # Ambiguous: either nothing cleared the confidence bar, or 2+ different strong reads disagree
    # (e.g. two candidate boxes on the same crop reading "7" and "17") -- never pick a winner here,
    # the caller escalates to the VLM instead.
    best = max(candidates, key=lambda c: c[1])
    return OcrRead(digits=best[0], confidence=best[1], is_confident=False, candidates=candidates)
