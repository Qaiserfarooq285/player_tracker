"""PARSeq-SoccerNet jersey-number reader (CLAUDE.md §7 "mkoshkina/jersey-number-pipeline
CHECKPOINTS ONLY" row, owner-authorized 2026-08-31).

**License scoping (read before touching this file):** the PARSeq **architecture code** used here
is the REAL upstream `baudm/parseq` package (`strhub`, Apache-2.0, installed as a git dependency --
see `pyproject.toml`'s `jersey_parseq` extra) -- never any code from `mkoshkina/jersey-number-
pipeline`, which is itself NC-licensed. Only the downloaded **checkpoint file**
(`models/jersey-parseq-soccernet/parseq_epoch=24-...ckpt`, SoccerNet-fine-tuned by mkoshkina) is
CC BY-NC 3.0 -- see that directory's own `MODEL_CARD.md`.

**Real, verified checkpoint-format mismatch (2026-08-31) -- read before changing the remap logic
below.** The brief's secondhand research assumed this checkpoint's output could be decoded via
`logits[:, :3, :11].softmax(-1)` (implying an 11-class digit-only decoder, max_label_length~2).
That does NOT match the actual downloaded checkpoint: its own stored `hyper_parameters` show
`charset_test` is the STANDARD 36-character alnum set (`0123456789abcdefghijklmnopqrstuvwxyz`,
not a jersey-specific digit vocabulary) and `max_label_length=25` (the standard PARSeq-base
default), confirming this was fine-tuned from the stock "parseq" experiment config, not a custom
small-vocabulary head. Verified directly (`model.hparams`, `logits.shape == (1, 26, 95)`) rather
than trusting the secondhand description. **Resolution:** use PARSeq's own OFFICIAL, documented
public API instead of any manual logit slicing -- `logits.softmax(-1)` then
`model.tokenizer.decode(probs)` (exactly the pattern in `baudm/parseq`'s own README) -- and
POST-FILTER the decoded string to accept only a pure `\\d{1,2}` result as a jersey-number read,
honestly rejecting anything else (Golden Rule 5: never force digits out of a non-numeric decode).

**Second real mismatch, also verified, not assumed:** loading this checkpoint via the documented
`strhub.models.utils.load_from_checkpoint` (a `pytorch_lightning.LightningModule.load_from_
checkpoint` classmethod) FAILS with a `state_dict` key mismatch -- the checkpoint's own keys are
UNPREFIXED (`encoder.*`, `decoder.*`, `head.*`, `pos_queries`, `text_embed.*`), while the current
upstream `main` branch's `PARSeq` class nests all of those under `self.model = Model(...)` (keys
`model.encoder.*`, etc). This is a real, dated upstream refactor -- confirmed by inspecting
`baudm/parseq`'s own git history, commit `4cdf0bf` ("Separate model definition from training logic
of PARSeq"), whose diff shows those exact five attribute groups (`encoder`, `decoder`, `head`,
`pos_queries`, `text_embed`) moved VERBATIM (byte-identical forward logic) into a new `model.py`
submodule -- this checkpoint (file-dated Sep 2023) predates that Feb-2024 refactor. Pinning to the
exact pre-refactor commit was tried and rejected: that old code imports
`pytorch_lightning.utilities.types.EPOCH_OUTPUT`, removed in the modern `pytorch-lightning`
(2.6.x) this project already depends on for other reasons -- old code only runs against an old
`pytorch-lightning`, which is a much larger, riskier dependency downgrade than the alternative
below. **Resolution actually used:** stay on current upstream `strhub` (Apache-2.0, unmodified
code), and remap the checkpoint's own state_dict keys by prefixing every one with `"model."`
before loading into the CURRENT `PARSeq` class. This is NOT a guessed workaround -- it was
exhaustively verified (2026-08-31): building `PARSeq(**checkpoint['hyper_parameters'])` (current
upstream code) and comparing `{"model." + k for k in checkpoint['state_dict']}` against that
model's own `state_dict().keys()` gives an EXACT match, 175/175 keys, zero missing, zero extra,
zero shape mismatches -- i.e. the remap reproduces the current architecture's own state_dict key
set one-for-one, not an approximate fit. `_remap_and_load` re-asserts this exact-match invariant
every time (never a silent `strict=False`) so a future checkpoint that doesn't fit this exact
historical shape fails loudly instead of loading wrong weights silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger

logger = get_logger(__name__)


class ParseqRead(BaseModel):
    """One PARSeq-SoccerNet pass over one (already legibility-gated) crop.

    `is_confident` requires BOTH a pure-digit decode (never a truncated/forced digit extraction
    from a mixed alnum decode, Golden Rule 5) AND the decoded sequence's own mean per-token
    confidence clearing `parseq_soccernet.min_confidence`.
    """

    digits: str | None
    confidence: float
    is_confident: bool
    raw_text: str | None = None
    source: Literal["parseq_soccernet"] = "parseq_soccernet"


def number_region(
    x1: int, y1: int, x2: int, y2: int, number_crop_cfg: dict
) -> tuple[int, int, int, int]:
    """Narrow a full-body player box to the upper-torso NUMBER REGION that PARSeq should read.

    ADR-21, measured 2026-09-01. PARSeq resizes its input to 128x32 (4:1, text-line shaped), so a
    full-body crop (tall and narrow) squashes the digits vertically into illegibility -- the model
    then returns a confident but WRONG read which is CONSISTENT across that track's frames, so
    temporal voting cannot filter it (agreement measures consistency, not correctness). That was
    the mechanism behind the owner-reported "player 8 detected as player 10". Measured on 5
    visually-confirmed tracks: full-body 0/5 correct, this region 5/5.

    Insets come from `configs/identity.yaml: parseq_soccernet.number_crop` (no magic numbers,
    CLAUDE.md §10). The result is clamped to stay inside the input box and to keep a non-degenerate
    area; a box too small to narrow meaningfully is returned unchanged rather than inverted, so the
    caller always gets a usable rect.

    NOTE: the legibility gate must keep seeing the FULL-BODY crop -- mkoshkina's resnet34 was
    trained on whole-player images and rejects tight number crops (measured 0/743 vs 336/743).
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return x1, y1, x2, y2

    nx1 = x1 + int(round(w * number_crop_cfg["left_inset_frac"]))
    nx2 = x2 - int(round(w * number_crop_cfg["right_inset_frac"]))
    ny1 = y1 + int(round(h * number_crop_cfg["top_inset_frac"]))
    ny2 = y1 + int(round(h * number_crop_cfg["bottom_frac"]))

    # degenerate after inset (very small boxes) -> fall back to the original box, never an
    # inverted/empty rect that would silently become a zero-size crop downstream
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def _remap_and_load(model: Any, checkpoint: dict) -> None:
    """Load a pre-refactor (`self.encoder`/`self.decoder`/...) checkpoint's `state_dict` into the
    CURRENT upstream `PARSeq` architecture (`self.model.encoder`/`self.model.decoder`/...) by
    prefixing every checkpoint key with `"model."`.

    Raises `RuntimeError` (never a silent partial/`strict=False` load) if the remapped key set
    does not EXACTLY match the live model's own `state_dict()` keys -- see this module's own
    docstring for why this exact-match check is the thing that makes the remap trustworthy rather
    than a guess.
    """
    old_state_dict = checkpoint["state_dict"]
    remapped = {f"model.{k}": v for k, v in old_state_dict.items()}
    current_keys = set(model.state_dict().keys())
    remapped_keys = set(remapped.keys())
    missing = current_keys - remapped_keys
    extra = remapped_keys - current_keys
    if missing or extra:
        raise RuntimeError(
            "PARSeq-SoccerNet checkpoint remap ('model.' prefix) does not exactly match the "
            f"current upstream strhub PARSeq architecture -- missing={sorted(missing)}, "
            f"extra={sorted(extra)}. Refusing to force-load (see src/identity/jersey_parseq.py's "
            "own module docstring for the verified 2026-08-31 remap this checkpoint used to "
            "match exactly) -- this means either strhub's upstream `main` branch has changed "
            "again, or a different checkpoint format was supplied."
        )
    model.load_state_dict(remapped, strict=True)


def load_parseq_soccernet(checkpoint_path: str | Path, device: str = "cuda") -> tuple[Any, Any]:
    """Load the PARSeq-SoccerNet reader ONCE per pipeline run (CLAUDE.md §11: never reload
    per-crop), same convention as `jersey_ocr.load_easyocr_reader`.

    Returns `(model, transform)`: `model` is a current-upstream `strhub.models.parseq.system.
    PARSeq` (a `torch.nn.Module`/`LightningModule`, used here purely for inference), `transform`
    is PARSeq's own documented `SceneTextDataModule.get_transform(model.hparams.img_size)`
    preprocessing pipeline (resize + ImageNet-style normalization built into `strhub` itself --
    reused verbatim, never reimplemented, since it must match what the checkpoint was trained
    with exactly).
    """
    import torch
    from strhub.data.module import SceneTextDataModule
    from strhub.models.parseq.system import PARSeq

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hparams = dict(checkpoint["hyper_parameters"])
    model = PARSeq(**hparams)
    _remap_and_load(model, checkpoint)
    model.eval()

    use_cuda = device == "cuda" and torch.cuda.is_available()
    model = model.to("cuda" if use_cuda else "cpu")
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    logger.info(
        "loaded PARSeq-SoccerNet reader (device=%s, img_size=%s, charset_test=%r)",
        "cuda" if use_cuda else "cpu",
        model.hparams.img_size,
        model.hparams.charset_test,
    )
    return model, transform


def free_parseq_soccernet(model: Any) -> None:
    """Explicitly free the PARSeq-SoccerNet model's VRAM (CLAUDE.md §11)."""
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jersey_number_parseq(
    crop_bgr: np.ndarray, model: Any, transform: Any, parseq_cfg: dict
) -> ParseqRead:
    """Run PARSeq-SoccerNet over one BGR crop (expected to already have cleared the legibility
    gate -- `src/identity/legibility.py` -- this function does not re-check legibility itself).

    Uses PARSeq's own OFFICIAL decode API (`logits.softmax(-1)` -> `model.tokenizer.decode`,
    exactly `baudm/parseq`'s own README usage) -- see this module's own docstring for why this
    replaces the brief's originally-assumed manual logit-slicing, which does not match this
    checkpoint's real output shape. A decode that isn't purely `\\d{1,2}` is an honest non-match
    (`digits=None`), never a forced/truncated digit extraction.
    """
    import torch
    from PIL import Image

    if crop_bgr.size == 0:
        return ParseqRead(digits=None, confidence=0.0, is_confident=False, raw_text=None)

    device = next(model.parameters()).device
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = logits.softmax(-1)
        labels, confs = model.tokenizer.decode(probs)

    raw_text = labels[0]
    confidence = float(confs[0].mean().item()) if len(confs[0]) else 0.0

    max_digits = parseq_cfg.get("max_digits", 2)
    digits_only_pattern = re.compile(rf"^\d{{1,{max_digits}}}$")
    if not digits_only_pattern.match(raw_text):
        # A legible crop that PARSeq reads as non-numeric (sponsor text, a letter, punctuation) or
        # a 3+ digit string (implausible for a jersey number, same ceiling as
        # `identity.yaml: ocr.max_digits`) is an honest non-match, never salvaged by stripping
        # non-digit characters out -- that would fabricate a digit string that was never itself
        # confidently, wholly read (Golden Rule 5).
        return ParseqRead(digits=None, confidence=confidence, is_confident=False, raw_text=raw_text)

    is_confident = confidence >= parseq_cfg["min_confidence"]
    return ParseqRead(
        digits=raw_text, confidence=confidence, is_confident=is_confident, raw_text=raw_text
    )
