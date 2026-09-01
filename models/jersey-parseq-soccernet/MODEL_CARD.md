# Model card — jersey-number legibility gate + PARSeq-SoccerNet reader

**Status:** ✅ owner-authorized 2026-08-31 (`CLAUDE.md` §7 "mkoshkina/jersey-number-pipeline
CHECKPOINTS ONLY" row) · ⚠️ **CC BY-NC 3.0, non-commercial only — see the license section below
before shipping anything that uses these two files.**

## Identity

| | File 1 | File 2 |
|---|---|---|
| Purpose | Legibility classifier (skip OCR on unreadable crops) | Jersey-number reader (PARSeq fine-tuned on SoccerNet) |
| File | `legibility_resnet34_soccer_20240215.pth` | `parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt` |
| Size | 85,289,629 bytes (~81.3 MiB) | 381,608,677 bytes (~363.9 MiB) |
| SHA-256 | `b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41` | `14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879` |
| Source | Google Drive, `id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw` | Google Drive, `id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm` |
| Provenance | `mkoshkina/jersey-number-pipeline` (fine-tuned on SoccerNet) | `mkoshkina/jersey-number-pipeline` (fine-tuned on SoccerNet, from the `baudm/parseq` base architecture) |
| Downloaded | 2026-08-31, via `gdown` (MIT) | 2026-08-31, via `gdown` (MIT) |
| Declared licence | **CC BY-NC 3.0** (non-commercial) | **CC BY-NC 3.0** (non-commercial) |

## License scoping — read this before touching either checkpoint

Per `CLAUDE.md` §7's own explicit scoping, the NC encumbrance is deliberately minimized to these
**two weight files only**:

- The **legibility classifier's model definition** (`src/identity/legibility.py`) is an
  independently-authored, ordinary `torchvision.models.resnet34` + `nn.Linear(512, 1)` — a
  standard architecture, verified directly against this checkpoint's own `state_dict` (218 keys,
  all prefixed `model_ft.`, final `fc.weight` shape `(1, 512)`). It contains none of
  `mkoshkina/jersey-number-pipeline`'s own source code.
- The **PARSeq architecture code** (`src/identity/jersey_parseq.py`) is the REAL upstream
  `baudm/parseq` package (`strhub`, Apache-2.0), installed as a git dependency
  (`pyproject.toml`'s `jersey_parseq` extra) — never any code from mkoshkina's repo.
- **Only the two weight files above are CC BY-NC 3.0.** This pipeline's output must not be sold,
  commercially distributed, or shipped in a closed/paid product while these two checkpoints are
  in use, without either removing them (the pipeline degrades gracefully to the pre-existing
  EasyOCR/Gemini-only chain when they're absent/disabled) or resolving their licensing first —
  same "flag, don't ship silently" discipline as ADR-9's SoccerNet-derived detector weights.

## Legibility classifier — architecture

- `torchvision.models.resnet34` backbone (ImageNet-pretrained before this fine-tune), final `fc`
  replaced with `nn.Linear(512, 1)`, sigmoid applied at inference for `P(legible)`.
- Input: 256×256 RGB, ImageNet mean/std normalization (`src/identity/legibility.py`'s own
  `_IMAGENET_MEAN`/`_IMAGENET_STD`).
- Verified 2026-08-31: loads with `strict=True`, zero shape mismatches, forward pass on a
  synthetic tensor produces a scalar sigmoid output in `[0, 1]`.

## PARSeq-SoccerNet reader — architecture + a real, documented format mismatch

- Base architecture: PARSeq-base (`embed_dim=384`, `enc_depth=12`, `dec_depth=1`,
  `img_size=[32,128]`, `patch_size=[4,8]`) — the standard `baudm/parseq` "parseq" experiment
  config, fine-tuned by mkoshkina on SoccerNet jersey crops.
- ⚠️ **`charset_test` is the standard 36-char alnum set** (`0123456789abcdefghijklmnopqrstuvwxyz`),
  **not** a jersey-specific digit-only vocabulary, and `max_label_length=25` (the stock default) —
  this checkpoint was NOT trained with a custom small output head. A read is accepted as a jersey
  number only when the decoded string matches `^\d{1,max_digits}$`; anything else (letters,
  punctuation, 3+ digit strings) is an honest non-match, never truncated/salvaged into a fake digit.
- ⚠️ **State-dict key mismatch, verified and resolved (see `src/identity/jersey_parseq.py`'s own
  module docstring for the full evidence trail):** this checkpoint (file-dated Sep 2023) predates
  a Feb-2024 upstream refactor (`baudm/parseq` commit `4cdf0bf`) that nested the model's
  encoder/decoder/head/pos_queries/text_embed under a new `self.model` submodule. Loading via the
  documented `strhub.models.utils.load_from_checkpoint` fails outright on this checkpoint. The
  fix: prefix every checkpoint key with `"model."` before loading into the CURRENT upstream
  architecture — verified exhaustively (175/175 keys match exactly, zero missing, zero extra,
  zero shape mismatches) rather than assumed. Pinning to the exact pre-refactor `strhub` commit
  was tried and rejected: that old code needs `pytorch_lightning.utilities.types.EPOCH_OUTPUT`,
  removed in the modern `pytorch-lightning` this project already depends on.
- Preprocessing: PARSeq's own `strhub.data.module.SceneTextDataModule.get_transform(img_size)`,
  reused verbatim (never reimplemented) so it matches training exactly.
- Decoding: PARSeq's own official public API — `logits.softmax(-1)` → `model.tokenizer.decode(...)`
  — not the brief's originally-assumed manual `logits[:, :3, :11]` slice, which does not match
  this checkpoint's real output shape (`(1, 26, 95)` on a 256-class-ish vocabulary, not 11).
- Reported performance (checkpoint filename, mkoshkina's own numbers, on their SoccerNet
  jersey-number eval split): val_accuracy 95.6044%, val_NED 96.3255%. **Not our own measured
  number** — same ADR-10 "expect a domain gap, measure on our own footage" discipline as the
  RF-DETR-SoccerNet checkpoint.

## Chain position (CLAUDE.md §7, `src/identity/verify.py` / `src/annotations/associate.py`)

legibility gate → if legible: PARSeq-SoccerNet → if PARSeq unconfident or the crop was flagged
illegible: fall back to the pre-existing EasyOCR path → escalate to Gemini VLM only if neither
succeeds. Both new stages are **additive and optional**: when either checkpoint fails to load
(missing file, disabled in config), the chain degrades exactly to the prior OCR-first/
VLM-escalation behaviour — see `configs/identity.yaml`'s own `legibility`/`parseq_soccernet`
blocks for the enable/threshold knobs.

## Operational notes (A2000 12 GB)

Two more models sharing the card alongside RF-DETR/EasyOCR/Gemini (API-only). Both are loaded
once per video (same convention as `jersey_ocr.load_easyocr_reader`) and freed immediately after.
Measured peak VRAM (2026-08-31, on the real verification run): report in the task's own follow-up
notes — both models are small relative to RF-DETR-Large (legibility: a single resnet34, ~21M
params; PARSeq-base: ~90M params) and are loaded/freed sequentially, never held in VRAM alongside
the heavier detection/tracking stages (CLAUDE.md §11: never hold multiple heavy models in VRAM at
once).
