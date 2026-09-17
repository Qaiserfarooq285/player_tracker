#!/usr/bin/env python3
"""Fetch the three model checkpoints the pipeline needs into `models/` (gitignored, ~1.9 GB).

Run once per machine (`make models`); `docker/runpod_bootstrap.sh` calls it on every boot and it
is a no-op when every file is already present with the right checksum. Checksums are the ones
recorded in each folder's MODEL_CARD.md at download time -- a mismatch is an error, never a
silently-accepted different file (CLAUDE.md Golden Rule 5 applied to weights).

Sources and licences (CLAUDE.md §7 / ADR-8 / ADR-9):
  * RF-DETR SoccerNet detector -- Hugging Face `julianzu9612/RFDETR-Soccernet`, Apache-2.0 as
    declared; trained on SoccerNet, so dev/eval only until ADR-9 is resolved.
  * Jersey legibility gate + PARSeq -- Google Drive, mkoshkina/jersey-number-pipeline
    checkpoints, CC BY-NC 3.0 (owner-authorized non-commercial exception, 2026-08-31).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

RFDETR_REPO = "julianzu9612/RFDETR-Soccernet"
RFDETR_DIR = MODELS / "rfdetr-soccernet"
# local filename -> (path inside the HF repo, sha256 or None). The repo moved the checkpoint under
# weights/ (2026-09-17, observed as a 404 on every pod boot); the local layout the configs point at
# (`models/rfdetr-soccernet/checkpoint_best_regular.pth`) stays the same.
RFDETR_FILES = {
    "checkpoint_best_regular.pth": (
        "weights/checkpoint_best_regular.pth",
        "b9ade4bcc2316259582674ebeebbd27bb2e956480428c54114ace567237eb5f9",
    ),
    "config.json": ("config.json", None),
    "model_metadata.json": ("model_metadata.json", None),
    "README.md": ("README.md", None),
}

JERSEY_DIR = MODELS / "jersey-parseq-soccernet"
JERSEY_FILES = {
    "legibility_resnet34_soccer_20240215.pth": (
        "18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
        "b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41",
    ),
    "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt": (
        "1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
        "14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ok(path: Path, expected: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return expected is None or _sha256(path) == expected


def _check(path: Path, expected: str | None, label: str) -> None:
    if not _ok(path, expected):
        raise SystemExit(f"{label}: checksum mismatch or missing file at {path}")
    print(f"  ok  {path.relative_to(ROOT)}")


def fetch_rfdetr() -> None:
    RFDETR_DIR.mkdir(parents=True, exist_ok=True)
    for name, (remote, sha) in RFDETR_FILES.items():
        dest = RFDETR_DIR / name
        if _ok(dest, sha):
            print(f"  have {dest.relative_to(ROOT)}")
            continue
        # Imported only when a download is actually needed: the `detect` extra brings it in, but
        # a machine that already has the weights must not need it just to pass this check.
        from huggingface_hub import hf_hub_download

        print(f"  downloading {RFDETR_REPO}/{remote} ...")
        got = Path(hf_hub_download(RFDETR_REPO, remote, local_dir=str(RFDETR_DIR)))
        if got.resolve() != dest.resolve():
            # local_dir mirrors the repo layout (weights/...), but the configs expect a flat folder.
            dest.parent.mkdir(parents=True, exist_ok=True)
            got.replace(dest)
        _check(dest, sha, name)


def fetch_jersey() -> None:
    JERSEY_DIR.mkdir(parents=True, exist_ok=True)
    for name, (gdrive_id, sha) in JERSEY_FILES.items():
        dest = JERSEY_DIR / name
        if _ok(dest, sha):
            print(f"  have {dest.relative_to(ROOT)}")
            continue
        import gdown  # same reason as above (`jersey_parseq` extra)

        print(f"  downloading Google Drive id={gdrive_id} -> {name} ...")
        gdown.download(id=gdrive_id, output=str(dest), quiet=False)
        _check(dest, sha, name)


def main() -> int:
    print(f"models/ -> {MODELS}")
    fetch_rfdetr()
    fetch_jersey()
    print("all model weights present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
