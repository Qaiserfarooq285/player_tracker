# Third-party attribution

Obligations we must satisfy when distributing this software or its outputs.
Keep in sync with the licence table in `CLAUDE.md` §7.

## Datasets

### football-players-detection (Roboflow Universe) — **CC BY 4.0, attribution required**
- Source: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- Version used: **v20 ("rf-detr-m")**, exported 2025-08-01, downloaded 2026-08-24
- Local path: `data/eval/roboflow-football-players-v20/` (gitignored)
- Contents: 372 images / 8,905 annotations — train 298, valid 49, test 25
- Classes: `ball`, `goalkeeper`, `player`, `referee`
- Licence: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/)
- Used for: detection **mAP evaluation** (`CLAUDE.md` §9). Not redistributed.

> **Required attribution text** (ship this in any product using it):
> "football-players-detection dataset by Roboflow user `roboflow-jvuqo`, licensed under
> CC BY 4.0. Source: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc"

Sibling datasets from the same publisher, same CC BY 4.0 terms, not yet downloaded:
`football-field-detection-f07vi` (keypoints), `football-ball-detection-rejhg` (ball).

## Model weights

### julianzu9612/RFDETR-Soccernet — Apache-2.0 *(declared)*, **⚠️ dev/eval use only**
- Source: https://huggingface.co/julianzu9612/RFDETR-Soccernet
- RF-DETR-Large, 128M params, DINOv2 backbone, 1280² input
- **Trained on SoccerNet-Tracking-2023, which is research/education licensed.**
  See `CLAUDE.md` **ADR-9** — this is a blocking item before any commercial distribution.
  Do not ship without resolving via one of: SoccerNet commercial terms, a fine-tune on the
  CC BY 4.0 Roboflow data, or falling back to RF-DETR-COCO.

## Software

Permissive licences per `CLAUDE.md` §7 (Apache-2.0 / MIT / BSD-3): RF-DETR, `supervision`,
SAHI, PySceneDetect, PyAV, TrackEval, `transformers`/`timm` (SigLIP). No AGPL dependencies —
Ultralytics YOLO and BoxMOT are deliberately excluded (Golden Rule 6).
