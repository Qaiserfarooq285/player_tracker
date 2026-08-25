"""Stage 2 fail-fast smoke test (CLAUDE.md task spec: run BEFORE building anything else on top of
`src/detect`). Loads the detector, runs it on one real frame from `input/clip2 77.mp4` at t~=5s,
prints per-class counts + confidences, saves an annotated JPG, and prints peak VRAM. If detections
are empty or nonsensical, this raises rather than reporting success.

Usage: ``.venv/bin/python scripts/smoke_detect.py``
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.io import load_yaml  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.video import decode_frames  # noqa: E402
from src.common.viz import draw_detections  # noqa: E402
from src.detect.detector import detect_frame, free_detector, load_detector  # noqa: E402

logger = get_logger("smoke_detect")

VIDEO_PATH = Path("input/clip2 77.mp4")
TARGET_T = 5.0
OUT_JPG = Path("work/smoke_detect_clip2_t5.jpg")
# CLAUDE.md task spec: "expect ~10-22 players on a wide soccer frame" — soft sanity band, not a
# hard requirement (real footage varies frame to frame); zero detections is the hard failure.
EXPECTED_PLAYER_RANGE = (10, 22)


def main() -> None:
    hardware_config = load_yaml("configs/hardware.yaml")
    detect_config = load_yaml("configs/detect.yaml")
    scale_width = hardware_config["decode"]["scale_width"]

    frame = None
    actual_t = None
    for _idx, t, f in decode_frames(
        VIDEO_PATH,
        fps=None,
        start=TARGET_T,
        end=TARGET_T + 0.1,
        scale_width=scale_width,
        use_nvdec=True,
    ):
        frame, actual_t = f, t
        break
    if frame is None:
        raise RuntimeError(f"FAIL: could not decode any frame from {VIDEO_PATH} at t={TARGET_T}s")
    logger.info("decoded frame from %s at t=%.3fs, shape=%s", VIDEO_PATH, actual_t, frame.shape)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    handle = load_detector(detect_config, hardware_config)
    conf_threshold = detect_config["inference"]["conf_threshold"]
    nms_iou = detect_config["inference"]["nms_iou_threshold"]

    detections = detect_frame(
        handle,
        frame,
        frame_index=0,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou,
        t=actual_t,
    )

    counts = Counter(d.cls.value for d in detections)
    confs = [d.conf for d in detections]

    print(f"checkpoint_used: {handle.checkpoint_used}")
    if handle.fallback_reason:
        print(f"*** FALLBACK TRIGGERED *** reason: {handle.fallback_reason}")
    print(f"image_size (block-size-snapped): {handle.image_size}")
    print(f"n_detections: {len(detections)}")
    print(f"per-class counts: {dict(counts)}")
    if confs:
        print(f"confidence range: [{min(confs):.3f}, {max(confs):.3f}]")
    else:
        print("confidence range: N/A (no detections)")

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"peak VRAM (torch.cuda.max_memory_allocated): {peak_vram_mb:.1f} MB")

    annotated = draw_detections(frame, [(d.bbox, d.cls, d.conf) for d in detections])
    OUT_JPG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_JPG), annotated)
    print(f"annotated frame -> {OUT_JPG}")

    free_detector(handle)

    if len(detections) == 0:
        raise SystemExit(
            "FAIL FAST: zero detections on a real frame. STOP — do not build on top of this "
            "until the detector produces sane output."
        )
    n_players = counts.get("player", 0)
    lo, hi = EXPECTED_PLAYER_RANGE
    if not (lo <= n_players <= hi):
        logger.warning(
            "player count %d is outside the expected sane band [%d, %d] for a wide soccer "
            "frame — not treated as a hard failure, but worth a second look",
            n_players,
            lo,
            hi,
        )
    logger.info("smoke test PASSED: %d detection(s), %d player(s)", len(detections), n_players)


if __name__ == "__main__":
    main()
