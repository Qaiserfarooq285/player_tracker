"""Stage 2 — RF-DETR player/ball/referee detection (CLAUDE.md §5 Stage 2, ADR-8/ADR-9/ADR-10).

**Checkpoint loading is NOT `rfdetr.RFDETR.from_checkpoint(...)`, and this is deliberate — read
before touching this file.** `from_checkpoint()` infers the model class from the checkpoint's
``pretrain_weights`` filename (``"rf-detr-large.pth"``). The installed `rfdetr` package has since
repointed its *current* ``RFDETRLarge`` symbol at a different 2026 architecture (patch16,
hidden_dim=256) that also default-downloads from a file named ``rf-detr-large.pth``, so
`from_checkpoint()`'s filename match resolves to the WRONG class for a checkpoint actually trained
under the older ("deprecated") Large config — and fails with a patch-size mismatch
(``patch_size=14`` in the checkpoint vs. ``16`` in the class `from_checkpoint()` picked). Measured
2026-08-24: this reproduces identically whether `from_checkpoint()` is called on the generic
`RFDETR` class or explicitly on `RFDETRLargeDeprecated` — the classmethod re-derives the model
class from the checkpoint's filename/args regardless of which class it was invoked on. The fix is
to skip class inference entirely: construct `RFDETRLargeDeprecated` directly, passing the local
checkpoint path as `pretrain_weights` so the *normal* weight-loading path
(`rfdetr.models.weights.load_pretrain_weights`) runs against the correct, already-known
architecture (confirmed against the checkpoint's own training `args`: `patch_size=14`,
`encoder=dinov2_windowed_base`, `hidden_dim=384`, `dec_layers=3` — all `RFDETRLargeDeprecatedConfig`
defaults, no per-field overrides needed beyond `num_classes`).

**Class count — a second, separate measured discrepancy.** The checkpoint's training `args`
declare `num_classes=4` and 4 `class_names` (ball/player/referee/goalkeeper — see the model card),
but its *saved* `class_embed.bias` tensor has shape `(4,)`, i.e. **3 foreground classes + 1
background slot**, not 4+1. Constructing with `num_classes=4` (matching the declared args) makes
`rfdetr` itself warn "Checkpoint has 3 classes but model is configured for 4" and then *silently
keeps* `model_config.num_classes=4` while the real head only has 3 live output slots — a
footgun that would surface as "goalkeeper never appears and nobody knows why". We construct with
`num_classes=3` instead, exactly matching `rfdetr`'s own suggested fix ("Pass num_classes=3 to
suppress this warning"), and confirmed empirically (a real frame from `clip2` at multiple
thresholds/resolutions) that `class_id=3` ("goalkeeper") is never produced — see
`configs/detect.yaml: model.measured_num_classes` and the Stage 2/3 run report. Goalkeepers still
get detected, just as `DetectionClass.PLAYER` (no dedicated slot) — a benign degradation, not a
correctness bug we can fix without retraining (Golden Rule 2: no training in Phase 1).

**Precision.** Rather than `RFDETR.inference(dtype=torch.float16)` (which *locks* the model to a
single fixed inference resolution chosen at optimize time — measured: calling `predict(shape=...)`
at any other resolution afterwards raises `ValueError: Resolution mismatch`), inference runs
inside `torch.autocast(device_type="cuda", dtype=torch.float16)` when `hardware.yaml`'s
`precision: fp16` and CUDA is available. This keeps `image_size` a genuinely per-call, per-config
knob (CLAUDE.md §11: fp16 "everywhere", but §3.2(5) also wants image_size left tunable/measurable)
without giving up that flexibility for a modest extra speedup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

from src.common.logging import DropCounter, get_logger
from src.common.types import BBox, Detection, DetectionClass

logger = get_logger(__name__)

# COCO class-name -> DetectionClass mapping for the ADR-2(b)/ADR-8 fallback path
# (configs/detect.yaml model.fallback.classes). RF-DETR's own `predict()` maps COCO-pretrained
# outputs to these exact strings (rfdetr.assets.coco_classes.COCO_CLASS_NAMES), so no numeric
# COCO category-id juggling is needed here.
_COCO_FALLBACK_CLASS_MAP: dict[str, DetectionClass] = {
    "person": DetectionClass.PLAYER,
    "sports ball": DetectionClass.BALL,
}

# The soccernet checkpoint's class_name strings already equal DetectionClass values exactly
# (CLAUDE.md model card: "These match DetectionClass in src/common/types.py exactly — no
# remapping needed") — this identity map exists only so both code paths share one lookup shape.
_SOCCERNET_CLASS_MAP: dict[str, DetectionClass] = {c.value: c for c in DetectionClass}


@dataclass
class DetectorHandle:
    """A loaded Stage 2 detector plus everything `detect_frame`/`detect_batch` need to run it."""

    model: Any  # an `rfdetr.RFDETR` subclass instance — untyped to avoid a hard rfdetr import
    class_map: dict[str, DetectionClass]
    checkpoint_used: str  # "soccernet" | "coco_fallback" — surfaced loudly in the run report
    device: str
    use_fp16_autocast: bool
    block_size: int  # predict()'s `shape` must be a multiple of this (patch_size * num_windows)
    image_size: int  # the (already block_size-snapped) square inference resolution actually used
    fallback_reason: str | None = field(default=None)


def _snap_to_block(image_size: int, block_size: int) -> int:
    """Round `image_size` up to the nearest multiple of `block_size`.

    RF-DETR's windowed-attention backbones require the inference `shape` passed to `predict()` to
    be divisible by `patch_size * num_windows` (raises `ValueError` otherwise) — `image_size` in
    `configs/hardware.yaml` (e.g. 1280) is a target, not guaranteed to already satisfy that, so
    every caller snaps through this one function rather than re-deriving the rule ad hoc.
    """
    if image_size <= 0 or block_size <= 0:
        raise ValueError(f"image_size/block_size must be positive, got {image_size}/{block_size}")
    return ((image_size + block_size - 1) // block_size) * block_size


def load_detector(detect_config: dict, hardware_config: dict) -> DetectorHandle:
    """Load the Stage 2 detector: the SoccerNet checkpoint (ADR-8), falling back to RF-DETR-COCO
    (ADR-2(b)) if it cannot be loaded at all. The fallback is logged loudly (`logger.error`), never
    taken silently (CLAUDE.md task spec / Golden Rule 6).
    """
    model_cfg = detect_config["model"]
    device = hardware_config["device"] if torch.cuda.is_available() else "cpu"
    if hardware_config["device"] == "cuda" and device == "cpu":
        logger.warning("hardware.yaml requests device=cuda but CUDA is unavailable; using cpu")
    use_fp16 = hardware_config["precision"] == "fp16" and device == "cuda"

    detect_stage_cfg = hardware_config["stages"]["detect"]
    block_size = model_cfg["patch_size"] * model_cfg["num_windows"]
    image_size = _snap_to_block(detect_stage_cfg["image_size"], block_size)
    if image_size != detect_stage_cfg["image_size"]:
        logger.info(
            "detect image_size %d is not a multiple of patch_size*num_windows=%d; snapped to %d",
            detect_stage_cfg["image_size"],
            block_size,
            image_size,
        )

    checkpoint_path = Path(model_cfg["local_checkpoint"])
    try:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"soccernet checkpoint not found at {checkpoint_path}")
        from rfdetr import RFDETRLargeDeprecated

        with warnings.catch_warnings():
            # RFDETRLargeDeprecated itself carries a library-level FutureWarning ("will be
            # removed in v2.0.0") unrelated to anything wrong with *this* checkpoint or load —
            # suppressed here so it doesn't read as a problem with our code; the real, actionable
            # ADR-9 licensing caveat is already surfaced via CLAUDE.md/the model card/this module.
            warnings.filterwarnings("ignore", category=FutureWarning)
            model = RFDETRLargeDeprecated(
                pretrain_weights=str(checkpoint_path),
                num_classes=model_cfg["measured_num_classes"],
                device=device,
                trust_checkpoint=True,  # local file we downloaded + SHA-256-verified ourselves
            )
        logger.info(
            "loaded Stage 2 detector: soccernet checkpoint %s (num_classes=%d, device=%s)",
            checkpoint_path,
            model_cfg["measured_num_classes"],
            device,
        )
        return DetectorHandle(
            model=model,
            class_map=_SOCCERNET_CLASS_MAP,
            checkpoint_used="soccernet",
            device=device,
            use_fp16_autocast=use_fp16,
            block_size=block_size,
            image_size=image_size,
        )
    except Exception as exc:  # noqa: BLE001 — any load failure triggers the documented fallback
        fallback_cfg = model_cfg["fallback"]
        logger.error(
            "SoccerNet checkpoint failed to load (%s: %s) — FALLING BACK to RF-DETR-COCO (%s). "
            "This is the ADR-2(b)/ADR-8 fallback path: classes are limited to person->player and "
            "sports_ball->ball (no referee/goalkeeper). This must be reported, not silently "
            "treated as a normal run.",
            type(exc).__name__,
            exc,
            fallback_cfg["variant"],
        )
        from rfdetr import RFDETRNano

        fallback_block_size = model_cfg["patch_size"] * model_cfg["num_windows"]
        fallback_image_size = _snap_to_block(detect_stage_cfg["image_size"], 16 * 2)
        model = RFDETRNano(device=device)
        return DetectorHandle(
            model=model,
            class_map=_COCO_FALLBACK_CLASS_MAP,
            checkpoint_used="coco_fallback",
            device=device,
            use_fp16_autocast=use_fp16,
            block_size=fallback_block_size,
            image_size=fallback_image_size,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def free_detector(handle: DetectorHandle) -> None:
    """Explicitly free the detector's VRAM (CLAUDE.md §11: unload each model before the next)."""
    del handle.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sv_detections_to_ours(
    dets: sv.Detections, frame_index: int, t: float, class_map: dict[str, DetectionClass]
) -> list[Detection]:
    """Convert one frame's `supervision.Detections` (from `RFDETR.predict`) to our `Detection`s.

    Classes absent from `class_map` (e.g. a stray COCO category on the fallback path) are dropped
    silently *here* — callers that care about drop provenance route through `detect_frame`, which
    logs them via a `DropCounter`.
    """
    out: list[Detection] = []
    names = dets.data.get("class_name", [])
    for i in range(len(dets)):
        cls_name = str(names[i]) if len(names) > i else ""
        cls = class_map.get(cls_name)
        if cls is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
        out.append(
            Detection(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                cls=cls,
                conf=float(dets.confidence[i]),
                frame_index=frame_index,
                t=t,
            )
        )
    return out


def _predict_one(
    handle: DetectorHandle, frame_rgb: np.ndarray, conf_threshold: float
) -> sv.Detections:
    """Run one RGB frame through `handle.model.predict`, under fp16 autocast if configured."""
    shape = (handle.image_size, handle.image_size)
    if handle.use_fp16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return handle.model.predict(frame_rgb, threshold=conf_threshold, shape=shape)
    return handle.model.predict(frame_rgb, threshold=conf_threshold, shape=shape)


def detect_frame(
    handle: DetectorHandle,
    frame_bgr: np.ndarray,
    frame_index: int,
    conf_threshold: float,
    nms_iou_threshold: float,
    t: float = 0.0,
    drops: DropCounter | None = None,
) -> list[Detection]:
    """Detect players/referees/goalkeepers/ball in one BGR frame (as decoded by `common.video`).

    Returns every detection whose class survives `handle.class_map` (already NMS'd); classes with
    no mapping (e.g. an unmapped COCO fallback category) are counted in `drops` as
    `dropped_unmapped_class` when a `DropCounter` is supplied.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets = _predict_one(handle, frame_rgb, conf_threshold)
    if len(dets) > 1:
        dets = dets.with_nms(threshold=nms_iou_threshold, class_agnostic=False)

    if drops is not None:
        names = dets.data.get("class_name", [])
        n_unmapped = sum(1 for n in names if str(n) not in handle.class_map)
        if n_unmapped:
            drops.drop("dropped_unmapped_class", n_unmapped)

    return _sv_detections_to_ours(dets, frame_index, t, handle.class_map)


def detect_batch(
    handle: DetectorHandle,
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    conf_threshold: float,
    nms_iou_threshold: float,
    ts: list[float] | None = None,
    drops: DropCounter | None = None,
) -> list[list[Detection]]:
    """Batched `detect_frame` — one `predict()` call over the whole list of frames.

    `rfdetr`'s `predict()` natively accepts a list of images and returns a list of `Detections`
    (one per image) when given one, which is what actually saves work per batch (vs. a Python
    loop over `detect_frame`) — a single forward pass over the stacked batch rather than one per
    frame.
    """
    if not frames_bgr:
        return []
    if ts is None:
        ts = [0.0] * len(frames_bgr)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    shape = (handle.image_size, handle.image_size)
    if handle.use_fp16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            results = handle.model.predict(frames_rgb, threshold=conf_threshold, shape=shape)
    else:
        results = handle.model.predict(frames_rgb, threshold=conf_threshold, shape=shape)
    if not isinstance(results, list):
        results = [results]

    out: list[list[Detection]] = []
    for frame_index, t, dets in zip(frame_indices, ts, results, strict=True):
        if len(dets) > 1:
            dets = dets.with_nms(threshold=nms_iou_threshold, class_agnostic=False)
        if drops is not None:
            names = dets.data.get("class_name", [])
            n_unmapped = sum(1 for n in names if str(n) not in handle.class_map)
            if n_unmapped:
                drops.drop("dropped_unmapped_class", n_unmapped)
        out.append(_sv_detections_to_ours(dets, frame_index, t, handle.class_map))
    return out
