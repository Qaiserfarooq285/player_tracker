"""Stage 2 CLI — RF-DETR detection + arrow/watermark masking + SAHI ball, cached per video
(CLAUDE.md §5 Stage 2). Wired to ``python -m src.detect.run --video "<path>"`` (see
`.claude/commands/process-match.md`).

Two independent decode passes run at their own `fps_sample` (CLAUDE.md §11 "per-stage fps
sampling"): the main player/referee/goalkeeper pass at `stages.detect.fps_sample`, and the SAHI
ball pass at `stages.ball.fps_sample` (higher — the ball moves fast and is small, CLAUDE.md §5).
Both share one loaded detector instance (freed once, at the end) rather than loading the 1.46GB
checkpoint twice. Ball detections from the *main* pass are deliberately dropped (not persisted) —
the ball is routed exclusively through the dedicated SAHI pass — and counted via `DropCounter` so
the drop is visible, not silent (CLAUDE.md §10).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import torch
import typer
from rich.console import Console

from src.common.io import (
    StageCache,
    load_models_parquet,
    load_yaml,
    save_models_parquet,
    work_dir_for,
)
from src.common.logging import DropCounter, get_logger
from src.common.types import BallDetection, Detection, DetectionClass
from src.common.video import decode_frames, probe
from src.detect.ball import detect_ball_sahi, interpolate_ball_gaps
from src.detect.detector import detect_batch, free_detector, load_detector
from src.detect.overlay_mask import ArrowHint, filter_arrow_overlap, filter_watermark, find_arrow

logger = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False)

_T = TypeVar("_T")


def _batched(iterable: Iterable[_T], n: int) -> Iterator[list[_T]]:
    """Yield `iterable` in chunks of (at most) `n` items — no external deps required."""
    batch: list[_T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def run_detect_stage(
    video_path: str | Path,
    hardware_config: dict,
    detect_config: dict,
    work_root: str | Path = "work",
    use_nvdec: bool = True,
) -> dict[str, Any]:
    """Run Stage 2 for one video: detections + arrow hints + SAHI ball, all cached under
    ``work/<slug>/detect/``. Returns a summary dict (counts/timings/VRAM/drops) for reporting.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    detect_dir = work_dir / "detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    detections_path = detect_dir / "detections.parquet"
    arrow_path = detect_dir / "arrow_hints.parquet"
    ball_path = detect_dir / "ball_detections.parquet"

    probe(video_path)  # fail fast if the video can't be read at all
    decode_cfg = hardware_config["decode"]
    detect_stage_cfg = hardware_config["stages"]["detect"]
    ball_stage_cfg = hardware_config["stages"]["ball"]

    video_identity = {
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
    }
    detect_cache_config = {
        **video_identity,
        "decode": decode_cfg,
        "detect_stage": detect_stage_cfg,
        "detect_config": detect_config,
    }
    detections_cache = StageCache(detections_path, detect_cache_config, stage="detect.detections")
    arrow_cache = StageCache(arrow_path, detect_cache_config, stage="detect.arrow_hints")
    ball_cache_config = {
        **video_identity,
        "decode": decode_cfg,
        "ball_stage": ball_stage_cfg,
        "detect_config": detect_config,
    }
    ball_cache = StageCache(ball_path, ball_cache_config, stage="detect.ball")

    both_main_hit = detections_cache.hit() and arrow_cache.hit()
    ball_hit = ball_cache.hit()

    if both_main_hit and ball_hit:
        detections = load_models_parquet(detections_path, Detection)
        ball_detections = load_models_parquet(ball_path, BallDetection)
        logger.info(
            "Stage 2 fully cached for %s — skipping detector load entirely", video_path.name
        )
        return {
            "video": video_path.name,
            "n_detections": len(detections),
            "n_ball": len(ball_detections),
            "wall_detect_s": 0.0,
            "wall_ball_s": 0.0,
            "peak_vram_mb": None,
            "checkpoint_used": "cached",
            "fallback_reason": None,
            "dropped": {},
        }

    handle = load_detector(detect_config, hardware_config)
    drops = DropCounter("detect")
    conf_threshold = detect_config["inference"]["conf_threshold"]
    nms_iou = detect_config["inference"]["nms_iou_threshold"]
    masking_cfg = detect_config["masking"]
    masking_enabled = masking_cfg["ignore_burned_in_graphics"]
    batch_size = detect_stage_cfg["batch_size"]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- main pass: player / referee / goalkeeper -----------------------------------------
    t0 = time.time()
    all_detections: list[Detection] = []
    all_arrow_hints: list[ArrowHint] = []
    if not both_main_hit:
        frame_iter = decode_frames(
            video_path,
            fps=detect_stage_cfg["fps_sample"],
            scale_width=decode_cfg["scale_width"],
            use_nvdec=use_nvdec,
        )
        for batch in _batched(frame_iter, batch_size):
            indices = [b[0] for b in batch]
            ts = [b[1] for b in batch]
            frames = [b[2] for b in batch]

            batch_dets = detect_batch(
                handle, frames, indices, conf_threshold, nms_iou, ts=ts, drops=drops
            )

            for frame_index, t, frame, dets in zip(indices, ts, frames, batch_dets, strict=True):
                arrow = None
                if masking_enabled:
                    arrow = find_arrow(frame, masking_cfg, frame_index=frame_index, t=t)
                    if arrow is not None:
                        all_arrow_hints.append(arrow)

                n_ball_main = sum(1 for d in dets if d.cls == DetectionClass.BALL)
                if n_ball_main:
                    drops.drop("dropped_ball_from_main_pass", n_ball_main)
                dets = [d for d in dets if d.cls != DetectionClass.BALL]

                dets = filter_arrow_overlap(dets, arrow, masking_cfg, drops)
                dets = filter_watermark(dets, frame.shape, masking_cfg, drops)
                all_detections.extend(dets)

        save_models_parquet(all_detections, detections_path)
        detections_cache.write_meta()
        save_models_parquet(all_arrow_hints, arrow_path)
        arrow_cache.write_meta()
    else:
        all_detections = load_models_parquet(detections_path, Detection)
    wall_detect_s = time.time() - t0

    # --- SAHI ball pass ---------------------------------------------------------------------
    t0 = time.time()
    all_ball_detections: list[BallDetection] = []
    if not ball_hit:
        sampled_frames: list[tuple[int, float]] = []
        observed: list[BallDetection] = []
        for frame_index, t, frame in decode_frames(
            video_path,
            fps=ball_stage_cfg["fps_sample"],
            scale_width=decode_cfg["scale_width"],
            use_nvdec=use_nvdec,
        ):
            sampled_frames.append((frame_index, t))
            bd = detect_ball_sahi(handle, frame, frame_index, detect_config, hardware_config, t=t)
            if bd is not None:
                observed.append(bd)

        if not observed:
            drops.drop("ball_never_detected", 1)
        all_ball_detections = interpolate_ball_gaps(
            observed, sampled_frames, detect_config["ball_interpolation"]
        )
        n_interp = sum(1 for d in all_ball_detections if d.interpolated)
        if n_interp:
            drops.drop("ball_interpolated_frames", n_interp)
        n_unfilled = len(sampled_frames) - len(all_ball_detections)
        if n_unfilled:
            drops.drop("ball_gap_unfilled", n_unfilled)

        save_models_parquet(all_ball_detections, ball_path)
        ball_cache.write_meta()
    else:
        all_ball_detections = load_models_parquet(ball_path, BallDetection)
    wall_ball_s = time.time() - t0

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
    checkpoint_used = handle.checkpoint_used
    fallback_reason = handle.fallback_reason
    free_detector(handle)

    dropped = drops.report()
    return {
        "video": video_path.name,
        "n_detections": len(all_detections),
        "n_ball": len(all_ball_detections),
        "n_arrow_hints": len(all_arrow_hints),
        "wall_detect_s": round(wall_detect_s, 2),
        "wall_ball_s": round(wall_ball_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
        "checkpoint_used": checkpoint_used,
        "fallback_reason": fallback_reason,
        "dropped": dropped,
    }


@app.command()
def main(
    video: Path = typer.Argument(..., help="Path to a video in input/"),
    hardware_config_path: Path = typer.Option(Path("configs/hardware.yaml")),
    detect_config_path: Path = typer.Option(Path("configs/detect.yaml")),
    work_root: Path = typer.Option(Path("work")),
) -> None:
    """Run Stage 2 (detection + arrow/watermark masking + SAHI ball) for one video."""
    hardware_config = load_yaml(hardware_config_path)
    detect_config = load_yaml(detect_config_path)

    summary = run_detect_stage(video, hardware_config, detect_config, work_root=work_root)
    console.print(summary)


if __name__ == "__main__":
    app()
