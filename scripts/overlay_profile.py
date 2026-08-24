#!/usr/bin/env python
"""Stage 0 / 0.5 debug overlay artifact (CLAUDE.md §10: "after each stage: ... an annotated
overlay video artifact").

Renders a downscaled MP4 for one input video into ``work/<slug>/debug/overlay_profile.mp4``
showing, per frame: frame index, timestamp, the current take id, and the clip's overall motion
score/profile classification. Take boundaries (from `src.shots.boundaries`) get a brief red flash
+ "CUT" banner so cuts are visually obvious when scrubbing the output.

Usage::

    .venv/bin/python scripts/overlay_profile.py "input/clip4 77.mp4"

Cosmetic constants below (flash duration/colour, text styling) are named module-level constants
rather than YAML knobs, following the same convention `src/common/viz.py` documents: they affect
only this debug visualization, not any pipeline decision, so they aren't config material.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import typer

from src.common.io import load_yaml, work_dir_for
from src.common.logging import get_logger
from src.common.types import RunProfile, Take
from src.common.video import decode_frames, probe, write_video
from src.common.viz import draw_hud
from src.pipeline.profiler import build_run_profile
from src.shots.boundaries import detect_takes

logger = get_logger(__name__)
app = typer.Typer(add_completion=False)

_OVERLAY_MAX_WIDTH = 960  # task spec: "downscaled 960px-wide render so it's fast"
_CUT_FLASH_FRAMES = 10  # how many frames the red flash/CUT banner stays up after a cut
_CUT_FLASH_COLOR_BGR = (0, 0, 255)
_CUT_FLASH_ALPHA = 0.35
_BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_FONT_SCALE = 1.4
_BANNER_THICKNESS = 3
_FOOTER_FONT_SCALE = 0.6
_FOOTER_COLOR = (255, 255, 255)
_FOOTER_SHADOW = (0, 0, 0)


def _annotate(
    frame_iter: Iterator[tuple[int, float, np.ndarray]],
    takes: list[Take],
    run_profile: RunProfile,
) -> Iterator[np.ndarray]:
    """Draw the HUD + take id + cut flash onto each decoded frame."""
    cut_frames = {t.frame_start for t in takes if t.frame_start > 0}
    take_id = 0
    for index, t, frame in frame_iter:
        while take_id < len(takes) - 1 and index >= takes[take_id].frame_end:
            take_id += 1

        out = draw_hud(frame, index, t, stage="profile")

        footer = (
            f"take={take_id} motion={run_profile.motion_score:.2f}px/f "
            f"profile={run_profile.profile.value} quality={run_profile.quality_flag}"
        )
        y = out.shape[0] - 15
        cv2.putText(
            out, footer, (10, y), _BANNER_FONT, _FOOTER_FONT_SCALE, _FOOTER_SHADOW, 3, cv2.LINE_AA
        )
        cv2.putText(
            out, footer, (10, y), _BANNER_FONT, _FOOTER_FONT_SCALE, _FOOTER_COLOR, 1, cv2.LINE_AA
        )

        frames_since_cut = min((index - c for c in cut_frames if c <= index), default=None)
        if frames_since_cut is not None and 0 <= frames_since_cut < _CUT_FLASH_FRAMES:
            overlay = out.copy()
            overlay[:] = _CUT_FLASH_COLOR_BGR
            out = cv2.addWeighted(overlay, _CUT_FLASH_ALPHA, out, 1 - _CUT_FLASH_ALPHA, 0)
            text = "CUT"
            (tw, th), _ = cv2.getTextSize(text, _BANNER_FONT, _BANNER_FONT_SCALE, _BANNER_THICKNESS)
            cx, cy = (out.shape[1] - tw) // 2, (out.shape[0] + th) // 2
            cv2.putText(
                out,
                text,
                (cx, cy),
                _BANNER_FONT,
                _BANNER_FONT_SCALE,
                (255, 255, 255),
                _BANNER_THICKNESS,
                cv2.LINE_AA,
            )

        yield out


@app.command()
def main(
    video: Path = typer.Argument(..., help="Path to a video in input/"),
    hardware_config_path: Path = typer.Option(Path("configs/hardware.yaml")),
    profile_config_path: Path = typer.Option(Path("configs/profile.yaml")),
    shots_config_path: Path = typer.Option(Path("configs/shots.yaml")),
    work_root: Path = typer.Option(Path("work")),
) -> Path:
    """Render the Stage 0/0.5 debug overlay MP4 for `video`."""
    hardware_config = load_yaml(hardware_config_path)
    profile_config = load_yaml(profile_config_path)
    shots_config = load_yaml(shots_config_path)

    run_profile = build_run_profile(
        video, hardware_config, profile_config, shots_config, work_root=work_root
    )
    takes = detect_takes(video, shots_config, work_root=work_root)

    meta = probe(video)
    fps = meta["fps"]
    out_width = min(_OVERLAY_MAX_WIDTH, meta["width"])

    frame_iter = decode_frames(video, fps=None, scale_width=out_width, use_nvdec=True)
    first = next(frame_iter, None)
    if first is None:
        raise RuntimeError(f"no frames decoded from {video}")
    actual_h, actual_w = first[2].shape[:2]

    work_dir = work_dir_for(video, root=work_root)
    debug_dir = work_dir / "debug"
    out_path = debug_dir / "overlay_profile.mp4"

    annotated = _annotate(itertools.chain([first], frame_iter), takes, run_profile)
    write_video(annotated, out_path, fps=fps, size=(actual_w, actual_h), encoder="h264_nvenc")
    logger.info(
        "overlay -> %s (%d take(s), profile=%s)", out_path, len(takes), run_profile.profile.value
    )
    return out_path


if __name__ == "__main__":
    app()
