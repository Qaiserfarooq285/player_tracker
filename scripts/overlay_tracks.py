"""Stage 2+3 debug artifact (CLAUDE.md §10: "after each stage: an annotated overlay video").

Renders `work/<slug>/debug/overlay_tracks.mp4`: track boxes coloured by team + track ID, the ball
marker (distinct colour when interpolated), a "CUT -> take N" banner at every take boundary after
the first, and the burned-in red-arrow's outline/tip when present. Runs Stage 2+3 first if their
caches are missing (both are no-ops on a cache hit).

Usage: ``.venv/bin/python scripts/overlay_tracks.py "input/clip2 77.mp4"``
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import typer  # noqa: E402

from src.common.io import load_models_parquet, load_yaml, work_dir_for  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.types import BallDetection, Track  # noqa: E402
from src.common.video import decode_frames, write_video  # noqa: E402
from src.common.viz import (  # noqa: E402
    draw_arrow_outline,
    draw_ball,
    draw_banner,
    draw_hud,
    draw_tracks,
)
from src.detect.overlay_mask import ArrowHint  # noqa: E402
from src.shots.boundaries import detect_takes  # noqa: E402
from src.track.run import run_track_stage  # noqa: E402
from src.track.tracker import assign_take_id  # noqa: E402

logger = get_logger("overlay_tracks")
app = typer.Typer(add_completion=False)

# Cosmetic/behavioural constants for this debug artifact only (CLAUDE.md §10 wants a visual
# sanity check, not a pipeline-tunable) — not YAML config material, per viz.py's own convention.
OUTPUT_WIDTH_PX = 960
BANNER_SECONDS = 1.0  # how long the "CUT -> take N" banner stays on screen after a take starts


def _nearest(items: list, t: float, tolerance: float, key) -> object | None:
    """Return whichever item in `items` (sorted by `key(item)`) is closest to `t`, within
    `tolerance` seconds, or `None`. `items` need not be pre-sorted (sorted here)."""
    if not items:
        return None
    items_sorted = sorted(items, key=key)
    keys = [key(it) for it in items_sorted]
    pos = bisect_left(keys, t)
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(items_sorted)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(keys[i] - t))
    return items_sorted[best] if abs(keys[best] - t) <= tolerance else None


def render_overlay(video_path: Path, work_root: str | Path = "work") -> Path:
    hardware_config = load_yaml("configs/hardware.yaml")
    detect_config = load_yaml("configs/detect.yaml")
    track_config = load_yaml("configs/track.yaml")
    team_config = load_yaml("configs/team.yaml")
    shots_config = load_yaml("configs/shots.yaml")

    run_track_stage(
        video_path,
        hardware_config,
        detect_config,
        track_config,
        team_config,
        shots_config,
        work_root=work_root,
    )

    work_dir = work_dir_for(video_path, root=work_root)
    tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    ball_detections = load_models_parquet(
        work_dir / "detect" / "ball_detections.parquet", BallDetection
    )
    arrow_hints_path = work_dir / "detect" / "arrow_hints.parquet"
    arrow_hints = (
        load_models_parquet(arrow_hints_path, ArrowHint) if arrow_hints_path.exists() else []
    )
    takes = detect_takes(video_path, shots_config, work_root=work_root)

    fps_sample = hardware_config["stages"]["detect"]["fps_sample"]
    scale_width = hardware_config["decode"]["scale_width"]
    tolerance = 0.5 / fps_sample if fps_sample > 0 else 0.5

    out_width = OUTPUT_WIDTH_PX
    out_height = None

    frames_out = []
    n_frames = 0
    for _idx, t, frame in decode_frames(
        video_path, fps=fps_sample, scale_width=scale_width, use_nvdec=True
    ):
        take_id = assign_take_id(t, takes)
        take = next((tk for tk in takes if tk.id == take_id), None)

        active = []
        for tr in tracks:
            if tr.take_id != take_id:
                continue
            box = _nearest(tr.boxes, t, tolerance, key=lambda b: b.t)
            if box is not None:
                active.append((tr.id, box.bbox, tr.team))
        annotated = draw_tracks(frame, active)

        ball = _nearest(ball_detections, t, tolerance, key=lambda b: b.t)
        if ball is not None:
            annotated = draw_ball(annotated, ball.bbox, ball.conf, ball.interpolated)

        arrow = _nearest(arrow_hints, t, tolerance, key=lambda a: a.t)
        if arrow is not None:
            annotated = draw_arrow_outline(annotated, arrow.bbox, (arrow.tip_x, arrow.tip_y))

        if take is not None and take.id > 0 and take.t_start <= t < take.t_start + BANNER_SECONDS:
            annotated = draw_banner(annotated, f"CUT -> take {take.id}")

        annotated = draw_hud(annotated, _idx, t, stage="track")

        height, width = annotated.shape[:2]
        if out_height is None:
            out_height = (round(height * (out_width / width)) // 2) * 2

        resized = cv2.resize(annotated, (out_width, out_height), interpolation=cv2.INTER_AREA)
        frames_out.append(resized)
        n_frames += 1

    if not frames_out:
        raise RuntimeError(f"no frames decoded from {video_path}; cannot render overlay")

    out_path = work_dir / "debug" / "overlay_tracks.mp4"
    write_video(
        iter(frames_out),
        out_path,
        fps=fps_sample,
        size=(out_width, out_height),
        encoder="h264_nvenc",
    )
    logger.info("wrote %d frame(s) -> %s", n_frames, out_path)
    return out_path


@app.command()
def main(video: Path = typer.Argument(..., help="Path to a video in input/")) -> None:
    out_path = render_overlay(video)
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
