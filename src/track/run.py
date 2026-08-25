"""Stage 3 CLI — within-take ByteTrack + SigLIP/UMAP/KMeans team assignment (CLAUDE.md §5 Stage
3). Wired to ``python -m src.track.run --video "<path>"`` (see
`.claude/commands/process-match.md`).

Runs Stage 2 first if its cache is missing (`src.detect.run.run_detect_stage` is itself a no-op on
a cache hit, CLAUDE.md §10 resumability), then Stage 1's take boundaries
(`src.shots.boundaries.detect_takes`, cached, ADR-7 — runs for every profile), buckets Stage 2's
(take-agnostic) sampled detections into takes by timestamp, tracks each take with a **fresh**
`ByteTrack` (Golden Rule 3 — IDs reset at every cut), then assigns team per take.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from src.common.io import (
    StageCache,
    load_models_parquet,
    load_yaml,
    save_models_parquet,
    work_dir_for,
)
from src.common.logging import get_logger
from src.common.types import Detection, Track
from src.detect.run import run_detect_stage
from src.shots.boundaries import detect_takes
from src.team.classifier import assign_teams, collect_track_crops
from src.track.tracker import assign_take_id, track_take

logger = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False)


def run_track_stage(
    video_path: str | Path,
    hardware_config: dict,
    detect_config: dict,
    track_config: dict,
    team_config: dict,
    shots_config: dict,
    work_root: str | Path = "work",
    use_nvdec: bool = True,
) -> dict[str, Any]:
    """Run Stage 3 for one video: per-take ByteTrack + team assignment, cached to
    ``work/<slug>/track/tracks.parquet``. Returns a summary dict for reporting.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    tracks_path = work_dir / "track" / "tracks.parquet"

    # Stage 2 is a prerequisite; its own run is a cache no-op if already done.
    detect_summary = run_detect_stage(
        video_path, hardware_config, detect_config, work_root=work_root, use_nvdec=use_nvdec
    )
    takes = detect_takes(video_path, shots_config, work_root=work_root, use_nvdec=use_nvdec)

    video_identity = {
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
    }
    cache_config = {
        **video_identity,
        "track_config": track_config,
        "team_config": team_config,
        "detect_stage_fps_sample": hardware_config["stages"]["detect"]["fps_sample"],
        "team_stage": hardware_config["stages"]["team"],
        "n_takes": len(takes),
    }
    cache = StageCache(tracks_path, cache_config, stage="track.tracks")
    if cache.hit():
        tracks = load_models_parquet(tracks_path, Track)
        return {
            "video": video_path.name,
            "n_takes": len(takes),
            "n_tracks": len(tracks),
            "tracks_per_take": _counts_per_take(tracks),
            "wall_track_s": 0.0,
            "wall_team_s": 0.0,
            "team_method_per_take": {},
            "detect_summary": detect_summary,
        }

    detections_path = work_dir / "detect" / "detections.parquet"
    all_detections = load_models_parquet(detections_path, Detection)

    # Bucket Stage 2's take-agnostic sampled detections into takes by timestamp (ADR-7: takes
    # come from shot-boundary detection regardless of source profile; Stage 2 itself never had to
    # know about takes at all).
    detections_by_take: dict[int, dict[int, list[Detection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_unassigned = 0
    for det in all_detections:
        take_id = assign_take_id(det.t, takes)
        if take_id is None:
            n_unassigned += 1
            continue
        detections_by_take[take_id][det.frame_index].append(det)
    if n_unassigned:
        logger.warning(
            "%d detection(s) fell outside every take's [t_start, t_end) and were dropped from "
            "tracking (should be rare/zero — takes are gapless by construction)",
            n_unassigned,
        )

    t0 = time.time()
    all_tracks: list[Track] = []
    fps_sample = hardware_config["stages"]["detect"]["fps_sample"]
    for take in takes:
        frame_map = detections_by_take.get(take.id, {})
        frames_sorted = sorted(frame_map.items(), key=lambda kv: kv[0])
        detections_by_frame = [
            (frame_index, dets[0].t if dets else 0.0, dets) for frame_index, dets in frames_sorted
        ]
        take_tracks = track_take(detections_by_frame, take, track_config, fps_sample)
        all_tracks.extend(take_tracks)
        logger.info(
            "take %d: %d track(s) from %d sampled frame(s)",
            take.id,
            len(take_tracks),
            len(frames_sorted),
        )
    wall_track_s = time.time() - t0

    t0 = time.time()
    team_method_per_take: dict[int, str] = {}
    decode_cfg = hardware_config["decode"]
    team_stage_cfg = hardware_config["stages"]["team"]
    for take in takes:
        take_tracks = [tr for tr in all_tracks if tr.take_id == take.id]
        if not take_tracks:
            continue
        crops_by_track = collect_track_crops(
            video_path,
            take,
            take_tracks,
            decode_cfg,
            team_stage_cfg,
            team_config["sampling"],
            use_nvdec=use_nvdec,
        )
        dominant_class_by_track = {tr.id: tr.dominant_class for tr in take_tracks}
        team_by_track, confidence_by_track, method = assign_teams(
            crops_by_track, dominant_class_by_track, team_config, hardware_config
        )
        team_method_per_take[take.id] = method
        for tr in take_tracks:
            tr.team = team_by_track.get(tr.id)
            tr.team_confidence = confidence_by_track.get(tr.id, 0.0)
    wall_team_s = time.time() - t0

    save_models_parquet(all_tracks, tracks_path)
    cache.write_meta()

    return {
        "video": video_path.name,
        "n_takes": len(takes),
        "n_tracks": len(all_tracks),
        "tracks_per_take": _counts_per_take(all_tracks),
        "wall_track_s": round(wall_track_s, 2),
        "wall_team_s": round(wall_team_s, 2),
        "team_method_per_take": team_method_per_take,
        "detect_summary": detect_summary,
    }


def _counts_per_take(tracks: list[Track]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for tr in tracks:
        counts[tr.take_id] += 1
    return dict(sorted(counts.items()))


@app.command()
def main(
    video: Path = typer.Argument(..., help="Path to a video in input/"),
    hardware_config_path: Path = typer.Option(Path("configs/hardware.yaml")),
    detect_config_path: Path = typer.Option(Path("configs/detect.yaml")),
    track_config_path: Path = typer.Option(Path("configs/track.yaml")),
    team_config_path: Path = typer.Option(Path("configs/team.yaml")),
    shots_config_path: Path = typer.Option(Path("configs/shots.yaml")),
    work_root: Path = typer.Option(Path("work")),
) -> None:
    """Run Stage 3 (within-take ByteTrack + team assignment) for one video."""
    hardware_config = load_yaml(hardware_config_path)
    detect_config = load_yaml(detect_config_path)
    track_config = load_yaml(track_config_path)
    team_config = load_yaml(team_config_path)
    shots_config = load_yaml(shots_config_path)

    summary = run_track_stage(
        video,
        hardware_config,
        detect_config,
        track_config,
        team_config,
        shots_config,
        work_root=work_root,
    )
    console.print(summary)


if __name__ == "__main__":
    app()
