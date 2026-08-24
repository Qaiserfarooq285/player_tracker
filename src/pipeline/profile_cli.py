"""Stage 0.5 CLI — profile every video in `input/` (CLAUDE.md §5 Stage 0.5).

Wired to `make profile` (see Makefile): ``python -m src.pipeline.profile_cli input``. Prints a
rich summary table and writes each video's `RunProfile` to ``work/<slug>/profile.json`` (via
`src.pipeline.profiler.build_run_profile`, which handles the resumable cache).
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.common.io import load_yaml
from src.common.logging import get_logger
from src.ingest.discovery import find_videos
from src.pipeline.profiler import build_run_profile

logger = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_dir: Path = typer.Argument(Path("input"), help="Directory of input videos"),
    run_config_path: Path = typer.Option(Path("configs/run.yaml"), help="CLAUDE.md §3.1 config"),
    ingest_config_path: Path = typer.Option(Path("configs/ingest.yaml")),
    hardware_config_path: Path = typer.Option(Path("configs/hardware.yaml")),
    profile_config_path: Path = typer.Option(Path("configs/profile.yaml")),
    shots_config_path: Path = typer.Option(Path("configs/shots.yaml")),
    work_root: Path = typer.Option(Path("work")),
) -> None:
    """Run the Stage 0.5 source profiler over every video in `input_dir` (CLAUDE.md Stage 0.5)."""
    run_config = load_yaml(run_config_path)
    ingest_config = load_yaml(ingest_config_path)
    hardware_config = load_yaml(hardware_config_path)
    profile_config = load_yaml(profile_config_path)
    shots_config = load_yaml(shots_config_path)

    videos = find_videos(
        input_dir,
        run_config["filename_convention_regex"],
        ingest_config["video_extensions"],
    )
    if not videos:
        console.print(f"[yellow]No videos found in {input_dir}[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"Source profile — {input_dir}")
    for col in (
        "file",
        "resolution",
        "fps",
        "duration",
        "cuts",
        "cuts/min",
        "motion (px/frame)",
        "profile",
        "quality",
        "target_jersey",
        "wall (s)",
    ):
        table.add_column(col)

    for ref in videos:
        t0 = time.time()
        run_profile = build_run_profile(
            ref.path,
            hardware_config,
            profile_config,
            shots_config,
            work_root=work_root,
        )
        wall = time.time() - t0
        for note in run_profile.notes:
            logger.info("[%s] note: %s", ref.path.name, note)
        table.add_row(
            ref.path.name,
            f"{run_profile.width}x{run_profile.height}",
            f"{run_profile.fps:.2f}",
            f"{run_profile.duration:.2f}s",
            str(run_profile.n_cuts),
            f"{run_profile.cuts_per_min:.2f}",
            f"{run_profile.motion_score:.3f}",
            run_profile.profile.value,
            run_profile.quality_flag,
            str(ref.target_jersey) if ref.target_jersey is not None else "-",
            f"{wall:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
