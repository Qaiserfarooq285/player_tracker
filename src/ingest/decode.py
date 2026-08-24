"""Stage 0 — decode input video, sample frames per stage config (CLAUDE.md Stage 0).

Builds a resumable **frame manifest** (`index`, `t`, `chunk_id`) rather than dumping every decoded
frame to disk (CLAUDE.md §10/§11: a 90-min match at 25-30 fps is ~135k frames — writing that many
JPEGs by default would be both slow and a disk-space trap). The manifest records which frames
*would* be sampled at a given fps for a given stage; downstream stages re-decode the (small) window
of frames they actually need via `common.video.decode_frames`, keyed off this manifest.

Chunked + resumable (CLAUDE.md §11 "chunk the match ... make every run resumable"): the video is
split into fixed `chunk_seconds` windows, each decoded and cached independently via
:class:`~src.common.io.StageCache`. A crash mid-run only redoes the chunk(s) that hadn't finished;
a second full run is a single cache hit at the merged-manifest level.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from src.common.io import StageCache, load_models_parquet, save_models_parquet, work_dir_for
from src.common.logging import DropCounter, get_logger
from src.common.video import decode_frames, probe

logger = get_logger(__name__)

# Cache of one-shot NVDEC preflight results, keyed by video path, so repeated chunk decodes for
# the same video don't re-run the preflight probe (CLAUDE.md task: "log which [decode] path used").
_NVDEC_PROBE_CACHE: dict[str, bool] = {}


class FrameManifestRow(BaseModel):
    """One row of the Stage 0 frame manifest: a frame that would be sampled, and where."""

    index: int
    t: float
    chunk_id: int


def chunk_ranges(duration: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Split ``[0, duration)`` into fixed-size ``[start, end)`` chunks of `chunk_seconds`.

    The final chunk is clipped to `duration` (may be shorter than `chunk_seconds`). Pure function,
    no I/O — the chunk boundaries downstream resumability keys off (CLAUDE.md §11).
    """
    if duration <= 0 or chunk_seconds <= 0:
        return []
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        ranges.append((start, end))
        start = end
    return ranges


def _nvdec_supported(path: str | Path) -> bool:
    """One-shot preflight: does ``ffmpeg -hwaccel cuda -c:v h264_cuvid`` decode >=1 frame of `path`?

    Result is cached per video path for the process lifetime and logged exactly once, so it's
    clear from the log which decode path (NVDEC vs CPU) a given video actually used, independent
    of `common.video.decode_frames`'s own per-attempt fallback logging.
    """
    key = str(path)
    if key in _NVDEC_PROBE_CACHE:
        return _NVDEC_PROBE_CACHE[key]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        supported = result.returncode == 0 and len(result.stdout) > 0
    except (OSError, subprocess.TimeoutExpired):
        supported = False
    _NVDEC_PROBE_CACHE[key] = supported
    logger.info(
        "decode path for %s: %s",
        path,
        "NVDEC (h264_cuvid)" if supported else "CPU (NVDEC unavailable)",
    )
    return supported


def build_frame_manifest(
    video_path: str | Path,
    hardware_config: dict,
    work_root: str | Path = "work",
    stage: str = "ingest",
    use_nvdec: bool = True,
    dump_frames: bool = False,
    dump_max_frames: int = 40,
) -> Path:
    """Build (or load from cache) Stage 0's frame manifest for `video_path`.

    Reads `hardware_config["decode"]` (`scale_width`, `chunk_seconds`) and
    `hardware_config["stages"][stage]["fps_sample"]` (CLAUDE.md §11 per-stage fps sampling) —
    caller passes the already-loaded `configs/hardware.yaml` dict.

    Returns the path to the merged manifest parquet at
    ``work/<slug>/ingest/frame_manifest.parquet``. When `dump_frames` is set, also writes an
    evenly-sampled subset (<= `dump_max_frames`) of frames as PNGs to
    ``work/<slug>/frames_debug/`` for eyeballing decode correctness — opt-in only.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    ingest_dir = work_dir / "ingest"
    chunks_dir = ingest_dir / "chunks"
    manifest_path = ingest_dir / "frame_manifest.parquet"

    meta = probe(video_path)
    decode_cfg = hardware_config["decode"]
    fps_sample = hardware_config["stages"][stage]["fps_sample"]
    scale_width = decode_cfg["scale_width"]
    chunk_seconds = decode_cfg["chunk_seconds"]

    nvdec_ok = use_nvdec and _nvdec_supported(video_path)
    ranges = chunk_ranges(meta["duration"], chunk_seconds)

    manifest_config = {
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
        "fps_sample": fps_sample,
        "scale_width": scale_width,
        "chunk_seconds": chunk_seconds,
        "n_chunks": len(ranges),
    }
    manifest_cache = StageCache(manifest_path, manifest_config, stage=f"{stage}.manifest")
    if manifest_cache.hit():
        if dump_frames:
            rows = load_models_parquet(manifest_path, FrameManifestRow)
            _dump_debug_frames(
                video_path, work_dir, rows, fps_sample, scale_width, nvdec_ok, dump_max_frames
            )
        return manifest_path

    drops = DropCounter(stage)
    all_rows: list[FrameManifestRow] = []
    global_index = 0
    for chunk_id, (start, end) in enumerate(ranges):
        chunk_path = chunks_dir / f"chunk_{chunk_id:04d}.parquet"
        chunk_config = {**manifest_config, "chunk_id": chunk_id, "start": start, "end": end}
        chunk_cache = StageCache(chunk_path, chunk_config, stage=f"{stage}.chunk")

        expected = max(0, round((end - start) * fps_sample))

        def _compute(start=start, end=end, chunk_id=chunk_id) -> list[FrameManifestRow]:
            return [
                FrameManifestRow(index=local_idx, t=t, chunk_id=chunk_id)
                for local_idx, t, _frame in decode_frames(
                    video_path,
                    fps=fps_sample,
                    start=start,
                    end=end,
                    scale_width=scale_width,
                    use_nvdec=nvdec_ok,
                )
            ]

        rows = chunk_cache.get_or_compute(
            _compute,
            lambda r, p: save_models_parquet(r, p),
            lambda p: load_models_parquet(p, FrameManifestRow),
        )
        if expected and len(rows) < expected:
            drops.drop(f"short_chunk_decode(chunk_{chunk_id})", expected - len(rows))

        for row in rows:
            all_rows.append(FrameManifestRow(index=global_index, t=row.t, chunk_id=chunk_id))
            global_index += 1

    save_models_parquet(all_rows, manifest_path)
    manifest_cache.write_meta()
    drops.report()

    if dump_frames:
        _dump_debug_frames(
            video_path, work_dir, all_rows, fps_sample, scale_width, nvdec_ok, dump_max_frames
        )

    return manifest_path


def _dump_debug_frames(
    video_path: Path,
    work_dir: Path,
    rows: list[FrameManifestRow],
    fps_sample: float,
    scale_width: int | None,
    use_nvdec: bool,
    max_frames: int,
) -> None:
    """Write an evenly-sampled subset (<= `max_frames`) of manifest frames as PNGs for debugging.

    Opt-in only (`--dump-frames`); never called by default (CLAUDE.md §10).

    Decodes the video **once, continuously**, at the same ``(fps_sample, scale_width)`` the
    manifest itself was built at, rather than seeking (``-ss``) to each target timestamp
    independently: measured 2026-08-24, a tiny (~1-frame) ``-ss``-seek window is enough to make
    NVDEC hang for ~90s before falling back to CPU decode on this footage -- repeating that per
    dumped frame made a 40-frame debug dump take minutes. A single sequential pass has none of
    that seek overhead and costs one full decode instead of `max_frames` partial ones.
    """
    if not rows:
        return
    debug_dir = work_dir / "frames_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    wanted = set(np.linspace(0, n - 1, num=min(max_frames, n), dtype=int).tolist())

    written = 0
    for running_index, (_idx, t, frame) in enumerate(
        decode_frames(video_path, fps=fps_sample, scale_width=scale_width, use_nvdec=use_nvdec)
    ):
        if running_index not in wanted:
            continue
        out_path = debug_dir / f"frame_{running_index:06d}_t{t:.3f}.png"
        cv2.imwrite(str(out_path), frame)
        written += 1
        if written >= len(wanted):
            break
    logger.info("dumped %d debug frame(s) -> %s", written, debug_dir)


if __name__ == "__main__":
    import typer

    from src.common.io import load_yaml

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(
        video: Path = typer.Argument(..., help="Path to a video in input/"),
        dump_frames: bool = typer.Option(False, "--dump-frames", help="Write a debug frame subset"),
        hardware_config_path: Path = typer.Option(Path("configs/hardware.yaml")),
        ingest_config_path: Path = typer.Option(Path("configs/ingest.yaml")),
    ) -> None:
        """Build the Stage 0 frame manifest for a single video (debug/manual invocation)."""
        hardware_config = load_yaml(hardware_config_path)
        ingest_config = load_yaml(ingest_config_path)
        max_frames = ingest_config["debug_dump"]["max_frames"]
        path = build_frame_manifest(
            video, hardware_config, dump_frames=dump_frames, dump_max_frames=max_frames
        )
        logger.info("manifest -> %s", path)

    app()
