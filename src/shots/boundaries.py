"""Stage 1 — camera-take segmentation (all profiles, ADR-7) + replay/close-up/scoreboard-OCR
(broadcast-gated). See configs/shots.yaml.

ADR-7 (CLAUDE.md §5.1): shot-boundary detection runs for **every** profile, not just `broadcast`
— clip4 proved a single-camera source can still contain real cuts. This module is therefore also
the engine the Stage 0.5 source profiler (`src/pipeline/profiler.py`) calls for its `n_cuts`/
`cuts_per_min`, so the `scenedetect` knobs live once, in `configs/shots.yaml`.

PySceneDetect's `ContentDetector` is driven **frame-by-frame** via its public `process_frame`
API rather than through `scenedetect`'s own `SceneManager`/`open_video` (which decodes through
OpenCV with no hardware acceleration). Instead we feed it frames from
`src.common.video.decode_frames` — the same NVDEC-first, CPU-fallback decode path the rest of the
pipeline uses (ADR-1) — which both reuses existing, tested decode machinery and is measurably
faster on this 4K footage.
"""

from __future__ import annotations

from pathlib import Path

from scenedetect.common import FrameTimecode
from scenedetect.detectors import ContentDetector

from src.common.io import StageCache, load_models_parquet, save_models_parquet, work_dir_for
from src.common.logging import get_logger
from src.common.types import Take
from src.common.video import decode_frames, probe
from src.ingest.decode import _nvdec_supported

logger = get_logger(__name__)


def takes_from_cut_frames(cut_frames: list[int], n_frames: int, fps: float) -> list[Take]:
    """Convert a list of cut frame indices into a gapless, non-overlapping :class:`Take` list
    covering ``[0, n_frames)``.

    Pure function, no video I/O — cut frame index `f` means "a new take starts at frame `f`"
    (equivalently: the previous take ends at frame `f - 1`). Out-of-range (``<= 0`` or
    ``>= n_frames``) and duplicate cut indices are ignored so the result always has full coverage
    with no gaps or overlaps, even for a pathological/empty `cut_frames`.
    """
    if n_frames <= 0:
        return []
    cuts = sorted({c for c in cut_frames if 0 < c < n_frames})
    boundaries = [0, *cuts, n_frames]
    takes: list[Take] = []
    for i in range(len(boundaries) - 1):
        frame_start, frame_end = boundaries[i], boundaries[i + 1]
        takes.append(
            Take(
                id=i,
                t_start=frame_start / fps,
                t_end=frame_end / fps,
                frame_start=frame_start,
                frame_end=frame_end,
                kind="unknown",
            )
        )
    return takes


def resolve_scenedetect_threshold(sd_cfg: dict, slug: str) -> float:
    """ADR-16: the `ContentDetector` threshold to use for one video, by its `work_dir_for()` slug.

    A per-video entry in `sd_cfg["overrides"]` (keyed by slug) wins over `sd_cfg["threshold"]`
    when present; every video without an entry (the original 5 `clip<N> <jersey>.mp4` clips as of
    this writing) falls back to the global default unchanged. Pure/config-only so it's unit
    testable without touching a real video file — see `tests/test_shots.py`.
    """
    return sd_cfg.get("overrides", {}).get(slug, sd_cfg["threshold"])


def detect_takes(
    video_path: str | Path,
    shots_config: dict,
    work_root: str | Path = "work",
    use_nvdec: bool = True,
) -> list[Take]:
    """Run `ContentDetector` over the whole of `video_path` and return its :class:`Take` list.

    `shots_config` is the loaded `configs/shots.yaml` dict. Cached per video at
    ``work/<slug>/shots/takes.parquet`` (CLAUDE.md §10), keyed on the `scenedetect` config plus
    the source video's size/mtime so replacing the file invalidates the cache too.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    takes_path = work_dir / "shots" / "takes.parquet"

    sd_cfg = shots_config["scenedetect"]
    meta = probe(video_path)
    fps = meta["fps"]

    # ADR-16: a per-video override wins over the global default when the video's own slug (the
    # same one work_dir_for() uses for work/output dirs) is present in `overrides` — absent for
    # every one of the original 5 clips, so their behaviour is provably unchanged.
    slug = work_dir.name
    threshold = resolve_scenedetect_threshold(sd_cfg, slug)
    if slug in sd_cfg.get("overrides", {}):
        logger.info(
            "shots: using per-video threshold override for slug=%s: %.1f (global default %.1f)",
            slug,
            threshold,
            sd_cfg["threshold"],
        )

    cache_config = {
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
        **sd_cfg,
        "resolved_threshold": threshold,
    }
    cache = StageCache(takes_path, cache_config, stage="shots.boundaries")
    if cache.hit():
        return load_models_parquet(takes_path, Take)

    nvdec_ok = use_nvdec and _nvdec_supported(video_path)

    detector = ContentDetector(
        threshold=threshold,
        min_scene_len=sd_cfg["min_scene_len_frames"],
    )
    cut_events: list[FrameTimecode] = []
    last_index = 0
    n_decoded = 0
    for index, _t, frame in decode_frames(
        video_path,
        fps=None,  # full native fps — cut detection must not miss single-frame cuts
        scale_width=sd_cfg["downscale_width"],
        use_nvdec=nvdec_ok,
    ):
        cut_events += detector.process_frame(FrameTimecode(index, fps), frame)
        last_index = index
        n_decoded += 1
    cut_events += detector.post_process(FrameTimecode(last_index, fps))
    cut_frames = sorted({c.frame_num for c in cut_events})

    takes = takes_from_cut_frames(cut_frames, n_decoded, fps)
    save_models_parquet(takes, takes_path)
    cache.write_meta()

    logger.info(
        "shots: %s -> %d take(s) from %d cut(s) at t=%s",
        video_path.name,
        len(takes),
        len(cut_frames),
        [round(c / fps, 3) for c in cut_frames],
    )
    return takes
