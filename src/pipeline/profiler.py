"""Stage 0.5 — source profiler (CLAUDE.md §5 Stage 0.5).

Classifies input as `broadcast` / `single-static` / `single-panning` so downstream stages branch
correctly (CLAUDE.md §3.1, ADR-7). Zero model loads — cuts come from `src.shots.boundaries`
(PySceneDetect), motion from classical (non-learned) optical flow. Output is a :class:`RunProfile`,
cached per video at ``work/<slug>/profile.json``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common.io import StageCache, load_json, save_json, work_dir_for
from src.common.logging import DropCounter, get_logger
from src.common.types import RunProfile, SourceProfile
from src.common.video import decode_frames, probe
from src.shots.boundaries import detect_takes

logger = get_logger(__name__)


def compute_motion_score(
    video_path: str | Path,
    hardware_config: dict,
    profile_config: dict,
    use_nvdec: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Estimate global camera motion for `video_path`, in px/**native video frame** at native
    resolution (matching CLAUDE.md §3.2's own `vidstabdetect`-measured "~1 px/frame" figure and
    `configs/profile.yaml`'s `static_motion_threshold`/`panning_motion_threshold`).

    **Estimator: sparse good-features-to-track + Lucas-Kanade flow, fit to one dominant motion
    model per anchor via `cv2.estimateAffinePartial2D(..., RANSAC)`, aggregated across anchors by
    median.** This is the third design tried on this footage — the first two were built, measured,
    and rejected empirically (not just in theory), which is worth recording:

    1. *Dense Farneback flow, grid-binned + median-across-cells, on frames sampled at
       `fps_sample`.* Rejected: measured motion_score was wildly resolution-dependent (0.07 at
       full 4K vs 6.1 at 240px on the same clip, on what should be a resolution-invariant
       quantity) — dense flow on this footage's largely textureless grass produces
       downscale-amplified noise, not a real signal, and it doesn't converge until you decode at
       (slow) near-native resolution.
    2. *Sparse KLT + RANSAC affine, but computed between frames sampled `fps_sample` apart* (0.5s
       / 15 native frames at `fps_sample=2`, 30fps). Rejected: KLT's small-inter-frame-displacement
       assumption breaks down over a 15-frame gap, producing wildly inconsistent per-anchor
       results (many spurious exact-zero readings from `-ss`-seek frame duplication, others
       inflated by real but oversized inter-sample motion) with a very low RANSAC inlier rate.

    This (third) design fixes both: KLT/affine-fit is computed between **truly consecutive native
    frames** (one native-frame gap, well inside KLT's small-motion assumption) at each of
    `fps_sample`-per-second anchor points, decoded via a **single continuous sequential pass**
    (no repeated `-ss` seeking, which is what caused design (2)'s frame-duplication artifacts) —
    the (cheap, GPU-side) full native-fps decode is kept, but the (CPU-side) expensive
    tracking/RANSAC step only runs on frame pairs that land on an anchor. `RANSAC` is what makes
    each anchor's estimate robust to players (a minority of tracked points at this camera's wide
    framing, CLAUDE.md §3.2): their motion gets voted down as outliers against the majority
    background (treeline/stands/pitch-line) consensus — and empirically the majority *is*
    background: `goodFeaturesToTrack` naturally avoids the low-texture grass and clusters on
    exactly those high-contrast structures.

    **Median, not mean, across anchors**: eyeballing the decoded frames confirmed all 5 of
    CLAUDE.md §3.2's clips share an un-documented characteristic — a genuine (not noise, not an
    artifact) camera **settling pan lasting ~3-5s at the very start of every clip**, before the
    shot goes fully static for the remainder. A mean over the whole clip lets that transient skew
    short clips heavily (it's a bigger fraction of an 11s clip than a 110s one); the median is
    dominated by the long static remainder instead. This is a real, reportable footage
    characteristic in its own right, not something being smoothed away to force a "static" answer
    — see the run report for the actual per-clip numbers, including where this still pushes a
    clip's classification to `single-panning`.

    `start`/`end` (seconds, optional): when given, scopes the WHOLE estimate to that window of
    `video_path` instead of the entire video — same anchor-based algorithm, same
    `decode_frames`-level `start`/`end` windowing every other windowed-decode caller in this
    codebase already uses (e.g. `src/events/goals.py::detect_goals_for_take`), just restricting
    the OUTPUT to one take instead of the whole video. Added for `src/goal/detect.py` (Stage C,
    "streamed-gathering-treehouse" plan), which needs a PER-TAKE motion score (ADR-11: consume the
    take's own measured signal, never the whole-video aggregate, when a take-specific decision —
    how often to re-localize the tracked goal structure — is being made). `None`/`None` (the
    default) is the original whole-video behaviour, byte-for-byte unchanged.
    """
    video_path = Path(video_path)
    meta = probe(video_path)
    native_width = meta["width"]
    native_fps = meta["fps"]
    duration = (end - start) if (start is not None and end is not None) else meta["duration"]

    fps_sample = hardware_config["stages"]["profile"]["fps_sample"]
    motion_cfg = profile_config["motion"]
    flow_width = motion_cfg["flow_width"]

    drops = DropCounter("profile.motion")

    if duration <= 0 or fps_sample <= 0:
        drops.drop("insufficient_frames_for_motion", 1)
        drops.report()
        return 0.0

    n_anchors = max(1, round(duration * fps_sample))
    anchor_frames = {round(i * native_fps / fps_sample) for i in range(n_anchors)}

    magnitudes: list[float] = []
    prev: np.ndarray | None = None
    native_scale = 1.0
    for index, _t, frame in decode_frames(
        video_path, fps=None, start=start, end=end, scale_width=flow_width, use_nvdec=use_nvdec
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None and index in anchor_frames:
            native_scale = native_width / gray.shape[1] if gray.shape[1] else 1.0
            mag = _anchor_translation_magnitude(prev, gray, motion_cfg)
            if mag is None:
                drops.drop("anchor_flow_failed", 1)
            else:
                magnitudes.append(mag * native_scale)
        prev = gray

    drops.report()
    if not magnitudes:
        return 0.0
    return float(np.median(magnitudes))


def _anchor_translation_magnitude(
    prev: np.ndarray, curr: np.ndarray, motion_cfg: dict
) -> float | None:
    """Sparse-feature RANSAC-affine translation magnitude between two consecutive frames.

    Returns ``None`` (a dropped anchor, not a fabricated ``0.0``) when too few features are found,
    tracked, or agree on a model — see :func:`compute_motion_score`'s docstring for why RANSAC
    (not a plain mean/median of point displacements) is the robust step here.
    """
    pts = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=motion_cfg["max_corners"],
        qualityLevel=motion_cfg["quality_level"],
        minDistance=motion_cfg["min_distance"],
        blockSize=motion_cfg["block_size"],
    )
    if pts is None or len(pts) < motion_cfg["min_tracked_points"]:
        return None

    win = motion_cfg["lk_win_size"]
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev, curr, pts, None, winSize=(win, win), maxLevel=motion_cfg["lk_max_level"]
    )
    ok = status.reshape(-1).astype(bool)
    p0, p1 = pts[ok], next_pts[ok]
    if len(p0) < motion_cfg["min_tracked_points"]:
        return None

    matrix, _inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=motion_cfg["ransac_reproj_threshold"]
    )
    if matrix is None:
        return None
    tx, ty = matrix[0, 2], matrix[1, 2]
    return float(np.hypot(tx, ty))


def _classify(
    n_cuts: int,
    cuts_per_min: float,
    motion_score: float,
    profile_config: dict,
) -> tuple[SourceProfile, list[str]]:
    """Pure classification decision table (CLAUDE.md §5 Stage 0.5 / this task's spec).

    Kept separate from I/O (`build_run_profile`) so it's directly unit-testable with synthetic
    ``(cuts_per_min, motion_score)`` pairs, including the ambiguous motion band.
    """
    classification_cfg = profile_config["classification"]
    motion_cfg = profile_config["motion"]
    broadcast_thr = classification_cfg["broadcast_cuts_per_min_threshold"]
    static_thr = motion_cfg["static_motion_threshold"]
    panning_thr = motion_cfg["panning_motion_threshold"]

    notes: list[str] = []

    if cuts_per_min >= broadcast_thr:
        profile = SourceProfile.BROADCAST
    elif motion_score < static_thr:
        profile = SourceProfile.SINGLE_STATIC
    elif motion_score > panning_thr:
        profile = SourceProfile.SINGLE_PANNING
    else:
        dist_to_static = motion_score - static_thr
        dist_to_panning = panning_thr - motion_score
        profile = (
            SourceProfile.SINGLE_STATIC
            if dist_to_static <= dist_to_panning
            else SourceProfile.SINGLE_PANNING
        )
        notes.append(
            f"motion_score={motion_score:.3f} px/frame falls in the ambiguous band between "
            f"static_motion_threshold={static_thr} and panning_motion_threshold={panning_thr}; "
            f"classified as {profile.value} (nearer threshold)."
        )

    if profile != SourceProfile.BROADCAST and n_cuts > 0:
        notes.append(
            f"single-camera source still contains {n_cuts} cut(s) (ADR-7 / CLAUDE.md §3.2(2)) — "
            "shot-boundary detection ran regardless of profile; downstream tracking must still "
            "reset IDs at every take boundary."
        )

    return profile, notes


def build_run_profile(
    video_path: str | Path,
    hardware_config: dict,
    profile_config: dict,
    shots_config: dict,
    work_root: str | Path = "work",
    use_nvdec: bool = True,
) -> RunProfile:
    """Compute (or load from cache) the Stage 0.5 :class:`RunProfile` for `video_path`.

    Cached at ``work/<slug>/profile.json`` (CLAUDE.md §10), keyed on the relevant config values
    plus the source video's size/mtime.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    profile_path = work_dir / "profile.json"

    cache_config = {
        "decode": hardware_config["decode"],
        "profile_fps_sample": hardware_config["stages"]["profile"]["fps_sample"],
        "profile_config": profile_config,
        "scenedetect": shots_config["scenedetect"],
        "video": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime": video_path.stat().st_mtime,
    }
    cache = StageCache(profile_path, cache_config, stage="profile")
    if cache.hit():
        return RunProfile.model_validate(load_json(profile_path))

    meta = probe(video_path)

    takes = detect_takes(video_path, shots_config, work_root=work_root, use_nvdec=use_nvdec)
    n_cuts = max(0, len(takes) - 1)
    cuts_per_min = (n_cuts / (meta["duration"] / 60.0)) if meta["duration"] > 0 else 0.0

    motion_score = compute_motion_score(
        video_path, hardware_config, profile_config, use_nvdec=use_nvdec
    )

    quality_min_res = profile_config["classification"]["quality_min_resolution_px"]
    quality_flag = "low_res" if min(meta["width"], meta["height"]) < quality_min_res else "ok"

    profile, notes = _classify(n_cuts, cuts_per_min, motion_score, profile_config)

    run_profile = RunProfile(
        profile=profile,
        n_cuts=n_cuts,
        cuts_per_min=round(cuts_per_min, 4),
        motion_score=round(motion_score, 4),
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        duration=meta["duration"],
        quality_flag=quality_flag,
        notes=notes,
    )

    save_json(run_profile, profile_path)
    cache.write_meta()
    logger.info(
        "profile: %s -> %s (cuts=%d, cuts/min=%.2f, motion=%.3f px/frame)",
        video_path.name,
        profile.value,
        n_cuts,
        cuts_per_min,
        motion_score,
    )
    return run_profile
