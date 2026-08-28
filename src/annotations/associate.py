"""ADR-19 -- manual-mode track association by colour + timing (CLAUDE.md §14.3 Stage 4).

For each parsed `Annotation`, best-effort-find which (if any) of that take's own tracks is a
plausible visual match for the client's own stated `team_colour`, so the annotated video's red
"TARGET" box can be drawn at that instant. Reuses `src.team.classifier`'s existing torso-crop +
CIELAB machinery VERBATIM (Golden Rule: no new model, no schema change) -- this module only adds
the one step ADR-12's own per-take colour clustering never needed: matching a colour WORD (the
client's own "in white"/"in red"/...) to a measured Lab value, rather than comparing two tracks to
each other.

A track close enough in time to the annotation's own `t` is a location candidate; the candidate
whose own torso Lab colour sits closest to `team_colour`'s reference Lab
(`configs/annotations.yaml: colour_reference_lab`) is picked, PROVIDED that distance clears
`colour_match_max_distance` -- otherwise no track is confidently associated (`None`), and the
caller (Stage 5's renderer) shows the event as a CAPTION only, never a fabricated box (Golden
Rule 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.types import Annotation, Take, Track
from src.common.video import decode_frames
from src.team.classifier import per_track_lab_median, torso_region

__all__ = [
    "associate_annotation_to_track",
    "candidate_tracks_near_time",
    "nearest_track_by_colour",
]


def candidate_tracks_near_time(tracks: list[Track], t: float, tolerance_s: float) -> list[Track]:
    """Every track with at least one box within `tolerance_s` seconds of `t` -- the LOCATION
    candidate pool for one annotation instant (pure, no I/O, directly unit-testable)."""
    return [tr for tr in tracks if any(abs(b.t - t) <= tolerance_s for b in tr.boxes)]


def nearest_track_by_colour(
    lab_by_track_id: dict[int, np.ndarray],
    team_colour: str,
    colour_cfg: dict,
) -> tuple[int | None, float | None]:
    """Pure colour-matching step: which (if any) of `lab_by_track_id`'s tracks has a torso Lab
    colour closest to `team_colour`'s own reference Lab (`colour_cfg['colour_reference_lab']`),
    within `colour_cfg['colour_match_max_distance']`.

    `(None, None)` when `team_colour` isn't a known reference colour, or there are no candidates
    with a usable Lab measurement at all. `(None, best_distance)` when a nearest candidate DOES
    exist but sits further than `colour_match_max_distance` -- kept distinct from the "no data at
    all" case so a caller can log HOW far off the nearest candidate actually was (Golden Rule 5:
    never a forced match, but never a silently unexplained one either).
    """
    reference = colour_cfg["colour_reference_lab"].get(team_colour.lower())
    if reference is None or not lab_by_track_id:
        return None, None
    ref = np.array(reference, dtype=np.float64)

    best_id: int | None = None
    best_dist: float | None = None
    for tid, lab in lab_by_track_id.items():
        dist = float(np.linalg.norm(lab - ref))
        if best_dist is None or dist < best_dist:
            best_dist, best_id = dist, tid

    if best_dist is None or best_dist > colour_cfg["colour_match_max_distance"]:
        return None, best_dist
    return best_id, best_dist


def associate_annotation_to_track(
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    annotation: Annotation,
    colour_cfg: dict,
    decode_cfg: dict,
    sampling_cfg: dict,
    use_nvdec: bool = True,
) -> tuple[int | None, float | None]:
    """Full pipeline for ONE annotation: candidate tracks near `annotation.t` -> a SHORT native
    decode window around that instant -> per-candidate torso-crop Lab (reusing
    `src.team.classifier`'s existing crop/median helpers verbatim) -> `nearest_track_by_colour`.

    Decodes a short window (`sampling_cfg['window_s']` either side of `annotation.t`, clamped into
    the take) rather than the take's own full span -- only THIS instant's identity matters here,
    unlike `src.team.classifier.collect_track_crops`'s take-wide sampling (built to answer "this
    track's colour ALL TAKE" for team clustering, a different question).

    Returns `(None, None)` immediately, no decode performed, when there is no candidate track near
    `annotation.t` at all -- cheap, and honest (Golden Rule 5: nothing to associate to).
    """
    tolerance_s = sampling_cfg["match_tolerance_s"]
    candidates = candidate_tracks_near_time(take_tracks, annotation.t, tolerance_s)
    if not candidates:
        return None, None

    window_s = sampling_cfg["window_s"]
    start = max(take.t_start, annotation.t - window_s)
    end = min(take.t_end, annotation.t + window_s)
    top_frac = sampling_cfg["torso_top_frac"]
    bottom_frac = sampling_cfg["torso_bottom_frac"]
    x_inset_frac = sampling_cfg["torso_x_inset_frac"]
    min_box_height_px = sampling_cfg["min_box_height_px"]

    crops_by_track: dict[int, list[np.ndarray]] = {tr.id: [] for tr in candidates}
    for _idx, t, frame in decode_frames(
        video_path,
        fps=sampling_cfg["fps_sample"],
        start=start,
        end=end,
        scale_width=decode_cfg.get("scale_width"),
        use_nvdec=use_nvdec,
    ):
        height, width = frame.shape[:2]
        for tr in candidates:
            box = next((b for b in tr.boxes if abs(b.t - t) <= tolerance_s), None)
            if box is None or box.bbox.height < min_box_height_px:
                continue
            torso = torso_region(box.bbox, top_frac, bottom_frac, x_inset_frac)
            x1, y1, x2, y2 = (int(round(v)) for v in torso.to_xyxy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:  # degenerate after clipping to frame bounds
                continue
            crops_by_track[tr.id].append(frame[y1:y2, x1:x2].copy())

    lab_by_track_id: dict[int, np.ndarray] = {}
    for tid, crops in crops_by_track.items():
        lab = per_track_lab_median(crops)
        if lab is not None:
            lab_by_track_id[tid] = lab

    return nearest_track_by_colour(lab_by_track_id, annotation.team_colour, colour_cfg)
