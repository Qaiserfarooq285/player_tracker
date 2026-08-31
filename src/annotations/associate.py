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

**Owner request, 2026-08-31** (a burned-in red arrow already marks the target player in several
takes of `input/video1/Jordan Thomas Highlight Video.mp4` -- CLAUDE.md §3.2 consequence 1/§3.3):
the arrow's own tip location is an ADDITIONAL, CORROBORATING signal, reused generically here for
every manual-mode video, not gated on any filename. `nearest_track_by_arrow_hint` applies the
EXACT SAME "tip contained in / nearest to a track's own box, within a measured pixel threshold"
test Stage 5's own `vote_seed_tracks` (`src.highlights.selection`) already trusts for auto seed
selection -- scoped to one annotation instant rather than voted across a whole take. Colour stays
the PRIMARY test (`associate_annotation_to_track`'s own docstring); the arrow signal is combined,
never substituted: it corroborates an agreeing colour pick, is logged (not silently dropped, Golden
Rule 5) when it disagrees with one, and is used as an honest fallback ONLY when colour itself
found no confident match at all. Neither signal confident -> still `(None, ...)`, caption-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.types import Annotation, Take, Track
from src.common.video import decode_frames
from src.detect.overlay_mask import ArrowHint
from src.highlights.selection import _center_distance, _contains, _nearest_box
from src.team.classifier import per_track_lab_median, torso_region

__all__ = [
    "associate_annotation_to_track",
    "candidate_tracks_near_time",
    "nearest_track_by_arrow_hint",
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


def nearest_track_by_arrow_hint(
    arrow_hints: list[ArrowHint],
    candidates: list[Track],
    t: float,
    hint_window_s: float,
    match_tolerance_s: float,
    max_dist_px: float,
) -> tuple[int | None, float | None]:
    """Best-effort ADDITIONAL corroborating signal for manual-mode association (owner request,
    2026-08-31): reuses the EXACT SAME arrow-tip -> track geometry test Stage 5's own
    `vote_seed_tracks` (`src.highlights.selection`) already applies for auto seed selection --
    scoped to ONE annotation instant rather than voted across a whole take, since here we only
    need "does a candidate track sit under the arrow's tip right now", not "which track does the
    arrow mostly point at all take".

    Picks the arrow hint nearest in time to `t` (must be within `hint_window_s`, the same
    "only this instant matters" window `associate_annotation_to_track` already decodes for colour
    -- `configs/annotations.yaml: track_association.window_s`); among `candidates`, returns
    whichever track's own box (nearest in time to that hint, within `match_tolerance_s`) either
    CONTAINS the tip, or -- if none does -- is nearest to it by centroid distance, within
    `max_dist_px` (both thresholds `configs/highlights.yaml: selection.arrow_match_*`, the SAME
    measured values Stage 5 already trusts for this exact judgement).

    `(None, None)` when there is no arrow hint within `hint_window_s` of `t` at all, no candidate
    has a usable box near that hint's own timestamp, or the nearest candidate sits further than
    `max_dist_px` -- an absent/weak arrow signal is exactly as honest a "no answer" as an absent
    colour match (Golden Rule 5), never a forced guess.
    """
    if not arrow_hints or not candidates:
        return None, None

    hint = min(arrow_hints, key=lambda h: abs(h.t - t))
    if abs(hint.t - t) > hint_window_s:
        return None, None

    containing: int | None = None
    nearest_id: int | None = None
    nearest_dist: float | None = None
    for tr in candidates:
        box = _nearest_box(tr, hint.t, match_tolerance_s)
        if box is None:
            continue
        if containing is None and _contains(box.bbox, hint.tip_x, hint.tip_y):
            containing = tr.id
        dist = _center_distance(box.bbox, hint.tip_x, hint.tip_y)
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist, nearest_id = dist, tr.id

    if containing is not None:
        return containing, 0.0
    if nearest_id is not None and nearest_dist is not None and nearest_dist <= max_dist_px:
        return nearest_id, nearest_dist
    return None, None


def associate_annotation_to_track(
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    annotation: Annotation,
    colour_cfg: dict,
    decode_cfg: dict,
    sampling_cfg: dict,
    use_nvdec: bool = True,
    arrow_hints: list[ArrowHint] | None = None,
    selection_cfg: dict | None = None,
) -> tuple[int | None, float | None, dict]:
    """Full pipeline for ONE annotation: candidate tracks near `annotation.t` -> a SHORT native
    decode window around that instant -> per-candidate torso-crop Lab (reusing
    `src.team.classifier`'s existing crop/median helpers verbatim) -> `nearest_track_by_colour`,
    COMBINED with the burned-in-arrow corroborating signal (`nearest_track_by_arrow_hint`) when
    `arrow_hints`/`selection_cfg` are supplied (owner request, 2026-08-31; generic to every
    manual-mode video, not gated on a filename).

    Decodes a short window (`sampling_cfg['window_s']` either side of `annotation.t`, clamped into
    the take) rather than the take's own full span -- only THIS instant's identity matters here,
    unlike `src.team.classifier.collect_track_crops`'s take-wide sampling (built to answer "this
    track's colour ALL TAKE" for team clustering, a different question).

    Combination rule (colour stays PRIMARY, arrow is corroborating, never substituting -- see
    module docstring): colour's own pick wins whenever it has one, whether or not the arrow agrees
    (a disagreement is logged in the returned `evidence` dict, never silently dropped, Golden
    Rule 5); the arrow's pick is used as a fallback ONLY when colour itself found nothing
    confident. Neither confident -> `(None, None, evidence)` -- caption-only, no fabricated box.

    Returns `(track_id, distance, evidence)`. `evidence` always carries
    `{"arrow_track_id", "arrow_distance_px", "agreement"}` for `annotation_report.json`'s own
    traceability (Golden Rule 5) -- `agreement` is one of `"no_candidates"`, `"colour_only"`,
    `"arrow_fallback"`, `"corroborated"`, `"disagreement"`, or `"no_confident_signal"`.
    `(None, None, {"arrow_track_id": None, "arrow_distance_px": None, "agreement":
    "no_candidates"})` immediately, no decode performed, when there is no candidate track near
    `annotation.t` at all -- cheap, and honest (nothing to associate to, colour or arrow).
    """
    tolerance_s = sampling_cfg["match_tolerance_s"]
    candidates = candidate_tracks_near_time(take_tracks, annotation.t, tolerance_s)
    if not candidates:
        return (
            None,
            None,
            {"arrow_track_id": None, "arrow_distance_px": None, "agreement": "no_candidates"},
        )

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

    colour_track_id, colour_distance = nearest_track_by_colour(
        lab_by_track_id, annotation.team_colour, colour_cfg
    )

    arrow_track_id: int | None = None
    arrow_distance: float | None = None
    if arrow_hints and selection_cfg:
        arrow_track_id, arrow_distance = nearest_track_by_arrow_hint(
            arrow_hints,
            candidates,
            annotation.t,
            hint_window_s=window_s,
            match_tolerance_s=selection_cfg["arrow_match_tolerance_s"],
            max_dist_px=selection_cfg["arrow_match_max_dist_px"],
        )

    if colour_track_id is not None and arrow_track_id is not None:
        agreement = "corroborated" if colour_track_id == arrow_track_id else "disagreement"
        final_id, final_distance = colour_track_id, colour_distance
    elif colour_track_id is not None:
        agreement = "colour_only"
        final_id, final_distance = colour_track_id, colour_distance
    elif arrow_track_id is not None:
        agreement = "arrow_fallback"
        final_id, final_distance = arrow_track_id, arrow_distance
    else:
        agreement = "no_confident_signal"
        final_id, final_distance = None, None

    evidence = {
        "arrow_track_id": arrow_track_id,
        "arrow_distance_px": arrow_distance,
        "agreement": agreement,
    }
    return final_id, final_distance, evidence
