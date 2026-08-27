"""Stage 5 — target-player selection (CLAUDE.md §5 Stage 5, Golden Rule 4/7).

Phase 1's rule is that a HUMAN picks the track. This module implements **auto-selection with the
human seam fully intact** so an overnight/unattended run can still complete end to end:

1. **Arrow vote** (`vote_seed_tracks`): for every hinted frame in `arrow_hints.parquet`
   (`src/detect/overlay_mask.py`), find the nearest/containing track to the arrow's TIP point and
   tally votes per take. This never reads a jersey number (Golden Rule 7) — geometry only.
2. **Within-take stitching** (`stitch_timeline`): greedily extend the seed track by joining other
   SAME-take fragments that are close in time and space (see `configs/highlights.yaml:
   selection`). This is inference on top of the raw tracker output, so the result gets its own,
   penalised `id_confidence` and records exactly which fragment ids were joined.
3. **No-arrow fallback** (`heuristic_fallback_seed`): for a take with no arrow evidence (most of
   `clip4` after the arrow disappears; all of `clip1`/`clip3`/`clip5`), pick a seed by a documented
   heuristic and mark it `heuristic_fallback` with a low confidence ceiling — never silently
   blended with an arrow-anchored pick.
4. **`SelectionResult`** is written to `work/<slug>/selection.json` with an explicit
   `needs_human_confirmation` flag that is `True` unless every take was human-overridden.
5. `--track-id` (see `src/pipeline/run.py`'s CLI) is the actual Phase-1 human-in-the-loop seam: a
   human-supplied override always wins over both the arrow vote and the fallback heuristic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from src.common.logging import get_logger
from src.common.types import BBox, DetectionClass, Take, Track, TrackBox
from src.detect.overlay_mask import ArrowHint
from src.track.continuity import stitch_timeline as stitch_timeline  # re-export, see below
from src.track.tracker import assign_take_id

logger = get_logger(__name__)

SelectionMethod = Literal["arrow_vote", "heuristic_fallback", "manual_override", "none"]


class TakeSelection(BaseModel):
    """The selected target timeline for ONE take (a `Track`/timeline never crosses a take,
    Golden Rule 3)."""

    take_id: int
    method: SelectionMethod
    seed_track_id: int | None
    track_ids: list[int]  # every fragment track id stitched into this take's timeline, in the
    # order they were joined (seed first) — empty when method == "none" (no usable track at all).
    confidence: float
    coverage_seconds: float  # total seconds of `[t_start, t_end]` union actually covered by the
    # stitched timeline's own boxes
    take_duration_seconds: float
    vote_share: float | None = None  # arrow_vote only: seed's votes / total votes cast this take

    # ADR-15 (CLAUDE.md §3.3/§5.1): populated ONLY for a filename-less video (`target_jersey is
    # None`) by `src.identity.verify.verify_identities_for_video`, called from
    # `src.pipeline.run` AFTER this take's own (location-only) selection above is already built —
    # never for the original 5 `clip<N> <jersey>.mp4` clips, whose `identity_status` stays `None`
    # forever (a real gate, not just an unused default: CLAUDE.md §13.1 renders NO red box for a
    # take unless `identity_status == "verified"`). `jersey_number` here is a VERIFIED number read
    # from visible evidence with temporal agreement across >= 2 independent frames — never the
    # same thing as `seed_track_id`/`method` above, which are a pure LOCATION prior that never
    # reads a digit (Golden Rule 7 / ADR-15).
    jersey_number: int | None = None
    identity_status: Literal["verified", "unverified"] | None = None
    identity_confidence: float = 0.0
    identity_evidence_frames: list[int] = Field(default_factory=list)


class SelectionResult(BaseModel):
    """Stage 5's output artifact (`work/<slug>/selection.json`). Never claims human confirmation
    unless every take was explicitly `manual_override`-ed (Golden Rule 4)."""

    video: str
    target_jersey: int | None
    overall_confidence: float
    needs_human_confirmation: bool
    takes: list[TakeSelection]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _nearest_box(track: Track, t: float, tolerance_s: float) -> TrackBox | None:
    """The track's own box closest in time to `t`, if within `tolerance_s` seconds."""
    best: TrackBox | None = None
    best_dt = tolerance_s
    for box in track.boxes:
        dt = abs(box.t - t)
        if dt <= best_dt:
            best, best_dt = box, dt
    return best


def _contains(bbox: BBox, x: float, y: float) -> bool:
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def _center_distance(bbox: BBox, x: float, y: float) -> float:
    return ((bbox.cx - x) ** 2 + (bbox.cy - y) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# 1. arrow vote
# ---------------------------------------------------------------------------


def vote_seed_tracks(
    arrow_hints: list[ArrowHint],
    tracks: list[Track],
    takes: list[Take],
    selection_cfg: dict,
) -> dict[int, dict[int, int]]:
    """Vote for a seed track per take from the arrow's tip point.

    For each hinted frame: resolve its take (`assign_take_id`), then among that take's own tracks
    find the one whose box (nearest in time, within `arrow_match_tolerance_s`) either CONTAINS the
    tip, or — if none does — is nearest to it by centroid distance, within `arrow_match_max_dist_px`
    (MEASURED on `clip2`: the tip sits 51-129px from its actual target's centroid, since it marks
    the head/shoulder area, not the box centre — see `configs/highlights.yaml` for the numbers).
    Each qualifying hinted frame casts exactly one vote.

    Returns `{take_id: {track_id: n_votes}}` — a take with no arrow evidence at all is simply
    absent from the result (its caller falls back to `heuristic_fallback_seed`).
    """
    tolerance_s = selection_cfg["arrow_match_tolerance_s"]
    max_dist = selection_cfg["arrow_match_max_dist_px"]

    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for hint in arrow_hints:
        take_id = assign_take_id(hint.t, takes)
        if take_id is None:
            continue
        candidates = tracks_by_take.get(take_id, [])
        if not candidates:
            continue

        containing: int | None = None
        nearest_id: int | None = None
        nearest_dist: float | None = None
        for tr in candidates:
            box = _nearest_box(tr, hint.t, tolerance_s)
            if box is None:
                continue
            if containing is None and _contains(box.bbox, hint.tip_x, hint.tip_y):
                containing = tr.id
            dist = _center_distance(box.bbox, hint.tip_x, hint.tip_y)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist, nearest_id = dist, tr.id

        chosen = containing
        if chosen is None and nearest_id is not None and nearest_dist is not None:
            if nearest_dist <= max_dist:
                chosen = nearest_id
        if chosen is not None:
            votes[take_id][chosen] += 1

    return {tid: dict(v) for tid, v in votes.items()}


# ---------------------------------------------------------------------------
# 2. within-take stitching
# ---------------------------------------------------------------------------

# `stitch_timeline` (gap/distance greedy forward-extension + its `_spatial_jump`/`_team_agrees`
# helpers) moved to `src/track/continuity.py` (ADR-13/14): the best-effort event heuristics
# (`src/events/{touches,possession,tackles}.py`) need the SAME "same person across a brief
# occlusion" judgement for EVERY player in a take, not just this module's one arrow/fallback seed,
# so the core greedy-chaining logic is shared rather than duplicated. Re-imported above as
# `stitch_timeline` — behaviour-preserving, this module's own public API/tests are unchanged.


def timeline_coverage_seconds(track_ids: list[int], take_tracks: list[Track]) -> float:
    """Total seconds spanned by the union of the stitched tracks' own box timestamps — a coarse
    but honest "how much of the take does this timeline actually have evidence for" measure
    (adjacent/overlapping fragments' spans are merged, not double-counted)."""
    by_id = {tr.id: tr for tr in take_tracks}
    spans = sorted(
        (by_id[tid].boxes[0].t, by_id[tid].boxes[-1].t)
        for tid in track_ids
        if tid in by_id and by_id[tid].boxes
    )
    if not spans:
        return 0.0
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


# ---------------------------------------------------------------------------
# 3. no-arrow fallback
# ---------------------------------------------------------------------------


def _minmax_normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-9:
        return [1.0 for _ in values]  # all tied -- treat as equally maximal, not zeroed out
    return [(v - lo) / (hi - lo) for v in values]


def heuristic_fallback_seed(
    take_tracks: list[Track], frame_width: float, frame_height: float, selection_cfg: dict
) -> int | None:
    """No-arrow fallback (CLAUDE.md task spec): the longest-lived track with the highest mean
    detection confidence nearest the frame centre — an outfield-player track only (referees are
    excluded; goalkeepers are kept since the target player could plausibly be either).

    Each of the three raw signals (duration, mean confidence, inverse distance-from-centre) is
    min-max normalised across THIS TAKE's own candidate tracks before weighting
    (`configs/highlights.yaml: selection.heuristic.*_weight`), so the weights stay comparable
    regardless of each signal's raw scale. Returns `None` when no candidate track clears
    `min_track_duration_s` at all (an empty/degenerate take).
    """
    cfg = selection_cfg["heuristic"]
    candidates = [
        tr
        for tr in take_tracks
        if tr.dominant_class != DetectionClass.REFEREE
        and tr.boxes
        and tr.duration >= cfg["min_track_duration_s"]
    ]
    if not candidates:
        return None

    cx, cy = frame_width / 2.0, frame_height / 2.0
    durations = [tr.duration for tr in candidates]
    mean_confs = [sum(b.conf for b in tr.boxes) / len(tr.boxes) for tr in candidates]
    center_dists = [min(_center_distance(b.bbox, cx, cy) for b in tr.boxes) for tr in candidates]
    inv_center = [-d for d in center_dists]  # closer to centre -> larger normalised score

    norm_dur = _minmax_normalize(durations)
    norm_conf = _minmax_normalize(mean_confs)
    norm_center = _minmax_normalize(inv_center)

    scores = [
        cfg["duration_weight"] * nd + cfg["confidence_weight"] * nc + cfg["center_weight"] * nctr
        for nd, nc, nctr in zip(norm_dur, norm_conf, norm_center, strict=True)
    ]
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx].id


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------


def select_targets(
    video_path: str,
    target_jersey: int | None,
    takes: list[Take],
    tracks: list[Track],
    arrow_hints: list[ArrowHint],
    frame_width: float,
    frame_height: float,
    selection_cfg: dict,
    manual_overrides: dict[int, int] | None = None,
) -> SelectionResult:
    """Run Stage 5 target selection for every take of one video.

    `manual_overrides` is the actual Phase-1 human-in-the-loop seam (`{take_id: track_id}`, from
    `src/pipeline/run.py`'s `--track-id` CLI flag) — a human-supplied track always wins over both
    the arrow vote and the fallback heuristic for that take, and is the ONLY case whose
    `TakeSelection.method` is `"manual_override"`.
    """
    manual_overrides = manual_overrides or {}
    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    votes_by_take = vote_seed_tracks(arrow_hints, tracks, takes, selection_cfg)

    take_selections: list[TakeSelection] = []
    for take in takes:
        take_tracks = tracks_by_take.get(take.id, [])
        take_duration = take.t_end - take.t_start

        if take.id in manual_overrides:
            seed = manual_overrides[take.id]
            if not any(tr.id == seed for tr in take_tracks):
                logger.warning(
                    "manual override track_id=%d for take=%d not found among that take's tracks "
                    "-- ignoring override, falling back to automatic selection",
                    seed,
                    take.id,
                )
            else:
                track_ids, _stitch_conf = stitch_timeline(seed, take_tracks, selection_cfg)
                confidence = selection_cfg["manual_override_confidence"]
                take_selections.append(
                    TakeSelection(
                        take_id=take.id,
                        method="manual_override",
                        seed_track_id=seed,
                        track_ids=track_ids,
                        confidence=confidence,
                        coverage_seconds=timeline_coverage_seconds(track_ids, take_tracks),
                        take_duration_seconds=take_duration,
                    )
                )
                continue

        take_votes = votes_by_take.get(take.id, {})
        if take_votes:
            total_votes = sum(take_votes.values())
            seed_candidate = max(take_votes, key=lambda tid: take_votes[tid])
            vote_share = take_votes[seed_candidate] / total_votes
            min_vote_share = selection_cfg["arrow_min_vote_share"]
            if vote_share >= min_vote_share:
                track_ids, stitch_conf = stitch_timeline(seed_candidate, take_tracks, selection_cfg)
                confidence = vote_share * stitch_conf
                take_selections.append(
                    TakeSelection(
                        take_id=take.id,
                        method="arrow_vote",
                        seed_track_id=seed_candidate,
                        track_ids=track_ids,
                        confidence=confidence,
                        coverage_seconds=timeline_coverage_seconds(track_ids, take_tracks),
                        take_duration_seconds=take_duration,
                        vote_share=vote_share,
                    )
                )
                continue
            # MEASURED on clip4's take 0 (this task's own Stage 5 build): 148 "arrow" hints
            # scattered across many different tracks with NO dominant winner (top share ~0.20) --
            # almost certainly a recurring false-positive red object (CLAUDE.md §3.2(1) already
            # documents clip4 needed arrow_max_area_px/arrow_min_height_width_ratio guards against
            # a running-track surface and a red-brick house; evidently not every false positive is
            # caught), NOT a genuine single target-marking arrow. A plurality this weak must not be
            # trusted as an anchor -- fall through to the heuristic fallback instead of silently
            # reporting a falsely-confident `arrow_vote` pick.
            logger.warning(
                "take=%d: arrow votes too scattered to trust (top candidate share=%.2f < "
                "arrow_min_vote_share=%.2f over %d total vote(s) across %d candidate track(s)) "
                "-- treating as NO reliable arrow evidence, falling back to the heuristic seed",
                take.id,
                vote_share,
                min_vote_share,
                total_votes,
                len(take_votes),
            )

        seed = heuristic_fallback_seed(take_tracks, frame_width, frame_height, selection_cfg)
        if seed is None:
            logger.warning("take=%d: no usable track for target selection at all", take.id)
            take_selections.append(
                TakeSelection(
                    take_id=take.id,
                    method="none",
                    seed_track_id=None,
                    track_ids=[],
                    confidence=0.0,
                    coverage_seconds=0.0,
                    take_duration_seconds=take_duration,
                )
            )
            continue

        track_ids, stitch_conf = stitch_timeline(seed, take_tracks, selection_cfg)
        ceiling = selection_cfg["heuristic_fallback_confidence_ceiling"]
        confidence = min(ceiling, stitch_conf)
        take_selections.append(
            TakeSelection(
                take_id=take.id,
                method="heuristic_fallback",
                seed_track_id=seed,
                track_ids=track_ids,
                confidence=confidence,
                coverage_seconds=timeline_coverage_seconds(track_ids, take_tracks),
                take_duration_seconds=take_duration,
            )
        )

    usable = [t for t in take_selections if t.method != "none"]
    overall_confidence = sum(t.confidence for t in usable) / len(usable) if usable else 0.0
    needs_confirmation = any(t.method != "manual_override" for t in take_selections)

    return SelectionResult(
        video=str(video_path),
        target_jersey=target_jersey,
        overall_confidence=overall_confidence,
        needs_human_confirmation=needs_confirmation,
        takes=take_selections,
    )
