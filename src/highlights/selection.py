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
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.common.logging import get_logger
from src.common.types import BBox, DetectionClass, Take, Track, TrackBox
from src.detect.overlay_mask import ArrowHint
from src.track.continuity import stitch_timeline as stitch_timeline  # re-export, see below
from src.track.tracker import assign_take_id

# `src.track.target`/`src.track.target_verify` are imported LAZILY (inside the functions that
# actually call into them below), never at module level: `src.track.target` -> `src.track.
# click_reid` -> `src.identity.verify` -> `src.highlights.selection` (this module, for its own
# `select_targets` re-export) is a real circular import at module-load time. `from __future__
# import annotations` (above) already defers every type annotation in this file to a string, so
# the type hints below (`TargetProfile`, `KitColourSample`, `TargetVerdict`, ...) never need a
# runtime import at all -- only the few call sites that actually construct/compare these types
# (`_update_profile_after_take`, `_select_take_with_profile`) import them locally.
if TYPE_CHECKING:
    from src.track.target import KitColourSample, TargetProfile
    from src.track.target_verify import TargetVerdict

logger = get_logger(__name__)

# "streamed-gathering-treehouse" plan Stage 4: `target_reidentified` (a non-override take resolved
# by verifying the persistent TargetProfile against its own candidate tracks) and `target_lost`
# (no candidate -- including a manual override that itself failed verification -- ever cleared
# ACCEPT) join the pre-existing four. Both are ONLY ever produced when a `TargetProfile` is
# supplied to `select_targets` (see `_select_take_with_profile` below); the no-profile path below
# never constructs either literal, so every pre-existing caller/test is unaffected.
SelectionMethod = Literal[
    "arrow_vote",
    "heuristic_fallback",
    "manual_override",
    "none",
    "target_reidentified",
    "target_lost",
]


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

    # "streamed-gathering-treehouse" plan Stage 4: the full `TargetVerdict.evidence` trail behind
    # a `target_reidentified`/`target_lost` decision (or a profile-verified `manual_override`) --
    # Golden Rule 5, every accept/reject traceable. Left at its empty-dict default for every
    # pre-existing method (arrow_vote/heuristic_fallback/none, and a profile-less
    # manual_override) -- the no-profile selection LOGIC is unaffected; only the serialized
    # `selection.json` gains one new, harmless `"evidence": {}` key per take.
    evidence: dict = Field(default_factory=dict)

    # Stage B of the "streamed-gathering-treehouse" plan (per-frame target identity gate,
    # `src/track/target_state.py`): the SUBSET of `track_ids` that may ever render a red/amber box
    # or be counted into the target's own stats -- "Candidate != Target". Populated ONLY by
    # `_select_take_with_profile` below (i.e. only when a `TargetProfile` drives selection):
    # `[override_track_id]`/`[winner_id]` for an ACCEPTed `manual_override`/`target_reidentified`
    # (the click itself, or whichever candidate(s) independently passed `verify_candidate` -- see
    # that function's own comment for why ALL of `accepts`, not just the winner, are kept), `[]`
    # for `target_lost`. Left at its empty-list default for every pre-existing profile-less method
    # (arrow_vote/heuristic_fallback/none, and a profile-less manual_override) -- those methods
    # have no separate "verified vs merely-geometry-stitched" distinction at all, so
    # `src.pipeline.run._accepted_target_ids` falls back to the full `track_ids` for them,
    # preserving their exact pre-existing behaviour (CLAUDE.md §14's auto flow, ADR-15, ADR-19).
    verified_track_ids: list[int] = Field(default_factory=list)


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
# 3b. target-profile-driven per-take resolution ("streamed-gathering-treehouse" plan Stage 4) --
# used ONLY by `select_targets` when it is given a `TargetProfile`; the arrow-vote/heuristic
# fallback code above is a completely separate branch this never calls into.
# ---------------------------------------------------------------------------


def _reasonable_candidates(take_tracks: list[Track], selection_cfg: dict) -> list[Track]:
    """Which of this take's own tracks are even worth verifying against the target profile --
    the SAME outfield-player-only, minimum-duration filter `heuristic_fallback_seed` already uses
    for its own candidate pool (`configs/highlights.yaml: selection.heuristic.
    min_track_duration_s`), reused rather than inventing a second bar: a fragment too short to
    trust for the no-profile heuristic is equally not worth a full `verify_candidate` call for the
    profile-driven path."""
    min_duration = selection_cfg["heuristic"]["min_track_duration_s"]
    return [
        tr
        for tr in take_tracks
        if tr.dominant_class != DetectionClass.REFEREE and tr.boxes and tr.duration >= min_duration
    ]


def _lost_selection(take: Take, take_duration: float, evidence: dict) -> TakeSelection:
    """A take where no candidate ever cleared `verify_candidate`'s ACCEPT bar -- `track_ids=[]`,
    `confidence=0.0`, NEVER a substitute player (plan Stage 4's own central rule)."""
    return TakeSelection(
        take_id=take.id,
        method="target_lost",
        seed_track_id=None,
        track_ids=[],
        confidence=0.0,
        coverage_seconds=0.0,
        take_duration_seconds=take_duration,
        evidence=evidence,
    )


def _last_position_evidence(track_ids: list[int], take_tracks: list[Track]) -> dict:
    """`{last_cx, last_cy, last_bbox_height}` from the LAST box (by time) across `track_ids`' own
    stitched fragments -- populates the three keys `src.track.target_verify.verify_candidate`'s
    own `_trajectory_score` reads out of `profile.links[-1].evidence` (its docstring names this
    exact population step as deferred to "a later, out-of-scope pipeline stage": this module).
    Empty dict when `track_ids` resolves to no boxes at all (nothing to record)."""
    by_id = {tr.id: tr for tr in take_tracks}
    all_boxes = [b for tid in track_ids if (tr := by_id.get(tid)) is not None for b in tr.boxes]
    if not all_boxes:
        return {}
    last = max(all_boxes, key=lambda b: b.t)
    return {
        "last_cx": last.bbox.cx,
        "last_cy": last.bbox.cy,
        "last_bbox_height": last.bbox.height,
    }


def _update_profile_after_take(
    profile: TargetProfile,
    take: Take,
    selection: TakeSelection,
    take_tracks: list[Track],
    kit_by_track: dict[int, KitColourSample],
    target_cfg: dict,
) -> None:
    """Append this take's own resolution to `profile.links` (mutating `profile` IN PLACE -- see
    `select_targets`'s own docstring for why this is a deliberate side effect, not an oversight),
    and -- only for an accepted link at/above `target_cfg['memory_update_min_confidence']` -- fold
    its winning track's own kit-colour sample into the memory bank via `update_memory_bank`
    (Stage 1's anti-drift rule: an unaccepted/`target_lost` link never reaches this call at all).
    """
    from src.track.target import TargetLink, TargetState, update_memory_bank  # see module-level

    # import-order comment: `src.track.target` cannot be imported at module scope here.

    if selection.method == "target_lost":
        state = TargetState.SEARCHING
        evidence = dict(selection.evidence)
    else:
        state = TargetState.CONFIRMED
        evidence = {
            **selection.evidence,
            **_last_position_evidence(selection.track_ids, take_tracks),
        }

    profile.links.append(
        TargetLink(
            take_id=take.id,
            track_ids=list(selection.track_ids),
            state=state,
            confidence=selection.confidence,
            evidence=evidence,
            t_start=take.t_start,
            t_end=take.t_end,
        )
    )

    if selection.method == "target_lost" or selection.seed_track_id is None:
        return
    sample = kit_by_track.get(selection.seed_track_id)
    if sample is None:
        return
    updated = update_memory_bank(profile, sample, selection.confidence, target_cfg)
    profile.kit_bank = updated.kit_bank
    profile.kit = updated.kit


def _select_take_with_profile(
    take: Take,
    take_tracks: list[Track],
    take_duration: float,
    profile: TargetProfile,
    target_cfg: dict,
    kit_by_track: dict[int, KitColourSample],
    jersey_by_track: dict[int, str],
    override_track_id: int | None,
    selection_cfg: dict,
    motion,
) -> TakeSelection:
    """Resolve ONE take against the persistent `TargetProfile` -- `heuristic_fallback_seed` and
    the arrow vote are never called from anywhere in this function. Mutates `profile` in place via
    `_update_profile_after_take` before returning (see that function + `select_targets`'s own
    docstring)."""
    from src.track.target_verify import VerdictDecision, verify_candidate  # see module-level

    # import-order comment: `src.track.target_verify` cannot be imported at module scope here.

    by_id = {tr.id: tr for tr in take_tracks}

    if override_track_id is not None:
        candidate = by_id.get(override_track_id)
        if candidate is None:
            # Plan Stage 4's own fix for the pre-existing bug at (what was) selection.py:296-303:
            # a manual override whose track isn't even IN this take's tracks used to silently fall
            # through to automatic (heuristic/arrow) selection. With a profile in play that must
            # become target_lost instead -- never a substitute player.
            selection = _lost_selection(
                take,
                take_duration,
                {
                    "reason": "manual_override_track_not_found",
                    "seed_track_id": override_track_id,
                },
            )
        else:
            verdict = verify_candidate(
                profile, candidate, kit_by_track, jersey_by_track, target_cfg
            )
            if verdict.decision == VerdictDecision.ACCEPT:
                track_ids, _stitch_conf = stitch_timeline(
                    override_track_id, take_tracks, selection_cfg, motion
                )
                selection = TakeSelection(
                    take_id=take.id,
                    method="manual_override",
                    seed_track_id=override_track_id,
                    track_ids=track_ids,
                    confidence=verdict.score,
                    coverage_seconds=timeline_coverage_seconds(track_ids, take_tracks),
                    take_duration_seconds=take_duration,
                    evidence=verdict.evidence,
                    # Stage B: the click itself is the ONLY accepted fragment -- everything else
                    # `stitch_timeline` joined above is geometry-only, never independently
                    # verified, so it stays a candidate (green box), never red (plan's "Candidate
                    # != Target").
                    verified_track_ids=[override_track_id],
                )
            else:
                # Stage 2's own rule: UNCERTAIN is never a weak accept, treated exactly like
                # REJECT here -- the target stays LOST, never assigned to a contradicted click.
                selection = _lost_selection(take, take_duration, verdict.evidence)
    else:
        candidates = _reasonable_candidates(take_tracks, selection_cfg)
        accepts: list[tuple[int, TargetVerdict]] = []
        all_evidence: dict[int, dict] = {}
        for tr in candidates:
            verdict = verify_candidate(profile, tr, kit_by_track, jersey_by_track, target_cfg)
            all_evidence[tr.id] = verdict.evidence
            if verdict.decision == VerdictDecision.ACCEPT:
                accepts.append((tr.id, verdict))

        if not accepts:
            selection = _lost_selection(
                take,
                take_duration,
                {
                    "reason": "no_accept",
                    "candidates_considered": [tr.id for tr in candidates],
                    "verdicts": all_evidence,
                },
            )
        else:
            max_score = max(v.score for _, v in accepts)
            winners = [(tid, v) for tid, v in accepts if v.score == max_score]
            if len(winners) > 1:
                # Plan Stage 4's own rule: "Ties/no ACCEPT -> LOST" -- an ambiguous best match is
                # not evidence for either candidate, so neither is trusted over the other.
                selection = _lost_selection(
                    take,
                    take_duration,
                    {
                        "reason": "tie",
                        "tied_track_ids": [tid for tid, _ in winners],
                        "verdicts": all_evidence,
                    },
                )
            else:
                winner_id, verdict = winners[0]
                track_ids, _stitch_conf = stitch_timeline(
                    winner_id, take_tracks, selection_cfg, motion
                )
                selection = TakeSelection(
                    take_id=take.id,
                    method="target_reidentified",
                    seed_track_id=winner_id,
                    track_ids=track_ids,
                    confidence=verdict.score,
                    coverage_seconds=timeline_coverage_seconds(track_ids, take_tracks),
                    take_duration_seconds=take_duration,
                    evidence=verdict.evidence,
                    # Stage B: EVERY candidate that independently passed `verify_candidate` this
                    # take, not just the highest-scoring `winner_id` -- each one was tested against
                    # the SAME strict hard-reject-then-score gate on its own merits (e.g. two
                    # ByteTrack fragments of the same real player either side of a brief ID reset,
                    # both plausibly the target). A fragment merely joined to the winner by
                    # `stitch_timeline`'s own geometry (in `track_ids` but NOT in `accepts`) never
                    # went through that gate and stays a candidate (green), never red.
                    verified_track_ids=[tid for tid, _ in accepts],
                )

    _update_profile_after_take(profile, take, selection, take_tracks, kit_by_track, target_cfg)
    return selection


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
    camera_motion_by_take: dict | None = None,
    target_profile: TargetProfile | None = None,
    target_cfg: dict | None = None,
    kit_by_track_by_take: dict[int, dict[int, KitColourSample]] | None = None,
    jersey_by_track_by_take: dict[int, dict[int, str]] | None = None,
) -> SelectionResult:
    """Run Stage 5 target selection for every take of one video.

    `manual_overrides` is the actual Phase-1 human-in-the-loop seam (`{take_id: track_id}`, from
    `src/pipeline/run.py`'s `--track-id` CLI flag) — a human-supplied track always wins over both
    the arrow vote and the fallback heuristic for that take, and is the ONLY case whose
    `TakeSelection.method` is `"manual_override"`.

    **"streamed-gathering-treehouse" plan Stage 4 — `target_profile`.** When `None` (every
    pre-existing caller), this function is completely unchanged: arrow vote -> within-take
    stitching -> `heuristic_fallback_seed`, exactly as before. When a `TargetProfile` IS supplied
    (`target_cfg` becomes required in that case -- `configs/target.yaml` merged with
    `configs["events"]["kit_colour"]` under its own `"kit_colour"` key, see
    `src.track.target_verify.verify_candidate`'s own docstring for why the caller owns that
    merge), every take is instead resolved by `_select_take_with_profile` below:
    `heuristic_fallback_seed` (and the arrow vote) are NEVER consulted -- the single most
    important line of this plan (owner's own words) -- and a manual override for that take is
    itself VERIFIED against the profile rather than trusted blindly. A take with no `ACCEPT`
    candidate (including a manual override that fails verification, or one whose track id isn't
    even in that take at all -- the plan Stage 4 fix for the old silent-fallback bug at
    `selection.py:296-303`) becomes `method="target_lost"`, `track_ids=[]`, `confidence=0.0`,
    never a substitute player. `target_profile` is MUTATED in place (a new `TargetLink` appended
    per take, and its memory bank updated via `update_memory_bank` for every accepted link) so its
    caller can persist the evolved profile after this call returns -- kept a side effect
    deliberately, rather than a second return value, so this function's return type stays
    `SelectionResult` unconditionally regardless of whether a profile was supplied (a hard
    requirement: the no-profile path must remain byte-for-byte unaffected, including its own
    return type).
    """
    if target_profile is not None and target_cfg is None:
        raise ValueError("target_cfg is required when target_profile is given")

    manual_overrides = manual_overrides or {}
    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    votes_by_take = vote_seed_tracks(arrow_hints, tracks, takes, selection_cfg)

    def _motion_for(take: Take):
        """That take's own camera-motion model, or `None` (which makes every stitch comparison
        fall back to the raw image-space distance, i.e. the exact pre-2026-09-01 behaviour)."""
        if not camera_motion_by_take:
            return None
        return camera_motion_by_take.get(take.id)

    take_selections: list[TakeSelection] = []
    for take in takes:
        take_tracks = tracks_by_take.get(take.id, [])
        take_duration = take.t_end - take.t_start

        if target_profile is not None:
            # Plan Stage 4's own branch, fully separate from the no-profile code below it (never
            # falls through into it) -- this is what structurally guarantees
            # `heuristic_fallback_seed`/the arrow vote are never reached when a profile exists,
            # not just an informal avoidance.
            selection = _select_take_with_profile(
                take,
                take_tracks,
                take_duration,
                target_profile,
                target_cfg,
                (kit_by_track_by_take or {}).get(take.id, {}),
                (jersey_by_track_by_take or {}).get(take.id, {}),
                manual_overrides.get(take.id),
                selection_cfg,
                _motion_for(take),
            )
            take_selections.append(selection)
            continue

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
                track_ids, _stitch_conf = stitch_timeline(
                    seed, take_tracks, selection_cfg, _motion_for(take)
                )
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
                track_ids, stitch_conf = stitch_timeline(
                    seed_candidate, take_tracks, selection_cfg, _motion_for(take)
                )
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

        track_ids, stitch_conf = stitch_timeline(
            seed, take_tracks, selection_cfg, _motion_for(take)
        )
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

    # "target_lost" (plan Stage 4) is the profile-driven equivalent of "none" -- no usable track
    # at all for that take -- so it is excluded from the confidence average the same way.
    usable = [t for t in take_selections if t.method not in ("none", "target_lost")]
    overall_confidence = sum(t.confidence for t in usable) / len(usable) if usable else 0.0
    needs_confirmation = any(t.method != "manual_override" for t in take_selections)

    return SelectionResult(
        video=str(video_path),
        target_jersey=target_jersey,
        overall_confidence=overall_confidence,
        needs_human_confirmation=needs_confirmation,
        takes=take_selections,
    )
