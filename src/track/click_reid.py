"""Click-anchored re-identification: extend a human-clicked player's chain across a within-take
ByteTrack fragmentation break, using the clicked player's OWN measured colour/height profile as
the matching signal.

CLAUDE.md Golden Rule 4: "a human-provided [identity] is the strongest evidence we have." A click
already IS the target -- this module's only job is to keep following that SAME physical player
through the raw tracker's own fragmentation, never to re-derive who they are from scratch. It runs
strictly AFTER `src.highlights.selection.stitch_timeline` has already produced its own forward
chain from the clicked seed (`src/pipeline/run.py`'s manual-override branch); it only picks up
fragments that chain missed because they fall outside its own gap/distance thresholds
(`configs/highlights.yaml: selection.stitch_max_gap_s/stitch_max_dist`).

Why this, not a trained ReID model: evidence-checked and rejected earlier in this project (see
CLAUDE.md's ADR log) -- no good permissively-licensed appearance-ReID option exists (`trackers`
dropped its ReID model in v2.1; StrongSORT is GPL-3.0; torchreid's PyPI package is stale with
academically-restricted weights). Team cluster and box height are already computed for every RAW
track by Stage 3 (`src.team.classifier.assign_teams`) and persisted in `tracks.parquet`, so this
needs **no extra video decode and no new dependency** -- it only combines signals the pipeline
already has. Jersey-number corroboration is accepted as an OPTIONAL extra signal when a caller
already has one (e.g. a UI's own advisory read via `src.identity.jersey_parseq`), never required.
"""

from __future__ import annotations

from pathlib import Path

from src.common.types import Track
from src.common.video import decode_frames
from src.identity.jersey_ocr import upscale_crop
from src.identity.jersey_parseq import number_region, read_jersey_number_parseq
from src.identity.legibility import is_legible
from src.identity.verify import FrameRead, _scale_bbox_to_native, aggregate_take_identity


def _median_height(track: Track) -> float:
    """Median box height across `track`'s own boxes -- robust to a couple of bad detections,
    unlike a mean. `0.0` for an empty track (never divides by zero downstream)."""
    heights = sorted(b.bbox.height for b in track.boxes)
    if not heights:
        return 0.0
    mid = len(heights) // 2
    return heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0


def build_click_profile(anchor_track: Track) -> dict:
    """Summarise the clicked player's own measured signals -- team cluster and median box height
    -- for `candidate_score` to compare every OTHER same-take fragment against."""
    return {
        "team": anchor_track.team,
        "median_height": _median_height(anchor_track),
    }


def candidate_score(
    candidate: Track,
    profile: dict,
    reid_cfg: dict,
    jersey_by_track_id: dict[int, str] | None = None,
    anchor_track_id: int | None = None,
) -> tuple[bool, dict]:
    """`(accept, evidence)` -- whether `candidate` is plausibly the SAME physical player as the
    clicked anchor behind `profile`, and the full evidence trail behind that decision (Golden Rule
    5: every join or rejection is explainable, never a silent guess).

    - **Team cluster**: `None` on either side means Stage 3's team assignment didn't run or wasn't
      confident for that take -- treated as "no evidence either way" (`team_match: None`), not a
      rejection (a low-confidence team call is common on crowded footage, ADR-12). A confident
      MISMATCH is a hard reject.
    - **Height**: compared as a RATIO (`min/max`), never an absolute pixel difference -- a player
      further from camera reads smaller, which is not evidence of being a different person.
    - **Jersey (optional)**: only consulted when the caller supplies confident reads for BOTH the
      anchor and the candidate (`jersey_by_track_id`, already filtered by the caller to whatever
      confidence bar it trusts -- this function does no thresholding of its own). Agreement is
      strong corroboration; a DISAGREEMENT is a hard veto regardless of colour/height, since a
      printed number is the strongest available disambiguator between two similarly-dressed
      teammates.
    """
    evidence: dict = {"candidate_track_id": candidate.id}

    team_a, team_b = profile.get("team"), candidate.team
    if team_a is not None and team_b is not None:
        team_match = team_a == team_b
        evidence["team_match"] = team_match
        if not team_match:
            evidence["rejected_reason"] = "different_team_cluster"
            return False, evidence
    else:
        evidence["team_match"] = None

    height_a = profile.get("median_height", 0.0)
    height_b = _median_height(candidate)
    if height_a > 0 and height_b > 0:
        ratio = min(height_a, height_b) / max(height_a, height_b)
        evidence["height_ratio"] = round(ratio, 3)
        if ratio < reid_cfg["min_height_ratio"]:
            evidence["rejected_reason"] = "height_mismatch"
            return False, evidence
    else:
        evidence["height_ratio"] = None

    if jersey_by_track_id and anchor_track_id is not None:
        digits_a = jersey_by_track_id.get(anchor_track_id)
        digits_b = jersey_by_track_id.get(candidate.id)
        if digits_a is not None and digits_b is not None:
            evidence["jersey_agrees"] = digits_a == digits_b
            if digits_a != digits_b:
                evidence["rejected_reason"] = "jersey_number_disagrees"
                return False, evidence

    evidence["accepted"] = True
    return True, evidence


def _sample_track_timestamps(track: Track, max_samples: int) -> list[float]:
    """Up to `max_samples` timestamps evenly spread across `track`'s OWN box lifespan -- same
    "don't judge a player from one instant" reasoning as
    `src.annotations.associate._candidate_target_timestamps` (a player side-on in one frame may be
    facing the camera elsewhere in the same track), simplified here since there is no take-anchored
    window to widen: this track's own boxes ARE the full population to sample from."""
    if not track.boxes:
        return []
    if len(track.boxes) <= max_samples:
        return [b.t for b in track.boxes]
    step = len(track.boxes) / max_samples
    return [track.boxes[int(i * step)].t for i in range(max_samples)]


def collect_track_jersey_digits(
    video_path: str | Path,
    take_tracks: list[Track],
    scale_x: float,
    scale_y: float,
    native_w: int,
    native_h: int,
    identity_cfg: dict,
    legibility_model,
    parseq_model,
    parseq_transform,
    max_samples_per_track: int,
    aggregation_cfg: dict,
    use_nvdec: bool = True,
) -> dict[int, str]:
    """Best-effort confident jersey-number digit string per raw track in `take_tracks` --
    corroborating evidence for `candidate_score`'s jersey check, so click-anchored chain extension
    is not left with colour+height alone (measured 2026-09-01, see `configs/highlights.yaml:
    click_reid`'s own comment: colour alone let a DIFFERENT physical player join a clicked chain).

    Reuses the exact ADR-21 chain (`src/identity/jersey_parseq.py::number_region` -- gate on the
    full-body crop, read the tight number region -- the fix for PARSeq's own 128x32 text-line input
    shape) and the SAME per-item aggregation rule `src.identity.verify.aggregate_take_identity`
    already applies for ADR-15's own per-take verification (>= `min_agreeing_frames` reads must
    agree, a strict plurality, clearing `min_agreement_fraction`/`min_verified_confidence`) --
    applied here PER TRACK instead of per take. A track with no confident majority is simply absent
    from the returned dict (an honest "no jersey evidence", not a guess) -- `candidate_score`
    already treats a missing entry as "no evidence either way", never a rejection.

    `legibility_model`/`parseq_model` may be `None` (checkpoint unavailable) -- returns `{}` rather
    than raising, same "optional stage, never fatal" contract as
    `src.identity.jersey_models.load_optional_jersey_stack`'s own callers.
    """
    if legibility_model is None or parseq_model is None or not take_tracks:
        return {}

    crop_cfg = identity_cfg["crop"]
    parseq_cfg = identity_cfg["parseq_soccernet"]
    legibility_cfg = identity_cfg["legibility"]
    min_height = crop_cfg["min_crop_height_frac"] * native_h
    expand = crop_cfg["torso_crop_expand"]
    upscale_factor = crop_cfg["crop_upscale_factor"]

    target_ts_by_track = {
        tr.id: _sample_track_timestamps(tr, max_samples_per_track) for tr in take_tracks
    }
    all_ts = [t for ts in target_ts_by_track.values() for t in ts]
    if not all_ts:
        return {}

    # ONE decode pass over the union of every track's own sample timestamps -- never a per-track
    # or per-sample re-seek (matches src/identity/verify.py's/src/annotations/associate.py's own
    # "one decode, bucket into instants" discipline).
    decoded_frames: list[tuple[float, object]] = []
    for _idx, frame_t, frame in decode_frames(
        str(video_path),
        fps=identity_cfg["crop"]["identity_fps_sample"],
        start=max(0.0, min(all_ts) - 0.5),
        end=max(all_ts) + 0.5,
        scale_width=None,
        use_nvdec=use_nvdec,
    ):
        decoded_frames.append((frame_t, frame))
    if not decoded_frames:
        return {}

    tolerance_s = 1.0 / identity_cfg["crop"]["identity_fps_sample"]
    reads_by_track: dict[int, list[FrameRead]] = {}
    frame_index = 0
    for tr in take_tracks:
        boxes_by_t = {b.t: b for b in tr.boxes}
        reads: list[FrameRead] = []
        for target_t in target_ts_by_track[tr.id]:
            box = boxes_by_t.get(target_t)
            if box is None:
                continue
            frame = min(decoded_frames, key=lambda f: abs(f[0] - target_t), default=None)
            if frame is None or abs(frame[0] - target_t) > tolerance_s:
                continue
            _t, frame_bgr = frame
            x1, y1, x2, y2 = _scale_bbox_to_native(
                box.bbox, scale_x, scale_y, expand, native_w, native_h
            )
            if (y2 - y1) < min_height or x2 <= x1 or y2 <= y1:
                continue
            frame_index += 1
            body_crop = upscale_crop(frame_bgr[y1:y2, x1:x2], upscale_factor)
            leg = is_legible(body_crop, legibility_model, legibility_cfg)
            if body_crop.size == 0 or not leg.is_legible:
                continue
            nx1, ny1, nx2, ny2 = number_region(x1, y1, x2, y2, parseq_cfg["number_crop"])
            number_crop = upscale_crop(frame_bgr[ny1:ny2, nx1:nx2], upscale_factor)
            read = read_jersey_number_parseq(
                number_crop, parseq_model, parseq_transform, parseq_cfg
            )
            if read.is_confident:
                reads.append(
                    FrameRead(
                        t=target_t,
                        frame_index=frame_index,
                        source="parseq_soccernet",
                        digits=read.digits,
                        confidence=read.confidence,
                        raw=read.raw_text,
                    )
                )
        if reads:
            reads_by_track[tr.id] = reads

    jersey_by_track_id: dict[int, str] = {}
    for track_id, reads in reads_by_track.items():
        agg = aggregate_take_identity(reads, aggregation_cfg)
        if agg["status"] == "verified" and agg["jersey_number"] is not None:
            jersey_by_track_id[track_id] = str(agg["jersey_number"])
    return jersey_by_track_id


def prune_chain_by_jersey(
    seed_track_id: int,
    chain_track_ids: list[int],
    jersey_by_track_id: dict[int, str],
) -> tuple[list[int], list[dict]]:
    """Drop any fragment of a clicked player's chain whose own confident jersey read CONTRADICTS
    the seed's. Returns `(kept_ids, evidence_trail)`.

    This is the SUBTRACTIVE counterpart to `extend_chain_with_profile`, and unlike that additive
    one it is safe to run by default: it can only ever REMOVE a fragment, never introduce a new
    physical player into the chain. Worst case (no confident reads anywhere) it is a no-op.

    Measured need, 2026-09-01 on `chelsea_burnley_target10` take 0 at 25 fps: the owner clicked
    Chelsea's blue #10 (raw track 1, which the ADR-21 reader confirms as "10" on 15 of 16 sampled
    frames -- 94%). `stitch_timeline` then joined raw track 47 across a 0.76s gap, and track 47 is
    visually a CLARET BURNLEY #21 -- a different team, let alone a different player. The chain
    therefore covered 60.5s of the take while showing the wrong person for its last 35s, which is
    exactly the owner-reported "the target frame switch to another player".

    Neither of the pipeline's other signals catches it: `Track.team` is degenerate on this take
    (194 of 231 raw tracks share one label), and a torso-colour median is heavily contaminated by
    grass background (track 47 measures near-neutral, not claret). The printed NUMBER does catch
    it -- the two fragments read "10" and "21" respectively -- so a disagreement veto is the one
    signal here that reflects reality rather than noise.

    A fragment with NO confident read is KEPT, never dropped: absence of evidence is not evidence
    of a different player (Golden Rule 5), and dropping unread fragments would silently gut the
    chain on footage where numbers are rarely legible (the 4K amateur clips read 0/801).
    """
    seed_digits = jersey_by_track_id.get(seed_track_id)
    if seed_digits is None:
        return chain_track_ids, [{"skipped": "seed has no confident jersey read"}]

    kept: list[int] = []
    trail: list[dict] = []
    for tid in chain_track_ids:
        digits = jersey_by_track_id.get(tid)
        if digits is not None and digits != seed_digits:
            trail.append(
                {
                    "track_id": tid,
                    "dropped": True,
                    "reason": "jersey_disagrees_with_clicked_seed",
                    "seed_jersey": seed_digits,
                    "fragment_jersey": digits,
                }
            )
            continue
        kept.append(tid)
        trail.append({"track_id": tid, "dropped": False, "fragment_jersey": digits})
    return kept, trail


def extend_chain_with_profile(
    seed_track_id: int,
    stitched_track_ids: list[int],
    take_tracks: list[Track],
    reid_cfg: dict,
    jersey_by_track_id: dict[int, str] | None = None,
) -> tuple[list[int], list[dict]]:
    """Extend `stitched_track_ids` (already `stitch_timeline`'s own forward chain from the clicked
    seed) with any OTHER same-take fragment that plausibly continues the SAME physical player --
    a break `stitch_timeline` itself didn't bridge (a gap/jump outside its own thresholds) but
    that colour+height (+jersey, if available) evidence says is still the same person.

    Never joins a fragment that overlaps in TIME with an already-accepted one: that would mean two
    different physical people were on screen as "the same track" at once, and picking one would be
    a silent, unjustified merge (Golden Rule 5). Returns the extended id list plus a full evidence
    trail covering every candidate considered -- accepted AND rejected -- for the run report.
    """
    by_id = {tr.id: tr for tr in take_tracks}
    anchor = by_id.get(seed_track_id)
    if anchor is None:
        return stitched_track_ids, []

    profile = build_click_profile(anchor)
    accepted_ids = list(stitched_track_ids)
    accepted_spans = [
        (by_id[tid].t_start, by_id[tid].t_end) for tid in accepted_ids if tid in by_id
    ]
    evidence_trail: list[dict] = []

    remaining = [tr for tr in take_tracks if tr.id not in accepted_ids]
    for candidate in sorted(remaining, key=lambda tr: (tr.t_start, tr.id)):
        overlaps = any(
            candidate.t_start < end and start < candidate.t_end for start, end in accepted_spans
        )
        if overlaps:
            evidence_trail.append(
                {
                    "candidate_track_id": candidate.id,
                    "rejected_reason": "time_overlap_with_accepted",
                }
            )
            continue
        accept, evidence = candidate_score(
            candidate, profile, reid_cfg, jersey_by_track_id, anchor_track_id=seed_track_id
        )
        evidence_trail.append(evidence)
        if accept:
            accepted_ids.append(candidate.id)
            accepted_spans.append((candidate.t_start, candidate.t_end))

    return accepted_ids, evidence_trail
