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

from src.common.types import Annotation, BBox, Take, Track
from src.common.video import decode_frames
from src.detect.overlay_mask import ArrowHint
from src.highlights.selection import _center_distance, _contains, _nearest_box
from src.identity.jersey_ocr import read_jersey_digits, upscale_crop
from src.identity.jersey_parseq import number_region, read_jersey_number_parseq
from src.identity.jersey_vlm import classify_jersey_number
from src.identity.legibility import is_legible
from src.identity.verify import FrameRead, aggregate_take_identity
from src.team.classifier import per_track_lab_median, torso_region

__all__ = [
    "associate_annotation_to_track",
    "candidate_tracks_near_time",
    "expanded_bbox_region",
    "nearest_track_by_arrow_hint",
    "nearest_track_by_colour",
    "opencv_lab_to_cielab",
    "read_jersey_number_for_candidates",
]


def candidate_tracks_near_time(tracks: list[Track], t: float, tolerance_s: float) -> list[Track]:
    """Every track with at least one box within `tolerance_s` seconds of `t` -- the LOCATION
    candidate pool for one annotation instant (pure, no I/O, directly unit-testable)."""
    return [tr for tr in tracks if any(abs(b.t - t) <= tolerance_s for b in tr.boxes)]


def opencv_lab_to_cielab(raw_lab: np.ndarray) -> np.ndarray:
    """Convert `src.team.classifier.crop_lab_mean`'s own OpenCV-scaled output into TRUE CIELAB
    units.

    **Bug fix, 2026-08-31.** OpenCV's `COLOR_BGR2LAB` returns an 8-bit-scaled encoding, not true
    CIELAB: ``L_cv = L_true * 255/100``, ``a_cv = a_true + 128``, ``b_cv = b_true + 128`` (OpenCV's
    own documented convention). `crop_lab_mean`/`per_track_lab_median` never undo this scaling --
    correctly so, since every OTHER consumer of those functions (ADR-12's own team-colour
    clustering in `src.team.classifier`) only ever compares tracks against EACH OTHER in that same
    raw scale, which is perfectly self-consistent and must NOT be changed here. But
    `configs/annotations.yaml: colour_reference_lab` is written in TRUE CIELAB units (e.g.
    ``white: [100.0, 0.0, 0.0]``, standard sRGB -> CIELAB conversions -- see that file's own
    comment), and `nearest_track_by_colour` compares a track's Lab directly against that table.
    Without this conversion the two sides are in different units: confirmed on real cached data
    (`work/chelsea_burnley_target10/track/tracks.parquet`, t=57.0s) where every candidate track's
    raw L value came out 126-146 -- impossible for true CIELAB, which caps at 100 -- inflating
    every distance by roughly 100+ units and silently defeating `colour_match_max_distance` for
    every colour, on every video. This is the one conversion point that fixes it; apply it ONLY
    where a track's own measured Lab is about to be compared against `colour_reference_lab` (i.e.
    here, in `associate_annotation_to_track`), never inside `src.team.classifier` itself.
    """
    l_cv, a_cv, b_cv = raw_lab
    return np.array([l_cv * 100.0 / 255.0, a_cv - 128.0, b_cv - 128.0])


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


def expanded_bbox_region(bbox: BBox, expand_frac: float) -> BBox:
    """The FULL player bbox, expanded by `expand_frac` on each side -- the crop shape ADR-15's own
    `src.identity.verify._scale_bbox_to_native` uses for jersey OCR/VLM, deliberately NOT
    `torso_region`'s tighter chest-band crop (`configs/annotations.yaml: track_association.
    torso_top_frac/torso_bottom_frac/torso_x_inset_frac`, tuned for CIELAB colour sampling, not
    digit legibility -- a jersey number printed low on the back or large on the chest can sit
    outside that band, or be clipped by its own `x_inset_frac`). Pure geometry, no clipping to
    frame bounds (the caller clips, since that needs the frame's own shape) -- same split as
    `torso_region` itself.
    """
    width = bbox.x2 - bbox.x1
    height = bbox.y2 - bbox.y1
    return BBox(
        x1=bbox.x1 - width * expand_frac,
        y1=bbox.y1 - height * expand_frac,
        x2=bbox.x2 + width * expand_frac,
        y2=bbox.y2 + height * expand_frac,
    )


def _candidate_lifespan(tr: Track, take: Take, t: float, window_s: float) -> tuple[float, float]:
    """`(start, end)` -- candidate `tr`'s own box lifespan, widened by `window_s` either side of
    `t` (so a track whose lifespan is very short around `t` still gets a real span to sample from),
    clamped into `take`. Split out from `_candidate_target_timestamps` so the DECODE window
    (`read_jersey_number_for_candidates`) can be sized from the SAME true lifespan bounds every
    candidate's sample points are drawn from -- bug caught during test-writing 2026-08-31: sizing
    the decode window from the scattered SAMPLE points' own min/max instead (they sit strictly
    inside the lifespan, `span * 0.5/max_frames` short of each edge) can miss frames right at the
    lifespan's own edges, exactly where a track's most-recent real box data lives.
    """
    if not tr.boxes:
        start = end = t
    else:
        start = min(b.t for b in tr.boxes)
        end = max(b.t for b in tr.boxes)
    start = min(start, t - window_s)
    end = max(end, t + window_s)
    start = max(take.t_start, start)
    end = min(take.t_end, end)
    if end <= start:
        end = start
    return start, end


def _candidate_target_timestamps(
    tr: Track, take: Take, t: float, window_s: float, max_frames: int
) -> list[float]:
    """Up to `max_frames` timestamps spread across candidate `tr`'s OWN box lifespan
    (`_candidate_lifespan`) -- owner-reported bug fix, 2026-08-31: a single crop near the
    annotation's own instant `t` gives jersey confirmation only one chance; a player side-on at `t`
    may be facing the camera at another point in the SAME track's own lifespan (confirmed on real
    broadcast footage: a locked track's box height stayed 70-97px, at/below EasyOCR's reliable
    floor, across its ENTIRE lifespan, but ONE specific pose/angle within that span could still be
    legible).

    Falls back to `[t]` alone when `tr` has no boxes (defensive) or its lifespan collapses to a
    single instant. Deliberately bounded to `max_frames` (never "scan the whole track") -- this is
    the cost-bounding knob `jersey_reid_cfg['max_frames_per_candidate']` controls.
    """
    lifespan_start, lifespan_end = _candidate_lifespan(tr, take, t, window_s)
    if lifespan_end <= lifespan_start or max_frames <= 1:
        return [t]
    span = lifespan_end - lifespan_start
    return [lifespan_start + span * (i + 0.5) / max_frames for i in range(max_frames)]


def read_jersey_number_for_candidates(
    video_path: str | Path,
    take: Take,
    candidates: list[Track],
    t: float,
    claimed_jersey_number: int,
    identity_cfg: dict,
    jersey_reid_cfg: dict,
    decode_cfg: dict,
    sampling_cfg: dict,
    reader,
    gemini_api_key: str | None,
    use_nvdec: bool = True,
    priority_track_id: int | None = None,
    legibility_model=None,
    parseq_model=None,
    parseq_transform=None,
) -> tuple[int | None, str | None, dict]:
    """Stage B ("streamed-gathering-treehouse" plan, revised 2026-08-31 after a real owner-found
    misidentification on `chelsea_burnley_target10`) -- best-effort jersey-NUMBER read across
    `candidates`, looking specifically for TEMPORALLY-AGREEING digit reads matching
    `claimed_jersey_number` (the annotation's own stated number): the strongest possible
    re-identification signal available, since two teammates in identical kit are only
    distinguishable by the printed number, not by colour.

    **Reading chain, owner-authorized 2026-08-31 (CLAUDE.md §7), same order as
    `src.identity.verify._collect_reads_for_take`:** a legibility gate
    (`src/identity/legibility.py`) first screens each crop; a legible crop is read by
    PARSeq-SoccerNet (`src/identity/jersey_parseq.py`) before falling back to the PRE-EXISTING
    EasyOCR/Gemini chain. `legibility_model`/`parseq_model`/`parseq_transform` default to `None`,
    which degrades this exactly to the prior OCR-first/VLM-escalation behaviour (missing/disabled
    checkpoints are not an error here -- see `src.identity.jersey_models.load_optional_jersey_stack`,
    which the caller in `src.pipeline.manual_events` uses to load these once per video).

    **Multi-frame-per-candidate, not one snapshot** (the real fix): for EACH candidate, up to
    `jersey_reid_cfg['max_frames_per_candidate']` crops are sampled across THAT candidate's OWN
    box lifespan (`_candidate_target_timestamps`), not just one crop near `t` -- a player side-on
    at `t` may be legible elsewhere in the same track. Each crop is UPSCALED
    (`src.identity.jersey_ocr.upscale_crop`, `identity_cfg['crop']['crop_upscale_factor']`) before
    OCR/VLM -- confirmed on real 720p broadcast footage that a locked track's own box height
    (70-97px across its whole lifespan) sits at/below EasyOCR's practical reliable floor without
    this. Every read (OCR and/or VLM, any mix) for one candidate is reduced via
    `src.identity.verify.aggregate_take_identity` -- the SAME temporal-agreement rule ADR-15 itself
    uses (`identity_cfg['aggregation']`, `min_agreeing_frames` independent frames must agree before
    a digit counts) -- so a single lucky/unlucky misread can no longer single-handedly decide a
    candidate's fate. Decodes ONE pass over the WHOLE TAKE (not per-candidate re-seeking, matching
    `src/identity/verify.py`'s own "one decode, bucket by candidate" discipline) at
    `decode_cfg`'s own resolution (matching whatever resolution `candidates`' own boxes are already
    expressed in).

    **Cost bound**: OCR is cheap/local and runs on every sampled crop; Gemini VLM escalation is
    capped at `jersey_reid_cfg['max_vlm_escalations_per_call']` TOTAL for this one invocation
    (shared across every candidate, not per-candidate) -- a take with many same-kit candidates
    still cannot turn this into an unbounded VLM scan.

    **`priority_track_id`, bug fix 2026-08-31** (real broadcast data:
    `chelsea_burnley_target10`, 15 near-time candidates, a shared VLM budget of 6 -- OCR was
    ambiguous on every one of 45 sampled crops, so the budget was entirely consumed by whichever
    TWO candidates happened to sort first by raw track id, and the colour-selected candidate
    itself -- track 221 -- was never even reached before the budget ran out). When given (the
    caller's own colour/arrow pick, if any), that candidate is checked FIRST, before the
    deterministic track-id order for everyone else -- the candidate colour ALREADY thinks is the
    target gets first claim on the shared VLM budget, rather than being crowded out by unrelated
    nearby players. `None` (the default) preserves plain deterministic track-id order.

    Returns `(track_id, source, evidence)`:
    - `(track_id, "parseq_soccernet"|"ocr"|"vlm"|"mixed", evidence)` when EXACTLY ONE candidate's
      own aggregated reads verify to `str(claimed_jersey_number)`.
    - `(None, None, evidence)` when no candidate's reads verify to the claimed number, OR when MORE
      THAN ONE candidate's reads do (an honest "can't disambiguate", never an arbitrary pick --
      Golden Rule 5; `evidence['outcome']` names which case happened).
    `evidence` always carries `{n_crops_considered, n_too_small, n_legible, n_parseq_confident,
    n_ocr_confident, n_vlm_calls, n_vlm_failed, reads: [{track_id, source, digits, confidence}, ...],
    outcome}` -- the FULL per-candidate-per-frame trail, not just the winner (Golden Rule 5).
    """
    evidence: dict = {
        "n_crops_considered": 0,
        "n_too_small": 0,
        "n_legible": 0,
        "n_parseq_confident": 0,
        "n_ocr_confident": 0,
        "n_vlm_calls": 0,
        "n_vlm_failed": 0,
        "reads": [],
    }
    if not candidates:
        evidence["outcome"] = "no_candidates"
        return None, None, evidence

    tolerance_s = sampling_cfg["match_tolerance_s"]
    window_s = sampling_cfg["window_s"]
    expand = identity_cfg["crop"]["torso_crop_expand"]
    min_height_frac = identity_cfg["crop"]["min_crop_height_frac"]
    upscale_factor = identity_cfg["crop"]["crop_upscale_factor"]
    ocr_cfg = identity_cfg["ocr"]
    vlm_cfg = identity_cfg["vlm"]
    agg_cfg = identity_cfg["aggregation"]
    legibility_cfg = identity_cfg.get("legibility", {})
    parseq_cfg = identity_cfg.get("parseq_soccernet", {})
    max_vlm = jersey_reid_cfg["max_vlm_escalations_per_call"]
    max_frames_per_candidate = jersey_reid_cfg["max_frames_per_candidate"]
    min_disambiguation_margin = jersey_reid_cfg["min_disambiguation_margin"]

    lifespans_by_track: dict[int, tuple[float, float]] = {
        tr.id: _candidate_lifespan(tr, take, t, window_s) for tr in candidates
    }
    target_ts_by_track: dict[int, list[float]] = {
        tr.id: _candidate_target_timestamps(tr, take, t, window_s, max_frames_per_candidate)
        for tr in candidates
    }
    # Decode window sized from the TRUE lifespan bounds (not the scattered sample points' own
    # min/max, which sit strictly inside each lifespan and can miss frames right at its edges --
    # see `_candidate_lifespan`'s own docstring).
    decode_start = max(take.t_start, min(s for s, _e in lifespans_by_track.values()))
    decode_end = min(take.t_end, max(e for _s, e in lifespans_by_track.values()))

    # ONE decode pass over the union of every candidate's own lifespan -- never a
    # separate re-seek per candidate or per frame (matches src/identity/verify.py's own "one
    # decode, bucket into instants" discipline).
    decoded_frames: list[tuple[float, np.ndarray]] = []
    for _idx, frame_t, frame in decode_frames(
        video_path,
        fps=sampling_cfg["fps_sample"],
        start=decode_start,
        end=decode_end,
        scale_width=decode_cfg.get("scale_width"),
        use_nvdec=use_nvdec,
    ):
        decoded_frames.append((frame_t, frame.copy()))

    def _best_crop_at(tr: Track, target_t: float) -> np.ndarray | None:
        """Nearest decoded frame to `target_t` that has a box for `tr` within `tolerance_s`, as an
        upscaled, size-gated crop -- `None` when no such frame/box/size exists."""
        best: tuple[float, np.ndarray] | None = None
        for frame_t, frame in decoded_frames:
            box = next((b for b in tr.boxes if abs(b.t - frame_t) <= tolerance_s), None)
            if box is None:
                continue
            dt = abs(frame_t - target_t)
            if best is not None and best[0] <= dt:
                continue
            height, width = frame.shape[:2]
            min_height_px = min_height_frac * height
            region = expanded_bbox_region(box.bbox, expand)
            x1, y1 = max(0, int(round(region.x1))), max(0, int(round(region.y1)))
            x2, y2 = min(width, int(round(region.x2))), min(height, int(round(region.y2)))
            if x2 - x1 < 2 or (y2 - y1) < min_height_px:
                continue
            best = (dt, frame[y1:y2, x1:x2].copy())
        return best[1] if best is not None else None

    vlm_budget = max_vlm
    frame_index = 0
    verified_track_ids: list[int] = []
    supporting_sources_by_track: dict[int, set[str]] = {}
    agreement_by_track: dict[int, float] = {}

    # Deterministic order, EXCEPT the caller's own colour/arrow pick (if any) goes first -- see
    # this function's own `priority_track_id` docstring for the real cost-exhaustion bug this
    # closes.
    ordered_candidates = sorted(candidates, key=lambda tr: (tr.id != priority_track_id, tr.id))
    for tr in ordered_candidates:
        candidate_reads: list[FrameRead] = []
        for target_t in target_ts_by_track[tr.id]:
            crop = _best_crop_at(tr, target_t)
            if crop is None:
                evidence["n_too_small"] += 1
                continue
            crop = upscale_crop(crop, upscale_factor)
            evidence["n_crops_considered"] += 1
            frame_index += 1

            # Owner-authorized 2026-08-31 (CLAUDE.md §7): legibility gate -> PARSeq-SoccerNet,
            # ahead of the pre-existing EasyOCR path (same chain/order as
            # `src.identity.verify._collect_reads_for_take`). `legibility_model`/`parseq_model`
            # are `None` whenever the NC-restricted checkpoints weren't loaded -- this block is
            # then a no-op and control falls straight through to the unchanged OCR/VLM chain.
            if legibility_model is not None and parseq_model is not None:
                # Gate on the FULL-BODY crop (what mkoshkina's resnet34 was trained on), but read
                # the tight upper-torso NUMBER REGION -- ADR-21. `crop` IS the full-body box here,
                # so the region is taken against its own bounds; the insets are fractions, so
                # applying them after `upscale_crop` is geometrically identical.
                leg = is_legible(crop, legibility_model, legibility_cfg)
                if leg.is_legible:
                    evidence["n_legible"] += 1
                    ch, cw = crop.shape[:2]
                    nx1, ny1, nx2, ny2 = number_region(
                        0, 0, cw, ch, parseq_cfg["number_crop"]
                    )
                    parseq_read = read_jersey_number_parseq(
                        crop[ny1:ny2, nx1:nx2], parseq_model, parseq_transform, parseq_cfg
                    )
                    if parseq_read.is_confident:
                        evidence["n_parseq_confident"] += 1
                        candidate_reads.append(
                            FrameRead(
                                t=target_t,
                                frame_index=frame_index,
                                source="parseq_soccernet",
                                digits=parseq_read.digits,
                                confidence=parseq_read.confidence,
                                raw=parseq_read.raw_text,
                            )
                        )
                        evidence["reads"].append(
                            {
                                "track_id": tr.id,
                                "source": "parseq_soccernet",
                                "digits": parseq_read.digits,
                                "confidence": parseq_read.confidence,
                            }
                        )
                        continue

            ocr_read = read_jersey_digits(reader, crop, ocr_cfg)
            if ocr_read.is_confident:
                evidence["n_ocr_confident"] += 1
                candidate_reads.append(
                    FrameRead(
                        t=target_t,
                        frame_index=frame_index,
                        source="ocr",
                        digits=ocr_read.digits,
                        confidence=ocr_read.confidence,
                    )
                )
                evidence["reads"].append(
                    {
                        "track_id": tr.id,
                        "source": "ocr",
                        "digits": ocr_read.digits,
                        "confidence": ocr_read.confidence,
                    }
                )
                continue

            if gemini_api_key and vlm_budget > 0:
                vlm_budget -= 1
                evidence["n_vlm_calls"] += 1
                digits, conf, raw = classify_jersey_number(crop, gemini_api_key, vlm_cfg)
                if raw.startswith("CALL_FAILED"):
                    evidence["n_vlm_failed"] += 1
                candidate_reads.append(
                    FrameRead(
                        t=target_t,
                        frame_index=frame_index,
                        source="vlm",
                        digits=digits,
                        confidence=conf,
                    )
                )
                evidence["reads"].append(
                    {"track_id": tr.id, "source": "vlm", "digits": digits, "confidence": conf}
                )

        if not candidate_reads:
            continue
        agg = aggregate_take_identity(candidate_reads, agg_cfg)
        if agg["status"] == "verified" and agg["jersey_number"] == claimed_jersey_number:
            verified_track_ids.append(tr.id)
            supporting_sources_by_track[tr.id] = {
                r.source for r in candidate_reads if r.frame_index in agg["evidence_frames"]
            }
            # share of THIS candidate's own reads backing the claimed number -- the tie-break
            # signal below. Deliberately the raw agreement fraction rather than `agg["confidence"]`,
            # which is half made of PARSeq's own self-reported confidence and is uncalibrated on
            # small crops (ADR-21) -- a tie-break must not be decided by the miscalibrated half.
            agreement_by_track[tr.id] = sum(
                1 for r in candidate_reads if r.digits == str(claimed_jersey_number)
            ) / len(candidate_reads)

    if not verified_track_ids:
        evidence["outcome"] = "no_match"
        return None, None, evidence

    if len(verified_track_ids) > 1:
        # More than one track verified as the SAME number, which is physically impossible -- only
        # one player wears it. Measured cause on real footage (video2 take 0): a ByteTrack ID
        # SWITCH, where one track drifts across two different people (track 201 showed a claret
        # player at t=39s and Chelsea's blue #10 at t=50s), so it accumulates a partial set of the
        # real holder's reads. That makes the read counts genuinely lopsided rather than tied.
        #
        # Prefer the clearly-better-supported track when the margin is decisive; otherwise keep the
        # honest "ambiguous -> no box" outcome (Golden Rule 5) rather than guessing between two
        # comparable claims. Both the pick and its margin go into the evidence trail.
        ranked = sorted(verified_track_ids, key=lambda t: (-agreement_by_track[t], t))
        best, runner_up = ranked[0], ranked[1]
        margin = agreement_by_track[best] - agreement_by_track[runner_up]
        evidence["ambiguous_track_ids"] = sorted(verified_track_ids)
        evidence["agreement_by_track"] = {
            t: round(agreement_by_track[t], 3) for t in sorted(verified_track_ids)
        }
        if margin < min_disambiguation_margin:
            evidence["outcome"] = "ambiguous_multiple_candidates_matched"
            evidence["disambiguation_margin"] = round(margin, 3)
            return None, None, evidence
        evidence["outcome"] = "matched_after_disambiguation"
        evidence["disambiguation_margin"] = round(margin, 3)
        evidence["disambiguated_from"] = sorted(verified_track_ids)
        sources = supporting_sources_by_track[best]
        return best, ("mixed" if len(sources) > 1 else next(iter(sources))), evidence

    track_id = verified_track_ids[0]
    sources = supporting_sources_by_track[track_id]
    source = "mixed" if len(sources) > 1 else next(iter(sources))
    evidence["outcome"] = "matched"
    return track_id, source, evidence


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
            # Bug fix 2026-08-31 (see opencv_lab_to_cielab's own docstring): per_track_lab_median
            # returns raw OpenCV-scaled Lab; colour_reference_lab is true CIELAB. Convert HERE,
            # at the point of comparison, never inside src.team.classifier itself.
            lab_by_track_id[tid] = opencv_lab_to_cielab(lab)

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
