"""ADR-19 -- manual-events pipeline entrypoint (CLAUDE.md §14.1/§14.3).

Dispatched from `src.pipeline.run.main()` whenever a `<video_basename>.annotations.txt` sidecar
exists next to the input -- checked BEFORE the filename-jersey and ADR-15 branches (§14.1: "the
client's file winning is the trigger, not a quality score"). Detection/tracking still run via the
SAME shared, cached `run_track_stage` machinery every other flow uses.

Identity (jersey number + team colour) comes straight from the annotations -- there is no OCR/VLM
verification step in this mode at all (a human directly watching the footage IS the verification,
Golden Rule 4). This also means the ADR-15 "skip unverified takes" gate does not apply here: EVERY
take containing at least one annotation gets processed, never silently skipped for lack of a
verified identity (there is nothing to verify in the first place). `src.events.goals`'s
scoreboard/goal-region detectors still never run in this mode -- a goal is either in the sidecar or
`not available` here, same as before.

**EVENT COUNTING, revised 2026-09-02 (owner decision).** The parsed sidecar's own annotation lines
are preserved verbatim in `annotation_report.json` as ground truth and drive IDENTITY (which raw
tracks belong to which jersey number, via `build_manual_identity_by_take`) -- but the STAT CARD's
touches/passes/turnovers/sprints/shots/dribbles/tackles/saves are now genuinely AUTO-DETECTED,
using the exact same `src.events.aggregate.compute_take_all_events` heuristics `run.py`'s own auto
branch uses, attributed per jersey via `identity_by_take`'s own `jersey_by_track_id`. Before this,
manual mode ran zero auto detection at all, so every touch/pass accuracy fix elsewhere in this
project never reached a sidecar-driven video -- exactly the videos the owner has been checking.
"""

from __future__ import annotations

import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from src.annotations.associate import (
    associate_annotation_to_track,
    candidate_tracks_near_time,
    read_jersey_number_for_candidates,
)
from src.annotations.parse import parse_annotations
from src.common.io import load_models_parquet, save_json, work_dir_for
from src.common.logging import get_logger
from src.common.types import Annotation, BallDetection, Event, EventType, Take, Track
from src.common.video import probe
from src.detect.overlay_mask import ArrowHint
from src.events.aggregate import attribute_events_to_target, compute_take_all_events
from src.events.manual_touches import manual_touch_events, merge_manual_touches, parse_touch_times
from src.events.possession import compute_distance_covered
from src.identity.jersey_models import free_optional_jersey_stack, load_optional_jersey_stack
from src.identity.jersey_ocr import free_easyocr_reader, load_easyocr_reader
from src.identity.verify import TakeIdentityResult
from src.pipeline.annotated_video import render_full_annotated_video
from src.pipeline.player_output import write_player_output
from src.pipeline.profiler import build_run_profile
from src.track.continuity import build_take_identities
from src.track.run import run_track_stage
from src.track.tracker import assign_take_id

logger = get_logger(__name__)


def _effective_frame_size(video_path: str | Path, decode_cfg: dict) -> tuple[float, float]:
    """Mirrors `src.pipeline.run._effective_frame_size` / `src.pipeline.extended_output`'s own
    copy exactly (kept independently duplicated per CLAUDE.md §10's "stages independently
    runnable/debuggable" convention, same as those two)."""
    meta = probe(video_path)
    width, height = meta["width"], meta["height"]
    scale_width = decode_cfg.get("scale_width")
    if scale_width and scale_width < width:
        out_width = scale_width
        out_height = (round(height * (scale_width / width)) // 2) * 2
        return float(out_width), float(out_height)
    return float(width), float(height)


def bucket_annotations_by_take(
    annotations: list[Annotation], takes: list[Take]
) -> tuple[dict[int, list[Annotation]], int]:
    """Assign every parsed annotation to the take containing its own timestamp (pure, no I/O).
    Returns `(annotations_by_take, n_unassigned)` -- an annotation whose `t` falls outside every
    take's `[t_start, t_end)` window is counted but excluded, never silently merged into the
    nearest take (Golden Rule 5: no guessing which take a timestamp belongs to)."""
    by_take: dict[int, list[Annotation]] = defaultdict(list)
    n_unassigned = 0
    for ann in annotations:
        take_id = assign_take_id(ann.t, takes)
        if take_id is None:
            n_unassigned += 1
            continue
        by_take[take_id].append(ann)
    return dict(by_take), n_unassigned


def build_manual_identity_by_take(
    takes: list[Take],
    tracks_by_take: dict[int, list[Track]],
    annotations_by_take: dict[int, list[Annotation]],
    video_path: str | Path,
    configs: dict,
    use_nvdec: bool = True,
    arrow_hints: list[ArrowHint] | None = None,
    gemini_api_key: str | None = None,
) -> tuple[dict[int, TakeIdentityResult], dict[str, dict]]:
    """Best-effort per-take track association (ADR-19): try to associate EACH of a take's own
    annotations to a real track by colour+timing (`src.annotations.associate`), CORROBORATED by
    the burned-in-arrow hint when one exists near that instant (owner request, 2026-08-31 --
    generic to every manual-mode video, not gated on a filename; see
    `src.annotations.associate`'s own module docstring for the combination rule: colour stays
    primary, the arrow is a corroborating/fallback signal, never a silent override). A take with
    at least one confidently-associated annotation gets a `TakeIdentityResult` (red box on the
    UNION of associated track ids across that take's own annotated instants -- best-effort
    continuity, same spirit as CLAUDE.md §13.1's "ID must not visibly reset" rule); a take where
    NONE of its own annotations associate confidently gets no identity at all (green boxes only in
    the renderer) -- the event CAPTIONS (Stage 5) still fire regardless, per ADR-19's own
    "caption-only, no red box" rule.

    `arrow_hints` defaults to `None`/empty -- every existing caller that doesn't pass it gets
    exactly the prior colour-only behaviour (`associate_annotation_to_track` treats an empty/`None`
    `arrow_hints` as "no arrow signal available", not an error).

    Returns `(identity_by_take, association_debug)` -- `association_debug` is a flat
    per-annotation audit trail (`{"<take_id>:<t>": {"track_id", "distance", "arrow_track_id",
    "arrow_distance_px", "agreement"}}`) written verbatim into `annotation_report.json` (Golden
    Rule 5: every association attempt, and whether the arrow corroborated/disagreed/filled in for
    colour, is traceable -- not just the ones that succeeded).

    **Identity lock, bug fix 2026-08-31.** Each annotation used to be associated INDEPENDENTLY --
    a fresh colour(+arrow) search across every candidate track near that instant, with no memory
    of which physical player a PRIOR annotation of the SAME jersey number had already resolved to.
    On real footage many teammates share the exact same kit colour, so two independent searches at
    two different timestamps can (and, confirmed on real data --
    `output/chelsea_burnley_target10/annotation_report.json`, jersey #10 resolved to raw track 208
    at t=57.0s and a DIFFERENT real track 210 at t=60.0s -- did) land on two different physical
    players. Fix: partition this take's own tracks into stitched "same physical player" identity
    chains ONCE (`src.track.continuity.build_take_identities`, already tested, no new stitching
    logic invented here). Per jersey number, the FIRST annotation in the take that resolves
    confidently LOCKS that jersey number to the resolved track's own stitched chain for the rest of
    the take; every LATER annotation of the SAME jersey number is restricted to candidates that are
    members of that SAME chain -- never re-opened to an independent whole-field search, even when a
    closer-by-colour distractor sits nearby. If no locked-chain member has a visible box near a
    later instant, the honest result is caption-only (`track_id=None`,
    `agreement="locked_identity_absent_at_instant"`), never a fallback to an unrestricted search
    (that would silently reopen the exact swap this fix closes). A jersey number whose first
    annotation itself has no confident match establishes no lock yet -- the next annotation of that
    jersey number still gets a fresh independent search. Generic to any number of distinct jersey
    numbers per take, each locking independently. When `selection_cfg` is absent (no
    `configs["highlights"]["selection"]`, e.g. a minimal test config), `build_take_identities` has
    no thresholds to stitch with, so this take falls back to the pre-fix independent-search
    behaviour for every annotation -- exactly as before this fix, never a crash.

    **Jersey-number OCR/VLM re-identification, Stage B ("streamed-gathering-treehouse" plan,
    2026-08-31).** Real testing on broadcast footage found the identity LOCK above is correct but
    locks onto a raw ByteTrack chain with real, large coverage gaps (26.6%/52.0% of a 61.3s take)
    -- dense, visually-similar players fragment the tracker badly, exactly where goals happen. Two
    teammates in identical kit are only distinguishable by the printed number, which colour
    matching structurally cannot read. `src.annotations.associate.read_jersey_number_for_candidates`
    (ADR-15's own OCR-first/VLM-escalation stack, reused verbatim) is now consulted at exactly two
    points, never a per-frame scan:
    (1) **establishing the initial lock** -- alongside colour(+arrow), a confident digit read
    matching the annotation's own claimed jersey number is preferred over/corroborates the colour
    pick (`association_signal="jersey_ocr"|"jersey_vlm"` when it wins, `"colour"` when only the
    colour/arrow test resolved something, `"none"` when neither did);
    (2) **re-acquiring after the locked chain is absent at a later instant** -- before falling back
    to the honest `"locked_identity_absent_at_instant"` result, the search reopens among ALL
    near-time candidate tracks (not just the locked chain), but is accepted ONLY on a confident
    jersey-number match (`association_signal="jersey_ocr_reacquired"|"jersey_vlm_reacquired"`) --
    NEVER on colour proximity alone outside the lock, which would reintroduce the exact
    cross-player swap the lock itself exists to prevent.
    Gated on `configs["identity"]` being present AND `configs["annotations"]["jersey_reid"]
    ["enabled"]` (both absent in older/minimal test configs -- falls back to the pre-Stage-B
    colour(+arrow)-only behaviour unconditionally, never a crash). `gemini_api_key=None` (no
    `GEMINI_API_KEY` set) still allows the cheap, local EasyOCR half of the signal to run; only the
    VLM escalation is skipped. `association_signal` is recorded in `debug` for every annotation
    (Golden Rule 5 -- full audit trail even when the outcome is "none").
    """
    colour_cfg = configs["annotations"]
    sampling_cfg = configs["annotations"]["track_association"]
    decode_cfg = configs["hardware"]["decode"]
    selection_cfg = configs.get("highlights", {}).get("selection")
    identity_cfg = configs.get("identity")
    jersey_reid_cfg = configs["annotations"].get("jersey_reid", {})
    jersey_reid_enabled = bool(identity_cfg) and jersey_reid_cfg.get("enabled", False)
    require_confirmed_jersey = bool(jersey_reid_cfg.get("require_confirmed_jersey", False))

    identity_by_take: dict[int, TakeIdentityResult] = {}
    debug: dict[str, dict] = {}
    takes_by_id = {t.id: t for t in takes}

    # One EasyOCR reader for the whole video (loading it is the expensive part) -- only created
    # when Stage B's jersey-reid signal is actually enabled and there is at least one annotation to
    # process; freed in the `finally` below regardless of how the loop below exits.
    reader = None
    legibility_model = None
    parseq_model = None
    parseq_transform = None
    if jersey_reid_enabled and any(annotations_by_take.values()):
        reader = load_easyocr_reader(identity_cfg["ocr"])
        # Owner-authorized 2026-08-31 (CLAUDE.md §7): same optional, fail-soft legibility ->
        # PARSeq-SoccerNet stack ADR-15's own `src.identity.verify` loads, reused here so
        # manual-mode jersey re-identification (Stage B) gets the same stronger signal ahead of
        # the pre-existing EasyOCR/Gemini chain. Missing/disabled checkpoints degrade this to
        # exactly the prior behaviour (`load_optional_jersey_stack` never raises).
        legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(identity_cfg)

    # Only needed when Stage B's jersey-reid signal is enabled (see the two call sites below) --
    # computed defensively via `.get` so a minimal test config without this key (jersey_reid
    # disabled in that case anyway) never KeyErrors on an unused value.
    tolerance_s = sampling_cfg.get("match_tolerance_s")
    try:
        for take_id, anns in annotations_by_take.items():
            take = takes_by_id.get(take_id)
            take_tracks = tracks_by_take.get(take_id, [])
            if take is None or not anns:
                continue

            # Stitched "same physical player" chains for this take, computed once (pure, no I/O).
            # `identity_of` is `{}` whenever `selection_cfg` is unavailable -- see docstring -- and
            # the lock logic below is a no-op in that case (falls through to independent search
            # always).
            # `motion` left at its default `None` (2026-09-03): manual-mode's own pipeline
            # (`src.pipeline.manual_events`) never estimates a `TakeCameraMotion` -- only
            # `src.pipeline.run`'s auto flow does, for its own `select_targets` call. Adding a
            # fresh estimate here for this call alone would be new per-take decode + optical-flow
            # work this entrypoint doesn't already do, which the fix this comment documents
            # explicitly avoids ("do not add expensive new motion estimation to a call site that
            # doesn't have it"). A real camera pan during manual-mode association therefore still
            # measures in raw image space -- an honest, documented gap, not a silent one.
            identity_of: dict[int, int] = {}
            if selection_cfg:
                identity_of, _identity_confidence = build_take_identities(
                    take_tracks, selection_cfg
                )

            associated_ids: set[int] = set()
            # owner-reported bug fix 2026-09-01: remember WHICH jersey number each associated
            # track belongs to, so a take naming two different target players (an assist/goal
            # pair) labels each box with its own number instead of one take-level number.
            jersey_by_track_id: dict[int, int] = {}
            # owner-reported bug fix 2026-09-02: remember WHEN each associated track was actually
            # the subject of an annotation, so its red box only renders near ITS OWN event(s) --
            # without this, a take naming two different players (assist/goal pair) drew BOTH
            # persistently red for the entire take, including during the OTHER player's own event.
            track_event_times: dict[int, list[float]] = defaultdict(list)
            any_jersey_confirmed = False  # owner-reported bug fix 2026-08-31: True only when AT
            # LEAST ONE of this take's own annotations actually resolved via a real digit read
            # (association_signal starting "jersey_"), never just because SOME track got a
            # colour-only box -- feeds TakeIdentityResult.association_confirmed_by_jersey so the
            # renderer never claims more certainty ("VERIFIED") than the evidence supports.
            locked_chain_id_by_jersey: dict[int, int] = {}
            for ann in sorted(anns, key=lambda a: a.t):
                locked_chain_id = (
                    locked_chain_id_by_jersey.get(ann.jersey_number) if identity_of else None
                )
                association_signal = "none"

                if locked_chain_id is not None:
                    # Restricted search: only candidates that are members of the SAME stitched
                    # chain the jersey number already locked onto earlier in this take.
                    locked_tracks = [
                        tr for tr in take_tracks if identity_of.get(tr.id) == locked_chain_id
                    ]
                    track_id, distance, arrow_evidence = associate_annotation_to_track(
                        video_path,
                        take,
                        locked_tracks,
                        ann,
                        colour_cfg,
                        decode_cfg,
                        sampling_cfg,
                        use_nvdec,
                        arrow_hints=arrow_hints,
                        selection_cfg=selection_cfg,
                    )
                    if track_id is not None:
                        association_signal = "locked_chain"
                    else:
                        # Stage B re-acquisition: before giving up, reopen the search among ALL
                        # near-time candidates (not just the locked chain) -- but accept a result
                        # ONLY on a confident jersey-number match, never on colour proximity alone
                        # outside the lock (that would reintroduce the exact cross-player swap the
                        # lock exists to prevent).
                        reacq_evidence: dict | None = None
                        if jersey_reid_enabled:
                            all_candidates = candidate_tracks_near_time(
                                take_tracks, ann.t, tolerance_s
                            )
                            reacq_id, reacq_source, reacq_evidence = (
                                read_jersey_number_for_candidates(
                                    video_path,
                                    take,
                                    all_candidates,
                                    ann.t,
                                    ann.jersey_number,
                                    identity_cfg,
                                    jersey_reid_cfg,
                                    decode_cfg,
                                    sampling_cfg,
                                    reader,
                                    gemini_api_key,
                                    use_nvdec,
                                    legibility_model=legibility_model,
                                    parseq_model=parseq_model,
                                    parseq_transform=parseq_transform,
                                )
                            )
                            if reacq_id is not None:
                                track_id = reacq_id
                                association_signal = f"jersey_{reacq_source}_reacquired"
                        # Honest caption-only when re-acquisition also found nothing: the locked
                        # identity has no visible/confirmable candidate near this instant.
                        arrow_evidence = {
                            **arrow_evidence,
                            "agreement": (
                                "locked_identity_absent_at_instant"
                                if track_id is None
                                else arrow_evidence.get("agreement")
                            ),
                        }
                        if reacq_evidence is not None:
                            arrow_evidence = {
                                **arrow_evidence,
                                "jersey_reid_evidence": reacq_evidence,
                            }
                    arrow_evidence = {**arrow_evidence, "locked_chain_id": locked_chain_id}
                else:
                    # No lock yet for this jersey number in this take -- fresh independent search
                    # across every candidate near this instant (the pre-fix behaviour), now
                    # corroborated/overridden by a jersey-number read when Stage B is enabled.
                    track_id, distance, arrow_evidence = associate_annotation_to_track(
                        video_path,
                        take,
                        take_tracks,
                        ann,
                        colour_cfg,
                        decode_cfg,
                        sampling_cfg,
                        use_nvdec,
                        arrow_hints=arrow_hints,
                        selection_cfg=selection_cfg,
                    )
                    association_signal = "colour" if track_id is not None else "none"

                    if jersey_reid_enabled:
                        candidates = candidate_tracks_near_time(take_tracks, ann.t, tolerance_s)
                        jersey_id, jersey_source, jersey_evidence = (
                            read_jersey_number_for_candidates(
                                video_path,
                                take,
                                candidates,
                                ann.t,
                                ann.jersey_number,
                                identity_cfg,
                                jersey_reid_cfg,
                                decode_cfg,
                                sampling_cfg,
                                reader,
                                gemini_api_key,
                                use_nvdec,
                                # bug fix 2026-08-31: give colour's OWN pick first claim on the
                                # shared VLM budget -- real broadcast data showed a crowded field
                                # of near-time candidates could exhaust the budget on unrelated
                                # players before ever reaching the one colour already selected.
                                priority_track_id=track_id,
                                legibility_model=legibility_model,
                                parseq_model=parseq_model,
                                parseq_transform=parseq_transform,
                            )
                        )
                        arrow_evidence = {
                            **arrow_evidence,
                            "jersey_reid_evidence": jersey_evidence,
                        }
                        if jersey_id is not None:
                            # A confident digit read matching the annotation's own claimed number
                            # is the strongest available signal -- it wins over (and, when the two
                            # agree, corroborates) the colour/arrow pick. Any disagreement is
                            # logged, never silently dropped (Golden Rule 5).
                            if track_id is not None and track_id != jersey_id:
                                arrow_evidence = {
                                    **arrow_evidence,
                                    "jersey_colour_disagreement": {
                                        "colour_track_id": track_id,
                                        "jersey_track_id": jersey_id,
                                    },
                                }
                            track_id = jersey_id
                            association_signal = f"jersey_{jersey_source}"
                        elif require_confirmed_jersey:
                            # A kit colour only narrows the field to several teammates; it cannot
                            # establish the claimed printed number.  Keeping the colour-selected
                            # id here used to render a confident-looking TARGET box on whichever
                            # teammate happened to be closest in Lab space. Preserve that useful
                            # candidate evidence, but render this event caption-only until a
                            # temporally-agreeing jersey read actually identifies the player.
                            if track_id is not None:
                                arrow_evidence = {
                                    **arrow_evidence,
                                    "unverified_colour_candidate_track_id": track_id,
                                }
                            track_id = None
                            distance = None
                            association_signal = "jersey_unverified"

                    if track_id is not None and identity_of:
                        # First confident resolution for this jersey number in this take -- lock
                        # its STITCHED chain id (not the bare raw track id) for every later
                        # annotation of the same jersey number in this take.
                        chain_id = identity_of.get(track_id, track_id)
                        locked_chain_id_by_jersey[ann.jersey_number] = chain_id
                        arrow_evidence = {**arrow_evidence, "locked_chain_id": chain_id}

                debug[f"{take_id}:{ann.t:.2f}"] = {
                    "track_id": track_id,
                    "distance": distance,
                    "association_signal": association_signal,
                    **arrow_evidence,
                }
                if track_id is not None:
                    associated_ids.add(track_id)
                    jersey_by_track_id[track_id] = ann.jersey_number
                    track_event_times[track_id].append(ann.t)
                if association_signal.startswith("jersey_"):
                    any_jersey_confirmed = True

            if not associated_ids:
                logger.info(
                    "manual mode take=%d: none of its %d annotation(s) could be confidently "
                    "associated to the claimed jersey -- events for this take will render as "
                    "captions only, no red box (Golden Rule 5)",
                    take_id,
                    len(anns),
                )
                continue

            # A take's annotations should all name the same target jersey number in practice; if
            # the client's own lines disagree, the lowest number is used as a deterministic,
            # documented tiebreak (never an arbitrary dict-ordering accident) -- the FULL
            # per-number event split still happens correctly downstream regardless
            # (events_by_number groups by each annotation's OWN jersey_number, not by this
            # take-level identity record).
            jersey_number = min(a.jersey_number for a in anns)
            pre_roll_s = configs["highlights"]["clip"]["pre_roll_s"]
            post_roll_s = configs["highlights"]["clip"]["post_roll_s"]
            track_active_windows = {
                track_id: [(t - pre_roll_s, t + post_roll_s) for t in times]
                for track_id, times in track_event_times.items()
            }
            identity_by_take[take_id] = TakeIdentityResult(
                take_id=take_id,
                jersey_number=jersey_number,
                status="verified",  # ADR-19: human-provided identity needs no further verification
                confidence=1.0,  # a human directly watching the footage, Golden Rule 4
                evidence_frames=[],
                location_method="manual_annotation_colour_match",
                location_track_ids=sorted(associated_ids),
                jersey_by_track_id=jersey_by_track_id,
                track_active_windows=track_active_windows,
                association_confirmed_by_jersey=any_jersey_confirmed,
            )
    finally:
        if reader is not None:
            free_easyocr_reader(reader)
        free_optional_jersey_stack(legibility_model, parseq_model)

    return identity_by_take, debug


def run_manual_events_pipeline_for_video(
    video_path: str | Path,
    configs: dict[str, dict],
    annotations_path: str | Path,
    target_jersey: int | None = None,
    work_root: str | Path = "work",
    output_root: str | Path = "output",
    use_nvdec: bool = True,
    manual_touch_times: str | None = None,
) -> dict:
    """ADR-19's manual-events entrypoint for ONE video with a `.annotations.txt` sidecar.

    Detection/tracking (Stage 0.5-3) run via the same cached `run_track_stage` every other flow
    uses. Everything after that is replaced: the parsed sidecar becomes every player's own event
    list directly (`annotation_to_event`). When `target_jersey` is supplied (the web UI's jersey
    field), it is the ONE primary player: only annotations for that number are allowed to establish
    a red target track or receive a player stat card. Annotations naming other players remain in
    the video as event captions, but can never replace the requested player as the target.
    Deletes any stale prior output first, same contract as `run_extended_pipeline_for_video`.

    `manual_touch_times` (Plan Stage 2, "streamed-gathering-treehouse", wired here for parity with
    `src.pipeline.run.run_pipeline_for_video`): an optional client-typed, comma-/newline-separated
    list of touch times, merged into `target_jersey`'s own event list only -- when `target_jersey`
    is `None` this mode has no single well-defined "the target player" (a sidecar can name several
    jersey numbers, see `jersey_numbers` below), so touch times are honestly left unmerged rather
    than guessed onto an arbitrary one (Golden Rule 5).
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    output_dir = Path(output_root) / work_dir.name

    if output_dir.exists():
        logger.info("wiping stale output at %s before writing this manual-mode run", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations, problems = parse_annotations(annotations_path, configs["annotations"])
    if problems:
        logger.warning(
            "manual-events mode: %d line(s) in %s could not be parsed -- see "
            "annotation_report.json for the full list (never silently dropped, CLAUDE.md §10)",
            len(problems),
            annotations_path,
        )

    t0 = time.time()
    profile = build_run_profile(
        video_path,
        configs["hardware"],
        configs["profile"],
        configs["shots"],
        work_root=work_root,
        use_nvdec=use_nvdec,
    )
    logger.info("profile: %s (%.1fs)", profile.profile.value, time.time() - t0)

    t0 = time.time()
    track_summary = run_track_stage(
        video_path,
        configs["hardware"],
        configs["detect"],
        configs["track"],
        configs["team"],
        configs["shots"],
        work_root=work_root,
        use_nvdec=use_nvdec,
    )
    logger.info(
        "detect+track+team: n_takes=%s n_tracks=%s (%.1fs)",
        track_summary.get("n_takes"),
        track_summary.get("n_tracks"),
        time.time() - t0,
    )

    takes = load_models_parquet(work_dir / "shots" / "takes.parquet", Take)
    tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    balls = load_models_parquet(work_dir / "detect" / "ball_detections.parquet", BallDetection)
    arrow_path = work_dir / "detect" / "arrow_hints.parquet"
    arrow_hints = load_models_parquet(arrow_path, ArrowHint) if arrow_path.exists() else []

    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    annotations_by_take, n_unassigned = bucket_annotations_by_take(annotations, takes)
    if n_unassigned:
        logger.warning(
            "%d annotation(s) fell outside every take's [t_start, t_end) window and were "
            "excluded from event generation",
            n_unassigned,
        )

    # Stage B ("streamed-gathering-treehouse" plan): jersey-number OCR/VLM re-identification.
    # Same env-var convention as src/pipeline/run.py / extended_output.py -- absent key still
    # allows the cheap, local EasyOCR half of the signal to run, only VLM escalation is skipped.
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or None
    jersey_reid_cfg = configs.get("annotations", {}).get("jersey_reid", {})
    if not gemini_api_key and configs.get("identity") and jersey_reid_cfg.get("enabled", False):
        logger.warning(
            "GEMINI_API_KEY not set -- jersey-number re-identification (Stage B) proceeds on "
            "EasyOCR alone (no VLM escalation for OCR-ambiguous crops)"
        )

    # The user-selected jersey is the primary identity.  For example, "#10 assists #2" must
    # never cause #2's goal annotation to become the red TARGET player merely because it is the
    # last or lowest-numbered annotation in the take.
    identity_annotations_by_take = annotations_by_take
    if target_jersey is not None:
        identity_annotations_by_take = {
            take_id: [ann for ann in anns if ann.jersey_number == target_jersey]
            for take_id, anns in annotations_by_take.items()
        }
        identity_annotations_by_take = {
            take_id: anns for take_id, anns in identity_annotations_by_take.items() if anns
        }

    identity_by_take, association_debug = build_manual_identity_by_take(
        takes,
        dict(tracks_by_take),
        identity_annotations_by_take,
        video_path,
        configs,
        use_nvdec,
        arrow_hints=arrow_hints,
        gemini_api_key=gemini_api_key,
    )

    annotation_report = {
        "video": str(video_path),
        "annotations_path": str(annotations_path),
        "target_jersey": target_jersey,
        "parsed": [a.model_dump(mode="json") for a in annotations],
        "problems": problems,
        "unassigned_annotations": n_unassigned,
        "track_association": association_debug,
    }
    save_json(annotation_report, output_dir / "annotation_report.json")

    # Owner decision, 2026-09-02: for a video with a sidecar, the STAT CARD reports
    # AUTO-DETECTED events only -- the sidecar stays the AUTHORITATIVE source for IDENTITY (who is
    # #10/#2, already resolved above into `identity_by_take`) and is preserved verbatim in
    # `annotation_report.json` as ground truth, but touches/passes/turnovers/sprints/shots/
    # dribbles/tackles/saves are now genuinely measured by the SAME Stage 4/4.9 heuristics
    # `run.py`'s own auto branch uses (`compute_take_all_events`), not re-derived from the
    # sidecar's own coarse phrase-mapped lines. Before this, manual mode ran ZERO auto detection
    # at all -- every touch/pass/sprint fix landed elsewhere this session never applied to a
    # sidecar-driven video, which is exactly the video the owner has been checking.
    frame_width, frame_height = _effective_frame_size(video_path, configs["hardware"]["decode"])
    events_cfg = configs["events"]
    selection_cfg = (configs.get("highlights") or {}).get("selection", {})
    attribution_cfg = (configs.get("highlights") or {}).get("attribution", {})
    fps_sample = configs["hardware"]["stages"]["track"]["fps_sample"]

    balls_by_take: dict[int, list[BallDetection]] = defaultdict(list)
    for b in balls:
        b_take_id = assign_take_id(b.t, takes)
        if b_take_id is not None:
            balls_by_take[b_take_id].append(b)

    events_by_number: dict[int, list[Event]] = defaultdict(list)
    takes_used_by_number: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for take in takes:
        identity = identity_by_take.get(take.id)
        if identity is None or not identity.jersey_by_track_id:
            continue  # no confidently-associated jersey in this take -- nothing to attribute to
        take_tracks = tracks_by_take.get(take.id, [])
        take_balls = balls_by_take.get(take.id, [])
        all_events, auto_identity_of, _identity_conf = compute_take_all_events(
            take, take_tracks, take_balls, frame_width, events_cfg, selection_cfg, fps_sample
        )
        # this take can name more than one jersey (an assist/goal pair) -- attribute its own
        # auto-detected events to EACH jersey separately, using that jersey's own associated
        # raw track ids from `jersey_by_track_id` (never the take-wide union).
        raw_ids_by_jersey: dict[int, list[int]] = defaultdict(list)
        for raw_id, jersey in identity.jersey_by_track_id.items():
            raw_ids_by_jersey[jersey].append(raw_id)
        for jersey, raw_ids in raw_ids_by_jersey.items():
            attributed = attribute_events_to_target(
                all_events, take_tracks, take_balls, raw_ids, auto_identity_of, attribution_cfg
            )
            events_by_number[jersey].extend(attributed)
            takes_used_by_number[jersey].append((take.id, raw_ids))

    final_video_path = output_dir / "original_annotated_video.mp4"
    render_full_annotated_video(
        video_path,
        work_dir,
        final_video_path,
        takes,
        tracks,
        balls,
        identity_by_take,
        dict(events_by_number),
        frame_width,
        frame_height,
        use_nvdec=use_nvdec,
        # `.get`, not `configs["goal_region"]`: manual mode never touches goal detection itself
        # (ADR-19 -- the sidecar is the sole event source), so unlike run.py/extended_output.py it
        # has no pre-existing hard dependency on this key being present in every caller's configs
        # dict (e.g. a minimal test fixture) -- `render_full_annotated_video` already treats a
        # missing/`None` goal_region_cfg as "no polygon configured", the correct default anyway.
        goal_region_cfg=configs.get("goal_region"),
        # `.get` chain for the same reason as `goal_region` above: a minimal test fixture's
        # configs dict need not carry every stage's block, and the renderer already treats a
        # missing/`None` selection_cfg as "don't stitch display identities", falling back to raw
        # track ids -- the pre-existing behaviour, not a failure.
        selection_cfg=(configs.get("highlights") or {}).get("selection"),
    )
    logger.info("manual-mode annotated video render -> %s", final_video_path)

    # Plan Stage 2 ("streamed-gathering-treehouse"): optional client-typed touch times, wired here
    # for parity with `src.pipeline.run.run_pipeline_for_video` -- parsed ONCE for the whole run
    # (see this function's own docstring for why merging is scoped to `target_jersey` only).
    manual_touch_cfg = configs["events"]["manual_touch"]
    manual_touch_problems: list[str] = []
    manual_touch_evts: list[Event] = []
    if manual_touch_times:
        touch_seconds, manual_touch_problems = parse_touch_times(
            manual_touch_times, manual_touch_cfg
        )
        manual_touch_evts = manual_touch_events(touch_seconds, manual_touch_cfg)
        if manual_touch_problems:
            logger.warning(
                "manual touch times for %s: %d unparseable entr(ies): %s",
                video_path.name,
                len(manual_touch_problems),
                manual_touch_problems,
            )
    manual_touch_suppressed = 0

    takes_by_id = {t.id: t for t in takes}
    jersey_numbers = (
        [target_jersey] if target_jersey is not None else sorted(events_by_number.keys())
    )
    for number in jersey_numbers:
        events = events_by_number.get(number, [])
        if manual_touch_evts and number == target_jersey:
            # Owner's merge rule ("their times win, auto-detection fills the gaps") -- see
            # `src.pipeline.run.run_pipeline_for_video`'s identical splice for the full reasoning.
            events, n_suppressed = merge_manual_touches(events, manual_touch_evts, manual_touch_cfg)
            events_by_number[number] = events
            manual_touch_suppressed += n_suppressed
        # Owner decision 2026-09-02: possession/distance are now genuinely measured the same way
        # run.py's own auto branch computes them, since events are auto-detected (see above) --
        # no longer a hardcoded None/"uncertain" now that there is a real heuristic behind them.
        possession_seconds = sum(
            ev.t_end - ev.t_start for ev in events if ev.type == EventType.POSSESSION
        )
        total_distance = total_samples = total_fragments = 0.0
        for take_id, raw_ids in takes_used_by_number.get(number, []):
            result = compute_distance_covered(raw_ids, tracks_by_take.get(take_id, []), events_cfg)
            total_distance += result["distance"]
            total_samples += result["n_speed_samples"]
            total_fragments += result["n_track_fragments"]
        distance_result = {
            "distance": total_distance,
            "unit": "bbox_heights",
            "calibrated": False,
            "n_track_fragments": total_fragments,
            "n_speed_samples": total_samples,
        }
        player_dir = output_dir / "players" / f"player_{number}"
        write_player_output(
            player_dir,
            number,
            events,
            possession_seconds=possession_seconds,
            distance_result=distance_result,
            takes_by_id=takes_by_id,
            video_path=final_video_path,
            highlights_cfg=configs["highlights"],
            # Revised 2026-09-02 (owner decision, see the module docstring's own "EVENT COUNTING"
            # section): events are now AUTO-DETECTED, not read off the sidecar, so a `None` here
            # would be WRONG -- it would claim "the event source is authoritative, this zero is
            # certain" when in fact goal-line geometry (Stage 6) doesn't exist yet and auto
            # detection genuinely cannot see a goal even when the sidecar itself records one (the
            # owner's own stated expectation when choosing this mode: "your assist/goal would only
            # appear if the auto detectors independently find them -- they currently cannot").
            # A non-None reason here means `render_statcard_markdown` shows "not available"
            # instead of a confident 0, honestly matching goals/assists everywhere else in this
            # project (no scoreboard, no marked goal region -- see `src/events/goals.py`).
            goal_reason=(
                "not available (auto goal-line detection is not yet implemented; this video's "
                "own annotations may record a goal, but manual-mode stats are now auto-detected "
                "per the owner's own choice, so an unconfirmed auto-goal reads as uncertain, "
                "never a guessed 0 or a silently-adopted sidecar count)"
            ),
            identity_status="Human-provided (manual annotation)",
        )
        logger.info(
            "player_%d (manual annotation): %d event(s) -- statcard/highlights written",
            number,
            len(events),
        )

    if not jersey_numbers:
        logger.warning(
            "manual-events mode: the sidecar for %s produced zero usable annotations -- no "
            "players/ folder written, see annotation_report.json for why",
            video_path.name,
        )

    summary = {
        "video": str(video_path),
        "mode": "manual_events",
        "n_takes": len(takes),
        "n_annotations_parsed": len(annotations),
        "n_problems": len(problems),
        "jersey_numbers": jersey_numbers,
        "annotation_report_path": str(output_dir / "annotation_report.json"),
        "annotated_video_path": str(final_video_path),
        "player_dirs": [str(output_dir / "players" / f"player_{n}") for n in jersey_numbers],
    }
    if manual_touch_times:
        summary["manual_touch"] = {
            "raw": manual_touch_times,
            "n_parsed": len(manual_touch_evts),
            "problems": manual_touch_problems,
            "n_suppressed_auto_touch": manual_touch_suppressed,
        }
    return summary
