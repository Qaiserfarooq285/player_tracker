"""ADR-19 -- manual-events pipeline entrypoint (CLAUDE.md §14.1/§14.3).

Dispatched from `src.pipeline.run.main()` whenever a `<video_basename>.annotations.txt` sidecar
exists next to the input -- checked BEFORE the filename-jersey and ADR-15 branches (§14.1: "the
client's file winning is the trigger, not a quality score"). Detection/tracking still run via the
SAME shared, cached `run_track_stage` machinery every other flow uses -- only EVENT GENERATION
changes: the parsed sidecar is the SOLE, authoritative event source. None of
`src.events.aggregate`'s auto heuristics (touch/pass/possession/tackle/...) or `src.events.goals`'s
scoreboard/goal-region detectors ever run in this mode -- the client already told us what happened.

Identity (jersey number + team colour) comes straight from the annotations too -- there is no
OCR/VLM verification step in this mode at all (a human directly watching the footage IS the
verification, Golden Rule 4). This also means the ADR-15 "skip unverified takes" gate does not
apply here: EVERY take containing at least one annotation gets processed, never silently skipped
for lack of a verified identity (there is nothing to verify in the first place).
"""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from pathlib import Path

from src.annotations.associate import associate_annotation_to_track
from src.annotations.parse import annotation_to_event, parse_annotations
from src.common.io import load_models_parquet, save_json, work_dir_for
from src.common.logging import get_logger
from src.common.types import Annotation, BallDetection, Event, Take, Track
from src.common.video import probe
from src.detect.overlay_mask import ArrowHint
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
    """
    colour_cfg = configs["annotations"]
    sampling_cfg = configs["annotations"]["track_association"]
    decode_cfg = configs["hardware"]["decode"]
    selection_cfg = configs.get("highlights", {}).get("selection")

    identity_by_take: dict[int, TakeIdentityResult] = {}
    debug: dict[str, dict] = {}
    takes_by_id = {t.id: t for t in takes}

    for take_id, anns in annotations_by_take.items():
        take = takes_by_id.get(take_id)
        take_tracks = tracks_by_take.get(take_id, [])
        if take is None or not anns:
            continue

        # Stitched "same physical player" chains for this take, computed once (pure, no I/O).
        # `identity_of` is `{}` whenever `selection_cfg` is unavailable -- see docstring -- and the
        # lock logic below is a no-op in that case (falls through to independent search always).
        identity_of: dict[int, int] = {}
        if selection_cfg:
            identity_of, _identity_confidence = build_take_identities(take_tracks, selection_cfg)

        associated_ids: set[int] = set()
        locked_chain_id_by_jersey: dict[int, int] = {}
        for ann in sorted(anns, key=lambda a: a.t):
            locked_chain_id = locked_chain_id_by_jersey.get(ann.jersey_number) if identity_of else None

            if locked_chain_id is not None:
                # Restricted search: only candidates that are members of the SAME stitched chain
                # the jersey number already locked onto earlier in this take.
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
                if track_id is None:
                    # Honest caption-only: the locked identity has no visible candidate near this
                    # instant -- never fall back to an unrestricted search (that would silently
                    # reopen the exact cross-player swap this fix closes).
                    arrow_evidence = {
                        **arrow_evidence,
                        "agreement": "locked_identity_absent_at_instant",
                    }
                arrow_evidence = {**arrow_evidence, "locked_chain_id": locked_chain_id}
            else:
                # No lock yet for this jersey number in this take -- fresh independent search
                # across every candidate near this instant (the pre-fix behaviour).
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
                if track_id is not None and identity_of:
                    # First confident resolution for this jersey number in this take -- lock its
                    # STITCHED chain id (not the bare raw track id) for every later annotation of
                    # the same jersey number in this take.
                    chain_id = identity_of.get(track_id, track_id)
                    locked_chain_id_by_jersey[ann.jersey_number] = chain_id
                    arrow_evidence = {**arrow_evidence, "locked_chain_id": chain_id}

            debug[f"{take_id}:{ann.t:.2f}"] = {
                "track_id": track_id,
                "distance": distance,
                **arrow_evidence,
            }
            if track_id is not None:
                associated_ids.add(track_id)

        if not associated_ids:
            logger.info(
                "manual mode take=%d: none of its %d annotation(s) could be confidently "
                "associated to a track by colour -- events for this take will render as "
                "captions only, no red box (Golden Rule 5)",
                take_id,
                len(anns),
            )
            continue

        # A take's annotations should all name the same target jersey number in practice; if the
        # client's own lines disagree, the lowest number is used as a deterministic, documented
        # tiebreak (never an arbitrary dict-ordering accident) -- the FULL per-number event split
        # still happens correctly downstream regardless (events_by_number groups by each
        # annotation's OWN jersey_number, not by this take-level identity record).
        jersey_number = min(a.jersey_number for a in anns)
        identity_by_take[take_id] = TakeIdentityResult(
            take_id=take_id,
            jersey_number=jersey_number,
            status="verified",  # ADR-19: human-provided identity needs no further verification
            confidence=1.0,  # a human directly watching the footage, Golden Rule 4
            evidence_frames=[],
            location_method="manual_annotation_colour_match",
            location_track_ids=sorted(associated_ids),
        )

    return identity_by_take, debug


def run_manual_events_pipeline_for_video(
    video_path: str | Path,
    configs: dict[str, dict],
    annotations_path: str | Path,
    work_root: str | Path = "work",
    output_root: str | Path = "output",
    use_nvdec: bool = True,
) -> dict:
    """ADR-19's manual-events entrypoint for ONE video with a `.annotations.txt` sidecar.

    Detection/tracking (Stage 0.5-3) run via the same cached `run_track_stage` every other flow
    uses. Everything after that is replaced: the parsed sidecar becomes every player's own event
    list directly (`annotation_to_event`), grouped by the jersey number each line itself names --
    never by an inferred/stitched identity. Deletes any stale prior output first, same contract as
    `run_extended_pipeline_for_video`.
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

    identity_by_take, association_debug = build_manual_identity_by_take(
        takes,
        dict(tracks_by_take),
        annotations_by_take,
        video_path,
        configs,
        use_nvdec,
        arrow_hints=arrow_hints,
    )

    annotation_report = {
        "video": str(video_path),
        "annotations_path": str(annotations_path),
        "parsed": [a.model_dump(mode="json") for a in annotations],
        "problems": problems,
        "unassigned_annotations": n_unassigned,
        "track_association": association_debug,
    }
    save_json(annotation_report, output_dir / "annotation_report.json")

    events_by_number: dict[int, list[Event]] = defaultdict(list)
    for take_id, anns in annotations_by_take.items():
        for ann in anns:
            events_by_number[ann.jersey_number].append(
                annotation_to_event(ann, configs["annotations"], take_id)
            )

    frame_width, frame_height = _effective_frame_size(video_path, configs["hardware"]["decode"])

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
    )
    logger.info("manual-mode annotated video render -> %s", final_video_path)

    takes_by_id = {t.id: t for t in takes}
    jersey_numbers = sorted(events_by_number.keys())
    for number in jersey_numbers:
        events = events_by_number[number]
        player_dir = output_dir / "players" / f"player_{number}"
        write_player_output(
            player_dir,
            number,
            events,
            possession_seconds=None,  # ADR-19: no possession heuristic runs over annotation-only
            distance_result=None,  # events -- CLAUDE.md §13.2's own "uncertain" rule, not a guess
            takes_by_id=takes_by_id,
            video_path=final_video_path,
            highlights_cfg=configs["highlights"],
            # Bug fix 2026-08-31: `goal_reason=None` unconditionally in manual mode -- the sidecar
            # is the sole, authoritative event source (Golden Rule 4), so a player with zero GOAL
            # annotations genuinely has zero goals, not an "uncertain"/"not available" zero. See
            # `render_statcard_markdown`'s own docstring for how `None` is now interpreted.
            goal_reason=None,
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

    return {
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
