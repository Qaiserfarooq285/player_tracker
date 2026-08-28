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
from src.identity.verify import TakeIdentityResult
from src.pipeline.annotated_video import render_full_annotated_video
from src.pipeline.player_output import write_player_output
from src.pipeline.profiler import build_run_profile
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
) -> tuple[dict[int, TakeIdentityResult], dict[str, dict]]:
    """Best-effort per-take track association (ADR-19): try to associate EACH of a take's own
    annotations to a real track by colour+timing (`src.annotations.associate`). A take with at
    least one confidently-associated annotation gets a `TakeIdentityResult` (red box on the UNION
    of associated track ids across that take's own annotated instants -- best-effort continuity,
    same spirit as CLAUDE.md §13.1's "ID must not visibly reset" rule); a take where NONE of its
    own annotations associate confidently gets no identity at all (green boxes only in the
    renderer) -- the event CAPTIONS (Stage 5) still fire regardless, per ADR-19's own
    "caption-only, no red box" rule.

    Returns `(identity_by_take, association_debug)` -- `association_debug` is a flat
    per-annotation audit trail (`{"<take_id>:<t>": {"track_id", "distance"}}`) written verbatim
    into `annotation_report.json` (Golden Rule 5: every association attempt is traceable, not just
    the ones that succeeded).
    """
    colour_cfg = configs["annotations"]
    sampling_cfg = configs["annotations"]["track_association"]
    decode_cfg = configs["hardware"]["decode"]

    identity_by_take: dict[int, TakeIdentityResult] = {}
    debug: dict[str, dict] = {}
    takes_by_id = {t.id: t for t in takes}

    for take_id, anns in annotations_by_take.items():
        take = takes_by_id.get(take_id)
        take_tracks = tracks_by_take.get(take_id, [])
        if take is None or not anns:
            continue

        associated_ids: set[int] = set()
        for ann in anns:
            track_id, distance = associate_annotation_to_track(
                video_path, take, take_tracks, ann, colour_cfg, decode_cfg, sampling_cfg, use_nvdec
            )
            debug[f"{take_id}:{ann.t:.2f}"] = {"track_id": track_id, "distance": distance}
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
        takes, dict(tracks_by_take), annotations_by_take, video_path, configs, use_nvdec
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
        goal_count = sum(1 for ev in events if ev.type.value == "goal")
        goal_reason = (
            None
            if goal_count > 0
            else (
                "not available (no GOAL annotation was present in the manual-annotation sidecar "
                "for this player -- ADR-19: manual mode never runs the scoreboard/goal-region "
                "auto-detectors, the client's own sidecar is the sole authoritative event source)"
            )
        )
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
            goal_reason=goal_reason,
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
