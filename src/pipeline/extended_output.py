"""ADR-15 — extended pipeline entrypoint for filename-less input (CLAUDE.md §3.3/§13).

Sibling to `src.pipeline.run.run_pipeline_for_video`, which stays completely unmodified and is
the ONLY entrypoint ever used for the original 5 `clip<N> <jersey>.mp4` clips. This module is
dispatched from `src.pipeline.run.main()` whenever a discovered video's `target_jersey is None`.

Runs Stage 0.5-3 with the exact same shared, cached machinery (`run_track_stage`) as the original
flow, then:

- **Stage 5.5** identity verification (`src.identity.verify`) — per-take verified jersey number.
- **Stage 6-ext** the FULL best-effort event suite (`src.events.aggregate`), attributed to
  whichever verified jersey number each take belongs to, PLUS a cheap-prefilter/Gemini
  celebration-or-key-moment pass (`src.events.key_moments`) on each verified take's own stitched
  timeline.
- **Stage 6-ext** per-player output (`src.pipeline.player_output`): `statcard.md` (owner's exact
  §13.2 template), category highlight reels, `event_timeline.json` — one folder per DISTINCT
  verified jersey number (there may be one, several, or zero — never forced).
- `identity_report.json` — every take's identity result, verified or not (CLAUDE.md §13.5: makes
  an absent `player_<N>` folder always explainable).
- **§13.1** the full-resolution annotated original video (`src.pipeline.annotated_video`).

Distance-covered note (Golden Rule 3): raw `Track.id`s reset per take, so a jersey number verified
in MULTIPLE, non-contiguous takes must have its distance computed ONE TAKE AT A TIME (each take's
own id-space is self-consistent) and then summed — never by pooling raw ids from different takes
into a single lookup, which would silently collide two different takes' same-numbered fragments.
"""

from __future__ import annotations

import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.common.io import StageCache, load_json, load_models_parquet, save_json, work_dir_for
from src.common.logging import DropCounter, get_logger
from src.common.types import BallDetection, Event, EventType, Take, Track
from src.common.video import probe
from src.detect.overlay_mask import ArrowHint
from src.events.aggregate import (
    attribute_events_to_target,
    compute_take_all_events,
    target_identity_id,
)
from src.events.goals import detect_goals_for_video
from src.events.key_moments import classify_candidate_windows, find_candidate_windows
from src.events.possession import compute_distance_covered
from src.identity.verify import IdentityReport, verify_identities_for_video
from src.pipeline.annotated_video import render_full_annotated_video
from src.pipeline.player_output import write_player_output
from src.pipeline.profiler import build_run_profile
from src.track.continuity import build_take_identities
from src.track.run import run_track_stage
from src.track.tracker import assign_take_id

logger = get_logger(__name__)


def _effective_frame_size(video_path: str | Path, decode_cfg: dict) -> tuple[float, float]:
    """Mirrors `src.pipeline.run._effective_frame_size` exactly (kept independently per CLAUDE.md
    §10 "stages independently runnable/debuggable" rather than importing a private helper across
    the sibling-module boundary)."""
    meta = probe(video_path)
    width, height = meta["width"], meta["height"]
    scale_width = decode_cfg.get("scale_width")
    if scale_width and scale_width < width:
        out_width = scale_width
        out_height = (round(height * (scale_width / width)) // 2) * 2
        return float(out_width), float(out_height)
    return float(width), float(height)


def _stitched_virtual_track(take_id: int, take_tracks: list[Track], ids: list[int]) -> Track:
    """Merge the boxes of several raw take-scoped fragments (one verified target's own
    location-selection `track_ids`) into ONE synthetic `Track`, time-sorted — needed because
    `src.events.key_moments.find_candidate_windows` (like `sprints.track_speed_series`) operates
    on a single `Track`'s own box sequence, not a list of fragments."""
    by_id = {tr.id: tr for tr in take_tracks}
    boxes = []
    for tid in ids:
        tr = by_id.get(tid)
        if tr is not None:
            boxes.extend(tr.boxes)
    boxes.sort(key=lambda b: b.t)
    return Track(id=-1, take_id=take_id, boxes=boxes)


def run_extended_pipeline_for_video(
    video_path: str | Path,
    configs: dict[str, dict],
    work_root: str | Path = "work",
    output_root: str | Path = "output",
    use_nvdec: bool = True,
    human_confirmed_jersey: int | None = None,
) -> dict:
    """Run the full ADR-15 extended pipeline for one filename-less video end to end. Deletes any
    stale prior output at `output/<slug>/` first (CLAUDE.md task spec: never mix a new run with a
    previous partial one).

    `human_confirmed_jersey` (ADR-18 (1)) is `src/pipeline/run.py::main()`'s optional
    `--target-jersey` human-in-the-loop override, threaded straight into Stage 5.5's identity
    verification (`src.identity.verify.verify_identities_for_video`) -- see that module's own
    docstring for the exact semantics.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    output_dir = Path(output_root) / work_dir.name

    if output_dir.exists():
        logger.info("wiping stale output at %s before writing this run", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or None
    if not gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY not set -- identity verification proceeds on EasyOCR alone (no VLM "
            "escalation) and zero key-moment events will ever be emitted. Both are honest "
            "degraded modes (Golden Rule 5), never a guess."
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

    frame_width, frame_height = _effective_frame_size(video_path, configs["hardware"]["decode"])
    fps_sample = configs["hardware"]["stages"]["track"]["fps_sample"]

    # --- Stage 5.5: identity verification (cached) ---------------------------------------------
    identity_path = work_dir / "identity.json"
    identity_cache_config = {
        "identity_cfg": configs["identity"],
        "selection_cfg": configs["highlights"]["selection"],
        "n_tracks": len(tracks),
        "n_takes": len(takes),
        "n_arrow_hints": len(arrow_hints),
        # ADR-18 (1): part of the cache key -- changing (or adding/removing) the human-confirmed
        # override between runs must invalidate a stale cached identity result, never silently
        # reuse a pre-override verification.
        "human_confirmed_jersey": human_confirmed_jersey,
    }
    identity_cache = StageCache(identity_path, identity_cache_config, stage="identity")
    if identity_cache.hit():
        identity_report = IdentityReport.model_validate(load_json(identity_path))
    else:
        t0 = time.time()
        identity_report = verify_identities_for_video(
            video_path,
            takes,
            tracks,
            arrow_hints,
            frame_width,
            frame_height,
            configs["highlights"]["selection"],
            configs["identity"],
            gemini_api_key,
            use_nvdec=use_nvdec,
            human_confirmed_jersey=human_confirmed_jersey,
        )
        save_json(identity_report, identity_path)
        identity_cache.write_meta()
        logger.info("identity verification: %.1fs", time.time() - t0)

    identity_by_take = {t.take_id: t for t in identity_report.takes}
    save_json(identity_report, output_dir / "identity_report.json")

    verified_numbers = sorted(
        {t.jersey_number for t in identity_report.takes if t.status == "verified"}
    )
    logger.info(
        "identity: %d/%d take(s) verified -> distinct jersey number(s): %s",
        sum(1 for t in identity_report.takes if t.status == "verified"),
        len(identity_report.takes),
        verified_numbers,
    )

    # --- Stage 6-ext: full event suite per take, attributed per verified jersey number ----------
    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    balls_by_take: dict[int, list[BallDetection]] = defaultdict(list)
    for b in balls:
        take_id = assign_take_id(b.t, takes)
        if take_id is not None:
            balls_by_take[take_id].append(b)

    events_cfg = configs["events"]
    selection_cfg = configs["highlights"]["selection"]
    attribution_cfg = configs["highlights"]["attribution"]
    key_moment_cfg = configs["key_moments"]

    # --- ADR-17: goals + assists, once for the whole video ------------------------------------
    # `identity_of_by_take` reuses the SAME `build_take_identities` partition
    # `compute_take_all_events` computes per-take below (cheap, pure arithmetic on tracks -- no
    # OCR/decode) so goal/assist attribution lands in the identical identity id-space every other
    # extended-pipeline event already uses, rather than raw track ids.
    t0 = time.time()
    identity_of_by_take: dict[int, dict[int, int]] = {
        take.id: build_take_identities(tracks_by_take.get(take.id, []), selection_cfg)[0]
        for take in takes
    }
    goal_result = detect_goals_for_video(
        video_path,
        takes,
        dict(tracks_by_take),
        dict(balls_by_take),
        events_cfg,
        identity_of_by_take=identity_of_by_take,
        ocr_cfg=configs["shots"]["ocr"],
        use_nvdec=use_nvdec,
    )
    logger.info("goal detection: %s (%.1fs)", goal_result.reason, time.time() - t0)
    goal_assist_by_take: dict[int, list[Event]] = defaultdict(list)
    for ev in goal_result.events:
        if ev.take_id is not None:
            goal_assist_by_take[ev.take_id].append(ev)

    drops = DropCounter("extended_events")
    events_by_number: dict[int, list[Event]] = defaultdict(list)
    takes_used_by_number: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)

    for take in takes:
        tir = identity_by_take.get(take.id)
        if tir is None or tir.status != "verified" or tir.jersey_number is None:
            continue
        take_tracks = tracks_by_take.get(take.id, [])
        take_balls = balls_by_take.get(take.id, [])
        if not take_tracks:
            continue

        all_events, identity_of, _identity_confidence = compute_take_all_events(
            take,
            take_tracks,
            take_balls,
            frame_width,
            events_cfg,
            selection_cfg,
            fps_sample,
            drops=drops,
        )
        all_events = all_events + goal_assist_by_take.get(take.id, [])
        target_events = attribute_events_to_target(
            all_events,
            take_tracks,
            take_balls,
            tir.location_track_ids,
            identity_of,
            attribution_cfg,
        )

        # ADR-14: celebration/key-moment pre-filter + Gemini, on this take's own stitched target
        # timeline, excluding windows already explained by a sprint/dribble/possession event.
        virtual_track = _stitched_virtual_track(take.id, take_tracks, tir.location_track_ids)
        existing_windows = [
            (ev.t_start, ev.t_end)
            for ev in target_events
            if ev.type in (EventType.SPRINT, EventType.DRIBBLE, EventType.POSSESSION)
        ]
        candidates = find_candidate_windows(
            virtual_track, key_moment_cfg["prefilter"], existing_windows
        )
        # Event.player_track_id is a TRACK/IDENTITY id, never a jersey number (ADR-15: the two id
        # spaces must never be conflated) -- resolve the target's own identity id the same way
        # attribute_events_to_target does, falling back to its first raw location track id (still
        # a real, traceable track id) on the rare miss where the identity partition disagrees.
        key_moment_player_id = target_identity_id(tir.location_track_ids, identity_of)
        if key_moment_player_id is None and tir.location_track_ids:
            key_moment_player_id = tir.location_track_ids[0]
        key_events = classify_candidate_windows(
            video_path,
            take.id,
            key_moment_player_id,
            candidates,
            gemini_api_key,
            key_moment_cfg["vlm"],
            use_nvdec=use_nvdec,
        )
        target_events = target_events + key_events

        events_by_number[tir.jersey_number].extend(target_events)
        takes_used_by_number[tir.jersey_number].append((take.id, tir.location_track_ids))

    dropped = drops.report()
    if dropped:
        logger.info("extended-event drops: %s", dropped)

    # --- §13.1: full-resolution annotated original video ----------------------------------------
    # ADR-18 (2): rendered BEFORE the per-player loop below (not after, as originally wired) so
    # that `write_player_output` -> `_cut_category_reel` -> `cut_clips` cuts each player's
    # highlight clips from THIS annotated video (red/green boxes, ball marker, live panel, CUT
    # banners already burned in) rather than from the raw, unannotated source -- CLAUDE.md §13.4
    # is explicit that highlight clips must show "the same red/green boxes + ball tracking +
    # captions as the main video". Total render cost is unchanged (same work, just reordered);
    # `cut_clips`/ffmpeg cutting a pre-rendered mp4 works identically to cutting the raw source
    # (it is still just a video file on disk).
    t0 = time.time()
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
    )
    logger.info("annotated video render: %.1fs -> %s", time.time() - t0, final_video_path)

    # --- Stage 6-ext: per-player output -----------------------------------------------------------
    takes_by_id = {t.id: t for t in takes}
    for number in verified_numbers:
        events = events_by_number.get(number, [])
        possession_seconds = sum(
            ev.t_end - ev.t_start for ev in events if ev.type == EventType.POSSESSION
        )
        total_distance = 0.0
        total_samples = 0
        total_fragments = 0
        for take_id, ids in takes_used_by_number.get(number, []):
            result = compute_distance_covered(ids, tracks_by_take.get(take_id, []), events_cfg)
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
            possession_seconds,
            distance_result,
            takes_by_id,
            # ADR-18 (2): the ANNOTATED video (rendered above), not the raw `video_path` -- every
            # category highlight clip must show the same red/green boxes + ball marker + live
            # panel as the primary output (CLAUDE.md §13.4), which cutting from the raw source
            # would silently fail to do.
            final_video_path,
            configs["highlights"],
            goal_result.reason,
        )
        logger.info(
            "player_%d: %d event(s) across %d take(s) -- statcard/highlights written",
            number,
            len(events),
            len(takes_used_by_number.get(number, [])),
        )

    if not verified_numbers:
        logger.warning(
            "no jersey number ever verified across %d take(s) of %s -- no players/ folder "
            "written, per-player output honestly reflects nothing was confirmed",
            len(takes),
            video_path.name,
        )

    return {
        "video": str(video_path),
        "n_takes": len(takes),
        "verified_numbers": verified_numbers,
        "identity_report_path": str(output_dir / "identity_report.json"),
        "annotated_video_path": str(final_video_path),
        "player_dirs": [str(output_dir / "players" / f"player_{n}") for n in verified_numbers],
    }
