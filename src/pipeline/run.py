"""Stage 4-6 orchestrator (CLAUDE.md §5 Stages 4-6) + the full pipeline entrypoint.

Wired to `make run` (see Makefile: `$(PYTHON) -m src.pipeline.run`). For each video discovered in
`input/` (or explicitly named on the CLI): profile -> takes+detect+track+team (Stage 0.5-3, via
`src.track.run.run_track_stage` — READ-ONLY from this module's point of view: it is itself fully
resumable/cached, so calling it here never forces a recompute of already-cached work) -> events
(Stage 4) -> target selection (Stage 5, arrow prior + fragment stitching, human seam via
`--track-id`) -> ranking + cutting + reel (Stage 6) -> stat card + run report (Stage 6).

Every new stage this module owns (events/selection) is itself cached under `work/<slug>/` the same
way (`StageCache`); Stage 6's cutting/reel/stat-card outputs go straight to `output/<slug>/` and
are cheap, deterministic recomputations over already-cached upstream artifacts, so they are always
regenerated rather than separately cached (avoids a second cache-invalidation surface for a few
seconds of ffmpeg work — the expensive stages, detection/tracking, are the ones that actually skip
via cache).
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.common.io import (
    StageCache,
    config_hash,
    load_json,
    load_models_parquet,
    load_yaml,
    save_json,
    work_dir_for,
)
from src.common.logging import DropCounter, get_logger
from src.common.types import (
    BallDetection,
    Event,
    EventType,
    RunReport,
    StageTiming,
    Take,
    Track,
)
from src.common.video import probe
from src.detect.overlay_mask import ArrowHint
from src.events.aggregate import (
    attribute_events_to_target,
    compute_take_all_events,
    target_identity_id,
)
from src.events.goals import GoalDetectionResult, detect_goals_for_video
from src.events.key_moments import classify_candidate_windows, find_candidate_windows
from src.events.possession import compute_distance_covered
from src.events.shots import detect_shots
from src.events.sprints import detect_sprints
from src.goal.detect import detect_goal_structures_for_video
from src.highlights.cutting import cut_clips
from src.highlights.ranking import rank_events
from src.highlights.reel import build_reel, select_for_export
from src.highlights.selection import (
    SelectionResult,
    select_targets,
    timeline_coverage_seconds,
)
from src.identity.verify import TakeIdentityResult
from src.ingest.discovery import VideoRef, find_videos, parse_filename
from src.pipeline.annotated_video import render_full_annotated_video
from src.pipeline.player_output import write_player_output
from src.pipeline.profiler import build_run_profile
from src.stats.stats import build_player_stats, write_stat_card
from src.track.click_reid import extend_chain_with_profile
from src.track.run import run_track_stage
from src.track.tracker import assign_take_id

logger = get_logger(__name__)
console = Console()

app = typer.Typer(add_completion=False)


def parse_track_id_overrides(raw: str | None) -> dict[int, int]:
    """Parse the `--track-id` human-in-the-loop override flag.

    Accepts either a bare track id (`"16"`, applied to take 0 — the common case: a single-take
    clip like `clip2`) or a comma-separated `take_id:track_id` list (`"0:16,2:9"` — needed once a
    clip has multiple takes, e.g. `clip4`'s 4). This is the ONLY way `TakeSelection.method` can
    ever become `"manual_override"` (CLAUDE.md Golden Rule 4) — never invented elsewhere.
    """
    if not raw:
        return {}
    overrides: dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            take_str, track_str = part.split(":", 1)
            overrides[int(take_str)] = int(track_str)
        else:
            overrides[0] = int(part)
    return overrides


def _effective_frame_size(video_path: str | Path, decode_cfg: dict) -> tuple[float, float]:
    """The `(width, height)` a `Track`'s/`BallDetection`'s pixel bboxes actually live in —
    mirrors `src.common.video.decode_frames`' own scale-then-keep-even computation exactly, since
    that is what Stage 2 decoded at when it produced those bboxes.
    """
    meta = probe(video_path)
    width, height = meta["width"], meta["height"]
    scale_width = decode_cfg.get("scale_width")
    if scale_width and scale_width < width:
        out_width = scale_width
        out_height = (round(height * (scale_width / width)) // 2) * 2
        return float(out_width), float(out_height)
    return float(width), float(height)


def _bucket_by_take(
    balls: list[BallDetection], takes: list[Take]
) -> dict[int, list[BallDetection]]:
    by_take: dict[int, list[BallDetection]] = defaultdict(list)
    n_unassigned = 0
    for b in balls:
        take_id = assign_take_id(b.t, takes)
        if take_id is None:
            n_unassigned += 1
            continue
        by_take[take_id].append(b)
    if n_unassigned:
        logger.warning(
            "%d ball detection(s) fell outside every take's [t_start, t_end) and were excluded "
            "from shot detection",
            n_unassigned,
        )
    return dict(by_take)


def _compute_events(
    tracks: list[Track],
    takes: list[Take],
    balls: list[BallDetection],
    frame_width: float,
    events_cfg: dict,
    fps_sample: float,
    drops: DropCounter,
) -> list[Event]:
    """Stage 4: sprint events per track + shot events per take's own ball detections. Goals are
    handled separately by `check_goal_availability` (never a per-track/per-take loop — it is an
    unconditional "not available" on this footage, see `src/events/goals.py`).
    """
    events: list[Event] = []
    for track in tracks:
        events.extend(detect_sprints(track, events_cfg, fps_sample, drops))

    balls_by_take = _bucket_by_take(balls, takes)
    for take in takes:
        events.extend(
            detect_shots(balls_by_take.get(take.id, []), take.id, frame_width, events_cfg, drops)
        )
    return events


def _stitched_virtual_track(take_id: int, take_tracks: list[Track], ids: list[int]) -> Track:
    """Merge the boxes of several raw take-scoped fragments (one target's own location-selection
    `track_ids`) into ONE synthetic `Track`, time-sorted — mirrors
    `src.pipeline.extended_output._stitched_virtual_track` EXACTLY (kept independently duplicated
    per CLAUDE.md §10's "stages independently runnable/debuggable" convention, same as
    `_effective_frame_size` above). Needed because `find_candidate_windows` operates on a single
    `Track`'s own box sequence, not a list of fragments.
    """
    by_id = {tr.id: tr for tr in take_tracks}
    boxes = []
    for tid in ids:
        tr = by_id.get(tid)
        if tr is not None:
            boxes.extend(tr.boxes)
    boxes.sort(key=lambda b: b.t)
    return Track(id=-1, take_id=take_id, boxes=boxes)


def _load_stage2_3_artifacts(
    work_dir: Path,
) -> tuple[list[Take], list[Track], list[BallDetection], list[ArrowHint]]:
    """READ-ONLY load of Stage 0.5/1/2/3's own cached artifacts — this module never regenerates
    them itself (that is `run_track_stage`'s job, already called before this)."""
    takes = load_models_parquet(work_dir / "shots" / "takes.parquet", Take)
    tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    balls = load_models_parquet(work_dir / "detect" / "ball_detections.parquet", BallDetection)
    arrow_path = work_dir / "detect" / "arrow_hints.parquet"
    arrow_hints = load_models_parquet(arrow_path, ArrowHint) if arrow_path.exists() else []
    return takes, tracks, balls, arrow_hints


def run_pipeline_for_video(
    video_path: str | Path,
    configs: dict[str, dict],
    target_jersey: int | None,
    manual_overrides: dict[int, int] | None = None,
    work_root: str | Path = "work",
    output_root: str | Path = "output",
    use_nvdec: bool = True,
) -> RunReport:
    """Run Stages 4-6 for one already-detected/tracked video end to end (Stage 0.5-3 are invoked
    here too, via `run_track_stage`, but are no-ops on a cache hit — see module docstring).
    Writes `output/<slug>/{reel.mp4, stat_card.json, stat_card.md, run_report.json}` and returns
    the `RunReport`.
    """
    video_path = Path(video_path)
    work_dir = work_dir_for(video_path, root=work_root)
    output_dir = Path(output_root) / work_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: list[StageTiming] = []
    all_dropped: dict[str, int] = {}

    def _merge_dropped(prefix: str, dropped: dict[str, int]) -> None:
        for reason, count in dropped.items():
            all_dropped[f"{prefix}.{reason}"] = all_dropped.get(f"{prefix}.{reason}", 0) + count

    # --- Stage 0.5: source profile -----------------------------------------------------------
    t0 = time.time()
    profile = build_run_profile(
        video_path,
        configs["hardware"],
        configs["profile"],
        configs["shots"],
        work_root=work_root,
        use_nvdec=use_nvdec,
    )
    timings.append(
        StageTiming(stage="profile", wall_seconds=round(time.time() - t0, 2), vram_peak_mb=None)
    )

    # --- Stage 0.5/1/2/3: takes + detect + track + team (cached, may be a full compute) -------
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
    detect_summary = track_summary.get("detect_summary", {})
    n_takes, n_tracks = track_summary.get("n_takes"), track_summary.get("n_tracks")
    timings.append(
        StageTiming(
            stage="detect_track_team",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=detect_summary.get("peak_vram_mb"),
            notes=f"n_takes={n_takes} n_tracks={n_tracks}",
        )
    )
    _merge_dropped("detect", detect_summary.get("dropped", {}))

    takes, tracks, balls, arrow_hints = _load_stage2_3_artifacts(work_dir)
    frame_width, frame_height = _effective_frame_size(video_path, configs["hardware"]["decode"])
    fps_sample = configs["hardware"]["stages"]["track"]["fps_sample"]
    takes_by_id = {t.id: t for t in takes}

    # --- Stage 4: events (cached) --------------------------------------------------------------
    # JSON, not parquet: `Event.evidence` is a heterogeneous dict whose KEYS differ by event type
    # (sprint vs. shot) -- parquet/arrow needs a single struct schema across all rows, which a
    # mixed-type-per-row dict column does not reliably give it. JSON round-trips it exactly.
    t0 = time.time()
    events_path = work_dir / "events" / "events.json"
    events_cache_config = {
        "events_cfg": configs["events"],
        "n_tracks": len(tracks),
        "n_balls": len(balls),
        "n_takes": len(takes),
        "frame_width": frame_width,
    }
    events_cache = StageCache(events_path, events_cache_config, stage="events")
    event_drops = DropCounter("events")
    if events_cache.hit():
        events = [Event.model_validate(d) for d in load_json(events_path)]
    else:
        events = _compute_events(
            tracks, takes, balls, frame_width, configs["events"], fps_sample, event_drops
        )
        save_json(events, events_path)
        events_cache.write_meta()
    _merge_dropped("events", event_drops.report())

    timings.append(
        StageTiming(
            stage="events",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"n_events={len(events)} (sprint/shot)",
        )
    )

    # --- Stage C: auto-tracked goal structure ("streamed-gathering-treehouse" plan) ------------
    # VLM-localize + native-res-CV-refine (src/goal/detect.py). Manages its own resumable cache
    # (work/<slug>/goal/goal_structures.json); an absent GEMINI_API_KEY makes every take's own
    # attempt an honest, cheap no-op (checked BEFORE any decode -- see that function's own
    # docstring), so this is safe to call unconditionally, same "cheap and harmless when unused"
    # discipline as the other optional stages.
    goal_structure_gemini_key = os.environ.get("GEMINI_API_KEY") or None
    if not goal_structure_gemini_key:
        logger.warning(
            "GEMINI_API_KEY not set for %s -- goal-structure auto-detection (Stage C) skipped; "
            "goal/assist detection falls back to configs/goal_region.yaml's manual polygon (if "
            "any) for this video",
            video_path.name,
        )
    tracks_by_take_for_goal_structure: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take_for_goal_structure[tr.take_id].append(tr)
    goal_structure_report = detect_goal_structures_for_video(
        video_path,
        takes,
        dict(tracks_by_take_for_goal_structure),
        frame_width,
        frame_height,
        configs["goal_structure"],
        goal_structure_gemini_key,
        work_root=work_root,
        use_nvdec=use_nvdec,
    )
    goal_structures_by_take = {gs.take_id: gs for gs in goal_structure_report.takes}

    # --- Stage 4b: goals + assists (ADR-17, extended by ADR-20 + Stage D, cached) --------------
    # Real detection now (see src/events/goals.py) -- still correctly reports "not available" on
    # this footage (measured, CLAUDE.md §3.2(3)), just for a specific, auditable reason instead of
    # an unconditional stub. Cached like `events` above: cheap (low-fps decode + corner-crop OCR
    # only, no detector/tracker rerun), but resumable per CLAUDE.md §10.
    t0 = time.time()
    goals_path = work_dir / "events" / "goals.json"
    goals_cache_config = {
        "goal_cfg": configs["events"]["goal"],
        "assist_cfg": configs["events"]["assist"],
        "goal_region_cfg": configs["events"]["goal_region"],
        # ADR-20: the polygon (if any) actually configured for THIS slug, not the whole
        # goal_region.yaml file -- editing another video's polygon must not invalidate this one's
        # cache, but editing (or newly adding) this slug's own polygon must.
        "goal_region_polygon": configs["goal_region"].get("regions", {}).get(work_dir.name),
        # Stage D: Stage C's own detected structures feed the SAME cache key as the manual
        # polygon above -- a fresh/changed detection must invalidate this cache exactly like a
        # newly-drawn manual polygon does.
        "goal_structures": [gs.model_dump(mode="json") for gs in goal_structure_report.takes],
        "n_tracks": len(tracks),
        "n_balls": len(balls),
        "n_takes": len(takes),
    }
    goals_cache = StageCache(goals_path, goals_cache_config, stage="goals")
    if goals_cache.hit():
        cached = load_json(goals_path)
        goal_result = GoalDetectionResult(
            available=cached["available"],
            reason=cached["reason"],
            events=[Event.model_validate(d) for d in cached["events"]],
        )
    else:
        tracks_by_take_for_goals: dict[int, list[Track]] = defaultdict(list)
        for tr in tracks:
            tracks_by_take_for_goals[tr.take_id].append(tr)
        balls_by_take_for_goals = _bucket_by_take(balls, takes)
        goal_result = detect_goals_for_video(
            video_path,
            takes,
            tracks_by_take_for_goals,
            balls_by_take_for_goals,
            configs["events"],
            identity_of_by_take=None,
            ocr_cfg=configs["shots"]["ocr"],
            use_nvdec=use_nvdec,
            frame_width=frame_width,
            frame_height=frame_height,
            goal_region_cfg=configs["goal_region"],
            slug=work_dir.name,
            goal_structures_by_take=goal_structures_by_take,
            goal_structure_cfg=configs["goal_structure"],
            hardware_cfg=configs["hardware"],
            profile_cfg=configs["profile"],
        )
        save_json(goal_result.model_dump(mode="json"), goals_path)
        goals_cache.write_meta()
    events = events + goal_result.events
    timings.append(
        StageTiming(
            stage="goals",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"goals={goal_result.reason}",
        )
    )

    # --- Stage 5: target selection (cached) ----------------------------------------------------
    t0 = time.time()
    selection_path = work_dir / "selection.json"
    click_reid_cfg = configs["highlights"].get("click_reid", {})
    selection_cache_config = {
        "selection_cfg": configs["highlights"]["selection"],
        "click_reid_cfg": click_reid_cfg,
        "n_tracks": len(tracks),
        "n_arrow_hints": len(arrow_hints),
        "n_takes": len(takes),
        "target_jersey": target_jersey,
        "manual_overrides": manual_overrides or {},
    }
    selection_cache = StageCache(selection_path, selection_cache_config, stage="selection")
    if selection_cache.hit():
        selection = SelectionResult.model_validate(load_json(selection_path))
    else:
        selection = select_targets(
            str(video_path),
            target_jersey,
            takes,
            tracks,
            arrow_hints,
            frame_width,
            frame_height,
            configs["highlights"]["selection"],
            manual_overrides=manual_overrides,
        )
        # Click-anchored re-identification (owner request 2026-09-01, `src/track/click_reid.py`):
        # extend a MANUAL-OVERRIDE take's own stitched chain across a break `stitch_timeline`
        # itself didn't bridge, using the clicked player's own team-cluster + height as
        # corroborating evidence. Deliberately scoped to `manual_override` selections only -- the
        # click is the strongest identity evidence this pipeline has (Golden Rule 4); auto-selected
        # takes (arrow_vote/heuristic_fallback) are untouched.
        if click_reid_cfg.get("enabled", False) and manual_overrides:
            tracks_by_take: dict[int, list] = defaultdict(list)
            for tr in tracks:
                tracks_by_take[tr.take_id].append(tr)
            for sel_take in selection.takes:
                if sel_take.method != "manual_override" or sel_take.seed_track_id is None:
                    continue
                extended_ids, _evidence = extend_chain_with_profile(
                    sel_take.seed_track_id,
                    sel_take.track_ids,
                    tracks_by_take.get(sel_take.take_id, []),
                    click_reid_cfg,
                )
                if extended_ids != sel_take.track_ids:
                    logger.info(
                        "click_reid: take=%d extended manual-override chain from %d to %d "
                        "fragment(s) (seed=%d)",
                        sel_take.take_id,
                        len(sel_take.track_ids),
                        len(extended_ids),
                        sel_take.seed_track_id,
                    )
                    sel_take.track_ids = extended_ids
                    sel_take.coverage_seconds = timeline_coverage_seconds(
                        extended_ids, tracks_by_take.get(sel_take.take_id, [])
                    )
        save_json(selection, selection_path)
        selection_cache.write_meta()
    timings.append(
        StageTiming(
            stage="selection",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=(
                f"overall_confidence={selection.overall_confidence:.3f} "
                f"methods={[t.method for t in selection.takes]}"
            ),
        )
    )

    # --- Stage 6: ranking + cutting + reel + stats ---------------------------------------------
    t0 = time.time()
    timeline_tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for take_selection in selection.takes:
        ids = set(take_selection.track_ids)
        timeline_tracks_by_take[take_selection.take_id] = [
            tr for tr in tracks if tr.take_id == take_selection.take_id and tr.id in ids
        ]
    balls_by_take = _bucket_by_take(balls, takes)

    ranked = rank_events(
        events, selection, timeline_tracks_by_take, balls_by_take, configs["highlights"]
    )

    cutting_drops = DropCounter("cutting")
    clips = cut_clips(
        ranked, takes_by_id, video_path, output_dir / "clips", configs["highlights"], cutting_drops
    )
    _merge_dropped("cutting", cutting_drops.report())

    score_by_event_id = {ev.id: score for ev, score in ranked}
    clips_with_scores = [(c, score_by_event_id.get(c.event_id, c.rank_score)) for c in clips]
    export_clips = select_for_export(clips_with_scores, configs["highlights"]["export"])

    reel_path = output_dir / "reel.mp4"
    if export_clips:
        build_reel(export_clips, reel_path)
    else:
        logger.warning(
            "no clips survived ranking/dedupe for %s -- no reel written", video_path.name
        )

    timings.append(
        StageTiming(
            stage="ranking_cutting_reel",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"n_ranked={len(ranked)} n_clips_cut={len(clips)} n_in_reel={len(export_clips)}",
        )
    )

    # --- Stage 6: stat card ----------------------------------------------------------------------
    t0 = time.time()
    clips_by_event_id = {c.event_id: c for c in clips}
    attributed_event_ids = {ev.id for ev, _ in ranked}
    attributed_events = [ev for ev in events if ev.id in attributed_event_ids]
    all_timeline_track_ids = sorted(
        {tid for take_selection in selection.takes for tid in take_selection.track_ids}
    )
    player_ref = f"{work_dir.name}#jersey_{target_jersey}" if target_jersey else work_dir.name
    stats = build_player_stats(player_ref, all_timeline_track_ids, attributed_events)
    selection_summary = {
        "methods_by_take": {t.take_id: t.method for t in selection.takes},
        "method": selection.takes[0].method if len(selection.takes) == 1 else "mixed_per_take",
        "confidence": selection.overall_confidence,
        "needs_human_confirmation": selection.needs_human_confirmation,
        "seed_track_ids_by_take": {t.take_id: t.seed_track_id for t in selection.takes},
        "stitched_track_ids_by_take": {t.take_id: t.track_ids for t in selection.takes},
        "coverage_seconds_by_take": {
            t.take_id: {"covered": t.coverage_seconds, "take_duration": t.take_duration_seconds}
            for t in selection.takes
        },
    }
    write_stat_card(
        stats,
        attributed_events,
        clips_by_event_id,
        output_dir,
        target_jersey,
        selection_summary,
        goal_result.reason,
    )
    timings.append(
        StageTiming(stage="stats", wall_seconds=round(time.time() - t0, 2), vram_peak_mb=None)
    )

    # --- Stage 6b: full event suite + full-res annotated video + per-player output (ADR-21) ------
    # Unifies this Phase-1 flow with `src.pipeline.extended_output`'s own event suite / statcard /
    # annotated-video / highlight-reel machinery -- previously wired ONLY for filename-less videos,
    # even though the owner's exact statcard template + full event suite lives here and is fully
    # tested. The identity SOURCE differs (a human-selected/arrow-confirmed track + the filename's
    # own known jersey number, never OCR/VLM) but every downstream detector/writer below is the
    # SAME already-tested code `run_extended_pipeline_for_video` calls. Purely additive: nothing
    # above this point (reel.mp4/stat_card.md/clips/) is touched.
    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or None
    if not gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY not set for %s -- key-moment classification proceeds in its honest "
            "degraded mode (zero KEY_MOMENT events, never a guess)",
            video_path.name,
        )

    t0 = time.time()
    identity_by_take: dict[int, TakeIdentityResult] = {}
    for sel in selection.takes:
        is_located = sel.method != "none" and bool(sel.track_ids)
        identity_by_take[sel.take_id] = TakeIdentityResult(
            take_id=sel.take_id,
            jersey_number=target_jersey,
            status="verified" if is_located else "unverified",
            confidence=sel.confidence,
            evidence_frames=[],
            # This take's identity comes from the Phase-1 human-selection seam (arrow prior,
            # heuristic fallback, or an explicit `--track-id` override), never OCR/VLM -- honestly
            # distinguished from ADR-15's verified-jersey provenance via `identity_status` below,
            # where it actually surfaces (Golden Rule 5: never conflate two different kinds of
            # evidence behind the same label).
            location_method=str(sel.method),
            location_track_ids=sel.track_ids,
        )

    full_tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        full_tracks_by_take[tr.take_id].append(tr)
    full_balls_by_take = _bucket_by_take(balls, takes)

    goal_assist_by_take: dict[int, list[Event]] = defaultdict(list)
    for ev in goal_result.events:
        if ev.take_id is not None:
            goal_assist_by_take[ev.take_id].append(ev)

    full_events_cfg = configs["events"]
    full_selection_cfg = configs["highlights"]["selection"]
    full_attribution_cfg = configs["highlights"]["attribution"]
    full_key_moment_cfg = configs["key_moments"]

    full_drops = DropCounter("full_events")
    events_by_number: dict[int, list[Event]] = defaultdict(list)
    takes_used_by_number: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)

    for take in takes:
        tir = identity_by_take.get(take.id)
        if tir is None or tir.status != "verified" or tir.jersey_number is None:
            continue
        take_tracks = full_tracks_by_take.get(take.id, [])
        take_balls = full_balls_by_take.get(take.id, [])
        if not take_tracks:
            continue

        all_events, identity_of, _identity_confidence = compute_take_all_events(
            take,
            take_tracks,
            take_balls,
            frame_width,
            full_events_cfg,
            full_selection_cfg,
            fps_sample,
            drops=full_drops,
        )
        all_events = all_events + goal_assist_by_take.get(take.id, [])
        target_events = attribute_events_to_target(
            all_events,
            take_tracks,
            take_balls,
            tir.location_track_ids,
            identity_of,
            full_attribution_cfg,
        )

        virtual_track = _stitched_virtual_track(take.id, take_tracks, tir.location_track_ids)
        existing_windows = [
            (ev.t_start, ev.t_end)
            for ev in target_events
            if ev.type in (EventType.SPRINT, EventType.DRIBBLE, EventType.POSSESSION)
        ]
        candidates = find_candidate_windows(
            virtual_track, full_key_moment_cfg["prefilter"], existing_windows
        )
        key_moment_player_id = target_identity_id(tir.location_track_ids, identity_of)
        if key_moment_player_id is None and tir.location_track_ids:
            key_moment_player_id = tir.location_track_ids[0]
        key_events = classify_candidate_windows(
            video_path,
            take.id,
            key_moment_player_id,
            candidates,
            gemini_api_key,
            full_key_moment_cfg["vlm"],
            use_nvdec=use_nvdec,
        )
        target_events = target_events + key_events

        events_by_number[tir.jersey_number].extend(target_events)
        takes_used_by_number[tir.jersey_number].append((take.id, tir.location_track_ids))

    full_dropped = full_drops.report()
    if full_dropped:
        logger.info("full-event drops: %s", full_dropped)
    timings.append(
        StageTiming(
            stage="full_events",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"n_jersey_numbers={len(events_by_number)}",
        )
    )

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
        goal_region_cfg=configs["goal_region"],
        goal_structures_by_take=goal_structures_by_take,
        goal_structure_cfg=configs["goal_structure"],
        hardware_cfg=configs["hardware"],
        profile_cfg=configs["profile"],
        selection_cfg=configs["highlights"]["selection"],
    )
    timings.append(
        StageTiming(
            stage="annotated_video", wall_seconds=round(time.time() - t0, 2), vram_peak_mb=None
        )
    )

    t0 = time.time()
    for number in sorted(events_by_number.keys()):
        number_events = events_by_number.get(number, [])
        possession_seconds = sum(
            ev.t_end - ev.t_start for ev in number_events if ev.type == EventType.POSSESSION
        )
        total_distance = 0.0
        total_samples = 0
        total_fragments = 0
        for take_id, ids in takes_used_by_number.get(number, []):
            result = compute_distance_covered(
                ids, full_tracks_by_take.get(take_id, []), full_events_cfg
            )
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
            number_events,
            possession_seconds,
            distance_result,
            takes_by_id,
            final_video_path,
            configs["highlights"],
            goal_result.reason,
            identity_status="Verified (arrow/manual track selection)",
        )
        logger.info(
            "player_%d: %d event(s) across %d take(s) -- statcard/highlights written",
            number,
            len(number_events),
            len(takes_used_by_number.get(number, [])),
        )
    timings.append(
        StageTiming(
            stage="player_output", wall_seconds=round(time.time() - t0, 2), vram_peak_mb=None
        )
    )
    if not events_by_number:
        logger.warning(
            "no take of %s ever had a located target track -- no players/ folder written",
            video_path.name,
        )

    hashed_config = config_hash({**configs, "target_jersey": target_jersey})
    report = RunReport(
        run_id=f"{work_dir.name}-{int(time.time())}",
        video=str(video_path),
        config_hash=hashed_config,
        profile=profile,
        timings=timings,
        dropped=all_dropped,
        created_at=datetime.now(),
    )
    report_dict = report.model_dump(mode="json")
    report_dict["goals"] = goal_result.reason
    report_dict["speed"] = (
        "uncalibrated (no metric pitch homography available for this footage -- ADR-6, "
        "CLAUDE.md §3.2(4))"
    )
    report_dict["selection"] = selection_summary
    save_json(report_dict, output_dir / "run_report.json")

    logger.info(
        "pipeline done for %s -> %s (reel: %s, %d clip(s))",
        video_path.name,
        output_dir,
        "written" if export_clips else "NOT written (no surviving clips)",
        len(export_clips),
    )
    return report


def _load_all_configs() -> dict[str, dict]:
    return {
        "run": load_yaml("configs/run.yaml"),
        "ingest": load_yaml("configs/ingest.yaml"),
        "hardware": load_yaml("configs/hardware.yaml"),
        "profile": load_yaml("configs/profile.yaml"),
        "shots": load_yaml("configs/shots.yaml"),
        "detect": load_yaml("configs/detect.yaml"),
        "track": load_yaml("configs/track.yaml"),
        "team": load_yaml("configs/team.yaml"),
        "events": load_yaml("configs/events.yaml"),
        "goal_region": load_yaml("configs/goal_region.yaml"),  # ADR-20: human-marked goal-mouth
        # polygons, empty by default -- loaded unconditionally (cheap, tiny file) same as
        # "identity"/"key_moments" below; every clip's own slug is simply absent until the owner
        # draws a polygon for it, so this is a no-op for all 5 original clips today.
        "highlights": load_yaml("configs/highlights.yaml"),
        # ADR-15/14: only ever consumed by src.pipeline.extended_output (target_jersey is None
        # path, CLAUDE.md §3.3) -- loaded here unconditionally too since the original 5 clips'
        # own run_pipeline_for_video simply never reads these two keys (no behavior change).
        "identity": load_yaml("configs/identity.yaml"),
        "key_moments": load_yaml("configs/key_moments.yaml"),
        # ADR-19: only ever consumed by src.pipeline.manual_events (a `.annotations.txt` sidecar
        # is present next to the input, CLAUDE.md §14.1) -- loaded here unconditionally, same
        # "cheap and harmless when unused" reasoning as "identity"/"key_moments" above.
        "annotations": load_yaml("configs/annotations.yaml"),
        # Stage C ("streamed-gathering-treehouse" plan) -- VLM-localize + native-res-CV-refine
        # goal-structure detection (src/goal/detect.py). Loaded unconditionally, same "cheap and
        # harmless when unused" reasoning as the other stage-optional configs above; only actually
        # exercised when GEMINI_API_KEY is set (src/goal/detect.py's own honest no-op otherwise).
        "goal_structure": load_yaml("configs/goal_structure.yaml"),
    }


@app.command()
def main(
    videos: list[Path] | None = typer.Argument(
        None, help="Specific video path(s) to process. Omit to process every video in input/."
    ),
    input_dir: Path = typer.Option(Path("input"), help="Directory scanned when no VIDEOS given"),
    work_root: Path = typer.Option(Path("work")),
    output_root: Path = typer.Option(Path("output")),
    track_id: str | None = typer.Option(
        None,
        "--track-id",
        help=(
            "Phase-1 human-in-the-loop seam (CLAUDE.md Golden Rule 4): override auto target "
            "selection with a manually-confirmed track id. Bare form '16' applies to take 0 "
            "(single-take clips); 'take:track' pairs, comma-separated (e.g. '0:16,2:9'), for a "
            "multi-take clip. Only meaningful when exactly one video is being processed."
        ),
    ),
    target_jersey: int | None = typer.Option(
        None,
        "--target-jersey",
        help=(
            "ADR-18 (1) human-in-the-loop seam (CLAUDE.md Golden Rule 4), for a FILENAME-LESS "
            "video (no `clip<N> <jersey>.mp4` name for the source profiler to parse a jersey "
            "number from, e.g. a broadcast download): tell the pipeline the jersey number you "
            "watched and confirmed in the footage yourself. Threaded into Stage 5.5's identity "
            "verification, where a human-confirmed number needs only ONE supporting OCR/VLM "
            "frame to verify (vs. the normal multi-frame vote) -- see "
            "src/identity/verify.py::aggregate_take_identity. Only meaningful when exactly one "
            "filename-less video is being processed; a video that already has a filename-parsed "
            "jersey number is unaffected (that flow has no identity-verification stage at all)."
        ),
    ),
) -> None:
    """Run the full Stage 0.5-6 pipeline (CLAUDE.md §5/§6) for video(s), writing
    `output/<slug>/{reel.mp4, stat_card.json, stat_card.md, run_report.json}` for each.
    """
    configs = _load_all_configs()
    overrides = parse_track_id_overrides(track_id)

    if videos:
        pattern = re.compile(configs["run"]["filename_convention_regex"])
        refs = []
        for v in videos:
            _idx, jersey = parse_filename(Path(v).stem, pattern)
            refs.append(VideoRef(path=Path(v), target_jersey=jersey))
    else:
        refs = find_videos(
            input_dir,
            configs["run"]["filename_convention_regex"],
            configs["ingest"]["video_extensions"],
        )

    if not refs:
        console.print(f"[yellow]No videos found ({input_dir if not videos else videos})[/yellow]")
        raise typer.Exit(code=0)

    if overrides and len(refs) > 1:
        logger.warning(
            "--track-id was given but %d videos are being processed -- applying the same "
            "override to every one of them, which is almost certainly not what you want. Pass a "
            "single video path to target the override correctly.",
            len(refs),
        )

    filenameless_refs = [r for r in refs if r.target_jersey is None]
    if target_jersey is not None and len(filenameless_refs) > 1:
        logger.warning(
            "--target-jersey was given but %d filename-less videos are being processed -- "
            "applying the same human-confirmed jersey number to all of them is almost certainly "
            "not what you want. Pass a single filename-less video path to target the override "
            "correctly.",
            len(filenameless_refs),
        )

    for ref in refs:
        console.rule(f"[bold]{ref.path.name}[/bold]")

        # ADR-19 / CLAUDE.md §14.1: branch selection, in this exact order. A `.annotations.txt`
        # sidecar next to the input wins UNCONDITIONALLY -- checked before either the filename-
        # jersey or the ADR-15 auto branch below, because "the client's own file winning is the
        # trigger" (not a quality score, not whether a jersey number happens to be parseable).
        annotations_path = ref.path.parent / f"{ref.path.stem}.annotations.txt"
        if annotations_path.exists():
            # Imported lazily, same convention as the ADR-15 import below, so importing this
            # module never pulls the annotation/associate/annotated-video stack unless this
            # branch actually runs.
            from src.pipeline.manual_events import run_manual_events_pipeline_for_video

            summary = run_manual_events_pipeline_for_video(
                ref.path,
                configs,
                annotations_path,
                target_jersey=target_jersey,
                work_root=work_root,
                output_root=output_root,
            )
            console.print(summary)
            continue

        if ref.target_jersey is None:
            # ADR-15 (CLAUDE.md §3.3): no filename jersey number -> verified-identity extended
            # pipeline, never the Phase-1 single-target flow below. Imported lazily so importing
            # this module (e.g. from tests) never pulls the identity/events-aggregate/annotated-
            # video stack unless this branch actually runs.
            from src.pipeline.extended_output import run_extended_pipeline_for_video

            summary = run_extended_pipeline_for_video(
                ref.path,
                configs,
                work_root=work_root,
                output_root=output_root,
                human_confirmed_jersey=target_jersey,
            )
            console.print(summary)
            continue
        report = run_pipeline_for_video(
            ref.path,
            configs,
            ref.target_jersey,
            manual_overrides=overrides,
            work_root=work_root,
            output_root=output_root,
        )
        console.print(report.model_dump(mode="json"))


if __name__ == "__main__":
    app()
