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

import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import typer
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
    RunReport,
    StageTiming,
    Take,
    Track,
)
from src.common.video import probe
from src.detect.overlay_mask import ArrowHint
from src.events.goals import check_goal_availability
from src.events.shots import detect_shots
from src.events.sprints import detect_sprints
from src.highlights.cutting import cut_clips
from src.highlights.ranking import rank_events
from src.highlights.reel import build_reel, select_for_export
from src.highlights.selection import SelectionResult, select_targets
from src.ingest.discovery import VideoRef, find_videos, parse_filename
from src.pipeline.profiler import build_run_profile
from src.stats.stats import build_player_stats, write_stat_card
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

    goal_result = check_goal_availability(profile, configs["events"]["goal"])
    events = events + goal_result.events  # always [] on this footage -- see src/events/goals.py
    timings.append(
        StageTiming(
            stage="events",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"n_events={len(events)} (sprint/shot only; goals={goal_result.reason})",
        )
    )

    # --- Stage 5: target selection (cached) ----------------------------------------------------
    t0 = time.time()
    selection_path = work_dir / "selection.json"
    selection_cache_config = {
        "selection_cfg": configs["highlights"]["selection"],
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
        "highlights": load_yaml("configs/highlights.yaml"),
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

    for ref in refs:
        console.rule(f"[bold]{ref.path.name}[/bold]")
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
