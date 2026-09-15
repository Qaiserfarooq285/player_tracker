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

import numpy as np
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
from src.events.manual_touches import manual_touch_events, merge_manual_touches, parse_touch_times
from src.events.possession import compute_distance_covered
from src.events.shots import detect_shots
from src.events.sprints import detect_sprints
from src.goal.detect import detect_goal_structures_for_video
from src.highlights.cutting import cut_clips
from src.highlights.ranking import rank_events
from src.highlights.reel import build_reel, select_for_export
from src.highlights.selection import (
    SelectionResult,
    TakeSelection,
    select_targets,
    timeline_coverage_seconds,
)
from src.identity.jersey_models import free_optional_jersey_stack, load_optional_jersey_stack
from src.identity.verify import TakeIdentityResult
from src.ingest.discovery import VideoRef, find_videos, parse_filename
from src.pipeline.annotated_video import render_full_annotated_video
from src.pipeline.player_output import write_player_output
from src.pipeline.profiler import build_run_profile
from src.stats.stats import build_player_stats, write_stat_card
from src.track.camera_motion import CameraMotionReport, estimate_camera_motion_for_video
from src.track.click_reid import (
    collect_track_jersey_digits,
    extend_chain_with_profile,
    prune_chain_by_jersey,
)
from src.track.kit_wiring import build_take_kit_colour
from src.track.run import run_track_stage
from src.track.target import KitColourSample, TargetProfile, save_target_profile
from src.track.target_state import TargetFrameStatus, build_target_timeline
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


def normalize_manual_overrides(
    raw: dict[int, int | list[int]] | None,
) -> dict[int, list[int]]:
    """ "streamed-gathering-treehouse" plan Stage 1 (multiple click anchors): the one place this
    module normalizes `manual_overrides` before it reaches `select_targets`, whose own parameter
    is `dict[int, list[int]] | None`.

    `parse_track_id_overrides` and the `--track-id` CLI grammar stay UNCHANGED (3 existing tests
    depend on the exact `dict[int, int]` shape it returns) -- this function is what bridges that
    legacy shape, and `apps/api/main.py`'s own multi-anchor `dict[int, list[int]]` shape, into one
    canonical form. A bare `int` becomes a single-element list; a `list[int]` is deduped
    (order-preserving, first occurrence wins) and passed through; a take whose own list ends up
    empty is dropped from the result entirely (equivalent to no override for that take at all,
    never a dict entry pointing at nothing).
    """
    if not raw:
        return {}
    normalized: dict[int, list[int]] = {}
    for take_id, value in raw.items():
        if isinstance(value, int):
            ids = [value]
        else:
            seen: set[int] = set()
            ids = []
            for tid in value:
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        if ids:
            normalized[take_id] = ids
    return normalized


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


def _track_active_windows(
    track_ids: list[int], take_tracks: list[Track]
) -> dict[int, list[tuple[float, float]]]:
    """Each of `track_ids`' own real `[first_box.t, last_box.t]` span ("streamed-gathering-
    treehouse" plan Stage 5, second half) -- the same population `src.pipeline.manual_events`
    already does for its own multi-target-per-take case, generalised here for the single-target
    Phase-1/ADR-15 flow's own stitched chain. A track id absent from `take_tracks`, or with no
    boxes at all, is simply omitted (falls back to `is_target_active_at`'s own "no entry -> always
    active" default for that one id, never a crash)."""
    by_id = {tr.id: tr for tr in take_tracks}
    windows: dict[int, list[tuple[float, float]]] = {}
    for tid in track_ids:
        tr = by_id.get(tid)
        if tr is not None and tr.boxes:
            windows[tid] = [(tr.boxes[0].t, tr.boxes[-1].t)]
    return windows


def _accepted_target_ids(sel: TakeSelection, target_profile_active: bool) -> list[int]:
    """The SAME accepted-fragment id set that gates the target's red/amber box (Stage A/C,
    `src.track.target_state.build_target_timeline`) and Stage D's own event attribution -- the
    plan's own explicit "box and statistics share one identity" requirement. A `TargetProfile`-
    driven take (`target_profile_active=True`) uses ONLY `sel.verified_track_ids` -- as of the
    "streamed-gathering-treehouse" (recheck) plan's own **Fix A**, this is no longer just the
    human-established seed(s) / whichever candidate(s) independently passed `verify_candidate`: it
    is that VERIFIED CHAIN -- those seeds/candidates plus every other same-take fragment
    `stitch_timeline` joined onto them that does NOT actively contradict the profile (confident
    different kit colour or jersey number), classified by
    `src.highlights.selection._verify_chain_fragments`. A fragment that DOES contradict is still
    excluded (never a red box, never counted into stats, "Candidate != Target" still holds) -- it
    is only the "no contradicting evidence" bar that widened, fixing the plan's own Finding A
    (every legitimately-stitched fragment used to be silently dropped from the stats count). Every
    pre-existing profile-LESS method (arrow_vote/heuristic_fallback/none, and a profile-less
    manual_override -- CLAUDE.md §14's auto flow, ADR-15, ADR-19) has no such verified/candidate
    distinction at all, so this falls back to the full stitched `sel.track_ids`, EXACTLY the
    pre-existing behaviour for every one of those flows (unchanged by either plan)."""
    if target_profile_active:
        return sel.verified_track_ids
    return sel.track_ids


def _resolve_manual_touch_target_jersey(
    typed_target_jersey: int | None,
    profile_jersey_number: int | None,
    produced_jersey_numbers: list[int],
) -> tuple[int | None, str]:
    """ "streamed-gathering-treehouse" (recheck) plan **Fix B**: resolve which jersey number a
    client's typed manual touch-times attach to, replacing the old `number == target_jersey`
    equality test (`src.pipeline.run.run_pipeline_for_video`'s own touch-merge loop) that silently
    dropped every typed touch time whenever `target_jersey is None` -- which is the NORMAL case
    now that the jersey field is optional and identity comes from a verified click instead of a
    typed number (see this plan's own Context section, finding B).

    Resolution order (owner's own words):
    1. the client's own TYPED `target_jersey`, when given -- the strongest, most explicit signal;
    2. else the persistent `TargetProfile`'s own `jersey_number` (`profile_jersey_number`) -- set
       from a verified click (`jersey_source="click"`) or a filename/human-confirmed number, so it
       is still a real, non-guessed identity, just not RE-typed alongside this touch submission;
    3. else, when this run produced events for EXACTLY ONE jersey number
       (`produced_jersey_numbers`), that one -- there is no genuine ambiguity left to resolve, only
       a field the client didn't bother re-typing;
    4. otherwise genuinely AMBIGUOUS (several jerseys, no profile number, no typed number) -- the
       touch times attach to NONE of them (never guessed onto an arbitrary one) and the caller must
       report this in the run report (CLAUDE.md §10: "log everything dropped"), never the old
       silent drop.

    Returns `(resolved_number_or_None, reason)` -- `reason` is always populated (Golden Rule 5):
    one of `"typed"`, `"profile"`, `"single_jersey_inferred"`, `"ambiguous_not_attached"`.
    """
    if typed_target_jersey is not None:
        return typed_target_jersey, "typed"
    if profile_jersey_number is not None:
        return profile_jersey_number, "profile"
    if len(produced_jersey_numbers) == 1:
        return produced_jersey_numbers[0], "single_jersey_inferred"
    return None, "ambiguous_not_attached"


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


def _serialize_kit_colour_by_take(
    kit_lab_by_track_id_by_take: dict[int, dict[int, np.ndarray]],
    kit_sample_by_track_id_by_take: dict[int, dict[int, KitColourSample]],
) -> dict:
    """JSON-safe encoding for `work/<slug>/track/kit_colour.json` -- a `np.ndarray`/pydantic-model
    caching format (not a `.parquet` file, despite the plan's own sketch naming one): the plan's
    Stage 3 asks for two DIFFERENT per-track shapes side by side (a bare Lab triple, and a
    torso/shorts/socks `KitColourSample`), and a `.parquet` file needs one flat row schema —
    forcing both into one table would mean either duplicating the write into two separate parquet
    files (two cache-invalidation surfaces instead of one) or flattening `KitColourSample` into
    loose columns and losing its own type on read-back. JSON (the same convention `events.json`
    already uses for exactly this "heterogeneous structure, not a flat table" reason, see this
    module's own `events` caching comment) keeps both shapes exact and keeps this one cache file
    the single source for both.
    """
    return {
        "kit_lab": {
            str(take_id): {str(tid): lab.tolist() for tid, lab in by_track.items()}
            for take_id, by_track in kit_lab_by_track_id_by_take.items()
        },
        "kit_sample": {
            str(take_id): {
                str(tid): sample.model_dump(mode="json") for tid, sample in by_track.items()
            }
            for take_id, by_track in kit_sample_by_track_id_by_take.items()
        },
    }


def _deserialize_kit_colour_by_take(
    data: dict,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, KitColourSample]]]:
    """Inverse of `_serialize_kit_colour_by_take`."""
    kit_lab_by_track_id_by_take = {
        int(take_id): {int(tid): np.array(lab) for tid, lab in by_track.items()}
        for take_id, by_track in data.get("kit_lab", {}).items()
    }
    kit_sample_by_track_id_by_take = {
        int(take_id): {
            int(tid): KitColourSample.model_validate(sample) for tid, sample in by_track.items()
        }
        for take_id, by_track in data.get("kit_sample", {}).items()
    }
    return kit_lab_by_track_id_by_take, kit_sample_by_track_id_by_take


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


def _collect_jersey_by_track_by_take(
    video_path: Path,
    takes: list[Take],
    tracks: list[Track],
    frame_width: float,
    frame_height: float,
    configs: dict[str, dict],
    use_nvdec: bool,
) -> dict[int, dict[int, str]]:
    """Per-take `{track_id: confident_jersey_digit_string}` for EVERY track in EVERY take
    ("streamed-gathering-treehouse" plan Stage 4/6) -- `src.track.click_reid.
    collect_track_jersey_digits`, called once per take, reusing the SAME ADR-21 legibility-gate +
    tight-number-region PARSeq chain the pre-existing click_reid corroboration block already uses
    for a narrower (manual-override-only) case. Loads the legibility/PARSeq checkpoint stack ONCE
    for the whole video (CLAUDE.md §11), frees it in `finally`. A take with no confident majority
    for a track is simply absent from that take's dict (the existing, honest
    `collect_track_jersey_digits` contract) -- `{}` everywhere when the checkpoint itself is
    unavailable/disabled, never a guess, never a raise.
    """
    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]
    scale_x = native_w / frame_width if frame_width else 1.0
    scale_y = native_h / frame_height if frame_height else 1.0

    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(
        configs["identity"]
    )
    max_samples = (
        configs["highlights"].get("click_reid", {}).get("max_jersey_samples_per_track", 40)
    )
    try:
        result: dict[int, dict[int, str]] = {}
        for take in takes:
            result[take.id] = collect_track_jersey_digits(
                video_path,
                tracks_by_take.get(take.id, []),
                scale_x,
                scale_y,
                native_w,
                native_h,
                configs["identity"],
                legibility_model,
                parseq_model,
                parseq_transform,
                max_samples,
                configs["identity"]["aggregation"],
                use_nvdec=use_nvdec,
            )
        return result
    finally:
        free_optional_jersey_stack(legibility_model, parseq_model)


def run_pipeline_for_video(
    video_path: str | Path,
    configs: dict[str, dict],
    target_jersey: int | None,
    manual_overrides: dict[int, int | list[int]] | None = None,
    work_root: str | Path = "work",
    output_root: str | Path = "output",
    use_nvdec: bool = True,
    target_profile: TargetProfile | None = None,
    manual_touch_times: str | None = None,
) -> RunReport:
    """Run Stages 4-6 for one already-detected/tracked video end to end (Stage 0.5-3 are invoked
    here too, via `run_track_stage`, but are no-ops on a cache hit — see module docstring).
    Writes `output/<slug>/{reel.mp4, stat_card.json, stat_card.md, run_report.json}` and returns
    the `RunReport`.

    `target_profile` ("streamed-gathering-treehouse" plan Stage 4/6, `src.track.target.
    TargetProfile`): `None` (every pre-existing caller) leaves Stage 5 target selection completely
    unchanged (`select_targets` with no profile — arrow vote / heuristic fallback, CLAUDE.md §14's
    auto flow). When supplied (only ever by `apps/api/main.py`'s click endpoint today), Stage 5
    becomes strict target-driven re-identification instead — see `select_targets`'s own docstring.
    `target_profile` is MUTATED in place by `select_targets` (new `TargetLink`s appended, memory
    bank updated), and this function persists the evolved profile back to
    `work/<slug>/target.json` itself before returning — the caller is only responsible for
    building/loading the profile before calling in (`apps/api/main.py`'s click endpoint does
    exactly that).

    `manual_overrides` accepts EITHER the legacy `dict[int, int]` shape (`parse_track_id_overrides`,
    the CLI's own `--track-id` flag) or the newer `dict[int, list[int]]` shape (`apps/api/main.py`'s
    multi-anchor click flow, "streamed-gathering-treehouse" plan Stage 1) -- `normalize_manual_
    overrides` below is the ONE place that reconciles both into the canonical `dict[int, list[int]]`
    `select_targets` itself requires, so nothing downstream of this function ever needs to branch
    on which shape was originally supplied.

    `manual_touch_times` (Plan Stage 2, "streamed-gathering-treehouse"): an optional client-typed,
    comma-/newline-separated list of `M:SS`/`H:MM:SS` ball-touch times (`configs/events.yaml:
    manual_touch`, `src.events.manual_touches`), submitted alongside a click. This flow's own
    target identity is now resolved, not just read off `target_jersey` verbatim -- **Fix B**,
    "streamed-gathering-treehouse" (recheck) plan: `target_jersey` is `None` whenever the owner
    relies on a click instead of also typing a jersey number (the now-recommended flow), so the
    typed touch times are attached via `_resolve_manual_touch_target_jersey` (typed field -> the
    active `TargetProfile`'s own jersey number -> the run's single produced jersey number ->
    genuinely ambiguous, reported, attached to none) rather than a bare `number == target_jersey`
    equality test that used to silently drop them whenever `target_jersey` was `None`. Merged into
    the resolved jersey number's own event list, once, right before `write_player_output` -- see
    the merge site below for why (Golden Rule 5: "their times win, auto-detection fills the gaps",
    never a silent double-count and never a silently dropped typo).
    """
    video_path = Path(video_path)
    manual_overrides = normalize_manual_overrides(manual_overrides)
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

    # --- Stage 3 ("streamed-gathering-treehouse" plan): per-track kit colour, cached ------------
    # Wires `src.track.kit_wiring.build_take_kit_colour` into production -- previously
    # `detect_passes`'s own `kit_lab_by_track_id` parameter was dead code (nothing ever
    # constructed it), and `src.track.target_verify.verify_candidate`'s `kit_by_track` argument
    # had no producer either. Built ONCE per take here and reused for BOTH: (a) every take's own
    # `detect_passes` call (Stage 4b goals below, and the Stage 6b per-take event loop further
    # down) via `kit_lab_by_track_id`, and (b) Stage 4/6's `verify_candidate` calls (via
    # `select_targets(..., kit_by_track_by_take=...)` below) via `kit_sample_by_track_id`. Runs
    # UNCONDITIONALLY (not gated on `target_profile` being supplied) -- ADR-20's colour-aware
    # pass/turnover attribution benefits every run, profile or not.
    t0 = time.time()
    kit_colour_path = work_dir / "track" / "kit_colour.json"
    native_meta_for_kit = probe(video_path)
    native_w_for_kit, native_h_for_kit = native_meta_for_kit["width"], native_meta_for_kit["height"]
    scale_x_for_kit = native_w_for_kit / frame_width if frame_width else 1.0
    scale_y_for_kit = native_h_for_kit / frame_height if frame_height else 1.0
    kit_colour_cache_config = {
        "identity_crop_cfg": configs["identity"]["crop"],
        "number_crop_cfg": configs["identity"]["parseq_soccernet"]["number_crop"],
        "target_bands_cfg": configs["target"]["bands"],
        "target_kit_colour_cfg": configs["target"]["kit_colour"],
        "target_band_weights": configs["target"]["band_confidence_weights"],
        "events_kit_colour_cfg": configs["events"]["kit_colour"],
        "n_tracks": len(tracks),
        "n_takes": len(takes),
    }
    kit_colour_cache = StageCache(kit_colour_path, kit_colour_cache_config, stage="kit_colour")
    if kit_colour_cache.hit():
        kit_lab_by_track_id_by_take, kit_sample_by_track_id_by_take = (
            _deserialize_kit_colour_by_take(load_json(kit_colour_path))
        )
    else:
        tracks_by_take_for_kit_colour: dict[int, list[Track]] = defaultdict(list)
        for tr in tracks:
            tracks_by_take_for_kit_colour[tr.take_id].append(tr)
        kit_lab_by_track_id_by_take = {}
        kit_sample_by_track_id_by_take = {}
        for take in takes:
            lab_by_track, sample_by_track = build_take_kit_colour(
                video_path,
                take,
                tracks_by_take_for_kit_colour.get(take.id, []),
                scale_x_for_kit,
                scale_y_for_kit,
                native_w_for_kit,
                native_h_for_kit,
                configs["identity"],
                configs["target"],
                configs["events"]["kit_colour"],
                configs["highlights"].get("click_reid", {}).get("max_jersey_samples_per_track", 40),
                use_nvdec=use_nvdec,
            )
            kit_lab_by_track_id_by_take[take.id] = lab_by_track
            kit_sample_by_track_id_by_take[take.id] = sample_by_track
        save_json(
            _serialize_kit_colour_by_take(
                kit_lab_by_track_id_by_take, kit_sample_by_track_id_by_take
            ),
            kit_colour_path,
        )
        kit_colour_cache.write_meta()
    timings.append(
        StageTiming(
            stage="kit_colour",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"n_takes_with_lab={len(kit_lab_by_track_id_by_take)}",
        )
    )

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
        # Stage 3 kit colour feeds this stage's own assist attribution (detect_goals_for_take's
        # lazy detect_passes call) -- a changed kit-colour cache must invalidate this one too.
        "n_kit_lab_tracks": sum(len(v) for v in kit_lab_by_track_id_by_take.values()),
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
            kit_lab_by_track_id_by_take=kit_lab_by_track_id_by_take,
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

    # --- Stage 4.9: camera motion (cached) -----------------------------------------------------
    # Fragment stitching compares positions in IMAGE space, where a fast camera pan makes a
    # stationary player look like a sprinter -- measured on real footage: during the 24-26s pan in
    # `chelsea_burnley_target10`, camera translation spikes from a 2.01 px/frame median to 12.84,
    # and the CORRECT continuation of the clicked #10 reads 3.45 bbox-heights/s raw (rejected) vs
    # 0.24 compensated (accepted), while the WRONG one goes the other way, 3.07 raw -> 6.41
    # compensated. See src/track/camera_motion.py's module docstring.
    t0 = time.time()
    camera_motion_path = work_dir / "track" / "camera_motion.json"
    camera_motion_cfg = configs["profile"]["motion"]
    camera_motion_cache = StageCache(
        camera_motion_path,
        {
            "motion_cfg": camera_motion_cfg,
            "fps": configs["hardware"]["stages"]["track"]["fps_sample"],
            "scale_width": configs["hardware"]["decode"].get("scale_width"),
            "n_takes": len(takes),
        },
        stage="camera_motion",
    )
    if camera_motion_cache.hit():
        camera_motion = CameraMotionReport.model_validate(load_json(camera_motion_path))
    else:
        camera_motion = estimate_camera_motion_for_video(
            str(video_path),
            takes,
            camera_motion_cfg,
            fps=configs["hardware"]["stages"]["track"]["fps_sample"],
            scale_width=configs["hardware"]["decode"].get("scale_width"),
            use_nvdec=use_nvdec,
        )
        save_json(camera_motion.model_dump(mode="json"), camera_motion_path)
        camera_motion_cache.write_meta()
    camera_motion_by_take = camera_motion.by_take()
    timings.append(
        StageTiming(
            stage="camera_motion",
            wall_seconds=round(time.time() - t0, 2),
            vram_peak_mb=None,
            notes=f"takes={len(camera_motion.takes)}",
        )
    )

    # --- Stage 4/6 ("streamed-gathering-treehouse" plan): target-profile jersey evidence --------
    # Only ever built when a persistent `TargetProfile` is actually in play (apps/api/main.py's
    # click endpoint) -- `verify_candidate`'s jersey hard-reject/scoring step needs a confident
    # read per CANDIDATE track across EVERY take, not just a manual-override take's own chain
    # (unlike the pre-existing click_reid corroboration block below, which is scoped to exactly
    # that narrower case). `None` for every profile-less run -- zero extra cost.
    target_cfg: dict | None = None
    jersey_by_track_by_take_for_selection: dict[int, dict[int, str]] | None = None
    if target_profile is not None:
        target_cfg = {**configs["target"], "kit_colour": configs["events"]["kit_colour"]}
        jersey_by_track_by_take_for_selection = _collect_jersey_by_track_by_take(
            video_path,
            takes,
            tracks,
            frame_width,
            frame_height,
            configs,
            use_nvdec,
        )

    # --- Stage 5: target selection (cached, except when a TargetProfile drives it) -------------
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
        "camera_motion_takes": len(camera_motion.takes),
    }
    selection_cache = StageCache(selection_path, selection_cache_config, stage="selection")
    # A persistent TargetProfile is MUTATED in place by `select_targets` (a new `TargetLink`
    # appended per take, its memory bank updated on every accepted link) -- the caller re-saves it
    # after this call expecting real forward progress each time. A cache hit would silently skip
    # that mutation on every call after the first, which would break the whole click-driven
    # re-identification flow (the profile would never actually learn/record anything past its
    # first use). So a profile-driven run always recomputes Stage 5, never reads/writes this
    # cache -- profile-less runs are completely unaffected (this only ever triggers when
    # `target_profile is not None`, i.e. never for CLAUDE.md §14's own auto flow).
    if target_profile is None and selection_cache.hit():
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
            camera_motion_by_take=camera_motion_by_take,
            target_profile=target_profile,
            target_cfg=target_cfg,
            kit_by_track_by_take=kit_sample_by_track_id_by_take if target_profile else None,
            jersey_by_track_by_take=jersey_by_track_by_take_for_selection,
        )
        # Click-anchored re-identification (owner request 2026-09-01, `src/track/click_reid.py`):
        # extend a MANUAL-OVERRIDE take's own stitched chain across a break `stitch_timeline`
        # itself didn't bridge, using the clicked player's own team-cluster + height as
        # corroborating evidence. Deliberately scoped to `manual_override` selections only -- the
        # click is the strongest identity evidence this pipeline has (Golden Rule 4); auto-selected
        # takes (arrow_vote/heuristic_fallback) are untouched. Also skipped entirely once a
        # `TargetProfile` is in play: Stage 2's `verify_candidate` (already run for every
        # profile-driven `manual_override`/`target_reidentified` take above) supersedes this
        # additive mechanism (plan's own "Out of scope" note) -- running both would mean a second,
        # uncoordinated chain-extension pass over a selection Stage 4 already resolved.
        if (
            (
                click_reid_cfg.get("enabled", False)
                or click_reid_cfg.get("prune_jersey_disagreement", False)
            )
            and manual_overrides
            and target_profile is None
        ):
            tracks_by_take: dict[int, list] = defaultdict(list)
            for tr in tracks:
                tracks_by_take[tr.take_id].append(tr)

            # Jersey-number corroboration (2026-09-01 fix for the measured failure documented in
            # configs/highlights.yaml: click_reid's own comment -- colour cluster alone let a
            # DIFFERENT physical player join a clicked chain, because Track.team_confidence reads
            # as an uninformative constant on real broadcast footage). Loaded/freed once per run,
            # only when this stage actually runs (a manual override exists and the feature is
            # enabled) -- never for the common auto-selected path.
            native_meta = probe(video_path)
            native_w, native_h = native_meta["width"], native_meta["height"]
            scale_x = native_w / frame_width if frame_width else 1.0
            scale_y = native_h / frame_height if frame_height else 1.0
            legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(
                configs["identity"]
            )
            try:
                for sel_take in selection.takes:
                    if sel_take.method != "manual_override" or sel_take.seed_track_id is None:
                        continue
                    take_tracks = tracks_by_take.get(sel_take.take_id, [])
                    jersey_by_track_id = collect_track_jersey_digits(
                        video_path,
                        take_tracks,
                        scale_x,
                        scale_y,
                        native_w,
                        native_h,
                        configs["identity"],
                        legibility_model,
                        parseq_model,
                        parseq_transform,
                        click_reid_cfg["max_jersey_samples_per_track"],
                        configs["identity"]["aggregation"],
                        use_nvdec=use_nvdec,
                    )
                    if jersey_by_track_id:
                        logger.info(
                            "click_reid: take=%d got confident jersey reads for %d/%d track(s)",
                            sel_take.take_id,
                            len(jersey_by_track_id),
                            len(take_tracks),
                        )
                    # SUBTRACTIVE pass first: drop any fragment `stitch_timeline` joined whose own
                    # confident jersey read CONTRADICTS the clicked seed's. Safe to run by default
                    # (it can only remove, never introduce, a physical player) -- see
                    # `prune_chain_by_jersey`'s own docstring for the measured blue-#10 -> claret
                    # -#21 join this exists to stop.
                    if click_reid_cfg.get("prune_jersey_disagreement", False):
                        kept_ids, prune_trail = prune_chain_by_jersey(
                            sel_take.seed_track_id, sel_take.track_ids, jersey_by_track_id
                        )
                        dropped = [e for e in prune_trail if e.get("dropped")]
                        if dropped:
                            for e in dropped:
                                logger.warning(
                                    "click_reid: take=%d DROPPED fragment track=%s from the "
                                    "clicked chain -- it reads jersey #%s but the clicked player "
                                    "reads #%s (a different player was stitched in)",
                                    sel_take.take_id,
                                    e["track_id"],
                                    e["fragment_jersey"],
                                    e["seed_jersey"],
                                )
                            sel_take.track_ids = kept_ids
                            sel_take.coverage_seconds = timeline_coverage_seconds(
                                kept_ids, take_tracks
                            )

                    if not click_reid_cfg.get("enabled", False):
                        continue

                    extended_ids, _evidence = extend_chain_with_profile(
                        sel_take.seed_track_id,
                        sel_take.track_ids,
                        take_tracks,
                        click_reid_cfg,
                        jersey_by_track_id,
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
                            extended_ids, take_tracks
                        )
            finally:
                free_optional_jersey_stack(legibility_model, parseq_model)
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

    if target_profile is not None:
        # Persist the profile `select_targets` just mutated in place (new `TargetLink`s appended,
        # memory bank updated) -- `work/<slug>/target.json` is this profile's own canonical home
        # (the same `work_dir` every other Stage 0.5-6 cache in this function already lives
        # under), so THIS function owns writing it back rather than pushing that responsibility
        # out to every caller that happens to supply a profile.
        save_target_profile(target_profile, work_dir / "target.json")

    # --- Stage A/B ("streamed-gathering-treehouse" plan): per-frame target state machine --------
    # Only ever built when a persistent `TargetProfile` actually drove this run's own target
    # selection above -- every pre-existing profile-less flow (CLAUDE.md §14's filename-jersey auto
    # flow, ADR-15's extended pipeline, ADR-19's manual-events mode) never reaches this block, so
    # its own rendering/stats stay byte-for-byte unchanged. See `src.track.target_state`'s own
    # module docstring for the three holes this closes (stale box / silent ID switch / whole-span
    # active window).
    target_timeline_by_take: dict[int, list[TargetFrameStatus]] = {}
    if target_profile is not None:
        t0 = time.time()
        tracks_by_take_for_timeline: dict[int, list[Track]] = defaultdict(list)
        for tr in tracks:
            tracks_by_take_for_timeline[tr.take_id].append(tr)
        frame_state_cfg = configs["target"]["frame_state"]
        timeline_path = work_dir / "target_timeline.json"
        timeline_cache_config = {
            "frame_state_cfg": frame_state_cfg,
            "verified_track_ids_by_take": {
                sel.take_id: sel.verified_track_ids for sel in selection.takes
            },
            "n_tracks": len(tracks),
        }
        timeline_cache = StageCache(timeline_path, timeline_cache_config, stage="target_timeline")
        if timeline_cache.hit():
            cached_timeline = load_json(timeline_path)
            target_timeline_by_take = {
                int(take_id_str): [TargetFrameStatus.model_validate(s) for s in statuses]
                for take_id_str, statuses in cached_timeline.items()
            }
        else:
            for sel in selection.takes:
                take = takes_by_id.get(sel.take_id)
                if take is None:
                    continue
                target_timeline_by_take[sel.take_id] = build_target_timeline(
                    take,
                    sel.verified_track_ids,
                    tracks_by_take_for_timeline.get(sel.take_id, []),
                    frame_state_cfg,
                )
            save_json(
                {
                    str(take_id): [status.model_dump(mode="json") for status in statuses]
                    for take_id, statuses in target_timeline_by_take.items()
                },
                timeline_path,
            )
            timeline_cache.write_meta()
        timings.append(
            StageTiming(
                stage="target_timeline",
                wall_seconds=round(time.time() - t0, 2),
                vram_peak_mb=None,
                notes=f"n_takes={len(target_timeline_by_take)}",
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
    full_tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        full_tracks_by_take[tr.take_id].append(tr)
    full_balls_by_take = _bucket_by_take(balls, takes)

    identity_by_take: dict[int, TakeIdentityResult] = {}
    for sel in selection.takes:
        # `is_located` and therefore `status` are UNCHANGED by the "streamed-gathering-treehouse"
        # plan's Stage 5 render-tier work: `method != "none"` already covers the new
        # `target_lost` literal too (its own `track_ids` is always `[]`, so `is_located` is
        # already False for it), and a profile-less `arrow_vote`/`heuristic_fallback` pick is
        # DELIBERATELY still `status="verified"` here -- CLAUDE.md §13.2's stat-card/live-panel
        # machinery already gates on `status == "verified"`, and the plan's own owner decision
        # ("keep tracking the pick, but stop calling it verified") is about the RENDERED CLAIM,
        # not about silently zeroing out today's stats for the unclicked filename-jersey flow.
        # `src.pipeline.annotated_video.render_tier` is what actually downgrades that claim (amber
        # box + "UNVERIFIED PICK" panel) by reading `location_method` below, never `status` itself
        # -- see that module's own docstring for the full three-tier rule.
        is_located = sel.method != "none" and bool(sel.track_ids)
        take_tracks_for_windows = full_tracks_by_take.get(sel.take_id, [])
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
            # Plan Stage 5 (second half): each stitched fragment's OWN real box-time-span, not the
            # pre-existing default "always active for the whole take" ({} -- see
            # `is_target_active_at`'s own docstring). Mirrors `src.pipeline.manual_events`'s own
            # population of this same field. A gap between two stitched fragments (an occlusion,
            # a break `stitch_timeline` didn't bridge) now genuinely renders as "not active" rather
            # than silently inheriting whatever the LAST fragment's box happened to be nearby.
            track_active_windows=_track_active_windows(sel.track_ids, take_tracks_for_windows),
        )

    goal_assist_by_take: dict[int, list[Event]] = defaultdict(list)
    for ev in goal_result.events:
        if ev.take_id is not None:
            goal_assist_by_take[ev.take_id].append(ev)

    full_events_cfg = configs["events"]
    full_selection_cfg = configs["highlights"]["selection"]
    full_attribution_cfg = configs["highlights"]["attribution"]
    full_key_moment_cfg = configs["key_moments"]

    # Stage D ("streamed-gathering-treehouse" plan): box and statistics share ONE identity. Every
    # per-take lookup below uses `_accepted_target_ids` -- the SAME accepted-fragment set that
    # gates the target's own red/amber box (Stage A/B/C) -- instead of `tir.location_track_ids`
    # directly, so a fragment the box gate refuses to draw (an unverified, merely geometry-stitched
    # candidate) can never still silently feed this take's own stats. A no-op for every
    # profile-less take (`_accepted_target_ids` falls back to `sel.track_ids` == the pre-existing
    # `tir.location_track_ids` exactly) -- see that function's own docstring.
    sel_by_take_id = {sel.take_id: sel for sel in selection.takes}
    target_profile_active = target_profile is not None

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

        sel = sel_by_take_id.get(take.id)
        accepted_ids = (
            _accepted_target_ids(sel, target_profile_active)
            if sel is not None
            else tir.location_track_ids
        )

        all_events, identity_of, _identity_confidence = compute_take_all_events(
            take,
            take_tracks,
            take_balls,
            frame_width,
            full_events_cfg,
            full_selection_cfg,
            fps_sample,
            drops=full_drops,
            kit_lab_by_track_id=kit_lab_by_track_id_by_take.get(take.id),
        )
        all_events = all_events + goal_assist_by_take.get(take.id, [])
        target_events = attribute_events_to_target(
            all_events,
            take_tracks,
            take_balls,
            accepted_ids,
            identity_of,
            full_attribution_cfg,
        )

        virtual_track = _stitched_virtual_track(take.id, take_tracks, accepted_ids)
        existing_windows = [
            (ev.t_start, ev.t_end)
            for ev in target_events
            if ev.type in (EventType.SPRINT, EventType.DRIBBLE, EventType.POSSESSION)
        ]
        candidates = find_candidate_windows(
            virtual_track, full_key_moment_cfg["prefilter"], existing_windows
        )
        key_moment_player_id = target_identity_id(accepted_ids, identity_of)
        if key_moment_player_id is None and accepted_ids:
            key_moment_player_id = accepted_ids[0]
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
        takes_used_by_number[tir.jersey_number].append((take.id, accepted_ids))

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
        target_timelines=target_timeline_by_take or None,
    )
    timings.append(
        StageTiming(
            stage="annotated_video", wall_seconds=round(time.time() - t0, 2), vram_peak_mb=None
        )
    )

    # --- Plan Stage 2 ("streamed-gathering-treehouse"): optional client-typed touch times --------
    # Parsed ONCE for the whole run (a single free-text field, not per-take/per-number) -- merged
    # in below, scoped to `target_jersey` only (the one player this flow's own click/filename
    # identity names; see this function's own docstring). Never raises on a malformed entry
    # (CLAUDE.md §10): unparseable entries are logged and carried into the run report, exactly like
    # `src.annotations.parse.parse_annotations`'s own `problems` list.
    manual_touch_cfg = configs["events"]["manual_touch"]
    manual_touch_problems: list[str] = []
    manual_touch_evts: list[Event] = []
    resolved_touch_jersey: int | None = None
    touch_jersey_reason = "no_manual_touch_times"
    if manual_touch_times:
        touch_seconds, manual_touch_problems = parse_touch_times(
            manual_touch_times, manual_touch_cfg
        )
        manual_touch_evts = manual_touch_events(touch_seconds, manual_touch_cfg)
        if manual_touch_problems:
            logger.warning(
                "manual touch times for %s: %d unparseable entr(ies) (see run_report.json): %s",
                video_path.name,
                len(manual_touch_problems),
                manual_touch_problems,
            )
        # Fix B: resolve the target jersey rather than requiring the (now-optional) typed field --
        # see `_resolve_manual_touch_target_jersey`'s own docstring for the full order/reasoning.
        resolved_touch_jersey, touch_jersey_reason = _resolve_manual_touch_target_jersey(
            target_jersey,
            target_profile.jersey_number if target_profile is not None else None,
            sorted(events_by_number.keys()),
        )
        if resolved_touch_jersey is None:
            logger.warning(
                "manual touch times for %s: %d typed time(s) could not be attached to any "
                "player -- ambiguous target (jerseys produced this run: %s, no typed jersey, no "
                "profile jersey) -- reported in run_report.json, never silently dropped",
                video_path.name,
                len(manual_touch_evts),
                sorted(events_by_number.keys()),
            )
        elif resolved_touch_jersey != target_jersey:
            logger.info(
                "manual touch times for %s: attached to jersey #%d (%s) -- not the typed "
                "target_jersey field (%s)",
                video_path.name,
                resolved_touch_jersey,
                touch_jersey_reason,
                target_jersey,
            )
    manual_touch_suppressed = 0

    t0 = time.time()
    for number in sorted(events_by_number.keys()):
        number_events = events_by_number.get(number, [])
        if manual_touch_evts and number == resolved_touch_jersey:
            # Owner's merge rule ("their times win, auto-detection fills the gaps"): applied ONCE,
            # scoped to the target jersey's own accumulated events across every take it appeared
            # in -- not per-take, since the client's typed times are absolute video timestamps that
            # could fall in any one of this player's takes, and a global merge here is simpler and
            # exactly as correct as a per-take merge would be (a manual touch and an auto touch it
            # suppresses are always compared within the same jersey number's own event list either
            # way). Logged, never silently dropped (CLAUDE.md §10).
            number_events, n_suppressed = merge_manual_touches(
                number_events, manual_touch_evts, manual_touch_cfg
            )
            events_by_number[number] = number_events
            manual_touch_suppressed += n_suppressed
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

    if manual_touch_times:
        all_dropped["manual_touch.suppressed_auto_touch"] = manual_touch_suppressed
        if manual_touch_problems:
            all_dropped["manual_touch.unparseable_entries"] = len(manual_touch_problems)
        if resolved_touch_jersey is None and manual_touch_evts:
            # Fix B: genuinely ambiguous target -- reported here (CLAUDE.md §10: "log everything
            # dropped"), never the old silent drop that `number == target_jersey` produced whenever
            # `target_jersey` was `None`.
            all_dropped["manual_touch.ambiguous_target_not_attached"] = len(manual_touch_evts)

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
    if manual_touch_times:
        report_dict["manual_touch"] = {
            "raw": manual_touch_times,
            "n_parsed": len(manual_touch_evts),
            "problems": manual_touch_problems,
            "n_suppressed_auto_touch": manual_touch_suppressed,
            # Fix B: which jersey the typed times actually attached to, and why -- see
            # `_resolve_manual_touch_target_jersey`'s own docstring for the resolution order.
            "resolved_target_jersey": resolved_touch_jersey,
            "target_jersey_resolution": touch_jersey_reason,
        }
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
        # Stages 1-4/6 ("streamed-gathering-treehouse" plan) -- the persistent `TargetProfile`
        # object (`src/track/target.py`) + its `verify_candidate` thresholds
        # (`src/track/target_verify.py`). Loaded unconditionally (cheap, tiny file, same "harmless
        # when unused" reasoning as the other stage-optional configs above); only actually
        # consulted when `target_profile is not None` is threaded into
        # `run_pipeline_for_video`/`select_targets` (apps/api/main.py's click endpoint today).
        "target": load_yaml("configs/target.yaml"),
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
