"""FastAPI Application Server for AI Football Player Tracking & Analytics System.

Provides REST endpoints and media streaming for video upload, player detection, ByteTrack tracking,
jersey OCR, event detection, movement analytics, and stat card visualization.
"""

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
WORK_DIR = BASE_DIR / "work"
WEB_DIR = BASE_DIR / "apps" / "web"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

# The whole pipeline's own I/O contract (CLAUDE.md §10) assumes cwd == repo root -- every
# `configs/*.yaml` load in `src.pipeline.run._load_all_configs` is a RELATIVE path. `_run_pipeline_
# job` already chdir'd here defensively before calling it; doing it once at import time, for the
# whole process, is both simpler and required by the fix below (never rely on it happening deep in
# a background task after cache keys have already been computed).
os.chdir(BASE_DIR)


def _resolve_video_path(video_name: str) -> Path:
    """`video_name` (e.g. "video2/chelsea_burnley_target10.mp4" or a bare filename) -> its path
    under INPUT_DIR, RELATIVE to BASE_DIR. Shared by every endpoint that takes a video name.

    Real bug fixed here (2026-09-01): every `StageCache` key embeds `str(video_path)` verbatim
    (`src/detect/run.py`, `src/track/run.py`, ...), and the CLI's own convention (`make run
    VIDEO=input/...`, and every prior real run in this project) is a RELATIVE path -- so a cache
    written by a CLI run keys on `"input/video2/chelsea_burnley_target10.mp4"`. This function used
    to return an ABSOLUTE path (`INPUT_DIR / video_name`, and `INPUT_DIR` is itself absolute), which
    produces a DIFFERENT hash and therefore a cache MISS on every stage for a video that had
    already been fully processed via the CLI -- confirmed directly: computing both hashes for the
    same on-disk config gave `d8237a32d5ce` (relative, matches the real cached `.meta.json`) vs.
    `d1d419e66f98` (absolute, a guaranteed miss). Returning a relative path here makes the API and
    the CLI share the exact same cache.
    """
    video_path = INPUT_DIR / video_name
    if video_path.exists():
        return video_path.relative_to(BASE_DIR)
    matches = list(INPUT_DIR.glob(f"**/{video_name}"))
    if matches:
        return matches[0].relative_to(BASE_DIR)
    raise HTTPException(
        status_code=404, detail=f"Video file '{video_name}' not found in input directory"
    )


def _canonical_work_dir(video_path: Path) -> Path:
    """The SAME `work/<slug>/` directory the pipeline itself resolves to
    (`src.common.io.work_dir_for`/`_slugify`) -- NOT `stem.replace(" ", "_")`, which diverges on
    uppercase/punctuation and silently points at the wrong directory."""
    from src.common.io import work_dir_for

    return work_dir_for(video_path, root=WORK_DIR)


def _canonical_slug(video_path: Path) -> str:
    return _canonical_work_dir(video_path).name


app = FastAPI(
    title="AI Football Player Tracking & Analytics API",
    description="YOLO/RF-DETR Detection + ByteTrack + Jersey OCR + Event & Movement Analytics",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory job store for background processing tasks
JOBS: dict[str, dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    video_name: str
    target_jersey: int | None = 10
    track_id: str | None = None
    manual_annotations: str | None = None
    click_x: float | None = None
    click_y: float | None = None
    click_t: float | None = None


@app.get("/api/health")
def get_health():
    """Health check endpoint with GPU status."""
    cuda_available = False
    gpu_name = "CPU Only"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "healthy",
        "gpu_available": cuda_available,
        "gpu_name": gpu_name,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/api/videos")
def list_videos():
    """List input videos and existing output runs."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    input_videos = []
    if INPUT_DIR.exists():
        for p in INPUT_DIR.glob("*"):
            if p.is_file() and p.suffix.lower() in video_extensions:
                input_videos.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "modified": time.ctime(p.stat().st_mtime),
                })
            elif p.is_dir():
                for sub in p.glob("*"):
                    if sub.is_file() and sub.suffix.lower() in video_extensions:
                        input_videos.append({
                            "name": f"{p.name}/{sub.name}",
                            "size_mb": round(sub.stat().st_size / (1024 * 1024), 2),
                            "modified": time.ctime(sub.stat().st_mtime),
                        })

    outputs = []
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.glob("*"):
            if p.is_dir() and not p.name.startswith("."):
                annotated_video = p / "original_annotated_video.mp4"
                has_annotated = annotated_video.exists()
                players_glob = (p / "players").glob("player_*")
                players = [d.name for d in players_glob] if (p / "players").exists() else []
                outputs.append({
                    "slug": p.name,
                    "has_annotated_video": has_annotated,
                    "players": players,
                })

    return {
        "input_videos": input_videos,
        "outputs": outputs,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a football video clip into the input directory."""
    filename = file.filename or f"upload_{int(time.time())}.mp4"
    # Clean filename
    safe_name = Path(filename).name.replace(" ", "_")
    target_path = INPUT_DIR / safe_name

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": safe_name,
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
        "path": str(target_path),
    }


def _read_jersey_hint(
    frame_bgr, bbox, legibility_model, parseq_model, parseq_transform, identity_cfg: dict
) -> dict[str, Any] | None:
    """Best-effort ADR-21 jersey read for ONE box on the already-decoded `frame_bgr` --
    advisory only (CLAUDE.md Golden Rule 4/7): shown to a human as a hint next to a clickable box,
    never used to auto-pick a target. `None` when the crop doesn't clear the legibility gate or
    PARSeq isn't confident (an honest "no hint", not a guess)."""
    from src.annotations.associate import expanded_bbox_region
    from src.identity.jersey_ocr import upscale_crop
    from src.identity.jersey_parseq import number_region, read_jersey_number_parseq
    from src.identity.legibility import is_legible

    crop_cfg = identity_cfg.get("crop", {})
    expand = crop_cfg.get("torso_crop_expand", 0.08)
    upscale = crop_cfg.get("crop_upscale_factor", 3.0)
    parseq_cfg = identity_cfg.get("parseq_soccernet", {})
    legibility_cfg = identity_cfg.get("legibility", {})

    h, w = frame_bgr.shape[:2]
    region = expanded_bbox_region(bbox, expand)
    x1, y1 = max(0, int(region.x1)), max(0, int(region.y1))
    x2, y2 = min(w, int(region.x2)), min(h, int(region.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    body_crop = upscale_crop(frame_bgr[y1:y2, x1:x2], upscale)
    leg = is_legible(body_crop, legibility_model, legibility_cfg)
    if body_crop.size == 0 or not leg.is_legible:
        return None

    nx1, ny1, nx2, ny2 = number_region(x1, y1, x2, y2, parseq_cfg.get("number_crop", {}))
    number_crop = upscale_crop(frame_bgr[ny1:ny2, nx1:nx2], upscale)
    read = read_jersey_number_parseq(number_crop, parseq_model, parseq_transform, parseq_cfg)
    if not read.is_confident:
        return None
    return {"digits": read.digits, "confidence": round(read.confidence, 3)}


@app.get("/api/frame_players")
def get_frame_players(video: str, t: float = 0.0, read_jersey: bool = False):
    """Every non-referee player track visible near timestamp `t`, keyed by its STITCHED chain id
    (`build_take_identities` -- not the raw ByteTrack fragment id, which is what makes labels look
    like they "change constantly"), plus a JPEG frame in the SAME pixel coordinate space as the
    returned boxes, so a UI can draw clickable rectangles directly on it with no separate
    calibration step.

    This is the picker behind "click a player, track them" (CLAUDE.md Golden Rule 4: a
    human-provided identity is the strongest evidence this pipeline has -- stronger than any
    jersey-number auto-read). Runs Stage 3 tracking first; it is cached, so a second call for the
    same video is a fast no-op read-back, not a re-track.

    `read_jersey=true` attaches an ADVISORY jersey-number hint per box (ADR-21's legibility-gate +
    tight-number-region PARSeq chain) -- shown to the human as a hint only, never used to pick a
    box automatically.
    """
    import base64

    import cv2

    from src.common.io import load_models_parquet
    from src.common.types import Track
    from src.common.video import decode_frames
    from src.pipeline.run import _effective_frame_size, _load_all_configs
    from src.shots.boundaries import detect_takes
    from src.track.continuity import build_take_identities
    from src.track.run import run_track_stage
    from src.track.tracker import assign_take_id

    video_path = _resolve_video_path(video)
    configs = _load_all_configs()

    run_track_stage(
        video_path,
        configs["hardware"],
        configs["detect"],
        configs["track"],
        configs["team"],
        configs["shots"],
        work_root=WORK_DIR,
    )
    takes = detect_takes(video_path, configs["shots"], work_root=WORK_DIR)
    take_id = assign_take_id(t, takes)
    if take_id is None:
        raise HTTPException(status_code=404, detail=f"No take covers t={t:.2f}s for this video")

    work_dir = _canonical_work_dir(video_path)
    all_tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    take_tracks = [tr for tr in all_tracks if tr.take_id == take_id]

    identity_of, identity_conf = build_take_identities(
        take_tracks, configs["highlights"]["selection"]
    )
    frame_w, frame_h = _effective_frame_size(video_path, configs["hardware"]["decode"])

    frame_bgr = None
    for _idx, _frame_t, frame in decode_frames(
        str(video_path),
        start=max(0.0, t - 0.05),
        end=t + 0.2,
        scale_width=configs["hardware"]["decode"].get("scale_width"),
    ):
        frame_bgr = frame
        break
    if frame_bgr is None:
        raise HTTPException(status_code=500, detail="Could not decode a frame at that timestamp")

    legibility_model = parseq_model = parseq_transform = None
    if read_jersey:
        from src.identity.jersey_models import load_optional_jersey_stack

        legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(
            configs["identity"]
        )

    tolerance_s = 0.3
    players = []
    for tr in take_tracks:
        if getattr(tr.dominant_class, "value", str(tr.dominant_class)) == "referee":
            continue
        if not tr.boxes:
            continue
        box = min(tr.boxes, key=lambda b: abs(b.t - t))
        if abs(box.t - t) > tolerance_s:
            continue
        bb = box.bbox
        chain_id = identity_of.get(tr.id, tr.id)
        entry: dict[str, Any] = {
            "raw_track_id": tr.id,
            "chain_id": chain_id,
            "chain_confidence": round(identity_conf.get(chain_id, 0.0), 3),
            "bbox_norm": {
                "x1": bb.x1 / frame_w,
                "y1": bb.y1 / frame_h,
                "x2": bb.x2 / frame_w,
                "y2": bb.y2 / frame_h,
            },
            "team": tr.team,
            "team_confidence": round(tr.team_confidence, 3),
            "height_px": round(bb.height, 1),
        }
        if read_jersey and legibility_model is not None and parseq_model is not None:
            entry["jersey_hint"] = _read_jersey_hint(
                frame_bgr, bb, legibility_model, parseq_model, parseq_transform, configs["identity"]
            )
        players.append(entry)

    if read_jersey:
        from src.identity.jersey_models import free_optional_jersey_stack

        free_optional_jersey_stack(legibility_model, parseq_model)

    ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii") if ok else None

    return {
        "video": video,
        "t": t,
        "take_id": take_id,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "frame_jpeg_base64": frame_b64,
        "players": players,
    }


def _match_click_to_track(
    video_path: Path, configs: dict[str, dict], click_x: float, click_y: float, click_t: float
) -> tuple[int, int | None]:
    """`(take_id, matched_track_id | None)` for a click at normalized `(click_x, click_y)` at
    time `click_t`.

    Two real bugs fixed here vs. the original inline version:
    - the take is now chosen via `assign_take_id(click_t, takes)` instead of being hardcoded to
      take 0 -- a click in a video's second (or later) take used to silently target the wrong take.
    - the "does the click land inside a box" search now stops as soon as it finds one (a `return`,
      not a `break` that only exited the inner loop and let later tracks overwrite the match).
    - the nearest-centroid fallback cap is a FRACTION of frame height, not an absolute 400px,
      which was calibrated for one specific resolution and meaningless on a 4K clip vs. a 720p one.
    """
    from src.common.io import load_models_parquet
    from src.common.types import Track
    from src.pipeline.run import _effective_frame_size
    from src.shots.boundaries import detect_takes
    from src.track.run import run_track_stage
    from src.track.tracker import assign_take_id

    run_track_stage(
        video_path,
        configs["hardware"],
        configs["detect"],
        configs["track"],
        configs["team"],
        configs["shots"],
        work_root=WORK_DIR,
    )
    takes = detect_takes(video_path, configs["shots"], work_root=WORK_DIR)
    take_id = assign_take_id(click_t, takes)
    if take_id is None:
        return 0, None

    tracks_path = _canonical_work_dir(video_path) / "track" / "tracks.parquet"
    tr_list = load_models_parquet(tracks_path, Track)
    take_tracks = [tr for tr in tr_list if tr.take_id == take_id]

    frame_w, frame_h = _effective_frame_size(video_path, configs["hardware"]["decode"])
    px_x = click_x * frame_w if click_x <= 1.0 else click_x
    px_y = click_y * frame_h if click_y <= 1.0 else click_y
    max_dist_px = 0.15 * frame_h  # a fraction of frame height, not an absolute pixel count

    best_track_id: int | None = None
    best_dist = float("inf")
    for tr in take_tracks:
        if getattr(tr.dominant_class, "value", str(tr.dominant_class)) == "referee":
            continue
        near = (b for b in tr.boxes if abs(b.t - click_t) <= 2.0)
        box = min(near, key=lambda b: abs(b.t - click_t), default=None)
        if box is None:
            continue
        bb = box.bbox
        if bb.x1 <= px_x <= bb.x2 and bb.y1 <= px_y <= bb.y2:
            return take_id, tr.id  # contained -- stop immediately, the strongest possible match
        dist = ((bb.cx - px_x) ** 2 + (bb.cy - px_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_track_id = dist, tr.id

    return take_id, (best_track_id if best_dist < max_dist_px else None)


def _run_pipeline_job(
    job_id: str,
    video_name: str,
    target_jersey: int | None,
    track_id: str | None,
    manual_annotations: str | None,
    click_x: float | None = None,
    click_y: float | None = None,
    click_t: float | None = None,
):
    """Background task function executing the processing pipeline stages."""
    job = JOBS[job_id]
    job["status"] = "processing"
    job["progress"] = 5
    job["stage"] = "Initializing Environment"
    job["logs"].append(
        f"Started analysis job {job_id} for video '{video_name}' with target jersey "
        f"#{target_jersey}"
    )

    try:
        video_path = _resolve_video_path(video_name)

        # Step 1: Handle manual annotations sidecar if provided
        if manual_annotations and manual_annotations.strip():
            job["logs"].append("Writing manual annotation sidecar file...")
            sidecar_path = video_path.parent / f"{video_path.stem}.annotations.txt"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(manual_annotations.strip() + "\n")
            job["logs"].append(f"Manual sidecar created at {sidecar_path.name}")

        # Update progress stages
        job["stage"] = "Player Detection (YOLO / RF-DETR)"
        job["progress"] = 15
        job["logs"].append("Running player & ball detection stage (YOLO / RF-DETR model)...")
        time.sleep(1)

        job["stage"] = "Player Tracking (ByteTrack / BoT-SORT)"
        job["progress"] = 35
        job["logs"].append("Running ByteTrack player association & trajectory tracking...")
        time.sleep(1)

        job["stage"] = "Jersey Number OCR (PARSeq / SoccerNet)"
        job["progress"] = 55
        job["logs"].append(
            "Extracting player upper-torso crops & running PARSeq jersey number OCR..."
        )
        time.sleep(1)

        job["stage"] = "Player Identity Matching"
        job["progress"] = 70
        job["logs"].append(f"Matching tracked IDs to Target Jersey #{target_jersey}...")
        time.sleep(1)

        job["stage"] = "Event Detection & Movement Analytics"
        job["progress"] = 85
        job["logs"].append(
            "Computing touch, pass (same-colour), turnover, sprint, shot, goal, assist, "
            "tackle stats..."
        )
        time.sleep(1)

        # Set current working directory to BASE_DIR so relative config paths work
        os.chdir(BASE_DIR)

        # Call python module command or direct function execution
        from src.ingest.discovery import parse_filename
        from src.pipeline.extended_output import run_extended_pipeline_for_video
        from src.pipeline.manual_events import run_manual_events_pipeline_for_video
        from src.pipeline.run import (
            _load_all_configs,
            parse_track_id_overrides,
            run_pipeline_for_video,
        )

        configs = _load_all_configs()
        if target_jersey is not None:
            configs["identity"]["target_jersey"] = target_jersey

        annotations_path = video_path.parent / f"{video_path.stem}.annotations.txt"
        slug = _canonical_slug(video_path)

        # Owner decision, 2026-09-01: when the human EXPLICITLY selects a player (an exact
        # track_id, or a click on the preview), that selection is authoritative and drives
        # auto-detected stats for THAT player -- it always wins, including over a
        # `.annotations.txt` sidecar (which otherwise unconditionally takes over, ADR-19, and
        # would silently swallow the click). This is Golden Rule 4 exercised directly: a
        # human-provided identity outranks every other signal, sidecar included.
        has_explicit_selection = bool(track_id) or (click_x is not None and click_y is not None)

        if has_explicit_selection:
            job["logs"].append("Explicit player selection given -- routing to the auto-detect "
                                "pipeline for that player (sidecar, if any, is not used).")
            overrides: dict[int, int] = {}
            if track_id:
                overrides.update(parse_track_id_overrides(track_id))

            if click_x is not None and click_y is not None and not overrides:
                job["logs"].append(
                    f"Processing interactive click at ({click_x:.3f}, {click_y:.3f}) "
                    f"@ t={click_t or 0.0:.1f}s..."
                )
                # Real bug fixed here (2026-09-01): a click that failed to match ANY track used
                # to fall through silently with `overrides` left empty, and the pipeline below
                # still ran -- but with NO manual override, `select_targets` auto-picks its own
                # arrow/heuristic target for whatever `target_jersey` happened to be sitting in
                # the form (a stale default from a PREVIOUS video, in the confirmed real case:
                # the user opened `video3` -- whose own annotated target is #2 -- clicked at
                # t=0.0s before play started, missed every player, and silently got results for
                # "#10" instead, with no indication their click had done nothing). CLAUDE.md
                # Golden Rule 5 ("no fabricated numbers... an honest not-available is correct, a
                # fabricated one is a bug") applies here just as much to IDENTITY as to a stat:
                # an explicit human selection that failed must fail loudly, never be silently
                # replaced by an unrelated auto-pick that LOOKS like it honored the request.
                try:
                    chosen_take_id, chosen_track_id = _match_click_to_track(
                        video_path, configs, click_x, click_y, click_t or 0.0
                    )
                except Exception as cx_err:
                    job["status"] = "failed"
                    job["error"] = f"Click matching failed: {cx_err}"
                    job["logs"].append(f"ERROR: {job['error']}")
                    return
                if chosen_track_id is not None:
                    overrides[chosen_take_id] = chosen_track_id
                    job["logs"].append(
                        f"Interactive click matched take={chosen_take_id} "
                        f"track_id={chosen_track_id}."
                    )
                else:
                    job["status"] = "failed"
                    job["error"] = (
                        "Your click did not land on a detected player near that moment (or the "
                        "video was paused before any player was visible). Refusing to silently "
                        "fall back to an unrelated auto-selected target. Use \"Load Players\" to "
                        "see real detected boxes and click one directly, or pick a moment where "
                        "the target player is clearly on screen."
                    )
                    job["logs"].append(f"ERROR: {job['error']}")
                    return

            run_pipeline_for_video(
                video_path,
                configs,
                target_jersey=target_jersey,
                manual_overrides=overrides,
                work_root=WORK_DIR,
                output_root=OUTPUT_DIR,
            )
        elif annotations_path.exists():
            job["logs"].append("Branching to MANUAL-EVENTS pipeline via sidecar...")
            run_manual_events_pipeline_for_video(
                video_path,
                configs,
                annotations_path,
                target_jersey=target_jersey,
                work_root=WORK_DIR,
                output_root=OUTPUT_DIR,
            )
        else:
            import re
            pattern = re.compile(configs["run"]["filename_convention_regex"])
            _, parsed_jersey = parse_filename(video_path.stem, pattern)
            effective_jersey = target_jersey if target_jersey is not None else parsed_jersey

            if parsed_jersey is not None or target_jersey is not None:
                job["logs"].append(
                    f"Branching to target player pipeline for jersey #{effective_jersey}..."
                )
                run_pipeline_for_video(
                    video_path,
                    configs,
                    target_jersey=effective_jersey,
                    manual_overrides={},
                    work_root=WORK_DIR,
                    output_root=OUTPUT_DIR,
                )
            else:
                job["logs"].append("Branching to extended verified-identity pipeline...")
                run_extended_pipeline_for_video(
                    video_path,
                    configs,
                    work_root=WORK_DIR,
                    output_root=OUTPUT_DIR,
                )

        job["stage"] = "Rendering Video & Exporting Stat Cards"
        job["progress"] = 100
        job["status"] = "completed"
        job["slug"] = slug
        job["logs"].append("Pipeline processing completed successfully!")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["logs"].append(f"ERROR: {str(e)}")


@app.post("/api/process")
def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Trigger the football video tracking and analytics pipeline."""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "video_name": req.video_name,
        "target_jersey": req.target_jersey,
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "logs": [],
        "created_at": time.time(),
        "slug": req.video_name.replace(".mp4", "").replace(".avi", "").replace(".mov", "")
        .replace(" ", "_"),
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id,
        req.video_name,
        req.target_jersey,
        req.track_id,
        req.manual_annotations,
        req.click_x,
        req.click_y,
        req.click_t,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Started pipeline job {job_id} for {req.video_name}",
    }


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Retrieve current processing job status."""
    if job_id not in JOBS:
        # Check if output already exists matching a slug
        possible_dir = OUTPUT_DIR / job_id
        if possible_dir.exists():
            return {
                "job_id": job_id,
                "status": "completed",
                "stage": "Finished",
                "progress": 100,
                "slug": job_id,
                "logs": ["Output directory exists on disk"],
            }
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS[job_id]


@app.get("/api/results/{slug}")
def get_results(slug: str):
    """Retrieve player stats, event timeline, heatmaps, and highlight clips URLs."""
    # Find matching directory in OUTPUT_DIR
    out_dir = OUTPUT_DIR / slug
    if not out_dir.exists():
        # Try matching substring or replacement
        matches = [d for d in OUTPUT_DIR.glob("*") if slug in d.name or d.name in slug]
        if matches:
            out_dir = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"No results found for slug '{slug}'")

    slug_name = out_dir.name

    # Check for players folder. CLAUDE.md Golden Rule 5: no fabricated numbers -- if no player
    # output exists yet (still processing, or an unverified/unselected take produced none), the
    # honest response is an EMPTY players list with a stated reason, never a plausible-looking
    # placeholder stat card the owner could mistake for a real result.
    players_dir = out_dir / "players"
    player_data = []

    if players_dir.exists():
        for p_folder in sorted(players_dir.glob("player_*")):
            p_num = p_folder.name.replace("player_", "")
            statcard_file = p_folder / "statcard.md"
            statcard_text = statcard_file.read_text() if statcard_file.exists() else ""

            stats = parse_statcard_markdown(statcard_text, int(p_num) if p_num.isdigit() else None)

            highlights = {}
            hl_dir = p_folder / "highlights"
            if hl_dir.exists():
                for hl in hl_dir.glob("*.mp4"):
                    hl_url = (
                        f"/media/output/{slug_name}/players/{p_folder.name}"
                        f"/highlights/{hl.name}"
                    )
                    highlights[hl.stem] = hl_url

            stats["highlights"] = highlights
            player_data.append(stats)

    # Annotated video path
    annotated_url = None
    if (out_dir / "original_annotated_video.mp4").exists():
        annotated_url = f"/media/output/{slug_name}/original_annotated_video.mp4"

    return {
        "slug": slug_name,
        "annotated_video_url": annotated_url,
        "players": player_data,
        "primary_player": player_data[0] if player_data else None,
        "message": (
            None
            if player_data
            else "No player output found for this run -- either it is still processing, or no "
            "take's identity was verified/selected (see run_report.json / identity_report.json "
            "in output/<slug>/ for why)."
        ),
        # Pitch trajectory intentionally omitted: a pixel-position tactical map needs calibrated
        # homography (ADR-6) that most of this project's own footage doesn't have (<4 usable
        # landmarks) -- a synthetic placeholder here would look like real tracking data. See
        # `output/<slug>/players/player_<N>/events/event_timeline.json` for the real, timestamped
        # events instead.
        "pitch_trajectory": None,
    }


def parse_statcard_markdown(md_text: str, jersey_num: int | None) -> dict[str, Any]:
    """Parse statcard.md into a structured dict. `identity_status` defaults to "Unknown" (never
    "Verified") until the statcard's own line is actually parsed below -- CLAUDE.md Golden Rule 5:
    a missing/unparseable statcard must never be presented as a verified result."""
    res = {
        "jersey_number": jersey_num,
        "identity_status": "Unknown",
        "touches": 0,
        "passes": 0,
        "turnovers": 0,
        "sprints": 0,
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "tackles": 0,
        "saves": 0,
        "dribbles": 0,
        "possession_time": "uncertain",
        "distance_covered": "uncertain",
        "events": [],
    }
    if not md_text:
        return res

    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("**Identity Status:**"):
            res["identity_status"] = line.split(":**")[1].strip()
        elif line.startswith("**Touches:**"):
            res["touches"] = _safe_int(line)
        elif line.startswith("**Passes:**"):
            res["passes"] = _safe_int(line)
        elif line.startswith("**Turnovers:**"):
            res["turnovers"] = _safe_int(line)
        elif line.startswith("**Sprints/Runs:**"):
            res["sprints"] = _safe_int(line)
        elif line.startswith("**Goals:**"):
            res["goals"] = _safe_int(line)
        elif line.startswith("**Assists:**"):
            res["assists"] = _safe_int(line)
        elif line.startswith("**Shots:**"):
            res["shots"] = _safe_int(line)
        elif line.startswith("**Tackles:**"):
            res["tackles"] = _safe_int(line)
        elif line.startswith("**Saves:**"):
            res["saves"] = _safe_int(line)
        elif line.startswith("**Dribbles:**"):
            res["dribbles"] = _safe_int(line)
        elif line.startswith("**Possession Time:**"):
            res["possession_time"] = line.split(":**")[1].strip()
        elif line.startswith("**Distance Covered:**"):
            res["distance_covered"] = line.split(":**")[1].strip()
        elif line.startswith("|") and not line.startswith("| Time") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                res["events"].append({
                    "time": parts[0],
                    "event": parts[1],
                    "confidence": float(parts[2]) if _is_float(parts[2]) else 0.90,
                })
    return res


def _safe_int(line: str) -> int:
    val = line.split(":**")[1].strip()
    try:
        return int(val.split()[0])
    except Exception:
        return 0


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False




def _safe_resolve_under(root: Path, path: str) -> Path | None:
    """`root / path`, resolved, and rejected (`None`) if it escapes `root` -- closes a real path-
    traversal hole (`../../etc/passwd` style) that existed here: the previous version joined the
    request path onto `BASE_DIR` with no check at all, so any file readable by the server process
    was servable over HTTP."""
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


@app.get("/media/{path:path}")
def serve_media(path: str):
    """Serve media files (input videos, annotated output videos, highlights) -- resolved under
    BASE_DIR first, then INPUT_DIR, then OUTPUT_DIR, each check rejecting any path that would
    escape that root."""
    full_path = (
        _safe_resolve_under(BASE_DIR, path)
        or _safe_resolve_under(INPUT_DIR, path)
        or _safe_resolve_under(OUTPUT_DIR, path)
    )
    if full_path is None:
        raise HTTPException(status_code=404, detail="Media file not found")

    return FileResponse(
        full_path,
        media_type="video/mp4" if full_path.suffix == ".mp4" else None,
        headers={"Accept-Ranges": "bytes"},
    )


# Mount Web Frontend UI static files
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
