"""The gateway's job queue: what the user pressed Run on, and where it is right now.

A job here wraps a pipeline job on the pod (`apps/api/main.py::process_video`). The gateway owns
the parts the pod cannot: waiting for a GPU, pushing the uploaded file across, and surviving a
pod that was recreated (or a browser that was refreshed) mid-way. The status shape mirrors the
pod's `/api/status/{job_id}` -- `status`, `stage`, `progress`, `logs`, `slug`, `error` -- so the
existing frontend polling (`apps/web/js/app.js::checkJobStatus`) needs no new branches; the only
additions are the pre-pod statuses and `pod_phase`.

Statuses:
    queued          waiting for the single worker (jobs run one at a time -- one GPU, ~$14 credit)
    waiting_gpu     RunPod has no GPU right now; the worker retries until one appears
    starting_gpu    pod being started / created / booting
    uploading       pushing the video to the pod
    running         the pod's pipeline is working; stage/progress/logs mirror the pod's
    completed / failed

Persistence: the queue is a JSON file under the gateway's data dir, rewritten on every change, so
a VPS reboot or a gateway restart resumes queued jobs and re-attaches to a running one instead of
forgetting it. Terminal jobs are kept (capped) so a refreshed page can still show them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from apps.gateway.pod_client import PodClient
from apps.gateway.pod_manager import ONLINE, PodManager, PodUnavailable

logger = logging.getLogger(__name__)

QUEUED = "queued"
WAITING_GPU = "waiting_gpu"
STARTING_GPU = "starting_gpu"
UPLOADING = "uploading"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
ACTIVE_STATUSES = (QUEUED, WAITING_GPU, STARTING_GPU, UPLOADING, RUNNING)

MAX_KEPT_JOBS = 50
MAX_LOG_LINES = 400
POD_POLL_S = 2.0
# The pod's idle watchdog never stops a pod with a job in flight, so a pod that stops answering
# mid-job has really died (or RunPod took the host). Give it this long before failing the job.
POD_UNREACHABLE_GRACE_S = 10 * 60.0
IDLE_SLEEP_S = 2.0


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load()

    # ---------------------------------------------------------------- persistence

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("jobs file %s unreadable (%s); starting empty", self.path, exc)
            return
        self._jobs = {j["job_id"]: j for j in raw.get("jobs", []) if "job_id" in j}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"])
        # Keep every active job plus the newest terminal ones.
        active = [j for j in jobs if j["status"] in ACTIVE_STATUSES]
        done = [j for j in jobs if j["status"] not in ACTIVE_STATUSES][-MAX_KEPT_JOBS:]
        keep = sorted(active + done, key=lambda j: j["created_at"])
        self._jobs = {j["job_id"]: j for j in keep}
        tmp.write_text(json.dumps({"jobs": keep}, indent=1))
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- CRUD

    def create(self, video_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            job_id = uuid.uuid4().hex[:8]
            job = {
                "job_id": job_id,
                "video_name": video_name,
                "payload": payload,
                "status": QUEUED,
                "stage": "Queued",
                "progress": 0,
                "logs": ["Queued -- waiting for the GPU pod."],
                "slug": None,
                "error": None,
                "pod_phase": "",
                "pod_job_id": None,
                "created_at": time.time(),
                "updated_at": time.time(),
                "finished_at": None,
            }
            self._jobs[job_id] = job
            self._save()
            return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        """What the browser sees: everything but the original request payload."""
        return {k: v for k, v in job.items() if k != "payload"}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.public(j) for j in sorted(self._jobs.values(), key=lambda j: j["created_at"])]

    def active(self) -> list[dict[str, Any]]:
        return [j for j in self.list() if j["status"] in ACTIVE_STATUSES]

    def next_queued(self) -> dict[str, Any] | None:
        with self._lock:
            for job in sorted(self._jobs.values(), key=lambda j: j["created_at"]):
                if job["status"] in ACTIVE_STATUSES:
                    return dict(job)
            return None

    def video_in_flight(self, video_name: str) -> bool:
        with self._lock:
            return any(j["video_name"] == video_name and j["status"] in ACTIVE_STATUSES for j in self._jobs.values())

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            log = fields.pop("log", None)
            job.update(fields)
            if log:
                if not job["logs"] or job["logs"][-1] != log:
                    job["logs"].append(log)
                    del job["logs"][:-MAX_LOG_LINES]
            job["updated_at"] = time.time()
            if job["status"] in (COMPLETED, FAILED) and not job.get("finished_at"):
                job["finished_at"] = time.time()
            self._save()
            return dict(job)


class JobWorker:
    """One background thread: takes the oldest active job, gets the pod online, pushes the file,
    starts the pipeline, mirrors its progress. Jobs are strictly sequential."""

    def __init__(
        self,
        store: JobStore,
        pods: PodManager,
        pod_client: PodClient,
        input_dir: Path,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.pods = pods
        self.pod = pod_client
        self.input_dir = input_dir
        self._sleep = sleep
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="gateway-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._sleep(IDLE_SLEEP_S)
                continue
            try:
                self.run_job(job)
            except Exception as exc:  # never let one job kill the worker
                logger.exception("job %s crashed", job["job_id"])
                self.store.update(job["job_id"], status=FAILED, stage="Failed", error=str(exc), log=f"Gateway error: {exc}")

    # ---------------------------------------------------------------- one job

    def run_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        video_name = job["video_name"]
        local = self.input_dir / video_name

        # A gateway restart mid-run: the pod may still be working on it -- re-attach, don't restart.
        if job["status"] == RUNNING and job.get("pod_job_id"):
            self._ensure_pod(job_id)
            self._mirror_until_done(job_id, job["pod_job_id"])
            return

        # 1. GPU pod
        self._ensure_pod(job_id)

        # 2. The file. Uploaded to the VPS; the pod's volume may already have it from a past run.
        if local.exists():
            try:
                present = self.pod.has_input_video(video_name, local.stat().st_size)
            except requests.RequestException as exc:
                present = False
                logger.warning("could not list pod videos: %s", exc)
            if not present:
                self.store.update(job_id, status=UPLOADING, stage="Uploading video to the GPU pod", progress=0,
                                  log=f"Uploading {video_name} to the GPU pod...")

                def on_upload(frac: float) -> None:
                    self.store.update(job_id, stage=f"Uploading video to the GPU pod ({int(frac * 100)}%)")

                self.pod.upload_video(local, upload_id=f"gw{job_id}{uuid.uuid4().hex[:16]}", on_progress=on_upload)
                self.store.update(job_id, log="Upload to the GPU pod complete.")
        else:
            # Not on the VPS (uploaded straight to the pod in the pre-gateway days): the pod must
            # already have it, or /api/process there will 404 with a clear message.
            logger.info("job %s: %s is not on the VPS; relying on the pod's copy", job_id, video_name)

        # 3. Start the pipeline on the pod.
        self.store.update(job_id, status=RUNNING, stage="Starting pipeline", progress=0, log="Starting the pipeline on the GPU pod...")
        started = self.pod.start_process(job["payload"])
        pod_job_id = started.get("job_id")
        if not pod_job_id:
            raise RuntimeError(f"pod returned no job id: {started}")
        self.store.update(job_id, pod_job_id=pod_job_id)

        # 4. Mirror progress until it ends.
        self._mirror_until_done(job_id, pod_job_id)

    def _ensure_pod(self, job_id: str) -> None:
        def on_progress(phase: str, message: str) -> None:
            status = WAITING_GPU if phase == "waiting_for_gpu" else STARTING_GPU
            stage = {
                "waiting_for_gpu": "Waiting for a free GPU",
                "starting": "Starting the GPU pod",
                "booting": "GPU pod booting",
                "online": "GPU pod online",
                "error": "GPU pod error",
            }.get(phase, "Preparing the GPU pod")
            self.store.update(job_id, status=status, stage=stage, progress=0, pod_phase=phase, log=message)

        if self.pods.online:
            self.pods.refresh()
        if self.pods.state.phase != ONLINE:
            on_progress(self.pods.state.phase or "starting", "Checking the GPU pod...")
        try:
            self.pods.ensure_online(on_progress)
        except PodUnavailable as exc:
            raise RuntimeError(f"GPU pod unavailable: {exc}") from exc
        self.store.update(job_id, pod_phase=ONLINE)

    def _mirror_until_done(self, job_id: str, pod_job_id: str) -> None:
        # The pod keeps the full pipeline log; mirror all of it after whatever the gateway logged
        # before handing over, so nothing between two polls is lost.
        current = self.store.get(job_id) or {}
        prefix_logs = [line for line in current.get("logs", []) if not line.startswith("[pod] ")]
        unreachable_since: float | None = None
        while not self._stop.is_set():
            try:
                pod_job = self.pod.job_status(pod_job_id)
                unreachable_since = None
            except requests.RequestException as exc:
                now = self._clock()
                unreachable_since = unreachable_since or now
                waited = now - unreachable_since
                self.store.update(job_id, stage=f"GPU pod not answering ({int(waited)} s) -- retrying",
                                  log=f"GPU pod not answering: {exc}")
                if waited > POD_UNREACHABLE_GRACE_S:
                    raise RuntimeError("the GPU pod stopped answering mid-job; it may have died -- run it again") from exc
                self._sleep(max(POD_POLL_S, 5.0))
                continue

            status = pod_job.get("status", RUNNING)
            fields: dict[str, Any] = {
                "stage": pod_job.get("stage") or "Processing",
                "progress": pod_job.get("progress", 0),
                "slug": pod_job.get("slug"),
            }
            pod_logs = [f"[pod] {line}" for line in (pod_job.get("logs") or [])]
            fields["logs"] = (prefix_logs + pod_logs)[-MAX_LOG_LINES:]
            if status == COMPLETED:
                fields.update(status=COMPLETED, progress=100)
                self.store.update(job_id, **fields)
                return
            if status == FAILED:
                fields.update(status=FAILED, error=pod_job.get("error") or "pipeline failed on the pod")
                self.store.update(job_id, **fields)
                return
            fields["status"] = RUNNING
            self.store.update(job_id, **fields)
            self._sleep(POD_POLL_S)
