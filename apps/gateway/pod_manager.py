"""Keeps the GPU pod alive exactly when someone needs it (docs/DEPLOY.md "Always-on gateway").

The gateway on the VPS is always up; the RunPod pod is not -- it stops itself when idle
(`apps/api/idle_stop.py`) and, because a STOPPED pod does not reserve its GPU, it often cannot
simply be started again. `PodManager.ensure_online()` is the one place that knows how to get from
"whatever state RunPod is in" to "the app answers on its proxy URL":

    no pod            -> create one on any GPU from the preference list
    create refused    -> WAITING_FOR_GPU: tell the user, retry every `retry_s`, never give up
    pod stopped       -> Start; if RunPod refuses (GPU taken) -> terminate it, then create
    pod running       -> wait for /api/health (BOOTING); a boot that never comes up is torn
                         down and recreated once the cap passes

Every transition is reported through `on_progress(phase, message)` so the job that is waiting
can show the user exactly what is going on ("No GPU available right now -- waiting"). The
manager never raises for "no GPU": that is a normal state here, not an error. It raises only for
things a retry cannot fix (a bad API key), so the job can fail loudly instead of spinning.

Pure-ish by construction: `clock` and `sleep` are injectable, and the RunPod client is a
parameter, so the whole state machine is unit-tested with a fake API and no real time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

from apps.gateway import runpod_pods as rp
from apps.gateway.runpod_pods import PodInfo, RunPodClient, RunPodError

logger = logging.getLogger(__name__)

# Phases, as shown to the user. Strings on purpose: they go straight into JSON.
UNKNOWN = "unknown"  # never looked yet, or RunPod unreachable
OFFLINE = "offline"  # pod stopped (or none exists) and nobody is asking for it
STARTING = "starting"  # Start accepted / pod created; waiting for RunPod to run the container
BOOTING = "booting"  # container running, app not answering yet
WAITING_FOR_GPU = "waiting_for_gpu"  # RunPod has no GPU of the wanted types right now
ONLINE = "online"  # /api/health answers
ERROR = "error"  # something a retry won't fix (auth); a human must look

DEFAULT_POLL_S = 10.0  # while starting/booting
DEFAULT_RETRY_S = 60.0  # while waiting for a GPU
DEFAULT_BOOT_CAP_S = 30 * 60.0  # a first boot installs deps (~10 min); anything past this is stuck
DEFAULT_TERMINATE_WAIT_S = 90.0
# A replacement pod is created WITH a gpuTypeId, so it coming up GPU-less means something is
# wrong at RunPod's end; don't burn credit recreating forever.
MAX_NO_GPU_REPLACEMENTS = 3


@dataclass
class PodConfig:
    name: str = rp.DEFAULT_POD_NAME
    volume_name: str = rp.DEFAULT_VOLUME_NAME
    volume_gb: int = rp.DEFAULT_VOLUME_GB
    datacenter: str = rp.DEFAULT_DATACENTER
    cloud: str = rp.DEFAULT_CLOUD
    gpu_types: tuple[str, ...] = rp.DEFAULT_GPU_TYPES
    port: int = rp.DEFAULT_PORT
    # Everything the bootstrap needs on the pod (docker/runpod.env.example).
    pod_env: dict[str, str] = field(default_factory=dict)
    poll_s: float = DEFAULT_POLL_S
    retry_s: float = DEFAULT_RETRY_S
    boot_cap_s: float = DEFAULT_BOOT_CAP_S
    terminate_wait_s: float = DEFAULT_TERMINATE_WAIT_S


@dataclass
class PodState:
    phase: str = UNKNOWN
    message: str = ""
    pod_id: str = ""
    proxy_url: str = ""
    gpu_type: str = ""
    gpu_name: str = ""  # what torch on the pod reports, once online
    cost_per_hr: float = 0.0
    since: float = 0.0  # wall-clock when `phase` was entered
    checked_at: float = 0.0
    last_error: str = ""
    waiting_since: float = 0.0  # wall-clock when WAITING_FOR_GPU began (0 when not waiting)

    def public(self) -> dict[str, Any]:
        """What `/api/health` exposes without a login: no pod id, no cost."""
        return {"phase": self.phase, "message": self.message, "gpu_name": self.gpu_name}

    def full(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "message": self.message,
            "pod_id": self.pod_id,
            "proxy_url": self.proxy_url,
            "gpu_type": self.gpu_type,
            "gpu_name": self.gpu_name,
            "cost_per_hr": self.cost_per_hr,
            "since": self.since,
            "checked_at": self.checked_at,
            "last_error": self.last_error,
            "waiting_since": self.waiting_since,
        }


class PodUnavailable(Exception):
    """`ensure_online` gave up: only for errors a retry cannot fix (auth) or a caller-imposed deadline."""


ProgressFn = Callable[[str, str], None]


def _no_progress(_phase: str, _message: str) -> None:
    pass


def _fmt_minutes(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    return f"{minutes} min"


class PodManager:
    def __init__(
        self,
        client: RunPodClient,
        config: PodConfig,
        *,
        health_session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.cfg = config
        self._health_session = health_session or requests.Session()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()  # serialises ensure_online callers; refresh() is lock-free
        self.state = PodState()
        self._volume_id: str = ""

    # ---------------------------------------------------------------- state helpers

    def _set(self, phase: str, message: str, pod: PodInfo | None = None, error: str = "") -> None:
        now = self._clock()
        st = self.state
        if phase != st.phase:
            st.since = now
        st.phase = phase
        st.message = message
        st.checked_at = now
        if error:
            st.last_error = error
        if pod is not None:
            st.pod_id = pod.id
            st.proxy_url = rp.proxy_url(pod.id, self.cfg.port)
            st.gpu_type = pod.gpu_type
            st.cost_per_hr = pod.cost_per_hr
        elif phase in (OFFLINE, WAITING_FOR_GPU):
            st.pod_id = ""
            st.proxy_url = ""
            st.gpu_type = ""
            st.cost_per_hr = 0.0
        if phase == WAITING_FOR_GPU:
            if not st.waiting_since:
                st.waiting_since = now
        else:
            st.waiting_since = 0.0
        if phase != ONLINE:
            st.gpu_name = ""
        logger.info("pod %s: %s", phase, message)

    @property
    def online(self) -> bool:
        return self.state.phase == ONLINE

    @property
    def proxy_url(self) -> str:
        return self.state.proxy_url

    # ---------------------------------------------------------------- read-only refresh

    def refresh(self) -> PodState:
        """Look at RunPod + the proxy once and update `state` WITHOUT changing anything. Used by
        the background status poller so the UI badge is honest even when no job is running."""
        try:
            pod = self.client.find_pod(self.cfg.name)
        except RunPodError as exc:
            self._set(self.state.phase if self.state.phase != UNKNOWN else UNKNOWN,
                      f"Cannot reach RunPod: {exc}", error=str(exc))
            return self.state
        if pod is None:
            if self.state.phase != WAITING_FOR_GPU:
                self._set(OFFLINE, "No GPU pod exists yet -- one is created when you press Run.")
            return self.state
        if not pod.is_running:
            self._set(OFFLINE, "GPU pod is stopped -- it starts automatically when you press Run.", pod)
            return self.state
        if not pod.has_gpu:
            self._set(OFFLINE, "Pod is running WITHOUT a GPU (RunPod resumed it on CPU) -- it is "
                      "replaced automatically when you press Run.", pod)
            return self.state
        health = rp.probe_health(rp.proxy_url(pod.id, self.cfg.port), self._health_session)
        if health and not health.get("gpu_available", True):
            self._set(OFFLINE, "Pod is up but sees no GPU -- it is replaced automatically when you press Run.", pod)
        elif health:
            self._mark_online(pod, health)
        else:
            self._set(BOOTING, f"GPU pod is running, app still starting ({_fmt_minutes(pod.uptime_s)} up).", pod)
        return self.state

    def _mark_online(self, pod: PodInfo, health: dict) -> None:
        gpu = str(health.get("gpu_name") or pod.gpu_type or "GPU")
        self._set(ONLINE, f"GPU pod online ({gpu}).", pod)
        self.state.gpu_name = gpu

    # ---------------------------------------------------------------- the state machine

    def ensure_online(self, on_progress: ProgressFn = _no_progress, deadline_s: float | None = None) -> PodState:
        """Block until the pod's app answers `/api/health`, doing whatever RunPod needs along the
        way. Reports each phase change via `on_progress`. Raises `PodUnavailable` only for an
        auth error or when `deadline_s` (seconds from now, `None` = wait forever) passes."""
        with self._lock:
            return self._ensure_online_locked(on_progress, deadline_s)

    def _ensure_online_locked(self, on_progress: ProgressFn, deadline_s: float | None) -> PodState:
        started = self._clock()
        deadline = started + deadline_s if deadline_s is not None else None
        last_reported: tuple[str, str] | None = None
        boot_started: float | None = None
        no_gpu_replacements = 0

        def report() -> None:
            nonlocal last_reported
            key = (self.state.phase, self.state.message)
            if key != last_reported:
                last_reported = key
                on_progress(*key)

        def wait(seconds: float) -> None:
            if deadline is not None and self._clock() + seconds > deadline:
                raise PodUnavailable(self.state.message or "gave up waiting for the GPU pod")
            self._sleep(seconds)

        while True:
            try:
                pod = self.client.find_pod(self.cfg.name)
            except RunPodError as exc:
                if exc.is_auth_error:
                    self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                    report()
                    raise PodUnavailable(self.state.message) from exc
                self._set(self.state.phase, f"Cannot reach RunPod ({exc}); retrying.", error=str(exc))
                report()
                wait(self.cfg.poll_s)
                continue

            # --- no pod: create one -------------------------------------------------------
            if pod is None:
                try:
                    pod = self._create_pod()
                except RunPodError as exc:
                    if exc.is_auth_error:
                        self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                        report()
                        raise PodUnavailable(self.state.message) from exc
                    # Anything else from POST /pods is, in practice, "no stock for these GPU
                    # types right now" -- RunPod phrases it several ways. Keep the raw reason.
                    waited = self._clock() - (self.state.waiting_since or self._clock())
                    self._set(
                        WAITING_FOR_GPU,
                        "No GPU available on RunPod right now -- waiting; processing starts "
                        f"automatically when one frees up (checked every {int(self.cfg.retry_s)} s, "
                        f"waiting {_fmt_minutes(waited)} so far).",
                        error=str(exc),
                    )
                    report()
                    wait(self.cfg.retry_s)
                    continue
                boot_started = self._clock()
                self._set(STARTING, f"GPU pod created on {pod.gpu_type or 'a GPU'}; starting up.", pod)
                report()
                wait(self.cfg.poll_s)
                continue

            # --- pod exists but is stopped: Start, or tear down if its GPU is gone ----------
            if not pod.is_running:
                try:
                    pod = self.client.start_pod(pod.id)
                except RunPodError as exc:
                    if exc.is_auth_error:
                        self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                        report()
                        raise PodUnavailable(self.state.message) from exc
                    if exc.status == 0:
                        # Network blip -- not evidence the GPU is gone. Try again shortly.
                        self._set(OFFLINE, f"Cannot reach RunPod ({exc}); retrying.", pod, error=str(exc))
                        report()
                        wait(self.cfg.poll_s)
                        continue
                    self._set(
                        STARTING,
                        "The stopped pod's GPU was taken by someone else -- replacing the pod "
                        "(your uploads and results are on the persistent volume).",
                        pod,
                        error=str(exc),
                    )
                    report()
                    self._terminate_and_wait(pod.id)
                    continue  # next loop: no pod -> create
                if not pod.is_running or not pod.has_gpu:
                    # 2xx, but either the pod is still not RUNNING or RunPod resumed it on CPU
                    # only (its fallback when the card is gone) -- both are a refusal to us.
                    self._set(
                        STARTING,
                        "RunPod could not give the stopped pod its GPU back -- replacing the pod "
                        "(your uploads and results are on the persistent volume).",
                        pod,
                        error="resumed without a GPU" if pod.is_running else "start did not take",
                    )
                    report()
                    self._terminate_and_wait(pod.id)
                    continue
                boot_started = self._clock()
                self._set(STARTING, f"GPU pod starting on {pod.gpu_type or 'its GPU'}.", pod)
                report()
                wait(self.cfg.poll_s)
                continue

            # --- pod running: does it have a GPU, and is the app up? -------------------------
            if not pod.has_gpu:
                no_gpu_replacements += 1
                if no_gpu_replacements > MAX_NO_GPU_REPLACEMENTS:
                    self._set(ERROR, "Every replacement pod comes up without a GPU -- check the "
                              "RunPod console.", pod, error="repeated CPU-only pods")
                    report()
                    raise PodUnavailable(self.state.message)
                self._set(STARTING, "The running pod has no GPU (RunPod resumed it on CPU) -- replacing it.",
                          pod, error="running without a GPU")
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue
            health = rp.probe_health(rp.proxy_url(pod.id, self.cfg.port), self._health_session)
            if health and not health.get("gpu_available", True):
                # The app itself is the last word: no CUDA device means no GPU, whatever RunPod says.
                no_gpu_replacements += 1
                if no_gpu_replacements > MAX_NO_GPU_REPLACEMENTS:
                    self._set(ERROR, "Every replacement pod comes up without a GPU -- check the "
                              "RunPod console.", pod, error="repeated CPU-only pods")
                    report()
                    raise PodUnavailable(self.state.message)
                self._set(STARTING, "The pod booted without a usable GPU -- replacing it.", pod,
                          error="app reports no CUDA device")
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue
            if health:
                self._mark_online(pod, health)
                report()
                return self.state
            if boot_started is None:
                # It was already running when we arrived; count from RunPod's own uptime.
                boot_started = self._clock() - pod.uptime_s
            booting_for = self._clock() - boot_started
            if booting_for > self.cfg.boot_cap_s:
                self._set(
                    STARTING,
                    f"GPU pod has been booting for {_fmt_minutes(booting_for)} without coming up -- "
                    "replacing it.",
                    pod,
                    error="boot cap exceeded",
                )
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue
            self._set(BOOTING, f"GPU pod is running; app starting ({_fmt_minutes(booting_for)} so far).", pod)
            report()
            wait(self.cfg.poll_s)

    # ---------------------------------------------------------------- RunPod actions

    def _volume(self) -> str:
        if self._volume_id:
            return self._volume_id
        vol = self.client.find_volume(self.cfg.volume_name)
        if vol is None:
            logger.info("creating network volume %r (%d GB, %s)", self.cfg.volume_name, self.cfg.volume_gb, self.cfg.datacenter)
            vol = self.client.create_volume(self.cfg.volume_name, self.cfg.volume_gb, self.cfg.datacenter)
        self._volume_id = str(vol["id"])
        return self._volume_id

    def _create_pod(self) -> PodInfo:
        body = rp.pod_create_body(
            name=self.cfg.name,
            gpu_type_ids=list(self.cfg.gpu_types),
            volume_id=self._volume(),
            env=self.cfg.pod_env,
            port=self.cfg.port,
            cloud=self.cfg.cloud,
        )
        return self.client.create_pod(body)

    def _terminate_and_wait(self, pod_id: str) -> None:
        """DELETE the pod and wait until RunPod no longer lists it (the volume is untouched)."""
        waited = 0.0
        while True:
            try:
                self.client.terminate_pod(pod_id)
            except RunPodError as exc:
                if exc.status == 404:
                    return
                logger.warning("terminate %s: %s", pod_id, exc)
            self._sleep(self.cfg.poll_s)
            waited += self.cfg.poll_s
            try:
                if self.client.get_pod(pod_id) is None:
                    return
            except RunPodError as exc:
                logger.warning("get_pod after terminate: %s", exc)
            if waited >= self.cfg.terminate_wait_s:
                logger.warning("pod %s still listed %.0fs after terminate; carrying on", pod_id, waited)
                return
