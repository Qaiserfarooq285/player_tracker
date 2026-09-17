"""Idle auto-stop watchdog for a hosted RunPod pod (docs/DEPLOY.md "Idle auto-stop").

With "less credit" (owner, 2026-09-16) the single most expensive mistake is forgetting to press
Stop on the RunPod console. This module watches for real activity (any authenticated,
non-public API request -- see `apps.api.main.access_gate`) and, once nothing has happened for
`threshold_s` AND no pipeline job is in flight, calls RunPod's REST API to stop the pod itself.
Nothing is lost: the network volume (checkout, `input/`, `work/`, `output/`) persists across a
stop; the owner starts the pod again from the RunPod console.

Disabled by default -- both `RUNPOD_API_KEY` and `RUNPOD_POD_ID` must be set (`RUNPOD_POD_ID` is
injected automatically by RunPod into every pod's environment; `RUNPOD_API_KEY` is the one thing
the owner has to create and paste in, docs/DEPLOY.md). A failed stop call is logged and retried on
the next tick -- it never raises and never takes the app down (CLAUDE.md §10: log, don't crash).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import requests

from src.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_IDLE_MINUTES = 30.0
DEFAULT_CHECK_INTERVAL_S = 60.0
_STOP_REQUEST_TIMEOUT_S = 15.0


def should_stop(now: float, last_activity: float, jobs_in_flight: int, threshold_s: float) -> bool:
    """Pure decision: stop only when idle for `threshold_s` AND nothing is running.

    A pipeline job legitimately runs for minutes with no new HTTP request in between (the client
    just polls `/api/status/{job_id}`, which itself counts as activity, but there is no reason to
    depend on that) -- `jobs_in_flight` is the hard veto so a long-running job is never stopped out
    from under itself.
    """
    if jobs_in_flight > 0:
        return False
    return (now - last_activity) >= threshold_s


class IdleWatchdog:
    """Polls `should_stop` on a background thread and stops the pod via RunPod's REST API.

    `clock` is injectable (defaults to `time.monotonic`) so tests can drive time without sleeping.
    """

    def __init__(
        self,
        get_inflight_count: Callable[[], int],
        api_key: str,
        pod_id: str,
        threshold_s: float,
        api_base: str = DEFAULT_API_BASE,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.get_inflight_count = get_inflight_count
        self.api_key = api_key
        self.pod_id = pod_id
        self.threshold_s = threshold_s
        self.api_base = api_base.rstrip("/")
        self.check_interval_s = check_interval_s
        self._clock = clock
        self._last_activity = clock()
        self.stopped = False

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and bool(self.pod_id) and self.threshold_s > 0

    def touch(self) -> None:
        """Record real activity now. Called from `access_gate` for every non-public request."""
        self._last_activity = self._clock()

    def stop_pod(self) -> bool:
        """POST the RunPod REST "stop pod" call. Never raises -- logs and returns False on any
        failure so the caller can just retry on the next tick."""
        url = f"{self.api_base}/pods/{self.pod_id}/stop"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            # `json={}` is load-bearing: the endpoint answers HTTP 500 "unexpected end of JSON
            # input" to a body-less POST (observed 2026-09-17), which would make every tick fail
            # and the pod never stop -- the exact bill this watchdog exists to prevent.
            resp = requests.post(url, headers=headers, json={}, timeout=_STOP_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            logger.warning(
                "idle-stop: POST %s raised %s: %s -- will retry next tick",
                url,
                type(exc).__name__,
                exc,
            )
            return False
        if 200 <= resp.status_code < 300:
            logger.info("idle-stop: pod %s stopped (HTTP %d)", self.pod_id, resp.status_code)
            return True
        logger.warning(
            "idle-stop: stop request failed HTTP %d: %s -- will retry next tick",
            resp.status_code,
            resp.text[:200],
        )
        return False

    def tick(self, now: float | None = None) -> None:
        """One check. Fires `stop_pod` at most once (`self.stopped` latches on success) -- a
        stopped pod doesn't need repeated stop calls, and a failed one is retried by the next
        natural `tick`, not by looping here."""
        if not self.enabled or self.stopped:
            return
        if now is None:
            now = self._clock()
        if should_stop(now, self._last_activity, self.get_inflight_count(), self.threshold_s):
            logger.info(
                "idle-stop: no activity for >= %.1f min and no job in flight -- stopping pod %s",
                self.threshold_s / 60.0,
                self.pod_id,
            )
            if self.stop_pod():
                self.stopped = True

    def _run_loop(self) -> None:
        while True:
            time.sleep(self.check_interval_s)
            self.tick()

    def start(self) -> None:
        """Start the background daemon thread. No-op (one log line) when not `enabled`."""
        if not self.enabled:
            logger.info(
                "idle-stop watchdog disabled (need both RUNPOD_API_KEY and RUNPOD_POD_ID set, "
                "and PV_IDLE_STOP_MINUTES > 0)"
            )
            return
        thread = threading.Thread(target=self._run_loop, name="idle-stop-watchdog", daemon=True)
        thread.start()
        logger.info(
            "idle-stop watchdog started: stop after %.1f min idle (checked every %.0fs)",
            self.threshold_s / 60.0,
            self.check_interval_s,
        )

    def status(self) -> dict:
        seconds_since_activity = self._clock() - self._last_activity
        return {
            "enabled": self.enabled,
            "idle_minutes": round(self.threshold_s / 60.0, 2),
            "seconds_since_activity": round(seconds_since_activity, 1),
        }


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def watchdog_from_env(get_inflight_count: Callable[[], int]) -> IdleWatchdog:
    """Build an `IdleWatchdog` from the environment (docker/runpod.env.example).

    `RUNPOD_POD_ID` is injected by RunPod automatically; `RUNPOD_API_KEY` is the owner-created
    key (RunPod -> Settings -> API Keys -> Restricted, pods read/write). Both blank by default
    off-RunPod, so the watchdog is disabled unless actually running on a pod with the key set.
    """
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    idle_minutes = _float_env("PV_IDLE_STOP_MINUTES", DEFAULT_IDLE_MINUTES)
    api_base = os.environ.get("PV_RUNPOD_API_BASE", "").strip() or DEFAULT_API_BASE
    check_interval_s = _float_env("PV_IDLE_CHECK_SECONDS", DEFAULT_CHECK_INTERVAL_S)
    return IdleWatchdog(
        get_inflight_count=get_inflight_count,
        api_key=api_key,
        pod_id=pod_id,
        threshold_s=idle_minutes * 60.0,
        api_base=api_base,
        check_interval_s=check_interval_s,
    )
