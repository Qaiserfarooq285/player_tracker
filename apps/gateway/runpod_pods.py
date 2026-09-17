"""Thin RunPod REST client for the ONE pod this project runs (docs/DEPLOY.md "Always-on gateway").

Everything the gateway (`apps/gateway/pod_manager.py`) and the command-line provisioner
(`docker/runpod_provision.py`) need to say to RunPod lives here, once: find the pod by name,
start / stop / terminate it, create a fresh one on the network volume, and probe its HTTP proxy.
The pod *spec* (`pod_create_body`) is the single source of truth for what a PitchVision pod looks
like -- CLAUDE.md §10, no duplicated magic values between the two callers.

Stdlib + `requests` only (a core dependency already), so the VPS gateway's venv stays tiny.
Nothing here retries or sleeps: callers own the waiting policy.

RunPod REST quirks learned the hard way (2026-09-17):
  * `POST /pods/{id}/start|stop` answer HTTP 500 to a body-less POST -- always send `{}`.
  * Cloudflare (error 1010) rejects urllib's / requests' default User-Agent -- set our own.
  * The `<id>-<port>.proxy.runpod.net` host answers a stopped pod, and a booting one whose port
    isn't registered yet, with an EMPTY 404 -- not a 502.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

REST_BASE = "https://rest.runpod.io/v1"
USER_AGENT = "pitchvision-gateway/1.0"
REQUEST_TIMEOUT_S = 60.0
HEALTH_TIMEOUT_S = 15.0

# What a PitchVision pod is made of. `docker/runpod_bootstrap.sh` is fetched on every boot from
# the (private) repo with the pod's GITHUB_TOKEN and runs the app; the network volume at
# /workspace carries the checkout, venv, models, uploads and outputs across pod recreations.
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh"
)
CONTAINER_DISK_GB = 20
VOLUME_MOUNT_PATH = "/workspace"
DEFAULT_POD_NAME = "pitchvision"
DEFAULT_VOLUME_NAME = "pitchvision-data"
DEFAULT_VOLUME_GB = 40  # models 2 GB + venv ~7 GB + room for clips and outputs; grows, never shrinks
DEFAULT_PORT = 8000
DEFAULT_DATACENTER = "EUR-IS-1"
DEFAULT_CLOUD = "SECURE"  # network volumes only attach to Secure Cloud pods
# In order of preference; RunPod picks the first type with stock. (What actually had stock in
# EUR-IS-1 on 2026-09-17: only the 4090. The RTX 2000 Ada is excluded on purpose -- every boot
# on that host died on the model download.)
DEFAULT_GPU_TYPES = (
    "NVIDIA L4",
    "NVIDIA RTX A4500",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
)


class RunPodError(Exception):
    """A RunPod REST call that did not succeed. `status` is the HTTP status (0 = no response)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status

    @property
    def is_auth_error(self) -> bool:
        return self.status in (401, 403)


def proxy_url(pod_id: str, port: int = DEFAULT_PORT) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def pod_create_body(
    *,
    name: str,
    gpu_type_ids: list[str],
    volume_id: str,
    env: dict[str, str],
    port: int = DEFAULT_PORT,
    cloud: str = DEFAULT_CLOUD,
    debug: bool = False,
) -> dict[str, Any]:
    """The `POST /pods` payload for a PitchVision pod. `env` must already carry the secrets the
    bootstrap needs (`GITHUB_TOKEN`, `PV_ACCESS_PASSWORD`, optional `RUNPOD_API_KEY` for idle
    auto-stop, `PUBLIC_KEY` for SSH).

    The repo is private, so raw.githubusercontent.com needs the token too. `$GITHUB_TOKEN` is
    left for the container's shell to expand at boot, so the token never appears in the command.
    """
    fetch = f'curl -fsSL -H "Authorization: token $GITHUB_TOKEN" {BOOTSTRAP_URL}'
    if debug:
        # Keep the container alive after a failed bootstrap and leave its output on the volume,
        # so a crash-looping pod can be inspected over SSH instead of guessed at from a blank
        # log viewer.
        start_cmd = (
            "/start.sh >/workspace/runpod-start.log 2>&1 & export PV_SKIP_RUNPOD_SERVICES=1; "
            f"{fetch} -o /workspace/bootstrap.sh && "
            "bash /workspace/bootstrap.sh >/workspace/bootstrap.log 2>&1; "
            "echo EXIT=$? >>/workspace/bootstrap.log; sleep infinity"
        )
    else:
        start_cmd = f"{fetch} | bash"
    return {
        "name": name,
        "imageName": IMAGE,
        "cloudType": cloud,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuCount": 1,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "networkVolumeId": volume_id,
        "ports": [f"{port}/http", "22/tcp"],
        "env": dict(env),
        "dockerStartCmd": ["bash", "-c", start_cmd],
        "supportPublicIp": True,
    }


@dataclass(frozen=True)
class PodInfo:
    """The few fields of a RunPod pod record the gateway reasons about."""

    id: str
    name: str
    desired_status: str  # "RUNNING" | "EXITED" | ... as RunPod reports it
    gpu_type: str
    uptime_s: int
    cost_per_hr: float

    @property
    def is_running(self) -> bool:
        return self.desired_status == "RUNNING"

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> PodInfo:
        machine = raw.get("machine") or {}
        runtime = raw.get("runtime") or {}
        return cls(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            desired_status=str(raw.get("desiredStatus", "")),
            gpu_type=str(machine.get("gpuTypeId") or raw.get("gpuTypeId") or ""),
            uptime_s=int(runtime.get("uptimeInSeconds") or 0),
            cost_per_hr=float(raw.get("costPerHr") or 0.0),
        )


class RunPodClient:
    """One authenticated REST session. `session` is injectable so tests never touch the network."""

    def __init__(
        self,
        api_key: str,
        base_url: str = REST_BASE,
        session: requests.Session | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_s = timeout_s

    # ---------------------------------------------------------------- raw call

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        try:
            resp = self.session.request(method, url, json=body, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise RunPodError(f"{method} {path}: {type(exc).__name__}: {exc}") from exc
        if not 200 <= resp.status_code < 300:
            raise RunPodError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:400]}", resp.status_code)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # ---------------------------------------------------------------- pods

    def list_pods(self) -> list[PodInfo]:
        raw = self._call("GET", "/pods") or []
        return [PodInfo.from_api(p) for p in raw]

    def find_pod(self, name: str) -> PodInfo | None:
        for pod in self.list_pods():
            if pod.name == name:
                return pod
        return None

    def get_pod(self, pod_id: str) -> PodInfo | None:
        try:
            return PodInfo.from_api(self._call("GET", f"/pods/{pod_id}"))
        except RunPodError as exc:
            if exc.status == 404:
                return None
            raise

    def start_pod(self, pod_id: str) -> PodInfo:
        # `{}` is load-bearing: a body-less POST is answered with HTTP 500.
        return PodInfo.from_api(self._call("POST", f"/pods/{pod_id}/start", {}))

    def stop_pod(self, pod_id: str) -> None:
        self._call("POST", f"/pods/{pod_id}/stop", {})

    def terminate_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{pod_id}")

    def create_pod(self, body: dict[str, Any]) -> PodInfo:
        return PodInfo.from_api(self._call("POST", "/pods", body))

    # ---------------------------------------------------------------- network volume

    def find_volume(self, name: str) -> dict[str, Any] | None:
        for vol in self._call("GET", "/networkvolumes") or []:
            if vol.get("name") == name:
                return vol
        return None

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> dict[str, Any]:
        return self._call("POST", "/networkvolumes", {"name": name, "size": size_gb, "dataCenterId": datacenter})


def probe_health(url: str, session: requests.Session | None = None, timeout_s: float = HEALTH_TIMEOUT_S) -> dict | None:
    """GET `<proxy>/api/health`. The parsed JSON when the app answers 200, else `None` (stopped,
    booting, or unreachable -- the caller doesn't need to tell those apart)."""
    sess = session or requests.Session()
    try:
        resp = sess.get(f"{url.rstrip('/')}/api/health", timeout=timeout_s, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
