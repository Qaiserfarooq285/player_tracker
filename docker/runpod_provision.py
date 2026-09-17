#!/usr/bin/env python3
"""Provision (or re-point) the PitchVision RunPod pod from the command line (docs/DEPLOY.md).

Does through the REST API what docs/DEPLOY.md §1-3 describe clicking through the console: create
the network volume once, then a pod that mounts it at /workspace and runs docker/runpod_bootstrap.sh
on every boot. Idempotent on the volume (looked up by name) and refuses to create a second pod
with the same name, so re-running it after a failed attempt is safe.

    RUNPOD_API_KEY=... python docker/runpod_provision.py \
        --password '<site password>' --gpu 'NVIDIA RTX A4000' --datacenter EUR-IS-1

Prints the pod id and its proxy URL; feed the id to docker/vps/setup_vps.sh as POD_ID. Never
commits or logs the password beyond the pod's own env. Stdlib only (urllib), no SDK needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh"
)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_VOLUME_GB = 40  # models 2 GB + venv ~7 GB + room for clips and outputs; grows, never shrinks


def _call(method: str, url: str, key: str, body: dict | None = None) -> dict | list:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "pitchvision-provision/1.0")  # Cloudflare 1010s the urllib default
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:800]}") from e
    return json.loads(raw) if raw else {}


def rest(method: str, path: str, key: str, body: dict | None = None):
    return _call(method, f"{REST}{path}", key, body)


def graphql(query: str, key: str) -> dict:
    out = _call("POST", GRAPHQL, key, {"query": query})
    if out.get("errors"):
        raise SystemExit(f"graphql error: {out['errors']}")
    return out["data"]


def ensure_volume(key: str, name: str, size_gb: int, datacenter: str) -> dict:
    for v in rest("GET", "/networkvolumes", key) or []:
        if v.get("name") == name:
            print(f"[provision] reusing network volume {v['id']} ({v.get('size')} GB, {v.get('dataCenterId')})")
            return v
    print(f"[provision] creating network volume {name!r}: {size_gb} GB in {datacenter}")
    return rest("POST", "/networkvolumes", key, {"name": name, "size": size_gb, "dataCenterId": datacenter})


def find_pod(key: str, name: str) -> dict | None:
    for p in rest("GET", "/pods", key) or []:
        if p.get("name") == name:
            return p
    return None


def create_pod(key: str, args: argparse.Namespace, volume_id: str) -> dict:
    env = {
        "PV_ACCESS_PASSWORD": args.password,
        "PV_IDLE_STOP_MINUTES": str(args.idle_minutes),
        "PORT": str(args.port),
    }
    if args.idle_stop:
        # The pod uses the same key to stop itself when idle (apps/api/idle_stop.py).
        env["RUNPOD_API_KEY"] = key
    if args.gemini_key:
        env["GEMINI_API_KEY"] = args.gemini_key
    if args.github_token:
        env["GITHUB_TOKEN"] = args.github_token
    if args.ssh_public_key:
        # RunPod's own /start.sh (kept alive by the bootstrap) installs this for `ssh root@<ip> -p <port>`.
        env["PUBLIC_KEY"] = args.ssh_public_key
    # The repo is private: raw.githubusercontent.com needs the token too. `$GITHUB_TOKEN` is left
    # for the container's shell to expand at boot, so the token never appears in the command itself.
    fetch = f'curl -fsSL -H "Authorization: token $GITHUB_TOKEN" {BOOTSTRAP_URL}'
    if args.debug:
        # Keep the container alive after a failed bootstrap and leave its output on the volume, so a
        # crash-looping pod can be inspected over SSH instead of guessed at from a blank log viewer.
        start_cmd = (
            "/start.sh >/workspace/runpod-start.log 2>&1 & export PV_SKIP_RUNPOD_SERVICES=1; "
            f"{fetch} -o /workspace/bootstrap.sh && "
            "bash /workspace/bootstrap.sh >/workspace/bootstrap.log 2>&1; "
            "echo EXIT=$? >>/workspace/bootstrap.log; sleep infinity"
        )
    else:
        start_cmd = f"{fetch} | bash"
    body = {
        "name": args.name,
        "imageName": IMAGE,
        "cloudType": args.cloud,
        "gpuTypeIds": [g.strip() for g in args.gpu.split(",") if g.strip()],
        "gpuCount": 1,
        "containerDiskInGb": 20,
        "volumeMountPath": "/workspace",
        "networkVolumeId": volume_id,
        "ports": [f"{args.port}/http", "22/tcp"],
        "env": env,
        "dockerStartCmd": ["bash", "-c", start_cmd],
        "supportPublicIp": True,
    }
    print(f"[provision] creating pod {args.name!r}: {args.gpu} ({args.cloud}) with volume {volume_id}")
    return rest("POST", "/pods", key, body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--password", required=True, help="PV_ACCESS_PASSWORD for the web UI")
    ap.add_argument("--gpu", default="NVIDIA RTX A4000", help="gpuTypeId(s), comma-separated in order of preference")
    ap.add_argument("--datacenter", default="EUR-IS-1", help="region for the volume + pod")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--name", default="pitchvision")
    ap.add_argument("--volume-name", default="pitchvision-data")
    ap.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--idle-minutes", type=int, default=30)
    ap.add_argument("--no-idle-stop", dest="idle_stop", action="store_false")
    ap.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""))
    ap.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""),
                    help="fine-grained PAT, Contents: read-only, this repo only (the repo is private)")
    ap.add_argument("--ssh-public-key", default="", help="installed for root SSH on the exposed 22/tcp")
    ap.add_argument("--debug", action="store_true",
                    help="log the bootstrap to /workspace/bootstrap.log and keep the container alive on failure")
    ap.add_argument("--wait", action="store_true", help="block until the pod reports RUNNING")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.github_token:
        print("--github-token / GITHUB_TOKEN is not set: the pod cannot clone the private repo", file=sys.stderr)
        return 2

    volume = ensure_volume(key, args.volume_name, args.volume_gb, args.datacenter)
    if volume.get("dataCenterId") and volume["dataCenterId"] != args.datacenter:
        print(f"[provision] note: volume lives in {volume['dataCenterId']}; the pod must go there too")
        args.datacenter = volume["dataCenterId"]

    pod = find_pod(key, args.name)
    if pod:
        print(f"[provision] pod {args.name!r} already exists: {pod['id']} ({pod.get('desiredStatus')})")
    else:
        pod = create_pod(key, args, volume["id"])
        print(f"[provision] created pod {pod['id']}")

    pod_id = pod["id"]
    if args.wait:
        for _ in range(60):
            pod = rest("GET", f"/pods/{pod_id}", key)
            status = pod.get("desiredStatus")
            runtime = pod.get("runtime") or {}
            print(f"[provision] status={status} uptime={runtime.get('uptimeInSeconds')}s", flush=True)
            if status == "RUNNING" and runtime.get("uptimeInSeconds"):
                break
            time.sleep(10)

    print()
    print(f"POD_ID={pod_id}")
    print(f"PROXY_URL=https://{pod_id}-{args.port}.proxy.runpod.net")
    print(f"HEALTH=https://{pod_id}-{args.port}.proxy.runpod.net/api/health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
