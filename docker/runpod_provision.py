#!/usr/bin/env python3
"""Provision (or re-point) the PitchVision RunPod pod from the command line (docs/DEPLOY.md).

Does through the REST API what docs/DEPLOY.md §1-3 describe clicking through the console: create
the network volume once, then a pod that mounts it at /workspace and runs docker/runpod_bootstrap.sh
on every boot. Idempotent on the volume (looked up by name) and refuses to create a second pod
with the same name, so re-running it after a failed attempt is safe.

    RUNPOD_API_KEY=... python docker/runpod_provision.py \
        --password '<site password>' --gpu 'NVIDIA RTX A4000' --datacenter EUR-IS-1

Prints the pod id and its proxy URL. With the always-on gateway in front (docs/DEPLOY.md), the
VPS needs no pod id at all -- it finds the pod by name -- so this script is for a first manual
deploy or for poking at RunPod from the command line. Never commits or logs the password beyond
the pod's own env. `requests` only, no SDK needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The REST client and the pod spec are shared with the always-on gateway (apps/gateway/
# pod_manager.py does at runtime what this script does by hand), so there is exactly one
# definition of "what a PitchVision pod looks like".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apps.gateway import runpod_pods as rp  # noqa: E402
from apps.gateway.runpod_pods import RunPodClient, RunPodError  # noqa: E402


def ensure_volume(client: RunPodClient, name: str, size_gb: int, datacenter: str) -> dict:
    vol = client.find_volume(name)
    if vol:
        print(f"[provision] reusing network volume {vol['id']} ({vol.get('size')} GB, {vol.get('dataCenterId')})")
        return vol
    print(f"[provision] creating network volume {name!r}: {size_gb} GB in {datacenter}")
    return client.create_volume(name, size_gb, datacenter)


def pod_env(args: argparse.Namespace, key: str) -> dict[str, str]:
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
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--password", required=True, help="PV_ACCESS_PASSWORD for the web UI")
    ap.add_argument("--gpu", default=",".join(rp.DEFAULT_GPU_TYPES),
                    help="gpuTypeId(s), comma-separated in order of preference")
    ap.add_argument("--datacenter", default=rp.DEFAULT_DATACENTER, help="region for the volume + pod")
    ap.add_argument("--cloud", default=rp.DEFAULT_CLOUD, choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--name", default=rp.DEFAULT_POD_NAME)
    ap.add_argument("--volume-name", default=rp.DEFAULT_VOLUME_NAME)
    ap.add_argument("--volume-gb", type=int, default=rp.DEFAULT_VOLUME_GB)
    ap.add_argument("--port", type=int, default=rp.DEFAULT_PORT)
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
    client = RunPodClient(key)

    try:
        volume = ensure_volume(client, args.volume_name, args.volume_gb, args.datacenter)
        if volume.get("dataCenterId") and volume["dataCenterId"] != args.datacenter:
            print(f"[provision] note: volume lives in {volume['dataCenterId']}; the pod must go there too")
            args.datacenter = volume["dataCenterId"]

        pod = client.find_pod(args.name)
        if pod:
            print(f"[provision] pod {args.name!r} already exists: {pod.id} ({pod.desired_status})")
        else:
            gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
            print(f"[provision] creating pod {args.name!r}: {gpus} ({args.cloud}) with volume {volume['id']}")
            body = rp.pod_create_body(
                name=args.name, gpu_type_ids=gpus, volume_id=str(volume["id"]), env=pod_env(args, key),
                port=args.port, cloud=args.cloud, debug=args.debug,
            )
            pod = client.create_pod(body)
            print(f"[provision] created pod {pod.id}")

        if args.wait:
            for _ in range(60):
                current = client.get_pod(pod.id)
                if current is None:
                    break
                print(f"[provision] status={current.desired_status} uptime={current.uptime_s}s", flush=True)
                if current.is_running and current.uptime_s:
                    break
                time.sleep(10)
    except RunPodError as exc:
        raise SystemExit(f"[provision] {exc}") from exc

    print()
    print(f"POD_ID={pod.id}")
    print(f"PROXY_URL={rp.proxy_url(pod.id, args.port)}")
    print(f"HEALTH={rp.proxy_url(pod.id, args.port)}/api/health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
