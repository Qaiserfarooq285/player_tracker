"""The always-on gateway's HTTP surface (`apps/gateway/main.py`) and job worker
(`apps/gateway/jobs.py`) with the pod faked -- nothing here reaches RunPod or a real pod. The
point of the gateway is that the frontend works with NO pod: uploads land, Run queues, the status
says why it is waiting."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

import apps.gateway.jobs as jobs_mod
import apps.gateway.pod_manager as pm
from apps.gateway.jobs import JobStore, JobWorker


@pytest.fixture
def gw(tmp_path, monkeypatch):
    """A fresh gateway app bound to `tmp_path`, threads off, login gate off."""
    monkeypatch.setenv("PV_GATEWAY_DATA", str(tmp_path))
    monkeypatch.setenv("PV_GATEWAY_NO_THREADS", "1")
    monkeypatch.setenv("PV_ACCESS_PASSWORD", "off")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    import importlib

    import apps.gateway.main as main

    main = importlib.reload(main)
    main.PODS.state = pm.PodState(phase=pm.OFFLINE, message="GPU pod is stopped")
    main.request_wake = lambda: None  # never spawn the real wake thread in tests
    with TestClient(main.app) as client:
        yield client, main


def _upload(client, name: str, data: bytes, chunk: int = 4):
    total = max(1, -(-len(data) // chunk))
    out = None
    for i in range(total):
        out = client.post(
            "/api/upload/chunk",
            data={"upload_id": "abcdefgh1234", "index": i, "total": total, "filename": name},
            files={"chunk": (name, io.BytesIO(data[i * chunk : (i + 1) * chunk]))},
        )
        assert out.status_code == 200, out.text
    return out.json()


def test_frontend_and_uploads_work_with_pod_offline(gw, tmp_path):
    client, main = gw
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["gateway"] is True and health["gpu_available"] is False
    assert health["pod"]["phase"] == "offline"

    out = _upload(client, "my clip.mp4", b"0123456789")
    assert out["status"] == "success" and out["filename"] == "my_clip.mp4"
    assert (tmp_path / "input" / "my_clip.mp4").read_bytes() == b"0123456789"
    vids = client.get("/api/videos").json()
    assert [v["name"] for v in vids["input_videos"]] == ["my_clip.mp4"]
    # The preview streams from the VPS copy -- no pod needed.
    assert client.get("/media/my_clip.mp4").content == b"0123456789"


def test_run_queues_and_status_explains_waiting(gw):
    client, main = gw
    _upload(client, "a.mp4", b"xx")
    res = client.post("/api/process", json={"video_name": "a.mp4", "target_jersey": 7})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = client.get(f"/api/status/{job_id}").json()
    assert st["status"] == "queued" and st["logs"]
    assert "payload" not in st
    # Same video twice while queued -> refused, like the pod's own in-flight guard.
    assert client.post("/api/process", json={"video_name": "a.mp4"}).status_code == 409
    # A file nobody uploaded, with no pod to ask -> immediate 404, not a job that fails later.
    assert client.post("/api/process", json={"video_name": "ghost.mp4"}).status_code == 404
    assert client.get("/api/jobs").json()["active"][0]["job_id"] == job_id


def test_gpu_only_endpoints_say_why_when_pod_is_down(gw):
    client, main = gw
    main.PODS.state = pm.PodState(phase=pm.WAITING_FOR_GPU, message="No GPU available on RunPod right now")
    res = client.get("/api/frame_players?video=a.mp4&t=1")
    assert res.status_code == 503
    assert "No GPU available" in res.json()["detail"]
    assert client.get("/api/results/some_slug").status_code == 503
    assert client.get("/media/output/x/y.mp4").status_code == 503
    # /api/health stays public and cheap, and carries the phase for the badge.
    assert client.get("/api/health").json()["pod"]["phase"] == "waiting_for_gpu"


def test_login_gate_applies_to_the_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_GATEWAY_DATA", str(tmp_path))
    monkeypatch.setenv("PV_GATEWAY_NO_THREADS", "1")
    monkeypatch.setenv("PV_ACCESS_PASSWORD", "s3cret")
    import importlib

    import apps.gateway.main as main

    main = importlib.reload(main)
    with TestClient(main.app) as client:
        assert client.get("/api/videos").status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.post("/api/login", json={"password": "nope"}).status_code == 401
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        assert client.get("/api/videos").status_code == 200


# ---------------------------------------------------------------- the worker, with a fake pod


class FakePodClient:
    """Stands in for `apps/gateway/pod_client.PodClient`: remembers what was uploaded and
    walks a scripted list of pod-side status snapshots."""

    def __init__(self, statuses, has_video=False):
        self.statuses = list(statuses)
        self.has_video = has_video
        self.uploaded: list[str] = []
        self.process_payloads: list[dict] = []

    def has_input_video(self, name, size):
        return self.has_video

    def upload_video(self, path, upload_id, on_progress=None):
        self.uploaded.append(path.name)
        if on_progress:
            on_progress(1.0)
        return {"status": "success", "filename": path.name}

    def start_process(self, payload):
        self.process_payloads.append(payload)
        return {"job_id": "podjob1", "status": "queued"}

    def job_status(self, pod_job_id):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


class FakePods:
    def __init__(self, phases_before_online=()):
        self.state = pm.PodState(phase=pm.OFFLINE)
        self.phases = list(phases_before_online)

    @property
    def online(self):
        return self.state.phase == pm.ONLINE

    def refresh(self):
        return self.state

    def ensure_online(self, on_progress=lambda *a: None, deadline_s=None):
        for phase, msg in self.phases:
            self.state.phase = phase
            on_progress(phase, msg)
        self.state = pm.PodState(phase=pm.ONLINE, proxy_url="https://pod", gpu_name="RTX 4090")
        on_progress(pm.ONLINE, "GPU pod online")
        return self.state


def _worker(tmp_path, pod, pods):
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"video")
    worker = JobWorker(store, pods, pod, tmp_path / "input", sleep=lambda s: None, clock=lambda: 0.0)
    return store, worker


def test_worker_waits_for_gpu_then_uploads_then_mirrors_pod_progress(tmp_path):
    pod = FakePodClient([
        {"status": "running", "stage": "Detection", "progress": 20, "logs": ["detecting"]},
        {"status": "running", "stage": "Tracking", "progress": 60, "logs": ["detecting", "tracking"]},
        {"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": ["detecting", "tracking", "done"]},
    ])
    pods = FakePods([(pm.WAITING_FOR_GPU, "No GPU available on RunPod right now -- waiting"),
                     (pm.STARTING, "GPU pod created"), (pm.BOOTING, "app starting")])
    store, worker = _worker(tmp_path, pod, pods)
    job = store.create("a.mp4", {"video_name": "a.mp4", "target_jersey": 9})

    worker.run_job(job)

    final = store.get(job["job_id"])
    assert final["status"] == "completed" and final["slug"] == "a" and final["progress"] == 100
    assert pod.uploaded == ["a.mp4"]
    assert pod.process_payloads == [{"video_name": "a.mp4", "target_jersey": 9}]
    logs = final["logs"]
    assert any("No GPU available" in line for line in logs)
    assert logs[-3:] == ["[pod] detecting", "[pod] tracking", "[pod] done"]
    # Persisted: a fresh store from the same file sees the finished job.
    assert JobStore(tmp_path / "jobs.json").get(job["job_id"])["status"] == "completed"


def test_worker_skips_upload_when_pod_volume_already_has_the_file(tmp_path):
    pod = FakePodClient([{"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": []}], has_video=True)
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    worker.run_job(job)
    assert pod.uploaded == []
    assert store.get(job["job_id"])["status"] == "completed"


def test_worker_marks_failed_when_pod_pipeline_fails(tmp_path):
    pod = FakePodClient([{"status": "failed", "stage": "Detection", "progress": 10, "error": "boom", "logs": ["x"]}])
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    worker.run_job(job)
    final = store.get(job["job_id"])
    assert final["status"] == "failed" and final["error"] == "boom"


def test_worker_reattaches_to_a_running_pod_job_after_restart(tmp_path):
    pod = FakePodClient([{"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": []}])
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    store.update(job["job_id"], status=jobs_mod.RUNNING, pod_job_id="podjob1")
    worker.run_job(store.get(job["job_id"]))
    assert pod.uploaded == [] and pod.process_payloads == []
    assert store.get(job["job_id"])["status"] == "completed"


def test_worker_fails_job_when_pod_vanishes_mid_run(tmp_path):
    class Vanishing(FakePodClient):
        def job_status(self, pod_job_id):
            raise requests.ConnectionError("gone")

    now = {"t": 0.0}

    def clock():
        now["t"] += 120.0
        return now["t"]

    pod = Vanishing([])
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"v")
    worker = JobWorker(store, FakePods(), pod, tmp_path / "input", sleep=lambda s: None, clock=clock)
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    with pytest.raises(RuntimeError, match="stopped answering"):
        worker.run_job(job)
