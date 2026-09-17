"""The always-on gateway's pod state machine (`apps/gateway/pod_manager.py`, docs/DEPLOY.md
"Always-on gateway") against a scripted fake of RunPod's REST API -- no network, no real time
(`clock`/`sleep` are injected). Each test is one of the situations the owner actually hit on
2026-09-17: a stopped pod whose GPU is gone, no stock at all, a normal start."""

from __future__ import annotations

import pytest

import apps.gateway.pod_manager as pm
from apps.gateway.pod_manager import PodConfig, PodManager, PodUnavailable
from apps.gateway.runpod_pods import PodInfo, RunPodError


class FakeRunPod:
    """Scripted RunPod. `pods` is the live table; `create_ok` / `start_ok` decide whether the
    next create/start is refused (the "no GPU" cases); `healthy_after_polls` says how many health
    probes a running pod needs before its app answers."""

    def __init__(self, pods=None, *, create_ok=True, start_ok=True, cpu_resume=False, healthy_after_polls=1):
        self.pods: dict[str, dict] = {p["id"]: p for p in (pods or [])}
        self.create_ok = create_ok
        self.cpu_resume = cpu_resume
        self.start_ok = start_ok
        self.healthy_after_polls = healthy_after_polls
        self.calls: list[str] = []
        self._next_id = 1
        self.health_polls: dict[str, int] = {}

    # --- RunPodClient surface ---
    def find_pod(self, name):
        self.calls.append("find")
        for p in self.pods.values():
            if p["name"] == name:
                return PodInfo.from_api(p)
        return None

    def get_pod(self, pod_id):
        self.calls.append(f"get:{pod_id}")
        p = self.pods.get(pod_id)
        return PodInfo.from_api(p) if p else None

    def start_pod(self, pod_id):
        self.calls.append(f"start:{pod_id}")
        if not self.start_ok:
            raise RunPodError("POST /pods/x/start -> HTTP 500: not enough free GPUs on the host machine", 500)
        self.pods[pod_id]["desiredStatus"] = "RUNNING"
        if self.cpu_resume:
            # What RunPod actually did on 2026-09-17: 2xx, RUNNING, but no GPU attached.
            self.pods[pod_id]["machine"] = {}
            self.pods[pod_id]["gpuCount"] = None
        return PodInfo.from_api(self.pods[pod_id])

    def terminate_pod(self, pod_id):
        self.calls.append(f"terminate:{pod_id}")
        self.pods.pop(pod_id, None)

    def create_pod(self, body):
        self.calls.append("create")
        if not self.create_ok:
            raise RunPodError("POST /pods -> HTTP 500: There are no longer any instances available", 500)
        pod_id = f"new{self._next_id}"
        self._next_id += 1
        self.pods[pod_id] = {"id": pod_id, "name": body["name"], "desiredStatus": "RUNNING", "gpuCount": 1,
                             "machine": {"gpuTypeId": body["gpuTypeIds"][0]}, "runtime": {"uptimeInSeconds": 0}}
        return PodInfo.from_api(self.pods[pod_id])

    def find_volume(self, name):
        return {"id": "vol1", "name": name}

    def create_volume(self, *a, **k):  # pragma: no cover - never reached with find_volume above
        raise AssertionError("volume already exists")

    # --- the proxy's /api/health, as seen by pod_manager.probe_health ---
    def probe(self, url, session=None, timeout_s=None):
        pod_id = url.split("//")[1].split("-")[0]
        p = self.pods.get(pod_id)
        if not p or p["desiredStatus"] != "RUNNING":
            return None
        self.health_polls[pod_id] = self.health_polls.get(pod_id, 0) + 1
        if self.health_polls[pod_id] >= self.healthy_after_polls:
            has_gpu = bool(p.get("gpuCount"))
            return {"status": "healthy", "gpu_available": has_gpu, "gpu_name": "NVIDIA GeForce RTX 4090" if has_gpu else "CPU Only"}
        return None


class FakeClock:
    def __init__(self):
        self.now = 1_000.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


STOPPED_POD = {"id": "old1", "name": "pitchvision", "desiredStatus": "EXITED", "gpuCount": 1,
               "machine": {"gpuTypeId": "NVIDIA GeForce RTX 4090"}}


@pytest.fixture
def make(monkeypatch):
    def _make(fake: FakeRunPod, **cfg):
        monkeypatch.setattr(pm.rp, "probe_health", fake.probe)
        clock = FakeClock()
        config = PodConfig(pod_env={"GITHUB_TOKEN": "t", "PV_ACCESS_PASSWORD": "p"}, poll_s=10, retry_s=60, **cfg)
        mgr = PodManager(fake, config, clock=clock, sleep=clock.sleep)  # type: ignore[arg-type]
        return mgr, clock

    return _make


def test_stopped_pod_starts_in_place(make):
    fake = FakeRunPod([dict(STOPPED_POD)])
    mgr, clock = make(fake)
    phases = []
    st = mgr.ensure_online(lambda ph, msg: phases.append(ph))
    assert st.phase == pm.ONLINE
    assert st.pod_id == "old1"
    assert st.gpu_name == "NVIDIA GeForce RTX 4090"
    assert "start:old1" in fake.calls and "create" not in fake.calls
    assert phases[0] == pm.STARTING and phases[-1] == pm.ONLINE


def test_gpu_taken_terminates_and_recreates(make):
    """The owner's 2026-09-17 case: Start refused because the stopped pod's GPU is gone."""
    fake = FakeRunPod([dict(STOPPED_POD)], start_ok=False)
    mgr, clock = make(fake)
    st = mgr.ensure_online()
    assert st.phase == pm.ONLINE
    assert st.pod_id == "new1"
    assert fake.calls.index("terminate:old1") < fake.calls.index("create")
    assert "old1" not in fake.pods


def test_cpu_only_resume_is_treated_as_a_refusal(make):
    """What RunPod really did on 2026-09-17: Start returned 2xx but resumed the pod on CPU only
    ($0.37/h, app says "CPU Only"). The gateway must replace it, not call that online."""
    fake = FakeRunPod([dict(STOPPED_POD)], cpu_resume=True)
    mgr, clock = make(fake)
    messages = []
    st = mgr.ensure_online(lambda ph, msg: messages.append(msg))
    assert st.phase == pm.ONLINE and st.pod_id == "new1"
    assert st.gpu_name == "NVIDIA GeForce RTX 4090"
    assert "terminate:old1" in fake.calls
    assert any("could not give the stopped pod its GPU back" in m for m in messages)


def test_running_cpu_only_pod_is_replaced_and_capped(make):
    cpu_pod = {"id": "cpu1", "name": "pitchvision", "desiredStatus": "RUNNING", "gpuCount": None, "machine": {}}
    fake = FakeRunPod([cpu_pod])
    mgr, clock = make(fake)
    assert mgr.refresh().phase == pm.OFFLINE and "WITHOUT a GPU" in mgr.state.message
    st = mgr.ensure_online()
    assert st.phase == pm.ONLINE and st.pod_id == "new1" and "terminate:cpu1" in fake.calls

    class AlwaysCpu(FakeRunPod):
        def create_pod(self, body):
            pod = super().create_pod(body)
            self.pods[pod.id]["gpuCount"] = None
            self.pods[pod.id]["machine"] = {}
            return PodInfo.from_api(self.pods[pod.id])

    mgr2, _ = make(AlwaysCpu([dict(cpu_pod)]))
    with pytest.raises(PodUnavailable):
        mgr2.ensure_online()
    assert mgr2.state.phase == pm.ERROR


def test_no_gpu_anywhere_waits_and_retries_until_stock_appears(make):
    fake = FakeRunPod([], create_ok=False)
    mgr, clock = make(fake)
    seen = []

    def on_progress(phase, message):
        seen.append((phase, message))
        # Stock appears after the third refusal.
        if seen.count((pm.WAITING_FOR_GPU, message)) and fake.calls.count("create") >= 3:
            fake.create_ok = True

    st = mgr.ensure_online(on_progress)
    assert st.phase == pm.ONLINE
    waiting = [m for ph, m in seen if ph == pm.WAITING_FOR_GPU]
    assert waiting and "No GPU available" in waiting[0]
    # It kept the user informed AND slept the retry interval between attempts, never spun.
    assert clock.slept.count(60) >= 3
    assert any("waiting" in m and "min" in m for m in waiting[1:])


def test_deadline_gives_up_with_pod_unavailable(make):
    fake = FakeRunPod([], create_ok=False)
    mgr, clock = make(fake)
    with pytest.raises(PodUnavailable):
        mgr.ensure_online(deadline_s=100)
    assert mgr.state.phase == pm.WAITING_FOR_GPU


def test_auth_error_is_not_retried(make):
    class Unauthorized(FakeRunPod):
        def find_pod(self, name):
            raise RunPodError("GET /pods -> HTTP 401: unauthorized", 401)

    mgr, clock = make(Unauthorized())
    with pytest.raises(PodUnavailable):
        mgr.ensure_online()
    assert mgr.state.phase == pm.ERROR
    assert clock.slept == []


def test_running_pod_waits_for_app_to_boot(make):
    running = {"id": "run1", "name": "pitchvision", "desiredStatus": "RUNNING", "gpuCount": 1,
               "machine": {"gpuTypeId": "NVIDIA L4"}, "runtime": {"uptimeInSeconds": 5}}
    fake = FakeRunPod([running], healthy_after_polls=3)
    mgr, clock = make(fake)
    phases = []
    st = mgr.ensure_online(lambda ph, msg: phases.append(ph))
    assert st.phase == pm.ONLINE
    assert pm.BOOTING in phases
    assert "start:run1" not in fake.calls and "create" not in fake.calls


def test_boot_that_never_comes_up_is_replaced(make):
    running = {"id": "stuck", "name": "pitchvision", "desiredStatus": "RUNNING", "gpuCount": 1,
               "machine": {"gpuTypeId": "NVIDIA L4"}, "runtime": {"uptimeInSeconds": 0}}
    fake = FakeRunPod([running])
    mgr, clock = make(fake, boot_cap_s=120)

    real_probe = fake.probe

    def probe(url, session=None, timeout_s=None):
        # The replacement pod boots fine; only "stuck" never answers.
        if "stuck" in url:
            return None
        return real_probe(url, session, timeout_s)

    pm.rp.probe_health = probe
    st = mgr.ensure_online()
    assert st.phase == pm.ONLINE and st.pod_id == "new1"
    assert "terminate:stuck" in fake.calls


def test_refresh_reports_without_acting(make):
    fake = FakeRunPod([dict(STOPPED_POD)])
    mgr, clock = make(fake)
    st = mgr.refresh()
    assert st.phase == pm.OFFLINE
    assert st.pod_id == "old1"
    assert not any(c.startswith(("start", "create", "terminate")) for c in fake.calls)
    assert st.public() == {"phase": "offline", "message": st.message, "gpu_name": ""}
