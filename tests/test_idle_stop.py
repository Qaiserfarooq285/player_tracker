"""Idle auto-stop watchdog (`apps/api/idle_stop.py`, docs/DEPLOY.md "Idle auto-stop") -- the pure
`should_stop` decision, the RunPod REST stop call (network mocked, never a real HTTP request), and
env-var parsing. All `requests.post` calls in this file are monkeypatched -- nothing here reaches
the real RunPod API."""

from __future__ import annotations

import threading

import requests

import apps.api.idle_stop as idle_stop
from apps.api.idle_stop import IdleWatchdog, should_stop, watchdog_from_env

# ---------------------------------------------------------------- should_stop (pure)


def test_should_stop_false_below_threshold():
    assert should_stop(now=100.0, last_activity=95.0, jobs_in_flight=0, threshold_s=30.0) is False


def test_should_stop_true_above_threshold_and_idle():
    assert should_stop(now=200.0, last_activity=100.0, jobs_in_flight=0, threshold_s=30.0) is True


def test_should_stop_false_when_job_in_flight_even_past_threshold():
    assert should_stop(now=200.0, last_activity=100.0, jobs_in_flight=1, threshold_s=30.0) is False


# ---------------------------------------------------------------- IdleWatchdog.stop_pod


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _watchdog(**overrides) -> IdleWatchdog:
    kwargs = dict(
        get_inflight_count=lambda: 0,
        api_key="testkey",
        pod_id="pod123",
        threshold_s=30.0,
        clock=lambda: 1000.0,
    )
    kwargs.update(overrides)
    return IdleWatchdog(**kwargs)


def test_stop_pod_posts_correct_url_and_bearer_header_and_returns_true_on_2xx(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(idle_stop.requests, "post", fake_post)
    wd = _watchdog()
    assert wd.stop_pod() is True
    assert captured["url"] == "https://rest.runpod.io/v1/pods/pod123/stop"
    assert captured["headers"] == {"Authorization": "Bearer testkey"}
    # RunPod 500s a body-less stop POST; an empty JSON object is the minimum it accepts.
    assert captured["json"] == {}
    assert captured["timeout"] == idle_stop._STOP_REQUEST_TIMEOUT_S


def test_stop_pod_returns_false_on_5xx_without_raising(monkeypatch):
    monkeypatch.setattr(idle_stop.requests, "post", lambda *a, **k: _FakeResponse(500, "boom"))
    wd = _watchdog()
    assert wd.stop_pod() is False


def test_stop_pod_returns_false_on_network_exception_without_raising(monkeypatch):
    def raise_it(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(idle_stop.requests, "post", raise_it)
    wd = _watchdog()
    assert wd.stop_pod() is False  # never raises out of stop_pod


# ---------------------------------------------------------------- tick fires once only


def test_tick_stops_pod_once_then_stays_stopped(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return _FakeResponse(200)

    monkeypatch.setattr(idle_stop.requests, "post", fake_post)
    wd = _watchdog(threshold_s=10.0, clock=lambda: 1000.0)
    wd.tick(now=1000.0 + 10.0)  # idle long enough -> stops
    assert wd.stopped is True
    assert len(calls) == 1
    wd.tick(now=1000.0 + 1000.0)  # way past threshold again, but already stopped
    assert len(calls) == 1


def test_tick_does_nothing_while_job_in_flight(monkeypatch):
    monkeypatch.setattr(idle_stop.requests, "post", lambda *a, **k: _FakeResponse(200))
    wd = _watchdog(get_inflight_count=lambda: 1, threshold_s=10.0)
    wd.tick(now=1000.0 + 999.0)
    assert wd.stopped is False


def test_tick_retries_on_the_next_call_after_a_failed_stop(monkeypatch):
    responses = [_FakeResponse(500, "boom"), _FakeResponse(200)]
    monkeypatch.setattr(idle_stop.requests, "post", lambda *a, **k: responses.pop(0))
    wd = _watchdog(threshold_s=10.0, clock=lambda: 1000.0)
    wd.tick(now=1000.0 + 10.0)  # first attempt fails (500) -> not marked stopped
    assert wd.stopped is False
    wd.tick(now=1000.0 + 20.0)  # second attempt succeeds
    assert wd.stopped is True


# ---------------------------------------------------------------- enabled / start()


def test_disabled_when_api_key_missing():
    assert _watchdog(api_key="").enabled is False


def test_disabled_when_pod_id_missing():
    assert _watchdog(pod_id="").enabled is False


def test_disabled_when_threshold_not_positive():
    assert _watchdog(threshold_s=0.0).enabled is False


def test_enabled_when_key_and_pod_id_and_threshold_all_set():
    assert _watchdog().enabled is True


def test_start_does_not_spawn_thread_when_disabled():
    wd = _watchdog(api_key="")
    before = threading.active_count()
    wd.start()
    assert threading.active_count() == before


def test_start_spawns_a_daemon_thread_when_enabled():
    # A long check_interval_s means the thread just sleeps and never calls tick() (so never makes
    # a real network call) for the lifetime of this test -- it's a daemon thread so the process
    # doesn't wait on it to exit.
    wd = _watchdog(check_interval_s=1000.0)
    before = threading.active_count()
    wd.start()
    assert threading.active_count() == before + 1


# ---------------------------------------------------------------- status()


def test_status_reports_enabled_and_seconds_since_activity():
    now = {"t": 1000.0}
    wd = _watchdog(threshold_s=600.0, clock=lambda: now["t"])
    now["t"] = 1042.5
    status = wd.status()
    assert status == {"enabled": True, "idle_minutes": 10.0, "seconds_since_activity": 42.5}


def test_touch_resets_seconds_since_activity():
    now = {"t": 1000.0}
    wd = _watchdog(clock=lambda: now["t"])
    now["t"] = 1100.0
    wd.touch()
    now["t"] = 1105.0
    assert wd.status()["seconds_since_activity"] == 5.0


# ---------------------------------------------------------------- watchdog_from_env


def test_watchdog_from_env_parses_minutes(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_POD_ID", "p")
    monkeypatch.setenv("PV_IDLE_STOP_MINUTES", "2")
    monkeypatch.delenv("PV_RUNPOD_API_BASE", raising=False)
    monkeypatch.delenv("PV_IDLE_CHECK_SECONDS", raising=False)
    wd = watchdog_from_env(lambda: 0)
    assert wd.threshold_s == 120.0
    assert wd.api_base == idle_stop.DEFAULT_API_BASE
    assert wd.check_interval_s == idle_stop.DEFAULT_CHECK_INTERVAL_S
    assert wd.enabled is True


def test_watchdog_from_env_defaults_to_30_minutes_when_unset(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_POD_ID", "p")
    monkeypatch.delenv("PV_IDLE_STOP_MINUTES", raising=False)
    wd = watchdog_from_env(lambda: 0)
    assert wd.threshold_s == 30.0 * 60.0


def test_watchdog_from_env_disabled_when_key_missing(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "p")
    wd = watchdog_from_env(lambda: 0)
    assert wd.enabled is False


def test_watchdog_from_env_honours_custom_api_base(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_POD_ID", "p")
    monkeypatch.setenv("PV_RUNPOD_API_BASE", "http://localhost:9999")
    wd = watchdog_from_env(lambda: 0)
    assert wd.api_base == "http://localhost:9999"
