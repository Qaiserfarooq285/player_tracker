"""Hosted-deployment API behaviours (docs/DEPLOY.md, 2026-09-16): the chunked upload endpoint the
web UI now uses (Cloudflare caps one request body at 100 MB, a 4K clip is bigger) and the
`PV_ACCESS_PASSWORD` session gate, which is ON BY DEFAULT (built-in `admin1122`,
`apps/api/access.py`) unless explicitly turned `off`. Same isolation pattern as
`test_api_download_statcard.py`: `INPUT_DIR` is monkeypatched to `tmp_path`, so nothing here
touches the real `input/`."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
from apps.api.access import DEFAULT_ACCESS_PASSWORD, resolve_access_password


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "INPUT_DIR", tmp_path / "input")
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(api_main, "_UPLOADS", {})
    return TestClient(api_main.app)


def _chunk(client, upload_id, index, total, data, filename="my clip.mp4"):
    return client.post(
        "/api/upload/chunk",
        data={"upload_id": upload_id, "index": index, "total": total, "filename": filename},
        files={"chunk": (filename, data, "application/octet-stream")},
    )


# ---------------------------------------------------------------- chunked upload


def test_chunks_reassemble_in_order_into_input_dir(client, tmp_path):
    parts = [b"a" * 10, b"b" * 10, b"c" * 3]
    uid = "upload-0001"
    for i, part in enumerate(parts[:-1]):
        r = _chunk(client, uid, i, 3, part)
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "partial", "upload_id": uid, "received": i + 1, "total": 3}
    r = _chunk(client, uid, 2, 3, parts[-1])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["filename"] == "my_clip.mp4"  # spaces normalised like /api/upload
    assert (tmp_path / "input" / "my_clip.mp4").read_bytes() == b"".join(parts)
    # the partial file is gone and no bookkeeping leaks
    assert not list((tmp_path / "input" / ".uploads").glob("*.part"))
    assert uid not in api_main._UPLOADS


def test_out_of_order_chunk_is_rejected_not_appended(client, tmp_path):
    uid = "upload-0002"
    assert _chunk(client, uid, 0, 3, b"x" * 4).status_code == 200
    r = _chunk(client, uid, 2, 3, b"z" * 4)
    assert r.status_code == 409
    assert "expected chunk 1" in r.json()["detail"]
    part = tmp_path / "input" / ".uploads" / f"{uid}.part"
    assert part.read_bytes() == b"x" * 4  # nothing was appended


def test_chunk_for_unknown_upload_is_404(client):
    r = _chunk(client, "never-started-1", 1, 3, b"x")
    assert r.status_code == 404


def test_non_video_extension_is_rejected_by_both_upload_paths(client):
    r = _chunk(client, "upload-0003", 0, 1, b"x", filename="notes.txt")
    assert r.status_code == 400
    r = client.post("/api/upload", files={"file": ("evil.sh", b"x", "text/plain")})
    assert r.status_code == 400


def test_bad_upload_id_is_rejected(client):
    r = _chunk(client, "../../etc", 0, 1, b"x")
    assert r.status_code == 400


# ---------------------------------------------------------------- access gate


def test_gate_is_off_when_access_password_is_empty(client, monkeypatch):
    # What `PV_ACCESS_PASSWORD=off` resolves to (apps.api.access.resolve_access_password) -- the
    # gate itself only ever looks at `ACCESS_PASSWORD`, so patching it directly here is equivalent
    # and keeps this test decoupled from env-var parsing (covered separately below).
    monkeypatch.setattr(api_main, "ACCESS_PASSWORD", "")
    assert client.get("/api/auth/status").json() == {"required": False, "authenticated": True}
    assert client.get("/api/videos").status_code == 200


def test_gate_is_on_by_default_with_admin1122(client, monkeypatch):
    # Simulates a deployment where PV_ACCESS_PASSWORD was never set: main.py's own module-level
    # resolution already does this at import time (via resolve_access_password), so this exercises
    # the same default value the real gate falls back to.
    monkeypatch.setattr(api_main, "ACCESS_PASSWORD", api_main.DEFAULT_ACCESS_PASSWORD)
    assert client.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    assert client.get("/api/videos").status_code == 401
    monkeypatch.setattr(api_main, "_LOGIN_FAIL_DELAY_S", 0.0)
    r = client.post("/api/login", json={"password": "admin1122"})
    assert r.status_code == 200
    assert api_main._SESSION_COOKIE in r.cookies
    assert client.get("/api/videos").status_code == 200


@pytest.mark.parametrize(
    "raw, expected_password, expected_is_default",
    [
        (None, DEFAULT_ACCESS_PASSWORD, True),
        ("custom", "custom", False),
        ("off", "", False),
        ("OFF", "", False),
        ("  ", DEFAULT_ACCESS_PASSWORD, True),
    ],
)
def test_resolve_access_password(raw, expected_password, expected_is_default):
    assert resolve_access_password(raw) == (expected_password, expected_is_default)


def test_gate_blocks_api_until_login_then_allows(client, monkeypatch):
    monkeypatch.setattr(api_main, "ACCESS_PASSWORD", "s3cret")
    # public surface still reachable so the UI can render its login overlay
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    # everything else is 401
    assert client.get("/api/videos").status_code == 401
    assert client.get("/media/anything.mp4").status_code == 401
    assert _chunk(client, "upload-0004", 0, 1, b"x").status_code == 401
    # wrong password
    monkeypatch.setattr(api_main, "_LOGIN_FAIL_DELAY_S", 0.0)
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    assert client.get("/api/videos").status_code == 401
    # right password sets the session cookie and unlocks the API
    r = client.post("/api/login", json={"password": "s3cret"})
    assert r.status_code == 200
    assert api_main._SESSION_COOKIE in r.cookies
    assert client.get("/api/videos").status_code == 200
    assert client.get("/api/auth/status").json() == {"required": True, "authenticated": True}
    # logout drops it again
    assert client.post("/api/logout").status_code == 200
    assert client.get("/api/videos").status_code == 401


def test_forged_cookie_is_not_a_session(client, monkeypatch):
    monkeypatch.setattr(api_main, "ACCESS_PASSWORD", "s3cret")
    client.cookies.set(api_main._SESSION_COOKIE, "0" * 64)
    assert client.get("/api/videos").status_code == 401
