"""FastAPI endpoint tests for Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14):
`GET /api/download/statcard/{slug}/{jersey}`, `/api/results/{slug}`'s new `statcard_pdf_url`
field, and the `/media/{path:path}` security fix (BASE_DIR probe removed).

Uses `fastapi.testclient.TestClient` directly against `apps.api.main.app` -- no existing test file
in this repo exercises the API layer yet, so this establishes the pattern. `OUTPUT_DIR`/
`INPUT_DIR` are monkeypatched to an isolated `tmp_path` per test so nothing here touches the real
repo's `output/`/`input/` directories.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main

client = TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Point OUTPUT_DIR/INPUT_DIR at an isolated tmp_path for every test in this file -- never
    read/write the real repo's own output/input directories."""
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    output_dir.mkdir()
    input_dir.mkdir()
    monkeypatch.setattr(api_main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(api_main, "INPUT_DIR", input_dir)
    return output_dir, input_dir


def _write_player_with_pdf(output_dir: Path, slug: str, jersey: int) -> Path:
    player_dir = output_dir / slug / "players" / f"player_{jersey}"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text("# Player Statistics\n")
    pdf_path = player_dir / "statcard.pdf"
    # A minimal, real, valid PDF byte string -- good enough for FileResponse/`%PDF-` sniffing;
    # this test does not need a reportlab-rendered document, only a real file on disk.
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    return pdf_path


# ---------------------------------------------------------------------------------------------
# GET /api/download/statcard/{slug}/{jersey}
# ---------------------------------------------------------------------------------------------


def test_download_statcard_pdf_200_with_real_pdf(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    _write_player_with_pdf(output_dir, "myslug", 7)

    res = client.get("/api/download/statcard/myslug/7")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")
    assert "player_7_statcard.pdf" in res.headers.get("content-disposition", "")


def test_download_statcard_pdf_404_missing_slug(_isolated_dirs):
    res = client.get("/api/download/statcard/does_not_exist/7")
    assert res.status_code == 404


def test_download_statcard_pdf_404_missing_player(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    (output_dir / "myslug").mkdir()
    (output_dir / "myslug" / "players").mkdir()

    res = client.get("/api/download/statcard/myslug/99")

    assert res.status_code == 404


def test_download_statcard_pdf_404_missing_pdf_file(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    player_dir = output_dir / "myslug" / "players" / "player_7"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text("# Player Statistics\n")
    # Deliberately no statcard.pdf -- e.g. reportlab wasn't installed when this run happened.

    res = client.get("/api/download/statcard/myslug/7")

    assert res.status_code == 404


def test_download_statcard_pdf_rejects_path_traversal_in_jersey(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    _write_player_with_pdf(output_dir, "myslug", 7)

    res = client.get("/api/download/statcard/myslug/../../../etc/passwd")

    assert res.status_code == 404
    assert b"root:" not in res.content


def test_download_statcard_pdf_rejects_path_traversal_in_slug(_isolated_dirs):
    res = client.get("/api/download/statcard/..%2f..%2f..%2fetc/7")
    assert res.status_code == 404


# ---------------------------------------------------------------------------------------------
# /api/results/{slug} gains statcard_pdf_url
# ---------------------------------------------------------------------------------------------


def test_results_includes_statcard_pdf_url_when_pdf_exists(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    _write_player_with_pdf(output_dir, "myslug", 7)

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert len(data["players"]) == 1
    assert data["players"][0]["statcard_pdf_url"] == "/api/download/statcard/myslug/7"
    assert data["primary_player"]["statcard_pdf_url"] == "/api/download/statcard/myslug/7"

    # And the URL it advertises is actually downloadable.
    dl = client.get(data["players"][0]["statcard_pdf_url"])
    assert dl.status_code == 200


def test_results_statcard_pdf_url_is_none_when_pdf_missing(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    player_dir = output_dir / "myslug" / "players" / "player_7"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text("# Player Statistics\n")

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert data["players"][0]["statcard_pdf_url"] is None


# ---------------------------------------------------------------------------------------------
# /media/{path:path} security fix -- BASE_DIR probe dropped
# ---------------------------------------------------------------------------------------------


def test_media_env_file_404s(_isolated_dirs):
    """Anti-regression test for the security fix: `/media/.env` must 404, never serve the repo
    root's secrets file. Uses the REAL, unpatched BASE_DIR (this repo checks out a real `.env`
    locally, see CLAUDE.md §8) -- INPUT_DIR/OUTPUT_DIR are patched to an isolated tmp_path by the
    autouse fixture, but BASE_DIR itself is untouched, which is exactly what this test needs to
    confirm the BASE_DIR probe is really gone (a `.env` under the isolated tmp_path would prove
    nothing)."""
    res = client.get("/media/.env")
    assert res.status_code == 404


def test_media_serves_real_file_under_output_dir(_isolated_dirs):
    output_dir, _ = _isolated_dirs
    (output_dir / "myslug").mkdir()
    (output_dir / "myslug" / "clip.mp4").write_bytes(b"not a real video, just bytes")

    res = client.get("/media/output/myslug/clip.mp4")

    assert res.status_code == 200
    assert res.content == b"not a real video, just bytes"


def test_media_serves_real_file_under_input_dir(_isolated_dirs):
    _, input_dir = _isolated_dirs
    (input_dir / "clip.mp4").write_bytes(b"input bytes")

    res = client.get("/media/clip.mp4")

    assert res.status_code == 200
    assert res.content == b"input bytes"


def test_media_rejects_traversal_out_of_input_and_output(_isolated_dirs):
    res = client.get("/media/..%2f..%2f..%2fetc%2fpasswd")
    assert res.status_code == 404
