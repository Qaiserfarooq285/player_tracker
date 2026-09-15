"""FastAPI endpoint tests for Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14):
`GET /api/download/statcard/{slug}/{jersey}`, `/api/results/{slug}`'s new `statcard_pdf_url`
field, and the `/media/{path:path}` security fix (BASE_DIR probe removed).

Extended for the Plan's re-check Fix D (2026-09-15): `/api/results/{slug}`'s `primary_player`
resolution (`target.json`/`selection.json`, never a lexicographic accident) and on-demand
`statcard.pdf` generation from an archived `statcard.md` when no PDF exists yet.

Uses `fastapi.testclient.TestClient` directly against `apps.api.main.app` -- no existing test file
in this repo exercises the API layer yet, so this establishes the pattern. `OUTPUT_DIR`/
`INPUT_DIR`/`WORK_DIR` are monkeypatched to an isolated `tmp_path` per test so nothing here touches
the real repo's own `output/`/`input/`/`work/` directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
from src.track.target import TargetProfile, save_target_profile

client = TestClient(api_main.app)

# A real statcard.md, same shape as the owner's own on-disk
# `output/chelsea_burnley_target10/players/player_2/statcard.md` (CLAUDE.md §13.2's exact
# template) -- including the literal "not available (...)" Goals/Assists reason text the Fix D
# on-demand PDF path must carry through VERBATIM, never coerced to a bare "0".
_GOAL_NOT_AVAILABLE_REASON = (
    "not available (auto goal-line detection is not yet implemented; this video's own "
    "annotations may record a goal, but manual-mode stats are now auto-detected per the owner's "
    "own choice, so an unconfirmed auto-goal reads as uncertain, never a guessed 0 or a "
    "silently-adopted sidecar count)"
)

_REALISTIC_STATCARD_MD = f"""# Player Statistics

## Player #2

**Identity Status:** Human-provided (manual annotation)

**Touches:** 4
**Passes:** 0
**Turnovers:** 0
**Sprints/Runs:** 3
**Goals:** {_GOAL_NOT_AVAILABLE_REASON}
**Assists:** {_GOAL_NOT_AVAILABLE_REASON}
**Shots:** 0
**Tackles:** 0
**Saves:** 0
**Dribbles:** 0
**Possession Time:** 0.0s
**Distance Covered:** 20.23 bbox_heights (uncalibrated)

## Event Timeline

| Time | Event | Confidence |
|------|-------|------------|
| 0:32.8 | Sprint | 0.42 |
| 0:42.7 | Touch | 0.23 |
"""


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Point OUTPUT_DIR/INPUT_DIR/WORK_DIR at an isolated tmp_path for every test in this file --
    never read/write the real repo's own output/input/work directories. `WORK_DIR` joined Plan Fix
    D's `primary_player` resolution tests below (`target.json`/`selection.json` live there)."""
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    work_dir = tmp_path / "work"
    output_dir.mkdir()
    input_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.setattr(api_main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(api_main, "INPUT_DIR", input_dir)
    monkeypatch.setattr(api_main, "WORK_DIR", work_dir)
    return output_dir, input_dir, work_dir


def _write_player_with_pdf(output_dir: Path, slug: str, jersey: int) -> Path:
    player_dir = output_dir / slug / "players" / f"player_{jersey}"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text("# Player Statistics\n")
    pdf_path = player_dir / "statcard.pdf"
    # A minimal, real, valid PDF byte string -- good enough for FileResponse/`%PDF-` sniffing;
    # this test does not need a reportlab-rendered document, only a real file on disk.
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    return pdf_path


def _write_player_with_markdown_only(
    output_dir: Path, slug: str, jersey: int, md_text: str = _REALISTIC_STATCARD_MD
) -> Path:
    """A player folder with a real `statcard.md` and deliberately NO `statcard.pdf` -- the exact
    on-disk shape of every run completed before Plan Stage 3 shipped the PDF feature (e.g. the
    real `output/chelsea_burnley_target10/players/player_2/`)."""
    player_dir = output_dir / slug / "players" / f"player_{jersey}"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text(md_text)
    return player_dir


# ---------------------------------------------------------------------------------------------
# GET /api/download/statcard/{slug}/{jersey}
# ---------------------------------------------------------------------------------------------


def test_download_statcard_pdf_200_with_real_pdf(_isolated_dirs):
    output_dir, _, _ = _isolated_dirs
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
    output_dir, _, _ = _isolated_dirs
    (output_dir / "myslug").mkdir()
    (output_dir / "myslug" / "players").mkdir()

    res = client.get("/api/download/statcard/myslug/99")

    assert res.status_code == 404


def test_download_statcard_pdf_404_when_neither_pdf_nor_markdown_exist(_isolated_dirs):
    """The one remaining genuine 404 case for this endpoint (Plan Fix D, 2026-09-15): a player
    folder that has neither a `statcard.pdf` NOR a `statcard.md` to regenerate one from. Before
    Fix D this exact on-disk shape (`statcard.md` present, no PDF) was ALSO a 404 -- see
    `test_download_statcard_pdf_generates_on_demand_from_markdown_only` below for that behaviour
    change; this test is what's left of the old `test_download_statcard_pdf_404_missing_pdf_file`
    once markdown-only is no longer a 404 case."""
    output_dir, _, _ = _isolated_dirs
    player_dir = output_dir / "myslug" / "players" / "player_7"
    player_dir.mkdir(parents=True)
    # Deliberately no statcard.md AND no statcard.pdf -- nothing on disk to serve OR regenerate
    # from (e.g. the run failed before `write_player_output` ever ran for this player).

    res = client.get("/api/download/statcard/myslug/7")

    assert res.status_code == 404


# ---------------------------------------------------------------------------------------------
# Plan Fix D ("streamed-gathering-treehouse" re-check, 2026-09-15): on-demand statcard.pdf
# generation from an archived statcard.md, for every run completed before Plan Stage 3 shipped
# the PDF feature.
# ---------------------------------------------------------------------------------------------


def test_download_statcard_pdf_generates_on_demand_from_markdown_only(_isolated_dirs):
    """THE behaviour change from Fix D: `statcard.md` exists, `statcard.pdf` does not (exactly
    the real, on-disk shape of e.g. `output/chelsea_burnley_target10/players/player_2/`, which
    completed before the PDF feature existed) -- this must now succeed with a real PDF, not 404."""
    pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")
    output_dir, _, _ = _isolated_dirs
    _write_player_with_markdown_only(output_dir, "myslug", 2)

    res = client.get("/api/download/statcard/myslug/2")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")
    assert "player_2_statcard.pdf" in res.headers.get("content-disposition", "")


def test_download_statcard_pdf_on_demand_preserves_not_available_verbatim(_isolated_dirs):
    """Correctness requirement (Plan Fix D): the regenerated PDF must carry the FULL "not
    available (...)" reason text for Goals/Assists verbatim -- `parse_statcard_markdown`'s
    `_safe_int` coercion (which the dashboard's OWN numeric fields still use) must never be what
    feeds this PDF's Goals/Assists lines, or it would silently flatten the real reason down to a
    bare `0` (the exact Golden Rule 5 regression `statcard_pdf.py`'s own module docstring already
    documents for the OLD frontend exporter this endpoint replaced)."""
    pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")
    output_dir, _, _ = _isolated_dirs
    _write_player_with_markdown_only(output_dir, "myslug", 2)

    res = client.get("/api/download/statcard/myslug/2")

    assert res.status_code == 200
    # pageCompression=0 (statcard_pdf.PAGE_COMPRESSION) keeps the PDF's own content stream
    # plain-text/greppable -- same convention `tests/test_statcard_pdf.py` already relies on.
    assert b"not available" in res.content
    assert b"auto goal-line detection is not yet implemented" in res.content
    assert b"Goals: 0" not in res.content
    assert b"Assists: 0" not in res.content
    # And the real numeric fields (never textual on this template) still show through normally.
    assert b"Touches: 4" in res.content


def test_download_statcard_pdf_on_demand_uses_real_event_timeline_json_when_present(
    _isolated_dirs,
):
    """When a player's own `events/event_timeline.json` (written by `write_player_output`
    alongside every real statcard.md) is present, the on-demand PDF uses ITS exact per-event
    `source` string rather than the honestly-labelled placeholder
    `_reconstruct_timeline_rows_from_markdown` falls back to when that JSON is missing."""
    pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")
    output_dir, _, _ = _isolated_dirs
    player_dir = _write_player_with_markdown_only(output_dir, "myslug", 2)
    events_dir = player_dir / "events"
    events_dir.mkdir()
    (events_dir / "event_timeline.json").write_text(
        json.dumps(
            [
                {
                    "t_start": 32.84,
                    "t_end": 35.2,
                    "type": "sprint",
                    "label": "Sprint",
                    "confidence": 0.42341252469468393,
                    "take_id": 0,
                    "source": "speed_heuristic_normalized_pixel",
                }
            ]
        )
    )

    res = client.get("/api/download/statcard/myslug/2")

    assert res.status_code == 200
    assert b"speed_heuristic_normalized_pixel" in res.content
    assert b"archived statcard.md" not in res.content


def test_download_statcard_pdf_on_demand_caches_pdf_to_disk(_isolated_dirs):
    """The generated PDF is written next to the markdown so a second request serves the cached
    file directly rather than regenerating -- "cache it next to the markdown, then serve it"."""
    pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")
    output_dir, _, _ = _isolated_dirs
    player_dir = _write_player_with_markdown_only(output_dir, "myslug", 2)

    res = client.get("/api/download/statcard/myslug/2")

    assert res.status_code == 200
    assert (player_dir / "statcard.pdf").exists()


def test_download_statcard_pdf_on_demand_404_not_500_when_reportlab_missing(
    _isolated_dirs, monkeypatch
):
    """Never a 500 when `reportlab` (the `api` extra, CLAUDE.md §7) isn't installed -- an honest
    404, the reason logged server-side. Simulated the same way `tests/test_statcard_pdf.py`
    already does: make Python's own import machinery raise `ImportError` for it, regardless of
    whether this test environment actually has it installed."""
    import builtins

    output_dir, _, _ = _isolated_dirs
    _write_player_with_markdown_only(output_dir, "myslug", 2)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("simulated: reportlab not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    res = client.get("/api/download/statcard/myslug/2")

    assert res.status_code == 404


def test_download_statcard_pdf_rejects_path_traversal_in_jersey(_isolated_dirs):
    output_dir, _, _ = _isolated_dirs
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
    output_dir, _, _ = _isolated_dirs
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
    output_dir, _, _ = _isolated_dirs
    player_dir = output_dir / "myslug" / "players" / "player_7"
    player_dir.mkdir(parents=True)
    (player_dir / "statcard.md").write_text("# Player Statistics\n")

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert data["players"][0]["statcard_pdf_url"] is None


# ---------------------------------------------------------------------------------------------
# Plan Fix D ("streamed-gathering-treehouse" re-check, 2026-09-15): `primary_player` resolution.
# Real bug: `/api/results/{slug}` used to hand back `sorted(glob("player_*"))[0]` -- a
# LEXICOGRAPHIC accident ("player_10" < "player_2" as strings) -- so a run that targeted #2 showed
# the dashboard #10's numbers. Every test below writes BOTH `player_10` and `player_2` (in that
# creation order, so a naive `sorted(...)[0]` would keep picking #10) to prove the fix actually
# changes the outcome, not just that it doesn't crash.
# ---------------------------------------------------------------------------------------------


def _write_two_players(output_dir: Path, slug: str) -> None:
    _write_player_with_pdf(output_dir, slug, 10)
    _write_player_with_pdf(output_dir, slug, 2)


def _write_target_profile(work_dir: Path, slug: str, jersey_number: int | None) -> None:
    profile = TargetProfile(
        jersey_number=jersey_number,
        jersey_source="click" if jersey_number is not None else None,
        established_take_id=0,
        established_track_id=1,
        established_t=0.0,
    )
    work_slug_dir = work_dir / slug
    work_slug_dir.mkdir(parents=True, exist_ok=True)
    save_target_profile(profile, work_slug_dir / "target.json")


def _write_selection_json(work_dir: Path, slug: str, target_jersey: int | None) -> None:
    work_slug_dir = work_dir / slug
    work_slug_dir.mkdir(parents=True, exist_ok=True)
    (work_slug_dir / "selection.json").write_text(
        json.dumps(
            {
                "video": f"input/{slug}.mp4",
                "target_jersey": target_jersey,
                "overall_confidence": 0.5,
                "needs_human_confirmation": False,
                "takes": [],
            }
        )
    )


def test_results_primary_player_uses_target_json_jersey_over_lexicographic_order(_isolated_dirs):
    output_dir, _, work_dir = _isolated_dirs
    _write_two_players(output_dir, "myslug")
    _write_target_profile(work_dir, "myslug", jersey_number=2)

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert data["primary_player"]["jersey_number"] == 2


def test_results_primary_player_uses_selection_json_when_no_target_json(_isolated_dirs):
    """`target.json` doesn't exist at all (the original filename/typed-jersey flow never writes
    one) -- falls back to `selection.json`'s own `target_jersey`."""
    output_dir, _, work_dir = _isolated_dirs
    _write_two_players(output_dir, "myslug")
    _write_selection_json(work_dir, "myslug", target_jersey=2)

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert data["primary_player"]["jersey_number"] == 2


def test_results_primary_player_target_json_takes_priority_over_selection_json(_isolated_dirs):
    output_dir, _, work_dir = _isolated_dirs
    _write_two_players(output_dir, "myslug")
    _write_target_profile(work_dir, "myslug", jersey_number=2)
    _write_selection_json(work_dir, "myslug", target_jersey=10)

    res = client.get("/api/results/myslug")

    assert res.json()["primary_player"]["jersey_number"] == 2


def test_results_primary_player_falls_back_to_sorted_order_when_no_profile_or_selection(
    _isolated_dirs,
):
    """Neither cache file exists (e.g. these players came from the original 5
    `clip<N> <jersey>.mp4` filename-based runs) -- falls back SILENTLY to the pre-existing
    first-sorted-entry behaviour, unchanged."""
    output_dir, _, _ = _isolated_dirs
    _write_two_players(output_dir, "myslug")

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    data = res.json()
    assert data["primary_player"]["jersey_number"] == 10  # unchanged pre-existing behaviour


def test_results_primary_player_falls_back_when_target_jersey_has_no_matching_player_folder(
    _isolated_dirs,
):
    """Judgement call (documented in the plan hand-off): a resolved jersey number that has no
    matching `player_<N>` folder in THIS run's own output (e.g. only #10/#2 were ever
    verified/selected, but `target.json` names #99 for some other reason) falls back to the
    pre-existing first-sorted entry, silently -- never a crash, never a `None` where a real
    (if not perfectly resolved) player is available."""
    output_dir, _, work_dir = _isolated_dirs
    _write_two_players(output_dir, "myslug")
    _write_target_profile(work_dir, "myslug", jersey_number=99)

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    assert res.json()["primary_player"]["jersey_number"] == 10


def test_results_primary_player_falls_back_when_target_json_is_malformed(_isolated_dirs):
    """A corrupt/partial `target.json` (e.g. an interrupted write) must degrade to the next
    resolution step, never a 500."""
    output_dir, _, work_dir = _isolated_dirs
    _write_two_players(output_dir, "myslug")
    work_slug_dir = work_dir / "myslug"
    work_slug_dir.mkdir(parents=True)
    (work_slug_dir / "target.json").write_text("{not valid json")
    _write_selection_json(work_dir, "myslug", target_jersey=2)

    res = client.get("/api/results/myslug")

    assert res.status_code == 200
    assert res.json()["primary_player"]["jersey_number"] == 2


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
    output_dir, _, _ = _isolated_dirs
    (output_dir / "myslug").mkdir()
    (output_dir / "myslug" / "clip.mp4").write_bytes(b"not a real video, just bytes")

    res = client.get("/media/output/myslug/clip.mp4")

    assert res.status_code == 200
    assert res.content == b"not a real video, just bytes"


def test_media_serves_real_file_under_input_dir(_isolated_dirs):
    _, input_dir, _ = _isolated_dirs
    (input_dir / "clip.mp4").write_bytes(b"input bytes")

    res = client.get("/media/clip.mp4")

    assert res.status_code == 200
    assert res.content == b"input bytes"


def test_media_rejects_traversal_out_of_input_and_output(_isolated_dirs):
    res = client.get("/media/..%2f..%2f..%2fetc%2fpasswd")
    assert res.status_code == 404
