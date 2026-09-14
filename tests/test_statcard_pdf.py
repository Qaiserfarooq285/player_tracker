"""Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14): tests for
`src/pipeline/statcard_pdf.py`. Real `reportlab` renders (no GPU/video I/O, small/fast) plus the
optional-dependency fail-soft path, exercised via `write_player_output` with the import mocked out
-- the anti-regression test for the frontend's old "not available (...)" -> 0 coercion bug
(Plan Context, CLAUDE.md Golden Rule 5).
"""

from __future__ import annotations

import builtins

import pytest

reportlab = pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")

from src.pipeline.statcard_pdf import render_statcard_pdf  # noqa: E402


def _read_pdf_bytes(path) -> bytes:
    return path.read_bytes()


def test_render_statcard_pdf_creates_valid_nonempty_pdf(tmp_path):
    out = tmp_path / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=7,
        counts={"touch": 3, "pass": 1},
        possession_seconds=12.5,
        distance_result={"distance": 42.0, "unit": "bbox_heights"},
        timeline_rows=[],
        output_path=out,
    )
    assert out.exists()
    data = _read_pdf_bytes(out)
    assert len(data) > 0
    # Valid PDF: starts with the standard header and ends with the standard trailer marker.
    assert data.startswith(b"%PDF-")
    assert b"%%EOF" in data


def test_render_statcard_pdf_contains_jersey_number(tmp_path):
    out = tmp_path / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=77,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        output_path=out,
    )
    data = _read_pdf_bytes(out)
    # pageCompression=0 (statcard_pdf.PAGE_COMPRESSION) keeps the content stream plain-text/
    # greppable -- see that module's own comment for why -- so the literal drawn string is
    # findable directly in the file bytes without a PDF-parsing dependency.
    assert b"Player #77" in data


def test_render_statcard_pdf_prints_not_available_verbatim_for_goals_and_assists(tmp_path):
    # THE anti-regression test (Plan Context): the old frontend exporter coerced the non-numeric
    # "not available (...)" string to a bare 0 via `_safe_int` -- this must never happen here.
    out = tmp_path / "statcard.pdf"
    reason = "not available (no legible scoreboard found by the region-activation scan)"
    render_statcard_pdf(
        jersey_number=10,
        counts={},  # zero goals/assists
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        goal_reason=reason,
        output_path=out,
    )
    data = _read_pdf_bytes(out)
    assert b"not available" in data
    # And specifically NOT flattened to a bare zero for Goals/Assists.
    assert b"Goals: 0" not in data
    assert b"Assists: 0" not in data


def test_render_statcard_pdf_zero_goals_with_no_reason_shows_real_zero(tmp_path):
    out = tmp_path / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=2,
        counts={"touch": 1},
        possession_seconds=None,
        distance_result=None,
        timeline_rows=[],
        goal_reason=None,
        identity_status="Human-provided (manual annotation)",
        output_path=out,
    )
    data = _read_pdf_bytes(out)
    assert b"Goals: 0" in data
    assert b"Assists: 0" in data
    assert b"not available" not in data
    assert b"uncertain" in data  # possession/distance, both None


def test_render_statcard_pdf_includes_timeline_rows_with_source(tmp_path):
    out = tmp_path / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=2,
        counts={"touch": 1},
        possession_seconds=5.0,
        distance_result={"distance": 1.0, "unit": "bbox_heights"},
        timeline_rows=[
            {
                "t_start": 26 * 60 + 56.0,
                "t_end": 26 * 60 + 56.0,
                "type": "touch",
                "label": "Touch",
                "confidence": 0.95,
                "take_id": 0,
                "source": "manual_touch",
            }
        ],
        output_path=out,
    )
    data = _read_pdf_bytes(out)
    assert b"manual_touch" in data
    assert b"Touch" in data


def test_render_statcard_pdf_empty_timeline_says_no_events(tmp_path):
    out = tmp_path / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=9,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        output_path=out,
    )
    data = _read_pdf_bytes(out)
    assert b"no events attributed" in data


def test_render_statcard_pdf_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "player_5" / "statcard.pdf"
    render_statcard_pdf(
        jersey_number=5,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        output_path=out,
    )
    assert out.exists()


def test_write_player_output_skips_pdf_gracefully_when_reportlab_missing(tmp_path, monkeypatch):
    """CLAUDE.md §7: reportlab is optional (`api` extra). When it can't be imported,
    write_player_output must still write statcard.md and must never raise -- it just logs a
    warning and skips statcard.pdf. Simulates a missing reportlab install by making Python's own
    import machinery raise ImportError for it, exactly as it would in an environment without the
    `api` extra installed.
    """
    from src.pipeline.player_output import write_player_output

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("simulated: reportlab not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    player_dir = tmp_path / "player_3"
    write_player_output(
        player_dir=player_dir,
        jersey_number=3,
        events=[],
        possession_seconds=None,
        distance_result=None,
        takes_by_id={},
        video_path=tmp_path / "does_not_matter.mp4",
        highlights_cfg={},
        goal_reason=None,
        identity_status="Verified",
    )

    assert (player_dir / "statcard.md").exists()
    assert not (player_dir / "statcard.pdf").exists()
