"""Pure-logic unit tests for src/pipeline/player_output.py (ADR-15, CLAUDE.md §13.2) — no
GPU/video I/O."""

from __future__ import annotations

import pytest

from src.common.types import Event, EventType
from src.pipeline.player_output import (
    _HIGHLIGHT_CATEGORIES,
    _key_moment_label,
    build_event_timeline_rows,
    render_statcard_markdown,
    write_player_output,
)


def _event(
    etype: EventType, t: float, confidence: float = 0.5, evidence: dict | None = None
) -> Event:
    return Event(
        id=f"{etype.value}-{t}",
        type=etype,
        t_start=t,
        t_end=t,
        player_track_id=1,
        take_id=0,
        confidence=confidence,
        source="test",
        evidence=evidence or {},
    )


def test_timeline_includes_save_but_excludes_possession():
    # Stage 6 (2026-08-31): SAVE gained a timeline row (owner's explicit "every save (for
    # goalkeepers)" ask, CLAUDE.md §13.2/§13.3) -- POSSESSION still has none, it only feeds the
    # "Possession Time" summary stat.
    events = [
        _event(EventType.SAVE, 1.0),
        _event(EventType.POSSESSION, 2.0),
        _event(EventType.TOUCH, 3.0),
    ]
    rows = build_event_timeline_rows(events)
    assert [r["label"] for r in rows] == ["Save", "Touch"]


def test_timeline_is_chronological():
    events = [
        _event(EventType.SHOT, 5.0),
        _event(EventType.TOUCH, 1.0),
        _event(EventType.PASS, 3.0),
    ]
    rows = build_event_timeline_rows(events)
    assert [r["t_start"] for r in rows] == [1.0, 3.0, 5.0]


def test_key_moment_splits_celebration_vs_other():
    celebration = _event(
        EventType.KEY_MOMENT,
        1.0,
        evidence={"gemini_raw_response": "VERDICT=YES; players celebrating a goal"},
    )
    other = _event(
        EventType.KEY_MOMENT,
        2.0,
        evidence={"gemini_raw_response": "VERDICT=YES; a heated argument breaks out"},
    )
    rows = build_event_timeline_rows([celebration, other])
    labels = {r["t_start"]: r["label"] for r in rows}
    assert labels[1.0] == "Celebration"
    assert labels[2.0] == "Other Key Moment"


def test_timeline_never_pads_empty_categories():
    # only ONE real event -- the timeline must not invent rows for every category
    rows = build_event_timeline_rows([_event(EventType.TACKLE, 1.0)])
    assert len(rows) == 1


def test_render_statcard_markdown_uses_exact_template_headers():
    md = render_statcard_markdown(
        jersey_number=7,
        counts={"touch": 3, "pass": 1},
        possession_seconds=12.5,
        distance_result={"distance": 42.0, "unit": "bbox_heights"},
        timeline_rows=[],
    )
    assert "# Player Statistics" in md
    assert "## Player #7" in md
    assert "**Identity Status:** Verified" in md
    assert "**Touches:** 3" in md
    assert "**Passes:** 1" in md
    assert "**Turnovers:** 0" in md  # ADR-20: own line, never fabricated -- genuinely zero
    assert "**Sprints/Runs:** 0" in md  # never fabricated -- genuinely zero, reported as such
    # Bug fix 2026-08-31: no `goal_reason` supplied at all (the default, `None`) now means "there
    # is nothing uncertain about this zero" -- the real count "0" is shown, same as every other
    # stat, rather than the old unconditional "not available" (see render_statcard_markdown's own
    # docstring). A genuinely-uncertain auto-mode zero is covered separately below by
    # test_render_statcard_markdown_not_available_carries_the_specific_reason, which DOES pass a
    # non-None goal_reason.
    assert "**Goals:** 0" in md
    assert "**Assists:** 0" in md
    assert "**Possession Time:** 12.5s" in md
    assert "(uncalibrated)" in md
    assert "## Event Timeline" in md


def test_render_statcard_markdown_counts_turnovers_separately_from_passes():
    # ADR-20: a turnover must never inflate the Passes count.
    md = render_statcard_markdown(
        jersey_number=7,
        counts={"pass": 2, "turnover": 3},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
    )
    assert "**Passes:** 2" in md
    assert "**Turnovers:** 3" in md


def test_render_statcard_markdown_identity_status_defaults_to_verified():
    md = render_statcard_markdown(
        jersey_number=7,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
    )
    assert "**Identity Status:** Verified" in md


def test_render_statcard_markdown_identity_status_manual_annotation():
    # ADR-19: manual-events mode never claims a stronger "Verified" identity than it earned.
    md = render_statcard_markdown(
        jersey_number=2,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        identity_status="Human-provided (manual annotation)",
    )
    assert "**Identity Status:** Human-provided (manual annotation)" in md
    assert "**Identity Status:** Verified" not in md


# ---------------------------------------------------------------------------
# ADR-19/20: TURNOVER/OUT_OF_BOUNDS timeline rows, passes/turnovers highlight categories, and the
# _key_moment_label extension for a manual-annotation-sourced KEY_MOMENT
# ---------------------------------------------------------------------------


def test_timeline_includes_turnover_and_out_of_bounds_rows():
    events = [_event(EventType.TURNOVER, 1.0), _event(EventType.OUT_OF_BOUNDS, 2.0)]
    rows = build_event_timeline_rows(events)
    assert [r["label"] for r in rows] == ["Turnover", "Out of bounds"]


def test_highlight_categories_include_passes_and_turnovers():
    assert _HIGHLIGHT_CATEGORIES["passes"] == [EventType.PASS]
    assert _HIGHLIGHT_CATEGORIES["turnovers"] == [EventType.TURNOVER]


def test_render_statcard_markdown_uncertain_possession_and_distance_when_none():
    # ADR-19: manual mode has no possession heuristic to derive these from -- must say
    # "uncertain" (owner's own word, CLAUDE.md §13.2), never a fabricated 0.0.
    md = render_statcard_markdown(
        jersey_number=2,
        counts={},
        possession_seconds=None,
        distance_result=None,
        timeline_rows=[],
    )
    assert "**Possession Time:** uncertain" in md
    assert "**Distance Covered:** uncertain" in md


def test_key_moment_label_reads_manual_annotation_action_phrase():
    # ADR-19: a manual-annotation-sourced KEY_MOMENT carries evidence["action_phrase"], not
    # evidence["gemini_raw_response"] -- the SAME "celebrat" substring test must still apply.
    celebration = _event(
        EventType.KEY_MOMENT, 1.0, evidence={"action_phrase": "celebrates with the fans"}
    )
    other = _event(EventType.KEY_MOMENT, 2.0, evidence={"action_phrase": "argues with the ref"})
    assert _key_moment_label(celebration) == "Celebration"
    assert _key_moment_label(other) == "Other Key Moment"


def test_render_statcard_markdown_shows_real_goal_and_assist_counts_when_present():
    # ADR-17: real detector found something for THIS run -- must show the real number, not
    # "not available".
    md = render_statcard_markdown(
        jersey_number=10,
        counts={"goal": 1, "assist": 1},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        goal_reason="1 goal(s) detected via scoreboard-OCR delta (1 with an assist credited)",
    )
    assert "**Goals:** 1" in md
    assert "**Assists:** 1" in md
    assert "not available" not in md


def test_render_statcard_markdown_zero_goals_with_no_reason_shows_real_zero():
    # Bug fix 2026-08-31 (ADR-19 manual mode): `goal_reason=None` with a zero count means the
    # event source is authoritative (e.g. the client's own sidecar) and simply recorded no goal --
    # the honest answer is a real "0", never "not available".
    md = render_statcard_markdown(
        jersey_number=2,
        counts={"touch": 1},
        possession_seconds=None,
        distance_result=None,
        timeline_rows=[],
        goal_reason=None,
        identity_status="Human-provided (manual annotation)",
    )
    assert "**Goals:** 0" in md
    assert "**Assists:** 0" in md
    assert "not available" not in md


def test_render_statcard_markdown_not_available_carries_the_specific_reason():
    # ADR-17: the OLD behaviour was a bare, unconditional "not available" -- now it must carry
    # the specific, auditable reason from GoalDetectionResult.reason (Golden Rule 5).
    reason = (
        "not available (no legible scoreboard found by the region-activation scan -- CLAUDE.md "
        "ADR-17: ...)"
    )
    md = render_statcard_markdown(
        jersey_number=10,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
        goal_reason=reason,
    )
    assert f"**Goals:** {reason}" in md
    assert f"**Assists:** {reason}" in md


def test_render_statcard_markdown_never_pads_timeline_to_look_complete():
    md = render_statcard_markdown(
        jersey_number=9,
        counts={},
        possession_seconds=0.0,
        distance_result={"distance": 0.0, "unit": "bbox_heights"},
        timeline_rows=[],
    )
    assert "no events attributed" in md
    # must not contain a padded row for every possible event type in the TIMELINE section (the
    # summary stat lines above it legitimately always say "Touches:"/"Shots:" etc regardless)
    timeline_section = md.split("## Event Timeline", 1)[1]
    for label in ("Touch", "Pass", "Sprint", "Dribble", "Shot", "Tackle", "Celebration"):
        assert label not in timeline_section


# ---------------------------------------------------------------------------
# Plan Stage 3 (streamed-gathering-treehouse, 2026-09-14): write_player_output also writes
# statcard.pdf beside statcard.md (src/pipeline/statcard_pdf.py). No real video/GPU I/O below --
# empty `events`/`takes_by_id` make every highlight category empty, so `_cut_category_reel`
# returns before touching `cut_clips`/ffmpeg (see that function's own early-return), keeping this
# file's "pure-logic, no GPU/video I/O" scope intact (module docstring above).
# ---------------------------------------------------------------------------

pytest.importorskip("reportlab", reason="reportlab (api extra) not installed")


def test_write_player_output_writes_statcard_pdf_alongside_markdown(tmp_path):
    player_dir = tmp_path / "player_9"
    write_player_output(
        player_dir=player_dir,
        jersey_number=9,
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
    pdf_path = player_dir / "statcard.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-")
