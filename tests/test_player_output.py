"""Pure-logic unit tests for src/pipeline/player_output.py (ADR-15, CLAUDE.md §13.2) — no
GPU/video I/O."""

from __future__ import annotations

from src.common.types import Event, EventType
from src.pipeline.player_output import (
    build_event_timeline_rows,
    render_statcard_markdown,
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


def test_timeline_excludes_save_and_possession():
    events = [
        _event(EventType.SAVE, 1.0),
        _event(EventType.POSSESSION, 2.0),
        _event(EventType.TOUCH, 3.0),
    ]
    rows = build_event_timeline_rows(events)
    assert [r["label"] for r in rows] == ["Touch"]


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
    assert "**Sprints/Runs:** 0" in md  # never fabricated -- genuinely zero, reported as such
    assert "**Goals:** not available" in md
    assert "**Assists:** not available" in md
    assert "**Possession Time:** 12.5s" in md
    assert "(uncalibrated)" in md
    assert "## Event Timeline" in md


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
