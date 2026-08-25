"""Pure-logic unit tests for Stage 6's stat card (CLAUDE.md task spec: no GPU/video decode).

Covers: count/confidence aggregation, the ADR-6 uncalibrated speed/distance re-derivation, and
that `write_stat_card` produces well-formed, internally-consistent JSON/Markdown with real
click-to-clip paths and the required "not available"/"UNCALIBRATED" wording (Golden Rule 5).
"""

from __future__ import annotations

from pathlib import Path

from src.common.types import Clip, Event, EventType, PlayerStats
from src.stats.stats import (
    build_player_stats,
    peak_sprint_speed,
    total_sprint_distance,
    write_stat_card,
)


def _sprint(
    eid: str, mean_speed: float, peak_speed: float, t0: float, t1: float, confidence: float
) -> Event:
    return Event(
        id=eid,
        type=EventType.SPRINT,
        t_start=t0,
        t_end=t1,
        player_track_id=1,
        take_id=0,
        confidence=confidence,
        source="speed_heuristic_normalized_pixel",
        evidence={
            "mean_speed": mean_speed,
            "peak_speed": peak_speed,
            "calibrated": False,
            "unit": "bbox_heights_per_second",
            "t_range": [t0, t1],
        },
    )


def _shot(eid: str, confidence: float) -> Event:
    return Event(
        id=eid,
        type=EventType.SHOT,
        t_start=10.0,
        t_end=10.5,
        player_track_id=None,
        take_id=0,
        confidence=confidence,
        source="heuristic_ball_motion",
        evidence={"calibrated": False, "peak_speed": 0.5, "unit": "frame_widths_per_second"},
    )


# ---------------------------------------------------------------------------
# build_player_stats
# ---------------------------------------------------------------------------


def test_build_player_stats_counts_and_mean_confidence_per_type():
    events = [
        _sprint("s1", mean_speed=5.0, peak_speed=6.0, t0=0.0, t1=1.0, confidence=0.4),
        _sprint("s2", mean_speed=7.0, peak_speed=8.0, t0=2.0, t1=3.0, confidence=0.6),
        _shot("h1", confidence=0.3),
    ]
    stats = build_player_stats("clip2_77#target", [1], events)
    assert stats.counts == {"sprint": 2, "shot": 1}
    assert stats.confidences["sprint"] == (0.4 + 0.6) / 2
    assert stats.confidences["shot"] == 0.3
    assert set(stats.events) == {"s1", "s2", "h1"}
    assert stats.track_ids == [1]


def test_build_player_stats_empty_events():
    stats = build_player_stats("p", [], [])
    assert stats.counts == {}
    assert stats.confidences == {}
    assert stats.events == []


# ---------------------------------------------------------------------------
# total_sprint_distance / peak_sprint_speed -- ADR-6 re-derivation, not new numbers
# ---------------------------------------------------------------------------


def test_total_sprint_distance_sums_mean_speed_times_duration():
    events = [
        _sprint("s1", mean_speed=5.0, peak_speed=6.0, t0=0.0, t1=1.0, confidence=0.4),  # 5.0 * 1.0
        _sprint("s2", mean_speed=4.0, peak_speed=8.0, t0=2.0, t1=4.0, confidence=0.6),  # 4.0 * 2.0
    ]
    assert total_sprint_distance(events) == 5.0 * 1.0 + 4.0 * 2.0


def test_total_sprint_distance_ignores_non_sprint_events():
    events = [_shot("h1", confidence=0.3)]
    assert total_sprint_distance(events) == 0.0


def test_peak_sprint_speed_is_the_max_across_events():
    events = [
        _sprint("s1", mean_speed=5.0, peak_speed=6.0, t0=0.0, t1=1.0, confidence=0.4),
        _sprint("s2", mean_speed=4.0, peak_speed=9.5, t0=2.0, t1=4.0, confidence=0.6),
    ]
    assert peak_sprint_speed(events) == 9.5


def test_peak_sprint_speed_zero_when_no_sprints():
    assert peak_sprint_speed([_shot("h1", confidence=0.3)]) == 0.0


# ---------------------------------------------------------------------------
# write_stat_card -- well-formed output, real click-to-clip paths, required wording
# ---------------------------------------------------------------------------


def test_write_stat_card_is_well_formed_and_internally_consistent(tmp_path: Path):
    events = [
        _sprint("s1", mean_speed=5.0, peak_speed=6.0, t0=0.0, t1=1.0, confidence=0.4),
        _shot("h1", confidence=0.3),
    ]
    stats = build_player_stats("clip2_77#jersey_77", [16], events)
    clip = Clip(
        event_id="s1",
        t_start=0.0,
        t_end=1.0,
        take_id=0,
        rank_score=20.0,
        path=Path("output/clip2_77/clips/clip_000.mp4"),
    )
    clips_by_event_id = {"s1": clip}  # h1 was ranked out / never cut -- deliberately absent
    goal_reason = "not available (no scoreboard detected in this footage)"
    selection_summary = {
        "method": "arrow_vote",
        "confidence": 0.62,
        "needs_human_confirmation": True,
    }

    import json

    json_path, md_path = write_stat_card(
        stats,
        events,
        clips_by_event_id,
        tmp_path,
        target_jersey=77,
        selection_summary=selection_summary,
        goal_reason=goal_reason,
    )
    assert json_path.exists() and json_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0

    card = json.loads(json_path.read_text())
    # internal consistency: counts match what's actually listed in events[]
    assert sum(card["counts"].values()) == len(card["events"])
    for event_type, count in card["counts"].items():
        assert count == sum(1 for e in card["events"] if e["type"] == event_type)
    # every confidence is in [0, 1]
    for c in card["mean_confidence_by_type"].values():
        assert 0.0 <= c <= 1.0
    for e in card["events"]:
        assert 0.0 <= e["confidence"] <= 1.0

    # click-to-clip: the cut event has a real path; the uncut one is explicitly None (not
    # decorative)
    by_id = {e["event_id"]: e for e in card["events"]}
    assert by_id["s1"]["clip_path"] == str(clip.path)
    assert by_id["h1"]["clip_path"] is None

    # Golden Rule 5 wording requirements
    assert card["goals"].startswith("not available")
    assert card["speed"]["calibrated"] is False
    assert "UNCALIBRATED" in card["speed"]["note"]
    assert card["target_jersey"] == 77

    md_text = md_path.read_text()
    assert "UNCALIBRATED" in md_text
    assert "not available" in md_text
    assert str(clip.path) in md_text


def test_write_stat_card_handles_zero_events(tmp_path: Path):
    stats = PlayerStats(player_ref="p", track_ids=[], counts={}, confidences={}, events=[])
    json_path, md_path = write_stat_card(
        stats,
        [],
        {},
        tmp_path,
        target_jersey=None,
        selection_summary={"method": "none", "confidence": 0.0, "needs_human_confirmation": True},
        goal_reason="not available (no scoreboard detected in this footage)",
    )
    import json

    card = json.loads(json_path.read_text())
    assert card["counts"] == {}
    assert card["events"] == []
    assert card["target_jersey"] is None
