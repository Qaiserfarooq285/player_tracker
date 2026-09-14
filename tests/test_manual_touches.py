"""Pure-logic unit tests for `src/events/manual_touches.py` (Plan Stage 2, "streamed-gathering-
treehouse", 2026-09-14) -- no I/O, no GPU."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.common.io import load_yaml
from src.common.types import Event, EventType
from src.events.manual_touches import manual_touch_events, merge_manual_touches, parse_touch_times

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")["manual_touch"]


def _event(event_type: EventType, t_start: float, source: str = "possession_heuristic") -> Event:
    return Event(
        id=str(uuid.uuid4()),
        type=event_type,
        t_start=t_start,
        t_end=t_start + 0.5,
        player_track_id=7,
        take_id=0,
        confidence=0.4,
        source=source,
        evidence={},
    )


# ---------------------------------------------------------------------------------------------
# parse_touch_times
# ---------------------------------------------------------------------------------------------


def test_parse_comma_separated_mm_ss():
    times, problems = parse_touch_times("12:34, 45:10", _cfg())
    assert problems == []
    assert times == [12 * 60 + 34.0, 45 * 60 + 10.0]


def test_parse_newline_separated():
    times, problems = parse_touch_times("1:02:03\n26:56", _cfg())
    assert problems == []
    assert times == [3723.0, 1616.0]


def test_parse_mixed_comma_and_newline():
    times, problems = parse_touch_times("1:00,\n2:00\n3:00, 4:00", _cfg())
    assert problems == []
    assert times == [60.0, 120.0, 180.0, 240.0]


def test_unparseable_entry_lands_in_problems_valid_ones_still_parse():
    times, problems = parse_touch_times("12:34, not-a-time, 45:10", _cfg())
    assert times == [12 * 60 + 34.0, 45 * 60 + 10.0]
    assert problems == ["not-a-time"]


def test_blank_and_whitespace_only_entries_are_skipped_not_problems():
    times, problems = parse_touch_times("12:34,, \n \n45:10", _cfg())
    assert problems == []
    assert times == [12 * 60 + 34.0, 45 * 60 + 10.0]


def test_empty_raw_string_yields_no_times_no_problems():
    times, problems = parse_touch_times("", _cfg())
    assert times == []
    assert problems == []


def test_entirely_malformed_string_is_one_problem():
    times, problems = parse_touch_times("garbage", _cfg())
    assert times == []
    assert problems == ["garbage"]


# ---------------------------------------------------------------------------------------------
# manual_touch_events
# ---------------------------------------------------------------------------------------------


def test_manual_touch_events_fields():
    cfg = _cfg()
    events = manual_touch_events([100.0, 200.0], cfg)
    assert len(events) == 2
    for ev, t in zip(events, [100.0, 200.0], strict=True):
        assert ev.type == EventType.TOUCH
        assert ev.source == "manual_touch"
        assert ev.player_track_id is None
        assert ev.take_id is None
        assert ev.t_start == t
        assert ev.t_end == t + cfg["event_window_seconds"]
        assert ev.confidence == cfg["confidence"]
        assert ev.evidence["raw_time"] == t
        assert ev.evidence["provenance"] == "user_supplied"


def test_manual_touch_events_empty_times_yields_empty_list():
    assert manual_touch_events([], _cfg()) == []


# ---------------------------------------------------------------------------------------------
# merge_manual_touches
# ---------------------------------------------------------------------------------------------


def test_merge_manual_touch_always_kept():
    cfg = _cfg()
    manual = manual_touch_events([100.0], cfg)
    merged, n_suppressed = merge_manual_touches([], manual, cfg)
    assert merged == manual
    assert n_suppressed == 0


def test_merge_auto_touch_within_window_suppressed_and_counted():
    cfg = _cfg()
    manual = manual_touch_events([100.0], cfg)
    auto_touch_close = _event(EventType.TOUCH, 100.0 + cfg["merge_window_s"] - 0.01)
    merged, n_suppressed = merge_manual_touches([auto_touch_close], manual, cfg)
    assert n_suppressed == 1
    assert auto_touch_close not in merged
    assert manual[0] in merged
    assert len(merged) == 1


def test_merge_auto_touch_outside_window_kept():
    cfg = _cfg()
    manual = manual_touch_events([100.0], cfg)
    auto_touch_far = _event(EventType.TOUCH, 100.0 + cfg["merge_window_s"] + 5.0)
    merged, n_suppressed = merge_manual_touches([auto_touch_far], manual, cfg)
    assert n_suppressed == 0
    assert auto_touch_far in merged
    assert manual[0] in merged
    assert len(merged) == 2


def test_merge_non_touch_events_always_untouched():
    cfg = _cfg()
    manual = manual_touch_events([100.0], cfg)
    sprint_close = _event(EventType.SPRINT, 100.0)
    pass_close = _event(EventType.PASS, 100.0)
    merged, n_suppressed = merge_manual_touches([sprint_close, pass_close], manual, cfg)
    assert n_suppressed == 0
    assert sprint_close in merged
    assert pass_close in merged
    assert manual[0] in merged
    assert len(merged) == 3


def test_merge_exact_boundary_is_inclusive():
    cfg = _cfg()
    manual = manual_touch_events([100.0], cfg)
    auto_touch_at_boundary = _event(EventType.TOUCH, 100.0 + cfg["merge_window_s"])
    merged, n_suppressed = merge_manual_touches([auto_touch_at_boundary], manual, cfg)
    assert n_suppressed == 1
    assert len(merged) == 1
