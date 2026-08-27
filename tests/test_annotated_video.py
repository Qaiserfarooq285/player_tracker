"""Pure-logic unit tests for src/pipeline/annotated_video.py (ADR-15, CLAUDE.md §13.1) — no
GPU/video I/O."""

from __future__ import annotations

from src.common.types import Event, EventType
from src.identity.verify import TakeIdentityResult
from src.pipeline.annotated_video import _nearest_by_time, _NumberProgress, red_box_track_ids


def _event(etype: EventType, t: float) -> Event:
    return Event(
        id=f"{etype.value}-{t}",
        type=etype,
        t_start=t,
        t_end=t,
        player_track_id=1,
        take_id=0,
        confidence=0.5,
        source="test",
        evidence={},
    )


def test_number_progress_accumulates_forward_only():
    events = [
        _event(EventType.TOUCH, 1.0),
        _event(EventType.TOUCH, 3.0),
        _event(EventType.PASS, 5.0),
    ]
    progress = _NumberProgress(events)
    assert progress.advance_to(0.5) == {}
    assert progress.advance_to(2.0) == {"touch": 1}
    assert progress.advance_to(4.0) == {"touch": 2}
    assert progress.advance_to(10.0) == {"touch": 2, "pass": 1}


def test_number_progress_never_double_counts_on_repeated_calls_at_same_time():
    events = [_event(EventType.SHOT, 1.0)]
    progress = _NumberProgress(events)
    progress.advance_to(1.0)
    progress.advance_to(1.0)
    assert progress.counts == {"shot": 1}


def test_nearest_by_time_within_tolerance():
    items = [{"t": 0.0}, {"t": 1.0}, {"t": 2.0}]
    found = _nearest_by_time(items, 1.05, tolerance=0.2, key=lambda i: i["t"])
    assert found == {"t": 1.0}


def test_nearest_by_time_outside_tolerance_returns_none():
    items = [{"t": 0.0}, {"t": 5.0}]
    assert _nearest_by_time(items, 2.5, tolerance=0.5, key=lambda i: i["t"]) is None


def test_nearest_by_time_empty_list_returns_none():
    assert _nearest_by_time([], 1.0, tolerance=1.0, key=lambda i: i) is None


# ---------------------------------------------------------------------------
# red_box_track_ids -- the verified-identity render gate (CLAUDE.md §13.1)
# ---------------------------------------------------------------------------


def _identity(
    status: str, jersey_number: int | None, location_track_ids: list[int]
) -> TakeIdentityResult:
    return TakeIdentityResult(
        take_id=0,
        jersey_number=jersey_number,
        status=status,
        confidence=0.8,
        evidence_frames=[1, 2],
        location_method="arrow_vote",
        location_track_ids=location_track_ids,
    )


def test_red_box_track_ids_verified_take_returns_its_location_ids():
    identity = _identity("verified", 7, [10, 11, 12])
    assert red_box_track_ids(identity) == {10, 11, 12}


def test_red_box_track_ids_unverified_take_returns_empty_never_a_fallback():
    identity = _identity("unverified", None, [10, 11])
    assert red_box_track_ids(identity) == set()


def test_red_box_track_ids_none_identity_returns_empty():
    assert red_box_track_ids(None) == set()
