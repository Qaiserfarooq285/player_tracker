"""Pure-logic unit tests for `scripts/overlay_target.py` (CLAUDE.md task spec: no video I/O).

Covers: live-counter increment logic (`running_counts_at`), which event type is currently
"active"/flashing at a given timestamp (`active_event_types_at`), its within-window progress
fraction (`active_event_progress`), and which `Take` a timestamp falls in (`find_active_take`).
Touches/passes/tackles/assists/saves are never counted (Golden Rule 7) -- verified explicitly.

Importing `scripts.overlay_target` here must NOT pull in torch/rfdetr: the module defers that
import to inside `render_overlay` (see its own docstring) precisely so this test file stays a
pure-logic, no-GPU test like `tests/test_detect.py`/`tests/test_track.py`.
"""

from __future__ import annotations

from scripts.overlay_target import (
    active_event_progress,
    active_event_types_at,
    find_active_take,
    running_counts_at,
)
from src.common.types import Take

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sprint_row(t0: float, t1: float, confidence: float = 0.4) -> dict:
    return {"t_start": t0, "t_end": t1, "type": "sprint", "confidence": confidence}


def _shot_row(t0: float, t1: float, confidence: float = 0.3) -> dict:
    return {"t_start": t0, "t_end": t1, "type": "shot", "confidence": confidence}


TIMELINE = [
    _sprint_row(1.0, 3.0),
    _shot_row(5.0, 5.5),
    _sprint_row(8.0, 10.0),
]

TAKES = [
    Take(id=0, t_start=0.0, t_end=60.0, frame_start=0, frame_end=1800),
    Take(id=1, t_start=60.0, t_end=90.0, frame_start=1800, frame_end=2700),
    Take(id=2, t_start=90.0, t_end=110.0, frame_start=2700, frame_end=3300),
]


# ---------------------------------------------------------------------------
# running_counts_at
# ---------------------------------------------------------------------------


def test_running_counts_zero_before_any_event_starts():
    counts = running_counts_at(0.0, TIMELINE)
    assert counts == {"sprint": 0, "shot": 0}


def test_running_counts_increments_as_events_start():
    # t=1.0: first sprint has just started
    assert running_counts_at(1.0, TIMELINE) == {"sprint": 1, "shot": 0}
    # t=4.0: first sprint over, shot not started yet
    assert running_counts_at(4.0, TIMELINE) == {"sprint": 1, "shot": 0}
    # t=5.0: shot has started
    assert running_counts_at(5.0, TIMELINE) == {"sprint": 1, "shot": 1}
    # t=9.0: second sprint has started (counted even mid-event, since t_start already passed)
    assert running_counts_at(9.0, TIMELINE) == {"sprint": 2, "shot": 1}


def test_running_counts_is_monotonically_non_decreasing_over_time():
    sample_times = [0.0, 1.5, 4.0, 5.2, 6.0, 8.5, 20.0]
    prev = running_counts_at(sample_times[0], TIMELINE)
    for t in sample_times[1:]:
        cur = running_counts_at(t, TIMELINE)
        assert cur["sprint"] >= prev["sprint"]
        assert cur["shot"] >= prev["shot"]
        prev = cur


def test_running_counts_never_counts_out_of_scope_categories():
    """Golden Rule 7: touches/passes/tackles/assists/saves must never appear as a counted key,
    even if such a row somehow ended up in the timeline (e.g. a stray Phase-2 event type)."""
    timeline_with_stray_type = TIMELINE + [
        {"t_start": 0.0, "t_end": 100.0, "type": "touch", "confidence": 0.9}
    ]
    counts = running_counts_at(50.0, timeline_with_stray_type)
    assert set(counts.keys()) == {"sprint", "shot"}
    assert "touch" not in counts


def test_running_counts_empty_timeline():
    assert running_counts_at(42.0, []) == {"sprint": 0, "shot": 0}


# ---------------------------------------------------------------------------
# active_event_types_at
# ---------------------------------------------------------------------------


def test_active_event_types_empty_between_events():
    assert active_event_types_at(0.5, TIMELINE) == set()
    assert active_event_types_at(4.0, TIMELINE) == set()


def test_active_event_types_sprint_window():
    assert active_event_types_at(2.0, TIMELINE) == {"sprint"}
    assert active_event_types_at(1.0, TIMELINE) == {"sprint"}  # inclusive at t_start
    assert active_event_types_at(3.0, TIMELINE) == {"sprint"}  # inclusive at t_end


def test_active_event_types_shot_window():
    assert active_event_types_at(5.2, TIMELINE) == {"shot"}


def test_active_event_types_can_overlap_multiple_types():
    overlapping = [_sprint_row(1.0, 4.0), _shot_row(2.0, 3.0)]
    assert active_event_types_at(2.5, overlapping) == {"sprint", "shot"}


# ---------------------------------------------------------------------------
# active_event_progress
# ---------------------------------------------------------------------------


def test_active_event_progress_halfway_through_window():
    progress = active_event_progress(2.0, TIMELINE)  # sprint window [1.0, 3.0]
    assert progress["sprint"] == 0.5


def test_active_event_progress_at_window_edges():
    assert active_event_progress(1.0, TIMELINE)["sprint"] == 0.0
    assert active_event_progress(3.0, TIMELINE)["sprint"] == 1.0


def test_active_event_progress_empty_when_nothing_active():
    assert active_event_progress(4.0, TIMELINE) == {}


def test_active_event_progress_zero_duration_window_reads_as_complete():
    zero_duration = [_shot_row(5.0, 5.0)]
    assert active_event_progress(5.0, zero_duration)["shot"] == 1.0


# ---------------------------------------------------------------------------
# find_active_take
# ---------------------------------------------------------------------------


def test_find_active_take_returns_correct_take_for_each_take_id():
    assert find_active_take(0.0, TAKES).id == 0
    assert find_active_take(59.9, TAKES).id == 0
    assert find_active_take(60.0, TAKES).id == 1
    assert find_active_take(89.9, TAKES).id == 1
    assert find_active_take(90.0, TAKES).id == 2


def test_find_active_take_at_the_very_last_take_end_is_still_covered():
    assert find_active_take(110.0, TAKES).id == 2  # closed-interval fallback, see assign_take_id


def test_find_active_take_returns_none_outside_every_take():
    assert find_active_take(-1.0, TAKES) is None
    assert find_active_take(200.0, TAKES) is None


def test_find_active_take_empty_takes_list():
    assert find_active_take(0.0, []) is None
