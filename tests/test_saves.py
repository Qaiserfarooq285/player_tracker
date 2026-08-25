"""Pure-logic unit tests for `src/events/saves.py` (ADR-13/14; no GPU/video decode).

⚠️ There is no real footage to validate this heuristic against (neither `work/clip2_77` nor
`work/clip4_77` contains a GOALKEEPER-class track -- see the module docstring); these are entirely
synthetic. The "should not fire without a goalkeeper" case is arguably the MOST important test
here, since it is the one behaviour this module is actually guaranteed to exhibit on the project's
current real input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, DetectionClass, EventType, Track, TrackBox
from src.events import saves

REPO_ROOT = Path(__file__).resolve().parents[1]


def _events_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")


def _box(t: float, cx: float, cy: float, height: float = 100.0, conf: float = 0.8) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 30)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=conf,
    )


def _ball(t: float, cx: float, cy: float, conf: float = 0.9) -> BallDetection:
    return BallDetection(
        bbox=BBox(x1=cx - 5.0, y1=cy - 5.0, x2=cx + 5.0, y2=cy + 5.0),
        conf=conf,
        frame_index=int(round(t * 30)),
        t=t,
    )


def _goalkeeper(tid: int, take_id: int, boxes: list[TrackBox]) -> Track:
    return Track(id=tid, take_id=take_id, boxes=boxes, dominant_class=DetectionClass.GOALKEEPER)


FRAME_WIDTH = 1920.0

# Shared shape constants for the synthetic "fast approach" ball sequences below.
_APPROACH_DT = 0.1
_APPROACH_N = 6
_APPROACH_MARGIN = 8.0  # see _fast_approach_then_stop_balls's docstring for why this needs to be
# large rather than close to 1.0.


def _approach_step(threshold: float) -> float:
    return threshold * FRAME_WIDTH * _APPROACH_DT * _APPROACH_MARGIN


def _approach_end_x(threshold: float) -> float:
    """The x-coordinate the ball holds during/after the stop in `_fast_approach_then_stop_balls`
    -- where a real goalkeeper would have to be standing for the "near the goalkeeper"
    precondition to hold."""
    return 1000.0 + _approach_step(threshold) * (_APPROACH_N - 1)


def _fast_approach_then_stop_balls(threshold: float) -> list[BallDetection]:
    """A ball approaching fast (well over `threshold` frame-widths/s) toward `_approach_end_x`,
    then sharply DECELERATING TO A STOP right after -- the canonical "should fire" save shape.

    ⚠️ Deliberately uses a LARGE margin (`_APPROACH_MARGIN`=8x `threshold`) rather than something
    closer to it: the underlying `src/events/shots.py::ball_speed_series` smoothing
    (`smoothing_window_frames`) means the 2-3 samples right after a sharp stop still read a fading
    FRACTION of the prior speed (a real trailing-moving-average artifact, not a test bug) -- too
    small a margin leaves that fading tail still above `save_deceleration_ratio *
    peak_speed_before`. An 8x margin pushes that whole fading tail back INTO the fast-approach
    segment itself (`find_threshold_segments` keeps swallowing samples that are still `>=
    threshold`), so the samples actually classified as "after the approach" are the FULLY SETTLED
    (genuinely zero-speed) ones -- verified numerically while writing this test, not asserted on
    faith.
    """
    step = _approach_step(threshold)
    xs = [1000.0 + step * i for i in range(_APPROACH_N)]
    n_stop = 6
    xs += [xs[-1]] * n_stop
    times = [i * _APPROACH_DT for i in range(_APPROACH_N + n_stop)]
    return [_ball(t, x, 500.0) for t, x in zip(times, xs, strict=True)]


def _approach_only_balls(threshold: float) -> list[BallDetection]:
    """Just the fast-approach half of `_fast_approach_then_stop_balls`, with NO samples following
    it at all -- for exercising the "no reaction window" rejection specifically."""
    step = _approach_step(threshold)
    xs = [1000.0 + step * i for i in range(_APPROACH_N)]
    return [_ball(i * _APPROACH_DT, xs[i], 500.0) for i in range(_APPROACH_N)]


# ---------------------------------------------------------------------------
# goalkeeper_tracks / the "no goalkeeper -> zero events" guarantee
# ---------------------------------------------------------------------------


def test_goalkeeper_tracks_filters_to_goalkeeper_class_only():
    gk = _goalkeeper(1, take_id=0, boxes=[_box(0.0, 1000.0, 500.0)])
    player = Track(
        id=2, take_id=0, boxes=[_box(0.0, 0.0, 0.0)], dominant_class=DetectionClass.PLAYER
    )
    assert saves.goalkeeper_tracks([gk, player]) == [gk]


def test_detect_saves_produces_nothing_without_any_goalkeeper_track():
    """The single most important behaviour on this project's ACTUAL footage: neither clip2 nor
    clip4 has a goalkeeper track at all, so this must produce zero events, never a forced/guessed
    one (CLAUDE.md task spec, Golden Rule 5) -- not an error, not a fallback guess."""
    cfg = _events_config()
    player = Track(
        id=1, take_id=0, boxes=[_box(0.0, 1000.0, 500.0)], dominant_class=DetectionClass.PLAYER
    )
    balls = _fast_approach_then_stop_balls(cfg["save"]["ball_speed_threshold"])
    events = saves.detect_saves(balls, [player], take_id=0, frame_width=FRAME_WIDTH, events_cfg=cfg)
    assert events == []


def test_detect_saves_rejects_mixed_take_ids():
    cfg = _events_config()
    gk_take0 = _goalkeeper(1, take_id=0, boxes=[_box(0.0, 1000.0, 500.0)])
    gk_take1 = _goalkeeper(2, take_id=1, boxes=[_box(0.0, 1000.0, 500.0)])
    balls = _fast_approach_then_stop_balls(cfg["save"]["ball_speed_threshold"])
    with pytest.raises(AssertionError):
        saves.detect_saves(
            balls, [gk_take0, gk_take1], take_id=0, frame_width=FRAME_WIDTH, events_cfg=cfg
        )


# ---------------------------------------------------------------------------
# _nearest_goalkeeper_within
# ---------------------------------------------------------------------------


def test_nearest_goalkeeper_within_finds_close_goalkeeper():
    gk = _goalkeeper(1, take_id=0, boxes=[_box(0.0, 1000.0, 500.0)])
    found = saves._nearest_goalkeeper_within(
        1000.0, 500.0, 0.0, [gk], proximity_px=50.0, match_tolerance_s=0.15
    )
    assert found is gk


def test_nearest_goalkeeper_within_none_when_too_far():
    gk = _goalkeeper(1, take_id=0, boxes=[_box(0.0, 1000.0, 500.0)])
    found = saves._nearest_goalkeeper_within(
        1000.0 + 10_000.0, 500.0, 0.0, [gk], proximity_px=50.0, match_tolerance_s=0.15
    )
    assert found is None


# ---------------------------------------------------------------------------
# save_confidence -- bounded correctly
# ---------------------------------------------------------------------------


def test_save_confidence_never_exceeds_max_confidence():
    cfg = _events_config()["save"]
    confidence = saves.save_confidence(
        peak_speed_before=100.0, peak_speed_after=0.0, reversed_direction=True, save_cfg=cfg
    )
    assert confidence == pytest.approx(cfg["max_confidence"])


def test_save_confidence_floors_at_min_confidence_when_no_signal():
    cfg = _events_config()["save"]
    # no deceleration (after == before) and no reversal -- weakest possible signal
    confidence = saves.save_confidence(
        peak_speed_before=1.0, peak_speed_after=1.0, reversed_direction=False, save_cfg=cfg
    )
    assert confidence == pytest.approx(cfg["min_confidence"])


# ---------------------------------------------------------------------------
# detect_saves -- should fire / should not fire, with a real goalkeeper present
# ---------------------------------------------------------------------------


def _dense_goalkeeper(tid: int, take_id: int, x: float, y: float, height: float = 150.0) -> Track:
    """A goalkeeper with a box at every 0.1s from 0.0 to 1.1s (matching
    `_fast_approach_then_stop_balls`'s own sample spacing/range) -- dense enough that
    `nearest_box_in_time`'s default tolerance always finds one, regardless of exactly which sample
    a given test's fast-approach segment happens to end on."""
    boxes = [_box(i * _APPROACH_DT, x, y, height=height) for i in range(_APPROACH_N + 6)]
    return _goalkeeper(tid, take_id, boxes)


def test_detect_saves_fires_for_fast_approach_and_stop_near_goalkeeper():
    cfg = _events_config()
    threshold = cfg["save"]["ball_speed_threshold"]
    gk = _dense_goalkeeper(1, take_id=5, x=_approach_end_x(threshold), y=500.0)
    balls = _fast_approach_then_stop_balls(threshold)
    events = saves.detect_saves(balls, [gk], take_id=5, frame_width=FRAME_WIDTH, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.SAVE
    assert ev.player_track_id == 1
    assert ev.take_id == 5
    assert ev.evidence["calibrated"] is False
    assert ev.evidence["decelerated"] is True
    assert 0.0 <= ev.confidence <= cfg["save"]["max_confidence"]


def test_detect_saves_does_not_fire_when_ball_never_approaches_fast():
    cfg = _events_config()
    gk = _dense_goalkeeper(1, take_id=0, x=1000.0, y=500.0)
    # ball barely moves at all -- never clears ball_speed_threshold
    balls = [_ball(i * 0.1, 1000.0 + i * 0.01, 500.0) for i in range(10)]
    events = saves.detect_saves(balls, [gk], take_id=0, frame_width=FRAME_WIDTH, events_cfg=cfg)
    assert events == []


def test_detect_saves_does_not_fire_when_fast_approach_is_far_from_goalkeeper():
    cfg = _events_config()
    threshold = cfg["save"]["ball_speed_threshold"]
    # goalkeeper sits far away from where the fast approach/stop actually happens
    gk = _dense_goalkeeper(1, take_id=0, x=_approach_end_x(threshold) - 5000.0, y=500.0)
    balls = _fast_approach_then_stop_balls(threshold)
    events = saves.detect_saves(balls, [gk], take_id=0, frame_width=FRAME_WIDTH, events_cfg=cfg)
    assert events == []


def test_detect_saves_does_not_fire_without_reaction_window():
    cfg = _events_config()
    threshold = cfg["save"]["ball_speed_threshold"]
    gk = _dense_goalkeeper(1, take_id=0, x=_approach_end_x(threshold), y=500.0)
    # fast approach with NO samples after it at all -- no reaction window to evaluate
    balls = _approach_only_balls(threshold)
    events = saves.detect_saves(balls, [gk], take_id=0, frame_width=FRAME_WIDTH, events_cfg=cfg)
    assert events == []
