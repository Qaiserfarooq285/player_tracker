"""Pure-logic unit tests for `src/events/tackles.py` (ADR-13/14; no GPU/video decode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, DetectionClass, EventType, Track, TrackBox
from src.events import possession, tackles

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


def _player(tid: int, take_id: int, boxes: list[TrackBox], team=None, team_confidence=0.0) -> Track:
    return Track(
        id=tid,
        take_id=take_id,
        boxes=boxes,
        team=team,
        team_confidence=team_confidence,
        dominant_class=DetectionClass.PLAYER,
    )


# ---------------------------------------------------------------------------
# closing_speed
# ---------------------------------------------------------------------------


def test_closing_speed_positive_when_approaching_ball():
    # tackler at x=0 moving to x=50 while ball stays at x=100 -- gap shrinks 100 -> 50 over 1s
    track_b = _player(2, take_id=0, boxes=[_box(0.0, 0.0, 0.0), _box(1.0, 50.0, 0.0)])
    balls = [_ball(0.0, 100.0, 0.0), _ball(1.0, 100.0, 0.0)]
    speed = tackles.closing_speed(track_b, balls, t_before=0.0, t_at=1.0, match_tolerance_s=0.15)
    assert speed is not None
    assert speed > 0.0


def test_closing_speed_negative_when_moving_away():
    track_b = _player(2, take_id=0, boxes=[_box(0.0, 90.0, 0.0), _box(1.0, 0.0, 0.0)])
    balls = [_ball(0.0, 100.0, 0.0), _ball(1.0, 100.0, 0.0)]
    speed = tackles.closing_speed(track_b, balls, t_before=0.0, t_at=1.0, match_tolerance_s=0.15)
    assert speed is not None
    assert speed < 0.0


def test_closing_speed_none_when_t_at_not_after_t_before():
    track_b = _player(2, take_id=0, boxes=[_box(0.0, 0.0, 0.0)])
    balls = [_ball(0.0, 100.0, 0.0)]
    speed = tackles.closing_speed(track_b, balls, t_before=1.0, t_at=1.0, match_tolerance_s=0.15)
    assert speed is None


def test_closing_speed_none_when_no_box_in_tolerance():
    track_b = _player(2, take_id=0, boxes=[_box(10.0, 0.0, 0.0)])
    balls = [_ball(0.0, 100.0, 0.0), _ball(1.0, 100.0, 0.0)]
    speed = tackles.closing_speed(track_b, balls, t_before=0.0, t_at=1.0, match_tolerance_s=0.15)
    assert speed is None


# ---------------------------------------------------------------------------
# detect_tackles -- should fire / should not fire / take-crossing safety
# ---------------------------------------------------------------------------


def _closing_run_pair(cfg, gap_s=0.2, far_x=-400.0):
    """A has the ball at t=0; B starts far away (`far_x`) and is at the ball's own position by
    `gap_s` later -- the canonical "should fire" tackle setup (comfortably clears
    `tackle_min_closing_speed` regardless of `gap_s`, as long as `gap_s` stays under
    `tackle_lookback_s` so the closing-speed lookback window covers B's whole run-in)."""
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b_boxes = [_box(0.0, far_x, 100.0, height=100.0), _box(gap_s, 100.0, 100.0, height=100.0)]
    b = _player(2, take_id=0, boxes=b_boxes, team=1, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(gap_s, 100.0, 100.0)]
    return a, b, balls


def test_detect_tackles_fires_for_fast_closing_opponent_possession_change():
    cfg = _events_config()
    a, b, balls = _closing_run_pair(cfg, gap_s=0.2)
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = tackles.detect_tackles(runs, [a, b], balls, take_id=0, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.TACKLE
    assert ev.player_track_id == 2  # tackler credited, per this module's own documented decision
    assert ev.evidence["tackler_identity"] == 2
    assert ev.evidence["tackled_identity"] == 1
    assert ev.evidence["calibrated"] is False
    assert 0.0 <= ev.confidence <= cfg["tackle"]["max_confidence"]


def test_detect_tackles_no_event_when_same_team():
    cfg = _events_config()
    a, b, balls = _closing_run_pair(cfg, gap_s=0.2)
    b.team = 0  # same team as a -- a genuine fast closedown, but not an "opponent" tackle
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = tackles.detect_tackles(runs, [a, b], balls, take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_tackles_degrades_gracefully_on_low_team_confidence():
    """ADR-12: poor team_confidence must suppress tackle emission entirely, never default to
    assuming opposition (which would fabricate tackles out of noisy team labels)."""
    cfg = _events_config()
    a, b, balls = _closing_run_pair(cfg, gap_s=0.2)
    a.team_confidence = 0.1
    b.team_confidence = 0.1
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = tackles.detect_tackles(runs, [a, b], balls, take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_tackles_no_event_when_gap_exceeds_max_duration():
    cfg = dict(_events_config())
    cfg["tackle"] = dict(cfg["tackle"])
    cfg["tackle"]["tackle_max_duration_s"] = 0.05
    a, b, balls = _closing_run_pair(cfg, gap_s=1.0)
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = tackles.detect_tackles(runs, [a, b], balls, take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_tackles_no_event_when_not_closing_fast_enough():
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    # B is already sitting right next to the ball the whole time -- no real closing speed
    b = _player(
        2,
        take_id=0,
        boxes=[_box(0.0, 100.0, 100.0, height=100.0), _box(0.2, 100.0, 100.0, height=100.0)],
        team=1,
        team_confidence=0.9,
    )
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.2, 100.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = tackles.detect_tackles(runs, [a, b], balls, take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_tackles_never_crosses_take_boundary():
    """The mandatory take-crossing rejection test: `detect_tackles` (like
    `possession_runs_for_take`/`detect_passes`/`detect_dribbles`) asserts its own `tracks`
    argument never spans more than one `take_id` -- raw `Track.id`s reset per take (Golden Rule
    3), so a mixed-take `tracks` list risks two different takes' fragments colliding on the same
    integer id and being treated as one person's possession change. Even a
    geometrically/temporally perfect "should fire" tackle setup (see `_closing_run_pair`) must be
    rejected outright, never silently evaluated, when the two possessors are stamped with
    different take_ids."""
    cfg = _events_config()
    a, b, balls = _closing_run_pair(cfg, gap_s=0.2)
    b.take_id = 1  # a DIFFERENT take -- must never be mixed with `a`'s take 0

    with pytest.raises(AssertionError):
        tackles.detect_tackles([], [a, b], balls, take_id=0, events_cfg=cfg)


def test_build_take_identities_also_rejects_this_exact_tackle_setup_mixed_across_takes():
    """Cross-check against `src/track/continuity.py`'s own guard (the identity-building step a
    real orchestrator would run before `detect_tackles`) using the EXACT same fixture, so the two
    layers of defence are shown to agree, not just `detect_tackles`'s own local assertion."""
    from src.track.continuity import build_take_identities

    cfg = _events_config()
    a, b, _balls = _closing_run_pair(cfg, gap_s=0.2)
    b.take_id = 1
    highlights_cfg = load_yaml(REPO_ROOT / "configs" / "highlights.yaml")
    with pytest.raises(AssertionError):
        build_take_identities([a, b], highlights_cfg["selection"])
