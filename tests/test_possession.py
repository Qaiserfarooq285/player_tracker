"""Pure-logic unit tests for `src/events/possession.py` (ADR-13/14; no GPU/video decode).

Covers possession-run building/merging (incl. the take-crossing rejection test), possession event
emission, pass detection (incl. the ADR-12 team-confidence gate degrading gracefully rather than
crashing/assuming team=0), dribble detection, and distance-covered integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, DetectionClass, EventType, Track, TrackBox
from src.events import possession

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
# possession_runs_for_take / _merge_possession_runs
# ---------------------------------------------------------------------------


def test_possession_runs_never_mix_tracks_from_different_takes():
    """The mandatory take-crossing rejection test for possession: raw `Track.id`s reset per take
    (Golden Rule 3), so `possession_runs_for_take` must reject a `tracks` list spanning more than
    one `take_id` outright, never silently treat a different take's fragment as a possessor --
    even when it is geometrically/temporally identical to a real same-take track."""
    cfg = _events_config()
    same_take = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    other_take = _player(2, take_id=1, boxes=[_box(0.0, 100.0, 100.0)])  # identical position/time
    balls = [_ball(0.0, 100.0, 100.0)]

    with pytest.raises(AssertionError):
        possession.possession_runs_for_take(balls, [same_take, other_take], cfg)

    # the CORRECT usage -- caller passes ONLY take 0's own tracks -- works fine and never
    # attributes anything to track id 2 (which doesn't even appear in the input).
    runs = possession.possession_runs_for_take(balls, [same_take], cfg)
    assert len(runs) == 1
    assert runs[0].identity == 1
    assert all(rid != 2 for _, rid, _, _, _ in runs[0].samples)


def test_merge_possession_runs_bridges_brief_loose_ball_gap_same_identity():
    series = [
        (0.0, 1, 5.0, 100.0, 100.0),
        (0.2, None, None, 500.0, 500.0),
        (0.3, 1, 5.0, 100.0, 100.0),
    ]
    runs = possession._merge_possession_runs(series, identity_of=None, merge_gap_s=0.5)
    assert len(runs) == 1
    assert runs[0].t_start == 0.0
    assert runs[0].t_end == 0.3


def test_merge_possession_runs_does_not_bridge_across_another_identity():
    series = [
        (0.0, 1, 5.0, 100.0, 100.0),
        (0.5, 2, 5.0, 500.0, 500.0),  # a different identity's own real touch in between
        (1.0, 1, 5.0, 100.0, 100.0),
    ]
    runs = possession._merge_possession_runs(series, identity_of=None, merge_gap_s=2.0)
    assert [r.identity for r in runs] == [1, 2, 1]


def test_merge_possession_runs_splits_when_gap_exceeds_merge_gap_s():
    series = [(0.0, 1, 5.0, 100.0, 100.0), (5.0, 1, 5.0, 100.0, 100.0)]
    runs = possession._merge_possession_runs(series, identity_of=None, merge_gap_s=0.5)
    assert len(runs) == 2


def test_possession_runs_apply_identity_map():
    cfg = _events_config()
    frag_a = _player(10, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    frag_b = _player(11, take_id=0, boxes=[_box(0.3, 100.0, 100.0)])
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.3, 100.0, 100.0)]
    runs = possession.possession_runs_for_take(
        balls, [frag_a, frag_b], cfg, identity_of={10: 10, 11: 10}
    )
    assert len(runs) == 1
    assert runs[0].identity == 10


# ---------------------------------------------------------------------------
# detect_possession
# ---------------------------------------------------------------------------


def test_detect_possession_fires_for_sustained_run():
    cfg = _events_config()
    track = _player(1, take_id=3, boxes=[_box(t, 100.0, 100.0) for t in [0.0, 0.2, 0.4, 0.6]])
    balls = [_ball(t, 100.0, 100.0) for t in [0.0, 0.2, 0.4, 0.6]]
    runs = possession.possession_runs_for_take(balls, [track], cfg)
    events = possession.detect_possession(runs, take_id=3, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.POSSESSION
    assert ev.player_track_id == 1
    assert ev.take_id == 3
    assert ev.evidence["calibrated"] is False
    assert 0.0 <= ev.confidence <= cfg["possession"]["max_confidence"]


def test_detect_possession_drops_run_below_min_duration():
    cfg = _events_config()
    track = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    balls = [_ball(0.0, 100.0, 100.0)]  # single sample -> zero-duration run
    runs = possession.possession_runs_for_take(balls, [track], cfg)
    events = possession.detect_possession(runs, take_id=0, events_cfg=cfg)
    assert events == []


# ---------------------------------------------------------------------------
# detect_passes
# ---------------------------------------------------------------------------


def test_detect_passes_fires_between_confident_teammates():
    cfg = _events_config()
    passer = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    receiver = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [passer, receiver], cfg)
    events = possession.detect_passes(runs, [passer, receiver], take_id=0, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.PASS
    assert ev.player_track_id == 1  # passer credited
    assert ev.evidence["receiver_identity"] == 2
    assert 0.0 <= ev.confidence <= cfg["pass"]["max_confidence"]


def test_detect_passes_between_opponents_emits_turnover_not_pass():
    """ADR-20 acceptance #3 (different-cluster case): a possession change to a confidently
    OPPOSING colour cluster is a TURNOVER, never a PASS -- and must not be counted as a pass."""
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=1, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)

    assert [ev.type for ev in events] == [EventType.TURNOVER]
    turnover = events[0]
    assert turnover.player_track_id == 1  # the player who LOST the ball
    assert turnover.source == "possession_change_opposing_colour"
    assert turnover.evidence["losing_identity"] == 1
    assert turnover.evidence["receiving_identity"] == 2
    assert turnover.evidence["losing_team"] == 0
    assert turnover.evidence["receiving_team"] == 1
    assert 0.0 <= turnover.confidence <= cfg["turnover"]["max_confidence"]
    # never counted as a completed pass
    assert sum(1 for ev in events if ev.type == EventType.PASS) == 0


def test_detect_passes_same_cluster_receiver_still_fires_a_pass():
    """ADR-20 acceptance #3 (same-cluster case, unchanged behaviour): a same-colour receiver still
    fires a real PASS, not a turnover."""
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert [ev.type for ev in events] == [EventType.PASS]


def test_detect_passes_low_team_confidence_emits_neither_pass_nor_turnover():
    """ADR-20 acceptance #3 (low-confidence case, unchanged): when the team signal itself isn't
    trustworthy (`teammates_gate` returns `None`), neither a PASS nor a TURNOVER is ever guessed."""
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.1)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=1, team_confidence=0.1)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_passes_turnover_disabled_by_config_drops_instead_of_emitting():
    cfg = _events_config()
    cfg = {**cfg, "turnover": {**cfg["turnover"], "enabled": False}}
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=1, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_passes_same_colour_required_false_treats_opposing_colour_as_pass():
    cfg = _events_config()
    cfg = {**cfg, "pass": {**cfg["pass"], "same_colour_required": False}}
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=1, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert [ev.type for ev in events] == [EventType.PASS]


def test_detect_passes_degrades_gracefully_on_low_team_confidence_never_assumes_team_zero():
    """ADR-12: when team_confidence is poor (as measured on real clip2 -- every one of its 47
    tracks scored <= 0.3), a pass heuristic must emit NOTHING rather than silently assuming
    team=0 for everyone (which would make every possession change read as a "pass")."""
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.1)
    b = _player(2, take_id=0, boxes=[_box(1.0, 200.0, 100.0)], team=0, team_confidence=0.1)
    balls = [_ball(0.0, 100.0, 100.0), _ball(1.0, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_passes_rejects_ball_travel_beyond_pass_max_dist():
    cfg = _events_config()
    far = cfg["pass"]["pass_max_dist"] + 500.0
    a = _player(1, take_id=0, boxes=[_box(0.0, 0.0, 0.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=0, boxes=[_box(1.0, far, 0.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 0.0, 0.0), _ball(1.0, far, 0.0)]
    runs = possession.possession_runs_for_take(balls, [a, b], cfg)
    events = possession.detect_passes(runs, [a, b], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_passes_rejects_mixed_take_ids():
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=1, boxes=[_box(1.0, 200.0, 100.0)], team=0, team_confidence=0.9)
    with pytest.raises(AssertionError):
        possession.detect_passes([], [a, b], take_id=0, events_cfg=cfg)


def test_detect_dribbles_rejects_mixed_take_ids():
    cfg = _events_config()
    a = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    b = _player(2, take_id=1, boxes=[_box(1.0, 200.0, 100.0)], team=1, team_confidence=0.9)
    with pytest.raises(AssertionError):
        possession.detect_dribbles([], [a, b], take_id=0, events_cfg=cfg)


def test_detect_passes_no_pass_when_same_identity_regains_ball():
    cfg = _events_config()
    track = _player(1, take_id=0, boxes=[_box(t, 100.0, 100.0) for t in [0.0, 0.5, 3.0, 3.5]])
    balls = [_ball(t, 100.0, 100.0) for t in [0.0, 0.5, 3.0, 3.5]]
    runs = possession.possession_runs_for_take(balls, [track], cfg)
    # even if runs got split (e.g. by a gap), same identity on both sides is never a "pass"
    events = possession.detect_passes(runs, [track], take_id=0, events_cfg=cfg)
    assert events == []


# ---------------------------------------------------------------------------
# detect_dribbles
# ---------------------------------------------------------------------------


def test_detect_dribbles_fires_when_contested_and_travelling():
    cfg = _events_config()
    dribble_cfg = cfg["dribble"]
    n = 12
    dt = dribble_cfg["dribble_min_duration_s"] / (n - 1)
    travel_total = dribble_cfg["dribble_min_travel"] * 100.0 + 50.0  # comfortably over the bar
    possessor_boxes = [_box(i * dt, 100.0 + i * (travel_total / n), 100.0) for i in range(n)]
    possessor = _player(1, take_id=0, boxes=possessor_boxes, team=0, team_confidence=0.9)
    # an opponent standing right next to the possessor's START position -- contested at t=0
    opponent = _player(
        2, take_id=0, boxes=[_box(0.0, 105.0, 100.0, height=100.0)], team=1, team_confidence=0.9
    )
    balls = [_ball(b.t, b.bbox.cx, b.bbox.cy) for b in possessor_boxes]
    runs = possession.possession_runs_for_take(balls, [possessor, opponent], cfg)
    events = possession.detect_dribbles(runs, [possessor, opponent], take_id=0, events_cfg=cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.DRIBBLE
    assert ev.player_track_id == 1
    assert ev.evidence["unit"] == "bbox_heights"
    assert ev.evidence["calibrated"] is False
    assert 0.0 <= ev.confidence <= cfg["dribble"]["max_confidence"]


def test_detect_dribbles_does_not_fire_when_uncontested():
    cfg = _events_config()
    dribble_cfg = cfg["dribble"]
    n = 12
    dt = dribble_cfg["dribble_min_duration_s"] / (n - 1)
    travel_total = dribble_cfg["dribble_min_travel"] * 100.0 + 50.0
    possessor_boxes = [_box(i * dt, 100.0 + i * (travel_total / n), 100.0) for i in range(n)]
    possessor = _player(1, take_id=0, boxes=possessor_boxes, team=0, team_confidence=0.9)
    # no opponents at all -- travelling with the ball uncontested is not a "dribble" by this
    # heuristic's own definition (CLAUDE.md task: distinguishes "controlling under pressure").
    balls = [_ball(b.t, b.bbox.cx, b.bbox.cy) for b in possessor_boxes]
    runs = possession.possession_runs_for_take(balls, [possessor], cfg)
    events = possession.detect_dribbles(runs, [possessor], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_dribbles_does_not_fire_when_standing_still_even_if_contested():
    cfg = _events_config()
    dribble_cfg = cfg["dribble"]
    n = 12
    dt = dribble_cfg["dribble_min_duration_s"] / (n - 1)
    # no net travel at all -- standing still with the ball loosely nearby
    possessor_boxes = [_box(i * dt, 100.0, 100.0) for i in range(n)]
    possessor = _player(1, take_id=0, boxes=possessor_boxes, team=0, team_confidence=0.9)
    opponent = _player(
        2, take_id=0, boxes=[_box(0.0, 105.0, 100.0, height=100.0)], team=1, team_confidence=0.9
    )
    balls = [_ball(b.t, b.bbox.cx, b.bbox.cy) for b in possessor_boxes]
    runs = possession.possession_runs_for_take(balls, [possessor, opponent], cfg)
    events = possession.detect_dribbles(runs, [possessor, opponent], take_id=0, events_cfg=cfg)
    assert events == []


def test_detect_dribbles_too_short_duration_never_fires():
    cfg = _events_config()
    # duration well under dribble_min_duration_s
    possessor = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(0.1, 200.0, 100.0)])
    opponent = _player(2, take_id=0, boxes=[_box(0.0, 105.0, 100.0)], team=1, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.1, 200.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [possessor, opponent], cfg)
    events = possession.detect_dribbles(runs, [possessor, opponent], take_id=0, events_cfg=cfg)
    assert events == []


# ---------------------------------------------------------------------------
# compute_distance_covered
# ---------------------------------------------------------------------------


def test_compute_distance_covered_zero_for_stationary_track():
    cfg = _events_config()
    track = _player(1, take_id=0, boxes=[_box(i * 0.1, 100.0, 100.0) for i in range(10)])
    result = possession.compute_distance_covered([1], [track], cfg)
    assert result["distance"] == pytest.approx(0.0, abs=1e-9)
    assert result["unit"] == "bbox_heights"
    assert result["calibrated"] is False


def test_compute_distance_covered_positive_for_moving_track():
    cfg = _events_config()
    # bbox height 100px, moving 50px per 0.1s -> 5.0 bbox-heights/s * 0.9s ~= 4.5 bbox-heights
    track = _player(1, take_id=0, boxes=[_box(i * 0.1, 100.0 + i * 50.0, 100.0) for i in range(10)])
    result = possession.compute_distance_covered([1], [track], cfg)
    assert result["distance"] > 0.0


def test_compute_distance_covered_unknown_track_id_contributes_nothing():
    cfg = _events_config()
    track = _player(1, take_id=0, boxes=[_box(i * 0.1, 100.0, 100.0) for i in range(10)])
    result = possession.compute_distance_covered([999], [track], cfg)
    assert result["distance"] == 0.0
    assert result["n_speed_samples"] == 0
