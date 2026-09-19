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


# ---------------------------------------------------------------------------
# pass_min_gap_s debounce -- real bug fix, measured 2026-09-02 on clip1_43: 9 PASS events fired
# in an 11s clip, 0.1-0.4s apart, for the same passer identity.
# ---------------------------------------------------------------------------


def test_detect_passes_debounces_rapid_repeats_for_the_same_passer():
    """Player 1 passes to 2 at t=0.6 (passer=1), then quickly regains the ball and 'passes'
    again to 3 at t=1.2 (would-be passer=1 again, 0.3s after the first -- well under
    pass_min_gap_s=1.0). Real
    bug this closes, measured 2026-09-02 on clip1_43: exactly this pattern fired 9 PASS events in
    an 11s clip. Only the FIRST pass by player 1 may count.

    Timings scaled x3 on 2026-09-16 when `pass.pass_min_flight_s` (0.25s) landed: the original
    0.1-0.2s ball-flight gaps were never physically plausible (a kicked ball is out of everyone's
    possession for longer than a frame or two) and are exactly the flicker that floor now
    rejects. Every debounce relationship is preserved; only the flights are now real.
    """
    cfg = _events_config()
    p1 = _player(
        1,
        take_id=0,
        boxes=[_box(0, 100.0, 100.0), _box(0.9, 100.0, 100.0)],
        team=0,
        team_confidence=0.9,
    )
    p2 = _player(2, take_id=0, boxes=[_box(0.6, 200.0, 100.0)], team=0, team_confidence=0.9)
    p3 = _player(3, take_id=0, boxes=[_box(1.2, 300.0, 100.0)], team=0, team_confidence=0.9)
    balls = [
        _ball(0, 100.0, 100.0),
        _ball(0.6, 200.0, 100.0),
        _ball(0.9, 100.0, 100.0),
        _ball(1.2, 300.0, 100.0),
    ]
    runs = possession.possession_runs_for_take(balls, [p1, p2, p3], cfg)
    events = possession.detect_passes(runs, [p1, p2, p3], take_id=0, events_cfg=cfg)
    passes_by_1 = [e for e in events if e.type == EventType.PASS and e.player_track_id == 1]
    assert len(passes_by_1) == 1, "the second rapid-fire pass by the same player must be debounced"
    assert passes_by_1[0].t_end == pytest.approx(0.6)


def test_detect_passes_allows_a_second_pass_after_the_debounce_window():
    """The SAME fixture, but the second sequence happens well after pass_min_gap_s -- both of
    player 1's passes must count as genuinely separate."""
    cfg = _events_config()
    p1 = _player(
        1,
        take_id=0,
        boxes=[_box(0, 100.0, 100.0), _box(15, 100.0, 100.0)],
        team=0,
        team_confidence=0.9,
    )
    p2 = _player(2, take_id=0, boxes=[_box(0.6, 200.0, 100.0)], team=0, team_confidence=0.9)
    p3 = _player(3, take_id=0, boxes=[_box(15.6, 300.0, 100.0)], team=0, team_confidence=0.9)
    balls = [
        _ball(0, 100.0, 100.0),
        _ball(0.6, 200.0, 100.0),
        _ball(15, 100.0, 100.0),
        _ball(15.6, 300.0, 100.0),
    ]
    runs = possession.possession_runs_for_take(balls, [p1, p2, p3], cfg)
    events = possession.detect_passes(runs, [p1, p2, p3], take_id=0, events_cfg=cfg)
    passes_by_1 = [e for e in events if e.type == EventType.PASS and e.player_track_id == 1]
    assert len(passes_by_1) == 2


def test_detect_passes_debounce_is_per_passer_not_global():
    """A debounced passer must not suppress an UNRELATED passer's own genuine pass at a nearby
    time -- the gap is tracked per identity, never globally."""
    cfg = _events_config()
    p1 = _player(
        1,
        take_id=0,
        boxes=[_box(0, 100.0, 100.0), _box(0.9, 100.0, 100.0)],
        team=0,
        team_confidence=0.9,
    )
    p2 = _player(2, take_id=0, boxes=[_box(0.6, 200.0, 100.0)], team=0, team_confidence=0.9)
    p3 = _player(3, take_id=0, boxes=[_box(1.2, 300.0, 100.0)], team=0, team_confidence=0.9)
    # A second, entirely independent passer/receiver pair, well OUTSIDE the first sequence's own
    # time range -- detect_passes pairs run ADJACENCY across the take's full merged timeline
    # (unaffected by this fix), so an overlapping-time pair would interleave with player 1's own
    # runs rather than forming its own independent adjacent pair; separating them in time isolates
    # exactly the property under test (debounce state is keyed per identity, never shared).
    p4 = _player(4, take_id=0, boxes=[_box(30, 900.0, 900.0)], team=0, team_confidence=0.9)
    p5 = _player(5, take_id=0, boxes=[_box(30.6, 1000.0, 900.0)], team=0, team_confidence=0.9)
    balls = [
        _ball(0, 100.0, 100.0),
        _ball(0.6, 200.0, 100.0),
        _ball(0.9, 100.0, 100.0),
        _ball(1.2, 300.0, 100.0),
        _ball(30, 900.0, 900.0),
        _ball(30.6, 1000.0, 900.0),
    ]
    runs = possession.possession_runs_for_take(balls, [p1, p2, p3, p4, p5], cfg)
    events = possession.detect_passes(runs, [p1, p2, p3, p4, p5], take_id=0, events_cfg=cfg)
    passers = {e.player_track_id for e in events if e.type == EventType.PASS}
    assert (
        4 in passers
    ), "player 4's own independent pass must not be suppressed by player 1's debounce"


# ---------------------------------------------------------------------------
# teammates_test -- kit-colour-first cascade (Stage 5, owner request: "also add the player jersy
# color for passes and assist")
# ---------------------------------------------------------------------------


def test_teammates_test_kit_colour_overrides_a_degenerate_team_cluster():
    """The exact real failure this fixes: team_confidence reads as a useless constant (measured
    0.30 on 229/231 real tracks), so the cluster gate alone always returns None. Kit colour must
    be able to resolve the pass anyway when it's confidently measured."""
    import numpy as np

    cfg = _events_config()
    tracks_by_id = {
        1: _player(1, 0, [_box(0.0, 0, 0)], team=1, team_confidence=0.30),
        2: _player(2, 0, [_box(0.0, 0, 0)], team=1, team_confidence=0.30),
    }
    kit_lab = {
        1: np.array([44.9, 7.5, -24.0]),
        2: np.array([47.2, 6.0, -22.0]),
    }  # both blue, dE=3.4
    result = possession.teammates_test(
        1, 2, tracks_by_id, cfg["pass"]["team_confidence_threshold"], cfg["kit_colour"], kit_lab
    )
    assert result is True


def test_teammates_test_falls_back_to_cluster_when_colour_undecided():
    """When kit colour itself lands in the undecided gray zone, fall through to the existing
    cluster gate rather than guessing."""
    import numpy as np

    cfg = _events_config()
    tracks_by_id = {
        1: _player(1, 0, [_box(0.0, 0, 0)], team=0, team_confidence=0.9),
        2: _player(2, 0, [_box(0.0, 0, 0)], team=0, team_confidence=0.9),
    }
    # dE=13.0, inside the measured undecided gap (8.0-24.0)
    kit_lab = {1: np.array([47.2, 6.0, -22.0]), 2: np.array([47.5, 6.5, -9.0])}
    result = possession.teammates_test(
        1, 2, tracks_by_id, cfg["pass"]["team_confidence_threshold"], cfg["kit_colour"], kit_lab
    )
    assert result is True  # resolved by the (confident) cluster gate instead


def test_teammates_test_none_kit_lab_dict_is_exactly_the_old_cluster_only_behaviour():
    """Backward compatibility: a caller passing kit_lab_by_track_id=None (the default -- no
    per-take colour sampling done yet) must get precisely the pre-existing cluster-only gate."""
    cfg = _events_config()
    tracks_by_id = {
        1: _player(1, 0, [_box(0.0, 0, 0)], team=1, team_confidence=0.30),
        2: _player(2, 0, [_box(0.0, 0, 0)], team=1, team_confidence=0.30),
    }
    result = possession.teammates_test(
        1, 2, tracks_by_id, cfg["pass"]["team_confidence_threshold"], cfg["kit_colour"], None
    )
    assert (
        result is None
    )  # low team_confidence, no colour data supplied -> can't tell, same as before


def test_detect_passes_rejects_a_single_frame_flight_as_flicker():
    """Owner-reported 2026-09-16 ("for 1 pass it counts 5"). Measured on clip5_77: every false
    positive had the ball "in flight" for 1-6 frames -- possession flickering between adjacent
    players. A kicked ball is out of everyone's possession for the whole time it travels, so a
    one-frame gap is physically only possible if the players are touching. Must be dropped and
    LOGGED (never a silent filter)."""
    from src.common.logging import DropCounter

    cfg = _events_config()
    p1 = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    p2 = _player(2, take_id=0, boxes=[_box(0.033, 300.0, 100.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.033, 300.0, 100.0)]  # one frame at 30fps
    runs = possession.possession_runs_for_take(balls, [p1, p2], cfg)
    drops = DropCounter("t")
    events = possession.detect_passes(runs, [p1, p2], take_id=0, events_cfg=cfg, drops=drops)
    assert not [e for e in events if e.type == EventType.PASS]
    assert drops.as_dict().get("pass_flight_too_short", 0) >= 1


def test_detect_passes_rejects_ball_that_did_not_travel():
    """Companion floor: a ball that 'moved' a few dozen px (measured 38px on clip5_77 at
    t=10.87) is detection jitter, not a pass -- even with a plausible flight time."""
    from src.common.logging import DropCounter

    cfg = _events_config()
    p1 = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    p2 = _player(2, take_id=0, boxes=[_box(0.5, 130.0, 100.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.5, 130.0, 100.0)]  # 30px, 0.5s
    runs = possession.possession_runs_for_take(balls, [p1, p2], cfg)
    drops = DropCounter("t")
    events = possession.detect_passes(runs, [p1, p2], take_id=0, events_cfg=cfg, drops=drops)
    assert not [e for e in events if e.type == EventType.PASS]
    assert drops.as_dict().get("pass_ball_travel_too_short", 0) >= 1


def test_detect_passes_still_counts_a_real_pass_with_plausible_flight_and_travel():
    """The floors must not eat a genuine pass: 0.5s in flight, 200px of travel, teammate."""
    cfg = _events_config()
    p1 = _player(1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)], team=0, team_confidence=0.9)
    p2 = _player(2, take_id=0, boxes=[_box(0.5, 300.0, 100.0)], team=0, team_confidence=0.9)
    balls = [_ball(0.0, 100.0, 100.0), _ball(0.5, 300.0, 100.0)]
    runs = possession.possession_runs_for_take(balls, [p1, p2], cfg)
    events = possession.detect_passes(runs, [p1, p2], take_id=0, events_cfg=cfg)
    assert len([e for e in events if e.type == EventType.PASS]) == 1


# ---------------------------------------------------------------------------
# detect_passes with ball kinematics -- 2026-09-19, "not counting the passes correctly"
# ---------------------------------------------------------------------------


def _ball_states_for(balls, cfg, ref_height=100.0):
    from src.events.ball_track import build_ball_state_series

    return build_ball_state_series(balls, cfg["ball"], ref_height)


def _short_pass_fixture(cfg, kicked: bool):
    """Two teammates 160px apart (their `possession_max_dist` zones nearly touch), so the ball
    is out of BOTH zones for only a frame or so -- a real short pass has exactly this signature.
    `kicked=True`: the ball sits at #1's feet for 0.5s, then is struck (stationary -> 25px/frame).
    `kicked=False`: the ball drifts at one constant 4px/frame the whole time -- it still crosses
    from #1's zone into #2's, but its trajectory never changes, i.e. nobody kicked it."""
    frames = range(0, 60)
    p1 = _player(1, take_id=0, boxes=[_box(t / 30, 100.0, 100.0) for t in frames],
                 team=0, team_confidence=0.9)
    p2 = _player(2, take_id=0, boxes=[_box(t / 30, 260.0, 100.0) for t in frames],
                 team=0, team_confidence=0.9)
    balls = []
    if kicked:
        for i in range(0, 16):
            balls.append(_ball(i / 30, 100.0, 100.0))
        for i in range(16, 30):
            balls.append(_ball(i / 30, 100.0 + 25.0 * (i - 15), 100.0))
    else:
        for i in range(0, 50):
            balls.append(_ball(i / 30, 100.0 + 4.0 * i, 100.0))
    return p1, p2, balls


def test_detect_passes_counts_a_short_pass_when_the_kick_is_verified():
    from src.common.logging import DropCounter

    cfg = _events_config()
    p1, p2, balls = _short_pass_fixture(cfg, kicked=True)
    runs = possession.possession_runs_for_take(balls, [p1, p2], cfg)
    b_start = next(b.t_start for b in runs if b.identity == 2)
    a_end = next(a.t_end for a in runs if a.identity == 1)
    gap = b_start - a_end
    assert gap < cfg["pass"]["pass_min_flight_s"], "fixture must be a SHORT flight"

    drops = DropCounter("t")
    # without kinematics the flight floor alone still (correctly, conservatively) says flicker
    assert not possession.detect_passes(runs, [p1, p2], 0, cfg, drops=drops)
    assert drops.as_dict().get("pass_flight_too_short", 0) >= 1

    states = _ball_states_for(balls, cfg)
    events = possession.detect_passes(runs, [p1, p2], 0, cfg, ball_states=states)
    passes = [e for e in events if e.type == EventType.PASS]
    assert len(passes) == 1
    assert passes[0].player_track_id == 1
    assert passes[0].evidence["kick_verified"] is True


def test_detect_passes_short_flight_without_a_kick_is_still_flicker():
    from src.common.logging import DropCounter

    cfg = _events_config()
    p1, p2, balls = _short_pass_fixture(cfg, kicked=False)
    runs = possession.possession_runs_for_take(balls, [p1, p2], cfg)
    assert [r.identity for r in runs] == [1, 2]
    states = _ball_states_for(balls, cfg)
    drops = DropCounter("t")
    events = possession.detect_passes(runs, [p1, p2], 0, cfg, drops=drops, ball_states=states)
    assert not [e for e in events if e.type == EventType.PASS]
    assert drops.as_dict().get("pass_flight_too_short", 0) >= 1


def test_detect_passes_drops_a_fly_by_past_a_third_player():
    """#1 kicks to #2; the ball rolls straight past #3 (one sample inside #3's zone, trajectory
    unchanged). #3 must NOT be credited with a 'pass' to #2 -- the pre-existing behaviour did
    exactly that whenever the #3->#2 gap cleared `pass_min_flight_s`."""
    from src.common.logging import DropCounter

    cfg = _events_config()
    frames = range(0, 90)
    p1 = _player(1, 0, [_box(t / 30, 100.0, 100.0) for t in frames], team=0, team_confidence=0.9)
    p3 = _player(3, 0, [_box(t / 30, 400.0, 100.0) for t in frames], team=0, team_confidence=0.9)
    p2 = _player(2, 0, [_box(t / 30, 700.0, 100.0) for t in frames], team=0, team_confidence=0.9)
    balls = [_ball(i / 30, 100.0 + i, 100.0) for i in range(0, 16)]  # at #1's feet, 0.5s
    # kicked at 0.5s: 20px/frame straight through #3's zone (x 330-470, ~7 frames) on to #2
    for i in range(16, 60):
        balls.append(_ball(i / 30, 115.0 + 20.0 * (i - 15), 100.0))
    runs = possession.possession_runs_for_take(balls, [p1, p2, p3], cfg)
    assert [r.identity for r in runs] == [1, 3, 2], [r.identity for r in runs]
    states = _ball_states_for(balls, cfg)
    drops = DropCounter("t")
    events = possession.detect_passes(runs, [p1, p2, p3], 0, cfg, drops=drops, ball_states=states)
    passers = sorted(e.player_track_id for e in events if e.type == EventType.PASS)
    assert 3 not in passers, "the fly-by player must never be credited"
    assert drops.as_dict().get("pass_passer_never_played_ball", 0) >= 1


def test_pass_confidence_does_not_veto_a_pass_inside_pass_max_dist():
    """A 700px pass (inside the measured 750px `pass_max_dist`) with ordinary run quality used to
    fall under `min_emit_confidence` purely through the closeness term reaching ~0."""
    cfg = _events_config()
    pass_cfg = cfg["pass"]
    min_emit = cfg["confidence"]["min_emit_confidence"]
    conf = possession.pass_confidence(0.7, 0.7, 700.0, pass_cfg)
    assert conf >= min_emit
    # ...and distance still lowers the (unclamped) closeness relative to a tap to a neighbour
    assert possession.travel_closeness(700.0, pass_cfg) < possession.travel_closeness(
        50.0, pass_cfg
    )
    floor = pass_cfg["distance_closeness_floor"]
    assert possession.travel_closeness(pass_cfg["pass_max_dist"], pass_cfg) == pytest.approx(floor)
    assert possession.travel_closeness(0.0, pass_cfg) == pytest.approx(1.0)
