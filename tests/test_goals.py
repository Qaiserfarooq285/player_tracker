"""Pure-logic unit tests for ADR-17's real goal + assist detector (`src/events/goals.py`).

Covers: scoreboard-text parsing, the region-activation self-check (`scan_candidate_regions`),
debounced increment detection (`find_score_increments`), the occurrence detector
(`detect_goals_scoreboard_delta`) against a mocked EasyOCR reader, team/scorer attribution
(`attribute_goal_team_and_scorer`), assist crediting (`find_assist_event`), the final
Golden-Rule-5 reporting decision table (`check_goal_availability`), and one end-to-end
`detect_goals_for_video` run with `decode_frames`/EasyOCR monkeypatched out (no GPU/ffmpeg/network
required). Every test in this file must pass without a GPU or a live API key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.io import load_yaml
from src.common.types import BallDetection, BBox, Event, EventType, Take, Track
from src.events import goals as goals_mod
from src.events.goals import (
    GoalDetectionResult,
    attribute_goal_team_and_scorer,
    check_goal_availability,
    detect_goals_goal_region,
    detect_goals_scoreboard_delta,
    find_assist_event,
    find_score_increments,
    parse_scoreboard_text,
    scan_candidate_regions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _events_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")


def _goal_cfg(**overrides) -> dict:
    cfg = dict(_events_config()["goal"])
    cfg.update(overrides)
    return cfg


def _assist_cfg(**overrides) -> dict:
    cfg = dict(_events_config()["assist"])
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# parse_scoreboard_text
# ---------------------------------------------------------------------------


def test_parse_scoreboard_text_dash_separator():
    assert parse_scoreboard_text("1 - 0") == (1, 0)


def test_parse_scoreboard_text_colon_separator_no_spaces():
    assert parse_scoreboard_text("2:1") == (2, 1)


def test_parse_scoreboard_text_extracts_from_noisy_joined_text():
    assert parse_scoreboard_text("HOME 3 - 2 AWAY") == (3, 2)


def test_parse_scoreboard_text_none_on_no_pattern():
    assert parse_scoreboard_text("veo") is None


def test_parse_scoreboard_text_none_on_empty_or_none():
    assert parse_scoreboard_text("") is None
    assert parse_scoreboard_text(None) is None


# ---------------------------------------------------------------------------
# scan_candidate_regions -- the self-activation check (Golden Rule 5 / ADR-11)
# ---------------------------------------------------------------------------


class _ShapeKeyedReader:
    """A fake EasyOCR reader whose `readtext` result depends only on the crop's own `(h, w)`
    shape and a single "control pixel" at `crop[0, 0, 0]` -- lets a test give two differently
    SIZED candidate regions of the SAME frame two independently controllable readings, without
    needing a real OCR model or real digit glyphs drawn into the frame."""

    def __init__(self, script: dict[tuple[int, int, int], str]) -> None:
        self.script = script

    def readtext(self, crop):
        key = (crop.shape[0], crop.shape[1], int(crop[0, 0, 0]))
        text = self.script.get(key, "")
        return [((0, 0), text, 0.9)] if text else []


def test_scan_candidate_regions_activates_the_stable_region_and_skips_the_noisy_one():
    region_a = [0.0, 0.0, 0.3, 0.2]  # -> crop shape (20, 30) on a 100x100 frame
    region_b = [0.6, 0.0, 1.0, 0.2]  # -> crop shape (20, 40)
    cfg = {"candidate_regions": [region_a, region_b], "min_stable_frames": 3}

    # region A's control pixel varies every frame (never a stable reading); region B's control
    # pixel is constant, reading a stable "1 - 0" every frame.
    frames = []
    for i in range(4):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0, 0, 0] = i  # region A control pixel -- different every frame
        frame[0, 60, 0] = 9  # region B control pixel -- constant
        frames.append(frame)

    reader = _ShapeKeyedReader({(20, 40, 9): "1 - 0"})  # region A never produces a script hit

    region, debug = scan_candidate_regions(reader, frames, cfg)
    assert region == (0.6, 0.0, 1.0, 0.2)
    assert debug["activated_region"] == [0.6, 0.0, 1.0, 0.2]
    assert debug["activated_reading"] == [1, 0]
    assert debug["candidates"][0]["best_count"] == 0  # region A: never a single reading


def test_scan_candidate_regions_returns_none_when_nothing_is_stable():
    region_a = [0.0, 0.0, 0.3, 0.2]
    cfg = {"candidate_regions": [region_a], "min_stable_frames": 3}
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
    reader = _ShapeKeyedReader({})  # every crop reads as empty text

    region, debug = scan_candidate_regions(reader, frames, cfg)
    assert region is None
    assert debug["activated_region"] is None


def test_scan_candidate_regions_requires_a_majority_not_a_bare_plurality():
    region_a = [0.0, 0.0, 0.3, 0.2]
    cfg = {"candidate_regions": [region_a], "min_stable_frames": 3}
    # 4 frames: only 2 share a reading -- below min_stable_frames=3, must NOT activate.
    frames = []
    for val in (1, 1, 2, 3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0, 0, 0] = val
        frames.append(frame)
    reader = _ShapeKeyedReader({(20, 30, 1): "1 - 0", (20, 30, 2): "2 - 0", (20, 30, 3): "3 - 0"})

    region, debug = scan_candidate_regions(reader, frames, cfg)
    assert region is None
    assert debug["candidates"][0]["best_count"] == 2


# ---------------------------------------------------------------------------
# find_score_increments -- debounce (Golden Rule 5: never fire on a single misread)
# ---------------------------------------------------------------------------


def test_find_score_increments_detects_a_debounced_plus_one():
    samples = [(0.0, 0), (1.0, 0), (2.0, 1), (3.0, 1), (4.0, 1)]
    cfg = {"increment_debounce_samples": 2}
    increments = find_score_increments(samples, cfg)
    assert increments == [{"t_before": 0.0, "t_after": 2.0, "score_before": 0, "score_after": 1}]


def test_find_score_increments_ignores_a_single_frame_misread():
    # a single stray "1" reverting immediately to "0" must never fire a goal
    samples = [(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 0)]
    cfg = {"increment_debounce_samples": 2}
    assert find_score_increments(samples, cfg) == []


def test_find_score_increments_never_guesses_a_multi_step_jump():
    # a persistent jump of +2 is ambiguous (could be 1 or 2 real goals) -- never emitted, Golden
    # Rule 5 -- but the baseline still resyncs so a LATER clean +1 is still detected.
    samples = [(0.0, 0), (1.0, 2), (2.0, 2), (3.0, 3), (4.0, 3)]
    cfg = {"increment_debounce_samples": 2}
    increments = find_score_increments(samples, cfg)
    assert increments == [{"t_before": 1.0, "t_after": 3.0, "score_before": 2, "score_after": 3}]


def test_find_score_increments_empty_samples_is_empty():
    assert find_score_increments([], {"increment_debounce_samples": 2}) == []


def test_find_score_increments_decrease_resyncs_without_emitting():
    samples = [(0.0, 1), (1.0, 0), (2.0, 0), (3.0, 1), (4.0, 1)]
    cfg = {"increment_debounce_samples": 2}
    increments = find_score_increments(samples, cfg)
    # 1 -> 0 is a decrease (OCR noise), resynced; then 0 -> 1 is a clean, debounced +1.
    assert increments == [{"t_before": 1.0, "t_after": 3.0, "score_before": 0, "score_after": 1}]


# ---------------------------------------------------------------------------
# detect_goals_scoreboard_delta -- real implementation, no longer NotImplementedError
# ---------------------------------------------------------------------------


def test_detect_goals_scoreboard_delta_no_activation_returns_no_events(monkeypatch):
    monkeypatch.setattr(
        goals_mod,
        "scan_candidate_regions",
        lambda reader, frames, cfg: (None, {"activated_region": None}),
    )
    events, debug = detect_goals_scoreboard_delta(
        reader=object(),
        frames=[np.zeros((5, 5, 3), dtype=np.uint8)],
        times=[0.0],
        take_id=0,
        goal_cfg=_goal_cfg(),
    )
    assert events == []
    assert debug["activated_region"] is None


def test_detect_goals_scoreboard_delta_emits_an_occurrence_only_goal_event(monkeypatch):
    region = (0.0, 0.0, 0.25, 0.12)
    monkeypatch.setattr(
        goals_mod,
        "scan_candidate_regions",
        lambda reader, frames, cfg: (region, {"activated_region": list(region)}),
    )
    # frame i's control pixel encodes the scripted reading for that sample.
    script = {0: "0 - 0", 1: "0 - 0", 2: "1 - 0", 3: "1 - 0", 4: "1 - 0"}
    frames = [np.full((10, 10, 3), i, dtype=np.uint8) for i in script]
    times = [float(i) for i in script]

    def fake_ocr_text(reader, crop):
        return script[int(crop[0, 0, 0])]

    monkeypatch.setattr(goals_mod, "_ocr_region_text", fake_ocr_text)

    cfg = _goal_cfg(increment_debounce_samples=2, occurrence_confidence=0.75)
    events, debug = detect_goals_scoreboard_delta(
        reader=object(), frames=frames, times=times, take_id=3, goal_cfg=cfg
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.GOAL
    assert ev.take_id == 3
    assert ev.player_track_id is None
    assert ev.confidence == cfg["occurrence_confidence"]
    assert ev.evidence["score_before"] == 0
    assert ev.evidence["score_after"] == 1
    assert ev.evidence["activated_region"] == list(region)
    assert debug["increments"] == [
        {"t_before": 0.0, "t_after": 2.0, "score_before": 0, "score_after": 1}
    ]


# ---------------------------------------------------------------------------
# attribute_goal_team_and_scorer (ADR-17 steps 2-3)
# ---------------------------------------------------------------------------


def _track(track_id: int, team: int | None) -> Track:
    return Track(id=track_id, take_id=0, boxes=[], team=team, team_confidence=0.9)


def _possession_event(
    identity: int, raw_track_id: int, t_start: float, t_end: float, eid: str | None = None
) -> Event:
    return Event(
        id=eid or f"poss-{identity}-{t_start}",
        type=EventType.POSSESSION,
        t_start=t_start,
        t_end=t_end,
        player_track_id=identity,
        take_id=0,
        confidence=0.3,
        source="ball_proximity_possession_heuristic",
        evidence={"raw_track_ids": [raw_track_id]},
    )


def _goal_event(t_after: float, confidence: float = 0.75) -> Event:
    return Event(
        id="goal-1",
        type=EventType.GOAL,
        t_start=t_after - 1.0,
        t_end=t_after,
        player_track_id=None,
        take_id=0,
        confidence=confidence,
        source="scoreboard_ocr_delta",
        evidence={"t_after": t_after, "score_before": 0, "score_after": 1},
    )


def test_attribute_goal_credits_the_team_with_more_possession_time():
    tracks_by_id = {1: _track(1, team=0), 2: _track(2, team=1)}
    # team 0 (track 1) holds the ball much longer than team 1 (track 2) in the lookback window.
    possession_events = [
        _possession_event(identity=10, raw_track_id=1, t_start=90.0, t_end=98.0),
        _possession_event(identity=20, raw_track_id=2, t_start=98.5, t_end=99.0),
    ]
    goal_event = _goal_event(t_after=100.0)
    cfg = _goal_cfg(possession_lookback_s=15.0)

    attributed = attribute_goal_team_and_scorer(goal_event, possession_events, tracks_by_id, cfg)
    assert attributed.evidence["credited_team"] == 0
    assert attributed.player_track_id == 10  # the LAST possession run of the credited team
    expected_conf = cfg["occurrence_confidence"] * cfg["scorer_confidence_multiplier"]
    assert attributed.confidence == expected_conf
    # never mutates the original event
    assert goal_event.player_track_id is None


def test_attribute_goal_picks_the_last_run_of_the_credited_team_as_scorer():
    tracks_by_id = {1: _track(1, team=0)}
    possession_events = [
        _possession_event(identity=10, raw_track_id=1, t_start=90.0, t_end=92.0, eid="p1"),
        _possession_event(identity=11, raw_track_id=1, t_start=95.0, t_end=99.0, eid="p2"),
    ]
    goal_event = _goal_event(t_after=100.0)
    cfg = _goal_cfg(possession_lookback_s=15.0)

    attributed = attribute_goal_team_and_scorer(goal_event, possession_events, tracks_by_id, cfg)
    assert attributed.player_track_id == 11
    assert attributed.evidence["supporting_possession_event_id"] == "p2"


def test_attribute_goal_undetermined_team_penalises_confidence_never_guesses():
    goal_event = _goal_event(t_after=100.0)
    cfg = _goal_cfg()
    attributed = attribute_goal_team_and_scorer(goal_event, [], {}, cfg)
    assert attributed.player_track_id is None
    assert attributed.confidence == max(
        cfg["min_confidence"], goal_event.confidence * cfg["team_unknown_confidence_penalty"]
    )
    assert "undetermined" in attributed.evidence["team_attribution"]


def test_attribute_goal_ignores_possession_outside_the_lookback_window():
    tracks_by_id = {1: _track(1, team=0)}
    # this possession ended long before the lookback window even starts
    possession_events = [_possession_event(identity=10, raw_track_id=1, t_start=1.0, t_end=2.0)]
    goal_event = _goal_event(t_after=100.0)
    cfg = _goal_cfg(possession_lookback_s=8.0)
    attributed = attribute_goal_team_and_scorer(goal_event, possession_events, tracks_by_id, cfg)
    assert attributed.player_track_id is None


# ---------------------------------------------------------------------------
# find_assist_event (ADR-17 step 4)
# ---------------------------------------------------------------------------


def _pass_event(
    passer: int, receiver: int, t_start: float, t_end: float, eid: str = "pass-1"
) -> Event:
    return Event(
        id=eid,
        type=EventType.PASS,
        t_start=t_start,
        t_end=t_end,
        player_track_id=passer,
        take_id=0,
        confidence=0.3,
        source="possession_change_teammate_heuristic",
        evidence={"passer_identity": passer, "receiver_identity": receiver},
    )


def _scored_goal_event(t_after: float, scorer: int) -> Event:
    return _goal_event(t_after=t_after).model_copy(
        update={"player_track_id": scorer, "t_end": t_after}
    )


def test_find_assist_event_credits_the_passer_of_a_qualifying_pass():
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [_pass_event(passer=7, receiver=11, t_start=97.0, t_end=98.0)]
    cfg = _assist_cfg(assist_window_seconds=15.0)

    assist = find_assist_event(goal_event, pass_events, cfg)
    assert assist is not None
    assert assist.type == EventType.ASSIST
    assert assist.player_track_id == 7
    assert assist.evidence["receiver_identity"] == 11
    assert assist.evidence["goal_event_id"] == goal_event.id
    expected = min(
        cfg["max_confidence"],
        max(cfg["min_confidence"], goal_event.confidence * cfg["assist_confidence_multiplier"]),
    )
    assert assist.confidence == expected


def test_find_assist_event_none_when_scorer_unknown():
    goal_event = _goal_event(t_after=100.0)  # player_track_id is None
    pass_events = [_pass_event(passer=7, receiver=11, t_start=97.0, t_end=98.0)]
    assert find_assist_event(goal_event, pass_events, _assist_cfg()) is None


def test_find_assist_event_none_when_no_qualifying_pass():
    goal_event = _goal_event(t_after=100.0).model_copy(update={"player_track_id": 11})
    # a pass to a DIFFERENT receiver never qualifies
    pass_events = [_pass_event(passer=7, receiver=99, t_start=97.0, t_end=98.0)]
    assert find_assist_event(goal_event, pass_events, _assist_cfg()) is None


def test_find_assist_event_none_when_pass_outside_window():
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [_pass_event(passer=7, receiver=11, t_start=10.0, t_end=11.0)]  # far too early
    cfg = _assist_cfg(assist_window_seconds=15.0)
    assert find_assist_event(goal_event, pass_events, cfg) is None


def test_find_assist_event_picks_the_nearest_preceding_pass():
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [
        _pass_event(passer=5, receiver=11, t_start=90.0, t_end=91.0, eid="earlier"),
        _pass_event(passer=7, receiver=11, t_start=97.0, t_end=98.0, eid="nearest"),
    ]
    cfg = _assist_cfg(assist_window_seconds=15.0)
    assist = find_assist_event(goal_event, pass_events, cfg)
    assert assist.evidence["supporting_pass_event_id"] == "nearest"
    assert assist.player_track_id == 7


# ---------------------------------------------------------------------------
# find_assist_event -- teammate check (Stage D bug fix, "streamed-gathering-treehouse" plan).
# The docstring always claimed "a teammate pass" but the code never verified it; these tests
# exercise the NEW `tracks_by_id`-gated re-check via the same `teammates_gate` tri-state test
# `pass`/`tackle`/`dribble` already use.
# ---------------------------------------------------------------------------


def _pass_event_with_raw(
    passer: int,
    receiver: int,
    passer_raw: int,
    t_start: float,
    t_end: float,
    eid: str = "pass-1",
) -> Event:
    return Event(
        id=eid,
        type=EventType.PASS,
        t_start=t_start,
        t_end=t_end,
        player_track_id=passer,
        take_id=0,
        confidence=0.3,
        source="possession_change_teammate_heuristic",
        evidence={
            "passer_identity": passer,
            "receiver_identity": receiver,
            "passer_raw_track_id": passer_raw,
        },
    )


def test_find_assist_event_credits_when_passer_confirmed_teammate_of_scorer():
    # scorer's own track id doubles as its "raw" id here (matches the real Phase-1 flow, where
    # identity_of is None and run.identity IS the raw Track.id -- see detect_goals_for_take's own
    # docstring). Both on team 0 -- confirmed teammates.
    tracks_by_id = {7: _track(7, team=0), 11: _track(11, team=0)}
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [
        _pass_event_with_raw(passer=7, receiver=11, passer_raw=7, t_start=97.0, t_end=98.0)
    ]
    cfg = _assist_cfg(assist_window_seconds=15.0)

    assist = find_assist_event(goal_event, pass_events, cfg, tracks_by_id)
    assert assist is not None
    assert assist.player_track_id == 7


def test_find_assist_event_rejects_when_passer_confirmed_opposing_team():
    # Real gap this fix closes: receiver_identity == scorer matches, but passer (team 1) and
    # scorer (team 0) are on CONFIRMED OPPOSING teams -- never a real assist.
    tracks_by_id = {7: _track(7, team=1), 11: _track(11, team=0)}
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [
        _pass_event_with_raw(passer=7, receiver=11, passer_raw=7, t_start=97.0, t_end=98.0)
    ]
    cfg = _assist_cfg(assist_window_seconds=15.0)

    assert find_assist_event(goal_event, pass_events, cfg, tracks_by_id) is None


def test_find_assist_event_rejects_when_team_signal_not_trustworthy():
    # `teammates_gate` returns None (untrustworthy) when either side's team_confidence is below
    # the threshold -- same "can't tell, don't guess" convention as pass/tackle/dribble: excluded,
    # never assumed to be a teammate just because no evidence says otherwise.
    tracks_by_id = {
        7: Track(id=7, take_id=0, boxes=[], team=0, team_confidence=0.1),
        11: _track(11, team=0),
    }
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [
        _pass_event_with_raw(passer=7, receiver=11, passer_raw=7, t_start=97.0, t_end=98.0)
    ]
    cfg = _assist_cfg(assist_window_seconds=15.0)

    assert find_assist_event(goal_event, pass_events, cfg, tracks_by_id) is None


def test_find_assist_event_skips_teammate_check_when_tracks_by_id_not_given():
    # Backward compatibility: tracks_by_id=None (the default) preserves the exact pre-fix
    # behaviour -- no re-check performed at all.
    goal_event = _scored_goal_event(t_after=100.0, scorer=11)
    pass_events = [_pass_event(passer=7, receiver=11, t_start=97.0, t_end=98.0)]
    cfg = _assist_cfg(assist_window_seconds=15.0)

    assist = find_assist_event(goal_event, pass_events, cfg)
    assert assist is not None
    assert assist.player_track_id == 7


# ---------------------------------------------------------------------------
# detect_goals_goal_region (ADR-20) -- acceptance #4
# ---------------------------------------------------------------------------


def _goal_region_cfg(**overrides) -> dict:
    cfg = dict(_events_config()["goal_region"])
    cfg.update(overrides)
    return cfg


def _region_ball(t: float, cx: float, cy: float = 0.5) -> BallDetection:
    return BallDetection(
        bbox=BBox(x1=cx - 5, y1=cy - 5, x2=cx + 5, y2=cy + 5), conf=0.9, frame_index=0, t=t
    )


SQUARE_POLYGON = [(0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)]  # right-edge 20% of frame


def test_detect_goals_goal_region_no_polygon_configured_returns_empty():
    cfg = _goal_region_cfg(regions={})
    balls = [_region_ball(1.0, cx=950.0)]
    events = detect_goals_goal_region(balls, [(0.0, 0.9)], "someslug", cfg, take_id=0)
    assert events == []


def test_detect_goals_goal_region_crossing_inside_shot_window_fires_one_goal():
    cfg = _goal_region_cfg(regions={"someslug": [SQUARE_POLYGON]})
    # ball centroid at cx=950 on an implicit 1000-wide frame (cx/cy passed as already-scaled
    # coordinates, matching how detect_goals_for_take feeds it -- polygon fractions * frame_width)
    balls = [_region_ball(t=1.0, cx=0.95 * 1.0)]  # cx as a FRACTION here (polygon test is unitless)
    # a raw fast-ball-motion window ending 0.5s before the crossing, inside shot_window_s=3.0
    fast_ball_windows = [(0.0, 0.5)]
    events = detect_goals_goal_region(balls, fast_ball_windows, "someslug", cfg, take_id=7)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.GOAL
    assert ev.source == "goal_region"
    assert ev.take_id == 7
    assert ev.player_track_id is None
    assert ev.confidence == cfg["confidence"]
    assert ev.evidence["goal_region_source"] == "manual"
    assert ev.evidence["slug"] == "someslug"


def test_detect_goals_goal_region_crossing_without_preceding_fast_window_does_not_fire():
    cfg = _goal_region_cfg(regions={"someslug": [SQUARE_POLYGON]})
    balls = [_region_ball(t=10.0, cx=0.95)]
    # no fast-ball-motion windows at all -- ball merely sitting/rolling in the region is not a
    # goal on its own
    events = detect_goals_goal_region(balls, [], "someslug", cfg, take_id=0)
    assert events == []


def test_detect_goals_goal_region_crossing_outside_shot_window_does_not_fire():
    cfg = _goal_region_cfg(regions={"someslug": [SQUARE_POLYGON]})
    balls = [_region_ball(t=10.0, cx=0.95)]
    fast_ball_windows = [
        (0.5, 1.0)
    ]  # ended 9s before the crossing -- far outside shot_window_s=3.0
    events = detect_goals_goal_region(balls, fast_ball_windows, "someslug", cfg, take_id=0)
    assert events == []


def test_detect_goals_goal_region_ball_outside_polygon_does_not_fire():
    cfg = _goal_region_cfg(regions={"someslug": [SQUARE_POLYGON]})
    balls = [_region_ball(t=1.0, cx=0.1)]  # far from the polygon (right 20% of frame)
    fast_ball_windows = [(0.0, 0.5)]
    events = detect_goals_goal_region(balls, fast_ball_windows, "someslug", cfg, take_id=0)
    assert events == []


def test_detect_goals_goal_region_debounces_one_continuous_crossing():
    cfg = _goal_region_cfg(regions={"someslug": [SQUARE_POLYGON]})
    fast_ball_windows = [(0.0, 0.5)]
    balls = [_region_ball(t, cx=0.95) for t in (1.0, 1.2, 1.4, 1.6)]  # one continuous crossing
    events = detect_goals_goal_region(balls, fast_ball_windows, "someslug", cfg, take_id=0)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# detect_goals_for_take -- Gap-2 fix: region-sourced goals now get full attribution,
# and lazy possession/pass computation is still skipped when there is truly nothing to attribute.
# ---------------------------------------------------------------------------


def _no_scoreboard_decode(
    video_path, fps=None, start=None, end=None, scale_width=None, use_nvdec=True
):
    for i in range(2):
        yield i, float(i), np.zeros((10, 10, 3), dtype=np.uint8)


class _EmptyReader:
    def readtext(self, crop):
        return []


def test_detect_goals_for_take_region_sourced_goal_gets_full_attribution_when_supported(
    monkeypatch,
):
    """Gap 2: a region-sourced GOAL (zero scoreboard occurrence events at all) must flow through
    the exact same team/scorer/assist attribution a scoreboard-sourced goal gets, when possession/
    pass data supports it -- not the old, deliberately-incomplete "region goals are occurrence-only"
    behaviour."""
    monkeypatch.setattr(goals_mod, "decode_frames", _no_scoreboard_decode)

    events_cfg = _events_config()
    take = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10, kind="main")
    frame_width = 1000.0
    frame_height = 1000.0

    # A fast, sustained rightward ball run ending inside the marked polygon's right-hand 15% of
    # the frame -- same synthetic trajectory as the existing goal-region end-to-end test, so it is
    # already known to fire exactly one region-sourced GOAL.
    balls = [
        BallDetection(
            bbox=BBox(x1=100.0 + i * 90.0 - 5, y1=45.0, x2=100.0 + i * 90.0 + 5, y2=55.0),
            conf=0.9,
            frame_index=i,
            t=i * 0.1,
        )
        for i in range(10)
    ]
    polygon = [(0.85, 0.0), (1.0, 0.0), (1.0, 1.0), (0.85, 1.0)]
    goal_region_cfg = {"regions": {"testslug": [polygon]}}

    take_tracks = [_track(1, team=0), _track(2, team=0)]
    # Identity 1 (team 0) makes a same-colour PASS to identity 2, who then holds the LAST
    # possession run right up to the goal -- exactly the data `attribute_goal_team_and_scorer` /
    # `find_assist_event` need to credit a real scorer + assist.
    possession_events = [
        _possession_event(identity=1, raw_track_id=1, t_start=0.3, t_end=0.6, eid="poss-1"),
        _possession_event(identity=2, raw_track_id=2, t_start=0.65, t_end=0.9, eid="poss-2"),
    ]
    pass_events = [_pass_event(passer=1, receiver=2, t_start=0.55, t_end=0.62, eid="pass-1")]
    monkeypatch.setattr(goals_mod, "possession_runs_for_take", lambda *a, **k: [])
    monkeypatch.setattr(goals_mod, "detect_possession", lambda *a, **k: possession_events)
    monkeypatch.setattr(goals_mod, "detect_passes", lambda *a, **k: pass_events)

    events, debug = goals_mod.detect_goals_for_take(
        reader=_EmptyReader(),
        video_path="fake.mp4",
        take=take,
        take_tracks=take_tracks,
        take_balls=balls,
        events_cfg=events_cfg,
        identity_of=None,
        use_nvdec=False,
        frame_width=frame_width,
        frame_height=frame_height,
        goal_region_cfg=goal_region_cfg,
        slug="testslug",
    )

    goal_events = [e for e in events if e.type == EventType.GOAL]
    assert len(goal_events) == 1
    goal = goal_events[0]
    assert goal.source == "goal_region"
    assert goal.player_track_id == 2  # attributed scorer, not left None
    region_cfg = events_cfg["goal_region"]
    goal_cfg = events_cfg["goal"]
    expected_goal_conf = region_cfg["confidence"] * goal_cfg["scorer_confidence_multiplier"]
    assert goal.confidence == expected_goal_conf
    # Confidence stacks down HONESTLY LOWER than the equivalent scoreboard-sourced case would --
    # same attribution multiplier, but starting from goal_region's own lower occurrence confidence.
    scoreboard_equivalent_conf = (
        goal_cfg["occurrence_confidence"] * goal_cfg["scorer_confidence_multiplier"]
    )
    assert goal.confidence < scoreboard_equivalent_conf

    assist_events = [e for e in events if e.type == EventType.ASSIST]
    assert len(assist_events) == 1
    assert assist_events[0].player_track_id == 1  # the passer credited with the assist


def test_detect_goals_for_take_skips_possession_computation_when_no_goals_of_either_kind(
    monkeypatch,
):
    """Cost-discipline check (the function's own docstring claim): when neither the scoreboard
    scan nor a configured goal region produces ANY goal for this take, the expensive lazy
    possession/pass computation must never run at all."""
    monkeypatch.setattr(goals_mod, "decode_frames", _no_scoreboard_decode)

    def _boom(*a, **k):
        raise AssertionError("possession_runs_for_take must not be called with zero goals")

    monkeypatch.setattr(goals_mod, "possession_runs_for_take", _boom)

    events_cfg = _events_config()
    take = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10, kind="main")

    events, debug = goals_mod.detect_goals_for_take(
        reader=_EmptyReader(),
        video_path="fake.mp4",
        take=take,
        take_tracks=[],
        take_balls=[],
        events_cfg=events_cfg,
        identity_of=None,
        use_nvdec=False,
        frame_width=None,
        frame_height=None,
        goal_region_cfg=None,
        slug=None,
    )
    assert events == []  # no exception raised -- the lazy step was correctly never invoked


# ---------------------------------------------------------------------------
# check_goal_availability -- the final Golden-Rule-5 decision table
# ---------------------------------------------------------------------------


def test_check_goal_availability_no_scan_attempted():
    result = check_goal_availability(
        scan_attempted=False, activated_any_region=False, events=[], goal_cfg=_goal_cfg()
    )
    assert result.available is False
    assert result.reason.startswith("not available")
    assert result.events == []


def test_check_goal_availability_scanned_but_no_region_activated():
    result = check_goal_availability(
        scan_attempted=True, activated_any_region=False, events=[], goal_cfg=_goal_cfg()
    )
    assert result.available is False
    assert result.reason.startswith("not available")
    assert "no legible scoreboard" in result.reason


def test_check_goal_availability_activated_but_no_increment():
    result = check_goal_availability(
        scan_attempted=True, activated_any_region=True, events=[], goal_cfg=_goal_cfg()
    )
    assert result.available is False
    assert result.reason.startswith("not available")
    assert "activated" in result.reason


def test_check_goal_availability_real_goal_found():
    goal_event = _goal_event(t_after=100.0).model_copy(update={"player_track_id": 11})
    result = check_goal_availability(
        scan_attempted=True, activated_any_region=True, events=[goal_event], goal_cfg=_goal_cfg()
    )
    assert result.available is True
    assert "1 goal(s)" in result.reason
    assert result.events == [goal_event]


# ---------------------------------------------------------------------------
# end-to-end: detect_goals_for_video with decode/OCR monkeypatched out (no GPU/ffmpeg)
# ---------------------------------------------------------------------------


def test_detect_goals_for_video_end_to_end_reports_available_with_no_team_data(monkeypatch):
    """Full wiring check: detect_goals_for_video -> detect_goals_for_take (decode+OCR
    monkeypatched) -> detect_goals_scoreboard_delta (real) -> attribute_goal_team_and_scorer
    (real, "team undetermined" branch since there are zero tracks/balls) -> find_assist_event
    (real, returns None since scorer is unknown) -> check_goal_availability (real)."""

    script = {0: "0 - 0", 1: "0 - 0", 2: "1 - 0", 3: "1 - 0", 4: "1 - 0"}

    def fake_decode_frames(
        video_path, fps=None, start=None, end=None, scale_width=None, use_nvdec=True
    ):
        for i in script:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            frame[0, 0, 0] = i
            yield i, float(i), frame

    class FakeReader:
        def readtext(self, crop):
            text = script[int(crop[0, 0, 0])]
            return [((0, 0), text, 0.9)]

    monkeypatch.setattr(goals_mod, "decode_frames", fake_decode_frames)
    monkeypatch.setattr(goals_mod, "load_easyocr_reader", lambda cfg: FakeReader())
    monkeypatch.setattr(goals_mod, "free_easyocr_reader", lambda reader: None)

    events_cfg = _events_config()
    events_cfg["goal"] = _goal_cfg(
        candidate_regions=[[0.0, 0.0, 1.0, 1.0]],
        activation_frames_count=5,
        min_stable_frames=2,
        increment_debounce_samples=2,
    )

    take = Take(id=0, t_start=0.0, t_end=4.0, frame_start=0, frame_end=4, kind="main")
    result = goals_mod.detect_goals_for_video(
        video_path="fake.mp4",
        takes=[take],
        tracks_by_take={0: []},
        balls_by_take={0: []},
        events_cfg=events_cfg,
        identity_of_by_take=None,
        ocr_cfg={"gpu": False},
        use_nvdec=False,
    )
    assert isinstance(result, GoalDetectionResult)
    assert result.available is True
    assert len(result.events) == 1
    goal = result.events[0]
    assert goal.type == EventType.GOAL
    assert goal.take_id == 0
    assert goal.player_track_id is None  # no possession data -> team/scorer undetermined


def test_detect_goals_for_video_no_takes_is_not_available():
    result = goals_mod.detect_goals_for_video(
        video_path="fake.mp4",
        takes=[],
        tracks_by_take={},
        balls_by_take={},
        events_cfg=_events_config(),
    )
    assert result.available is False
    assert result.reason.startswith("not available")


# ---------------------------------------------------------------------------
# check_goal_availability -- ADR-20 region-awareness (must not be masked by "no scoreboard")
# ---------------------------------------------------------------------------


def test_check_goal_availability_region_goal_available_even_with_no_scoreboard_activation():
    """The bug ADR-20 fixes: a video with NO scoreboard anywhere but a real region-sourced goal
    must report available=True, never masked by activated_any_region=False."""
    region_goal = Event(
        id="region-goal-1",
        type=EventType.GOAL,
        t_start=9.0,
        t_end=10.0,
        player_track_id=None,
        take_id=0,
        confidence=0.45,
        source="goal_region",
        evidence={},
    )
    result = check_goal_availability(
        scan_attempted=True,
        activated_any_region=False,
        events=[region_goal],
        goal_cfg=_goal_cfg(),
        goal_region_configured=True,
    )
    assert result.available is True
    assert "via a human-marked goal region" in result.reason


def test_check_goal_availability_neither_source_names_both_as_attempted():
    result = check_goal_availability(
        scan_attempted=True,
        activated_any_region=False,
        events=[],
        goal_cfg=_goal_cfg(),
        goal_region_configured=True,
    )
    assert result.available is False
    assert "human-marked goal region IS configured" in result.reason
    assert "no ball-crossing-a-marked-region goal was found" in result.reason


def test_check_goal_availability_no_region_configured_says_so():
    result = check_goal_availability(
        scan_attempted=True,
        activated_any_region=False,
        events=[],
        goal_cfg=_goal_cfg(),
        goal_region_configured=False,
    )
    assert "no human-marked goal region is configured" in result.reason


# ---------------------------------------------------------------------------
# detect_goals_for_video -- ADR-20 goal-region end-to-end (acceptance #4)
# ---------------------------------------------------------------------------


def test_detect_goals_for_video_goal_region_end_to_end(monkeypatch):
    """Acceptance #4: polygon configured for a test slug + a synthetic ball track crossing it
    inside the shot window -> exactly one GOAL with source="goal_region"; polygon removed ->
    goals read not available, no crash, no fabricated goal."""

    def fake_decode_frames(
        video_path, fps=None, start=None, end=None, scale_width=None, use_nvdec=True
    ):
        # no scoreboard text anywhere in this take -- purely exercising the goal-region source
        for i in range(2):
            yield i, float(i), np.zeros((10, 10, 3), dtype=np.uint8)

    class EmptyReader:
        def readtext(self, crop):
            return []

    monkeypatch.setattr(goals_mod, "decode_frames", fake_decode_frames)
    monkeypatch.setattr(goals_mod, "load_easyocr_reader", lambda cfg: EmptyReader())
    monkeypatch.setattr(goals_mod, "free_easyocr_reader", lambda reader: None)

    events_cfg = _events_config()
    take = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10, kind="main")
    frame_width = 1000.0
    frame_height = 1000.0

    # A fast, sustained rightward ball run (clears shot.ball_speed_threshold with full direction
    # consistency) ending at cx=910 -- inside the marked polygon's right-hand 15% of the frame.
    balls = [
        BallDetection(
            bbox=BBox(x1=100.0 + i * 90.0 - 5, y1=45.0, x2=100.0 + i * 90.0 + 5, y2=55.0),
            conf=0.9,
            frame_index=i,
            t=i * 0.1,
        )
        for i in range(10)
    ]
    polygon = [(0.85, 0.0), (1.0, 0.0), (1.0, 1.0), (0.85, 1.0)]
    goal_region_cfg = {**events_cfg["goal_region"], "regions": {"testslug": [polygon]}}

    result = goals_mod.detect_goals_for_video(
        video_path="fake.mp4",
        takes=[take],
        tracks_by_take={0: []},
        balls_by_take={0: balls},
        events_cfg=events_cfg,
        identity_of_by_take=None,
        ocr_cfg={"gpu": False},
        use_nvdec=False,
        frame_width=frame_width,
        frame_height=frame_height,
        goal_region_cfg=goal_region_cfg,
        slug="testslug",
    )
    assert result.available is True
    goal_events = [e for e in result.events if e.type == EventType.GOAL]
    assert len(goal_events) == 1
    assert goal_events[0].source == "goal_region"

    # polygon removed -> not available, no crash, no fabricated goal
    result_no_polygon = goals_mod.detect_goals_for_video(
        video_path="fake.mp4",
        takes=[take],
        tracks_by_take={0: []},
        balls_by_take={0: balls},
        events_cfg=events_cfg,
        identity_of_by_take=None,
        ocr_cfg={"gpu": False},
        use_nvdec=False,
        frame_width=frame_width,
        frame_height=frame_height,
        goal_region_cfg={**events_cfg["goal_region"], "regions": {}},
        slug="testslug",
    )
    assert result_no_polygon.available is False
    assert result_no_polygon.reason.startswith("not available")
    assert result_no_polygon.events == [] or all(
        e.type != EventType.GOAL for e in result_no_polygon.events
    )
