"""Pure-logic unit tests for src/track/click_reid.py -- no GPU/video I/O.

Click-anchored re-identification (owner request 2026-09-01: "keep in mind the colour and jersy
number and height") extends a human-clicked player's stitched chain across a break
`stitch_timeline` itself didn't bridge, using the clicked player's own measured team-cluster and
box-height signals, plus an optional jersey-number veto.
"""

from __future__ import annotations

from src.common.types import BBox, Track, TrackBox
from src.track.click_reid import (
    build_click_profile,
    candidate_score,
    extend_chain_with_profile,
)

REID_CFG = {"min_height_ratio": 0.60}


def _box(t: float, height: float) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=0, y1=0, x2=40, y2=height), conf=0.9)


def _track(tid: int, boxes: list[TrackBox], team: int | None = None) -> Track:
    return Track(id=tid, take_id=0, boxes=boxes, team=team)


# ---------------------------------------------------------------------------
# build_click_profile / _median_height
# ---------------------------------------------------------------------------


def test_build_click_profile_captures_team_and_median_height():
    anchor = _track(1, [_box(0.0, 100), _box(0.1, 110), _box(0.2, 90)], team=0)
    profile = build_click_profile(anchor)
    assert profile["team"] == 0
    assert profile["median_height"] == 100  # median of [90, 100, 110]


def test_build_click_profile_empty_track_has_zero_height():
    anchor = _track(1, [], team=0)
    assert build_click_profile(anchor)["median_height"] == 0.0


# ---------------------------------------------------------------------------
# candidate_score
# ---------------------------------------------------------------------------


def test_candidate_score_accepts_same_team_and_similar_height():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 105)], team=0)
    accept, evidence = candidate_score(candidate, profile, REID_CFG)
    assert accept is True
    assert evidence["team_match"] is True
    assert evidence["accepted"] is True


def test_candidate_score_rejects_different_team_cluster():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 100)], team=1)
    accept, evidence = candidate_score(candidate, profile, REID_CFG)
    assert accept is False
    assert evidence["rejected_reason"] == "different_team_cluster"


def test_candidate_score_missing_team_on_either_side_is_no_evidence_not_a_rejection():
    """A low-confidence Stage 3 team call is common on crowded footage (ADR-12) -- missing team
    data must not silently veto an otherwise-good height match."""
    anchor = _track(1, [_box(0.0, 100)], team=None)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 100)], team=1)
    accept, evidence = candidate_score(candidate, profile, REID_CFG)
    assert evidence["team_match"] is None
    assert accept is True  # height still agrees, so overall accept


def test_candidate_score_rejects_height_mismatch():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 40)], team=0)  # ratio 0.4 < 0.60 floor
    accept, evidence = candidate_score(candidate, profile, REID_CFG)
    assert accept is False
    assert evidence["rejected_reason"] == "height_mismatch"
    assert evidence["height_ratio"] == 0.4


def test_candidate_score_height_ratio_is_scale_invariant_not_absolute():
    """A player further from camera reads smaller -- that is not evidence of a different person.
    A distant anchor (30px) and an equally-distant candidate (33px) must still agree even though
    the absolute pixel gap (3px) is tiny and would look meaningless as an absolute threshold."""
    anchor = _track(1, [_box(0.0, 30)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 33)], team=0)
    accept, evidence = candidate_score(candidate, profile, REID_CFG)
    assert accept is True
    assert evidence["height_ratio"] == round(30 / 33, 3)


def test_candidate_score_jersey_agreement_is_corroborating_not_required():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 100)], team=0)
    jersey = {1: "10", 2: "10"}
    accept, evidence = candidate_score(candidate, profile, REID_CFG, jersey, anchor_track_id=1)
    assert accept is True
    assert evidence["jersey_agrees"] is True


def test_candidate_score_jersey_disagreement_is_a_hard_veto():
    """A printed number is the strongest available disambiguator between two similarly-dressed
    teammates -- a confident disagreement must override an otherwise-good colour+height match."""
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 100)], team=0)  # same team, same height -- would pass alone
    jersey = {1: "10", 2: "4"}
    accept, evidence = candidate_score(candidate, profile, REID_CFG, jersey, anchor_track_id=1)
    assert accept is False
    assert evidence["rejected_reason"] == "jersey_number_disagrees"


def test_candidate_score_jersey_missing_for_one_side_is_not_consulted():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_click_profile(anchor)
    candidate = _track(2, [_box(5.0, 100)], team=0)
    jersey = {1: "10"}  # candidate never read
    accept, evidence = candidate_score(candidate, profile, REID_CFG, jersey, anchor_track_id=1)
    assert accept is True
    assert "jersey_agrees" not in evidence


# ---------------------------------------------------------------------------
# extend_chain_with_profile
# ---------------------------------------------------------------------------


def test_extend_chain_joins_a_plausible_later_fragment():
    seed = _track(1, [_box(0.0, 100), _box(1.0, 100)], team=0)
    later = _track(2, [_box(5.0, 100), _box(6.0, 100)], team=0)  # same team+height, no time overlap
    tracks = [seed, later]
    extended, trail = extend_chain_with_profile(1, [1], tracks, REID_CFG)
    assert extended == [1, 2]
    assert any(e.get("candidate_track_id") == 2 and e.get("accepted") for e in trail)


def test_extend_chain_rejects_a_time_overlapping_fragment():
    """Two fragments overlapping in time cannot be the same physical track instance -- joining
    them would silently claim one player was in two places at once."""
    seed = _track(1, [_box(0.0, 100), _box(2.0, 100)], team=0)
    overlapping = _track(2, [_box(1.0, 100), _box(3.0, 100)], team=0)
    tracks = [seed, overlapping]
    extended, trail = extend_chain_with_profile(1, [1], tracks, REID_CFG)
    assert extended == [1]
    assert trail[0]["rejected_reason"] == "time_overlap_with_accepted"


def test_extend_chain_does_not_join_a_mismatched_fragment():
    seed = _track(1, [_box(0.0, 100)], team=0)
    different_player = _track(2, [_box(5.0, 100)], team=1)
    tracks = [seed, different_player]
    extended, trail = extend_chain_with_profile(1, [1], tracks, REID_CFG)
    assert extended == [1]
    assert trail[0]["rejected_reason"] == "different_team_cluster"


def test_extend_chain_unknown_seed_returns_input_unchanged():
    tracks = [_track(1, [_box(0.0, 100)], team=0)]
    extended, trail = extend_chain_with_profile(999, [999], tracks, REID_CFG)
    assert extended == [999]
    assert trail == []


def test_extend_chain_chains_two_separate_breaks_in_one_pass():
    seed = _track(1, [_box(0.0, 100)], team=0)
    mid_break = _track(2, [_box(5.0, 100)], team=0)
    late_break = _track(3, [_box(10.0, 100)], team=0)
    tracks = [seed, mid_break, late_break]
    extended, _trail = extend_chain_with_profile(1, [1], tracks, REID_CFG)
    assert extended == [1, 2, 3]
