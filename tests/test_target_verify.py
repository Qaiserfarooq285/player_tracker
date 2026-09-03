"""Pure-logic unit tests for `src/track/target_verify.py` (Stage 2 of the
"streamed-gathering-treehouse" plan: strict hard-reject-then-score target verification). No GPU,
no video decode.

`kit_colour_same_team` fixtures reuse the SAME real, measured CIELAB values as
`tests/test_kit_colour.py` (`chelsea_burnley_target10`, 2026-09-02) so the "known real failure must
be rejected" case (blue #10 vs claret #21, dE=24.5) is exercised with the actual measured numbers,
not an arbitrary synthetic pair.
"""

from __future__ import annotations

from pathlib import Path

from src.common.io import load_yaml
from src.common.types import BBox, Track, TrackBox
from src.track.target import KitColourSample, TargetLink, TargetProfile, TargetState
from src.track.target_verify import VerdictDecision, verify_candidate

REPO_ROOT = Path(__file__).resolve().parents[1]

# Real measured grass-suppressed CIELAB, chelsea_burnley_target10 take 0, 2026-09-02 (same values
# as tests/test_kit_colour.py).
BLUE_10 = (44.9, 7.5, -24.0)  # Chelsea #10, visually confirmed
CLARET_21 = (52.9, 5.0, -1.0)  # Burnley #21, visually confirmed -- dE from BLUE_10 is 24.5


def _cfg() -> dict:
    """`configs/target.yaml` merged with `configs/events.yaml: kit_colour`, exactly the shape
    `verify_candidate`'s own docstring documents as the caller's responsibility."""
    target_cfg = load_yaml(REPO_ROOT / "configs" / "target.yaml")
    events_cfg = load_yaml(REPO_ROOT / "configs" / "events.yaml")
    return {**target_cfg, "kit_colour": events_cfg["kit_colour"]}


def _box(t: float, height: float = 100.0) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=0, y1=0, x2=50, y2=height), conf=0.9)


def _track(tid: int, boxes: list[TrackBox], team: int | None = None) -> Track:
    return Track(id=tid, take_id=0, boxes=boxes, team=team)


def _kit(torso=None, shorts=None, socks=None) -> KitColourSample:
    return KitColourSample(
        torso_lab=torso, shorts_lab=shorts, socks_lab=socks, take_id=0, t=1.0, confidence=1.0
    )


def _profile(
    kit: KitColourSample | None = None,
    jersey_number: int | None = None,
    median_height: float = 100.0,
    links: list[TargetLink] | None = None,
) -> TargetProfile:
    return TargetProfile(
        jersey_number=jersey_number,
        jersey_source="click" if jersey_number is not None else None,
        kit=kit,
        kit_bank=[kit] if kit is not None else [],
        median_height=median_height,
        team_cluster=0,
        established_take_id=0,
        established_track_id=1,
        established_t=0.0,
        links=links or [],
    )


# ---------------------------------------------------------------------------
# Hard reject 1: kit colour, torso-dominant
# ---------------------------------------------------------------------------


def test_reject_wrong_kit_colour_reproduces_the_real_measured_failure():
    """The known real failure this whole plan exists to catch: blue #10 vs claret #21, dE=24.5,
    measured on real footage -- must be REJECTED with the real distance in evidence, not a
    replayed constant (the distance is computed at call time by `kit_colour_same_team`)."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=10)
    candidate = _track(47, [_box(0.0, 100)], team=0)
    kit_by_track = {47: _kit(torso=CLARET_21, shorts=CLARET_21, socks=CLARET_21)}
    verdict = verify_candidate(profile, candidate, kit_by_track, {}, _cfg())
    assert verdict.decision == VerdictDecision.REJECT
    assert verdict.evidence["rejected_reason"] == "wrong_kit_colour"
    assert verdict.evidence["kit_colour"]["torso"]["same"] is False
    assert (
        verdict.evidence["kit_colour"]["torso"]["distance"]
        > _cfg()["kit_colour"]["diff_team_min_dist"]
    )


def test_torso_dominates_over_disagreeing_shorts_and_socks():
    """Torso reads confidently SAME even though shorts/socks (contrived) would read confidently
    DIFFERENT -- the combined verdict must follow torso, per `_combined_kit_verdict`'s own
    documented rule, and must NOT hard-reject."""
    cfg = _cfg()
    same_max = cfg["kit_colour"]["same_team_max_dist"]
    diff_min = cfg["kit_colour"]["diff_team_min_dist"]
    assert same_max < diff_min  # sanity on the fixture itself
    torso_a = (50.0, 0.0, 0.0)
    torso_b = (50.0 + same_max * 0.5, 0.0, 0.0)  # well inside same_team_max_dist
    shorts_a = (50.0, 0.0, 0.0)
    shorts_b = (50.0 + diff_min * 1.5, 0.0, 0.0)  # well past diff_team_min_dist
    profile = _profile(kit=_kit(torso=torso_a, shorts=shorts_a, socks=shorts_a), jersey_number=None)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=torso_b, shorts=shorts_b, socks=shorts_b)}
    verdict = verify_candidate(profile, candidate, kit_by_track, {}, cfg)
    assert verdict.decision != VerdictDecision.REJECT or "wrong_kit_colour" not in str(
        verdict.evidence.get("rejected_reason")
    )
    assert verdict.evidence["kit_colour"]["decisive_band"] == "torso"


def test_shorts_socks_fallback_when_torso_has_no_evidence():
    """Torso missing on one side -- shorts/socks both confidently agree SAME -- combined verdict
    must fall back to that agreement rather than reporting no evidence at all."""
    cfg = _cfg()
    profile = _profile(kit=_kit(torso=None, shorts=BLUE_10, socks=BLUE_10), jersey_number=None)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=None, shorts=BLUE_10, socks=BLUE_10)}
    verdict = verify_candidate(profile, candidate, kit_by_track, {}, cfg)
    assert verdict.evidence["kit_colour"]["decisive_band"] == "shorts_socks_agree"
    assert verdict.decision != VerdictDecision.REJECT


def test_no_kit_evidence_anywhere_is_not_a_rejection():
    """Neither side has ANY kit sample at all -- absence of evidence, not evidence of a mismatch --
    must not hard-reject on kit colour."""
    profile = _profile(kit=None, jersey_number=None)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    assert verdict.evidence.get("rejected_reason") != "wrong_kit_colour"
    assert verdict.evidence["kit_colour"]["decisive_band"] is None


# ---------------------------------------------------------------------------
# Hard reject 2: jersey number
# ---------------------------------------------------------------------------


def test_reject_wrong_jersey_number_overrides_good_colour_match():
    """A confident jersey disagreement must hard-reject even when kit colour matches perfectly --
    mirrors `click_reid.candidate_score`'s own veto semantics exactly."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=10)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    jersey_by_track = {2: "4"}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, _cfg())
    assert verdict.decision == VerdictDecision.REJECT
    assert verdict.evidence["rejected_reason"] == "wrong_jersey_number"


def test_missing_jersey_read_on_candidate_is_not_a_rejection():
    """The profile has a confident jersey number but the candidate has no confident read at all --
    absence of evidence must never veto an otherwise-good match."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=10)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    verdict = verify_candidate(profile, candidate, kit_by_track, {}, _cfg())
    assert verdict.evidence.get("rejected_reason") != "wrong_jersey_number"
    assert verdict.evidence["jersey_agrees"] is None


def test_missing_jersey_read_on_profile_is_not_a_rejection():
    """The profile itself has no jersey number recorded at all -- also not evidence of a
    mismatch."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=None)
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    jersey_by_track = {2: "10"}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, _cfg())
    assert verdict.evidence.get("rejected_reason") != "wrong_jersey_number"
    assert verdict.evidence["jersey_agrees"] is None


def test_jersey_agreement_corroborates_and_can_accept():
    profile = _profile(
        kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10),
        jersey_number=10,
        median_height=100.0,
    )
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    jersey_by_track = {2: "10"}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, _cfg())
    assert verdict.evidence["jersey_agrees"] is True
    assert verdict.decision == VerdictDecision.ACCEPT


# ---------------------------------------------------------------------------
# Hard reject 3: height ratio, scale-invariant
# ---------------------------------------------------------------------------


def test_reject_height_mismatch():
    profile = _profile(kit=None, jersey_number=None, median_height=100.0)
    candidate = _track(2, [_box(0.0, 40)], team=0)  # ratio 0.4 < 0.60 floor
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    assert verdict.decision == VerdictDecision.REJECT
    assert verdict.evidence["rejected_reason"] == "height_mismatch"
    assert verdict.evidence["height_ratio"] == 0.4


def test_height_ratio_is_scale_invariant():
    """A distant profile (30px) and an equally-distant candidate (33px) must still pass even
    though the absolute pixel gap is tiny -- the ratio, not the absolute difference, is what's
    tested."""
    profile = _profile(kit=None, jersey_number=None, median_height=30.0)
    candidate = _track(2, [_box(0.0, 33)], team=0)
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    assert verdict.evidence.get("rejected_reason") != "height_mismatch"
    assert verdict.evidence["height_ratio"] == round(30 / 33, 3)


def test_missing_height_data_is_not_a_rejection():
    profile = _profile(kit=None, jersey_number=None, median_height=0.0)
    candidate = _track(2, [], team=0)  # no boxes -> zero median height
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    assert verdict.evidence.get("rejected_reason") != "height_mismatch"
    assert verdict.evidence["height_ratio"] is None


# ---------------------------------------------------------------------------
# Scoring / decision bands -- ACCEPT / UNCERTAIN / REJECT
# ---------------------------------------------------------------------------


def test_full_agreement_on_every_signal_is_a_confident_accept():
    profile = _profile(
        kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10),
        jersey_number=10,
        median_height=100.0,
    )
    candidate = _track(2, [_box(0.0, 100)], team=0)
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    jersey_by_track = {2: "10"}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, _cfg())
    assert verdict.decision == VerdictDecision.ACCEPT
    assert verdict.score >= _cfg()["strong_match"]


def test_no_evidence_anywhere_lands_in_reject_not_a_weak_accept():
    """Every soft signal is neutral (0.5) when there's no evidence anywhere -- the resulting score
    must sit BELOW `uncertain_low`, so a candidate with literally nothing corroborating it is
    REJECTED, never accepted or left uncertain, by construction of the weights."""
    cfg = _cfg()
    profile = _profile(kit=None, jersey_number=None, median_height=0.0)
    candidate = _track(2, [], team=0)
    verdict = verify_candidate(profile, candidate, {}, {}, cfg)
    assert verdict.score == 0.5
    assert verdict.score < cfg["uncertain_low"]
    assert verdict.decision == VerdictDecision.REJECT
    assert verdict.evidence["rejected_reason"] == "score_below_uncertain_low"


def test_uncertain_band_is_returned_not_a_weak_accept():
    """Construct a score that lands strictly between `uncertain_low` and `strong_match` (kit
    colour matches + jersey agrees, height/trajectory neutral -- no boxes/link history to measure
    either) and confirm it comes back UNCERTAIN, not ACCEPT -- the plan's own core requirement
    (§5/§14/§19: UNCERTAIN must never become a weak accept)."""
    cfg = _cfg()
    weights = cfg["scoring"]["weights"]
    # kit_colour=1.0, jersey=1.0, height/trajectory neutral 0.5 each ->
    # score = 0.5 + 0.5*(weights['kit_colour'] + weights['jersey'])
    expected_score = 0.5 + 0.5 * (weights["kit_colour"] + weights["jersey"])
    assert (
        cfg["uncertain_low"] <= expected_score < cfg["strong_match"]
    ), "fixture assumption broken -- adjust the profile/candidate below if config weights change"
    profile = _profile(
        kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10),
        jersey_number=10,
        median_height=0.0,  # -> height_ratio can't be computed -> neutral, not measured
        links=[],  # -> no trajectory history -> neutral, not measured
    )
    candidate = _track(2, [], team=0)  # no boxes -> its own median height is also 0.0
    kit_by_track = {2: _kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10)}
    jersey_by_track = {2: "10"}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, cfg)
    assert verdict.score == round(expected_score, 4)
    assert verdict.decision == VerdictDecision.UNCERTAIN
    assert "rejected_reason" not in verdict.evidence


def test_evidence_always_present_regardless_of_decision():
    """Golden Rule 5: a REJECT/UNCERTAIN must carry exactly as much evidence as an ACCEPT."""
    profile = _profile(kit=_kit(torso=BLUE_10, shorts=BLUE_10, socks=BLUE_10), jersey_number=10)
    candidate = _track(47, [_box(0.0, 100)], team=0)
    kit_by_track = {47: _kit(torso=CLARET_21, shorts=CLARET_21, socks=CLARET_21)}
    verdict = verify_candidate(profile, candidate, kit_by_track, {}, _cfg())
    assert verdict.decision == VerdictDecision.REJECT
    assert "kit_colour" in verdict.evidence
    assert "candidate_track_id" in verdict.evidence
    assert verdict.evidence["candidate_track_id"] == 47


# ---------------------------------------------------------------------------
# Trajectory continuity (judgement-call feature: reads last_cx/last_cy/last_bbox_height out of
# the most recent TargetLink's own evidence dict)
# ---------------------------------------------------------------------------


def test_trajectory_no_link_history_is_neutral_not_penalised():
    profile = _profile(kit=None, jersey_number=None, median_height=0.0, links=[])
    candidate = _track(2, [_box(0.0, 100)], team=0)
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    assert verdict.evidence["score_breakdown"]["trajectory"]["available"] is False


def test_trajectory_close_continuation_scores_high():
    link = TargetLink(
        take_id=0,
        track_ids=[1],
        state=TargetState.ACTIVE,
        confidence=1.0,
        evidence={"last_cx": 25.0, "last_cy": 50.0, "last_bbox_height": 100.0},
        t_start=0.0,
        t_end=1.0,
    )
    profile = _profile(kit=None, jersey_number=None, median_height=0.0, links=[link])
    candidate = _track(2, [_box(0.0, 100)], team=0)  # bbox x1=0,y1=0,x2=50,y2=100 -> centre (25,50)
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    traj = verdict.evidence["score_breakdown"]["trajectory"]
    assert traj["available"] is True
    assert traj["distance_bbox_heights"] == 0.0


def test_trajectory_far_jump_scores_low_but_is_not_a_hard_reject():
    link = TargetLink(
        take_id=0,
        track_ids=[1],
        state=TargetState.ACTIVE,
        confidence=1.0,
        evidence={"last_cx": 5000.0, "last_cy": 5000.0, "last_bbox_height": 100.0},
        t_start=0.0,
        t_end=1.0,
    )
    profile = _profile(kit=None, jersey_number=None, median_height=0.0, links=[link])
    candidate = _track(2, [_box(0.0, 100)], team=0)
    verdict = verify_candidate(profile, candidate, {}, {}, _cfg())
    traj = verdict.evidence["score_breakdown"]["trajectory"]
    assert traj["available"] is True
    assert traj["distance_bbox_heights"] > _cfg()["trajectory"]["max_plausible_bbox_heights"]
    # a bad trajectory score alone (one of four soft signals) must not itself be a REJECT reason
    # other than the ordinary low-score path (there is no dedicated trajectory hard-reject)
    assert verdict.evidence.get("rejected_reason") in (None, "score_below_uncertain_low")
