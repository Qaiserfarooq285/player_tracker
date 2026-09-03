"""Stage 2 of the "streamed-gathering-treehouse" plan (strict persistent target identity, see
`/home/qaiserfarooq/.claude/plans/streamed-gathering-treehouse.md`): strict verification of a
candidate track against the persistent `TargetProfile` (`src/track/target.py`), hard-rejecting on
a contradictory hard signal BEFORE any similarity score is even computed.

**Why hard-reject-then-score, not score-then-threshold (plan §5).** A single blended similarity
score can be pulled high by two agreeing soft signals (say, height + trajectory) even when a third
signal flatly contradicts the match -- exactly the failure this plan exists to prevent (the
`chelsea_burnley_target10` click-anchored chain that pulled a claret Burnley #21 into a blue
Chelsea #10's chain despite the two teams being visually obvious, `src/track/click_reid.py`'s own
module docstring). `verify_candidate` therefore checks kit colour, then jersey number, then height
in that order -- ANY of the three failing is an immediate, final REJECT, with the measured
evidence that caused it -- and only once all three have passed does it fall through to a weighted
score. A high score can never override an earlier hard reject, because scoring never runs after
one.

**Why UNCERTAIN must never become a weak ACCEPT (plan §5/§14/§19).** `verify_candidate` returns
three decisions, not two: ACCEPT (`score >= strong_match`), REJECT (a hard reject, OR a score below
`uncertain_low`), and UNCERTAIN (`uncertain_low <= score < strong_match`). Callers (a later,
out-of-scope pipeline stage) MUST treat UNCERTAIN exactly like REJECT for the purpose of keeping
the target's own state machine at TEMPORARILY_LOST/SEARCHING -- an ambiguous read is not evidence
either way, and accepting it "because nothing else looked better" is precisely how a profile drifts
onto the wrong physical player. This module enforces that at the TYPE level (`VerdictDecision` has
three members, not two) but the "stays LOST" behaviour itself belongs to the caller.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from src.common.types import Track
from src.events.kit_colour import kit_colour_same_team
from src.track.click_reid import _median_height  # reused verbatim, not copied (plan Stage 2)
from src.track.target import KitColourSample, TargetProfile


class VerdictDecision(str, Enum):  # noqa: UP042 -- matches src/common/types.py's own enum style
    """The three possible outcomes of `verify_candidate` (plan §5)."""

    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class TargetVerdict(BaseModel):
    """The full result of one `verify_candidate` call -- `evidence` always carries the complete
    trail (every band's colour verdict, the jersey comparison, the height ratio, the trajectory
    distance, and the final weighted score breakdown) regardless of `decision`, so a REJECT or
    UNCERTAIN is exactly as traceable as an ACCEPT (Golden Rule 5)."""

    decision: VerdictDecision
    score: float
    evidence: dict = Field(default_factory=dict)


def _band_verdict(
    profile_band: tuple[float, float, float] | None,
    candidate_band: tuple[float, float, float] | None,
    kit_colour_cfg: dict,
) -> tuple[bool | None, float]:
    """One band's own `kit_colour_same_team` call, or `(None, 0.0)` when either side has no
    reading for this band at all (never a guess from a missing sample)."""
    if profile_band is None or candidate_band is None:
        return None, 0.0
    return kit_colour_same_team(np.array(profile_band), np.array(candidate_band), kit_colour_cfg)


def _combined_kit_verdict(
    profile_kit: KitColourSample | None,
    candidate_kit: KitColourSample | None,
    kit_colour_cfg: dict,
) -> tuple[bool | None, dict]:
    """Combine the three bands' own `kit_colour_same_team` verdicts into one same-kit decision,
    **torso-dominant**: torso is the least occluded, least foreshortened band on a standing or
    running player, and the one band ADR-21 already proved reads cleanly from real footage (it is
    exactly the region `src.identity.jersey_parseq.number_region` isolates for jersey-number
    reading) -- so whenever torso itself has a confident verdict (True or False), that verdict IS
    the combined one, and shorts/socks are recorded in `evidence` purely for traceability, never
    consulted to override it.

    Only when torso has NO evidence (occluded crop, insufficient kit pixels) do shorts/socks act
    as a fallback: if the ones that DO have evidence unanimously agree, that agreement is the
    combined verdict; if they disagree with each other, or neither has evidence either, the
    combined verdict is `None` (undecided) -- never resolved by a coin-flip between two
    conflicting weak signals.
    """
    evidence: dict = {}
    if profile_kit is None or candidate_kit is None:
        evidence["torso"] = evidence["shorts"] = evidence["socks"] = None
        evidence["decisive_band"] = None
        return None, evidence

    torso_same, torso_dist = _band_verdict(
        profile_kit.torso_lab, candidate_kit.torso_lab, kit_colour_cfg
    )
    shorts_same, shorts_dist = _band_verdict(
        profile_kit.shorts_lab, candidate_kit.shorts_lab, kit_colour_cfg
    )
    socks_same, socks_dist = _band_verdict(
        profile_kit.socks_lab, candidate_kit.socks_lab, kit_colour_cfg
    )
    evidence["torso"] = {"same": torso_same, "distance": round(torso_dist, 2)}
    evidence["shorts"] = {"same": shorts_same, "distance": round(shorts_dist, 2)}
    evidence["socks"] = {"same": socks_same, "distance": round(socks_dist, 2)}

    if torso_same is not None:
        evidence["decisive_band"] = "torso"
        return torso_same, evidence

    fallback = [v for v in (shorts_same, socks_same) if v is not None]
    if not fallback:
        evidence["decisive_band"] = None
        return None, evidence
    if all(v is True for v in fallback):
        evidence["decisive_band"] = "shorts_socks_agree"
        return True, evidence
    if all(v is False for v in fallback):
        evidence["decisive_band"] = "shorts_socks_agree"
        return False, evidence
    evidence["decisive_band"] = "shorts_socks_disagree"
    return None, evidence


def verify_candidate(
    profile: TargetProfile,
    candidate: Track,
    kit_by_track: dict[int, KitColourSample],
    jersey_by_track: dict[int, str],
    cfg: dict,
) -> TargetVerdict:
    """Verify whether `candidate` is plausibly the SAME physical player as `profile`'s target.

    `cfg` is `configs/target.yaml`'s own dict, with one addition the caller is responsible for
    merging in: `cfg["kit_colour"]` must be `configs/events.yaml: kit_colour` (the module that
    calibrated `same_team_max_dist`/`diff_team_min_dist` against real footage) -- see
    `configs/target.yaml`'s own header comment for why those two numbers are deliberately NOT
    duplicated into this file. `kit_by_track`/`jersey_by_track` are keyed by `candidate.id`
    (raw track id) and hold that take's already-computed per-track kit sample / confident jersey
    digit string respectively (both produced by an out-of-scope, later pipeline stage) -- a track
    missing from either dict is treated as "no evidence for this track", never a rejection.

    **Order (plan §5) -- hard rejects before any scoring:**
    1. Kit colour: `_combined_kit_verdict` reads `False` (confidently a different kit) -> REJECT
       `wrong_kit_colour`, evidence carries every band's own measured distance.
    2. Jersey: BOTH sides have a confident read AND they disagree -> REJECT `wrong_jersey_number`.
       Mirrors `src.track.click_reid.candidate_score`'s exact veto semantics: a MISSING read on
       either side is never evidence of a different player, only a genuine disagreement is.
    3. Height: `min(profile.median_height, candidate's own median height) / max(...)` below
       `cfg['min_height_ratio']` -> REJECT `height_mismatch`. Scale-invariant (a ratio, never an
       absolute pixel gap) -- a player further from camera reads smaller, which is not evidence of
       being a different person (same reasoning as `click_reid.candidate_score`).

    **Only once all three pass** does `_score_candidate` blend colour + jersey + height +
    trajectory continuity per `cfg['scoring']['weights']`, and the decision follows
    `cfg['strong_match']`/`cfg['uncertain_low']` (plan §5): `>= strong_match` -> ACCEPT;
    `[uncertain_low, strong_match)` -> **UNCERTAIN** (never a weak accept -- see this module's own
    docstring); below `uncertain_low` -> REJECT `score_below_uncertain_low`.
    """
    evidence: dict = {"candidate_track_id": candidate.id}

    candidate_kit = kit_by_track.get(candidate.id)
    kit_same, kit_evidence = _combined_kit_verdict(profile.kit, candidate_kit, cfg["kit_colour"])
    evidence["kit_colour"] = kit_evidence
    if kit_same is False:
        evidence["rejected_reason"] = "wrong_kit_colour"
        return TargetVerdict(decision=VerdictDecision.REJECT, score=0.0, evidence=evidence)

    profile_digits = str(profile.jersey_number) if profile.jersey_number is not None else None
    candidate_digits = jersey_by_track.get(candidate.id)
    jersey_agrees: bool | None = None
    if profile_digits is not None and candidate_digits is not None:
        jersey_agrees = profile_digits == candidate_digits
        evidence["jersey_agrees"] = jersey_agrees
        if not jersey_agrees:
            evidence["rejected_reason"] = "wrong_jersey_number"
            return TargetVerdict(decision=VerdictDecision.REJECT, score=0.0, evidence=evidence)
    else:
        evidence["jersey_agrees"] = None

    candidate_height = _median_height(candidate)
    height_ratio: float | None = None
    if profile.median_height > 0 and candidate_height > 0:
        height_ratio = min(profile.median_height, candidate_height) / max(
            profile.median_height, candidate_height
        )
        evidence["height_ratio"] = round(height_ratio, 3)
        if height_ratio < cfg["min_height_ratio"]:
            evidence["rejected_reason"] = "height_mismatch"
            return TargetVerdict(decision=VerdictDecision.REJECT, score=0.0, evidence=evidence)
    else:
        evidence["height_ratio"] = None

    score, score_evidence = _score_candidate(
        profile, candidate, kit_same, jersey_agrees, height_ratio, cfg
    )
    evidence["score_breakdown"] = score_evidence

    if score >= cfg["strong_match"]:
        decision = VerdictDecision.ACCEPT
    elif score >= cfg["uncertain_low"]:
        decision = VerdictDecision.UNCERTAIN
    else:
        decision = VerdictDecision.REJECT
        evidence["rejected_reason"] = "score_below_uncertain_low"

    return TargetVerdict(decision=decision, score=round(score, 4), evidence=evidence)


def _trajectory_score(profile: TargetProfile, candidate: Track, cfg: dict) -> tuple[float, dict]:
    """Continuity score in `[0, 1]` between wherever the profile's most recent `TargetLink` recorded
    its last known position and `candidate`'s own first box.

    **Judgement call, not specified by the plan:** `TargetProfile` has no dedicated "last known
    pixel position" field, so this reads `last_cx`/`last_cy`/`last_bbox_height` out of the most
    recent link's own free-form `evidence` dict -- populating those three keys when a link is
    appended is the responsibility of an out-of-scope, later pipeline stage. No link, or a link
    that never recorded a position, scores a neutral `0.5` ("no evidence either way", the same
    treatment `click_reid.candidate_score` gives a missing team/jersey signal) rather than
    penalising a candidate for a gap in bookkeeping that isn't its own fault.
    """
    if not profile.links or not candidate.boxes:
        return 0.5, {"available": False}

    last_evidence = profile.links[-1].evidence
    required = ("last_cx", "last_cy", "last_bbox_height")
    if not all(key in last_evidence for key in required):
        return 0.5, {"available": False}

    last_cx, last_cy, last_h = (last_evidence[k] for k in required)
    first_box = candidate.boxes[0].bbox
    cand_h = first_box.height
    mean_h = (last_h + cand_h) / 2.0 if (last_h > 0 and cand_h > 0) else 0.0
    if mean_h <= 0:
        return 0.5, {"available": False}

    dist_px = float(np.hypot(first_box.cx - last_cx, first_box.cy - last_cy))
    dist_bbox_heights = dist_px / mean_h
    max_plausible = cfg["trajectory"]["max_plausible_bbox_heights"]
    score = max(0.0, 1.0 - dist_bbox_heights / max_plausible)
    return score, {"available": True, "distance_bbox_heights": round(dist_bbox_heights, 3)}


def _score_candidate(
    profile: TargetProfile,
    candidate: Track,
    kit_same: bool | None,
    jersey_agrees: bool | None,
    height_ratio: float | None,
    cfg: dict,
) -> tuple[float, dict]:
    """Weighted blend of the four soft signals, reached ONLY after every hard reject has already
    passed (`kit_same` can therefore never be `False` here, and `jersey_agrees` can never be
    `False` -- both would already have returned a REJECT). A `True`/agreeing signal scores `1.0`;
    a genuinely absent one (colour undecided, jersey unread on one side, height unmeasurable)
    scores a neutral `0.5` rather than penalising the candidate for a gap in evidence that isn't
    itself contradictory (Golden Rule 5: absence of evidence is not evidence of a mismatch).
    """
    weights = cfg["scoring"]["weights"]
    kit_score = 1.0 if kit_same is True else 0.5
    jersey_score = 1.0 if jersey_agrees is True else 0.5
    height_score = height_ratio if height_ratio is not None else 0.5
    trajectory_score, trajectory_evidence = _trajectory_score(profile, candidate, cfg)

    components = {
        "kit_colour": kit_score,
        "jersey": jersey_score,
        "height": height_score,
        "trajectory": trajectory_score,
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(weights[k] * components[k] for k in components)
    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return score, {
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": dict(weights),
        "trajectory": trajectory_evidence,
    }
