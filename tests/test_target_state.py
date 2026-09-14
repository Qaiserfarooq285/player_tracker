"""Stage F of the "streamed-gathering-treehouse" plan: tests for the spec's own scenarios (pure,
no GPU, no video I/O).

Covers `src/track/target_state.py` (Stage A's per-frame state machine) directly, plus the two
wiring points in `src/pipeline/run.py` and `src/highlights/selection.py` that feed it (Stage B):
`TakeSelection.verified_track_ids` (the "Candidate != Target" accepted-fragment set) and
`_accepted_target_ids` (the single helper that makes box-gate and stats-gate consume the identical
set, per the plan's own explicit Stage D requirement).
"""

from __future__ import annotations

from pathlib import Path

from src.common.io import load_yaml
from src.common.types import BBox, DetectionClass, Take, Track, TrackBox
from src.highlights.selection import TakeSelection, select_targets
from src.pipeline.run import _accepted_target_ids
from src.track.target import KitColourSample, TargetProfile
from src.track.target_state import (
    TargetFrameState,
    _bbox_iou,
    _strict_target_bbox_at,
    build_target_timeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------


def _frame_state_cfg(**overrides) -> dict:
    """Small, test-friendly numbers (NOT `configs/target.yaml`'s real ones -- those are tuned for
    a 25fps real tracking cadence, not a hand-built integer-second grid) so gap/threshold
    arithmetic is easy to reason about by eye in each test."""
    base = {
        "max_sample_age_s": 0.5,
        "occlusion_iou": 0.10,
        "partial_occlusion_iou": 0.15,
        "search_grace_s": 2.0,
        "reconnect_hold_s": 1.5,
    }
    base.update(overrides)
    return base


def _box(t: float, bbox: BBox, conf: float = 0.9) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=bbox, conf=conf)


def _track(tid: int, take_id: int, boxes: list[TrackBox]) -> Track:
    return Track(id=tid, take_id=take_id, boxes=boxes)


_TARGET_BOX = BBox(x1=0, y1=0, x2=10, y2=10)  # a stationary target -- keeps overlap arithmetic
# trivial (iou against itself is always 1.0) across every test below.


def _take(t_end: float = 10.0) -> Take:
    return Take(id=0, t_start=0.0, t_end=t_end, frame_start=0, frame_end=int(t_end * 25))


# ---------------------------------------------------------------------------
# The plan's own absolute rule: bbox is None in every OCCLUDED/LOST/SEARCHING entry
# ---------------------------------------------------------------------------


def test_bbox_is_always_none_in_occluded_lost_searching():
    """Owner's own words: "when choosing between (A) no target box or (B) a target box on a
    possibly wrong player, ALWAYS choose (A)". Sweep a timeline that visits every non-visible
    state and assert the absolute invariant directly."""
    target = _track(1, 0, [_box(0.0, _TARGET_BOX), _box(1.0, _TARGET_BOX)])
    occluder = _track(
        2, 0, [_box(2.0, _TARGET_BOX), _box(3.0, _TARGET_BOX)]
    )  # full overlap -> OCCLUDED
    other = _track(
        3, 0, [_box(4.0, BBox(x1=500, y1=500, x2=510, y2=510))]
    )  # no overlap -> LOST/SEARCHING
    take = _take()
    timeline = build_target_timeline(take, [1], [target, occluder, other], _frame_state_cfg())

    non_visible_states = {
        TargetFrameState.OCCLUDED,
        TargetFrameState.LOST,
        TargetFrameState.SEARCHING,
    }
    seen_states = {s.state for s in timeline}
    assert (
        non_visible_states & seen_states
    ), "fixture should exercise at least one non-visible state"
    for status in timeline:
        if status.state in non_visible_states:
            assert status.bbox is None
            assert status.tracker_id is None
            assert status.identity_confidence == 0.0
            assert status.reason  # always populated, Golden Rule 5


# ---------------------------------------------------------------------------
# Scenario 1: occluder crosses in front => OCCLUDED, bbox is None
# ---------------------------------------------------------------------------


def test_occluder_crosses_in_front_yields_occluded_with_no_box():
    target = _track(1, 0, [_box(0.0, _TARGET_BOX), _box(1.0, _TARGET_BOX), _box(4.0, _TARGET_BOX)])
    occluder = _track(
        2, 0, [_box(2.0, _TARGET_BOX), _box(3.0, _TARGET_BOX)]
    )  # exact same box -> iou=1.0
    take = _take()
    timeline = build_target_timeline(take, [1], [target, occluder], _frame_state_cfg())
    by_t = {s.t: s for s in timeline}

    assert by_t[0.0].state == TargetFrameState.VISIBLE
    assert by_t[1.0].state == TargetFrameState.VISIBLE
    for t in (2.0, 3.0):
        assert by_t[t].state == TargetFrameState.OCCLUDED
        assert by_t[t].bbox is None
        assert by_t[t].tracker_id is None
        assert "track=2" in by_t[t].reason


# ---------------------------------------------------------------------------
# Scenario 2: tracker ID changes mid-occlusion => persistent id unchanged, no box on the new id
# ---------------------------------------------------------------------------


def test_id_switch_onto_an_unverified_track_never_gets_a_box():
    """A ByteTrack lost-track re-association would put a NEW raw id (99) right where the target
    used to be. That id was never independently verified (never in `accepted_fragments`), so it
    must never receive the target's own box -- the plan's headline failure this whole stage
    exists to prevent."""
    target = _track(1, 0, [_box(0.0, _TARGET_BOX), _box(1.0, _TARGET_BOX)])
    hijacker = _track(  # geometrically PERFECT continuation -- but never verified
        99, 0, [_box(2.0, _TARGET_BOX), _box(3.0, _TARGET_BOX), _box(4.0, _TARGET_BOX)]
    )
    take = _take()
    timeline = build_target_timeline(take, [1], [target, hijacker], _frame_state_cfg())

    tracker_ids_used = {s.tracker_id for s in timeline if s.tracker_id is not None}
    assert 99 not in tracker_ids_used
    assert tracker_ids_used <= {1}
    for t in (2.0, 3.0, 4.0):
        status = next(s for s in timeline if s.t == t)
        assert status.bbox is None  # never a box on the hijacked id


# ---------------------------------------------------------------------------
# Scenario 3: camera cut => LOST, no box on anyone
# ---------------------------------------------------------------------------


def test_camera_cut_with_no_accepted_fragment_is_lost_for_everyone():
    """A fresh take (post-cut) where nothing was ever verified this take (`target_lost` --
    `accepted_fragments=[]`) must show bbox=None throughout, for EVERY track present -- and never
    OCCLUDED (there is no "last known position" yet for anything to overlap)."""
    other_player = _track(5, 0, [_box(0.0, _TARGET_BOX), _box(1.0, _TARGET_BOX)])
    take = _take(t_end=2.0)
    timeline = build_target_timeline(take, [], [other_player], _frame_state_cfg())

    assert timeline  # the grid still exists (other tracks provide it)
    for status in timeline:
        assert status.bbox is None
        assert status.tracker_id is None
        assert status.state in (TargetFrameState.LOST, TargetFrameState.SEARCHING)
        assert status.state != TargetFrameState.OCCLUDED


def test_empty_take_tracks_yields_empty_timeline():
    take = _take()
    assert build_target_timeline(take, [], [], _frame_state_cfg()) == []


# ---------------------------------------------------------------------------
# Scenario 4: unverified candidate never becomes the target
# ---------------------------------------------------------------------------


def test_unverified_candidate_never_becomes_the_target_even_if_present_in_take_tracks():
    verified = _track(1, 0, [_box(0.0, _TARGET_BOX)])
    unverified_candidate = _track(  # NOT in accepted_fragments -- e.g. a plausible but unverified
        2, 0, [_box(0.0, BBox(x1=100, y1=100, x2=110, y2=110))]  # kit/height match, never verified
    )
    take = _take(t_end=1.0)
    timeline = build_target_timeline(
        take, [1], [verified, unverified_candidate], _frame_state_cfg()
    )
    assert all(s.tracker_id != 2 for s in timeline)


# ---------------------------------------------------------------------------
# Scenario 5 (RECONNECTED): only after a gap that actually reached SEARCHING
# ---------------------------------------------------------------------------


def test_short_gap_never_reaching_searching_reconnects_silently_as_visible():
    """search_grace_s=2.0 -- a single-instant, 1-second gap never escalates to SEARCHING, so
    regaining the target right after it must be plain VISIBLE, not RECONNECTED."""
    target = _track(1, 0, [_box(0.0, _TARGET_BOX), _box(1.0, _TARGET_BOX), _box(3.0, _TARGET_BOX)])
    filler = _track(9, 0, [_box(2.0, BBox(x1=900, y1=900, x2=910, y2=910))])  # populates the grid
    take = _take(t_end=4.0)
    cfg = _frame_state_cfg(search_grace_s=2.0)
    timeline = build_target_timeline(take, [1], [target, filler], cfg)
    by_t = {s.t: s for s in timeline}

    assert by_t[2.0].state == TargetFrameState.LOST  # gap=1.0s < search_grace_s=2.0
    assert by_t[3.0].state == TargetFrameState.VISIBLE  # NOT reconnected -- never searched


def test_gap_reaching_searching_then_reconnects_and_reverts_after_hold():
    target = _track(
        1,
        0,
        [
            _box(0.0, _TARGET_BOX),
            _box(1.0, _TARGET_BOX),
            _box(5.0, _TARGET_BOX),
            _box(6.0, _TARGET_BOX),
            _box(7.0, _TARGET_BOX),
        ],
    )
    filler = _track(
        9,
        0,
        [_box(t, BBox(x1=900, y1=900, x2=910, y2=910)) for t in (2.0, 3.0, 4.0)],
    )
    take = _take(t_end=8.0)
    cfg = _frame_state_cfg(search_grace_s=2.0, reconnect_hold_s=1.5)
    timeline = build_target_timeline(take, [1], [target, filler], cfg)
    by_t = {s.t: s for s in timeline}

    assert by_t[2.0].state == TargetFrameState.LOST  # gap=1.0 < 2.0
    assert by_t[3.0].state == TargetFrameState.SEARCHING  # gap=2.0 >= 2.0
    assert by_t[4.0].state == TargetFrameState.SEARCHING  # gap=3.0
    assert by_t[5.0].state == TargetFrameState.RECONNECTED  # first real sample back
    assert by_t[5.0].bbox is not None
    assert by_t[6.0].state == TargetFrameState.RECONNECTED  # 6.0 <= 5.0 + 1.5
    assert by_t[7.0].state == TargetFrameState.VISIBLE  # 7.0 > 6.5 -- hold has expired


# ---------------------------------------------------------------------------
# Scenario 6: a stale sample beyond max_sample_age_s yields no box, not an old one
# ---------------------------------------------------------------------------


def test_stale_sample_beyond_max_gap_yields_no_box():
    target = _track(1, 0, [_box(0.0, _TARGET_BOX), _box(5.0, _TARGET_BOX)])  # 5s gap
    filler = _track(9, 0, [_box(2.0, BBox(x1=900, y1=900, x2=910, y2=910))])  # grid point mid-gap
    take = _take(t_end=6.0)
    cfg = _frame_state_cfg(max_sample_age_s=0.5)
    timeline = build_target_timeline(take, [1], [target, filler], cfg)
    status_at_2 = next(s for s in timeline if s.t == 2.0)
    assert status_at_2.bbox is None  # NOT the t=0.0 box, stale by 2 seconds
    assert status_at_2.tracker_id is None


def test_strict_target_bbox_at_bridges_only_within_max_gap():
    samples = [(0.0, _TARGET_BOX, 1), (0.3, _TARGET_BOX, 1)]
    ts = [0.0, 0.3]
    bbox, tid = _strict_target_bbox_at(samples, ts, 0.15, max_gap_s=0.5)
    assert bbox is not None and tid == 1  # bridged: gap=0.3 <= 0.5

    bbox2, tid2 = _strict_target_bbox_at(samples, ts, 0.15, max_gap_s=0.2)
    assert bbox2 is None and tid2 is None  # NOT bridged: gap=0.3 > 0.2


def test_strict_target_bbox_at_exact_match_short_circuits():
    samples = [(0.0, _TARGET_BOX, 7)]
    ts = [0.0]
    bbox, tid = _strict_target_bbox_at(samples, ts, 0.0, max_gap_s=0.01)
    assert bbox == _TARGET_BOX
    assert tid == 7


def test_strict_target_bbox_at_empty_is_none():
    assert _strict_target_bbox_at([], [], 1.0, max_gap_s=1.0) == (None, None)


# ---------------------------------------------------------------------------
# Partial occlusion: box still drawn while overlapped
# ---------------------------------------------------------------------------


def test_partial_occlusion_keeps_the_box_but_flags_it():
    target = _track(1, 0, [_box(0.0, _TARGET_BOX)])
    overlapper = _track(2, 0, [_box(0.0, BBox(x1=5, y1=0, x2=15, y2=10))])  # ~1/3 overlap
    take = _take(t_end=1.0)
    cfg = _frame_state_cfg(partial_occlusion_iou=0.10)
    timeline = build_target_timeline(take, [1], [target, overlapper], cfg)
    status = timeline[0]
    assert status.state == TargetFrameState.PARTIALLY_OCCLUDED
    assert status.bbox is not None  # still drawn -- a real accepted sample backs it
    assert status.tracker_id == 1


def test_low_overlap_below_partial_threshold_is_plain_visible():
    target = _track(1, 0, [_box(0.0, _TARGET_BOX)])
    barely_touching = _track(2, 0, [_box(0.0, BBox(x1=9.9, y1=0, x2=20, y2=10))])
    take = _take(t_end=1.0)
    cfg = _frame_state_cfg(partial_occlusion_iou=0.5)  # deliberately high bar
    timeline = build_target_timeline(take, [1], [target, barely_touching], cfg)
    assert timeline[0].state == TargetFrameState.VISIBLE


# ---------------------------------------------------------------------------
# _bbox_iou -- small pure helper, sanity-checked directly
# ---------------------------------------------------------------------------


def test_bbox_iou_identical_boxes_is_one():
    assert _bbox_iou(_TARGET_BOX, _TARGET_BOX) == 1.0


def test_bbox_iou_disjoint_boxes_is_zero():
    other = BBox(x1=1000, y1=1000, x2=1010, y2=1010)
    assert _bbox_iou(_TARGET_BOX, other) == 0.0


# ---------------------------------------------------------------------------
# Stage D: box-gate and stats-gate consume the IDENTICAL accepted set
# ---------------------------------------------------------------------------


def test_accepted_target_ids_profile_active_uses_verified_subset_only():
    """`src.pipeline.run._accepted_target_ids` is the SAME helper `run_pipeline_for_video` calls
    both to build the box-gating timeline (`build_target_timeline`'s own `accepted_fragments`
    argument) and to attribute stats (`attribute_events_to_target`'s own `location_track_ids`
    argument) -- asserting its behaviour here covers both call sites by construction (they are
    literally the same function call in `src/pipeline/run.py`)."""
    sel = TakeSelection(
        take_id=0,
        method="target_reidentified",
        seed_track_id=5,
        track_ids=[5, 6, 7],  # the full geometry-stitched chain
        confidence=0.9,
        coverage_seconds=3.0,
        take_duration_seconds=10.0,
        verified_track_ids=[5],  # only the winner independently passed verify_candidate
    )
    assert _accepted_target_ids(sel, target_profile_active=True) == [5]


def test_accepted_target_ids_profile_inactive_falls_back_to_full_chain():
    """Every profile-LESS method (arrow_vote/heuristic_fallback/manual_override without a
    profile) has no verified/candidate distinction at all -- must reproduce the exact pre-existing
    behaviour (the full stitched chain), unchanged by this plan."""
    sel = TakeSelection(
        take_id=0,
        method="arrow_vote",
        seed_track_id=5,
        track_ids=[5, 6, 7],
        confidence=0.5,
        coverage_seconds=3.0,
        take_duration_seconds=10.0,
        vote_share=0.8,
    )
    assert _accepted_target_ids(sel, target_profile_active=False) == [5, 6, 7]


def test_accepted_target_ids_target_lost_is_empty():
    sel = TakeSelection(
        take_id=0,
        method="target_lost",
        seed_track_id=None,
        track_ids=[],
        confidence=0.0,
        coverage_seconds=0.0,
        take_duration_seconds=10.0,
    )
    assert _accepted_target_ids(sel, target_profile_active=True) == []


# ---------------------------------------------------------------------------
# Stage B: `select_targets` (profile-driven) actually populates `verified_track_ids` --
# "Candidate != Target" starts here, not just inside `build_target_timeline`.
# ---------------------------------------------------------------------------


BLUE_10 = (44.9, 7.5, -24.0)  # same real measured CIELAB as tests/test_target_verify.py


def _target_cfg() -> dict:
    target_cfg = load_yaml(REPO_ROOT / "configs" / "target.yaml")
    events_cfg = load_yaml(REPO_ROOT / "configs" / "events.yaml")
    return {**target_cfg, "kit_colour": events_cfg["kit_colour"]}


def _selection_cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "highlights.yaml")["selection"]


def _kit(lab) -> KitColourSample:
    return KitColourSample(
        torso_lab=lab, shorts_lab=lab, socks_lab=lab, take_id=0, t=0.0, confidence=1.0
    )


def _profile() -> TargetProfile:
    return TargetProfile(
        jersey_number=10,  # matched by the seed's own confident read in the ACCEPT test below --
        # needed to clear `strong_match` (kit alone lands in UNCERTAIN territory, 0.775 < 0.90).
        jersey_source="click",
        kit=_kit(BLUE_10),
        kit_bank=[_kit(BLUE_10)],
        median_height=100.0,
        team_cluster=0,
        established_take_id=0,
        established_track_id=1,
        established_t=0.0,
        links=[],
    )


def _player_track(tid: int, boxes: list[TrackBox]) -> Track:
    return Track(id=tid, take_id=0, boxes=boxes, dominant_class=DetectionClass.PLAYER, team=0)


def test_select_targets_with_profile_manual_override_accept_sets_verified_ids_to_seed_only():
    seed = _player_track(1, [_box(0.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    # a second fragment close enough in time/space that stitch_timeline WILL geometrically join it
    # -- this is exactly the "candidate, never verified" fragment the plan's rule is about.
    joined = _player_track(2, [_box(0.2, BBox(x1=5, y1=0, x2=55, y2=100))])
    take = _take(t_end=1.0)
    kit_by_track = {1: _kit(BLUE_10), 2: _kit(BLUE_10)}

    result = select_targets(
        "fake.mp4",
        None,
        [take],
        [seed, joined],
        [],
        1920.0,
        1080.0,
        _selection_cfg(),
        # "streamed-gathering-treehouse" plan Stage 1: `select_targets`'s own `manual_overrides`
        # shape is now `dict[int, list[int]]` (one or more click anchors per take) -- a
        # single-anchor take is simply a one-element list.
        manual_overrides={0: [1]},
        target_profile=_profile(),
        target_cfg=_target_cfg(),
        kit_by_track_by_take={0: kit_by_track},
        jersey_by_track_by_take={0: {1: "10"}},
    )
    sel = result.takes[0]
    assert sel.method == "manual_override"
    assert 2 in sel.track_ids  # geometry DID stitch it in
    assert sel.verified_track_ids == [1]  # but only the click itself is "accepted"


def test_select_targets_with_profile_no_accept_is_target_lost_with_empty_verified_ids():
    candidate = _player_track(3, [_box(0.0, BBox(x1=500, y1=500, x2=510, y2=560))])  # tiny -> low
    # height ratio (60/100=0.6 borderline) AND wrong kit colour -> hard reject
    take = _take(t_end=1.0)
    claret = (52.9, 5.0, -1.0)
    kit_by_track = {3: _kit(claret)}

    result = select_targets(
        "fake.mp4",
        None,
        [take],
        [candidate],
        [],
        1920.0,
        1080.0,
        _selection_cfg(),
        target_profile=_profile(),
        target_cfg=_target_cfg(),
        kit_by_track_by_take={0: kit_by_track},
        jersey_by_track_by_take={0: {}},
    )
    sel = result.takes[0]
    assert sel.method == "target_lost"
    assert sel.verified_track_ids == []
    assert sel.track_ids == []


# ---------------------------------------------------------------------------
# "streamed-gathering-treehouse" plan Stage 1: multiple click anchors in one take (the player
# left frame and came back, giving the tracker a NEW track id -- one click can no longer cover
# the whole take, see the plan's own Context section).
# ---------------------------------------------------------------------------

CLARET_21 = (52.9, 5.0, -1.0)  # same measured Burnley #21 CIELAB as tests/test_click_target.py


def test_select_take_with_profile_two_seeds_both_accepted_unions_chains():
    """Two anchors in the SAME take, far enough apart in time that `stitch_timeline` never joins
    them on its own (5s gap vs. `stitch_max_gap_s`'s 1.5s) -- exactly the "left frame, came back,
    got a new track id" scenario. Both independently pass `verify_candidate` -> their chains are
    unioned (deduped, time-sorted) into ONE `track_ids`, both seeds land in `verified_track_ids`,
    and `seed_track_id` is the FIRST accepted one (chronologically first here too)."""
    seed1 = _player_track(1, [_box(0.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    seed2 = _player_track(5, [_box(5.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    take = _take(t_end=10.0)
    kit_by_track = {1: _kit(BLUE_10), 5: _kit(BLUE_10)}
    jersey_by_track = {1: "10", 5: "10"}

    result = select_targets(
        "fake.mp4",
        None,
        [take],
        [seed1, seed2],
        [],
        1920.0,
        1080.0,
        _selection_cfg(),
        manual_overrides={0: [1, 5]},
        target_profile=_profile(),
        target_cfg=_target_cfg(),
        kit_by_track_by_take={0: kit_by_track},
        jersey_by_track_by_take={0: jersey_by_track},
    )
    sel = result.takes[0]
    assert sel.method == "manual_override"
    assert sel.track_ids == [1, 5]  # unioned, time-sorted
    assert sel.verified_track_ids == [1, 5]  # both anchors independently verified
    assert sel.seed_track_id == 1  # first accepted, chronologically first here too


def test_select_take_with_profile_one_accepted_one_rejected_keeps_only_accepted_chain():
    """A second click that lands on a visually different player (wrong kit colour, same "Burnley
    claret vs. Chelsea blue" measured pair the rest of this plan uses) must be REJECTED and
    excluded -- never allowed to widen the accepted chain ("Candidate != Target")."""
    good = _player_track(1, [_box(0.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    bad = _player_track(7, [_box(5.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    take = _take(t_end=10.0)
    kit_by_track = {1: _kit(BLUE_10), 7: _kit(CLARET_21)}
    jersey_by_track = {1: "10", 7: "21"}

    result = select_targets(
        "fake.mp4",
        None,
        [take],
        [good, bad],
        [],
        1920.0,
        1080.0,
        _selection_cfg(),
        manual_overrides={0: [1, 7]},
        target_profile=_profile(),
        target_cfg=_target_cfg(),
        kit_by_track_by_take={0: kit_by_track},
        jersey_by_track_by_take={0: jersey_by_track},
    )
    sel = result.takes[0]
    assert sel.method == "manual_override"
    assert sel.track_ids == [1]  # the rejected fragment never joins
    assert sel.verified_track_ids == [1]
    assert sel.seed_track_id == 1
    assert sel.evidence["seed_evidence"][7]["decision"] == "reject"
    assert sel.evidence["seed_evidence"][7]["rejected_reason"] == "wrong_kit_colour"


def test_select_take_with_profile_no_seed_accepted_is_target_lost():
    """Every anchor for this take fails verification (one wrong-kit REJECT, one whose track id
    isn't even present in this take's tracks) -> `target_lost`, never a substitute player, exactly
    the single-seed rule extended to N seeds."""
    bad = _player_track(7, [_box(0.0, BBox(x1=0, y1=0, x2=50, y2=100))])
    take = _take(t_end=10.0)
    kit_by_track = {7: _kit(CLARET_21)}
    jersey_by_track = {7: "21"}

    result = select_targets(
        "fake.mp4",
        None,
        [take],
        [bad],
        [],
        1920.0,
        1080.0,
        _selection_cfg(),
        manual_overrides={0: [7, 99]},  # 99 is not in this take's tracks at all
        target_profile=_profile(),
        target_cfg=_target_cfg(),
        kit_by_track_by_take={0: kit_by_track},
        jersey_by_track_by_take={0: jersey_by_track},
    )
    sel = result.takes[0]
    assert sel.method == "target_lost"
    assert sel.track_ids == []
    assert sel.verified_track_ids == []
    assert sel.evidence["seed_evidence"][99]["rejected_reason"] == "manual_override_track_not_found"
