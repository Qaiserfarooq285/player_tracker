"""Pure-logic unit tests for `src/track/continuity.py` (ADR-13/14; CLAUDE.md task spec: no
GPU/video decode). Covers the extracted single-seed `stitch_timeline` (must behave exactly like
`src.highlights.selection`'s original inline implementation) and the new general-purpose
`build_take_identities`, including the mandatory take-crossing rejection tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import BBox, DetectionClass, Track, TrackBox
from src.track import continuity
from src.track.camera_motion import TakeCameraMotion

REPO_ROOT = Path(__file__).resolve().parents[1]


def _highlights_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "highlights.yaml")


def _selection_cfg() -> dict:
    return _highlights_config()["selection"]


def _box(t: float, cx: float, cy: float, height: float = 100.0, conf: float = 0.8) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 10)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=conf,
    )


# ---------------------------------------------------------------------------
# stitch_timeline -- must be behaviour-identical to the original selection.py implementation
# ---------------------------------------------------------------------------


def test_stitch_timeline_never_joins_across_different_take_ids():
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 0.0, 0.0), _box(0.1, 5.0, 0.0)])
    other_take = Track(id=2, take_id=1, boxes=[_box(0.2, 10.0, 0.0), _box(0.3, 15.0, 0.0)])
    with pytest.raises(AssertionError):
        continuity.stitch_timeline(1, [seed, other_take], _selection_cfg())


def test_stitch_timeline_joins_close_fragment_in_time_and_space():
    cfg = _selection_cfg()
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(0.5, 105.0, 100.0)])
    close_fragment = Track(
        id=2, take_id=0, boxes=[_box(1.0, 110.0, 100.0), _box(1.5, 115.0, 100.0)]
    )
    track_ids, confidence = continuity.stitch_timeline(1, [seed, close_fragment], cfg)
    assert track_ids == [1, 2]
    assert 0.0 < confidence <= 1.0


def test_stitch_timeline_rejects_fragment_beyond_max_gap():
    cfg = dict(_selection_cfg())
    cfg["stitch_max_gap_s"] = 0.5
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(0.5, 105.0, 100.0)])
    far_in_time = Track(id=2, take_id=0, boxes=[_box(5.0, 110.0, 100.0)])
    track_ids, _confidence = continuity.stitch_timeline(1, [seed, far_in_time], cfg)
    assert track_ids == [1]


def test_stitch_timeline_missing_seed_returns_empty():
    other = Track(id=2, take_id=0, boxes=[_box(0.0, 0.0, 0.0)])
    track_ids, confidence = continuity.stitch_timeline(999, [other], _selection_cfg())
    assert track_ids == []
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# build_take_identities -- the new general-purpose capability
# ---------------------------------------------------------------------------


def test_build_take_identities_rejects_mixed_take_ids():
    """The dedicated cross-take rejection test: even when two fragments are close in time/space,
    mixing take_ids in the INPUT must be rejected outright, never silently stitched (Golden Rule
    3) -- mirrors `stitch_timeline`'s own defensive assertion."""
    a = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    b = Track(id=2, take_id=1, boxes=[_box(0.01, 100.5, 100.0)])  # near-identical time/space
    with pytest.raises(AssertionError):
        continuity.build_take_identities([a, b], _selection_cfg())


def test_build_take_identities_never_links_across_takes_even_when_called_correctly():
    """The behavioural half of the take-crossing guarantee: build identities for take 0 and take 1
    SEPARATELY (as any real caller must, since a `Track` never itself spans two takes) and assert
    the two never end up sharing an identity id, no matter how close in time/space their raw boxes
    are -- there is no code path by which they even see each other's tracks."""
    cfg = _selection_cfg()
    take0 = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    take1 = Track(id=1, take_id=1, boxes=[_box(0.01, 100.5, 100.0)])  # same raw id, different take

    identity_of_0, _ = continuity.build_take_identities([take0], cfg)
    identity_of_1, _ = continuity.build_take_identities([take1], cfg)

    assert identity_of_0 == {1: 1}
    assert identity_of_1 == {1: 1}
    # each call only ever saw its own take's tracks -- there is no shared state/identity namespace
    # that could leak a take-0 identity into take-1's result.
    assert identity_of_0 is not identity_of_1


def test_build_take_identities_chains_close_fragments_into_one_identity():
    cfg = _selection_cfg()
    a = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    b = Track(id=2, take_id=0, boxes=[_box(0.5, 105.0, 100.0)])  # close in time+space to a
    identity_of, identity_confidence = continuity.build_take_identities([a, b], cfg)
    assert identity_of == {1: 1, 2: 1}
    assert set(identity_confidence.keys()) == {1}
    assert 0.0 < identity_confidence[1] <= 1.0


def test_build_take_identities_keeps_far_fragments_as_separate_identities():
    cfg = dict(_selection_cfg())
    cfg["stitch_max_gap_s"] = 0.5
    a = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    b = Track(id=2, take_id=0, boxes=[_box(5.0, 100.0, 100.0)])  # gap way beyond stitch_max_gap_s
    identity_of, identity_confidence = continuity.build_take_identities([a, b], cfg)
    assert identity_of == {1: 1, 2: 2}
    assert set(identity_confidence.keys()) == {1, 2}


def test_build_take_identities_partitions_multiple_people_correctly():
    """Two independent people, each fragmented into two pieces by a brief occlusion, interleaved
    in id order -- proves the whole-take partition finds BOTH chains correctly, not just "seed +
    everything else"."""
    cfg = _selection_cfg()
    person_a_1 = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0)])
    person_b_1 = Track(id=2, take_id=0, boxes=[_box(0.0, 900.0, 900.0)])
    person_a_2 = Track(id=3, take_id=0, boxes=[_box(0.5, 105.0, 100.0)])  # close to person_a_1
    person_b_2 = Track(id=4, take_id=0, boxes=[_box(0.5, 905.0, 900.0)])  # close to person_b_1

    identity_of, identity_confidence = continuity.build_take_identities(
        [person_a_1, person_b_1, person_a_2, person_b_2], cfg
    )
    assert identity_of == {1: 1, 2: 2, 3: 1, 4: 2}
    assert set(identity_confidence.keys()) == {1, 2}


def test_build_take_identities_empty_input_is_empty():
    identity_of, identity_confidence = continuity.build_take_identities([], _selection_cfg())
    assert identity_of == {}
    assert identity_confidence == {}


def test_build_take_identities_singleton_zero_box_track_does_not_crash():
    empty = Track(id=1, take_id=0, boxes=[], dominant_class=DetectionClass.PLAYER)
    other = Track(id=2, take_id=0, boxes=[_box(0.0, 0.0, 0.0)])
    identity_of, identity_confidence = continuity.build_take_identities(
        [empty, other], _selection_cfg()
    )
    assert identity_of[1] == 1
    assert identity_confidence[1] == 0.0
    assert identity_of[2] == 2


def test_build_take_identities_confidence_penalised_per_join():
    cfg = _selection_cfg()
    a = Track(id=1, take_id=0, boxes=[_box(0.0, 0.0, 0.0, conf=1.0)])
    b = Track(id=2, take_id=0, boxes=[_box(0.5, 5.0, 0.0, conf=1.0)])
    c = Track(id=3, take_id=0, boxes=[_box(1.0, 10.0, 0.0, conf=1.0)])

    identity_of_short, conf_short = continuity.build_take_identities([a, b], cfg)
    identity_of_long, conf_long = continuity.build_take_identities([a, b, c], cfg)
    assert identity_of_short == {1: 1, 2: 1}
    assert identity_of_long == {1: 1, 2: 1, 3: 1}
    assert conf_long[1] < conf_short[1]  # each extra join costs confidence, same as stitch_timeline


# ---------------------------------------------------------------------------
# Stage 7 ("streamed-gathering-treehouse" plan) -- three measured continuity gaps:
# (a) a zero-gap join bypassed the speed gate entirely; (b) build_take_identities never received
# camera motion; (c) no span-overlap rejection (click_reid.py has one, this module didn't).
# ---------------------------------------------------------------------------


def test_extend_chain_forward_rejects_zero_gap_join_even_when_close_in_space():
    """A candidate whose FIRST box lands at the exact same timestamp as the chain's own LAST box
    (`gap == 0`) is two DIFFERENT tracker fragments (a candidate is only ever considered when its
    id isn't already claimed) both reporting a detection at the SAME sampled instant -- not a
    reappearance, since nothing disappeared for anyone to reappear from. Before 2026-09-03 the old
    `gap > 0` conjunct let this bypass the speed gate entirely, falling through to only the
    gap-blind `stitch_max_dist` check -- so a same-instant pair comfortably under `stitch_max_dist`
    (one player supposedly occupying two simultaneous detections) was silently joined. Detection
    runs on a fixed grid (`configs/track.yaml: frame_rate`), so this is a reachable case, not a
    contrived one.
    """
    cfg = _selection_cfg()
    seed = Track(id=1, take_id=0, boxes=[_box(2.0, 100.0, 100.0, height=100.0)])
    # 2.0 bbox-heights away -- comfortably under the default stitch_max_dist (3.0) -- but at the
    # EXACT SAME timestamp as the seed's only box.
    same_instant = Track(id=2, take_id=0, boxes=[_box(2.0, 300.0, 100.0, height=100.0)])
    track_ids, _confidence = continuity.stitch_timeline(1, [seed, same_instant], cfg)
    assert track_ids == [1]  # never joined -- two simultaneous detections can't be one player


def test_extend_chain_forward_rejects_candidate_overlapping_an_earlier_chain_member():
    """Defensive parity with `src.track.click_reid.extend_chain_with_profile`'s own span-overlap
    guard (`click_reid.py:328-338`). Under NORMAL (per-track time-ordered) box lists this guard is
    provably redundant with the pre-existing `gap >= 0` check here -- each newly-accepted member's
    own `t_end` is a running maximum, so a candidate eligible on `gap` alone can never fall inside
    an EARLIER member's span. That argument depends on every individual track's own boxes being
    time-ordered, though -- this test constructs the one case where it breaks: an already-accepted
    fragment (id=2) whose own box list is NOT time-ordered (its LAST list entry precedes its FIRST)
    makes `current_end_t` (`current.boxes[-1].t` -- the only thing the `gap` check compares
    against) go BACKWARDS, letting a later candidate (id=3) whose real timestamp sits inside the
    FIRST fragment's (id=1) span slip past the `gap` check. The explicit overlap guard -- checked
    against every accepted span, not just `current`'s -- is what actually stops it.
    """
    cfg = dict(_selection_cfg())
    cfg["stitch_max_gap_s"] = 5.0  # generous enough that only the overlap guard is under test
    seed = Track(
        id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(10.0, 100.0, 100.0)]
    )  # occupies [0.0, 10.0]
    unsorted_fragment = Track(
        id=2, take_id=0, boxes=[_box(10.5, 105.0, 100.0), _box(2.0, 105.0, 100.0)]
    )  # its own boxes are OUT OF TIME ORDER: boxes[-1].t (2.0) precedes boxes[0].t (10.5)
    overlapping_candidate = Track(
        id=3, take_id=0, boxes=[_box(5.0, 100.0, 100.0)]
    )  # genuinely falls inside the SEED's [0.0, 10.0] span

    pool = {1: seed, 2: unsorted_fragment, 3: overlapping_candidate}
    used: set[int] = set()
    chain = continuity.extend_chain_forward(1, pool, used, cfg)

    assert [tr.id for tr in chain] == [1, 2]
    assert 3 not in used


def test_build_take_identities_uses_motion_to_compensate_camera_pan_when_supplied():
    """`build_take_identities` used to never receive a `TakeCameraMotion` at all
    (`continuity.py:282`, before 2026-09-03), so its whole-take partition ran in the raw,
    camera-motion-contaminated regime even when a real motion model was available for that take --
    exactly the failure `src/track/camera_motion.py`'s own module docstring measures for
    `stitch_timeline` (a fast pan makes a stationary player look like a sprinter). Synthetic
    version of that same measured case: a player who is REALLY stationary between t=0 and t=1
    while the camera pans 400px -- 4.0 bbox-heights at height=100, over the default
    `stitch_max_dist` of 3.0 -- so raw image positions make the two fragments look too far apart
    to join, while the camera-compensated position is unchanged and joins cleanly. Also proves
    `motion=None` (the default) keeps the original, uncompensated behaviour -- no existing caller
    is affected by this addition.
    """
    cfg = _selection_cfg()
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0, height=100.0)])
    # The SAME physical (stationary) player, recorded 400px further right purely because the
    # camera itself panned 400px rightward between t=0.0 and t=1.0 -- see `motion` below.
    panned_fragment = Track(id=2, take_id=0, boxes=[_box(1.0, 500.0, 100.0, height=100.0)])

    motion = TakeCameraMotion(
        take_id=0,
        times=[0.0, 1.0],
        # `cumulative[i]` maps the take's FIRST sampled frame into the frame at `times[i]`; a pure
        # +400px-in-x translation models "the camera panned 400px right by t=1.0" (see
        # `TakeCameraMotion`'s own docstring).
        cumulative=[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 400.0, 0.0, 1.0, 0.0]],
    )

    identity_of_raw, _ = continuity.build_take_identities([seed, panned_fragment], cfg)
    identity_of_compensated, _ = continuity.build_take_identities(
        [seed, panned_fragment], cfg, motion=motion
    )

    assert identity_of_raw == {
        1: 1,
        2: 2,
    }  # raw 4.0 bbox-h jump exceeds stitch_max_dist -- kept separate
    assert identity_of_compensated == {1: 1, 2: 1}  # compensated jump (~0) -- correctly joined
