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
