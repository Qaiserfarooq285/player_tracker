"""Pure-logic unit tests for Stage 5/6 (CLAUDE.md task spec: no GPU/video decode).

Covers: event ranking order, clip-range dedupe math, clip-snapping to take boundaries (Golden
Rule 3 — a clip must never cross a cut), within-take fragment stitching (never crosses a take,
respects the gap/distance thresholds), the arrow-tip vote, the no-arrow fallback heuristic, shot
attribution, and the `--track-id` human-override parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import (
    BallDetection,
    BBox,
    DetectionClass,
    Event,
    EventType,
    Take,
    Track,
    TrackBox,
)
from src.detect.overlay_mask import ArrowHint
from src.highlights import cutting, ranking, selection
from src.pipeline.run import normalize_manual_overrides, parse_track_id_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]


def _highlights_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "highlights.yaml")


def _box(t: float, cx: float, cy: float, height: float = 100.0, conf: float = 0.8) -> TrackBox:
    half_h = height / 2.0
    return TrackBox(
        frame_index=int(round(t * 10)),
        t=t,
        bbox=BBox(x1=cx - 20.0, y1=cy - half_h, x2=cx + 20.0, y2=cy + half_h),
        conf=conf,
    )


def _event(
    eid: str,
    etype: EventType,
    t_start: float,
    t_end: float,
    confidence: float,
    take_id: int | None = 0,
    player_track_id: int | None = None,
    evidence: dict | None = None,
) -> Event:
    return Event(
        id=eid,
        type=etype,
        t_start=t_start,
        t_end=t_end,
        player_track_id=player_track_id,
        take_id=take_id,
        confidence=confidence,
        source="test",
        evidence=evidence or {},
    )


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_event_rank_score_fixed_type_priority_beats_confidence():
    cfg = _highlights_config()["rank"]
    weights, cw = cfg["weights"], cfg["confidence_weight"]
    low_conf_shot = _event("a", EventType.SHOT, 0.0, 1.0, confidence=0.1)
    high_conf_sprint = _event("b", EventType.SPRINT, 0.0, 1.0, confidence=0.99)
    # CLAUDE.md §5 priority: shot > sprint, unconditionally on type -- even a barely-confident
    # shot must outrank an almost-certain sprint.
    assert ranking.event_rank_score(low_conf_shot, weights, cw) > ranking.event_rank_score(
        high_conf_sprint, weights, cw
    )


def test_event_rank_score_confidence_breaks_ties_within_a_type():
    cfg = _highlights_config()["rank"]
    weights, cw = cfg["weights"], cfg["confidence_weight"]
    low = _event("a", EventType.SPRINT, 0.0, 1.0, confidence=0.2)
    high = _event("b", EventType.SPRINT, 0.0, 1.0, confidence=0.8)
    assert ranking.event_rank_score(high, weights, cw) > ranking.event_rank_score(low, weights, cw)


def test_rank_events_only_keeps_events_attributed_to_the_timeline():
    cfg = _highlights_config()
    sel = selection.SelectionResult(
        video="v",
        target_jersey=77,
        overall_confidence=0.5,
        needs_human_confirmation=True,
        takes=[
            selection.TakeSelection(
                take_id=0,
                method="arrow_vote",
                seed_track_id=1,
                track_ids=[1],
                confidence=0.5,
                coverage_seconds=1.0,
                take_duration_seconds=2.0,
            )
        ],
    )
    mine = _event("mine", EventType.SPRINT, 0.0, 1.0, confidence=0.5, take_id=0, player_track_id=1)
    someone_elses = _event(
        "other", EventType.SPRINT, 0.0, 1.0, confidence=0.9, take_id=0, player_track_id=99
    )
    ranked = ranking.rank_events([mine, someone_elses], sel, {}, {}, cfg)
    assert [ev.id for ev, _ in ranked] == ["mine"]


def test_attribute_shot_true_when_timeline_near_ball():
    cfg = _highlights_config()["attribution"]
    track = Track(id=1, take_id=0, boxes=[_box(t=5.0, cx=100.0, cy=100.0, height=100.0)])
    ev = _event("s", EventType.SHOT, 5.0, 5.5, confidence=0.3, take_id=0)
    ball = BallDetection(
        bbox=BBox(x1=95.0, y1=95.0, x2=105.0, y2=105.0), conf=0.9, frame_index=50, t=5.0
    )
    assert ranking.attribute_shot(ev, [track], [ball], cfg) is True


def test_attribute_shot_false_when_timeline_far_from_ball():
    cfg = _highlights_config()["attribution"]
    track = Track(id=1, take_id=0, boxes=[_box(t=5.0, cx=100.0, cy=100.0, height=100.0)])
    ev = _event("s", EventType.SHOT, 5.0, 5.5, confidence=0.3, take_id=0)
    # 2000px away at bbox height 100 -> 20 bbox-heights, way over the default 4.0 ceiling
    ball = BallDetection(
        bbox=BBox(x1=2095.0, y1=95.0, x2=2105.0, y2=105.0), conf=0.9, frame_index=50, t=5.0
    )
    assert ranking.attribute_shot(ev, [track], [ball], cfg) is False


def test_attribute_shot_false_when_no_timeline_position_known():
    cfg = _highlights_config()["attribution"]
    ev = _event("s", EventType.SHOT, 5.0, 5.5, confidence=0.3, take_id=0)
    ball = BallDetection(
        bbox=BBox(x1=95.0, y1=95.0, x2=105.0, y2=105.0), conf=0.9, frame_index=50, t=5.0
    )
    assert ranking.attribute_shot(ev, [], [ball], cfg) is False


# ---------------------------------------------------------------------------
# dedupe / clip-range math
# ---------------------------------------------------------------------------


def test_time_iou_no_overlap_is_zero():
    assert cutting.time_iou((0.0, 1.0), (2.0, 3.0)) == 0.0


def test_time_iou_identical_ranges_is_one():
    assert cutting.time_iou((0.0, 5.0), (0.0, 5.0)) == pytest.approx(1.0)


def test_time_iou_partial_overlap():
    # intersection [1,2] = 1, union [0,3] = 3 -> 1/3
    assert cutting.time_iou((0.0, 2.0), (1.0, 3.0)) == pytest.approx(1.0 / 3.0)


def test_dedupe_clip_ranges_keeps_higher_ranked_of_overlapping_pair():
    best = _event("best", EventType.SHOT, 0.0, 0.0, confidence=0.9)
    worse = _event("worse", EventType.SPRINT, 0.0, 0.0, confidence=0.1)
    ranges = {"best": (0.0, 10.0), "worse": (1.0, 9.0)}  # heavily overlapping
    kept = cutting.dedupe_clip_ranges([best, worse], ranges, overlap_threshold=0.5)
    assert [ev.id for ev in kept] == ["best"]


def test_dedupe_clip_ranges_keeps_both_when_barely_overlapping():
    a = _event("a", EventType.SHOT, 0.0, 0.0, confidence=0.9)
    b = _event("b", EventType.SPRINT, 0.0, 0.0, confidence=0.5)
    ranges = {"a": (0.0, 10.0), "b": (9.9, 20.0)}  # tiny overlap
    kept = cutting.dedupe_clip_ranges([a, b], ranges, overlap_threshold=0.5)
    assert {ev.id for ev in kept} == {"a", "b"}


def test_clip_range_for_event_snaps_to_take_boundary():
    clip_cfg = {"pre_roll_s": 3.0, "post_roll_s": 2.0, "snap_to_take_boundaries": True}
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)
    # event near the take's start/end: naive padding would run to -3.0 .. 12.0, well outside
    ev = _event("e", EventType.SPRINT, 1.0, 9.0, confidence=0.5, take_id=0)
    start, end = cutting.clip_range_for_event(ev, take, clip_cfg)
    assert start == pytest.approx(0.0)  # clamped, never negative / before the take
    assert end == pytest.approx(10.0)  # clamped, never past the take's own end
    assert start >= take.t_start
    assert end <= take.t_end


def test_clip_range_for_event_unclamped_without_snapping():
    clip_cfg = {"pre_roll_s": 3.0, "post_roll_s": 2.0, "snap_to_take_boundaries": False}
    take = Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)
    ev = _event("e", EventType.SPRINT, 1.0, 9.0, confidence=0.5, take_id=0)
    start, end = cutting.clip_range_for_event(ev, take, clip_cfg)
    assert start == pytest.approx(-2.0)  # 1.0 - 3.0, NOT clamped when snapping is off
    assert end == pytest.approx(11.0)  # 9.0 + 2.0


# ---------------------------------------------------------------------------
# fragment stitching -- Golden Rule 3: NEVER crosses a take
# ---------------------------------------------------------------------------


def _selection_cfg() -> dict:
    return _highlights_config()["selection"]


def test_stitch_timeline_never_joins_across_different_take_ids():
    """`stitch_timeline` is only ever called with one take's own tracks by its caller
    (`src.highlights.selection.select_targets`); assert that invariant is enforced -- passing a
    mix of take ids must be rejected rather than silently joining across a cut.
    """
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 0.0, 0.0), _box(0.1, 5.0, 0.0)])
    other_take = Track(id=2, take_id=1, boxes=[_box(0.2, 10.0, 0.0), _box(0.3, 15.0, 0.0)])
    with pytest.raises(AssertionError):
        selection.stitch_timeline(1, [seed, other_take], _selection_cfg())


def test_stitch_timeline_joins_close_fragment_in_time_and_space():
    cfg = _selection_cfg()
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(0.5, 105.0, 100.0)])
    close_fragment = Track(
        id=2, take_id=0, boxes=[_box(1.0, 110.0, 100.0), _box(1.5, 115.0, 100.0)]
    )
    track_ids, confidence = selection.stitch_timeline(1, [seed, close_fragment], cfg)
    assert track_ids == [1, 2]
    assert 0.0 < confidence <= 1.0


def test_stitch_timeline_rejects_fragment_beyond_max_gap():
    cfg = dict(_selection_cfg())
    cfg["stitch_max_gap_s"] = 0.5
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0), _box(0.5, 105.0, 100.0)])
    far_in_time = Track(id=2, take_id=0, boxes=[_box(5.0, 110.0, 100.0)])  # gap = 4.5s >> 0.5s
    track_ids, _confidence = selection.stitch_timeline(1, [seed, far_in_time], cfg)
    assert track_ids == [1]  # fragment never joined


def test_stitch_timeline_rejects_fragment_beyond_max_dist():
    cfg = dict(_selection_cfg())
    cfg["stitch_max_gap_s"] = 5.0
    cfg["stitch_max_dist"] = 1.0  # 1 bbox-height allowed jump
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 100.0, 100.0, height=100.0)])
    far_in_space = Track(
        id=2, take_id=0, boxes=[_box(1.0, 10000.0, 100.0, height=100.0)]
    )  # ~9900px / 100px height = 99 bbox-heights away
    track_ids, _confidence = selection.stitch_timeline(1, [seed, far_in_space], cfg)
    assert track_ids == [1]


def test_stitch_timeline_confidence_penalised_per_join():
    cfg = _selection_cfg()
    seed = Track(id=1, take_id=0, boxes=[_box(0.0, 0.0, 0.0, conf=1.0)])
    frag1 = Track(id=2, take_id=0, boxes=[_box(0.5, 5.0, 0.0, conf=1.0)])
    frag2 = Track(id=3, take_id=0, boxes=[_box(1.0, 10.0, 0.0, conf=1.0)])
    ids_one_join, conf_one_join = selection.stitch_timeline(1, [seed, frag1], cfg)
    ids_two_joins, conf_two_joins = selection.stitch_timeline(1, [seed, frag1, frag2], cfg)
    assert len(ids_one_join) == 2
    assert len(ids_two_joins) == 3
    assert conf_two_joins < conf_one_join  # each extra join costs confidence


# ---------------------------------------------------------------------------
# arrow vote
# ---------------------------------------------------------------------------


def test_vote_seed_tracks_prefers_containing_track():
    takes = [Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)]
    target = Track(id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0, height=100.0)])
    bystander = Track(id=2, take_id=0, boxes=[_box(1.0, 500.0, 500.0, height=100.0)])
    hint = ArrowHint(
        frame_index=10,
        t=1.0,
        bbox=BBox(x1=90, y1=90, x2=110, y2=110),
        area_px=400,
        tip_x=100.0,
        tip_y=100.0,
    )
    votes = selection.vote_seed_tracks([hint], [target, bystander], takes, _selection_cfg())
    assert votes == {0: {1: 1}}


def test_vote_seed_tracks_falls_back_to_nearest_within_max_dist():
    cfg = dict(_selection_cfg())
    cfg["arrow_match_max_dist_px"] = 200
    takes = [Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)]
    # tip sits just outside the box (not contained) but well within max_dist of its centre
    nearby = Track(id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0, height=50.0)])
    hint = ArrowHint(
        frame_index=10,
        t=1.0,
        bbox=BBox(x1=140, y1=90, x2=160, y2=110),
        area_px=400,
        tip_x=150.0,
        tip_y=100.0,
    )
    votes = selection.vote_seed_tracks([hint], [nearby], takes, cfg)
    assert votes == {0: {1: 1}}


def test_vote_seed_tracks_ignores_hint_beyond_max_dist():
    cfg = dict(_selection_cfg())
    cfg["arrow_match_max_dist_px"] = 10
    takes = [Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)]
    far = Track(id=1, take_id=0, boxes=[_box(1.0, 100.0, 100.0, height=50.0)])
    hint = ArrowHint(
        frame_index=10,
        t=1.0,
        bbox=BBox(x1=990, y1=990, x2=1010, y2=1010),
        area_px=400,
        tip_x=1000.0,
        tip_y=1000.0,
    )
    votes = selection.vote_seed_tracks([hint], [far], takes, cfg)
    assert votes == {}


# ---------------------------------------------------------------------------
# heuristic fallback
# ---------------------------------------------------------------------------


def test_heuristic_fallback_seed_prefers_longer_more_confident_central_track():
    cfg = _selection_cfg()
    frame_w, frame_h = 1000.0, 1000.0
    long_confident_central = Track(
        id=1,
        take_id=0,
        boxes=[_box(t=i * 0.5, cx=500.0, cy=500.0, conf=0.9) for i in range(10)],
    )
    short_low_conf_edge = Track(
        id=2,
        take_id=0,
        boxes=[_box(t=i * 0.5, cx=10.0, cy=10.0, conf=0.2) for i in range(2)],
    )
    seed = selection.heuristic_fallback_seed(
        [long_confident_central, short_low_conf_edge], frame_w, frame_h, cfg
    )
    assert seed == 1


def test_heuristic_fallback_seed_excludes_referees():
    cfg = _selection_cfg()
    referee = Track(
        id=1,
        take_id=0,
        boxes=[_box(t=i * 0.5, cx=500.0, cy=500.0, conf=0.99) for i in range(10)],
        dominant_class=DetectionClass.REFEREE,
    )
    player = Track(
        id=2,
        take_id=0,
        boxes=[_box(t=i * 0.5, cx=100.0, cy=100.0, conf=0.5) for i in range(3)],
        dominant_class=DetectionClass.PLAYER,
    )
    seed = selection.heuristic_fallback_seed([referee, player], 1000.0, 1000.0, cfg)
    assert seed == 2


def test_heuristic_fallback_seed_returns_none_when_nothing_qualifies():
    cfg = _selection_cfg()
    too_short = Track(id=1, take_id=0, boxes=[_box(0.0, 0.0, 0.0)])
    assert selection.heuristic_fallback_seed([too_short], 1000.0, 1000.0, cfg) is None


# ---------------------------------------------------------------------------
# --track-id override parsing
# ---------------------------------------------------------------------------


def test_parse_track_id_overrides_bare_form_applies_to_take_zero():
    assert parse_track_id_overrides("16") == {0: 16}


def test_parse_track_id_overrides_multi_take_form():
    assert parse_track_id_overrides("0:16,2:9") == {0: 16, 2: 9}


def test_parse_track_id_overrides_empty_is_empty():
    assert parse_track_id_overrides(None) == {}
    assert parse_track_id_overrides("") == {}


# ---------------------------------------------------------------------------
# `normalize_manual_overrides` -- "streamed-gathering-treehouse" plan Stage 1: the pipeline-
# boundary bridge between `parse_track_id_overrides`'s unchanged `dict[int, int]` CLI shape and
# `apps/api/main.py`'s multi-anchor `dict[int, list[int]]` shape, into ONE canonical form before
# `select_targets` ever sees it.
# ---------------------------------------------------------------------------


def test_normalize_manual_overrides_none_and_empty():
    assert normalize_manual_overrides(None) == {}
    assert normalize_manual_overrides({}) == {}


def test_normalize_manual_overrides_bare_int_becomes_single_element_list():
    """The legacy `parse_track_id_overrides` shape (`dict[int, int]`, e.g. the CLI's own
    `--track-id` flag) passes straight through as a one-element list per take."""
    assert normalize_manual_overrides({0: 16, 2: 9}) == {0: [16], 2: [9]}


def test_normalize_manual_overrides_passes_list_through():
    assert normalize_manual_overrides({0: [16, 44]}) == {0: [16, 44]}


def test_normalize_manual_overrides_dedupes_within_a_take_order_preserving():
    """A duplicate anchor for the same take (the user clicked the same box twice, or the raw-
    coordinate click path and the picker both resolved to the same track) collapses to one entry,
    keeping the FIRST occurrence's position -- never silently doubling a chain's weight."""
    assert normalize_manual_overrides({0: [16, 44, 16]}) == {0: [16, 44]}


def test_normalize_manual_overrides_drops_takes_with_no_surviving_anchors():
    """An empty list for a take (e.g. every duplicate collapsed away, or the caller explicitly
    passed `[]`) is dropped from the result entirely -- equivalent to no override for that take at
    all, never a dict entry pointing at nothing."""
    assert normalize_manual_overrides({0: [16], 3: []}) == {0: [16]}


def test_normalize_manual_overrides_mixed_int_and_list_shapes():
    """A real call site could plausibly merge a CLI-style bare int for one take with an API-style
    list for another -- both normalize into the same canonical shape."""
    assert normalize_manual_overrides({0: 16, 1: [7, 8]}) == {0: [16], 1: [7, 8]}
