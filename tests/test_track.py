"""Pure-logic unit tests for Stage 3 (CLAUDE.md task spec: no GPU in unit tests).

Covers: per-take tracker reset (`src/track/tracker.py::track_take` — IDs must not continue across
takes, Golden Rule 3), team majority-vote logic, torso-crop geometry, and cluster-balance scoring
(`src/team/classifier.py`). Real `configs/track.yaml`/`configs/team.yaml` params are used (matching
`tests/test_profiler.py`'s pattern of exercising the actual configured values, not hand-copied
duplicates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.io import load_yaml
from src.common.types import BBox, Detection, DetectionClass, Take
from src.team.classifier import (
    balance_ratio,
    centroid_distance_confidence,
    crop_lab_mean,
    majority_vote_teams,
    per_track_lab_median,
    torso_region,
)
from src.track.tracker import assign_take_id, track_take

REPO_ROOT = Path(__file__).resolve().parents[1]


def _track_config() -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "track.yaml")
    # A generous matching threshold + low activation threshold keeps a single, slowly-moving
    # synthetic detection tracked as ONE continuous track across a handful of frames, which is
    # all these tests need from ByteTrack itself — the actual thing under test is per-take reset.
    return cfg


def _det(frame_index: int, t: float, x: float) -> Detection:
    return Detection(
        bbox=BBox(x1=x, y1=100.0, x2=x + 40.0, y2=180.0),
        cls=DetectionClass.PLAYER,
        conf=0.9,
        frame_index=frame_index,
        t=t,
    )


def _walking_detections(n_frames: int, start_frame: int = 0, start_t: float = 0.0) -> list:
    """One synthetic player drifting slowly right, frame by frame — easily tracked as one ID."""
    return [
        (
            start_frame + i,
            start_t + i * 0.1,
            [_det(start_frame + i, start_t + i * 0.1, 100.0 + i * 5)],
        )
        for i in range(n_frames)
    ]


def test_track_take_ids_start_fresh_each_call():
    """Golden Rule 3 / ADR-7: track only within a take; reset IDs at every cut.

    `track_take` constructs a brand-new `ByteTrack` per call (see its docstring) — calling it
    independently for two different takes must NOT continue track IDs from the first call into
    the second, even though both takes contain visually-identical synthetic detections.
    """
    cfg = _track_config()
    fps_sample = cfg["bytetrack"]["frame_rate"]

    take0 = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10)
    take1 = Take(id=1, t_start=1.0, t_end=2.0, frame_start=10, frame_end=20)

    tracks0 = track_take(_walking_detections(8), take0, cfg, fps_sample)
    tracks1 = track_take(_walking_detections(8), take1, cfg, fps_sample)

    assert len(tracks0) >= 1
    assert len(tracks1) >= 1
    # every take_id is correctly stamped on its own tracks
    assert all(tr.take_id == 0 for tr in tracks0)
    assert all(tr.take_id == 1 for tr in tracks1)
    # the crucial assertion: take1's IDs must look like a fresh start (same ID space as take0's),
    # not a continuation (e.g. take0 -> id 1, take1 -> id 2, 3, ... if state leaked across calls)
    assert sorted(tr.id for tr in tracks1) == sorted(tr.id for tr in tracks0)


def test_track_take_sets_dominant_class():
    cfg = _track_config()
    fps_sample = cfg["bytetrack"]["frame_rate"]
    take = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10)

    tracks = track_take(_walking_detections(5), take, cfg, fps_sample)
    assert len(tracks) >= 1
    assert all(tr.dominant_class == DetectionClass.PLAYER for tr in tracks)


def test_track_take_untracked_class_excluded():
    """A class not listed in `tracked_classes` never produces a track."""
    cfg = _track_config()
    fps_sample = cfg["bytetrack"]["frame_rate"]
    take = Take(id=0, t_start=0.0, t_end=1.0, frame_start=0, frame_end=10)

    ball_frames = [
        (
            i,
            i * 0.1,
            [
                Detection(
                    bbox=BBox(x1=100.0 + i * 5, y1=100.0, x2=110.0 + i * 5, y2=110.0),
                    cls=DetectionClass.BALL,
                    conf=0.7,
                    frame_index=i,
                    t=i * 0.1,
                )
            ],
        )
        for i in range(5)
    ]
    tracks = track_take(ball_frames, take, cfg, fps_sample)
    assert tracks == []  # ball is not in configs/track.yaml's tracked_classes


# ---------------------------------------------------------------------------
# assign_take_id
# ---------------------------------------------------------------------------


def test_assign_take_id_finds_containing_take():
    takes = [
        Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100),
        Take(id=1, t_start=10.0, t_end=20.0, frame_start=100, frame_end=200),
    ]
    assert assign_take_id(5.0, takes) == 0
    assert assign_take_id(10.0, takes) == 1  # boundary belongs to the take that starts there
    assert assign_take_id(19.999, takes) == 1
    assert assign_take_id(20.0, takes) == 1  # closed-interval fallback for the very last instant


def test_assign_take_id_outside_all_takes_returns_none():
    takes = [Take(id=0, t_start=0.0, t_end=10.0, frame_start=0, frame_end=100)]
    assert assign_take_id(50.0, takes) is None


# ---------------------------------------------------------------------------
# majority_vote_teams
# ---------------------------------------------------------------------------


def test_majority_vote_teams_picks_most_common_label_per_track():
    # track 1: 4 crops labelled team 0, 1 crop labelled team 1 -> team 0, confidence 4/5
    # track 2: all 3 crops labelled team 1 -> team 1, confidence 3/3
    crop_track_ids = [1, 1, 1, 1, 1, 2, 2, 2]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    team_by_track, confidence_by_track = majority_vote_teams(crop_track_ids, labels)

    assert team_by_track[1] == 0
    assert confidence_by_track[1] == 4 / 5
    assert team_by_track[2] == 1
    assert confidence_by_track[2] == 1.0


def test_majority_vote_teams_unanimous_gives_full_confidence():
    team_by_track, confidence_by_track = majority_vote_teams([7, 7, 7], [0, 0, 0])
    assert team_by_track[7] == 0
    assert confidence_by_track[7] == 1.0


def test_majority_vote_teams_tie_breaks_deterministically_via_first_seen():
    # Counter.most_common ties break by insertion order — first-seen label wins.
    team_by_track, confidence_by_track = majority_vote_teams([1, 1], [0, 1])
    assert team_by_track[1] == 0
    assert confidence_by_track[1] == 0.5


# ---------------------------------------------------------------------------
# torso_region
# ---------------------------------------------------------------------------


def test_torso_region_is_a_horizontal_band_within_bbox():
    bbox = BBox(x1=100.0, y1=200.0, x2=180.0, y2=440.0)  # 80 wide, 240 tall
    torso = torso_region(bbox, top_frac=0.40, bottom_frac=0.55, x_inset_frac=0.15)

    assert torso.y1 == pytest.approx(200.0 + 0.40 * 240.0)
    assert torso.y2 == pytest.approx(200.0 + 0.55 * 240.0)
    assert torso.x1 == pytest.approx(100.0 + 0.15 * 80.0)
    assert torso.x2 == pytest.approx(180.0 - 0.15 * 80.0)
    # strictly inside the original bbox on all sides
    assert bbox.x1 < torso.x1 < torso.x2 < bbox.x2
    assert bbox.y1 < torso.y1 < torso.y2 < bbox.y2


def test_torso_region_scales_with_bbox_size():
    small = torso_region(BBox(x1=0, y1=0, x2=10, y2=30), 0.4, 0.55, 0.15)
    big = torso_region(BBox(x1=0, y1=0, x2=100, y2=300), 0.4, 0.55, 0.15)
    assert big.x2 - big.x1 == pytest.approx((small.x2 - small.x1) * 10)
    assert big.y2 - big.y1 == pytest.approx((small.y2 - small.y1) * 10)


# ---------------------------------------------------------------------------
# balance_ratio
# ---------------------------------------------------------------------------


def test_balance_ratio_perfectly_balanced():
    teams = {1: 0, 2: 0, 3: 1, 4: 1}
    assert balance_ratio(teams) == pytest.approx(0.5)


def test_balance_ratio_degenerate_split_reads_low():
    # the exact failure mode the coordinator reported: 48 vs 3
    teams = {i: 0 for i in range(48)} | {48 + i: 1 for i in range(3)}
    assert balance_ratio(teams) == pytest.approx(3 / 51)


def test_balance_ratio_single_cluster_only_is_zero():
    teams = {1: 0, 2: 0, 3: 0}
    assert balance_ratio(teams) == 0.0


def test_balance_ratio_ignores_none_entries():
    teams = {1: 0, 2: 0, 3: 1, 4: 1, 5: None}
    assert balance_ratio(teams) == pytest.approx(0.5)


def test_balance_ratio_fewer_than_two_assigned_is_zero():
    assert balance_ratio({1: 0}) == 0.0
    assert balance_ratio({}) == 0.0


# ---------------------------------------------------------------------------
# crop_lab_mean / per_track_lab_median
# ---------------------------------------------------------------------------


def test_crop_lab_mean_distinguishes_dark_from_light():
    dark_blue = np.full((10, 10, 3), (200, 40, 20), dtype=np.uint8)  # BGR, dark/saturated
    white = np.full((10, 10, 3), (230, 230, 230), dtype=np.uint8)  # BGR, light/near-neutral
    dark_lab = crop_lab_mean(dark_blue)
    white_lab = crop_lab_mean(white)
    assert white_lab[0] > dark_lab[0]  # L* (lightness) is clearly higher for the white crop


def test_crop_lab_mean_uniform_crop_has_low_variance_across_pixels():
    # A uniform-colour crop's mean should equal (approximately) any single pixel's own Lab value.
    crop = np.full((8, 8, 3), (100, 150, 200), dtype=np.uint8)
    import cv2

    single_pixel_lab = cv2.cvtColor(crop[:1, :1], cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
    mean_lab = crop_lab_mean(crop)
    assert mean_lab == pytest.approx(single_pixel_lab, abs=1e-6)


def test_per_track_lab_median_is_robust_to_one_outlier_crop():
    # Four consistent "white-ish" crops plus one wildly different (motion-blur-like) outlier —
    # median should stay close to the consistent crops, not get pulled toward the outlier.
    consistent = np.full((10, 10, 3), (230, 230, 230), dtype=np.uint8)
    outlier = np.full((10, 10, 3), (10, 10, 10), dtype=np.uint8)  # near-black
    crops = [consistent, consistent, consistent, consistent, outlier]
    median_lab = per_track_lab_median(crops)
    consistent_lab = crop_lab_mean(consistent)
    assert median_lab == pytest.approx(consistent_lab, abs=1.0)


def test_per_track_lab_median_empty_crops_returns_none():
    assert per_track_lab_median([]) is None


# ---------------------------------------------------------------------------
# centroid_distance_confidence
# ---------------------------------------------------------------------------


def test_centroid_distance_confidence_on_own_centroid_is_full():
    own = np.array([0.0, 0.0, 0.0])
    other = np.array([10.0, 0.0, 0.0])
    assert centroid_distance_confidence(own, own, [other]) == pytest.approx(1.0)


def test_centroid_distance_confidence_midway_is_half():
    own = np.array([0.0, 0.0, 0.0])
    other = np.array([10.0, 0.0, 0.0])
    midpoint = np.array([5.0, 0.0, 0.0])
    assert centroid_distance_confidence(midpoint, own, [other]) == pytest.approx(0.5)


def test_centroid_distance_confidence_closer_to_other_is_below_half():
    own = np.array([0.0, 0.0, 0.0])
    other = np.array([10.0, 0.0, 0.0])
    near_other = np.array([8.0, 0.0, 0.0])
    assert centroid_distance_confidence(near_other, own, [other]) < 0.5


def test_centroid_distance_confidence_picks_nearest_other_centroid():
    own = np.array([0.0, 0.0, 0.0])
    far_other = np.array([100.0, 0.0, 0.0])
    near_other = np.array([4.0, 0.0, 0.0])
    point = np.array([2.0, 0.0, 0.0])
    # distance to own=2, distance to nearest other (near_other)=2 -> confidence 0.5, not
    # dominated by the far_other centroid
    result = centroid_distance_confidence(point, own, [far_other, near_other])
    assert result == pytest.approx(0.5)
