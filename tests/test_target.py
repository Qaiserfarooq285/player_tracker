"""Pure-logic unit tests for `src/track/target.py` (Stage 1 of the "streamed-gathering-treehouse"
plan: the persistent target profile). No GPU, no video decode -- synthetic frames/tracks only,
same style as `tests/test_kit_colour.py`/`tests/test_click_reid.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.io import load_yaml
from src.common.types import BBox, Track, TrackBox
from src.track.target import (
    KitColourSample,
    TargetLink,
    TargetState,
    build_target_profile,
    load_target_profile,
    sample_kit_bands,
    save_target_profile,
    update_memory_bank,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "target.yaml")


def _box(t: float, height: float = 100.0) -> TrackBox:
    return TrackBox(frame_index=int(t * 10), t=t, bbox=BBox(x1=0, y1=0, x2=50, y2=height), conf=0.9)


def _track(tid: int, boxes: list[TrackBox], team: int | None = None) -> Track:
    return Track(id=tid, take_id=0, boxes=boxes, team=team)


def _kit_sample(
    torso=(45.0, 5.0, -20.0), shorts=(45.0, 5.0, -20.0), socks=(45.0, 5.0, -20.0), confidence=1.0
) -> KitColourSample:
    return KitColourSample(
        torso_lab=torso, shorts_lab=shorts, socks_lab=socks, take_id=0, t=1.0, confidence=confidence
    )


# ---------------------------------------------------------------------------
# sample_kit_bands
# ---------------------------------------------------------------------------


def test_sample_kit_bands_all_grass_returns_none():
    """A crop that is entirely pitch-green has no kit anywhere -- every band should reject, and
    the whole sample must be `None`, never a zero-confidence stub."""
    frame = np.zeros((400, 200, 3), dtype=np.uint8)
    frame[:, :] = (40, 180, 40)  # solid saturated green (BGR), pure grass hue
    bbox = BBox(x1=20, y1=20, x2=180, y2=380)
    result = sample_kit_bands(frame, bbox, take_id=0, t=1.0, cfg=_cfg())
    assert result is None


def test_sample_kit_bands_degenerate_bbox_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = BBox(x1=50, y1=50, x2=50.2, y2=50.2)
    result = sample_kit_bands(frame, bbox, take_id=0, t=1.0, cfg=_cfg())
    assert result is None


def test_sample_kit_bands_solid_colour_all_bands_populated():
    """A solid, saturated non-grass colour fills every band -- all three should be sampled and
    confidence should be the full sum of `band_confidence_weights` (i.e. 1.0)."""
    frame = np.zeros((400, 200, 3), dtype=np.uint8)
    frame[:, :] = (200, 30, 30)  # solid saturated blue (BGR) -- not in the grass hue band
    bbox = BBox(x1=20, y1=20, x2=180, y2=380)
    cfg = _cfg()
    result = sample_kit_bands(frame, bbox, take_id=2, t=5.0, cfg=cfg)
    assert result is not None
    assert result.torso_lab is not None
    assert result.shorts_lab is not None
    assert result.socks_lab is not None
    assert result.take_id == 2
    assert result.t == 5.0
    weights = cfg["band_confidence_weights"]
    assert np.isclose(result.confidence, sum(weights.values()))


def test_sample_kit_bands_partial_grass_lowers_confidence():
    """Grass in the socks band only -- torso/shorts should still sample, socks should not, and
    confidence should equal torso+shorts weights only (never the full 1.0)."""
    frame = np.zeros((400, 200, 3), dtype=np.uint8)
    frame[:, :] = (200, 30, 30)  # solid blue everywhere
    cfg = _cfg()
    bbox = BBox(x1=20, y1=20, x2=180, y2=380)
    socks_band = cfg["bands"]["socks"]
    box_h = bbox.y2 - bbox.y1
    y1 = int(round(bbox.y1 + box_h * socks_band["top_frac"]))
    y2 = int(round(bbox.y1 + box_h * socks_band["bottom_frac"]))
    frame[y1:y2, :] = (40, 180, 40)  # paint the socks region pure grass
    result = sample_kit_bands(frame, bbox, take_id=0, t=1.0, cfg=cfg)
    assert result is not None
    assert result.socks_lab is None
    assert result.torso_lab is not None
    assert result.shorts_lab is not None
    weights = cfg["band_confidence_weights"]
    assert np.isclose(result.confidence, weights["torso"] + weights["shorts"])


# ---------------------------------------------------------------------------
# build_target_profile
# ---------------------------------------------------------------------------


def test_build_target_profile_reuses_median_height():
    anchor = _track(1, [_box(0.0, 90), _box(0.1, 100), _box(0.2, 110)], team=1)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample()],
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=0.0,
        cfg=_cfg(),
    )
    assert profile.median_height == 100  # median of [90, 100, 110], same as click_reid
    assert profile.team_cluster == 1
    assert profile.established_track_id == 1
    assert profile.jersey_number == 10
    assert profile.jersey_source == "click"
    assert profile.target_id == "TARGET_001"


def test_build_target_profile_never_a_tracker_id():
    """The plan is explicit: `target_id` must NEVER be a raw tracker id, even when a caller
    supplies one via `anchor_track`."""
    anchor = _track(777, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[],
        jersey_number=None,
        jersey_source=None,
        established_take_id=0,
        established_t=0.0,
        cfg=_cfg(),
    )
    assert profile.target_id != "777"
    assert profile.target_id == "TARGET_001"


def test_build_target_profile_aggregates_kit_from_samples():
    samples = [_kit_sample(torso=(45.0, 5.0, -20.0)), _kit_sample(torso=(47.0, 7.0, -18.0))]
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=samples,
        jersey_number=None,
        jersey_source=None,
        established_take_id=0,
        established_t=0.0,
        cfg=_cfg(),
    )
    assert profile.kit is not None
    assert np.allclose(profile.kit.torso_lab, (46.0, 6.0, -19.0))  # median of the two
    assert len(profile.kit_bank) == 2


def test_build_target_profile_empty_samples_has_no_kit():
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[],
        jersey_number=None,
        jersey_source=None,
        established_take_id=0,
        established_t=0.0,
        cfg=_cfg(),
    )
    assert profile.kit is None
    assert profile.kit_bank == []


def test_build_target_profile_caps_bank_at_max_bank_samples():
    cfg = _cfg()
    cap = cfg["max_bank_samples"]
    samples = [_kit_sample(torso=(float(i), 0.0, 0.0)) for i in range(cap + 5)]
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=samples,
        jersey_number=None,
        jersey_source=None,
        established_take_id=0,
        established_t=0.0,
        cfg=cfg,
    )
    assert len(profile.kit_bank) == cap
    # the most RECENT `cap` samples are kept (i.e. the tail of the input list)
    assert profile.kit_bank[0].torso_lab[0] == 5.0


# ---------------------------------------------------------------------------
# update_memory_bank -- the anti-drift rule (plan §4)
# ---------------------------------------------------------------------------


def test_update_memory_bank_refuses_low_confidence_sample():
    """A link accepted below `memory_update_min_confidence` must NEVER write to the bank -- this
    is the core anti-drift guarantee the whole plan hinges on."""
    cfg = _cfg()
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample()],
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=0.0,
        cfg=cfg,
    )
    low_confidence = cfg["memory_update_min_confidence"] - 0.01
    new_sample = _kit_sample(torso=(90.0, 90.0, 90.0))  # wildly different colour
    updated = update_memory_bank(profile, new_sample, low_confidence, cfg)
    assert updated == profile  # completely unchanged
    assert len(updated.kit_bank) == 1
    assert updated.kit.torso_lab == profile.kit.torso_lab


def test_update_memory_bank_accepts_at_exactly_the_floor():
    cfg = _cfg()
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample()],
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=0.0,
        cfg=cfg,
    )
    exactly_floor = cfg["memory_update_min_confidence"]
    new_sample = _kit_sample(torso=(50.0, 10.0, -15.0))
    updated = update_memory_bank(profile, new_sample, exactly_floor, cfg)
    assert len(updated.kit_bank) == 2


def test_update_memory_bank_accepts_high_confidence_and_recomputes_median():
    cfg = _cfg()
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample(torso=(40.0, 0.0, 0.0))],
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=0.0,
        cfg=cfg,
    )
    new_sample = _kit_sample(torso=(60.0, 0.0, 0.0))
    updated = update_memory_bank(profile, new_sample, 1.0, cfg)
    assert len(updated.kit_bank) == 2
    assert np.allclose(updated.kit.torso_lab, (50.0, 0.0, 0.0))  # median of 40 and 60


def test_update_memory_bank_caps_at_max_bank_samples():
    cfg = _cfg()
    cap = cfg["max_bank_samples"]
    anchor = _track(1, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample()] * cap,
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=0.0,
        cfg=cfg,
    )
    assert len(profile.kit_bank) == cap
    updated = update_memory_bank(profile, _kit_sample(torso=(1.0, 1.0, 1.0)), 1.0, cfg)
    assert len(updated.kit_bank) == cap  # still capped, oldest dropped
    assert updated.kit_bank[-1].torso_lab == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# save/load round trip
# ---------------------------------------------------------------------------


def test_target_profile_round_trips_through_save_and_load(tmp_path):
    anchor = _track(1, [_box(0.0, 100), _box(0.1, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[_kit_sample()],
        jersey_number=10,
        jersey_source="click",
        established_take_id=0,
        established_t=1.5,
        cfg=_cfg(),
    )
    profile.links.append(
        TargetLink(
            take_id=0,
            track_ids=[1],
            state=TargetState.ACTIVE,
            confidence=0.95,
            evidence={"note": "established by click"},
            t_start=0.0,
            t_end=2.0,
        )
    )
    path = tmp_path / "target.json"
    save_target_profile(profile, path)
    assert path.exists()

    loaded = load_target_profile(path)
    assert loaded == profile
    assert loaded.links[0].state == TargetState.ACTIVE
    assert loaded.kit.torso_lab == profile.kit.torso_lab


def test_target_profile_json_has_no_tracker_id_leakage(tmp_path):
    """Sanity check on the serialized form itself: `target_id` on disk is the fixed string, not
    whatever raw track id was used to establish the profile."""
    anchor = _track(42, [_box(0.0, 100)], team=0)
    profile = build_target_profile(
        anchor_track=anchor,
        kit_samples=[],
        jersey_number=None,
        jersey_source=None,
        established_take_id=0,
        established_t=0.0,
        cfg=_cfg(),
    )
    path = tmp_path / "target.json"
    save_target_profile(profile, path)
    raw = json.loads(path.read_text())
    assert raw["target_id"] == "TARGET_001"
    assert raw["established_track_id"] == 42  # the raw id is still recorded, just not AS the id
