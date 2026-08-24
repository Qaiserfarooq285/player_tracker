"""Pure-logic unit tests for the Stage 0.5 source profiler classification decision table
(CLAUDE.md §5 Stage 0.5). Feeds synthetic `(cuts_per_min, motion_score)` pairs into
`src.pipeline.profiler._classify` — no video decoding (CLAUDE.md task spec: pure-logic unit tests
only). Thresholds are read from the real `configs/profile.yaml` (like
`tests/test_filename_convention.py` does for the filename regex) so this test exercises the
actual configured decision table, not a hand-copied duplicate of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import load_yaml
from src.common.types import SourceProfile
from src.pipeline.profiler import _classify

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profile_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "profile.yaml")


@pytest.fixture(scope="module")
def thresholds(profile_config) -> dict:
    classification_cfg = profile_config["classification"]
    motion_cfg = profile_config["motion"]
    return {
        "broadcast_cuts_per_min": classification_cfg["broadcast_cuts_per_min_threshold"],
        "static": motion_cfg["static_motion_threshold"],
        "panning": motion_cfg["panning_motion_threshold"],
    }


def test_broadcast_wins_regardless_of_motion(profile_config, thresholds):
    # Many cuts/min => broadcast, even with dead-static motion (multi-camera cut pattern).
    profile, notes = _classify(
        n_cuts=20,
        cuts_per_min=thresholds["broadcast_cuts_per_min"] + 1,
        motion_score=0.0,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.BROADCAST
    assert notes == []  # no ambiguity, no single-cam-with-cuts note (it IS broadcast)


def test_broadcast_threshold_is_inclusive(profile_config, thresholds):
    profile, _ = _classify(
        n_cuts=3,
        cuts_per_min=thresholds["broadcast_cuts_per_min"],
        motion_score=0.0,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.BROADCAST


def test_single_static_below_static_threshold(profile_config, thresholds):
    profile, notes = _classify(
        n_cuts=0,
        cuts_per_min=0.0,
        motion_score=thresholds["static"] - 0.5,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.SINGLE_STATIC
    assert notes == []  # not ambiguous, no cuts to note


def test_single_panning_above_panning_threshold(profile_config, thresholds):
    profile, notes = _classify(
        n_cuts=0,
        cuts_per_min=0.0,
        motion_score=thresholds["panning"] + 1.0,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.SINGLE_PANNING
    assert notes == []


def test_ambiguous_band_nearer_static(profile_config, thresholds):
    static_thr, panning_thr = thresholds["static"], thresholds["panning"]
    # Just above the static threshold -> nearer to "static" than "panning".
    motion = static_thr + (panning_thr - static_thr) * 0.1
    profile, notes = _classify(
        n_cuts=0, cuts_per_min=0.0, motion_score=motion, profile_config=profile_config
    )
    assert profile == SourceProfile.SINGLE_STATIC
    assert any("ambiguous" in n for n in notes)


def test_ambiguous_band_nearer_panning(profile_config, thresholds):
    static_thr, panning_thr = thresholds["static"], thresholds["panning"]
    motion = static_thr + (panning_thr - static_thr) * 0.9
    profile, notes = _classify(
        n_cuts=0, cuts_per_min=0.0, motion_score=motion, profile_config=profile_config
    )
    assert profile == SourceProfile.SINGLE_PANNING
    assert any("ambiguous" in n for n in notes)


def test_ambiguous_band_midpoint_breaks_toward_static(profile_config, thresholds):
    static_thr, panning_thr = thresholds["static"], thresholds["panning"]
    midpoint = (static_thr + panning_thr) / 2.0
    profile, notes = _classify(
        n_cuts=0, cuts_per_min=0.0, motion_score=midpoint, profile_config=profile_config
    )
    # dist_to_static == dist_to_panning at the exact midpoint; ties resolve to static (<=).
    assert profile == SourceProfile.SINGLE_STATIC
    assert any("ambiguous" in n for n in notes)


def test_single_camera_with_cuts_gets_adr7_note(profile_config, thresholds):
    """The clip4 case: single-camera (not broadcast) profile that still has real cuts (ADR-7)."""
    profile, notes = _classify(
        n_cuts=1,
        cuts_per_min=0.5,
        motion_score=thresholds["static"] - 0.5,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.SINGLE_STATIC
    assert any("cut" in n.lower() and "ADR-7" in n for n in notes)


def test_single_camera_zero_cuts_no_adr7_note(profile_config, thresholds):
    profile, notes = _classify(
        n_cuts=0,
        cuts_per_min=0.0,
        motion_score=thresholds["static"] - 0.5,
        profile_config=profile_config,
    )
    assert profile == SourceProfile.SINGLE_STATIC
    assert notes == []


@pytest.mark.parametrize(
    "cuts_per_min,motion_offset,expected",
    [
        (0.0, -0.5, SourceProfile.SINGLE_STATIC),  # clip1/2/3/5-like: 0 cuts, near-static motion
    ],
)
def test_real_footage_shape_classifies_single_static(
    profile_config, thresholds, cuts_per_min, motion_offset, expected
):
    """Sanity check against the measured shape of CLAUDE.md §3.2's footage (near-static, ~0
    cuts/min) without hard-coding the exact measured motion_score (that's the smoke test's job)."""
    profile, _ = _classify(
        n_cuts=0,
        cuts_per_min=cuts_per_min,
        motion_score=thresholds["static"] + motion_offset,
        profile_config=profile_config,
    )
    assert profile == expected
