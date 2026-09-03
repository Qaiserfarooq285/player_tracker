"""Pure-logic unit tests for `src/events/kit_colour.py` (Stage 5b of the evidence-based event
redesign, owner request 2026-09-02: kit colour as the primary teammate signal for pass/assist).

`kit_colour_same_team`'s fixtures use the ACTUAL real Lab values measured 2026-09-02 on
`chelsea_burnley_target10` (6 visually-confirmed tracks, grass-suppressed) -- this test file is
also the record of that validation, not just synthetic edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.io import load_yaml
from src.events import kit_colour

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")["kit_colour"]


# Real measured grass-suppressed CIELAB, chelsea_burnley_target10 take 0, 2026-09-02.
BLUE_10 = np.array([44.9, 7.5, -24.0])  # track 1, Chelsea #10, visually confirmed
BLUE_CTRL_A = np.array([47.2, 6.0, -22.0])  # track 13, Chelsea, confirmed
BLUE_CTRL_B = np.array([47.5, 6.5, -9.0])  # track 36, Chelsea, confirmed -- the noisy outlier
CLARET_21 = np.array([52.9, 5.0, -1.0])  # track 47, Burnley #21, visually confirmed
CLARET_CTRL_A = np.array([55.5, -1.0, 1.0])  # track 43, Burnley, confirmed
CLARET_CTRL_B = np.array([72.7, -6.0, 5.0])  # track 37, Burnley, confirmed -- occluded/blurry crop


def test_none_input_is_undecided():
    assert kit_colour.kit_colour_same_team(None, BLUE_10, _cfg()) == (None, 0.0)
    assert kit_colour.kit_colour_same_team(BLUE_10, None, _cfg()) == (None, 0.0)
    assert kit_colour.kit_colour_same_team(None, None, _cfg()) == (None, 0.0)


def test_confirmed_same_team_pair_reads_true():
    """Track 1 and track 13: both real Chelsea blue, dE=3.4 -- must read confidently same-team."""
    same, dist = kit_colour.kit_colour_same_team(BLUE_10, BLUE_CTRL_A, _cfg())
    assert same is True
    assert dist < _cfg()["same_team_max_dist"]


def test_confirmed_same_team_claret_pair_reads_true():
    """Track 47 and track 43: both real Burnley claret, dE=6.8."""
    same, dist = kit_colour.kit_colour_same_team(CLARET_21, CLARET_CTRL_A, _cfg())
    assert same is True


def test_confirmed_cross_team_pair_reads_false():
    """Track 1 (blue) vs track 47 (claret): the main verified pair, dE=24.5 -- the exact real
    click-to-track pairing validated earlier the same day. This is the pair diff_team_min_dist=24.0
    (rather than a rounder 25.0) is specifically tuned to resolve confidently."""
    same, dist = kit_colour.kit_colour_same_team(BLUE_10, CLARET_21, _cfg())
    assert same is False
    assert dist > _cfg()["diff_team_min_dist"]


def test_confirmed_cross_team_pair_reads_false_second_example():
    """Track 13 (blue) vs track 43 (claret), dE=25.4 -- a second, independent cross-team pair."""
    same, _dist = kit_colour.kit_colour_same_team(BLUE_CTRL_A, CLARET_CTRL_A, _cfg())
    assert same is False


def test_noisy_outlier_pairs_abstain_rather_than_guess_wrong():
    """The two noisy real tracks (36's own suppressed sample sits unusually far from its own
    teammate; 37's crop was partially occluded) must NOT produce a confident wrong verdict.
    track36-track47 is actually CROSS-team but measures only dE=9.8 (close to same_team_max) --
    must NOT read as True (would be a false positive crediting a pass to an opponent)."""
    same, dist = kit_colour.kit_colour_same_team(BLUE_CTRL_B, CLARET_21, _cfg())
    assert same is None, "must abstain, not falsely confirm a cross-team pair as same-team"


def test_noisy_same_team_pair_also_abstains_rather_than_guess_wrong():
    """track47-track37 is actually SAME-team (both claret) but measures dE=23.4 (near
    diff_team_min_dist) -- must NOT read as False (would be a false negative, e.g. wrongly
    calling a real pass a turnover)."""
    same, dist = kit_colour.kit_colour_same_team(CLARET_21, CLARET_CTRL_B, _cfg())
    assert same is None, "must abstain, not falsely deny a same-team pair"


def test_identical_colour_is_confidently_same_team():
    lab = np.array([50.0, 10.0, -10.0])
    same, dist = kit_colour.kit_colour_same_team(lab, lab.copy(), _cfg())
    assert same is True
    assert dist == 0.0


# ---------------------------------------------------------------------------
# median_kit_lab
# ---------------------------------------------------------------------------


def test_median_kit_lab_empty_is_none():
    assert kit_colour.median_kit_lab([]) is None


def test_median_kit_lab_robust_to_one_outlier_sample():
    samples = [BLUE_10, BLUE_10, np.array([200.0, 200.0, 200.0])]  # one wild outlier
    median = kit_colour.median_kit_lab(samples)
    assert median is not None
    assert np.allclose(median, BLUE_10)  # median, not mean, ignores the outlier


# ---------------------------------------------------------------------------
# kit_lab_sample -- pure geometry/masking behaviour, no real image needed for the reject paths
# ---------------------------------------------------------------------------


def test_kit_lab_sample_degenerate_bbox_returns_none():
    from src.common.io import load_yaml as _ly
    from src.common.types import BBox

    identity_cfg = _ly(REPO_ROOT / "configs" / "identity.yaml")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = BBox(x1=50, y1=50, x2=50.5, y2=50.5)  # too small after number_region insets
    result = kit_colour.kit_lab_sample(
        frame, bbox, identity_cfg["parseq_soccernet"]["number_crop"], _cfg()
    )
    assert result is None


def test_kit_lab_sample_all_grass_returns_none():
    """A crop that is ENTIRELY pitch-green must yield no sample -- min_kept_pixel_frac/
    min_kept_pixels both correctly reject a torso region with no visible kit at all."""
    from src.common.io import load_yaml as _ly
    from src.common.types import BBox

    identity_cfg = _ly(REPO_ROOT / "configs" / "identity.yaml")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:, :] = (40, 180, 40)  # solid saturated green (BGR) -- pure grass hue
    bbox = BBox(x1=20, y1=20, x2=100, y2=180)
    result = kit_colour.kit_lab_sample(
        frame, bbox, identity_cfg["parseq_soccernet"]["number_crop"], _cfg()
    )
    assert result is None
