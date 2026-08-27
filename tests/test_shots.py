"""Pure-logic unit tests for Stage 1 shot-boundary segmentation (CLAUDE.md §5 Stage 1, ADR-16).

Covers `takes_from_cut_frames` (cut-frame -> gapless Take list) and
`resolve_scenedetect_threshold` (ADR-16's per-video override lookup) — both pure/config-only, no
video I/O, so these run with no GPU/network.
"""

from __future__ import annotations

from src.shots.boundaries import resolve_scenedetect_threshold, takes_from_cut_frames


# ---------------------------------------------------------------------------
# takes_from_cut_frames
# ---------------------------------------------------------------------------


def test_takes_from_cut_frames_no_cuts_is_one_take():
    takes = takes_from_cut_frames([], n_frames=300, fps=30.0)
    assert len(takes) == 1
    assert takes[0].frame_start == 0
    assert takes[0].frame_end == 300


def test_takes_from_cut_frames_gapless_coverage():
    takes = takes_from_cut_frames([100, 200], n_frames=300, fps=30.0)
    assert len(takes) == 3
    assert [t.frame_start for t in takes] == [0, 100, 200]
    assert [t.frame_end for t in takes] == [100, 200, 300]


def test_takes_from_cut_frames_ignores_out_of_range_and_duplicates():
    takes = takes_from_cut_frames([0, 100, 100, 300, 400], n_frames=300, fps=30.0)
    assert [t.frame_start for t in takes] == [0, 100]
    assert [t.frame_end for t in takes] == [100, 300]


def test_takes_from_cut_frames_empty_video():
    assert takes_from_cut_frames([50], n_frames=0, fps=30.0) == []


# ---------------------------------------------------------------------------
# resolve_scenedetect_threshold (ADR-16)
# ---------------------------------------------------------------------------


def test_resolve_threshold_falls_back_to_global_default_when_no_override():
    sd_cfg = {"threshold": 38.0, "overrides": {"jordan_thomas_highlight_video": 24.0}}
    # every one of the original 5 clips has no entry in `overrides`
    assert resolve_scenedetect_threshold(sd_cfg, "clip4_77") == 38.0
    assert resolve_scenedetect_threshold(sd_cfg, "clip2_77") == 38.0


def test_resolve_threshold_uses_per_video_override_when_present():
    sd_cfg = {"threshold": 38.0, "overrides": {"jordan_thomas_highlight_video": 24.0}}
    assert resolve_scenedetect_threshold(sd_cfg, "jordan_thomas_highlight_video") == 24.0


def test_resolve_threshold_handles_missing_overrides_key():
    # a shots.yaml without any `overrides` key at all (pre-ADR-16 shape) must not KeyError
    sd_cfg = {"threshold": 38.0}
    assert resolve_scenedetect_threshold(sd_cfg, "anything") == 38.0


def test_resolve_threshold_matches_real_configs_shots_yaml():
    from src.common.io import load_yaml

    sd_cfg = load_yaml("configs/shots.yaml")["scenedetect"]
    assert resolve_scenedetect_threshold(sd_cfg, "jordan_thomas_highlight_video") == 24.0
    assert resolve_scenedetect_threshold(sd_cfg, "clip4_77") == 38.0
