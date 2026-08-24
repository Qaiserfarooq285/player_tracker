"""Pure-logic unit tests for Stage 0 ingest (CLAUDE.md §5 Stage 0): filename parsing, Take
construction from synthetic cut lists, and chunking math. No video decoding here (CLAUDE.md task
spec: "no video decoding in unit tests") — decode-touching behaviour is exercised manually /
in the smoke test, not in this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.ingest.decode import chunk_ranges
from src.ingest.discovery import parse_filename
from src.shots.boundaries import takes_from_cut_frames

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def filename_pattern() -> re.Pattern[str]:
    config = yaml.safe_load((REPO_ROOT / "configs" / "run.yaml").read_text())
    return re.compile(config["filename_convention_regex"])


# ---------------------------------------------------------------------------
# parse_filename (clip<N> <jersey> -> target_jersey)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected_clip_index,expected_jersey",
    [
        ("clip1 43.mp4", 1, 43),
        ("clip2 77.mp4", 2, 77),
        ("clip3 77.mp4", 3, 77),
        ("clip4 77.mp4", 4, 77),
        ("clip5 77.mp4", 5, 77),
        ("clip10 9.mp4", 10, 9),
    ],
)
def test_parse_filename_positive(filename_pattern, filename, expected_clip_index, expected_jersey):
    stem = Path(filename).stem
    clip_index, target_jersey = parse_filename(stem, filename_pattern)
    assert clip_index == expected_clip_index
    assert target_jersey == expected_jersey


@pytest.mark.parametrize(
    "stem",
    [
        "clip1_43",  # underscore instead of space
        "clip 1 43",  # space between "clip" and the clip number
        "CLIP1 43",  # case-sensitive
        "clip1 43x",  # trailing junk
        "clip 43",  # missing clip number
        "clip1",  # missing jersey number
        "clip1  ",  # jersey missing after trailing whitespace
        "random_video",  # doesn't follow the convention
        "clip1 43.mp4",  # extension left in
    ],
)
def test_parse_filename_negative(filename_pattern, stem):
    clip_index, target_jersey = parse_filename(stem, filename_pattern)
    assert clip_index is None
    assert target_jersey is None


# ---------------------------------------------------------------------------
# takes_from_cut_frames (synthetic cut lists -> gapless, non-overlapping Take list)
# ---------------------------------------------------------------------------


def _assert_full_coverage_no_gaps_no_overlaps(takes, n_frames):
    assert takes, "expected at least one take"
    assert takes[0].frame_start == 0
    assert takes[-1].frame_end == n_frames
    for i, take in enumerate(takes):
        assert take.id == i
        assert take.frame_start < take.frame_end
        assert take.kind == "unknown"
    for a, b in zip(takes, takes[1:], strict=False):
        assert a.frame_end == b.frame_start  # gapless, non-overlapping


def test_takes_no_cuts_single_take_spans_whole_video():
    takes = takes_from_cut_frames(cut_frames=[], n_frames=300, fps=30.0)
    _assert_full_coverage_no_gaps_no_overlaps(takes, 300)
    assert len(takes) == 1
    assert takes[0].t_start == pytest.approx(0.0)
    assert takes[0].t_end == pytest.approx(10.0)


def test_takes_single_cut_clip4_like():
    # clip4 77.mp4: 110.88s @ 30fps -> 3326 frames, one cut at ~t=60.07s -> frame ~1802
    n_frames = 3326
    fps = 30.0
    cut_frame = round(60.07 * fps)
    takes = takes_from_cut_frames(cut_frames=[cut_frame], n_frames=n_frames, fps=fps)
    _assert_full_coverage_no_gaps_no_overlaps(takes, n_frames)
    assert len(takes) == 2
    assert takes[0].frame_start == 0
    assert takes[0].frame_end == cut_frame
    assert takes[1].frame_start == cut_frame
    assert takes[1].frame_end == n_frames
    assert takes[1].t_start == pytest.approx(60.07, abs=0.02)


def test_takes_multiple_cuts_full_coverage():
    n_frames = 1000
    fps = 25.0
    takes = takes_from_cut_frames(cut_frames=[100, 400, 900], n_frames=n_frames, fps=fps)
    _assert_full_coverage_no_gaps_no_overlaps(takes, n_frames)
    assert len(takes) == 4
    assert [t.frame_start for t in takes] == [0, 100, 400, 900]
    assert [t.frame_end for t in takes] == [100, 400, 900, n_frames]


def test_takes_ignores_out_of_range_and_duplicate_cuts():
    n_frames = 500
    fps = 30.0
    # 0 and n_frames are not valid interior cut points; -5/600 are out of range; 200 is duplicated
    takes = takes_from_cut_frames(
        cut_frames=[0, -5, 200, 200, 600, n_frames], n_frames=n_frames, fps=fps
    )
    _assert_full_coverage_no_gaps_no_overlaps(takes, n_frames)
    assert [t.frame_start for t in takes] == [0, 200]


def test_takes_empty_video_returns_no_takes():
    assert takes_from_cut_frames(cut_frames=[], n_frames=0, fps=30.0) == []


# ---------------------------------------------------------------------------
# chunk_ranges (chunking math)
# ---------------------------------------------------------------------------


def test_chunk_ranges_110_88s_at_180s_is_one_chunk():
    ranges = chunk_ranges(duration=110.88, chunk_seconds=180)
    assert ranges == [(0.0, 110.88)]


def test_chunk_ranges_110_88s_at_60s_is_two_chunks():
    ranges = chunk_ranges(duration=110.88, chunk_seconds=60)
    assert len(ranges) == 2
    assert ranges[0] == (0.0, 60.0)
    assert ranges[1][0] == 60.0
    assert ranges[1][1] == pytest.approx(110.88)


def test_chunk_ranges_exact_multiple_no_dangling_short_chunk():
    ranges = chunk_ranges(duration=120.0, chunk_seconds=60.0)
    assert ranges == [(0.0, 60.0), (60.0, 120.0)]


def test_chunk_ranges_shorter_than_one_chunk():
    ranges = chunk_ranges(duration=11.06, chunk_seconds=180)
    assert ranges == [(0.0, 11.06)]


def test_chunk_ranges_cover_full_duration_with_no_gaps():
    duration, chunk_seconds = 373.4, 47.0
    ranges = chunk_ranges(duration, chunk_seconds)
    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == pytest.approx(duration)
    for a, b in zip(ranges, ranges[1:], strict=False):
        assert a[1] == b[0]  # gapless


def test_chunk_ranges_zero_or_negative_duration_is_empty():
    assert chunk_ranges(duration=0.0, chunk_seconds=180) == []
    assert chunk_ranges(duration=-5.0, chunk_seconds=180) == []
