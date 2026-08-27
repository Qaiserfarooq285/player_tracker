"""Tests for the `clip<N> <jersey>` filename convention regex (CLAUDE.md §3.1) against the 5
real input clips, plus negative cases.

The regex is loaded straight from configs/run.yaml so this test exercises the actual config
value the pipeline will use, not a copy of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pattern() -> re.Pattern:
    config = yaml.safe_load((REPO_ROOT / "configs" / "run.yaml").read_text())
    return re.compile(config["filename_convention_regex"])


REAL_FILENAMES = [
    ("clip1 43.mp4", "1", "43"),
    ("clip2 77.mp4", "2", "77"),
    ("clip3 77.mp4", "3", "77"),
    ("clip4 77.mp4", "4", "77"),
    ("clip5 77.mp4", "5", "77"),
]


@pytest.mark.parametrize("filename,expected_n,expected_jersey", REAL_FILENAMES)
def test_real_filenames_match(pattern: re.Pattern, filename, expected_n, expected_jersey):
    stem = Path(filename).stem
    match = pattern.match(stem)
    assert match is not None, f"expected {filename!r} (stem {stem!r}) to match the convention"
    assert match.group("n") == expected_n
    assert match.group("jersey") == expected_jersey


NEGATIVE_CASES = [
    "clip1_43",  # underscore instead of space
    "clip 1 43",  # space between "clip" and the clip number
    "CLIP1 43",  # regex is case-sensitive, must be lowercase "clip"
    "clip1 43x",  # trailing junk after the jersey number
    "clip 43",  # missing the clip number entirely
    "clip1",  # missing the jersey number entirely
    "clip1  ",  # jersey number missing after trailing whitespace
    "random_video",  # doesn't follow the convention at all
    "clip1 43.mp4",  # extension left in (must apply the regex to the stem, not the full name)
]


@pytest.mark.parametrize("stem", NEGATIVE_CASES)
def test_negative_cases_do_not_match(pattern: re.Pattern, stem):
    assert pattern.match(stem) is None, f"expected {stem!r} NOT to match the convention"


def test_target_jersey_default_is_null():
    config = yaml.safe_load((REPO_ROOT / "configs" / "run.yaml").read_text())
    assert config["target_jersey"] is None


def test_input_clips_present_and_match(pattern: re.Pattern):
    """Sanity-check the actual files in input/, if present (input/ is gitignored so this is
    best-effort in environments where it's empty).

    CLAUDE.md §3.3/ADR-15: as of `Jordan Thomas Highlight Video.mp4` (a deliberate filename-less
    input, owner-authorized 2026-08-27), NOT every file in input/ is required to match the
    `clip<N> <jersey>` convention any more — `find_videos`/`parse_filename` (src/ingest/discovery)
    already handle a non-match by returning `target_jersey=None` rather than erroring. This test
    therefore only requires the ORIGINAL 5 `clip<N> <jersey>.mp4` files (still expected to match,
    unaffected by ADR-15) to match; any other filename found is logged, not asserted on, so this
    stays a real sanity check without re-imposing a convention ADR-15 explicitly lifted.
    """
    input_dir = REPO_ROOT / "input"
    if not input_dir.is_dir():
        pytest.skip("input/ not present in this environment")
    videos = [p for p in input_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".mov"}]
    if not videos:
        pytest.skip("no video files in input/ in this environment")
    clip_convention_videos = [v for v in videos if re.match(r"^clip\d+ ", v.stem)]
    for video in clip_convention_videos:
        match = pattern.match(video.stem)
        assert match is not None, f"input file {video.name!r} does not match the convention"
    non_matching = [v.name for v in videos if pattern.match(v.stem) is None]
    if non_matching:
        print(
            f"input/ contains {len(non_matching)} filename-less (ADR-15) video(s): {non_matching}"
        )
