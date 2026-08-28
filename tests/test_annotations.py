"""Pure-logic unit tests for src/annotations/parse.py (ADR-19, CLAUDE.md §14.3) -- no I/O beyond a
tmp_path sidecar file.
"""

from __future__ import annotations

from pathlib import Path

from src.annotations.parse import annotation_to_event, parse_annotations
from src.common.io import load_yaml
from src.common.types import EventType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _annotations_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "annotations.yaml")


CANONICAL_SIDECAR = """\
# a source URL or comment line is allowed and ignored
https://www.youtube.com/watch?v=wzvM66c8ZmE
Min 26:56 player #2 in white takes a touch
Min 26:58 player #2 in white makes a pass
Min 27:28 player #2 in white kicks the ball out of bounds
"""


def test_canonical_three_lines_parse_exactly(tmp_path):
    """Acceptance test #1: the three canonical lines parse to exactly
    [TOUCH@1616.0s, PASS@1618.0s, OUT_OF_BOUNDS@1648.0s], all jersey 2 / colour white."""
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text(CANONICAL_SIDECAR)

    annotations, problems = parse_annotations(sidecar, _annotations_config())

    assert problems == []
    assert len(annotations) == 3

    expected = [
        (EventType.TOUCH, 1616.0),
        (EventType.PASS, 1618.0),
        (EventType.OUT_OF_BOUNDS, 1648.0),
    ]
    for ann, (etype, t) in zip(annotations, expected, strict=True):
        assert ann.event_type == etype
        assert ann.t == t
        assert ann.jersey_number == 2
        assert ann.team_colour == "white"


def test_comment_and_url_lines_are_ignored_not_treated_as_problems(tmp_path):
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text(CANONICAL_SIDECAR)
    _annotations, problems = parse_annotations(sidecar, _annotations_config())
    assert problems == []


def test_malformed_line_lands_in_problems_not_dropped_not_crashing(tmp_path):
    """A deliberately malformed line (missing the required '#<jersey>' token) must be preserved in
    `problems`, not silently dropped, and must not raise."""
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text(
        "Min 26:56 player #2 in white takes a touch\n"
        "Min 40:00 player white takes a shot\n"  # malformed: no '#<jersey>'
    )

    annotations, problems = parse_annotations(sidecar, _annotations_config())

    assert len(annotations) == 1
    assert len(problems) == 1
    assert "line 2" in problems[0]
    assert "Min 40:00 player white takes a shot" in problems[0]


def test_unmapped_phrase_becomes_key_moment_with_raw_text_preserved(tmp_path):
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text("Min 10:00 player #9 in red does a rainbow flick\n")

    annotations, problems = parse_annotations(sidecar, _annotations_config())

    assert problems == []
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann.event_type == EventType.KEY_MOMENT
    assert ann.action_phrase == "does a rainbow flick"


def test_celebration_phrase_maps_to_key_moment_not_a_new_event_type(tmp_path):
    """CLAUDE.md §14.3: 'celebrat' maps to the EXISTING KEY_MOMENT EventType, never a new
    CELEBRATION type -- the timeline label ("Celebration") is a rendering-time text heuristic on
    top of KEY_MOMENT, not a second representation of the same concept."""
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text("Min 15:00 player #2 in white celebrates with the fans\n")

    annotations, _problems = parse_annotations(sidecar, _annotations_config())
    assert annotations[0].event_type == EventType.KEY_MOMENT


def test_hour_prefixed_time_parses():
    cfg = _annotations_config()
    from src.annotations.parse import _parse_time

    assert _parse_time("1:02:03") == 3723.0
    assert _parse_time("26:56") == 1616.0
    assert cfg["time_pattern"]  # sanity: config key exists and is truthy


def test_time_prefix_is_optional(tmp_path):
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text("26:56 player #2 in white takes a touch\n")  # no "Min " prefix
    annotations, problems = parse_annotations(sidecar, _annotations_config())
    assert problems == []
    assert annotations[0].t == 1616.0


def test_blank_lines_are_skipped(tmp_path):
    sidecar = tmp_path / "match.annotations.txt"
    sidecar.write_text("\n\nMin 26:56 player #2 in white takes a touch\n\n")
    annotations, problems = parse_annotations(sidecar, _annotations_config())
    assert problems == []
    assert len(annotations) == 1


def test_annotation_to_event_uses_event_window_and_default_confidence():
    cfg = _annotations_config()
    from src.common.types import Annotation

    ann = Annotation(
        t=100.0,
        jersey_number=2,
        team_colour="white",
        action_phrase="takes a touch",
        event_type=EventType.TOUCH,
        raw_line="Min 1:40 player #2 in white takes a touch",
    )
    ev = annotation_to_event(ann, cfg, take_id=3)
    assert ev.type == EventType.TOUCH
    assert ev.t_start == 100.0
    assert ev.t_end == 100.0 + cfg["event_window_seconds"]
    assert ev.confidence == cfg["default_confidence"]
    assert ev.source == "manual_annotation"
    assert ev.player_track_id is None
    assert ev.take_id == 3
    assert ev.evidence["jersey_number"] == 2
    assert ev.evidence["team_colour"] == "white"
    assert ev.evidence["raw_line"] == ann.raw_line
