"""Pure-logic unit tests for scripts/make_annotation_template.py (ADR-19, CLAUDE.md §14.3)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Load the script as a module without going through `sys.path` games twice (the script
    itself does its own `sys.path.insert` for standalone CLI use; importing it directly here
    exercises the exact same module)."""
    path = REPO_ROOT / "scripts" / "make_annotation_template.py"
    spec = importlib.util.spec_from_file_location("make_annotation_template", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mat = _load_module()


def test_build_template_text_lists_every_unverified_take():
    report = {
        "video": "input/match.mp4",
        "takes": [
            {"take_id": 0, "status": "verified", "confidence": 0.9},
            {"take_id": 1, "status": "unverified", "confidence": 0.0},
            {"take_id": 2, "status": "unverified", "confidence": 0.1},
        ],
    }
    text = mat.build_template_text(report)
    assert "Take 1:" in text
    assert "Take 2:" in text
    assert "Take 0:" not in text  # verified -- not named as needing a manual line
    assert text.count("Min MM:SS player #<N> in <colour> <action>") == 2


def test_build_template_text_all_verified_has_no_take_blocks():
    report = {
        "video": "input/match.mp4",
        "takes": [{"take_id": 0, "status": "verified", "confidence": 0.9}],
    }
    text = mat.build_template_text(report)
    assert "Take 0:" not in text
    assert "already VERIFIED" in text
    assert "Min MM:SS" not in text


def test_build_template_text_never_crashes_on_empty_takes():
    text = mat.build_template_text({"video": "input/match.mp4", "takes": []})
    assert "already VERIFIED" in text
