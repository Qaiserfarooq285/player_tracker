"""Stage 4 — goal detection (CLAUDE.md §5 Stage 5 goals row; §3.2 consequence 3, Golden Rule 5).

Two code paths, deliberately kept apart:

- `detect_goals_scoreboard_delta` — the eventual BROADCAST-path detector (OCR a scoreboard
  region, watch for either side's digit incrementing). This is DEAD CODE on this project's
  current input: it is never called from `src/pipeline/run.py` for the owner's clips, and it
  intentionally stops short of a real OCR/parsing implementation (raises `NotImplementedError`)
  rather than guessing at goal-detection behaviour against a scoreboard we already know does not
  exist in any of this footage (CLAUDE.md §3.2(3)). It exists so the broadcast branch is present
  in code per CLAUDE.md §5's stage table, ready to be finished once real broadcast/scoreboard
  input is available to validate it against.
- `check_goal_availability` — the ACTUAL Phase-1 entry point `src/pipeline/run.py` calls. It
  reports the already-established fact (§3.2(3): no scoreboard anywhere in this footage) directly,
  rather than running a doomed OCR pass for the sole purpose of getting a "no digits found" result
  we could have predicted for free. Golden Rule 5 forbids fabricating a goal count from absent
  evidence, so this returns an explicit "not available", never a guessed/heuristic one.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.types import Event, RunProfile

logger = get_logger(__name__)


class GoalDetectionResult(BaseModel):
    """Outcome of Stage 4's goal-detection attempt for one video (Golden Rule 5: explicit, not
    fabricated). `reason` is always human-readable and, when `available` is `False`, always
    literally starts with the string ``"not available"`` — the run report surfaces it verbatim.
    """

    available: bool
    reason: str
    events: list[Event] = []


def detect_goals_scoreboard_delta(
    frames: list[np.ndarray],
    times: list[float],
    goal_cfg: dict,
) -> list[Event]:
    """Broadcast-path scoreboard-delta goal detector — **DEAD CODE**, never invoked on this
    project's current (single-camera, scoreboard-less) input. See module docstring.

    Would OCR `goal_cfg["scoreboard_region_relative"]` on each frame (EasyOCR, matching
    `configs/shots.yaml: ocr`'s cadence/engine), parse each side's digit, and emit a `GOAL` event
    whenever a reading increments between two OCR passes. Stops at a `NotImplementedError` rather
    than a real (untestable, on this footage) OCR/parsing implementation — CLAUDE.md §3.2(3)
    established there is no scoreboard anywhere in this project's footage to validate the parsing
    logic against, so writing more of it now would be guessing at goal-post behaviour with zero
    ground truth to check it against.
    """
    raise NotImplementedError(
        "scoreboard-delta goal detection is a deliberately unfinished broadcast-path seam "
        "(CLAUDE.md §3.2(3)) -- it is never invoked on this project's single-camera, "
        "scoreboard-less input. See check_goal_availability() for what Stage 4 actually emits "
        "for goals on this footage."
    )


def check_goal_availability(profile: RunProfile | None, goal_cfg: dict) -> GoalDetectionResult:
    """The real Phase-1 goal-detection entry point (CLAUDE.md §3.2(3), Golden Rule 5).

    CLAUDE.md §3.2(3) measured (2026-08-24) that NONE of the owner's Veo clips show a scoreboard
    graphic anywhere. Running `detect_goals_scoreboard_delta` against a region we already know is
    empty would only manufacture "no digits found" noise for zero benefit, so this does not even
    attempt the OCR pass — it reports the *known* absence directly, unconditionally, for Phase 1's
    actual input set. `profile.profile == SourceProfile.BROADCAST` is the intended future trigger
    for calling `detect_goals_scoreboard_delta` for real; that branch is not exercised here since
    none of this project's clips are broadcast source.
    """
    reason = (
        "not available (no scoreboard detected in this footage -- CLAUDE.md §3.2(3): these are "
        "single-camera Veo youth-match clips with no broadcast scoreboard graphic anywhere; goal "
        "detection was not attempted rather than guessed, per Golden Rule 5)"
    )
    logger.info("goal detection: %s", reason)
    return GoalDetectionResult(available=False, reason=reason, events=[])
