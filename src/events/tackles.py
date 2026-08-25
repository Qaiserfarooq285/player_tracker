"""Best-effort tackle detection (ADR-13/14; CLAUDE.md §13.3 "Touch, Pass, Tackle, Save" row).

⚠️ Owner-authorized exception to Golden Rule 7 (CLAUDE.md §12/ADR-13): arithmetic on cached
`Track`/`BallDetection` artifacts, no training/new model (Golden Rule 2). A "tackle" here is a
possession change (`src/events/possession.py::possession_runs_for_take`) from identity A to a
confidently-OPPOSING identity B (`possession.teammates_gate`, inverted), where B was measurably
CLOSING distance on the ball just beforehand. This chains THREE separate heuristics (a possession
change is itself inferred; team-opposition is a soft ADR-12 signal; closing speed is a proxy for
"about to challenge", not contact) — accordingly this is one of the two lowest-confidence-ceiling
categories in `configs/events.yaml` (the other being `save`, see `src/events/saves.py`), and it
will read as noisy on any footage where team labels are weak (CLAUDE.md §3.2/ADR-12 — see
`configs/events.yaml: pass.team_confidence_threshold`'s own measured clip2/clip4 numbers, reused
here verbatim since it's the identical "is the team signal trustworthy at all" question).

**Attribution decision (CLAUDE.md task: "decide and document which direction you attribute")**:
the emitted `player_track_id` is the TACKLER (B), matching the conventional "Tackles: N" football
stat as tackles a player MADE (CLAUDE.md §13.2's stat-card template), not tackles they suffered.
`Event.evidence` carries both `tackler_identity`/`tackled_identity` so a future consumer could
still derive "times dispossessed" for the other player from the same event stream without
re-running this heuristic (Golden Rule 5 traceability).
"""

from __future__ import annotations

import uuid

from src.common.types import BallDetection, Event, EventType, Track
from src.events.possession import (
    DEFAULT_BALL_FPS,
    PossessionRun,
    possession_quality,
    teammates_gate,
)
from src.events.touches import nearest_box_in_time


def _ball_position_near(
    balls: list[BallDetection], t: float, tolerance_s: float
) -> tuple[float, float] | None:
    """Nearest-in-time ball centroid to `t`, within `tolerance_s` seconds, or `None`."""
    best: tuple[float, float] | None = None
    best_dt = tolerance_s
    for b in balls:
        dt = abs(b.t - t)
        if dt <= best_dt:
            best_dt = dt
            best = (b.bbox.cx, b.bbox.cy)
    return best


def closing_speed(
    track_b: Track,
    balls: list[BallDetection],
    t_before: float,
    t_at: float,
    match_tolerance_s: float,
) -> float | None:
    """How fast track `track_b` closed the distance to the BALL between `t_before` and `t_at`,
    in bbox-heights/second (same normalised unit as `sprint`/`tackle.tackle_min_closing_speed`) —
    positive means the gap shrank (closing in), negative means it grew. `None` when either
    instant's box/ball position can't be resolved within `match_tolerance_s`, or `t_at <= t_before`
    (never guessed, Golden Rule 5).
    """
    if t_at <= t_before:
        return None
    box_before = nearest_box_in_time(track_b, t_before, match_tolerance_s)
    box_at = nearest_box_in_time(track_b, t_at, match_tolerance_s)
    ball_before = _ball_position_near(balls, t_before, match_tolerance_s)
    ball_at = _ball_position_near(balls, t_at, match_tolerance_s)
    if box_before is None or box_at is None or ball_before is None or ball_at is None:
        return None

    dist_before = (
        (box_before.bbox.cx - ball_before[0]) ** 2 + (box_before.bbox.cy - ball_before[1]) ** 2
    ) ** 0.5
    dist_at = ((box_at.bbox.cx - ball_at[0]) ** 2 + (box_at.bbox.cy - ball_at[1]) ** 2) ** 0.5
    mean_h = (box_before.bbox.height + box_at.bbox.height) / 2.0
    if mean_h <= 0:
        return None
    return ((dist_before - dist_at) / (t_at - t_before)) / mean_h


def tackle_confidence(
    quality_a: float, quality_b: float, closing: float, tackle_cfg: dict
) -> float:
    """`mean(quality_a, quality_b) * speed_factor`, clamped to `tackle`'s own (lowest-of-all-
    categories) ceiling. `speed_factor` scales from 0.5 AT exactly `tackle_min_closing_speed` up
    to a cap of 1.0 at 2x that speed — closing in at merely the qualifying threshold shouldn't read
    as fully confident on its own, but an unrealistically fast closing speed shouldn't dominate the
    score either (same spirit as `src/events/shots.py::shot_confidence`'s own bounded excess-based
    scaling)."""
    threshold = tackle_cfg["tackle_min_closing_speed"]
    margin = closing / threshold if threshold > 0 else 1.0
    speed_factor = min(1.0, margin / 2.0)
    raw = ((quality_a + quality_b) / 2.0) * speed_factor
    return min(tackle_cfg["max_confidence"], max(tackle_cfg["min_confidence"], raw))


def detect_tackles(
    runs: list[PossessionRun],
    tracks: list[Track],
    balls: list[BallDetection],
    take_id: int | None,
    events_cfg: dict,
    ball_fps: float = DEFAULT_BALL_FPS,
    identity_confidence: dict[int, float] | None = None,
    drops=None,
) -> list[Event]:
    """`TACKLE` events between adjacent runs (from `possession.possession_runs_for_take`, same
    take, same "no other identity intervened" guarantee `detect_passes` relies on) by DIFFERENT,
    confidently-OPPOSING identities, within `tackle_max_duration_s`, where the new possessor (B)
    was closing distance on the ball at >= `tackle_min_closing_speed` over the preceding
    `tackle_lookback_s` (clamped to B's own track start, never extrapolated before it existed).

    Asserted defensively, same reasoning as `possession.possession_runs_for_take`/`detect_passes`:
    `tracks` must be one take's own (raw ids reset per take -- Golden Rule 3).
    """
    take_ids = {tr.take_id for tr in tracks}
    assert len(take_ids) <= 1, "detect_tackles must only ever see one take's tracks"

    tackle_cfg = events_cfg["tackle"]
    possession_cfg = events_cfg["possession"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    match_tolerance_s = possession_cfg["match_tolerance_s"]
    tracks_by_id = {tr.id: tr for tr in tracks}

    events: list[Event] = []
    for run_a, run_b in zip(runs, runs[1:], strict=False):
        if run_a.identity == run_b.identity:
            continue  # same person regaining the ball -- not a tackle

        gap = run_b.t_start - run_a.t_end
        if gap < 0 or gap > tackle_cfg["tackle_max_duration_s"]:
            if drops is not None:
                drops.drop("tackle_gap_too_large")
            continue

        raw_a = run_a.samples[-1][1]
        raw_b = run_b.samples[0][1]
        team_threshold = tackle_cfg["team_confidence_threshold"]
        opposes = teammates_gate(raw_a, raw_b, tracks_by_id, team_threshold)
        if opposes is not False:
            if drops is not None:
                reason = (
                    "tackle_team_confidence_too_low"
                    if opposes is None
                    else "tackle_same_team_not_opponent"
                )
                drops.drop(reason)
            continue

        track_b = tracks_by_id.get(raw_b)
        if track_b is None or not track_b.boxes:
            if drops is not None:
                drops.drop("tackle_missing_tackler_track")
            continue

        t_at = run_b.t_start
        t_before = max(t_at - tackle_cfg["tackle_lookback_s"], track_b.t_start)
        speed = closing_speed(track_b, balls, t_before, t_at, match_tolerance_s)
        if speed is None or speed < tackle_cfg["tackle_min_closing_speed"]:
            if drops is not None:
                drops.drop("tackle_insufficient_closing_speed")
            continue

        quality_a = possession_quality(run_a, possession_cfg, ball_fps)
        quality_b = possession_quality(run_b, possession_cfg, ball_fps)
        if identity_confidence is not None:
            quality_a *= identity_confidence.get(run_a.identity, 1.0)
            quality_b *= identity_confidence.get(run_b.identity, 1.0)
        confidence = tackle_confidence(quality_a, quality_b, speed, tackle_cfg)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("tackle_below_min_emit_confidence")
            continue

        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.TACKLE,
                t_start=t_before,
                t_end=t_at,
                player_track_id=run_b.identity,
                take_id=take_id,
                confidence=confidence,
                source="possession_change_closing_speed_heuristic",
                evidence={
                    "tackler_identity": run_b.identity,
                    "tackled_identity": run_a.identity,
                    "tackler_raw_track_id": raw_b,
                    "tackled_raw_track_id": raw_a,
                    "closing_speed": speed,
                    "tackle_min_closing_speed": tackle_cfg["tackle_min_closing_speed"],
                    "gap_s": gap,
                    "unit": "bbox_heights_per_second",
                    "calibrated": False,
                },
            )
        )
    return events
