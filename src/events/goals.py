"""Stage 4 — REAL goal + assist detection (ADR-17, CLAUDE.md §5.1/§3.2 consequence 3, Golden
Rule 5).

Supersedes the earlier dead-code stub: `detect_goals_scoreboard_delta` used to raise
`NotImplementedError` unconditionally, and `check_goal_availability` used to return a hardcoded
"not available" regardless of input. Both are now real, and — measured, not assumed — still
correctly report "not available" on all 5 original `clip<N> <jersey>.mp4` clips plus
`jordan_thomas_highlight_video`, because none of them has a scoreboard graphic anywhere
(CLAUDE.md §3.2(3)). ADR-17's own design, in four steps:

1. **Occurrence** (`detect_goals_scoreboard_delta`) — scan a SMALL set of candidate scoreboard
   corner regions (`configs/events.yaml: goal.candidate_regions`) across several early sampled
   frames of a take with the project's existing EasyOCR engine. A region only "activates" if it
   yields the SAME stable `digit [separator] digit` reading across multiple frames (a MEASURED
   self-check — `scan_candidate_regions` — never a hardcoded ROI guess, and never a trust in
   `RunProfile.profile`; ADR-11's own lesson in this codebase applies identically here: read the
   measured signal, not the label). Once activated, the region is watched across the whole take
   for a DEBOUNCED score increment (`find_score_increments`) — a reading must climb by exactly one
   and then STAY there for a couple of subsequent samples before it counts as a real goal, so a
   single OCR misread can never fire a phantom one.
2. **Scoring team** (`attribute_goal_team_and_scorer`) — deliberately sidesteps mapping the
   scoreboard's own left/right digit to home/away. Instead, whichever of OUR OWN two
   colour-clustered teams (`src/events/possession.py`, ADR-12) held the ball for the most
   cumulative time in the `possession_lookback_s` window immediately before the digit changed is
   credited.
3. **Scorer** (same function) — the identity with the LAST possession run in that window, gated to
   the credited team.
4. **Assist** (`find_assist_event`) — the owner's own rule, verbatim: "if the target pass the ball
   and the other score goal should be assist". The nearest PRECEDING `PASS` event whose
   `evidence['receiver_identity']` equals the credited scorer, from a teammate, within
   `assist_window_seconds`, credited to the PASSER.

Confidence stacks DOWN at each step (`goal_cfg['occurrence_confidence']` is the ceiling; each
attribution step only ever multiplies it further down) and every emitted `Event.evidence` carries
enough to audit it — which region/frames the OCR locked onto and its raw text, the possession
scores considered, and which `Event.id` supported an assist (Golden Rule 5: no black-box
confidence).

No visual ball-crosses-the-goal-line detection is built here (no goal-post/net detector exists,
and ADR-17 explicitly scoped that out as deferred, not part of this task).

`detect_goals_for_video` is the shared entrypoint both `src.pipeline.run` (the 5 original clips)
and `src.pipeline.extended_output` (ADR-15's filename-less flow) call identically.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.types import BallDetection, Event, EventType, Take, Track
from src.common.video import decode_frames
from src.events.possession import detect_passes, detect_possession, possession_runs_for_take
from src.events.shots import detect_shots
from src.identity.jersey_ocr import free_easyocr_reader, load_easyocr_reader

logger = get_logger(__name__)


class GoalDetectionResult(BaseModel):
    """Outcome of Stage 4's goal-detection attempt for one video (Golden Rule 5: explicit, not
    fabricated). `reason` is always human-readable and, when `available` is `False`, always
    literally starts with the string ``"not available"`` — the run report surfaces it verbatim.
    """

    available: bool
    reason: str
    events: list[Event] = []


# ---------------------------------------------------------------------------
# OCR text -> (left, right) parsing (pure)
# ---------------------------------------------------------------------------

_SCORE_PATTERN = re.compile(r"(\d{1,2})\s*[-:]\s*(\d{1,2})")


def parse_scoreboard_text(joined_text: str | None) -> tuple[int, int] | None:
    """Extract a plausible `digit [separator] digit` scoreboard reading (e.g. ``"1 - 0"``,
    ``"2:1"``) from a joined EasyOCR text string. `None` when no such pattern is present — the
    conservative, no-fabrication default (Golden Rule 5); EasyOCR frequently returns several
    fragments per crop (e.g. ``"1"``, ``"-"``, ``"0"``) or none at all.
    """
    if not joined_text:
        return None
    match = _SCORE_PATTERN.search(joined_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _bbox_x1(bbox: Any) -> float:
    """Best-effort left-x of an EasyOCR result box — real EasyOCR boxes are 4 `[x, y]` points, so
    ``bbox[0][0]`` is the top-left corner's x. Defensive against a test double's dummy bbox shape
    (e.g. the `(0, 0, 0, 0)` tuple `tests/test_identity.py` already uses for the sibling
    `jersey_ocr` module) — falls back to `0.0` rather than crashing on an unexpected shape.
    """
    try:
        return float(bbox[0][0])
    except (TypeError, IndexError):
        return 0.0


def _crop_region(
    frame: np.ndarray, region_relative: tuple[float, float, float, float]
) -> np.ndarray:
    """Crop `region_relative` (`[x1, y1, x2, y2]` as a fraction of frame width/height) out of
    `frame`."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region_relative
    return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def _ocr_region_text(reader: Any, crop: np.ndarray) -> str:
    """Run EasyOCR over `crop` with NO digit allowlist (unlike `src/identity/jersey_ocr.py`'s
    jersey-number OCR — a scoreboard reading needs the separator character too) and join every
    detected text fragment left-to-right into one string for `parse_scoreboard_text`."""
    if crop.size == 0:
        return ""
    raw = reader.readtext(crop)
    ordered = sorted(raw, key=lambda r: _bbox_x1(r[0]))
    return " ".join(str(text) for _bbox, text, _conf in ordered)


# ---------------------------------------------------------------------------
# step 1a: which (if any) candidate region is a real, stable scoreboard
# ---------------------------------------------------------------------------


def scan_candidate_regions(
    reader: Any, frames: list[np.ndarray], goal_cfg: dict
) -> tuple[tuple[float, float, float, float] | None, dict]:
    """Self-check WHICH (if any) of `goal_cfg['candidate_regions']` is a real scoreboard, from
    MEASURED OCR stability across `frames` — never a hardcoded ROI guess, never a trust in
    `RunProfile.profile` (ADR-11's own lesson: read the measured signal, not the label).

    A region "activates" when the SAME parsed `(left, right)` reading appears in at least
    `goal_cfg['min_stable_frames']` of `frames` — a real scoreboard graphic reads the same score
    frame after frame over a short early window (barring an implausibly fast kickoff goal), so
    stability across several samples is exactly the self-check that separates "this is a
    scoreboard" from "this crop of grass/sky/a sponsor board happened to OCR a plausible-looking
    digit pair once or twice".

    Returns `(activated_region_or_None, debug)` — `debug` always carries every candidate region's
    own reading tally (Golden Rule 5: auditable even when nothing activates), checked in
    `goal_cfg['candidate_regions']`'s own order and returning the FIRST region that activates.
    """
    candidates = goal_cfg["candidate_regions"]
    min_stable = goal_cfg["min_stable_frames"]
    debug: dict = {"candidates": [], "activated_region": None}

    for region in candidates:
        region_t = tuple(float(v) for v in region)
        readings: dict[tuple[int, int], int] = {}
        raw_texts: list[str] = []
        for frame in frames:
            text = _ocr_region_text(reader, _crop_region(frame, region_t))
            raw_texts.append(text)
            parsed = parse_scoreboard_text(text)
            if parsed is not None:
                readings[parsed] = readings.get(parsed, 0) + 1

        best_reading = max(readings.items(), key=lambda kv: kv[1]) if readings else None
        debug["candidates"].append(
            {
                "region": list(region_t),
                "raw_texts": raw_texts,
                "readings": {f"{k[0]}-{k[1]}": v for k, v in readings.items()},
                "best_count": best_reading[1] if best_reading else 0,
            }
        )
        if best_reading is not None and best_reading[1] >= min_stable:
            debug["activated_region"] = list(region_t)
            debug["activated_reading"] = list(best_reading[0])
            return region_t, debug

    return None, debug


# ---------------------------------------------------------------------------
# step 1b: debounced score-increment detection over one take's own low-fps series
# ---------------------------------------------------------------------------


def find_score_increments(samples: list[tuple[float, int]], goal_cfg: dict) -> list[dict]:
    """Walk a time-sorted `(t, total_score)` series (already parsed+summed from an activated
    region's own readings — gaps where OCR produced nothing are simply absent, never zero-filled)
    and return every DEBOUNCED `+1` increment as
    `{t_before, t_after, score_before, score_after}`.

    Debounce (`goal_cfg['increment_debounce_samples']`): a candidate `+1` step must hold at the
    NEW value for that many SUBSEQUENT samples before being trusted — a single-sample spike that
    reverts is OCR noise, not a goal. A jump of more than `+1`, or any decrease, is ambiguous (an
    OCR misread, or — if it turns out to persist — a real change this heuristic cannot tell how
    many goals it represents) and is never turned into a goal event (Golden Rule 5: no guessing);
    if such a reading itself proves debounce-stable, the baseline silently resyncs to it (so a
    later, cleaner `+1` from that new baseline can still be detected) without ever emitting an
    event for the ambiguous jump itself.
    """
    debounce = goal_cfg["increment_debounce_samples"]
    increments: list[dict] = []
    if not samples:
        return increments

    baseline_t, baseline_score = samples[0]
    i = 1
    while i < len(samples):
        t, score = samples[i]
        if score == baseline_score:
            i += 1
            continue

        window = samples[i : i + debounce]
        stable = len(window) == debounce and all(s == score for _, s in window)

        if score == baseline_score + 1 and stable:
            increments.append(
                {
                    "t_before": baseline_t,
                    "t_after": t,
                    "score_before": baseline_score,
                    "score_after": score,
                }
            )
            baseline_t, baseline_score = t, score
            i += debounce
            continue

        if stable:
            # An ambiguous (non-+1) but persistent change -- resync without emitting a goal.
            baseline_t, baseline_score = t, score
            i += debounce
            continue

        # Not stable across the debounce window -- a stray misread, ignore this one sample only.
        i += 1

    return increments


# ---------------------------------------------------------------------------
# step 1: occurrence detector (real implementation -- no longer NotImplementedError)
# ---------------------------------------------------------------------------


def detect_goals_scoreboard_delta(
    reader: Any,
    frames: list[np.ndarray],
    times: list[float],
    take_id: int | None,
    goal_cfg: dict,
) -> tuple[list[Event], dict]:
    """ADR-17 step 1 — the real scoreboard-OCR-delta goal-OCCURRENCE detector.

    `frames`/`times` are a LOW-FPS, time-sorted series spanning one take (`detect_goals_for_take`
    decodes them via `configs/events.yaml: goal.scoreboard_ocr_fps_sample`); this function itself
    is decode-agnostic, so it is independently testable against synthetic frames + a mocked
    EasyOCR `reader` (no GPU/video I/O needed).

    Team/scorer/assist attribution is a SEPARATE, later step (`attribute_goal_team_and_scorer` /
    `find_assist_event`) — every `Event` returned here has `player_track_id=None` and
    `confidence=goal_cfg['occurrence_confidence']` (ADR-17: the highest-confidence step in the
    chain; attribution only ever multiplies it further down), so this function stays a pure,
    auditable "did a goal OCCUR" detector.

    Returns `(occurrence_events, debug)`; `debug` is `scan_candidate_regions`'s own audit trail,
    plus (when a region activated) the raw per-sample score series and the increments found.
    """
    n_activation = min(len(frames), goal_cfg["activation_frames_count"])
    activated_region, debug = scan_candidate_regions(reader, frames[:n_activation], goal_cfg)
    if activated_region is None:
        return [], debug

    samples: list[tuple[float, int]] = []
    for t, frame in zip(times, frames, strict=True):
        crop = _crop_region(frame, activated_region)
        parsed = parse_scoreboard_text(_ocr_region_text(reader, crop))
        if parsed is not None:
            samples.append((t, parsed[0] + parsed[1]))

    increments = find_score_increments(samples, goal_cfg)
    debug["score_samples"] = samples
    debug["increments"] = increments

    events: list[Event] = []
    for inc in increments:
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.GOAL,
                t_start=inc["t_before"],
                t_end=inc["t_after"],
                player_track_id=None,
                take_id=take_id,
                confidence=goal_cfg["occurrence_confidence"],
                source="scoreboard_ocr_delta",
                evidence={
                    "activated_region": list(activated_region),
                    "score_before": inc["score_before"],
                    "score_after": inc["score_after"],
                    "t_before": inc["t_before"],
                    "t_after": inc["t_after"],
                },
            )
        )
    return events, debug


# ---------------------------------------------------------------------------
# steps 2-3: scoring team + scorer attribution
# ---------------------------------------------------------------------------


def attribute_goal_team_and_scorer(
    goal_event: Event,
    possession_events: list[Event],
    tracks_by_id: dict[int, Track],
    goal_cfg: dict,
) -> Event:
    """ADR-17 steps 2-3. Deliberately sidesteps mapping the scoreboard's own left/right digit to
    home/away — instead credits whichever of OUR OWN two colour-clustered teams (`Track.team`,
    ADR-12) held ball possession for the most cumulative time in the
    `goal_cfg['possession_lookback_s']` window immediately before the digit changed
    (`goal_event.evidence['t_after']`), via the already-computed `POSSESSION` events
    (`src/events/possession.py::detect_possession`). The SCORER is then the identity with the
    LAST such possession run in that window, gated to the credited team.

    Returns a NEW `Event` (never mutates `goal_event` in place — Golden Rule 5: every attribution
    step's own evidence must stay independently auditable) with `player_track_id` set to the
    scorer's identity id when determinable; when no team-confident possession exists in the
    window, `player_track_id` stays `None` and `confidence` is penalised, never guessed.
    """
    lookback = goal_cfg["possession_lookback_s"]
    window_end = goal_event.evidence.get("t_after", goal_event.t_end)
    window_start = window_end - lookback

    in_window = [
        ev
        for ev in possession_events
        if ev.type == EventType.POSSESSION and ev.t_end >= window_start and ev.t_start <= window_end
    ]

    def _team_of(ev: Event) -> int | None:
        raw_ids = ev.evidence.get("raw_track_ids") or []
        if not raw_ids:
            return None
        tr = tracks_by_id.get(raw_ids[-1])
        return tr.team if tr is not None else None

    team_duration: dict[int, float] = {}
    for ev in in_window:
        team = _team_of(ev)
        if team is None:
            continue
        team_duration[team] = team_duration.get(team, 0.0) + (ev.t_end - ev.t_start)

    def _clamp(value: float) -> float:
        return min(goal_cfg["max_confidence"], max(goal_cfg["min_confidence"], value))

    if not team_duration:
        new_evidence = {
            **goal_event.evidence,
            "team_attribution": (
                f"undetermined -- no team-confident POSSESSION event found in the {lookback}s "
                "lookback window before the goal"
            ),
            "possession_lookback_s": lookback,
        }
        return goal_event.model_copy(
            update={
                "evidence": new_evidence,
                "confidence": _clamp(
                    goal_event.confidence * goal_cfg["team_unknown_confidence_penalty"]
                ),
            }
        )

    credited_team = max(team_duration, key=lambda k: team_duration[k])
    team_events = [ev for ev in in_window if _team_of(ev) == credited_team]
    scorer_event = max(team_events, key=lambda ev: ev.t_end)
    scorer_identity = scorer_event.player_track_id

    new_evidence = {
        **goal_event.evidence,
        "credited_team": credited_team,
        "team_duration_seconds": team_duration,
        "scorer_identity": scorer_identity,
        "supporting_possession_event_id": scorer_event.id,
        "possession_lookback_s": lookback,
    }
    return goal_event.model_copy(
        update={
            "player_track_id": scorer_identity,
            "evidence": new_evidence,
            "confidence": _clamp(goal_event.confidence * goal_cfg["scorer_confidence_multiplier"]),
        }
    )


# ---------------------------------------------------------------------------
# step 4: assist
# ---------------------------------------------------------------------------


def find_assist_event(
    goal_event: Event, pass_events: list[Event], assist_cfg: dict
) -> Event | None:
    """ADR-17 step 4 — the owner's own rule, verbatim: "if the target pass the ball and the other
    score goal should be assist". The nearest PRECEDING `PASS` event
    (`src/events/possession.py::detect_passes`) whose `evidence['receiver_identity']` equals
    `goal_event`'s own credited scorer, ending within `assist_cfg['assist_window_seconds']` of the
    goal, becomes an `ASSIST` `Event` credited to the PASSER (`player_track_id` = the pass event's
    own `player_track_id`). Returns `None` — never fabricated — when the scorer itself is unknown
    (`goal_event.player_track_id is None`) or no qualifying pass exists.
    """
    scorer = goal_event.player_track_id
    if scorer is None:
        return None

    window_start = goal_event.t_end - assist_cfg["assist_window_seconds"]
    candidates = [
        ev
        for ev in pass_events
        if ev.type == EventType.PASS
        and ev.evidence.get("receiver_identity") == scorer
        and window_start <= ev.t_end <= goal_event.t_end
    ]
    if not candidates:
        return None
    supporting_pass = max(candidates, key=lambda ev: ev.t_end)

    confidence = min(
        assist_cfg["max_confidence"],
        max(
            assist_cfg["min_confidence"],
            goal_event.confidence * assist_cfg["assist_confidence_multiplier"],
        ),
    )
    return Event(
        id=str(uuid.uuid4()),
        type=EventType.ASSIST,
        t_start=supporting_pass.t_start,
        t_end=supporting_pass.t_end,
        player_track_id=supporting_pass.player_track_id,
        take_id=goal_event.take_id,
        confidence=confidence,
        source="goal_preceding_pass_heuristic",
        evidence={
            "goal_event_id": goal_event.id,
            "supporting_pass_event_id": supporting_pass.id,
            "passer_identity": supporting_pass.player_track_id,
            "receiver_identity": scorer,
            "assist_window_seconds": assist_cfg["assist_window_seconds"],
        },
    )


# ---------------------------------------------------------------------------
# ADR-20 -- human-marked goal-region detector (a second, independent occurrence source)
# ---------------------------------------------------------------------------


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test, pure/dependency-free (no need for a cv2/shapely
    import for this -- a handful of point tests per take is cheap either way). `polygon` is a list
    of `(x, y)` vertices in the SAME normalised-fraction coordinate space the ball centroid is
    converted into before calling this (see `detect_goals_goal_region`)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def detect_goals_goal_region(
    balls: list[BallDetection],
    shots: list[Event],
    slug: str,
    region_cfg: dict,
    take_id: int | None,
) -> list[Event]:
    """ADR-20 -- the human-marked goal-mouth-region occurrence detector: a SECOND, independent
    "did a goal occur" source alongside `detect_goals_scoreboard_delta`, for the static-camera
    footage that has no scoreboard at all (the owner's own `configs/goal_region.yaml`, one or more
    hand-drawn polygons per video slug -- CLAUDE.md ADR-3's own human-in-the-loop, no-training
    pattern, never an auto-detected goal line/net).

    `region_cfg['regions']` is `{slug: [[(x, y), ...], ...]}`, `(x, y)` as FRACTIONS of the
    detect-stage frame (same coordinate space as `configs/events.yaml: goal.candidate_regions`) --
    a `slug` absent from that mapping means no polygon was ever drawn for this video, so this
    returns `[]` immediately (Golden Rule 5: no polygon configured is a genuine "not available",
    never a guessed default region).

    A ball sample counts as a goal candidate only when its centroid falls inside ANY of the
    slug's own polygons AND a `SHOT` event (`shots`, this take's own `detect_shots` output) ended
    within `region_cfg['shot_window_s']` seconds before it -- the shot-window gate is what
    distinguishes an actual strike finding the net from the ball merely sitting/rolling through
    the marked area (a goal kick, a keeper's distribution, a corner taken from behind the line).
    Consecutive qualifying samples from the SAME continuous crossing are debounced into one event
    (`shot_window_s` is reused as the debounce gap too -- a real goal is one instant, not one event
    per ~0.03s ball sample crossing the line).
    """
    polygons = (region_cfg.get("regions") or {}).get(slug)
    if not polygons:
        return []

    window = region_cfg["shot_window_s"]
    shot_ends = sorted(s.t_end for s in shots)

    events: list[Event] = []
    debounce_until = float("-inf")
    for ball in sorted(balls, key=lambda b: b.t):
        if ball.t <= debounce_until:
            continue
        inside = any(_point_in_polygon(ball.bbox.cx, ball.bbox.cy, poly) for poly in polygons)
        if not inside:
            continue
        preceding_shot_ends = [t for t in shot_ends if t <= ball.t and (ball.t - t) <= window]
        if not preceding_shot_ends:
            continue

        shot_t_end = max(preceding_shot_ends)
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.GOAL,
                t_start=shot_t_end,
                t_end=ball.t,
                player_track_id=None,
                take_id=take_id,
                confidence=region_cfg["confidence"],
                source="goal_region",
                evidence={
                    "ball_t": ball.t,
                    "ball_cx": ball.bbox.cx,
                    "ball_cy": ball.bbox.cy,
                    "shot_t_end": shot_t_end,
                    "shot_window_s": window,
                    "goal_region_source": "manual",
                    "slug": slug,
                },
            )
        )
        debounce_until = ball.t + window
    return events


# ---------------------------------------------------------------------------
# per-take orchestration (does its own decode; lazily computes possession/pass only if needed)
# ---------------------------------------------------------------------------


def detect_goals_for_take(
    reader: Any,
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    take_balls: list[BallDetection],
    events_cfg: dict,
    identity_of: dict[int, int] | None,
    use_nvdec: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
    goal_region_cfg: dict | None = None,
    slug: str | None = None,
) -> tuple[list[Event], dict]:
    """Full ADR-17 pipeline for ONE take: decode a low-fps frame series spanning it, run the
    scoreboard occurrence scan, and — ONLY when at least one occurrence event is actually found —
    lazily compute this take's own possession runs + passes (`src/events/possession.py`) purely to
    attribute team/scorer/assist. A take with no scoreboard activation or no increment never pays
    that extra cost, which is what keeps this cheap on scoreboard-less footage (measured true for
    every take of all 5 original clips + `jordan_thomas_highlight_video`, this task's own
    verification run).

    ADR-20 adds a SECOND, independent occurrence source alongside the scoreboard scan: when
    `goal_region_cfg`/`slug`/`frame_width` are given and a polygon exists for `slug`
    (`detect_goals_goal_region`), this take's own `SHOT` events are computed (lazily, only then --
    `frame_width` being `None` skips this path entirely, same "don't pay for what isn't
    configured" discipline as the scoreboard branch above) and checked for a ball-crossing goal.
    Region-sourced `GOAL`s never go through team/scorer/assist attribution (they carry no
    possession-run context of their own to attribute from) -- they are appended to the result
    as-is, occurrence-only, exactly per ADR-20's own scope.

    `identity_of` is the SAME `build_take_identities` partition the caller's own event pipeline
    uses for this take (`None` for the original Phase-1 flow, which has no identity-stitching
    step at all and uses raw `Track.id`s directly, exactly like `detect_sprints` already does).
    """
    goal_cfg = events_cfg["goal"]
    assist_cfg = events_cfg["assist"]

    times: list[float] = []
    frames: list[np.ndarray] = []
    for _idx, t, frame in decode_frames(
        video_path,
        fps=goal_cfg["scoreboard_ocr_fps_sample"],
        start=take.t_start,
        end=take.t_end,
        scale_width=None,
        use_nvdec=use_nvdec,
    ):
        times.append(t)
        frames.append(frame.copy())

    occurrence_events, debug = detect_goals_scoreboard_delta(
        reader, frames, times, take.id, goal_cfg
    )

    region_events: list[Event] = []
    region_attempted = False
    # ADR-20's own config split (CLAUDE.md §5.1): `configs/goal_region.yaml` (`goal_region_cfg`
    # here) carries ONLY human-marked polygons (`{regions: {slug: [...]}}`), never thresholds --
    # `enabled`/`shot_window_s`/`confidence` live in `configs/events.yaml`'s own `goal_region:`
    # block (`events_cfg["goal_region"]`). The two must be merged before calling
    # `detect_goals_goal_region`, which expects all of them on one dict.
    region_behavior_cfg = events_cfg.get("goal_region", {})
    raw_polygons = (goal_region_cfg or {}).get("regions", {}).get(slug) if slug else None
    if (
        goal_region_cfg is not None
        and slug is not None
        and frame_width is not None
        and frame_height is not None
        and region_behavior_cfg.get("enabled", True)
        and raw_polygons
    ):
        region_attempted = True
        # `configs/goal_region.yaml` polygons are authored as [0,1] FRACTIONS of the frame (so one
        # polygon travels correctly across a re-run at a different decode scale_width) --
        # `detect_goals_goal_region` itself is unit-agnostic (it just compares ball coordinates
        # against polygon coordinates directly), so the fraction -> pixel scaling happens exactly
        # once, here, into a per-call copy of the config rather than mutating the shared one.
        scaled_polygons = [
            [(x * frame_width, y * frame_height) for x, y in poly] for poly in raw_polygons
        ]
        scaled_region_cfg = {**region_behavior_cfg, "regions": {slug: scaled_polygons}}
        take_shots = detect_shots(take_balls, take.id, frame_width, events_cfg)
        region_events = detect_goals_goal_region(
            take_balls, take_shots, slug, scaled_region_cfg, take.id
        )
    debug["goal_region_attempted"] = region_attempted
    debug["goal_region_events_found"] = len(region_events)

    if not occurrence_events:
        return region_events, debug

    tracks_by_id = {tr.id: tr for tr in take_tracks}
    runs = possession_runs_for_take(take_balls, take_tracks, events_cfg, identity_of)
    possession_events = detect_possession(runs, take.id, events_cfg)
    pass_events = detect_passes(runs, take_tracks, take.id, events_cfg)
    debug["n_possession_runs"] = len(runs)

    events: list[Event] = []
    for occ in occurrence_events:
        attributed = attribute_goal_team_and_scorer(occ, possession_events, tracks_by_id, goal_cfg)
        events.append(attributed)
        assist_event = find_assist_event(attributed, pass_events, assist_cfg)
        if assist_event is not None:
            events.append(assist_event)
    return events + region_events, debug


# ---------------------------------------------------------------------------
# final reporting step (pure decision table)
# ---------------------------------------------------------------------------


def check_goal_availability(
    scan_attempted: bool,
    activated_any_region: bool,
    events: list[Event],
    goal_cfg: dict,
    goal_region_configured: bool = False,
) -> GoalDetectionResult:
    """The final Golden-Rule-5 reporting step (ADR-17, extended by ADR-20): turns a scan's own
    OUTCOME into an honest `GoalDetectionResult`. Never does I/O itself — `detect_goals_for_video`
    is responsible for actually running the scan(s); this stays a pure, trivially-unit-testable
    decision table, so its result is provably driven by real input rather than a hardcoded "not
    available" regardless of what was actually found (the exact defect ADR-17 fixes).

    ADR-20: the availability decision is now made off `events` (whichever `GOAL`s survived from
    EITHER the scoreboard-OCR-delta source or the human-marked goal-region source) FIRST, not off
    `activated_any_region` (the scoreboard-only activation flag) alone -- a video with a configured
    goal region but no scoreboard anywhere must still correctly report `available=True` off a real
    region-sourced goal, never masked by "no scoreboard activated". `goal_region_configured` names,
    in the "not available" reason, whether a human-marked polygon was even in play for this video
    (`configs/goal_region.yaml`) so the reason always says which of the two sources this module
    itself attempted (a third source, the ADR-19 manual-annotation sidecar, is a separate pipeline
    entirely and is not this function's concern).
    """
    if not scan_attempted:
        return GoalDetectionResult(
            available=False,
            reason="not available (goal detection was not attempted -- this video has no takes)",
            events=[],
        )

    goal_events = [e for e in events if e.type == EventType.GOAL]
    region_note = (
        "a human-marked goal region IS configured for this video (configs/goal_region.yaml, "
        "ADR-20) and was also checked"
        if goal_region_configured
        else "no human-marked goal region is configured for this video either "
        "(configs/goal_region.yaml, ADR-20)"
    )

    if not goal_events:
        if not activated_any_region:
            scoreboard_note = (
                "no legible scoreboard found by the region-activation scan -- CLAUDE.md ADR-17: "
                "EasyOCR was run over each candidate corner region across multiple sampled frames "
                "per take and none yielded a stable digit-separator-digit reading; this is the "
                "measured, per-video result, never an assumption from the source profile label"
            )
        else:
            scoreboard_note = (
                "a scoreboard region activated on this footage, but no debounced score increment "
                "was ever observed across the video"
            )
        return GoalDetectionResult(
            available=False,
            reason=(
                f"not available ({scoreboard_note}; {region_note}, and no ball-crossing-a-marked-"
                "region goal was found)"
            ),
            events=events,
        )

    assist_events = [e for e in events if e.type == EventType.ASSIST]
    scoreboard_n = sum(1 for e in goal_events if e.source == "scoreboard_ocr_delta")
    region_n = sum(1 for e in goal_events if e.source == "goal_region")
    source_bits = []
    if scoreboard_n:
        source_bits.append(f"{scoreboard_n} via scoreboard-OCR delta")
    if region_n:
        source_bits.append(f"{region_n} via a human-marked goal region (ADR-20)")
    source_desc = "; ".join(source_bits) if source_bits else "via an unspecified source"
    return GoalDetectionResult(
        available=True,
        reason=(
            f"{len(goal_events)} goal(s) detected ({source_desc}) "
            f"({len(assist_events)} with an assist credited) -- see each Event.evidence for the "
            "full audit trail (activated region / marked polygon, before/after reading or "
            "crossing point, possession-window scores, supporting pass id), Golden Rule 5"
        ),
        events=events,
    )


# ---------------------------------------------------------------------------
# whole-video orchestration (shared by src.pipeline.run and src.pipeline.extended_output)
# ---------------------------------------------------------------------------


def detect_goals_for_video(
    video_path: str | Path,
    takes: list[Take],
    tracks_by_take: dict[int, list[Track]],
    balls_by_take: dict[int, list[BallDetection]],
    events_cfg: dict,
    identity_of_by_take: dict[int, dict[int, int]] | None = None,
    ocr_cfg: dict | None = None,
    use_nvdec: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
    goal_region_cfg: dict | None = None,
    slug: str | None = None,
) -> GoalDetectionResult:
    """ADR-17's real, per-video goal+assist detector (ADR-20 adds the human-marked goal-region
    source alongside it) — the single entrypoint both `src.pipeline.run` (the 5 original
    filename-based clips) and `src.pipeline.extended_output` (ADR-15's filename-less flow) call
    identically. Loads ONE EasyOCR reader for the whole video (CLAUDE.md §11: never reload
    per-take), matching `src/identity/verify.py`'s own load-once/free-in-`finally` convention.

    Never trusts `RunProfile.profile` (ADR-11's own lesson): every take gets the SAME measured
    region-activation scan regardless of what the source profiler labeled the video.

    `frame_width`/`frame_height`/`goal_region_cfg`/`slug` (ADR-20) are all optional and default to
    `None` -- omitting any of them simply skips the goal-region source entirely (never a crash,
    never a fabricated region), which is exactly what every caller that doesn't yet pass them gets
    (safe, behaviour-preserving default for any pre-ADR-20 call site).
    """
    goal_cfg = events_cfg["goal"]
    goal_region_configured = bool(slug and (goal_region_cfg or {}).get("regions", {}).get(slug))
    if not takes:
        return check_goal_availability(
            scan_attempted=False,
            activated_any_region=False,
            events=[],
            goal_cfg=goal_cfg,
            goal_region_configured=goal_region_configured,
        )

    reader = load_easyocr_reader(ocr_cfg or {"gpu": True})
    all_events: list[Event] = []
    activated_any = False
    try:
        for take in takes:
            take_tracks = tracks_by_take.get(take.id, [])
            take_balls = balls_by_take.get(take.id, [])
            identity_of = (identity_of_by_take or {}).get(take.id)
            events, debug = detect_goals_for_take(
                reader,
                video_path,
                take,
                take_tracks,
                take_balls,
                events_cfg,
                identity_of,
                use_nvdec=use_nvdec,
                frame_width=frame_width,
                frame_height=frame_height,
                goal_region_cfg=goal_region_cfg,
                slug=slug,
            )
            if debug.get("activated_region") is not None:
                activated_any = True
            all_events.extend(events)
    finally:
        free_easyocr_reader(reader)

    any_goal = any(e.type == EventType.GOAL for e in all_events)
    logger.info(
        "goal detection: %s (%d take(s) scanned, scoreboard region activated=%s, goal region "
        "configured=%s)",
        "available" if any_goal else "not available",
        len(takes),
        activated_any,
        goal_region_configured,
    )
    return check_goal_availability(
        scan_attempted=True,
        activated_any_region=activated_any,
        events=all_events,
        goal_cfg=goal_cfg,
        goal_region_configured=goal_region_configured,
    )
