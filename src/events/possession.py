"""Best-effort possession/pass/dribble/distance heuristics (ADR-13/14; CLAUDE.md §13.3).

⚠️ Owner-authorized exception to Golden Rule 7 (CLAUDE.md §12/ADR-13): everything below is
ARITHMETIC on cached `Detection`/`Track`/`BallDetection` artifacts, never a new trained model
(Golden Rule 2). Builds on `src/events/touches.py`'s "who has the ball" geometry
(`nearest_player_edge_distance`, `POSSESSOR_CLASSES`) — reused, not reimplemented, per the task
spec's own framing ("touches and possession share the same core logic").

Pipeline, in order:

1. `possession_runs_for_take` — for every eligible ball sample, find the nearest possessor (via
   `touches.nearest_player_edge_distance`) within `possession_max_dist`; merge consecutive
   same-identity samples into `PossessionRun`s (bridging brief "no-one close enough" gaps under
   `possession_merge_gap_s`, but NEVER bridging across a different identity's own intervening
   spell — see `_merge_possession_runs`'s docstring for the exact semantics this guarantees).
2. `detect_possession` — the subset of those runs lasting >= `possession_min_duration_s` becomes a
   `POSSESSION` `Event` per run.
3. `detect_passes` — walks the FULL (unfiltered) run list; two adjacent runs by DIFFERENT
   identities, close enough in time/space and confidently on the SAME team (ADR-12-gated), become
   a `PASS` `Event`. Using the unfiltered list means even a real-but-brief opposing touch in
   between correctly blocks a pass being read straight through it.
4. `detect_dribbles` — a single (min-duration-filtered) run, by one identity, that is CONTESTED by
   a confidently-opposing player for at least part of it AND covers real ground, becomes a
   `DRIBBLE` `Event`.
5. `compute_distance_covered` — NOT an `Event` (there is no natural single instant/interval for a
   running total): a plain traceable dict, integrating `src/events/sprints.py`'s own per-sample
   speed (imported, not reimplemented) over a set of track fragments, in the SAME bbox-heights
   unit, always `calibrated: False` (ADR-6). This is the natural home per this task's own file
   ownership list ("possession.py ... dribbles + possession + distance live here") — `sprints.py`
   itself is left untouched.

`src/events/tackles.py` also consumes `possession_runs_for_take`'s output directly (a tackle is
itself a possession change, just an OPPOSING-team one with a closing-speed precondition) — see
`teammates_gate`/`possession_quality` below, both public for exactly that reuse.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from src.common.types import BallDetection, DetectionClass, Event, EventType, Track, TrackBox
from src.events.sprints import track_speed_series
from src.events.touches import POSSESSOR_CLASSES, nearest_box_in_time, nearest_player_edge_distance

# `configs/hardware.yaml: stages.ball.fps_sample` — the ball's own sampling rate, used as the
# default "how many samples SHOULD exist over this duration" denominator for continuity scoring.
# Every caller in this project's actual pipeline already samples the ball at this rate; a caller
# feeding a different rate should simply pass its own `ball_fps`, not edit this default.
DEFAULT_BALL_FPS = 30.0


@dataclass
class PossessionRun:
    """One continuous stretch of the ball being nearest to a single (possibly continuity-stitched)
    identity. `samples` is `(t, raw_track_id, distance_px, ball_cx, ball_cy)` for every ball sample
    that contributed to this run, time-ordered — kept (not just start/end) so confidence/travel/
    contest checks can inspect the whole interval, not just its endpoints."""

    identity: int
    t_start: float
    t_end: float
    samples: list[tuple[float, int, float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. building possession runs
# ---------------------------------------------------------------------------


def _raw_possession_series(
    balls: list[BallDetection], candidates: list[Track], possession_cfg: dict
) -> list[tuple[float, int | None, float | None, float, float]]:
    """`(t, raw_track_id_or_None, distance_px_or_None, ball_cx, ball_cy)` per eligible ball
    sample, time-sorted. `raw_track_id` is `None` when no candidate track is close enough (within
    `possession_max_dist`) or time-aligned (within `match_tolerance_s`) at all -- a "loose ball"
    instant, not an error.
    """
    tol = possession_cfg["match_tolerance_s"]
    max_dist = possession_cfg["possession_max_dist"]
    min_ball_conf = possession_cfg["min_ball_conf"]

    series: list[tuple[float, int | None, float | None, float, float]] = []
    for ball in sorted(balls, key=lambda b: b.t):
        if ball.conf < min_ball_conf:
            continue
        nearest = nearest_player_edge_distance(ball, candidates, tol)
        if nearest is None or nearest[1] > max_dist:
            series.append((ball.t, None, None, ball.bbox.cx, ball.bbox.cy))
            continue
        raw_id, dist = nearest
        series.append((ball.t, raw_id, dist, ball.bbox.cx, ball.bbox.cy))
    return series


def _merge_possession_runs(
    series: list[tuple[float, int | None, float | None, float, float]],
    identity_of: dict[int, int] | None,
    merge_gap_s: float,
) -> list[PossessionRun]:
    """Merge consecutive same-identity samples into `PossessionRun`s.

    A `None` (loose-ball) sample never itself starts/extends a run, but does NOT reset `current`
    either -- so if the SAME identity's samples resume within `merge_gap_s` of the run's last
    sample, the loose-ball gap between them is bridged into one run (a brief ball-tracking miss
    mid-possession, ADR-6-style). Critically, if a DIFFERENT identity's sample appears in that gap
    instead, it starts its OWN new run immediately (the `current["identity"] == identity` check
    fails), so a real intervening possessor is NEVER silently bridged over -- only genuine "no one
    was close enough" stretches are.
    """
    runs: list[PossessionRun] = []
    current: PossessionRun | None = None
    for t, raw_id, dist, bx, by in series:
        if raw_id is None or dist is None:
            continue
        identity = identity_of.get(raw_id, raw_id) if identity_of else raw_id
        extends_current = current is not None and current.identity == identity
        if current is not None and extends_current and (t - current.t_end) < merge_gap_s:
            current.t_end = t
            current.samples.append((t, raw_id, dist, bx, by))
        else:
            current = PossessionRun(
                identity=identity, t_start=t, t_end=t, samples=[(t, raw_id, dist, bx, by)]
            )
            runs.append(current)
    return runs


def possession_runs_for_take(
    balls: list[BallDetection],
    tracks: list[Track],
    events_cfg: dict,
    identity_of: dict[int, int] | None = None,
) -> list[PossessionRun]:
    """Build the full (unfiltered by `possession_min_duration_s`) run list for one take's own
    `balls`/`tracks` (caller-bucketed, Golden Rule 3 — never crosses a take). Excludes referees
    from possession candidacy (`POSSESSOR_CLASSES`, same convention as `touches.detect_touches`).

    Asserted defensively, same reasoning as `touches.detect_touches`: raw `Track.id`s reset per
    take, so a `tracks` list spanning more than one `take_id` risks two takes' fragments colliding
    on the same integer id and being merged into one run as if they were the same person.
    """
    take_ids = {tr.take_id for tr in tracks}
    assert len(take_ids) <= 1, "possession_runs_for_take must only ever see one take's tracks"

    possession_cfg = events_cfg["possession"]
    candidates = [tr for tr in tracks if tr.dominant_class in POSSESSOR_CLASSES]
    series = _raw_possession_series(balls, candidates, possession_cfg)
    return _merge_possession_runs(series, identity_of, possession_cfg["possession_merge_gap_s"])


# ---------------------------------------------------------------------------
# confidence (shared building block: an UNCLAMPED [0,1] "quality" score)
# ---------------------------------------------------------------------------


def possession_quality(run: PossessionRun, possession_cfg: dict, ball_fps: float) -> float:
    """`continuity_weight * continuity + (1 - continuity_weight) * closeness`, both terms in
    `[0, 1]`, UNCLAMPED by any category's own confidence ceiling -- the same blend style as
    `src/events/sprints.py::segment_confidence`, but returned raw so `detect_passes`/
    `detect_tackles` (in `src/events/tackles.py`) can combine two runs' qualities and apply their
    OWN category ceiling afterwards, instead of compounding two already-capped-low numbers
    together (which would make a real pass/tackle almost never clear `min_emit_confidence`).

    `continuity` = (# ball samples actually in this run) / (# EXPECTED at `ball_fps` over its
    duration), capped at 1.0. `closeness` = how far the run's OWN mean ball-to-possessor distance
    sits below `possession_max_dist`.
    """
    duration = max(run.t_end - run.t_start, 1e-6)
    expected = max(1, round(duration * ball_fps) + 1)
    continuity = min(1.0, len(run.samples) / expected)

    mean_dist = sum(d for _, _, d, _, _ in run.samples) / len(run.samples)
    max_d = possession_cfg["possession_max_dist"]
    closeness = 1.0 - min(1.0, mean_dist / max_d) if max_d > 0 else 1.0

    weight = possession_cfg["continuity_weight"]
    return weight * continuity + (1.0 - weight) * closeness


def possession_confidence(
    run: PossessionRun,
    possession_cfg: dict,
    ball_fps: float = DEFAULT_BALL_FPS,
    identity_confidence: dict[int, float] | None = None,
) -> float:
    """`possession_quality(...) * identity_confidence[run.identity]`, hard-clamped to this
    category's own `[min_confidence, max_confidence]` (Golden Rule 5). A run built from a heavily
    continuity-stitched identity (many joins, `src/track/continuity.py`) reads as less certain
    than one from a single unbroken track fragment, same spirit as `stitch_timeline`'s own
    per-join confidence penalty.
    """
    quality = possession_quality(run, possession_cfg, ball_fps)
    id_conf = identity_confidence.get(run.identity, 1.0) if identity_confidence else 1.0
    raw = quality * id_conf
    return min(possession_cfg["max_confidence"], max(possession_cfg["min_confidence"], raw))


# ---------------------------------------------------------------------------
# team lookups (shared with src/events/tackles.py)
# ---------------------------------------------------------------------------


def track_team(raw_track_id: int, tracks_by_id: dict[int, Track]) -> tuple[int | None, float]:
    """`(team, team_confidence)` for a RAW track id, `(None, 0.0)` when the id is unknown."""
    tr = tracks_by_id.get(raw_track_id)
    if tr is None:
        return None, 0.0
    return tr.team, tr.team_confidence


def teammates_gate(
    raw_a: int, raw_b: int, tracks_by_id: dict[int, Track], threshold: float
) -> bool | None:
    """`True`/`False` when BOTH raw tracks' `team_confidence` clears `threshold` and both have a
    known `team`; `None` when the team signal isn't trustworthy enough to judge at all (ADR-12:
    team is a SOFT signal — ``src/events/tackles.py`` treats `None` the same way `pass`'s own
    caller does, as "can't tell", never as a silent `team=0` default).
    """
    team_a, conf_a = track_team(raw_a, tracks_by_id)
    team_b, conf_b = track_team(raw_b, tracks_by_id)
    if conf_a < threshold or conf_b < threshold:
        return None
    if team_a is None or team_b is None:
        return None
    return team_a == team_b


def teammates_test(
    raw_a: int,
    raw_b: int,
    tracks_by_id: dict[int, Track],
    team_threshold: float,
    kit_colour_cfg: dict,
    kit_lab_by_track_id: dict[int, np.ndarray] | None = None,
) -> bool | None:
    """Cascading "are these two players teammates" test (Stage 5, owner request 2026-09-02: kit
    colour as the PRIMARY signal for pass/assist attribution).

    Order: (1) kit colour, DIRECTLY measured and compared (`src.events.kit_colour.
    kit_colour_same_team` -- see that function's own docstring for why it is a two-sided
    real-data-measured gate, not a single threshold, and why it is never compared against a fixed
    reference palette); (2) if colour is undecided (or `kit_lab_by_track_id` wasn't supplied at
    all -- e.g. a caller that hasn't run the per-take colour-sampling pass yet), fall back to the
    pre-existing ADR-12 team-CLUSTER gate (`teammates_gate`, unchanged). Backward compatible: a
    caller passing `kit_lab_by_track_id=None` (the default) gets EXACTLY the old cluster-only
    behaviour.
    """
    from src.events.kit_colour import kit_colour_same_team

    if kit_lab_by_track_id is not None:
        lab_a = kit_lab_by_track_id.get(raw_a)
        lab_b = kit_lab_by_track_id.get(raw_b)
        colour_verdict, _dist = kit_colour_same_team(lab_a, lab_b, kit_colour_cfg)
        if colour_verdict is not None:
            return colour_verdict
    return teammates_gate(raw_a, raw_b, tracks_by_id, team_threshold)


# ---------------------------------------------------------------------------
# 2. possession events
# ---------------------------------------------------------------------------


def detect_possession(
    runs: list[PossessionRun],
    take_id: int | None,
    events_cfg: dict,
    ball_fps: float = DEFAULT_BALL_FPS,
    identity_confidence: dict[int, float] | None = None,
    drops=None,
) -> list[Event]:
    """`POSSESSION` events for every run in `runs` (from `possession_runs_for_take`) that clears
    `possession_min_duration_s`. `take_id` is stamped directly (Golden Rule 3: `runs` must already
    be one take's own, same convention as every other `detect_*` in this project)."""
    possession_cfg = events_cfg["possession"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]

    kept = [r for r in runs if (r.t_end - r.t_start) >= possession_cfg["possession_min_duration_s"]]
    if drops is not None and len(runs) > len(kept):
        drops.drop("possession_run_below_min_duration", len(runs) - len(kept))

    events: list[Event] = []
    for run in kept:
        confidence = possession_confidence(run, possession_cfg, ball_fps, identity_confidence)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("possession_below_min_emit_confidence")
            continue
        mean_dist = sum(d for _, _, d, _, _ in run.samples) / len(run.samples)
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.POSSESSION,
                t_start=run.t_start,
                t_end=run.t_end,
                player_track_id=run.identity,
                take_id=take_id,
                confidence=confidence,
                source="ball_proximity_possession_heuristic",
                evidence={
                    "raw_track_ids": sorted({rid for _, rid, _, _, _ in run.samples}),
                    "n_ball_samples": len(run.samples),
                    "mean_distance_px": mean_dist,
                    "possession_max_dist": possession_cfg["possession_max_dist"],
                    "calibrated": False,
                    "unit": "pixels_at_detect_stage_resolution",
                },
            )
        )
    return events


# ---------------------------------------------------------------------------
# 3. passes
# ---------------------------------------------------------------------------


def _ball_travel(run_a: PossessionRun, run_b: PossessionRun) -> float:
    """Pixel distance the ball itself moved between run A's LAST sample and run B's FIRST sample
    (not the players' own positions) — the most direct read of "did the ball actually travel a
    plausible pass distance", per `configs/events.yaml: pass.pass_max_dist`'s own measurement."""
    _, _, _, bx_a, by_a = run_a.samples[-1]
    _, _, _, bx_b, by_b = run_b.samples[0]
    return ((bx_b - bx_a) ** 2 + (by_b - by_a) ** 2) ** 0.5


def pass_confidence(quality_a: float, quality_b: float, travel: float, pass_cfg: dict) -> float:
    """`closeness(travel) * mean(quality_a, quality_b)`, clamped to `pass`'s own ceiling. Combines
    two UNCLAMPED possession qualities (see `possession_quality`) rather than two already-capped
    confidences, so a real pass doesn't get squared-away below `min_emit_confidence` just because
    each half of it was itself a low-ceiling possession event."""
    max_d = pass_cfg["pass_max_dist"]
    closeness = 1.0 - min(1.0, travel / max_d) if max_d > 0 else 1.0
    raw = closeness * ((quality_a + quality_b) / 2.0)
    return min(pass_cfg["max_confidence"], max(pass_cfg["min_confidence"], raw))


def turnover_confidence(
    quality_a: float, quality_b: float, travel: float, pass_cfg: dict, turnover_cfg: dict
) -> float:
    """ADR-20: the SAME `closeness(travel) * mean(quality_a, quality_b)` shape as `pass_confidence`
    (reuses `pass_cfg['pass_max_dist']` for the closeness term -- a turnover is measured against
    the identical "is this ball travel plausible for one continuous possession change" scale a
    pass is, it just landed on the WRONG colour), clamped to `turnover`'s own
    `configs/events.yaml` ceiling instead of `pass`'s."""
    max_d = pass_cfg["pass_max_dist"]
    closeness = 1.0 - min(1.0, travel / max_d) if max_d > 0 else 1.0
    raw = closeness * ((quality_a + quality_b) / 2.0)
    return min(turnover_cfg["max_confidence"], max(turnover_cfg["min_confidence"], raw))


def detect_passes(
    runs: list[PossessionRun],
    tracks: list[Track],
    take_id: int | None,
    events_cfg: dict,
    ball_fps: float = DEFAULT_BALL_FPS,
    identity_confidence: dict[int, float] | None = None,
    drops=None,
    kit_lab_by_track_id: dict[int, np.ndarray] | None = None,
) -> list[Event]:
    """`PASS` events between adjacent runs (in `runs`' own time order — i.e. no other identity's
    run sits between them, see `_merge_possession_runs`) by DIFFERENT, confidently-SAME-team
    identities, close enough in time (`pass_max_gap_s`) and ball-travel (`pass_max_dist`).

    `player_track_id` is the PASSER (run A's identity) — CLAUDE.md §13.2 lists "Passes" as a
    per-player stat, and the conventional football-stats reading credits the player who PLAYED the
    ball, not the one who received it; `Event.evidence` carries both identities regardless.

    ADR-20: the `teammates_gate` tri-state now drives THREE outcomes, not two. `True` (same ADR-12
    colour cluster) → `PASS`, unchanged. `False` (confidently OPPOSING cluster) → a NEW
    `EventType.TURNOVER` — a real, traceable possession loss, logged with its own evidence rather
    than silently dropped (Golden Rule 5: hiding a lost ball would be a dishonest filter, and the
    owner's own framing is explicit that a turnover is "not a completed pass", not "not an event
    at all"). `None` (team signal not trustworthy enough to judge either way) → unchanged: dropped,
    never guessed as either outcome.

    Asserted defensively, same reasoning as `possession_runs_for_take`: `tracks` must be one
    take's own (raw ids reset per take, so a mixed-take list risks a wrong team lookup silently
    overwriting another take's same-numbered fragment in `tracks_by_id`).
    """
    take_ids = {tr.take_id for tr in tracks}
    assert len(take_ids) <= 1, "detect_passes must only ever see one take's tracks"

    pass_cfg = events_cfg["pass"]
    possession_cfg = events_cfg["possession"]
    kit_colour_cfg = events_cfg["kit_colour"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    tracks_by_id = {tr.id: tr for tr in tracks}
    # Owner spec ("Implement event cooldown/state logic... FPS-aware") + measured real bug: on
    # clip1_43, adjacent possession-run pairs fired 9 PASS events in an 11s clip, 0.1-0.4s apart --
    # runs fragment easily (possession.possession_min_duration_s=0.3s), and nothing previously
    # stopped the SAME passer identity being credited with a second "pass" a fraction of a second
    # after the first. Compared in real SECONDS against run timestamps, so this is FPS-independent
    # by construction, identical reasoning to touch.touch_min_gap_s.
    last_pass_t: dict[int, float] = {}

    events: list[Event] = []
    for run_a, run_b in zip(runs, runs[1:], strict=False):
        if run_a.identity == run_b.identity:
            continue  # same person regaining the ball -- not a pass (could be a dribble instead)

        gap = run_b.t_start - run_a.t_end
        if gap < 0 or gap > pass_cfg["pass_max_gap_s"]:
            if drops is not None:
                drops.drop("pass_gap_too_large")
            continue

        last_t = last_pass_t.get(run_a.identity)
        if last_t is not None and (run_a.t_end - last_t) < pass_cfg["pass_min_gap_s"]:
            if drops is not None:
                drops.drop("pass_debounced")
            continue

        travel = _ball_travel(run_a, run_b)
        if travel > pass_cfg["pass_max_dist"]:
            if drops is not None:
                drops.drop("pass_ball_travel_too_far")
            continue

        raw_a = run_a.samples[-1][1]
        raw_b = run_b.samples[0][1]
        team_threshold = pass_cfg["team_confidence_threshold"]
        teammates = teammates_test(
            raw_a, raw_b, tracks_by_id, team_threshold, kit_colour_cfg, kit_lab_by_track_id
        )
        if teammates is None:
            if drops is not None:
                drops.drop("pass_team_confidence_too_low")
            continue

        # ADR-20: `pass.same_colour_required` documents (and can relax) the design decision that a
        # PASS strictly requires the SAME ADR-12 colour cluster. Defaulting True/absent preserves
        # the exact pre-ADR-20 gate; an explicit False treats an opposing-colour possession change
        # as a pass too (never emitting a TURNOVER at all) -- an owner-facing escape hatch for
        # footage where the colour clustering itself turns out to be unreliable, not something any
        # of this project's own clips currently need.
        if teammates is False and not pass_cfg.get("same_colour_required", True):
            teammates = True

        quality_a = possession_quality(run_a, possession_cfg, ball_fps)
        quality_b = possession_quality(run_b, possession_cfg, ball_fps)
        if identity_confidence is not None:
            quality_a *= identity_confidence.get(run_a.identity, 1.0)
            quality_b *= identity_confidence.get(run_b.identity, 1.0)

        if teammates is True:
            confidence = pass_confidence(quality_a, quality_b, travel, pass_cfg)
            if confidence < min_emit:
                if drops is not None:
                    drops.drop("pass_below_min_emit_confidence")
                continue
            last_pass_t[run_a.identity] = run_a.t_end
            events.append(
                Event(
                    id=str(uuid.uuid4()),
                    type=EventType.PASS,
                    t_start=run_a.t_end,
                    t_end=run_b.t_start,
                    player_track_id=run_a.identity,
                    take_id=take_id,
                    confidence=confidence,
                    source="possession_change_teammate_heuristic",
                    evidence={
                        "passer_identity": run_a.identity,
                        "receiver_identity": run_b.identity,
                        "passer_raw_track_id": raw_a,
                        "receiver_raw_track_id": raw_b,
                        "ball_travel_px": travel,
                        "gap_s": gap,
                        "calibrated": False,
                        "unit": "pixels_at_detect_stage_resolution",
                    },
                )
            )
            continue

        # ADR-20: `teammates is False` -- a confidently OPPOSING-colour possession change. This is
        # a TURNOVER (logged with full evidence, Golden Rule 5), never silently dropped and never
        # counted toward `Passes` (CLAUDE.md §13.2's own template keeps the two lines separate).
        # Reuses the exact same adjacency/gap/distance eligibility already checked above for a
        # pass -- the only thing that differs is which colour cluster ended up with the ball.
        turnover_cfg = events_cfg["turnover"]
        if not turnover_cfg["enabled"]:
            if drops is not None:
                drops.drop("turnover_disabled")
            continue
        confidence = turnover_confidence(quality_a, quality_b, travel, pass_cfg, turnover_cfg)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("turnover_below_min_emit_confidence")
            continue

        team_a, _conf_a = track_team(raw_a, tracks_by_id)
        team_b, _conf_b = track_team(raw_b, tracks_by_id)
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.TURNOVER,
                t_start=run_a.t_end,
                t_end=run_b.t_start,
                player_track_id=run_a.identity,
                take_id=take_id,
                confidence=confidence,
                source="possession_change_opposing_colour",
                evidence={
                    "losing_identity": run_a.identity,
                    "receiving_identity": run_b.identity,
                    "losing_raw_track_id": raw_a,
                    "receiving_raw_track_id": raw_b,
                    "losing_team": team_a,
                    "receiving_team": team_b,
                    "ball_travel_px": travel,
                    "gap_s": gap,
                    "calibrated": False,
                    "unit": "pixels_at_detect_stage_resolution",
                },
            )
        )
    return events


# ---------------------------------------------------------------------------
# 4. dribbles
# ---------------------------------------------------------------------------


def _bbox_height_normalized_distance(box_a: TrackBox, box_b: TrackBox) -> float:
    """Same bbox-height normalisation as `src/track/continuity.py::_spatial_jump`, generalised to
    two ARBITRARY boxes (not one track's own last box vs a candidate's first) — used for the
    dribble travel/contest checks, which compare two independent tracks' positions at the same
    instant, a different shape of query from stitching's own "chain continuation" check."""
    dx = box_a.bbox.cx - box_b.bbox.cx
    dy = box_a.bbox.cy - box_b.bbox.cy
    dist = (dx * dx + dy * dy) ** 0.5
    mean_h = (box_a.bbox.height + box_b.bbox.height) / 2.0
    return dist / mean_h if mean_h > 0 else float("inf")


def _is_contested(
    run: PossessionRun,
    tracks_by_id: dict[int, Track],
    all_tracks: list[Track],
    dribble_cfg: dict,
    match_tolerance_s: float,
) -> bool:
    """Whether some confidently-OPPOSING, non-referee track came within `dribble_contest_dist`
    (bbox-heights) of the possessor for at least one sample in `run`. Strides through at most ~20
    of the run's own samples (a performance detail, not a semantic threshold -- a genuine contest
    lasting even a fraction of a second shows up at any stride short of the whole run) rather than
    checking every single ~30fps ball sample against every other track.
    """
    threshold = dribble_cfg["team_confidence_threshold"]
    contest_dist = dribble_cfg["dribble_contest_dist"]
    samples = run.samples
    stride = max(1, len(samples) // 20)

    for t, raw_id, _dist, _bx, _by in samples[::stride]:
        possessor = tracks_by_id.get(raw_id)
        if possessor is None:
            continue
        p_box = nearest_box_in_time(possessor, t, match_tolerance_s)
        if p_box is None or possessor.team_confidence < threshold or possessor.team is None:
            continue
        for other in all_tracks:
            if other.id == possessor.id or other.dominant_class == DetectionClass.REFEREE:
                continue
            if other.team_confidence < threshold or other.team is None:
                continue
            if other.team == possessor.team:
                continue  # same team (or itself) -- not "opposing pressure"
            o_box = nearest_box_in_time(other, t, match_tolerance_s)
            if o_box is None:
                continue
            if _bbox_height_normalized_distance(p_box, o_box) <= contest_dist:
                return True
    return False


def detect_dribbles(
    runs: list[PossessionRun],
    tracks: list[Track],
    take_id: int | None,
    events_cfg: dict,
    ball_fps: float = DEFAULT_BALL_FPS,
    identity_confidence: dict[int, float] | None = None,
    drops=None,
) -> list[Event]:
    """`DRIBBLE` events: a single possession run lasting >= `dribble_min_duration_s`, CONTESTED by
    an opponent for some portion of it (`_is_contested`), whose possessor's own position moved at
    least `dribble_min_travel` (bbox-heights) between the run's first and last sample -- excludes
    "standing still with the ball loosely nearby" (Golden Rule 5: distinguishing signal from a
    coincidence is the whole point of this heuristic's own extra conditions).

    Asserted defensively, same reasoning as `possession_runs_for_take`/`detect_passes`: `tracks`
    must be one take's own.
    """
    take_ids = {tr.take_id for tr in tracks}
    assert len(take_ids) <= 1, "detect_dribbles must only ever see one take's tracks"

    dribble_cfg = events_cfg["dribble"]
    possession_cfg = events_cfg["possession"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    match_tolerance_s = possession_cfg["match_tolerance_s"]
    tracks_by_id = {tr.id: tr for tr in tracks}

    events: list[Event] = []
    long_runs = [r for r in runs if (r.t_end - r.t_start) >= dribble_cfg["dribble_min_duration_s"]]
    for run in long_runs:
        start_raw = run.samples[0][1]
        end_raw = run.samples[-1][1]
        start_tr = tracks_by_id.get(start_raw)
        end_tr = tracks_by_id.get(end_raw)
        if start_tr is None or end_tr is None:
            if drops is not None:
                drops.drop("dribble_missing_track_for_travel")
            continue
        start_box = nearest_box_in_time(start_tr, run.t_start, match_tolerance_s)
        end_box = nearest_box_in_time(end_tr, run.t_end, match_tolerance_s)
        if start_box is None or end_box is None:
            if drops is not None:
                drops.drop("dribble_no_box_near_interval_edge")
            continue

        travel = _bbox_height_normalized_distance(start_box, end_box)
        if travel < dribble_cfg["dribble_min_travel"]:
            if drops is not None:
                drops.drop("dribble_insufficient_travel")
            continue

        if not _is_contested(run, tracks_by_id, tracks, dribble_cfg, match_tolerance_s):
            if drops is not None:
                drops.drop("dribble_not_contested")
            continue

        quality = possession_quality(run, possession_cfg, ball_fps)
        id_conf = identity_confidence.get(run.identity, 1.0) if identity_confidence else 1.0
        raw_conf = quality * id_conf
        confidence = min(
            dribble_cfg["max_confidence"], max(dribble_cfg["min_confidence"], raw_conf)
        )
        if confidence < min_emit:
            if drops is not None:
                drops.drop("dribble_below_min_emit_confidence")
            continue

        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.DRIBBLE,
                t_start=run.t_start,
                t_end=run.t_end,
                player_track_id=run.identity,
                take_id=take_id,
                confidence=confidence,
                source="possession_contested_travel_heuristic",
                evidence={
                    "travel_bbox_heights": travel,
                    "dribble_min_travel": dribble_cfg["dribble_min_travel"],
                    "contested": True,
                    "raw_track_ids": sorted({rid for _, rid, _, _, _ in run.samples}),
                    "calibrated": False,
                    "unit": "bbox_heights",
                },
            )
        )
    return events


# ---------------------------------------------------------------------------
# 5. distance covered (a plain traceable dict, not an Event -- see module docstring)
# ---------------------------------------------------------------------------


def compute_distance_covered(
    identity_track_ids: list[int],
    tracks: list[Track],
    events_cfg: dict,
    smoothing_window: int | None = None,
) -> dict:
    """Integrate `src/events/sprints.py::track_speed_series`'s own per-sample speed (bbox-heights
    per second) over `dt` for every track fragment in `identity_track_ids`, summed into one total
    -- `speed * dt` integrated over time is a distance, in the SAME bbox-heights unit ADR-6
    already establishes for this project, never metres (`calibrated` is always `False`).

    `identity_track_ids` is typically one stitched identity's own fragment ids (e.g. a
    `SelectionResult.TakeSelection.track_ids`, or a `src/track/continuity.py::build_take_identities`
    chain) spanning one take, or several takes' worth concatenated by the caller for a whole-video
    total -- this function itself is take-agnostic, it just sums whatever fragments it's given.
    """
    sprint_cfg = events_cfg["sprint"]
    default_window = sprint_cfg["smoothing_window_frames"]
    window = smoothing_window if smoothing_window is not None else default_window
    by_id = {tr.id: tr for tr in tracks}

    total = 0.0
    n_speed_samples = 0
    for tid in identity_track_ids:
        tr = by_id.get(tid)
        if tr is None or len(tr.boxes) < 2:
            continue
        times, speeds = track_speed_series(tr, window)
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt > 0:
                total += speeds[i] * dt
                n_speed_samples += 1

    return {
        "distance": total,
        "unit": "bbox_heights",
        "calibrated": False,
        "track_ids": list(identity_track_ids),
        "n_track_fragments": len(identity_track_ids),
        "n_speed_samples": n_speed_samples,
    }
