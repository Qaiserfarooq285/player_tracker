"""Best-effort touch detection (ADR-13/14; CLAUDE.md §13.3 "Touch, Pass, Tackle, Save" row).

⚠️ Owner-authorized exception to Golden Rule 7 (CLAUDE.md §12/ADR-13): this is pure ARITHMETIC on
top of already-cached `Detection`/`Track`/`BallDetection` artifacts — no training, no new
pretrained model (Golden Rule 2 still holds). A "touch" here means "the ball's centroid came
within `touch_distance_px` of some player's bbox" — a coarse, honestly low-confidence proxy for
actual ball contact, never a verified event (Golden Rule 5): two players standing close together
can make this heuristic credit the wrong one, and a partially-occluded/interpolated ball detection
can be off by enough pixels to miss or falsely trigger a touch. See `configs/events.yaml: touch`
for the real measured pixel-distance distribution (on `work/clip2_77`/`work/clip4_77`) that
grounds `touch_distance_px`, and this module's own `nearest_player_edge_distance`/
`point_to_bbox_distance`, which `src/events/possession.py` imports rather than reimplementing
(CLAUDE.md task spec: touches and possession share the same "who has the ball" core geometry).

`player_track_id` on an emitted `TOUCH` event is the (optionally continuity-stitched) player
IDENTITY nearest the ball at that instant — pass `identity_of` (from
`src/track/continuity.py::build_take_identities`) so a touch during a brief occlusion still
attributes to the same person as the touch before/after it, rather than splitting across raw
tracker fragment ids. `identity_of` is optional: a caller that hasn't built identities yet gets
raw `Track.id`s back (harmless, just less continuous — CLAUDE.md Golden Rule 3 is unaffected
either way, since `take_id` is always stamped from the caller's own bucketing).
"""

from __future__ import annotations

import math
import uuid

from src.common.logging import DropCounter
from src.common.types import BallDetection, BBox, DetectionClass, Event, EventType, Track, TrackBox
from src.events.ball_track import BallState, nearest_ball_state

# Track classes that can plausibly "have" the ball for touch/possession purposes — a match
# official is never credited with a touch/possession (CLAUDE.md task; matches
# `src/highlights/selection.py::heuristic_fallback_seed`'s own referee exclusion).
POSSESSOR_CLASSES = {DetectionClass.PLAYER, DetectionClass.GOALKEEPER}


def point_to_bbox_distance(px: float, py: float, bbox: BBox) -> float:
    """Euclidean distance from point `(px, py)` to the nearest point of `bbox` — `0.0` when the
    point falls INSIDE (or on the boundary of) the box. Pure geometry, no config/state, reused by
    `src/events/possession.py` and `src/events/tackles.py`."""
    dx = max(bbox.x1 - px, 0.0, px - bbox.x2)
    dy = max(bbox.y1 - py, 0.0, py - bbox.y2)
    return (dx * dx + dy * dy) ** 0.5


def nearest_box_in_time(track: Track, t: float, tolerance_s: float) -> TrackBox | None:
    """The track's own box closest in time to `t`, if within `tolerance_s` seconds (mirrors
    `src.highlights.selection._nearest_box` — a tiny, config-free geometry helper, not "stitching
    logic" in the CLAUDE.md task-spec sense, so re-deriving it here rather than importing across
    the events/highlights module boundary is fine; it is itself reused by `src/events/possession.py`
    and `src/events/tackles.py` rather than being re-derived a third/fourth time)."""
    best: TrackBox | None = None
    best_dt = tolerance_s
    for box in track.boxes:
        dt = abs(box.t - t)
        if dt <= best_dt:
            best, best_dt = box, dt
    return best


def nearest_player_edge_distance(
    ball: BallDetection, tracks: list[Track], match_tolerance_s: float
) -> tuple[int, float] | None:
    """The `(raw_track_id, point_to_bbox_distance)` of whichever candidate track's own box
    (nearest in time to `ball.t`, within `match_tolerance_s`) sits closest to the ball's centroid.

    `tracks` should already be filtered to plausible possessors by the caller (excluding
    referees, e.g. via `POSSESSOR_CLASSES`/`dominant_class`) — this function itself makes no
    class judgement, only geometry, so both `touches.py` and `possession.py` can filter however
    is appropriate for their own call site. Returns `None` when no candidate has a box within
    `match_tolerance_s` of `ball.t` at all.
    """
    best_id: int | None = None
    best_dist: float | None = None
    for tr in tracks:
        box = nearest_box_in_time(tr, ball.t, match_tolerance_s)
        if box is None:
            continue
        dist = point_to_bbox_distance(ball.bbox.cx, ball.bbox.cy, box.bbox)
        if best_dist is None or dist < best_dist:
            best_dist, best_id = dist, tr.id
    if best_id is None or best_dist is None:
        return None
    return best_id, best_dist


def touch_confidence(distance_px: float, ball_conf: float, touch_cfg: dict) -> float:
    """Confidence scales with how much margin `distance_px` clears BELOW `touch_distance_px`
    (closer = more confident) and with the ball detection's own confidence, hard-clamped to
    `[min_confidence, max_confidence]` (Golden Rule 5: proximity is a coarse proxy for actual
    contact, never treated as certain — see module docstring)."""
    max_d = touch_cfg["touch_distance_px"]
    closeness = 1.0 - (distance_px / max_d) if max_d > 0 else 1.0
    closeness = max(0.0, min(1.0, closeness))
    span = touch_cfg["max_confidence"] - touch_cfg["min_confidence"]
    raw = touch_cfg["min_confidence"] + closeness * ball_conf * span
    return min(touch_cfg["max_confidence"], max(touch_cfg["min_confidence"], raw))


def has_contact_evidence(
    ball_states: list[BallState], t: float, touch_cfg: dict
) -> tuple[bool, dict]:
    """Whether the ball's OWN trajectory shows real evidence of contact around instant `t` --
    owner spec: "ball trajectory/velocity/direction changes after the interaction... Do NOT count
    a touch because the ball is nearby [or] bounding boxes overlap." Proximity alone (the caller's
    own distance/gate) is necessary but never sufficient.

    Compares the ball's velocity vector shortly BEFORE `t` to shortly AFTER `t`
    (`touch_cfg['contact_window_s']` each side) -- a single "how much did the vector change"
    magnitude naturally covers a direction change (a deflection/pass away), a speed increase (a
    kick), AND a speed decrease (a trap/first touch) without needing three separate tests.

    Returns `(has_evidence, debug)` -- `debug` always carries the before/after states (or `None`)
    and the computed delta, so a caller building `Event.evidence` never has to re-derive it.
    Missing or `known=False` ball state on EITHER side is an HONEST "cannot confirm" -> `False`,
    never treated as "no change detected, so nothing happened" being conflated with an actual
    steady-state reading (Golden Rule 5: absence of ball evidence is not evidence of no contact,
    but it is also not evidence FOR a touch -- accuracy over completeness, per spec).
    """
    window = touch_cfg["contact_window_s"]
    before = nearest_ball_state(ball_states, t - window, window)
    after = nearest_ball_state(ball_states, t + window, window)
    debug = {"before": before, "after": after, "delta_v": None}
    if before is None or after is None or not before.known or not after.known:
        return False, debug
    delta_v = math.hypot(after.vx - before.vx, after.vy - before.vy)
    debug["delta_v"] = delta_v
    return delta_v >= touch_cfg["min_velocity_change"], debug


def detect_touches(
    balls: list[BallDetection],
    tracks: list[Track],
    take_id: int | None,
    events_cfg: dict,
    ball_states: list[BallState],
    identity_of: dict[int, int] | None = None,
    drops: DropCounter | None = None,
) -> list[Event]:
    """Detect touch events within one take's own ball detections + tracks (`take_id` is stamped
    directly onto every emitted `Event` — the caller is responsible for having already bucketed
    both `balls` and `tracks` into a single take, so a touch can never span a cut, Golden Rule 3).

    Asserted defensively (not just documented): raw `Track.id`s RESET per take (Golden Rule 3), so
    silently accepting a `tracks` list spanning more than one `take_id` risks two different takes'
    fragments colliding on the same integer id and being merged as if they were one identity — the
    same failure mode `src/track/continuity.py`'s `stitch_timeline`/`build_take_identities` already
    guard against.

    Debounced per (stitched) identity via `touch_min_gap_s`: continuous ball proximity to the same
    player (e.g. a stationary close-control dribble) fires one touch per debounce window, not one
    per ball sample. This value is compared in real SECONDS against `ball.t` timestamps, so it is
    already FPS-independent by construction (a 25fps and a 10fps take with the same real-world
    dribble rate produce the same debounce behaviour) -- no separate "FPS-aware" conversion needed.

    `ball_states` (`src.events.ball_track.build_ball_state_series`, built ONCE per take by the
    caller) supplies the CONTACT evidence this heuristic was missing entirely before 2026-09-02:
    proximity alone used to be sufficient ("a coarse proxy for actual ball contact" -- this
    module's own prior docstring, now fixed). `has_contact_evidence` requires a measurable
    velocity-vector change in the ball's own trajectory around the candidate instant; proximity
    without it is dropped as `touch_no_contact_evidence`, never counted (owner spec: "Do NOT count
    a touch because... the ball is nearby").
    """
    take_ids = {tr.take_id for tr in tracks}
    assert len(take_ids) <= 1, "detect_touches must only ever see one take's tracks"

    touch_cfg = events_cfg["touch"]
    min_emit = events_cfg["confidence"]["min_emit_confidence"]
    candidates = [tr for tr in tracks if tr.dominant_class in POSSESSOR_CLASSES]

    last_touch_t: dict[int, float] = {}
    events: list[Event] = []
    for ball in sorted(balls, key=lambda b: b.t):
        if ball.conf < touch_cfg["min_ball_conf"]:
            if drops is not None:
                drops.drop("touch_ball_conf_too_low")
            continue

        nearest = nearest_player_edge_distance(ball, candidates, touch_cfg["match_tolerance_s"])
        if nearest is None:
            if drops is not None:
                drops.drop("touch_no_track_near_ball_in_time")
            continue
        raw_id, distance = nearest
        if distance > touch_cfg["touch_distance_px"]:
            if drops is not None:
                drops.drop("touch_ball_too_far_from_any_player")
            continue

        identity_id = identity_of.get(raw_id, raw_id) if identity_of else raw_id
        last_t = last_touch_t.get(identity_id)
        if last_t is not None and (ball.t - last_t) < touch_cfg["touch_min_gap_s"]:
            if drops is not None:
                drops.drop("touch_debounced")
            continue

        contact, contact_debug = has_contact_evidence(ball_states, ball.t, touch_cfg)
        if not contact:
            if drops is not None:
                drops.drop("touch_no_contact_evidence")
            continue

        confidence = touch_confidence(distance, ball.conf, touch_cfg)
        if confidence < min_emit:
            if drops is not None:
                drops.drop("touch_below_min_emit_confidence")
            continue

        last_touch_t[identity_id] = ball.t
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.TOUCH,
                t_start=ball.t,
                t_end=ball.t,
                player_track_id=identity_id,
                take_id=take_id,
                confidence=confidence,
                source="ball_proximity_and_velocity_change_heuristic",
                evidence={
                    "raw_track_id": raw_id,
                    "distance_px": distance,
                    "touch_distance_px": touch_cfg["touch_distance_px"],
                    "ball_confidence": ball.conf,
                    "ball_velocity_delta": contact_debug["delta_v"],
                    "calibrated": False,
                    "unit": "pixels_at_detect_stage_resolution",
                },
            )
        )
    return events
