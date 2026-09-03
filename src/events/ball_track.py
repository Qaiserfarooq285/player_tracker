"""Stage 1 of the evidence-based event redesign (owner spec, 2026-09-02): a per-take BALL
TRAJECTORY STATE series — position, velocity, direction, confidence, an explicit UNKNOWN state —
that every downstream detector (possession/touch/pass/shot/goal) consumes instead of re-deriving
raw ball motion itself.

Owner's own framing: "the system should maintain ball_position, ball_velocity, ball_direction,
ball_confidence, ball_track_history for every frame where possible... if ball confidence is too
low, ball_state = UNKNOWN and do not create a false event."

**Reuses, does not reimplement, two things that already exist and already work:**
- Short-gap bridging is ALREADY Stage 2's job (`src.detect.ball.interpolate_ball_gaps`,
  `configs/detect.yaml: ball_interpolation`) -- it fills gaps up to a measured `max_gap_frames`,
  flags `BallDetection.interpolated=True`, and applies a confidence penalty (never presents an
  interpolated sample as observed, Golden Rule 5). This module does NOT re-bridge gaps; it reads
  the flag that Stage 2 already set and treats a gap Stage 2 declined to bridge as an honest hole
  in the series, never invents a position across it.
- `moving_average` (`src/events/segments.py`) -- the same trailing-window smoothing
  `src/events/shots.py::ball_speed_series` already uses, extended here to a full 2D velocity
  vector in camera-compensated coordinates (that function's own frame-widths-per-second speed
  scalar stays as-is for the existing shot heuristic; this module is the new, richer series the
  redesigned possession/touch/pass/shot/goal detectors consume).

**Camera compensation** (new, and the actual point of this module): a fast pan measurably breaks a
raw ball-speed reading exactly the way it broke fragment stitching earlier this session (see
`src/track/camera_motion.py`'s own module docstring -- the 25s the target's own box appeared to
teleport 3.45 bbox-heights/s and turned out to be 0.24 once the camera's own motion was removed).
Ball velocity is computed in the SAME take-reference frame via `TakeCameraMotion.to_reference`,
falling back to raw image-space displacement (flagged `is_camera_compensated=False`) only for an
instant with no usable motion estimate -- never silently guessed as zero motion.

**Units**: bbox-heights/second, the same normalized unit already used by `sprint_speed_threshold`
and `src.track.continuity`'s stitching gates (CLAUDE.md ADR-6: never a fabricated metric unit).
The scale is one PER-TAKE reference height (median player bbox height across that take's own
tracks) rather than a per-frame nearest-player height, deliberately: the ball has no natural
"height" of its own, and a per-frame divisor risks blowing up when no player is near the ball at
that instant, which is common and not itself evidence of anything.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.common.types import BallDetection, Track
from src.events.segments import moving_average


class BallState(BaseModel):
    """One instant of the ball's own trajectory state. `known=False` (a genuine `UNKNOWN` state,
    owner's own term) means every downstream detector must treat this instant as having NO ball
    evidence at all -- never a reason to emit an event, and never treated as "ball is stationary".
    """

    t: float
    x: float  # take-reference-frame pixel position (camera-compensated when possible)
    y: float
    vx: float  # bbox-heights per second, take-reference frame
    vy: float
    speed: float  # hypot(vx, vy)
    direction_deg: float | None  # atan2(vy, vx) in degrees; None when speed is ~0 (no direction)
    confidence: float
    is_interpolated: bool  # passed through from Stage 2's own BallDetection.interpolated
    is_camera_compensated: bool  # False only when no motion estimate covered this instant
    known: bool = True


def reference_player_height(tracks: list[Track]) -> float:
    """Median player-box height across ONE take's own tracks -- the single scale `speed`/`vx`/`vy`
    are normalized by. `0.0` (never a crash) when a take has no boxes at all; callers must treat
    that as "cannot normalize" and fall back to leaving positions in raw pixels, flagged as such.
    """
    heights = [b.bbox.height for tr in tracks for b in tr.boxes if b.bbox.height > 0]
    if not heights:
        return 0.0
    heights.sort()
    mid = len(heights) // 2
    return heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0


def build_ball_state_series(
    balls: list[BallDetection],
    ball_cfg: dict,
    reference_height_px: float,
    motion=None,
) -> list[BallState]:
    """One take's own ball detections -> a time-ordered `BallState` series.

    Steps, in order:
    1. Drop any sample below `ball_cfg['min_conf_for_state']` entirely -- it becomes a hole in the
       series (an absence), never a low-confidence `BallState` object that a careless caller might
       still read a position out of.
    2. Camera-compensate each KEPT sample's position via `motion.to_reference` when a `motion`
       model is supplied and covers that instant; otherwise keep the raw position and flag
       `is_camera_compensated=False`.
    3. Smooth the (possibly mixed raw/compensated) x/y series with `moving_average` -- identical
       technique to `shots.py::ball_speed_series`, same `ball_cfg['smoothing_window_frames']`
       naming convention as `shot`/`save` in `configs/events.yaml`.
    4. Difference consecutive SMOOTHED samples for `(vx, vy)`, normalized into bbox-heights/second
       by `reference_height_px` (falls back to raw pixels/second, still flagged, if the take had
       no measurable player height at all -- `reference_height_px <= 0`).
    5. A gap between two KEPT samples wider than `ball_cfg['max_gap_bridge_s']` breaks the
       velocity computation across it (emits `vx=vy=0, known=False` for the sample right after the
       gap) -- Stage 2 already declined to bridge that gap, so this module must not manufacture a
       velocity across a hole Stage 2 itself found too large to trust.
    """
    import math

    kept = sorted(
        (b for b in balls if b.conf >= ball_cfg["min_conf_for_state"]), key=lambda b: b.t
    )
    if not kept:
        return []

    raw_x = [b.bbox.cx for b in kept]
    raw_y = [b.bbox.cy for b in kept]
    comp_x: list[float] = []
    comp_y: list[float] = []
    compensated_flags: list[bool] = []
    for b, x, y in zip(kept, raw_x, raw_y, strict=True):
        ref = motion.to_reference(b.t, x, y) if motion is not None else None
        if ref is not None:
            comp_x.append(ref[0])
            comp_y.append(ref[1])
            compensated_flags.append(True)
        else:
            comp_x.append(x)
            comp_y.append(y)
            compensated_flags.append(False)

    window = ball_cfg["smoothing_window_frames"]
    smooth_x = moving_average(comp_x, window)
    smooth_y = moving_average(comp_y, window)

    max_gap_s = ball_cfg["max_gap_bridge_s"]
    scale = reference_height_px if reference_height_px > 0 else 1.0

    series: list[BallState] = []
    for i, b in enumerate(kept):
        if i == 0:
            vx = vy = 0.0
            known = True  # a single leading sample has real position, just no velocity yet
        else:
            dt = b.t - kept[i - 1].t
            gap_too_large = dt <= 0 or dt > max_gap_s
            if gap_too_large:
                vx = vy = 0.0
                known = False
            else:
                vx = (smooth_x[i] - smooth_x[i - 1]) / dt / scale
                vy = (smooth_y[i] - smooth_y[i - 1]) / dt / scale
                known = True
        speed = math.hypot(vx, vy)
        direction = math.degrees(math.atan2(vy, vx)) if speed > 1e-6 else None
        series.append(
            BallState(
                t=b.t,
                x=comp_x[i],
                y=comp_y[i],
                vx=vx,
                vy=vy,
                speed=speed,
                direction_deg=direction,
                confidence=b.conf,
                is_interpolated=b.interpolated,
                is_camera_compensated=compensated_flags[i],
                known=known,
            )
        )
    return series


def nearest_ball_state(series: list[BallState], t: float, tolerance_s: float) -> BallState | None:
    """The series' own state closest in time to `t`, within `tolerance_s` -- `None` when nothing
    qualifies (an honest "no ball evidence here", same convention as
    `src.events.touches.nearest_box_in_time`). A caller must treat a `None` result AND a
    `known=False` result identically: no event may be emitted from either."""
    best: BallState | None = None
    best_dt = tolerance_s
    for state in series:
        dt = abs(state.t - t)
        if dt <= best_dt:
            best, best_dt = state, dt
    return best
