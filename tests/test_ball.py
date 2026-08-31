"""Pure-logic unit tests for Stage 1's ball-trajectory plausibility filter
(`src.detect.ball.filter_implausible_ball_jumps`, CLAUDE.md Stage 1 plan 2026-08-31) — no
GPU/video I/O.
"""

from __future__ import annotations

from src.common.logging import DropCounter
from src.common.types import BallDetection, BBox
from src.detect.ball import filter_implausible_ball_jumps

FRAME_WIDTH = 1920.0

CFG = {
    "max_plausible_speed_fw_per_s": 10.0,
    "reseed_run_length": 5,
}


def _ball(cx: float, cy: float, t: float, conf: float = 0.5) -> BallDetection:
    half = 10.0
    return BallDetection(
        bbox=BBox(x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half),
        conf=conf,
        frame_index=round(t * 30),
        t=t,
        interpolated=False,
    )


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_empty_and_single_observation_pass_through_unchanged():
    assert filter_implausible_ball_jumps([], FRAME_WIDTH, CFG) == []
    one = [_ball(900, 500, 0.0)]
    assert filter_implausible_ball_jumps(one, FRAME_WIDTH, CFG) == one


def test_smooth_trajectory_is_fully_accepted():
    # The ball drifts a few px/frame -- well under any plausible-speed threshold.
    balls = [_ball(900 + i * 5, 500, i * 0.033) for i in range(20)]
    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG)
    assert kept == balls


def test_single_frame_teleport_is_rejected_and_logged():
    # Real clip1 pattern (module docstring): the true ball sits near cx~880, a single-frame
    # detector swap jumps to cx~1750 (~24 fw/s at 33ms), then returns.
    balls = [
        _ball(880, 580, 0.000),
        _ball(1750, 345, 0.033),  # implausible jump -- a detector swap, not ball motion
        _ball(875, 580, 0.067),  # plausible relative to the ORIGINAL anchor two samples back
        _ball(870, 580, 0.100),
    ]
    drops = DropCounter("test")
    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG, drops)
    assert [b.bbox.cx for b in kept] == [880, 875, 870]
    assert drops.as_dict() == {"ball_implausible_jump": 1}


def test_a_lone_rejected_sample_never_triggers_reseed():
    # One bad frame, even if it's internally "plausible" only relative to itself, must not be
    # enough to flip the anchor -- reseed requires `reseed_run_length` CONSECUTIVE agreeing
    # rejections, not just one.
    balls = [
        _ball(880, 580, 0.000),
        _ball(1750, 345, 0.033),
        _ball(878, 581, 0.067),
    ]
    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG)
    assert [b.bbox.cx for b in kept] == [880, 878]


# ---------------------------------------------------------------------------
# The bad-anchor failure mode the plan explicitly calls out
# ---------------------------------------------------------------------------


def test_filter_implausible_ball_jumps_reseeds_after_bad_anchor():
    """If the FIRST accepted observation is itself a false positive, a naive
    "reject anything far from the last accepted point" filter would reject every subsequent
    REAL observation forever (they all look like a huge jump from a stale, wrong anchor).

    Construct exactly that: a bogus first point far from where the real ball then travels
    smoothly for many samples. The filter must recognise the sustained, mutually-consistent run
    as the real trajectory and re-seed onto it rather than discarding it wholesale.
    """
    bad_anchor = _ball(3800, 100, 0.0)  # a one-off false positive, e.g. a stray graphic
    real_trajectory = [_ball(200 + i * 5, 500, 0.033 * (i + 1)) for i in range(10)]
    balls = [bad_anchor, *real_trajectory]

    drops = DropCounter("test")
    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG, drops)

    # The bad anchor itself is discarded, but the entire real trajectory that follows it survives
    # -- NOT rejected wholesale just because each point individually looks far from bad_anchor.
    kept_cx = [b.bbox.cx for b in kept]
    assert bad_anchor.bbox.cx not in kept_cx
    assert kept_cx == [b.bbox.cx for b in real_trajectory]
    assert drops.as_dict() == {"ball_implausible_jump": 1}  # only the lone bad anchor is dropped


def test_a_short_run_of_agreeing_rejections_below_reseed_length_still_gets_dropped():
    # A run of mutually-consistent rejections that is SHORTER than reseed_run_length is still
    # noise, not a re-seed trigger -- the original anchor reasserts itself and the whole run is
    # rejected. This is what stops a brief 1-2 frame detector excursion (not the sustained-run
    # case above) from ever being promoted.
    anchor = _ball(200, 500, 0.0)
    short_excursion = [_ball(3000 + i * 5, 100, 0.033 * (i + 1)) for i in range(3)]  # len 3 < 5
    resumes_from_anchor = _ball(210, 500, 0.033 * 4)
    balls = [anchor, *short_excursion, resumes_from_anchor]

    drops = DropCounter("test")
    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG, drops)

    assert [b.bbox.cx for b in kept] == [200, 210]
    assert drops.as_dict() == {"ball_implausible_jump": 3}


def test_reseed_requires_mutual_consistency_not_just_a_long_run():
    # A long run of rejections that DISAGREE with each other (still erratic, not a real sustained
    # trajectory) never accumulates into one re-seedable pending run -- each new outlier resets
    # the pending run instead of extending it. Every point below is placed far enough from BOTH
    # the anchor and its erratic neighbour (comfortably above max_plausible_speed_fw_per_s even
    # accounting for the largest dt any pairing in this sequence could see) that none can ever be
    # read as a plausible continuation of anything else.
    anchor = _ball(200, 500, 0.0)
    erratic = [
        _ball(5000, 500, 0.033),
        _ball(5000, 6000, 0.067),
        _ball(10000, 500, 0.100),
        _ball(10000, 6000, 0.133),
        _ball(5000, 3000, 0.167),
    ]
    balls = [anchor, *erratic]

    kept = filter_implausible_ball_jumps(balls, FRAME_WIDTH, CFG)

    assert kept == [anchor]  # nothing erratic was ever accepted or re-seeded onto


def test_frame_width_non_positive_is_a_no_op():
    balls = [_ball(0, 0, 0.0), _ball(9999, 9999, 0.033)]
    assert filter_implausible_ball_jumps(balls, 0.0, CFG) == balls


def test_unsorted_input_is_sorted_by_time_before_filtering():
    ordered = [_ball(900, 500, 0.0), _ball(905, 500, 0.033), _ball(910, 500, 0.067)]
    shuffled = [ordered[2], ordered[0], ordered[1]]
    kept = filter_implausible_ball_jumps(shuffled, FRAME_WIDTH, CFG)
    assert kept == ordered
