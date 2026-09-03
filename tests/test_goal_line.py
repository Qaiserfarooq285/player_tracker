"""Pure-logic unit tests for `src/events/goal_line.py` (Stage 6 of the evidence-based event
redesign, owner spec 2026-09-02): GOAL = ball crosses the LINE between the posts, never a
bounding-box/polygon containment test. No GPU/video decode.
"""

from __future__ import annotations

from pathlib import Path

from src.common.io import load_yaml
from src.events import goal_line
from src.events.ball_track import BallState

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "events.yaml")["goal"]


def _state(t: float, x: float, y: float, conf: float = 0.9, known: bool = True) -> BallState:
    return BallState(
        t=t,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        speed=0.0,
        direction_deg=None,
        confidence=conf,
        is_interpolated=False,
        is_camera_compensated=True,
        known=known,
    )


# ---------------------------------------------------------------------------
# goal_line_from_posts
# ---------------------------------------------------------------------------


def test_goal_line_from_two_posts_uses_base_points():
    """Base point = bottom-center of each post's own bbox -- where the post meets the pitch."""
    posts = [(100.0, 50.0, 110.0, 200.0), (300.0, 50.0, 310.0, 200.0)]
    line = goal_line.goal_line_from_posts(posts)
    assert line == ((105.0, 200.0), (305.0, 200.0))


def test_goal_line_orders_left_to_right_regardless_of_input_order():
    posts = [(300.0, 50.0, 310.0, 200.0), (100.0, 50.0, 110.0, 200.0)]  # right post listed first
    line = goal_line.goal_line_from_posts(posts)
    assert line[0][0] < line[1][0]


def test_goal_line_fewer_than_two_posts_is_none():
    assert goal_line.goal_line_from_posts([]) is None
    assert goal_line.goal_line_from_posts([(100.0, 50.0, 110.0, 200.0)]) is None


# ---------------------------------------------------------------------------
# ball_crosses_goal_line -- the actual scoring test
# ---------------------------------------------------------------------------


def test_ball_crossing_directly_between_posts_is_a_goal():
    """Owner spec §22 (direct goal): the ball moves from clearly outside the goal to clearly
    inside it, crossing the line strictly between the two posts."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))  # horizontal goal line at y=200, x in [100,300]
    prev = _state(0.0, 200.0, 150.0)  # above the line (in front of goal), between the posts
    curr = _state(0.1, 200.0, 250.0)  # below the line (in the net)
    crossing = goal_line.ball_crosses_goal_line(prev, curr, line, cfg)
    assert crossing is not None
    assert 100.0 < crossing[0] < 300.0


def test_ball_passing_outside_the_posts_is_not_a_goal():
    """Owner spec §15/§17: the ball must cross BETWEEN the posts -- passing the goal LINE's own
    infinite extension outside the two posts (e.g. wide of the goal) must not count."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 400.0, 150.0)  # well wide of the right post
    curr = _state(0.1, 400.0, 250.0)
    crossing = goal_line.ball_crosses_goal_line(prev, curr, line, cfg)
    assert crossing is None


def test_ball_merely_approaching_without_crossing_is_not_a_goal():
    """Owner spec §17: 'ball is close to the goal line' must not, by itself, be a goal -- only an
    actual crossing counts."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 200.0, 150.0)
    curr = _state(0.1, 200.0, 195.0)  # very close to the line, but still on the same side
    crossing = goal_line.ball_crosses_goal_line(prev, curr, line, cfg)
    assert crossing is None


def test_goalpost_rebound_without_a_later_crossing_is_not_a_goal():
    """Owner spec §19/test 7: ball -> post -> rebounds away. The ball approaches the line but its
    OWN movement segment ends before crossing (bounces back), so no crossing point exists."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 200.0, 150.0)
    curr = _state(0.1, 200.0, 190.0)  # rebounds off the post/bar, stays on the approach side
    assert goal_line.ball_crosses_goal_line(prev, curr, line, cfg) is None


def test_missing_ball_state_on_either_side_is_never_a_goal():
    """Owner spec: never create a goal because the ball 'disappeared' -- a gap in tracking must
    read as no evidence, not as a crossing."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 200.0, 150.0, known=False)
    curr = _state(0.1, 200.0, 250.0)
    assert goal_line.ball_crosses_goal_line(prev, curr, line, cfg) is None


def test_low_confidence_ball_state_is_never_a_goal():
    """A goal deserves a stricter confidence floor than an ordinary touch/pass -- a
    likely-interpolated/uncertain ball position must never decide the highest-stakes event."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    low_conf = cfg["min_ball_confidence_for_goal"] - 0.01
    prev = _state(0.0, 200.0, 150.0, conf=low_conf)
    curr = _state(0.1, 200.0, 250.0)
    assert goal_line.ball_crosses_goal_line(prev, curr, line, cfg) is None


def test_crossing_gap_too_large_is_not_a_goal():
    """Two ball sightings either side of a long gap are two unrelated observations, not one
    continuous shot -- must not be treated as a crossing even if they land on opposite sides."""
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 200.0, 150.0)
    curr = _state(0.0 + cfg["max_crossing_gap_s"] + 0.5, 200.0, 250.0)
    assert goal_line.ball_crosses_goal_line(prev, curr, line, cfg) is None


def test_ball_staying_on_the_same_side_never_crosses():
    cfg = _cfg()
    line = ((100.0, 200.0), (300.0, 200.0))
    prev = _state(0.0, 150.0, 150.0)
    curr = _state(0.1, 250.0, 160.0)  # moved, but never reached y=200
    assert goal_line.ball_crosses_goal_line(prev, curr, line, cfg) is None
