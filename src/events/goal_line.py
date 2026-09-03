"""Stage 6 of the evidence-based event redesign (owner spec, 2026-09-02): a GOAL requires the
ball's own trajectory to cross the goal LINE between the two posts -- never a bounding-box/polygon
CONTAINMENT test.

Owner spec, verbatim: "DO NOT detect the penalty box or generic goal box as the primary goal
detector. The system does NOT need to detect the entire penalty area. The critical event is BALL
CROSSES GOAL LINE inside the space between the goal posts." And: "Do NOT call a goal because: ball
enters penalty area / ball is near goal / ball is near goalkeeper / player shoots / ball
disappears near goal / players celebrate / ball touches the post / ball is inside the six-yard box
/ ball is close to the goal line."

**What this replaces.** `src.events.goals.detect_goals_goal_region` already exists and is kept
UNCHANGED for the human-marked-polygon path (`configs/goal_region.yaml`, ADR-20's own
human-in-the-loop escape hatch -- a person drawing a box IS an accepted, explicit design, not the
same thing as an algorithm inventing one). But that same containment test was ALSO being used for
the auto-tracked `GoalStructure` (Stage C/D) case, via `_tracked_polygons_for_take` building a
polygon out of the crossbar+posts bounding box -- exactly the "box as goal proxy" anti-pattern the
owner is asking to stop. This module gives the auto-tracked case a REAL line-crossing test instead,
using the SAME underlying `GoalStructure.posts` geometry Stage C already computes and caches --
no new detection model, no new dependency (Golden Rule 2).

**Geometry.** The goal line is the segment connecting the two posts' own BASE points (bottom-center
of each post's bbox -- where a post meets the ground, which is where the actual goal line is
painted). A GOAL requires two temporally-CONSECUTIVE, sufficiently-confident ball positions whose
connecting segment intersects that goal-line segment, with the intersection strictly between the
two base points (never merely "close to" the line, never "inside" any enclosing box). Reuses
standard 2D segment-segment intersection -- no new geometry engine, plain arithmetic.
"""

from __future__ import annotations

from src.events.ball_track import BallState


def goal_line_from_posts(
    posts: list[tuple[float, float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The goal-line segment `(left_post_base, right_post_base)` from `GoalStructure.posts`
    (`src.goal.detect.GoalStructure`/`goal_posts_at`) -- `None` when fewer than 2 posts are
    tracked (an honest "no goal line to test against", never a guessed single-post line).

    A post's own BASE point is `((x1+x2)/2, y2)` -- bottom-center of its bbox, i.e. where the post
    meets the pitch. "Left"/"right" is resolved by base x-coordinate so the segment is always
    built consistently regardless of the order Stage C happened to store the two posts in.
    """
    if len(posts) < 2:
        return None
    bases = sorted(
        (((x1 + x2) / 2.0, y2) for x1, y1, x2, y2 in posts),
        key=lambda p: p[0],
    )
    return bases[0], bases[-1]


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Standard 2D segment-segment intersection point of `p1->p2` and `p3->p4`, or `None` when
    they don't cross (parallel, or crossing outside either segment's own bounds). Pure arithmetic,
    no external geometry library -- this is the entire "does the ball's own movement cross the
    goal line" test, deliberately NOT a bounding-box/polygon containment check.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # parallel (or degenerate) -- no single crossing point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None  # crossing point falls outside one of the two segments' own extents
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def ball_crosses_goal_line(
    prev: BallState,
    curr: BallState,
    goal_line: tuple[tuple[float, float], tuple[float, float]],
    goal_cfg: dict,
) -> tuple[float, float] | None:
    """Whether the ball's movement from `prev` to `curr` crosses `goal_line` BETWEEN the posts --
    the actual goal-scoring geometric test. Returns the crossing point, or `None`.

    Both `prev`/`curr` must be `known` (never test across a hole in the ball's own trajectory --
    owner spec: "do NOT immediately create a...goal because the ball disappeared") and clear
    `goal_cfg['min_ball_confidence_for_goal']` (a goal is the single highest-stakes call this
    project makes; it deserves a stricter confidence floor than an ordinary touch/pass). The two
    samples must also be close enough in time (`goal_cfg['max_crossing_gap_s']`) that "the ball
    was on one side, now it's on the other" is actually one continuous movement, not two unrelated
    sightings either side of a long gap.
    """
    if not prev.known or not curr.known:
        return None
    if prev.confidence < goal_cfg["min_ball_confidence_for_goal"]:
        return None
    if curr.confidence < goal_cfg["min_ball_confidence_for_goal"]:
        return None
    if (curr.t - prev.t) > goal_cfg["max_crossing_gap_s"]:
        return None
    return _segments_intersect((prev.x, prev.y), (curr.x, curr.y), goal_line[0], goal_line[1])
