"""Full best-effort event suite for one take + target attribution (ADR-13/14/15 combined).

`src/pipeline/run.py::_compute_events` (the original Phase-1 flow, `target_jersey is not None`)
only ever computes sprint+shot — the always-available Phase-1 events. This module is the ADR-15
extended-output path's own orchestrator: it ADDITIONALLY runs the ADR-13/14 heuristic modules
(touch/possession/pass/dribble/tackle/save — already built, tested, but never previously wired
into any pipeline entrypoint) across a WHOLE take's tracks (every player, not just one target,
since possession/pass/tackle need to reason about interactions BETWEEN two different players),
then filters the result down to one verified target's own events.
"""

from __future__ import annotations

from src.common.logging import DropCounter
from src.common.types import BallDetection, Event, EventType, Take, Track
from src.events.possession import (
    DEFAULT_BALL_FPS,
    detect_dribbles,
    detect_passes,
    detect_possession,
    possession_runs_for_take,
)
from src.events.saves import detect_saves
from src.events.shots import detect_shots
from src.events.sprints import detect_sprints
from src.events.tackles import detect_tackles
from src.events.touches import detect_touches
from src.highlights.ranking import attribute_shot
from src.track.continuity import build_take_identities


def compute_take_all_events(
    take: Take,
    take_tracks: list[Track],
    take_balls: list[BallDetection],
    frame_width: float,
    events_cfg: dict,
    selection_cfg: dict,
    fps_sample: float,
    ball_fps: float = DEFAULT_BALL_FPS,
    drops: DropCounter | None = None,
) -> tuple[list[Event], dict[int, int], dict[int, float]]:
    """Every best-effort event category for ONE take, across ALL of that take's own tracks.

    Returns `(events, identity_of, identity_confidence)` — the `build_take_identities` partition
    is returned alongside so a caller can attribute a subset to one verified target
    (`attribute_events_to_target` below) without recomputing identities a second time.
    """
    identity_of, identity_confidence = build_take_identities(take_tracks, selection_cfg)

    events: list[Event] = []
    for tr in take_tracks:
        events.extend(detect_sprints(tr, events_cfg, fps_sample, drops))
    events.extend(detect_shots(take_balls, take.id, frame_width, events_cfg, drops))
    events.extend(detect_touches(take_balls, take_tracks, take.id, events_cfg, identity_of, drops))

    runs = possession_runs_for_take(take_balls, take_tracks, events_cfg, identity_of)
    events.extend(
        detect_possession(runs, take.id, events_cfg, ball_fps, identity_confidence, drops)
    )
    events.extend(
        detect_passes(runs, take_tracks, take.id, events_cfg, ball_fps, identity_confidence, drops)
    )
    events.extend(
        detect_dribbles(
            runs, take_tracks, take.id, events_cfg, ball_fps, identity_confidence, drops
        )
    )
    events.extend(
        detect_tackles(
            runs, take_tracks, take_balls, take.id, events_cfg, ball_fps, identity_confidence, drops
        )
    )
    events.extend(
        detect_saves(
            take_balls,
            take_tracks,
            take.id,
            frame_width,
            events_cfg,
            events_cfg["possession"]["match_tolerance_s"],
            drops,
        )
    )
    return events, identity_of, identity_confidence


def target_identity_id(location_track_ids: list[int], identity_of: dict[int, int]) -> int | None:
    """Which `build_take_identities` identity id corresponds to the verified target, by MAJORITY
    overlap with the target's own location-selection track ids.

    Both partitions come from the same greedy forward-stitch algorithm (`extend_chain_forward`),
    just seeded differently (the target's own arrow/heuristic seed vs. `build_take_identities`'
    earliest-unclaimed-fragment order) — they agree in the overwhelming majority of cases, but are
    not guaranteed byte-identical, so this is a documented majority vote, never an assumed
    exact-set equality (CLAUDE.md Golden Rule 5: traceable, not silently assumed).
    """
    mapped = [identity_of[rid] for rid in location_track_ids if rid in identity_of]
    if not mapped:
        return None
    counts: dict[int, int] = {}
    for m in mapped:
        counts[m] = counts.get(m, 0) + 1
    return max(counts, key=lambda k: counts[k])


def attribute_events_to_target(
    events: list[Event],
    take_tracks: list[Track],
    take_balls: list[BallDetection],
    location_track_ids: list[int],
    identity_of: dict[int, int],
    attribution_cfg: dict,
) -> list[Event]:
    """Filter `events` (from `compute_take_all_events`, covering ALL players) down to the ones
    belonging to ONE verified target, identified by its location-selection `location_track_ids`
    (raw ids).

    Handles the THREE different `player_track_id` id-spaces this project's event modules use: a
    raw `Track.id` (sprint; save's goalkeeper), a `build_take_identities` identity id (touch,
    possession, pass, dribble, tackle), or `None` (shot — spatial-proximity attributed, reusing
    `src.highlights.ranking.attribute_shot` exactly as the original Phase-1 Stage 6 flow does).
    """
    target_id = target_identity_id(location_track_ids, identity_of)
    belongs_ids = set(location_track_ids)
    if target_id is not None:
        belongs_ids.add(target_id)

    location_tracks = [tr for tr in take_tracks if tr.id in set(location_track_ids)]
    attributed: list[Event] = []
    for ev in events:
        if ev.player_track_id is not None:
            if ev.player_track_id in belongs_ids:
                attributed.append(ev)
            continue
        if ev.type == EventType.SHOT and attribute_shot(
            ev, location_tracks, take_balls, attribution_cfg
        ):
            attributed.append(ev)
    return attributed
