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

import numpy as np

from src.common.logging import DropCounter
from src.common.types import BallDetection, Event, EventType, Take, Track
from src.events.ball_track import build_ball_state_series, reference_player_height
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
    camera_motion=None,
    kit_lab_by_track_id: dict[int, np.ndarray] | None = None,
) -> tuple[list[Event], dict[int, int], dict[int, float]]:
    """Every best-effort event category for ONE take, across ALL of that take's own tracks.

    Returns `(events, identity_of, identity_confidence)` — the `build_take_identities` partition
    is returned alongside so a caller can attribute a subset to one verified target
    (`attribute_events_to_target` below) without recomputing identities a second time.

    `camera_motion` (`src.track.camera_motion.TakeCameraMotion`, optional): when supplied, the
    ball trajectory state built here for touch detection is expressed in the take's own reference
    frame rather than raw image space (2026-09-02 -- a fast pan otherwise inflates the ball's
    apparent velocity the same way it inflated fragment-stitching jumps, measured earlier the same
    day). `None` degrades to the raw, uncompensated series -- never a hard requirement, since not
    every caller has a camera-motion model available yet. **Also threaded into
    `build_take_identities` (2026-09-03)** -- the exact same argument already reaches
    `build_ball_state_series` above, so passing it on costs nothing new here; without it the
    whole-take identity partition this function's own event heuristics (touch/possession/pass/
    tackle) key on would silently fall back to the raw, pan-inflated image-space distance during a
    fast camera pan, the same failure mode `src/track/camera_motion.py`'s module docstring
    measures for `stitch_timeline`.

    `kit_lab_by_track_id` ("streamed-gathering-treehouse" plan Stage 3, `src.track.kit_wiring.
    build_take_kit_colour`): per-track grass-suppressed kit Lab for THIS take, passed straight
    through to `detect_passes`'s own `teammates_test` (ADR-20) so pass/turnover attribution
    actually uses real measured colour instead of the ADR-12 team-cluster fallback alone. `None`
    (the default) reproduces the exact pre-existing cluster-only behaviour for any caller that
    hasn't wired this up yet.
    """
    identity_of, identity_confidence = build_take_identities(
        take_tracks, selection_cfg, motion=camera_motion
    )

    ball_states = build_ball_state_series(
        take_balls,
        events_cfg["ball"],
        reference_player_height(take_tracks),
        motion=camera_motion,
    )

    events: list[Event] = []
    for tr in take_tracks:
        events.extend(detect_sprints(tr, events_cfg, fps_sample, drops))
    events.extend(detect_shots(take_balls, take.id, frame_width, events_cfg, drops))
    events.extend(
        detect_touches(
            take_balls, take_tracks, take.id, events_cfg, ball_states, identity_of, drops
        )
    )

    runs = possession_runs_for_take(take_balls, take_tracks, events_cfg, identity_of)
    events.extend(
        detect_possession(runs, take.id, events_cfg, ball_fps, identity_confidence, drops)
    )
    events.extend(
        detect_passes(
            runs,
            take_tracks,
            take.id,
            events_cfg,
            ball_fps,
            identity_confidence,
            drops,
            kit_lab_by_track_id=kit_lab_by_track_id,
            # 2026-09-19: the passer must have actually played the ball (kick evidence from the
            # same series `detect_touches` uses) -- see `detect_passes`' own docstring.
            ball_states=ball_states,
        )
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
    # ADR-20: a possession change can independently qualify as BOTH a TACKLE (detect_tackles,
    # above) and a TURNOVER (detect_passes' own opposing-colour branch) since they share the same
    # possession-run adjacency for this take — suppress the redundant TURNOVER so one real
    # dispossession is never double-counted on two different players' stat cards.
    events = suppress_turnovers_covered_by_tackles(events, events_cfg["turnover"], drops)
    return events, identity_of, identity_confidence


def suppress_turnovers_covered_by_tackles(
    events: list[Event], turnover_cfg: dict, drops: DropCounter | None = None
) -> list[Event]:
    """ADR-20: drop any `TURNOVER` whose possession-change transition is ALREADY covered by a
    `TACKLE` elsewhere in `events` — otherwise one real dispossession double-counts as both a
    Tackle (on the tackler's own stat card) and a Turnover (on the dispossessed player's), when
    both `src/events/tackles.py::detect_tackles` and `src/events/possession.py::detect_passes`
    fired on the exact same possession-run adjacency (they share the identical
    `possession_runs_for_take` output for one take).

    A `TURNOVER` is suppressed when some `TACKLE` in `events` has the IDENTICAL raw track-id pair
    (`tackler_raw_track_id` == the turnover's own `receiving_raw_track_id`, `tackled_raw_track_id`
    == the turnover's own `losing_raw_track_id`) whose own `t_end` sits within
    `turnover_cfg['suppress_if_tackle_within_s']` seconds of the turnover's `t_end` — see that
    config key's own comment for why the two are expected to land at (near-)identical timestamps
    in the common case. Every suppression is logged via `drops`, never silently vanished (Golden
    Rule 5 / CLAUDE.md §10).
    """
    window = turnover_cfg["suppress_if_tackle_within_s"]
    tackles = [ev for ev in events if ev.type == EventType.TACKLE]

    kept: list[Event] = []
    for ev in events:
        if ev.type != EventType.TURNOVER:
            kept.append(ev)
            continue
        losing_raw = ev.evidence.get("losing_raw_track_id")
        receiving_raw = ev.evidence.get("receiving_raw_track_id")
        covered = any(
            tk.evidence.get("tackler_raw_track_id") == receiving_raw
            and tk.evidence.get("tackled_raw_track_id") == losing_raw
            and abs(tk.t_end - ev.t_end) <= window
            for tk in tackles
        )
        if covered:
            if drops is not None:
                drops.drop("turnover_suppressed_by_tackle")
            continue
        kept.append(ev)
    return kept


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


def acting_raw_track_ids(ev: Event) -> set[int] | None:
    """The RAW track id(s) of the player an event is credited to, read from the evidence every
    identity-space event module already records (pass -> `passer_raw_track_id`, turnover ->
    `losing_raw_track_id`, touch -> `raw_track_id`, tackle -> `tackler_raw_track_id`, possession/
    dribble -> `raw_track_ids`). `None` when the event carries no such evidence (sprint/save use a
    raw id AS `player_track_id`; shot has no player at all)."""
    ev_evidence = ev.evidence or {}
    if ev.type == EventType.PASS:
        key = "passer_raw_track_id"
    elif ev.type == EventType.TURNOVER:
        key = "losing_raw_track_id"
    elif ev.type == EventType.TOUCH:
        key = "raw_track_id"
    elif ev.type == EventType.TACKLE:
        key = "tackler_raw_track_id"
    elif ev.type in (EventType.POSSESSION, EventType.DRIBBLE):
        ids = ev_evidence.get("raw_track_ids")
        return set(ids) if ids else None
    else:
        return None
    raw = ev_evidence.get(key)
    return {raw} if raw is not None else None


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

    2026-09-19 (owner: "not counting the passes correctly"), a real attribution hole: the
    identity-space events used to be matched against ONE identity -- `target_identity_id`'s
    majority vote. `build_take_identities` seeds its chains earliest-first (its own docstring), so
    it regularly claims one of the target's later fragments into some other chain, or splits the
    target across two chains; every pass/touch the target made while tracked under the minority
    chain was then silently dropped from the stat card. Now:

    * an event whose evidence names the acting RAW track id (`acting_raw_track_ids`) is judged on
      that id alone -- `location_track_ids` is the human-verified, hard-signal-checked answer to
      "which raw fragments are the target" (`src.track.target_verify`), which is strictly more
      trustworthy than either partition's greedy stitching;
    * an event with only an identity id belongs to the target if that identity is one that ANY
      verified location fragment maps to (every such chain, not just the majority one).
    """
    location_set = set(location_track_ids)
    belongs_ids = set(location_set)
    belongs_ids.update(identity_of[rid] for rid in location_track_ids if rid in identity_of)

    location_tracks = [tr for tr in take_tracks if tr.id in location_set]
    attributed: list[Event] = []
    for ev in events:
        if ev.player_track_id is not None:
            raw_ids = acting_raw_track_ids(ev)
            if raw_ids is not None:
                if raw_ids & location_set:
                    attributed.append(ev)
            elif ev.player_track_id in belongs_ids:
                attributed.append(ev)
            continue
        if ev.type == EventType.SHOT and attribute_shot(
            ev, location_tracks, take_balls, attribution_cfg
        ):
            attributed.append(ev)
    return attributed
