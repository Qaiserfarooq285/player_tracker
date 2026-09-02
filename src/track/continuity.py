"""Generalised within-take fragment stitching (ADR-13/14; CLAUDE.md §13.1's continuity note).

`src/highlights/selection.py::stitch_timeline` already does this for exactly ONE arrow-selected
(or fallback-seeded) target track per take. The best-effort event heuristics added alongside this
module (`src/events/{touches,possession,tackles}.py`) need the SAME "is this brief gap really a
new person, or the same one reappearing" judgement for EVERY player in a take, not just the
target -- a possession/pass/tackle sequence has no stable identities to reason about across a
brief occlusion otherwise (Golden Rule 3 still holds: nothing here ever crosses a take boundary).

Rather than duplicating `stitch_timeline`'s gap/distance/team-tiebreak logic, the core greedy
forward-extension loop is extracted once (`extend_chain_forward`) and used by BOTH:

- `stitch_timeline` (this module) -- the exact same public signature/behaviour
  `src/highlights/selection.py` used to implement inline; that module now imports it from here
  (a behaviour-preserving refactor, not a change to its own tests/API).
- `build_take_identities` -- the new general case: partitions an ENTIRE take's track fragments
  into stitched "identity" chains by repeatedly seeding a fresh chain from whichever fragment is
  earliest-starting among those not yet claimed, and greedily extending it exactly as
  `stitch_timeline` does for its one externally-chosen seed.

Both read the SAME `configs/highlights.yaml: selection` knobs
(`stitch_max_gap_s`/`stitch_max_dist`/`team_match_confidence_threshold`/`join_retention_factor`) --
CLAUDE.md task spec: a fragment reappearing after an occlusion is the same physical phenomenon
whether or not that fragment happens to be the manually-selected target, so a second, possibly
drifting set of thresholds would be actively wrong, not just redundant. No new config knobs were
needed for the general case (see `build_take_identities` docstring for the one new, threshold-free
design decision it does add: a deterministic seed order).
"""

from __future__ import annotations

from src.common.logging import get_logger
from src.common.types import Track, TrackBox

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# shared geometry (moved from src/highlights/selection.py -- ADR-13/14 refactor)
# ---------------------------------------------------------------------------


def _spatial_jump(a: Track, b_start_box: TrackBox, motion=None) -> float:
    """Normalised distance (bbox-heights, same unit as `configs/events.yaml: sprint`) between
    track `a`'s LAST box and `b_start_box` (candidate fragment's first box).

    When a `TakeCameraMotion` is supplied, both centres are first mapped into the take's own
    reference frame so the result measures how far the PLAYER moved, not how far the CAMERA
    swung (`src/track/camera_motion.py` -- see its module docstring for the measured 14x
    inflation this removes during a real pan). Without one, this is the original raw
    image-space distance, unchanged.
    """
    from src.track.camera_motion import compensated_jump

    a_box = a.boxes[-1]
    mean_h = (a_box.bbox.height + b_start_box.bbox.height) / 2.0
    jump, _compensated = compensated_jump(
        a_box.t,
        a_box.bbox.cx,
        a_box.bbox.cy,
        b_start_box.t,
        b_start_box.bbox.cx,
        b_start_box.bbox.cy,
        mean_h,
        motion,
    )
    return jump


def _team_agrees(a: Track, b: Track, threshold: float) -> bool | None:
    """`True`/`False` when BOTH tracks' `team_confidence` clears `threshold` (a real signal to
    tiebreak on); `None` when it can't be judged (ADR-12: team is a SOFT signal only, never a
    hard filter) -- see `configs/highlights.yaml: selection.team_match_confidence_threshold`."""
    if a.team_confidence < threshold or b.team_confidence < threshold:
        return None
    if a.team is None or b.team is None:
        return None
    return a.team == b.team


# ---------------------------------------------------------------------------
# core greedy forward-extension (the shared, no-longer-duplicated logic)
# ---------------------------------------------------------------------------


def extend_chain_forward(
    seed_track_id: int,
    pool: dict[int, Track],
    used: set[int],
    stitch_cfg: dict,
    motion=None,
) -> list[Track]:
    """Greedily extend `seed_track_id` forward in time within `pool`, consuming any fragment not
    already in `used` that satisfies the join rule -- mutates `used` IN PLACE (adding every
    fragment consumed, including the seed itself) so a caller can build several independent chains
    from the same `pool` without two chains ever claiming the same fragment.

    Join rule, identical to the original `src.highlights.selection.stitch_timeline`: a candidate
    is eligible when (a) the time gap between the chain's current last box and the candidate's
    first box is under `stitch_max_gap_s`, AND (b) the spatial jump between those two boxes
    (bbox-heights, `_spatial_jump`) is under `stitch_max_dist`. Among all eligible candidates the
    one with the SMALLEST spatial jump wins; team agreement (`_team_agrees`) only breaks an exact
    tie -- never a hard filter (ADR-12).

    Returns `[]` when `seed_track_id` isn't in `pool` or is already `used` (nothing to extend);
    otherwise returns the ordered chain (seed first) as `Track` objects, not bare ids, so callers
    can compute their own confidence/evidence from the full objects without a second lookup.
    """
    if seed_track_id not in pool or seed_track_id in used:
        return []

    max_gap_s = stitch_cfg["stitch_max_gap_s"]
    max_dist = stitch_cfg["stitch_max_dist"]
    # Optional (absent -> gate disabled, preserving the exact prior behaviour for any caller whose
    # config predates this key) -- see the gate's own comment in the candidate loop below.
    max_speed = stitch_cfg.get("stitch_max_speed_bbox_heights_per_s")
    team_threshold = stitch_cfg["team_match_confidence_threshold"]

    chain: list[Track] = [pool[seed_track_id]]
    used.add(seed_track_id)

    while True:
        current = chain[-1]
        if not current.boxes:
            break
        current_end_t = current.boxes[-1].t
        best: Track | None = None
        best_dist: float | None = None
        best_agrees: bool | None = None
        for cand_id, cand in pool.items():
            if cand_id in used or not cand.boxes:
                continue
            gap = cand.boxes[0].t - current_end_t
            if gap < 0 or gap >= max_gap_s:
                continue
            dist = _spatial_jump(current, cand.boxes[0], motion)
            if dist >= max_dist:
                continue
            # Physical-plausibility gate (2026-09-01). `max_dist` alone is gap-BLIND: it allows the
            # same 3.0-bbox-height jump whether the fragment reappears 0.04s or 1.4s later, so a
            # short-gap join can imply a speed no human reaches. Measured failure it fixes: on
            # `chelsea_burnley_target10` take 0 the clicked Chelsea #10 (track 1) was joined to a
            # CLARET BURNLEY #21 (track 47) across a 0.76s gap and 2.34 bbox-heights -- i.e.
            # 3.08 bbox-heights/SECOND. This project's own measured speed distribution
            # (configs/events.yaml: sprint_speed_threshold's comment, 1977+17885 real samples on
            # two clips) puts p95 at 2.10-2.31 and the observed MAXIMUM at 3.70, so 3.08 is
            # faster than 95% of all real player movement ever measured here -- implausible for
            # one continuous player, and exactly the signature of a jump to a different person.
            if max_speed is not None and gap > 0 and (dist / gap) >= max_speed:
                continue
            agrees = _team_agrees(current, cand, team_threshold)
            if (
                best is None
                or dist < best_dist
                or (dist == best_dist and agrees and not best_agrees)
            ):
                best, best_dist, best_agrees = cand, dist, agrees
        if best is None:
            break
        chain.append(best)
        used.add(best.id)

    return chain


def chain_confidence(chain: list[Track], stitch_cfg: dict) -> float:
    """`join_retention_factor ** n_joins * quality` -- the exact formula
    `src.highlights.selection.stitch_timeline` used inline, factored out once so
    `build_take_identities` scores every chain (not just the one externally-chosen seed's) the
    same way. `quality` is the mean, across the chain's own fragments, of each fragment's own mean
    detection confidence. More joins => lower confidence (each join is itself an inference on top
    of the raw tracker output, never ground truth)."""
    join_retention = stitch_cfg["join_retention_factor"]
    n_joins = len(chain) - 1
    mean_track_confs = [
        (sum(b.conf for b in tr.boxes) / len(tr.boxes)) if tr.boxes else 0.0 for tr in chain
    ]
    quality = sum(mean_track_confs) / len(mean_track_confs) if mean_track_confs else 0.0
    return (join_retention**n_joins) * quality


# ---------------------------------------------------------------------------
# 1. single-seed stitching (the original `selection.py` API, now just a thin wrapper)
# ---------------------------------------------------------------------------


def stitch_timeline(
    seed_track_id: int,
    take_tracks: list[Track],
    selection_cfg: dict,
    motion=None,
) -> tuple[list[int], float]:
    """Extend `seed_track_id` forward within `take_tracks` (all belonging to the SAME take -- the
    caller guarantees this, and it is also asserted defensively below).

    This is `src.highlights.selection`'s original public function, moved here verbatim in
    BEHAVIOUR (see the module docstring) so both the target-specific and general-purpose stitching
    read one shared implementation. `src.highlights.selection.stitch_timeline` now just re-exports
    this function -- its existing callers/tests are unaffected.

    Returns `(ordered_track_ids, id_confidence)`, seed first.
    """
    by_id = {tr.id: tr for tr in take_tracks}
    if seed_track_id not in by_id:
        return [], 0.0
    take_ids = {tr.take_id for tr in take_tracks}
    assert len(take_ids) <= 1, "stitch_timeline must only ever see one take's tracks"

    used: set[int] = set()
    chain = extend_chain_forward(seed_track_id, by_id, used, selection_cfg, motion)
    confidence = chain_confidence(chain, selection_cfg)

    logger.info(
        "stitched timeline seed=%d take=%s: %d fragment(s) joined (%s), id_confidence=%.3f",
        seed_track_id,
        next(iter(take_ids), None),
        len(chain),
        [tr.id for tr in chain],
        confidence,
    )
    return [tr.id for tr in chain], confidence


# ---------------------------------------------------------------------------
# 2. generalised, whole-take identity partitioning (the new capability)
# ---------------------------------------------------------------------------


def build_take_identities(
    take_tracks: list[Track], selection_cfg: dict
) -> tuple[dict[int, int], dict[int, float]]:
    """Partition an ENTIRE take's track fragments into stitched "identity" chains, using exactly
    the same greedy forward-extension rule as `stitch_timeline` (`extend_chain_forward`) instead of
    extending just one externally-chosen seed.

    **Seed order** (the one new design decision this function adds, deliberately threshold-free so
    it can't drift from `configs/highlights.yaml`): process fragments in ascending `t_start` order
    (ties broken by `id`, for determinism). The earliest not-yet-claimed fragment always seeds the
    next chain -- since `extend_chain_forward` only ever looks FORWARD in time, a fragment that
    starts before all still-unclaimed fragments can never be a valid forward-extension of anything
    that comes later, so seeding earliest-first guarantees every fragment ends up in exactly one
    chain (never claimed twice, never left out) without needing a second pass.

    A track with zero boxes (should not occur in practice -- see `Track.dominant_class` docstring)
    is defensively given its own singleton, zero-confidence identity rather than crashing on
    `chain[-1]` inside the extension loop.

    Returns `(identity_of, identity_confidence)`:
      - `identity_of`: `{raw_track_id: identity_id}` for every fragment in `take_tracks`. The
        canonical `identity_id` for a chain is simply the `id` of its FIRST (earliest-starting)
        fragment -- identities live in the same integer space as raw track ids, so no new id
        allocator/namespace is introduced.
      - `identity_confidence`: `{identity_id: confidence}`, one entry per chain, via
        `chain_confidence` (the exact same formula `stitch_timeline` uses).

    Never crosses a take boundary: `take_tracks` must all share one `take_id` (asserted, same
    invariant as `stitch_timeline`) -- see `tests/test_continuity.py::
    test_build_take_identities_rejects_mixed_take_ids` and the dedicated cross-take rejection test.
    """
    if not take_tracks:
        return {}, {}
    take_ids = {tr.take_id for tr in take_tracks}
    assert len(take_ids) <= 1, "build_take_identities must only ever see one take's tracks"

    pool = {tr.id: tr for tr in take_tracks}
    ordered = sorted(take_tracks, key=lambda tr: (tr.t_start, tr.id))

    used: set[int] = set()
    identity_of: dict[int, int] = {}
    identity_confidence: dict[int, float] = {}

    for tr in ordered:
        if tr.id in used:
            continue
        if not tr.boxes:
            # defensive: extend_chain_forward would break on chain[-1].boxes[-1] for a genuinely
            # empty track; give it its own zero-confidence singleton identity instead of crashing.
            identity_of[tr.id] = tr.id
            identity_confidence[tr.id] = 0.0
            used.add(tr.id)
            continue
        chain = extend_chain_forward(tr.id, pool, used, selection_cfg)
        if not chain:
            continue
        identity_id = chain[0].id
        confidence = chain_confidence(chain, selection_cfg)
        for member in chain:
            identity_of[member.id] = identity_id
        identity_confidence[identity_id] = confidence

    logger.info(
        "built %d identity chain(s) from %d fragment(s) in take=%s",
        len(identity_confidence),
        len(take_tracks),
        next(iter(take_ids), None),
    )
    return identity_of, identity_confidence
