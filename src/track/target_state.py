"""Stage A of the "streamed-gathering-treehouse" plan (per-frame target identity gate, see
`/home/qaiserfarooq/.claude/plans/streamed-gathering-treehouse.md`): a per-TRACKING-SAMPLE state
machine for the persistent target (`src.track.target.TargetProfile`) within ONE take.

**Why this exists.** `src.track.target_verify.verify_candidate` (Stage 2) and
`src.highlights.selection._select_take_with_profile` (Stage 4) already decide identity ONCE PER
TAKE -- which raw track fragment(s), if any, are the verified target for the whole take. That
decision is not enough on its own: the render loop (`src.pipeline.annotated_video`) draws a box
once per NATIVE FRAME, and between two real tracking samples of the accepted fragment(s) there can
be a multi-second stretch (an occlusion, a lost/re-acquired track, a camera cut) where NOTHING
actually observed the target. The three holes this closes (plan Context section, read from the
code, not assumed):

- **Hole A (stale box):** the render loop's own `interpolated_bbox` falls back to a *nearest*
  sample up to 0.3s away when no real pair brackets the query time -- a real position, but a STALE
  one, drawn as if it were current.
- **Hole B (silent ID switch):** a lost ByteTrack id can be re-associated onto a DIFFERENT physical
  player when the occluder clears; the raw id is still in the take's own "target" id set, so the
  red box would silently follow the wrong player.
- **Hole C (whole-span active window):** `src.pipeline.run._track_active_windows` marks a fragment
  "active" for its ENTIRE `[first_box.t, last_box.t]` span, so an internal gap (exactly what an
  occlusion looks like) still renders as if the target were continuously visible throughout it.

`build_target_timeline` resolves all three, once per take, into a plain list of `TargetFrameStatus`
-- one entry per REAL tracking-sample instant across the take (the union of every track's own box
timestamps in this take, since ByteTrack samples every track at the same fixed cadence). The
render loop then does a cheap nearest-time lookup against this precomputed list instead of
re-deriving identity per rendered frame (see `src.pipeline.annotated_video`'s own wiring).

**Absolute rule (owner's own words, plan's non-negotiable):** *"When choosing between (A)
temporarily showing NO TARGET BOX or (B) showing the TARGET BOX ON A POSSIBLY WRONG PLAYER, ALWAYS
CHOOSE (A)."* `TargetFrameStatus.bbox`/`tracker_id` are BOTH `None` in every state except
`VISIBLE`/`PARTIALLY_OCCLUDED`/`RECONNECTED` -- there is no code path that can populate a bbox
without a genuine bracketing (or exact) real sample from an ACCEPTED fragment. Asserted directly in
`tests/test_target_state.py`.

**Candidate != Target (closes hole B).** `accepted_fragments` is NOT `TakeSelection.track_ids` (the
full geometry-stitched chain) -- it is the SUBSET of that chain that is either the human-established
seed (a click / `--track-id`) or independently passed `src.track.target_verify.verify_candidate`
(`TakeSelection.verified_track_ids`, populated by
`src.highlights.selection._select_take_with_profile`).
A fragment `stitch_timeline` joined purely by geometric continuity, never itself verified, is
therefore invisible to this module entirely -- it is drawn as an ordinary green "other player" box
by the renderer, never red/amber, exactly like every other candidate.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from enum import Enum

from pydantic import BaseModel

from src.common.types import BBox, Take, Track


class TargetFrameState(str, Enum):  # noqa: UP042 -- matches src/common/types.py's own enum style
    """One tracking-sample instant's own resolved target state (plan Stage A)."""

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    OCCLUDED = "occluded"
    LOST = "lost"
    SEARCHING = "searching"
    RECONNECTED = "reconnected"


class TargetFrameStatus(BaseModel):
    """One tracking-sample instant's resolved target state -- the atomic unit
    `build_target_timeline` produces and the renderer looks up per rendered frame.

    `bbox`/`tracker_id` are `None` in every state except `VISIBLE`/`PARTIALLY_OCCLUDED`/
    `RECONNECTED` -- the plan's own absolute rule, never violated (Golden Rule 5: an honest "no
    box" beats a guessed one). `reason` is always populated, win or lose, so a `LOST`/`OCCLUDED`
    verdict is exactly as traceable as a `VISIBLE` one (Golden Rule 5)."""

    t: float
    state: TargetFrameState
    tracker_id: int | None
    bbox: BBox | None
    identity_confidence: float
    reason: str


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Standard IoU -- deliberately re-implemented rather than importing `src.goal.detect._iou`
    (module-private there) or `src.highlights.cutting.time_iou` (a different, TIME-interval IoU):
    this project's own established precedent (`src.pipeline.run._effective_frame_size`'s own
    docstring) is a small, self-contained helper duplicated per module over a cross-module import
    of another module's private helper."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _max_overlap(other_boxes: list[tuple[int, BBox]], ref_bbox: BBox) -> tuple[int | None, float]:
    """Whichever of `other_boxes` (already narrowed to THIS exact instant -- see
    `build_target_timeline`'s own `other_boxes_by_t` grouping) overlaps `ref_bbox` the most, and by
    how much. `(None, 0.0)` when `other_boxes` is empty. This is the "cheap enumeration of other
    tracks" the plan's own task spec flags as a judgement call: rather than testing every track's
    full box list against every grid instant (O(n_tracks * n_grid_points)), every track's boxes are
    grouped by EXACT timestamp ONCE up front, so this is a dict lookup (O(1) amortised) plus a scan
    of however many players/refs genuinely have a box at that one instant (bounded by the number of
    people on screen, never the whole track list)."""
    best_id: int | None = None
    best_iou = 0.0
    for track_id, bbox in other_boxes:
        iou = _bbox_iou(bbox, ref_bbox)
        if iou > best_iou:
            best_id, best_iou = track_id, iou
    return best_id, best_iou


def _strict_target_bbox_at(
    accepted_samples: list[tuple[float, BBox, int]],
    accepted_ts: list[float],
    t: float,
    max_gap_s: float,
) -> tuple[BBox | None, int | None]:
    """The accepted-fragment target bbox at `t`, from an EXACT sample or from two REAL bracketing
    samples no more than `max_gap_s` apart -- never a nearest-sample fallback (closes hole A for
    the target specifically; mirrors `src.pipeline.annotated_video.interpolated_bbox`'s own
    linear-interpolation arithmetic, but with the nearest-sample escape hatch removed on purpose --
    that escape hatch is exactly what produces a stale box during a real gap).

    `accepted_samples`/`accepted_ts` are parallel, time-sorted (see `build_target_timeline`) --
    `accepted_ts` exists purely so `bisect_left` doesn't need a `key=` (pure-Python bisect has no
    such parameter). Returns `(None, None)` whenever `t` falls before the first sample, after the
    last, or between two samples more than `max_gap_s` apart -- every one of those is "no real
    evidence right now", not "extrapolate from what we last saw".
    """
    if not accepted_samples:
        return None, None
    pos = bisect_left(accepted_ts, t)
    if pos < len(accepted_ts) and accepted_ts[pos] == t:
        _, bbox, tracker_id = accepted_samples[pos]
        return bbox, tracker_id
    before = accepted_samples[pos - 1] if pos - 1 >= 0 else None
    after = accepted_samples[pos] if pos < len(accepted_samples) else None
    if before is None or after is None:
        return None, None
    t0, b0, _tid0 = before
    t1, b1, tid1 = after
    gap = t1 - t0
    if gap <= 0 or gap > max_gap_s:
        return None, None
    frac = (t - t0) / gap
    bbox = BBox(
        x1=b0.x1 + (b1.x1 - b0.x1) * frac,
        y1=b0.y1 + (b1.y1 - b0.y1) * frac,
        x2=b0.x2 + (b1.x2 - b0.x2) * frac,
        y2=b0.y2 + (b1.y2 - b0.y2) * frac,
    )
    # The AFTER sample's own id -- an interpolated instant sits between two real samples, and once
    # `t` reaches an after-sample's own fragment, attributing the box to that (more recent, still
    # live) fragment is the more useful "who is this" answer than the fragment we're leaving.
    return bbox, tid1


def build_target_timeline(
    take: Take,
    accepted_fragments: list[int],
    take_tracks: list[Track],
    cfg: dict,
) -> list[TargetFrameStatus]:
    """One `TargetFrameStatus` per real tracking-sample instant across `take` (the union of every
    track's own box timestamps in this take -- ByteTrack samples every track at the SAME fixed
    cadence, so this is the take's own real tracking-sample grid, not an assumed uniform one).

    `accepted_fragments` is `TakeSelection.verified_track_ids` (plan Stage B) -- the human-
    established seed and/or whichever candidates independently passed `verify_candidate` this take.
    Empty (a `target_lost` take, or a take this plan's machinery never resolved at all) yields a
    timeline of nothing but `LOST`/`SEARCHING` entries -- `bbox` is `None` throughout, by
    construction (there is no accepted sample for `_strict_target_bbox_at` to ever find).

    Per-instant resolution, in order:
    1. **Real observation only** (closes hole A) -- `_strict_target_bbox_at` on the accepted
       fragments' own stitched box sequence; `None` unless an exact or tightly-bracketed
       (`cfg['max_sample_age_s']`) real sample exists.
    2. A real bbox found: `RECONNECTED` if this is the first real bbox after a gap that reached
       `SEARCHING`, held for `cfg['reconnect_hold_s']`; else `PARTIALLY_OCCLUDED` if another
       track's box overlaps it at/above `cfg['partial_occlusion_iou']` (box still drawn -- the
       plan allows this while a real accepted sample backs it); else plain `VISIBLE`.
    3. No real bbox: `OCCLUDED` if another track's box overlaps the target's LAST known position
       at/above `cfg['occlusion_iou']`; otherwise `LOST`, escalating to `SEARCHING` once the
       current gap has lasted >= `cfg['search_grace_s']`. No prior sighting at all this take (gap
       duration `inf`) always lands in `LOST`/`SEARCHING`, never `OCCLUDED` -- there is no "last
       known position" to test an overlap against yet.

    `identity_confidence` is deliberately a coarse binary signal (`1.0` whenever `bbox` is not
    `None`, `0.0` otherwise) -- the plan does not specify a finer-grained per-frame confidence
    model, and a manufactured one would just be a second, undocumented threshold; `reason` is what
    actually carries the traceable detail (Golden Rule 5).
    """
    accepted_ids = set(accepted_fragments)

    accepted_samples: list[tuple[float, BBox, int]] = sorted(
        (
            (box.t, box.bbox, tr.id)
            for tr in take_tracks
            if tr.id in accepted_ids
            for box in tr.boxes
        ),
        key=lambda item: item[0],
    )
    accepted_ts = [item[0] for item in accepted_samples]

    other_boxes_by_t: dict[float, list[tuple[int, BBox]]] = defaultdict(list)
    for tr in take_tracks:
        if tr.id in accepted_ids:
            continue
        for box in tr.boxes:
            other_boxes_by_t[box.t].append((tr.id, box.bbox))

    grid = sorted({box.t for tr in take_tracks for box in tr.boxes})
    if not grid:
        return []

    max_gap_s = cfg["max_sample_age_s"]
    occlusion_iou = cfg["occlusion_iou"]
    partial_occlusion_iou = cfg["partial_occlusion_iou"]
    search_grace_s = cfg["search_grace_s"]
    reconnect_hold_s = cfg["reconnect_hold_s"]

    statuses: list[TargetFrameStatus] = []
    last_seen_t: float | None = None
    last_seen_bbox: BBox | None = None
    gap_reached_searching = False
    reconnect_deadline: float | None = None

    for t in grid:
        bbox, tracker_id = _strict_target_bbox_at(accepted_samples, accepted_ts, t, max_gap_s)

        if bbox is not None:
            if gap_reached_searching:
                # First real observation after a gap that escalated to SEARCHING -- start a
                # RECONNECTED window (plan's own "for the first reconnect_hold_s after a gap >=
                # search_grace_s closes" rule), then clear the flag so we don't re-trigger it on
                # every subsequent real sample of THIS same reconnection.
                reconnect_deadline = t + reconnect_hold_s
                gap_reached_searching = False

            if reconnect_deadline is not None and t <= reconnect_deadline:
                state = TargetFrameState.RECONNECTED
                reason = (
                    f"real observation reacquired (tracker={tracker_id}) within "
                    f"reconnect_hold_s={reconnect_hold_s}s of a gap that had reached SEARCHING"
                )
            else:
                reconnect_deadline = None
                overlap_id, overlap_iou = _max_overlap(other_boxes_by_t.get(t, []), bbox)
                if overlap_iou >= partial_occlusion_iou:
                    state = TargetFrameState.PARTIALLY_OCCLUDED
                    reason = (
                        f"real sample from tracker={tracker_id}, but track={overlap_id} overlaps "
                        f"it at iou={overlap_iou:.2f} (>= partial_occlusion_iou="
                        f"{partial_occlusion_iou})"
                    )
                else:
                    state = TargetFrameState.VISIBLE
                    reason = f"real tracking sample from accepted fragment track={tracker_id}"

            statuses.append(
                TargetFrameStatus(
                    t=t,
                    state=state,
                    tracker_id=tracker_id,
                    bbox=bbox,
                    identity_confidence=1.0,
                    reason=reason,
                )
            )
            last_seen_t, last_seen_bbox = t, bbox
            continue

        gap_duration = (t - last_seen_t) if last_seen_t is not None else float("inf")
        overlap_id, overlap_iou = (
            _max_overlap(other_boxes_by_t.get(t, []), last_seen_bbox)
            if last_seen_bbox is not None
            else (None, 0.0)
        )
        if overlap_iou >= occlusion_iou:
            state = TargetFrameState.OCCLUDED
            reason = (
                f"no accepted-fragment sample at t={t:.3f}s, but track={overlap_id} overlaps the "
                f"target's last known position at iou={overlap_iou:.2f} "
                f"(>= occlusion_iou={occlusion_iou})"
            )
        elif gap_duration >= search_grace_s:
            state = TargetFrameState.SEARCHING
            gap_reached_searching = True
            reason = (
                f"no accepted-fragment sample and no occluding track for "
                f"{gap_duration:.3f}s >= search_grace_s={search_grace_s}s -- actively searching "
                "for the original player"
            )
        else:
            state = TargetFrameState.LOST
            reason = (
                "target not yet located in this take (no prior sighting)"
                if last_seen_t is None
                else (
                    f"no accepted-fragment sample and no occluding track "
                    f"(gap={gap_duration:.3f}s < search_grace_s={search_grace_s}s)"
                )
            )

        statuses.append(
            TargetFrameStatus(
                t=t,
                state=state,
                tracker_id=None,
                bbox=None,
                identity_confidence=0.0,
                reason=reason,
            )
        )

    return statuses
