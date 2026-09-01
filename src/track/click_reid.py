"""Click-anchored re-identification: extend a human-clicked player's chain across a within-take
ByteTrack fragmentation break, using the clicked player's OWN measured colour/height profile as
the matching signal.

CLAUDE.md Golden Rule 4: "a human-provided [identity] is the strongest evidence we have." A click
already IS the target -- this module's only job is to keep following that SAME physical player
through the raw tracker's own fragmentation, never to re-derive who they are from scratch. It runs
strictly AFTER `src.highlights.selection.stitch_timeline` has already produced its own forward
chain from the clicked seed (`src/pipeline/run.py`'s manual-override branch); it only picks up
fragments that chain missed because they fall outside its own gap/distance thresholds
(`configs/highlights.yaml: selection.stitch_max_gap_s/stitch_max_dist`).

Why this, not a trained ReID model: evidence-checked and rejected earlier in this project (see
CLAUDE.md's ADR log) -- no good permissively-licensed appearance-ReID option exists (`trackers`
dropped its ReID model in v2.1; StrongSORT is GPL-3.0; torchreid's PyPI package is stale with
academically-restricted weights). Team cluster and box height are already computed for every RAW
track by Stage 3 (`src.team.classifier.assign_teams`) and persisted in `tracks.parquet`, so this
needs **no extra video decode and no new dependency** -- it only combines signals the pipeline
already has. Jersey-number corroboration is accepted as an OPTIONAL extra signal when a caller
already has one (e.g. a UI's own advisory read via `src.identity.jersey_parseq`), never required.
"""

from __future__ import annotations

from src.common.types import Track


def _median_height(track: Track) -> float:
    """Median box height across `track`'s own boxes -- robust to a couple of bad detections,
    unlike a mean. `0.0` for an empty track (never divides by zero downstream)."""
    heights = sorted(b.bbox.height for b in track.boxes)
    if not heights:
        return 0.0
    mid = len(heights) // 2
    return heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0


def build_click_profile(anchor_track: Track) -> dict:
    """Summarise the clicked player's own measured signals -- team cluster and median box height
    -- for `candidate_score` to compare every OTHER same-take fragment against."""
    return {
        "team": anchor_track.team,
        "median_height": _median_height(anchor_track),
    }


def candidate_score(
    candidate: Track,
    profile: dict,
    reid_cfg: dict,
    jersey_by_track_id: dict[int, str] | None = None,
    anchor_track_id: int | None = None,
) -> tuple[bool, dict]:
    """`(accept, evidence)` -- whether `candidate` is plausibly the SAME physical player as the
    clicked anchor behind `profile`, and the full evidence trail behind that decision (Golden Rule
    5: every join or rejection is explainable, never a silent guess).

    - **Team cluster**: `None` on either side means Stage 3's team assignment didn't run or wasn't
      confident for that take -- treated as "no evidence either way" (`team_match: None`), not a
      rejection (a low-confidence team call is common on crowded footage, ADR-12). A confident
      MISMATCH is a hard reject.
    - **Height**: compared as a RATIO (`min/max`), never an absolute pixel difference -- a player
      further from camera reads smaller, which is not evidence of being a different person.
    - **Jersey (optional)**: only consulted when the caller supplies confident reads for BOTH the
      anchor and the candidate (`jersey_by_track_id`, already filtered by the caller to whatever
      confidence bar it trusts -- this function does no thresholding of its own). Agreement is
      strong corroboration; a DISAGREEMENT is a hard veto regardless of colour/height, since a
      printed number is the strongest available disambiguator between two similarly-dressed
      teammates.
    """
    evidence: dict = {"candidate_track_id": candidate.id}

    team_a, team_b = profile.get("team"), candidate.team
    if team_a is not None and team_b is not None:
        team_match = team_a == team_b
        evidence["team_match"] = team_match
        if not team_match:
            evidence["rejected_reason"] = "different_team_cluster"
            return False, evidence
    else:
        evidence["team_match"] = None

    height_a = profile.get("median_height", 0.0)
    height_b = _median_height(candidate)
    if height_a > 0 and height_b > 0:
        ratio = min(height_a, height_b) / max(height_a, height_b)
        evidence["height_ratio"] = round(ratio, 3)
        if ratio < reid_cfg["min_height_ratio"]:
            evidence["rejected_reason"] = "height_mismatch"
            return False, evidence
    else:
        evidence["height_ratio"] = None

    if jersey_by_track_id and anchor_track_id is not None:
        digits_a = jersey_by_track_id.get(anchor_track_id)
        digits_b = jersey_by_track_id.get(candidate.id)
        if digits_a is not None and digits_b is not None:
            evidence["jersey_agrees"] = digits_a == digits_b
            if digits_a != digits_b:
                evidence["rejected_reason"] = "jersey_number_disagrees"
                return False, evidence

    evidence["accepted"] = True
    return True, evidence


def extend_chain_with_profile(
    seed_track_id: int,
    stitched_track_ids: list[int],
    take_tracks: list[Track],
    reid_cfg: dict,
    jersey_by_track_id: dict[int, str] | None = None,
) -> tuple[list[int], list[dict]]:
    """Extend `stitched_track_ids` (already `stitch_timeline`'s own forward chain from the clicked
    seed) with any OTHER same-take fragment that plausibly continues the SAME physical player --
    a break `stitch_timeline` itself didn't bridge (a gap/jump outside its own thresholds) but
    that colour+height (+jersey, if available) evidence says is still the same person.

    Never joins a fragment that overlaps in TIME with an already-accepted one: that would mean two
    different physical people were on screen as "the same track" at once, and picking one would be
    a silent, unjustified merge (Golden Rule 5). Returns the extended id list plus a full evidence
    trail covering every candidate considered -- accepted AND rejected -- for the run report.
    """
    by_id = {tr.id: tr for tr in take_tracks}
    anchor = by_id.get(seed_track_id)
    if anchor is None:
        return stitched_track_ids, []

    profile = build_click_profile(anchor)
    accepted_ids = list(stitched_track_ids)
    accepted_spans = [
        (by_id[tid].t_start, by_id[tid].t_end) for tid in accepted_ids if tid in by_id
    ]
    evidence_trail: list[dict] = []

    remaining = [tr for tr in take_tracks if tr.id not in accepted_ids]
    for candidate in sorted(remaining, key=lambda tr: (tr.t_start, tr.id)):
        overlaps = any(
            candidate.t_start < end and start < candidate.t_end for start, end in accepted_spans
        )
        if overlaps:
            evidence_trail.append(
                {
                    "candidate_track_id": candidate.id,
                    "rejected_reason": "time_overlap_with_accepted",
                }
            )
            continue
        accept, evidence = candidate_score(
            candidate, profile, reid_cfg, jersey_by_track_id, anchor_track_id=seed_track_id
        )
        evidence_trail.append(evidence)
        if accept:
            accepted_ids.append(candidate.id)
            accepted_spans.append((candidate.t_start, candidate.t_end))

    return accepted_ids, evidence_trail
