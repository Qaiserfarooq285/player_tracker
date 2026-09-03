"""Stage 1 of the "streamed-gathering-treehouse" plan (strict persistent target identity, see
`/home/qaiserfarooq/.claude/plans/streamed-gathering-treehouse.md`): the persistent `TargetProfile`
object the rest of the plan hinges on.

**Why this exists.** CLAUDE.md Golden Rule 4 already says a human-provided identity is the
strongest evidence available, but before this module there was NO on-disk representation of "the
target player" at all -- only a per-take list of raw tracker IDs (`src.highlights.selection.
TakeSelection`). A tracker ID resets at every cut and can silently fragment or merge; `target_id`
here is a fixed string (`"TARGET_001"` by default) that never changes for the life of a run, and
everything about WHO that target visually is -- jersey number, kit colour per body band, height,
team cluster -- is captured once, then updated only under a strict anti-drift rule
(`update_memory_bank`) so an uncertain re-identification can never quietly retrain the profile onto
a different physical player (the exact "occlusion -> follows #7 -> learns #7 -> permanently
believes #7 is #11" failure mode the plan's Context section documents).

This module is pure data + arithmetic -- no verification/decision logic (that is Stage 2,
`src/track/target_verify.py`) and no pipeline wiring (later, out-of-scope stages).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from src.common.io import load_json, save_json
from src.common.types import BBox, Track
from src.events.kit_colour import lab_median_from_region, median_kit_lab
from src.track.click_reid import _median_height  # reused verbatim, not copied (plan Stage 1)


class TargetState(str, Enum):  # noqa: UP042 -- (str, Enum) matches src/common/types.py's own
    # convention for every other enum in this codebase.
    """Lifecycle states for the persistent target (plan §9)."""

    ACTIVE = "active"
    CONFIRMED = "confirmed"
    TEMPORARILY_LOST = "temporarily_lost"
    SEARCHING = "searching"
    REIDENTIFYING = "reidentifying"


class KitColourSample(BaseModel):
    """One grass-suppressed kit-colour reading, torso/shorts/socks, from ONE frame (plan §3 items
    2/4/5). Each band is `None` when that band's own grass-suppression floor
    (`configs/target.yaml: kit_colour`) wasn't cleared that frame -- an honest "no evidence for
    this band", never a guess. `confidence` reflects how MUCH of the sample was actually observed
    (see `sample_kit_bands`'s own docstring), not a colour-matching confidence."""

    torso_lab: tuple[float, float, float] | None
    shorts_lab: tuple[float, float, float] | None
    socks_lab: tuple[float, float, float] | None
    take_id: int
    t: float
    confidence: float


class TargetLink(BaseModel):
    """One take's worth of the target's own resolved identity (plan §10's `identity_history`):
    which raw track(s) were judged to be the target, in what state, with what confidence, and the
    full evidence trail behind that judgement (Golden Rule 5 -- always traceable)."""

    take_id: int
    track_ids: list[int]
    state: TargetState
    confidence: float
    evidence: dict = Field(default_factory=dict)
    t_start: float
    t_end: float


class TargetProfile(BaseModel):
    """The persistent target identity for one run (plan §10) -- NEVER a tracker id. Cached to
    `work/<slug>/target.json` via `save_target_profile`/`load_target_profile`."""

    target_id: str = "TARGET_001"  # a fixed string, never a raw tracker id (plan §10)
    jersey_number: int | None = None
    jersey_source: Literal["click", "filename", "human_confirmed", "ocr_verified"] | None = None
    kit: KitColourSample | None = None  # per-band median of `kit_bank`, recomputed by
    # `update_memory_bank`/`build_target_profile`; not a fresh single-frame sample of its own.
    kit_bank: list[KitColourSample] = Field(default_factory=list)
    median_height: float = 0.0
    team_cluster: int | None = None
    established_take_id: int
    established_track_id: int
    established_t: float
    links: list[TargetLink] = Field(default_factory=list)


def sample_kit_bands(
    frame_bgr: np.ndarray, bbox: BBox, take_id: int, t: float, cfg: dict
) -> KitColourSample | None:
    """Grass-suppressed kit-colour sample for `bbox` in `frame_bgr`, split into THREE vertical
    sub-regions (torso/shorts/socks, insets in `configs/target.yaml: bands`) rather than the
    single upper-torso region `src.events.kit_colour.kit_lab_sample` reads for jersey-number
    purposes.

    Reuses `src.events.kit_colour.lab_median_from_region` (the grass-mask + CIELAB-median
    arithmetic `kit_lab_sample` itself is built on, factored out for exactly this reuse -- see
    that function's own docstring) for each band independently, rather than reimplementing grass
    suppression here.

    `KitColourSample.confidence` is the sum of `configs/target.yaml: band_confidence_weights` over
    every band that actually cleared its own grass-suppression floor -- i.e. "how much of the kit
    was actually observed this frame", weighted toward torso (the band most reliably readable on
    real footage, see that config's own comment), NOT a colour-matching confidence. Returns `None`
    (never a zero-confidence sample) when NOT ONE band cleared its floor -- an honest "no colour
    evidence this frame at all".

    Note: `take_id`/`t` are accepted as explicit parameters (beyond the plan's own abbreviated
    `sample_kit_bands(frame_bgr, bbox, cfg)` sketch) because `KitColourSample` requires them and
    this function has no other way to know which take/timestamp it was called for -- the same
    "the plan's one-line signature wasn't meant to be exhaustive" pattern already used elsewhere
    in this codebase (e.g. `collect_track_jersey_digits`'s real signature is far longer than its
    module docstring's one-line mention).
    """
    box_w = bbox.x2 - bbox.x1
    box_h = bbox.y2 - bbox.y1
    if box_w <= 0 or box_h <= 0:
        return None

    kit_colour_cfg = cfg["kit_colour"]
    band_labs: dict[str, tuple[float, float, float] | None] = {}
    for band_name in ("torso", "shorts", "socks"):
        band_cfg = cfg["bands"][band_name]
        x1 = int(round(bbox.x1 + box_w * band_cfg["left_inset_frac"]))
        x2 = int(round(bbox.x2 - box_w * band_cfg["right_inset_frac"]))
        y1 = int(round(bbox.y1 + box_h * band_cfg["top_frac"]))
        y2 = int(round(bbox.y1 + box_h * band_cfg["bottom_frac"]))
        sample = lab_median_from_region(frame_bgr, x1, y1, x2, y2, kit_colour_cfg)
        band_labs[band_name] = (
            (float(sample[0][0]), float(sample[0][1]), float(sample[0][2]))
            if sample is not None
            else None
        )

    weights = cfg["band_confidence_weights"]
    confidence = sum(weights[band] for band, lab in band_labs.items() if lab is not None)
    if confidence <= 0.0:
        return None  # not one band cleared the grass floor -- no colour evidence at all

    return KitColourSample(
        torso_lab=band_labs["torso"],
        shorts_lab=band_labs["shorts"],
        socks_lab=band_labs["socks"],
        take_id=take_id,
        t=t,
        confidence=round(confidence, 4),
    )


def _aggregate_kit(bank: list[KitColourSample]) -> KitColourSample | None:
    """Per-band median across `bank` (reuses `src.events.kit_colour.median_kit_lab`, the same
    robust-to-one-bad-frame aggregation ADR-12's own per-track colour already uses), or `None`
    when the bank is empty or no sample in it has any band evidence at all. The returned sample's
    own `take_id`/`t` are the MOST RECENT contributing sample's -- this is the profile's current
    best-estimate aggregate, not a fresh single-frame reading, so those two fields are only
    meaningful as "as of when this was last updated"."""
    if not bank:
        return None

    def _band(attr: str) -> tuple[float, float, float] | None:
        values = [np.array(getattr(s, attr)) for s in bank if getattr(s, attr) is not None]
        med = median_kit_lab(values)
        return (float(med[0]), float(med[1]), float(med[2])) if med is not None else None

    torso, shorts, socks = _band("torso_lab"), _band("shorts_lab"), _band("socks_lab")
    if torso is None and shorts is None and socks is None:
        return None
    return KitColourSample(
        torso_lab=torso,
        shorts_lab=shorts,
        socks_lab=socks,
        take_id=bank[-1].take_id,
        t=bank[-1].t,
        confidence=round(sum(s.confidence for s in bank) / len(bank), 4),
    )


def build_target_profile(
    anchor_track: Track,
    kit_samples: list[KitColourSample],
    jersey_number: int | None,
    jersey_source: Literal["click", "filename", "human_confirmed", "ocr_verified"] | None,
    established_take_id: int,
    established_t: float,
    cfg: dict,
) -> TargetProfile:
    """Build the initial `TargetProfile` from the anchor track (a click, or a filename/human-
    confirmed jersey number's own first located track) plus whatever kit-colour samples were
    gathered while establishing it. Reuses `src.track.click_reid.build_click_profile`'s own
    `_median_height` (imported, not copied -- plan Stage 1 explicitly calls this out) for the
    profile's `median_height`.

    `kit_samples` is capped to `cfg['max_bank_samples']` (most recent kept) before being stored as
    the initial `kit_bank` and aggregated into `kit` -- the SAME cap `update_memory_bank` enforces
    later, so a profile is never built already over its own bound.
    """
    kept = kit_samples[-cfg["max_bank_samples"] :] if kit_samples else []
    return TargetProfile(
        jersey_number=jersey_number,
        jersey_source=jersey_source,
        kit=_aggregate_kit(kept),
        kit_bank=kept,
        median_height=_median_height(anchor_track),
        team_cluster=anchor_track.team,
        established_take_id=established_take_id,
        established_track_id=anchor_track.id,
        established_t=established_t,
        links=[],
    )


def update_memory_bank(
    profile: TargetProfile, sample: KitColourSample, link_confidence: float, cfg: dict
) -> TargetProfile:
    """The anti-drift rule (plan §4): append `sample` to the profile's memory bank ONLY when the
    link that produced it was itself accepted at `>= cfg['memory_update_min_confidence']`.

    An UNCERTAIN or low-confidence/unaccepted link returns `profile` completely UNCHANGED -- this
    is the one thing standing between "occlusion -> follows a different player -> learns their
    colour -> permanently mis-identifies them as the target" and a profile that can only ever be
    updated by evidence strong enough to have already been trusted as the real target. The bank is
    capped at `cfg['max_bank_samples']` (oldest dropped first) and `profile.kit` is recomputed as
    the per-band median of the resulting bank (`_aggregate_kit`, itself built on
    `src.events.kit_colour.median_kit_lab` -- reused, not reimplemented).
    """
    if link_confidence < cfg["memory_update_min_confidence"]:
        return profile

    bank = [*profile.kit_bank, sample]
    if len(bank) > cfg["max_bank_samples"]:
        bank = bank[-cfg["max_bank_samples"] :]
    return profile.model_copy(update={"kit_bank": bank, "kit": _aggregate_kit(bank)})


def save_target_profile(profile: TargetProfile, path: str | Path) -> None:
    """Persist `profile` to `path` (conventionally `work/<slug>/target.json`), following the
    existing `save_json` helper (CLAUDE.md §10)."""
    save_json(profile, path)


def load_target_profile(path: str | Path) -> TargetProfile:
    """Load a `TargetProfile` written by `save_target_profile`."""
    return TargetProfile.model_validate(load_json(path))
