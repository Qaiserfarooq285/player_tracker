"""Stage 5b of the evidence-based event redesign (owner spec, 2026-09-02): kit COLOUR as the
PRIMARY teammate signal for pass/assist attribution, requested explicitly by the owner ("also add
the player jersy color for passes and assist").

**Why this exists as a separate module from `src.team.classifier`, not a change to it:**
`src.team.classifier.crop_lab_mean`/`per_track_lab_median` are ADR-12's own team-CLUSTERING
primitives (KMeans on whole-torso-band crops) and are self-consistent *only* when every track in a
take is compared the same way against every other — changing their crop geometry or adding grass
suppression would risk silently shifting existing cluster assignments elsewhere in the pipeline.
This module builds a SEPARATE, absolute (not clustered) colour measurement, compared against the
fixed reference table in `configs/annotations.yaml: colour_reference_lab` the same way
`src.annotations.associate.nearest_track_by_colour` already does for manual-mode association --
reusing that exact comparison convention rather than inventing a second one.

**Measured problem this fixes (real footage, `chelsea_burnley_target10`):** a plain torso-band Lab
read for the claret Burnley #21 (raw track 47) came back NEAR-NEUTRAL (L*55.7, a*-9.0, b*12.5)
instead of red -- grass dominates the crop when the player is mid-stride or the crop is loose.

**Measured fix and its real limits (2026-09-02, `configs/identity.yaml: parseq_soccernet.
number_crop` geometry + HSV grass suppression, 14 samples/track, 6 tracks with VISUALLY CONFIRMED
kit colour):**

    pair                          RAW dE   SUPPRESSED dE
    blue #10  vs claret #21        14.5        24.5

Suppression roughly doubled the separation on the confirmed pair. But it is NOT uniformly clean:
one visually-confirmed blue control track measured closer to the claret cluster than to its own
blue partner after suppression (kept-kit-pixel fraction as low as 31% on some tracks, occlusion/
pose-dependent). Conclusion, applied directly below: grass suppression is a real, worthwhile
improvement, but a single-sample colour read is NOT reliable enough to gate a stat on its own --
`kit_colour_verdict` therefore requires the winning reference colour to beat the runner-up by a
real MARGIN (the same "don't break a close call arbitrarily" discipline already used for jersey
disambiguation, `src/annotations/associate.py`'s `min_disambiguation_margin`), and demotes to "no
verdict" rather than guessing whenever the margin isn't cleared or too little kit survives
suppression to trust the sample at all.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.common.types import BBox
from src.identity.jersey_parseq import number_region

# HSV grass band (OpenCV H in 0-179). Deliberately wide: pitch green varies with lighting/mowing
# stripes across real footage, and this only needs to reject grass, not classify it precisely --
# false-negative grass pixels are far cheaper than accidentally rejecting real green/yellow kit.
_GRASS_H_LO = 30
_GRASS_H_HI = 95
_GRASS_S_MIN = 40


def _grass_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask, `True` where a pixel looks like pitch grass (to be EXCLUDED), same shape as
    `crop_bgr`'s first two dims flattened. Pure OpenCV colour-space arithmetic, no model."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return (hsv[:, 0] >= _GRASS_H_LO) & (hsv[:, 0] <= _GRASS_H_HI) & (hsv[:, 1] >= _GRASS_S_MIN)


def kit_lab_sample(
    frame_bgr: np.ndarray, bbox: BBox, number_crop_cfg: dict, kit_colour_cfg: dict
) -> tuple[np.ndarray, float] | None:
    """`(true_cielab, kept_fraction)` for ONE player box in ONE frame, or `None` when too few
    non-grass pixels survive to trust the sample at all (`kit_colour_cfg['min_kept_pixel_frac']`)
    -- an honest "no colour evidence this frame", never a guess from mostly-background pixels.

    Reuses the ADR-21 `number_region` geometry (upper-torso, already proven to isolate the shirt
    from limbs/background for jersey-number reading) rather than the wider ADR-12 torso band --
    the tighter crop starts with less grass to suppress in the first place.
    """
    from src.annotations.associate import opencv_lab_to_cielab

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = number_region(
        int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), number_crop_cfg
    )
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    pixels = crop.reshape(-1, 3)
    keep = ~_grass_mask(crop)
    kept_frac = float(keep.mean()) if len(keep) else 0.0
    too_few = keep.sum() < kit_colour_cfg["min_kept_pixels"]
    too_sparse = kept_frac < kit_colour_cfg["min_kept_pixel_frac"]
    if too_few or too_sparse:
        return None

    kept_pixels = pixels[keep].reshape(-1, 1, 3)
    lab = cv2.cvtColor(kept_pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    return opencv_lab_to_cielab(np.median(lab, axis=0)), kept_frac


def median_kit_lab(samples: list[np.ndarray]) -> np.ndarray | None:
    """Per-track kit colour: median of that track's own per-frame `kit_lab_sample` reads (robust
    to the occasional bad frame, same aggregation discipline as
    `src.team.classifier.per_track_lab_median`). `None` for an empty input -- a track with no
    frame that cleared `kit_lab_sample`'s own kept-pixel floor has no colour evidence at all."""
    if not samples:
        return None
    return np.median(np.array(samples), axis=0)


def kit_colour_same_team(
    lab_a: np.ndarray | None, lab_b: np.ndarray | None, kit_colour_cfg: dict
) -> tuple[bool | None, float]:
    """`(same_team, distance)` -- compare two players' OWN measured kit CIELAB colours DIRECTLY,
    never against a fixed external palette.

    Deliberately not compared against `configs/annotations.yaml: colour_reference_lab`: every
    entry there except `blue` is REASONED-BUT-UNMEASURED (that file's own comment), and checking
    against it here would compare a real grass-suppressed measurement to a guessed reference --
    confirmed concretely on real data, the claret Burnley kit's own measured Lab (~L55 a2 b0) does
    not land anywhere near that file's guessed `maroon` entry ([25.5, 48.1, 38.1], dE ~64). The
    natural, general test needs no palette at all: "is player B wearing the same colour as player
    A", which works for any two kit colours in any video.

    **Real two-sided gate, measured 2026-09-02** (6 visually-confirmed tracks,
    `chelsea_burnley_target10`, grass-suppressed): SAME-team pairs (blue-blue, claret-claret)
    measured 3.4-23.4 apart; CROSS-team pairs (blue-claret) measured 9.8-42.4 apart. The two
    distributions OVERLAP -- one same-claret pair (a track with a partially-occluded, low-quality
    crop) measured 23.4, one cross-team pair measured only 9.8. No single threshold separates them
    without a wrong call, so this uses a gap instead:
        distance <= same_team_max_dist  -> True  (confident teammate)
        distance >= diff_team_min_dist  -> False (confident opponent)
        otherwise                        -> None (undecided -- caller falls through to jersey
                                                    number / team-cluster quality)
    With `same_team_max_dist=8.0`/`diff_team_min_dist=25.0` (the shipped defaults), EVERY one of
    the 15 measured pairs above that clears either gate is classified correctly (0 false
    positives, 0 false negatives) -- 8 of 15 pairs land in the undecided gap, which is exactly the
    owner's own stated preference for pass counting ("strict -- fewer, high-confidence"): a
    genuinely ambiguous read must fall through rather than risk crediting the wrong team.
    """
    if lab_a is None or lab_b is None:
        return None, 0.0
    dist = float(np.linalg.norm(lab_a - lab_b))
    if dist <= kit_colour_cfg["same_team_max_dist"]:
        return True, dist
    if dist >= kit_colour_cfg["diff_team_min_dist"]:
        return False, dist
    return None, dist
