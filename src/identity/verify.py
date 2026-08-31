"""ADR-15 — per-take jersey-identity verification orchestrator (CLAUDE.md §3.3/§5.1).

Only ever invoked when a video has no filename jersey number (`target_jersey is None` —
`src/ingest/discovery.py::find_videos` already returns that for a non-matching filename like
`Jordan Thomas Highlight Video.mp4`). Never touches the original 5 `clip<N> <jersey>.mp4` clips'
Phase-1 flow.

Pipeline, per take:

1. **Location prior** — reuse `src.highlights.selection.select_targets` (arrow vote + within-take
   fragment stitching, unchanged) called with `target_jersey=None` to get, per take, a candidate
   track-id timeline. This is a LOCATION prior only (CLAUDE.md task spec: the arrow never reads a
   digit) — it never by itself confirms identity.
2. **Evidence collection** (`_collect_reads_for_take`) — ONE native-resolution decode pass over
   the whole video (never re-seeking per candidate frame, which would risk landing on the wrong
   take at a keyframe boundary — see module note below), cropping the location candidate's own box
   wherever it clears `identity.yaml: crop.min_crop_height_frac` (resolution-aware, Stage A of the
   "streamed-gathering-treehouse" plan), running EasyOCR first (cheap,
   local) and escalating only OCR-ambiguous-or-silent crops to Gemini, bounded by
   `identity.yaml: vlm.max_escalations_per_take` (cost control, mirrors ADR-14's own "cheap
   pre-filter, VLM only on survivors" pattern).
3. **Aggregation** (`aggregate_take_identity`, pure/testable) — requires temporal agreement across
   >= `identity.yaml: aggregation.min_agreeing_frames` INDEPENDENT frames (OCR and/or VLM, any
   mix) reading the SAME digit string before declaring `status="verified"`; a tie for the top
   count, or too few agreeing frames, is `"unverified"` — never broken arbitrarily (owner's
   explicit rule, CLAUDE.md ADR-15).

Decode-accuracy note: a per-candidate-frame `ffmpeg -ss <t>` seek is a KEYFRAME seek and can land
up to one GOP before the requested time — for evidence spanning a take boundary that risk is
exactly what ADR-16 exists to avoid recreating one level up. This module instead does ONE
`decode_frames` call over the WHOLE video (a single seek, at true t=0) and buckets every decoded
frame into its take via `assign_take_id`, so every frame's reported timestamp is accurate
throughout (see `src.common.video.decode_frames`'s own `-vf fps=N` resampling, which IS accurate
once a stream is being read sequentially).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.types import BBox, Take, Track, TrackBox
from src.common.video import decode_frames, probe
from src.detect.overlay_mask import ArrowHint
from src.highlights.selection import select_targets
from src.identity.jersey_ocr import (
    free_easyocr_reader,
    load_easyocr_reader,
    read_jersey_digits,
)
from src.identity.jersey_vlm import classify_jersey_number
from src.track.tracker import assign_take_id

logger = get_logger(__name__)


class FrameRead(BaseModel):
    """One independent digit read at one instant — the atomic unit of evidence."""

    t: float
    frame_index: int  # index within THIS stage's own single native-res decode pass
    source: Literal["ocr", "vlm"]
    digits: str | None
    confidence: float
    raw: str | None = None


class TakeIdentityResult(BaseModel):
    """Stage 5.5's per-take output — everything `identity_report.json` needs (CLAUDE.md §13.5),
    including the FULL evidence trail (Golden Rule 5: no black-box confidence)."""

    take_id: int
    jersey_number: int | None
    status: Literal["verified", "unverified"]
    confidence: float
    evidence_frames: list[int]
    location_method: str  # SelectionMethod from src/highlights/selection.py, as a plain str
    location_track_ids: list[int]
    n_crops_considered: int = 0
    n_too_small: int = 0
    n_ocr_confident: int = 0
    n_vlm_calls: int = 0
    n_vlm_failed: int = 0
    all_reads: list[FrameRead] = []
    human_override_note: str | None = None  # ADR-18 (1): set whenever a `--target-jersey`
    # human-confirmed number was supplied for this video and applied (or attempted) on this take
    # -- always populated (never silently applied) so a human override is as auditable as any
    # other verification path, including the case where it CONTRADICTS an OCR/VLM majority
    # (Golden Rule 5: never hide contradicting evidence, even though Golden Rule 4 has the human's
    # own authority win).


class IdentityReport(BaseModel):
    """Stage 5.5's whole-video output (`work/<slug>/identity.json`, mirrored into
    `output/<slug>/identity_report.json` — CLAUDE.md §13.5)."""

    video: str
    takes: list[TakeIdentityResult]


# ---------------------------------------------------------------------------
# aggregation (pure, GPU/network-free — see tests/test_identity.py)
# ---------------------------------------------------------------------------


def aggregate_take_identity(
    reads: list[FrameRead],
    agg_cfg: dict,
    human_confirmed_jersey: int | None = None,
) -> dict:
    """Reduce one take's `FrameRead`s to
    `{jersey_number, status, confidence, evidence_frames, human_override_note}`.

    Normal (no human override) path: requires >= `agg_cfg['min_agreeing_frames']` reads (any mix
    of OCR/VLM) agreeing on the SAME digit string, with that count STRICTLY the unique maximum
    among all digit strings seen (a tie for the top count is `"unverified"`, never broken
    arbitrarily — owner's explicit rule). `confidence` blends how much of the total evidence
    agrees with the mean confidence of the agreeing reads themselves; a verified result
    additionally must clear `agg_cfg['min_verified_confidence']`.

    ADR-18 (1) — human-in-the-loop override (CLAUDE.md Golden Rule 4): when
    `human_confirmed_jersey` is given AND at least one read in `reads` actually matches it, a
    SINGLE supporting read is enough to verify (a human who watched the actual footage is
    stronger evidence than an automated multi-frame vote — the normal `min_agreeing_frames` bar
    does not apply). If the OTHER evidence has a contradicting digit string with STRICTLY MORE
    supporting reads than the human's own number, the human's number still wins (Golden Rule 4:
    human authority), but `human_override_note` records the disagreement in full instead of
    silently accepting it (Golden Rule 5: never hide contradicting evidence). When
    `human_confirmed_jersey` is given but has ZERO supporting reads in this take, the override
    cannot manufacture evidence that was never read — it falls back to the normal automated vote
    below, with a note explaining why.
    """
    by_digits: dict[str, list[FrameRead]] = defaultdict(list)
    for r in reads:
        if r.digits is not None:
            by_digits[r.digits].append(r)

    human_note: str | None = None
    if human_confirmed_jersey is not None:
        human_digits = str(human_confirmed_jersey)
        human_supporting = by_digits.get(human_digits, [])
        if human_supporting:
            other_counts = {d: len(v) for d, v in by_digits.items() if d != human_digits}
            max_other = max(other_counts.values(), default=0)
            if max_other > len(human_supporting):
                top_other = max(other_counts, key=lambda d: other_counts[d])
                human_note = (
                    f"human-confirmed jersey #{human_confirmed_jersey} ACCEPTED (Golden Rule 4: "
                    "human authority wins) DESPITE a contradicting OCR/VLM majority -- digit "
                    f"'{top_other}' had {max_other} supporting read(s) vs "
                    f"{len(human_supporting)} for '{human_digits}'; flagged here, never hidden "
                    "(Golden Rule 5)"
                )
            else:
                human_note = (
                    f"human-confirmed jersey #{human_confirmed_jersey} accepted "
                    f"({len(human_supporting)} supporting read(s), no contradicting majority)"
                )
            confidence = min(
                1.0,
                max(
                    agg_cfg["min_verified_confidence"],
                    sum(r.confidence for r in human_supporting) / len(human_supporting),
                ),
            )
            return {
                "jersey_number": human_confirmed_jersey,
                "status": "verified",
                "confidence": confidence,
                "evidence_frames": sorted(r.frame_index for r in human_supporting),
                "human_override_note": human_note,
            }
        human_note = (
            f"human-confirmed jersey #{human_confirmed_jersey} was given, but this take had ZERO "
            "supporting OCR/VLM reads for it -- falling back to the normal automated vote below "
            "(the human override lowers the evidence bar, it never fabricates evidence that was "
            "never read, Golden Rule 5)"
        )

    if not by_digits:
        return {
            "jersey_number": None,
            "status": "unverified",
            "confidence": 0.0,
            "evidence_frames": [],
            "human_override_note": human_note,
        }

    max_count = max(len(v) for v in by_digits.values())
    top_digits = [d for d, v in by_digits.items() if len(v) == max_count]
    min_agree = agg_cfg["min_agreeing_frames"]

    if max_count < min_agree or len(top_digits) != 1:
        return {
            "jersey_number": None,
            "status": "unverified",
            "confidence": 0.0,
            "evidence_frames": [],
            "human_override_note": human_note,
        }

    digits = top_digits[0]
    supporting = by_digits[digits]
    agreement_fraction = len(supporting) / len(reads)
    mean_source_conf = sum(r.confidence for r in supporting) / len(supporting)
    confidence = min(1.0, 0.5 * agreement_fraction + 0.5 * mean_source_conf)

    if confidence < agg_cfg["min_verified_confidence"]:
        return {
            "jersey_number": None,
            "status": "unverified",
            "confidence": confidence,
            "evidence_frames": [],
            "human_override_note": human_note,
        }

    return {
        "jersey_number": int(digits),
        "status": "verified",
        "confidence": confidence,
        "evidence_frames": sorted(r.frame_index for r in supporting),
        "human_override_note": human_note,
    }


# ---------------------------------------------------------------------------
# geometry: map a Track's (downscaled) box into native-decode pixel space
# ---------------------------------------------------------------------------


def _nearest_box_among(
    tracks: list[Track], t: float, tolerance_s: float
) -> tuple[int, TrackBox] | None:
    """`(raw_track_id, box)` of whichever of `tracks`' own boxes is nearest in time to `t`, within
    `tolerance_s` — `None` if nothing qualifies."""
    best_id: int | None = None
    best_box = None
    best_dt = tolerance_s
    for tr in tracks:
        for box in tr.boxes:
            dt = abs(box.t - t)
            if dt <= best_dt:
                best_dt, best_id, best_box = dt, tr.id, box
    if best_id is None or best_box is None:
        return None
    return best_id, best_box


def _scale_bbox_to_native(
    bbox: BBox, scale_x: float, scale_y: float, expand: float, native_w: int, native_h: int
) -> tuple[int, int, int, int]:
    """Scale a detect-stage (downscaled) `bbox` into native-resolution pixel coordinates, expand
    by `expand` fraction on each side, and clamp into `[0, native_w) x [0, native_h)`."""
    x1, y1, x2, y2 = bbox.x1 * scale_x, bbox.y1 * scale_y, bbox.x2 * scale_x, bbox.y2 * scale_y
    w, h = x2 - x1, y2 - y1
    x1 -= w * expand
    x2 += w * expand
    y1 -= h * expand
    y2 += h * expand
    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(native_w, int(round(x2)))
    iy2 = min(native_h, int(round(y2)))
    return ix1, iy1, ix2, iy2


# ---------------------------------------------------------------------------
# evidence collection (one native-res decode pass, streamed OCR+VLM)
# ---------------------------------------------------------------------------


def _collect_reads_for_take(
    frames_by_take: dict[int, list[tuple[int, float, np.ndarray]]],
    take_tracks: list[Track],
    scale_x: float,
    scale_y: float,
    native_w: int,
    native_h: int,
    identity_cfg: dict,
    reader,
    gemini_api_key: str | None,
    identity_fps_sample: float,
) -> tuple[list[FrameRead], dict]:
    """Evidence for ONE take, given the frames already bucketed for it. Returns
    `(reads, counters)`; `counters` feeds `TakeIdentityResult`'s own bookkeeping fields."""
    crop_cfg = identity_cfg["crop"]
    ocr_cfg = identity_cfg["ocr"]
    vlm_cfg = identity_cfg["vlm"]
    # Stage A (resolution-aware crop gate): the gate is a FRACTION of native decode-frame height,
    # not a fixed pixel count, so it scales correctly on any input resolution (see
    # configs/identity.yaml: crop.min_crop_height_frac's own comment for the 4K/720p measurements
    # behind this fraction).
    min_height = crop_cfg["min_crop_height_frac"] * native_h
    expand = crop_cfg["torso_crop_expand"]
    tolerance_s = 1.0 / identity_fps_sample
    max_escalations = vlm_cfg["max_escalations_per_take"]

    reads: list[FrameRead] = []
    counters = {
        "n_crops_considered": 0,
        "n_too_small": 0,
        "n_ocr_confident": 0,
        "n_vlm_calls": 0,
        "n_vlm_failed": 0,
    }
    take_id = take_tracks[0].take_id if take_tracks else None

    for frame_index, t, frame in frames_by_take.get(take_id, []):
        found = _nearest_box_among(take_tracks, t, tolerance_s)
        if found is None:
            continue
        _raw_id, box = found
        x1, y1, x2, y2 = _scale_bbox_to_native(
            box.bbox, scale_x, scale_y, expand, native_w, native_h
        )
        if (y2 - y1) < min_height or x2 <= x1 or y2 <= y1:
            counters["n_too_small"] += 1
            continue
        counters["n_crops_considered"] += 1
        crop = frame[y1:y2, x1:x2]

        ocr_read = read_jersey_digits(reader, crop, ocr_cfg)
        if ocr_read.is_confident:
            counters["n_ocr_confident"] += 1
            reads.append(
                FrameRead(
                    t=t,
                    frame_index=frame_index,
                    source="ocr",
                    digits=ocr_read.digits,
                    confidence=ocr_read.confidence,
                )
            )
            continue

        if gemini_api_key and counters["n_vlm_calls"] < max_escalations:
            counters["n_vlm_calls"] += 1
            digits, conf, raw = classify_jersey_number(crop, gemini_api_key, vlm_cfg)
            if raw.startswith("CALL_FAILED"):
                counters["n_vlm_failed"] += 1
                logger.warning("take=%s: gemini call failed at t=%.2f: %s", take_id, t, raw)
            reads.append(
                FrameRead(
                    t=t,
                    frame_index=frame_index,
                    source="vlm",
                    digits=digits,
                    confidence=conf,
                    raw=raw,
                )
            )

    return reads, counters


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------


def verify_identities_for_video(
    video_path: str | Path,
    takes: list[Take],
    tracks: list[Track],
    arrow_hints: list[ArrowHint],
    detect_frame_width: float,
    detect_frame_height: float,
    selection_cfg: dict,
    identity_cfg: dict,
    gemini_api_key: str | None,
    use_nvdec: bool = True,
    human_confirmed_jersey: int | None = None,
) -> IdentityReport:
    """Run Stage 5.5 for one filename-less video end to end. Never invoked when the video HAS a
    filename jersey number (`src/pipeline/run.py` gates this).

    `human_confirmed_jersey` (ADR-18 (1), CLAUDE.md Golden Rule 4) is the optional
    `--target-jersey` CLI override: a human who watched this specific filename-less video and
    confirmed which number to look for. Threaded straight into `aggregate_take_identity` for
    every take -- see that function's own docstring for the exact "one supporting read is enough,
    but a contradicting majority is flagged, never hidden" semantics.
    """
    video_path = Path(video_path)
    location = select_targets(
        str(video_path),
        None,
        takes,
        tracks,
        arrow_hints,
        detect_frame_width,
        detect_frame_height,
        selection_cfg,
    )
    location_by_take = {t.take_id: t for t in location.takes}

    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]
    scale_x = native_w / detect_frame_width if detect_frame_width else 1.0
    scale_y = native_h / detect_frame_height if detect_frame_height else 1.0

    identity_fps_sample = identity_cfg["crop"]["identity_fps_sample"]
    tracks_by_take: dict[int, list[Track]] = defaultdict(list)
    for tr in tracks:
        tracks_by_take[tr.take_id].append(tr)

    eligible_take_ids = {
        ts.take_id for ts in location.takes if ts.method != "none" and ts.track_ids
    }

    results: list[TakeIdentityResult] = []
    if not eligible_take_ids:
        logger.warning(
            "no take in %s has ANY location candidate (arrow_vote/heuristic_fallback) -- "
            "identity verification has nothing to run against, every take will be unverified",
            video_path.name,
        )
        for take in takes:
            sel = location_by_take.get(take.id)
            results.append(
                TakeIdentityResult(
                    take_id=take.id,
                    jersey_number=None,
                    status="unverified",
                    confidence=0.0,
                    evidence_frames=[],
                    location_method=sel.method if sel else "none",
                    location_track_ids=sel.track_ids if sel else [],
                )
            )
        return IdentityReport(video=str(video_path), takes=results)

    # ONE native-resolution decode pass over the WHOLE video (see module docstring for why this
    # beats per-candidate-frame seeking), bucketed into takes by timestamp as we go.
    logger.info(
        "identity: single native-res decode pass over %s at %.1f fps (eligible takes: %s)",
        video_path.name,
        identity_fps_sample,
        sorted(eligible_take_ids),
    )
    frames_by_take: dict[int, list[tuple[int, float, np.ndarray]]] = defaultdict(list)
    for frame_index, t, frame in decode_frames(
        video_path, fps=identity_fps_sample, scale_width=None, use_nvdec=use_nvdec
    ):
        take_id = assign_take_id(t, takes)
        if take_id in eligible_take_ids:
            frames_by_take[take_id].append((frame_index, t, frame.copy()))

    reader = load_easyocr_reader(identity_cfg["ocr"])
    try:
        for take in takes:
            sel = location_by_take.get(take.id)
            if sel is None or take.id not in eligible_take_ids:
                results.append(
                    TakeIdentityResult(
                        take_id=take.id,
                        jersey_number=None,
                        status="unverified",
                        confidence=0.0,
                        evidence_frames=[],
                        location_method=sel.method if sel else "none",
                        location_track_ids=sel.track_ids if sel else [],
                    )
                )
                continue

            take_tracks = [
                tr for tr in tracks_by_take.get(take.id, []) if tr.id in set(sel.track_ids)
            ]
            reads, counters = _collect_reads_for_take(
                frames_by_take,
                take_tracks,
                scale_x,
                scale_y,
                native_w,
                native_h,
                identity_cfg,
                reader,
                gemini_api_key,
                identity_fps_sample,
            )
            agg = aggregate_take_identity(
                reads, identity_cfg["aggregation"], human_confirmed_jersey
            )
            logger.info(
                "take=%d: location=%s(%s) -> identity=%s number=%s conf=%.2f "
                "(%d crop(s), %d ocr-confident, %d vlm call(s), %d vlm failure(s))",
                take.id,
                sel.method,
                sel.track_ids,
                agg["status"],
                agg["jersey_number"],
                agg["confidence"],
                counters["n_crops_considered"],
                counters["n_ocr_confident"],
                counters["n_vlm_calls"],
                counters["n_vlm_failed"],
            )
            results.append(
                TakeIdentityResult(
                    take_id=take.id,
                    jersey_number=agg["jersey_number"],
                    status=agg["status"],
                    confidence=agg["confidence"],
                    evidence_frames=agg["evidence_frames"],
                    location_method=sel.method,
                    location_track_ids=sel.track_ids,
                    all_reads=reads,
                    human_override_note=agg.get("human_override_note"),
                    **counters,
                )
            )
    finally:
        free_easyocr_reader(reader)

    return IdentityReport(video=str(video_path), takes=results)
