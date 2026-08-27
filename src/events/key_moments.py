"""ADR-14 — celebration / other-key-moment detection (CLAUDE.md §13.3, §13.5 `key_moments.mp4`).

Two-stage, matching ADR-14's own "cheap pre-filter, VLM only on survivors" design (Golden Rule 1:
Gemini never touches player/ball detection):

1. `find_candidate_windows` (pure, no I/O) — a cheap motion pre-filter over the VERIFIED target's
   own track: a burst of elevated, sustained speed (reusing `src.events.sprints.track_speed_series`
   exactly as `sprint` does) that is NOT already covered by one of that same player's own
   already-counted event windows (sprint/dribble/possession — CLAUDE.md task spec: don't re-flag
   an already-explained sprint as a separate "key moment"). This pre-filter is DELIBERATELY weak/
   cheap (see `configs/key_moments.yaml` module comment) — its only job is bounding how many
   (expensive) Gemini calls happen; Gemini is the actual classifier.
2. `classify_candidate_windows` — decodes one representative native-resolution frame per surviving
   candidate window and asks Gemini "celebration or other notable key moment: yes/no", via the
   shared `src.common.gemini.call_gemini_vision` retry/backoff machinery (ADR-15's identity module
   uses the exact same helper for jersey-number classification). Only a Gemini "yes" ever produces
   a `KEY_MOMENT` `Event` — a "no", an abstention, or a failed call all emit nothing (Golden Rule
   5: never fabricate a key moment because the cheap pre-filter merely looked promising).
"""

from __future__ import annotations

import re
import uuid

from src.common.gemini import call_gemini_vision
from src.common.logging import get_logger
from src.common.types import Event, EventType, Track
from src.common.video import decode_frames
from src.events.segments import find_threshold_segments
from src.events.sprints import track_speed_series

logger = get_logger(__name__)

_VERDICT_PATTERN = re.compile(r"VERDICT\s*=\s*(YES|NO)", re.IGNORECASE)

_PROMPT = (
    "You are looking at ONE frame from a soccer/football video, at a moment flagged by a motion "
    "detector as unusually energetic for the highlighted/circled or most prominent player.\n\n"
    "Does this frame show a genuine celebration (goal celebration, fist pump, arms raised, "
    "teammates embracing, etc.) OR another clearly notable key moment (e.g. an injury, a heated "
    "argument/confrontation, a referee card, a crowd reaction)?\n\n"
    "Respond in EXACTLY this format, one line, nothing else:\n"
    "VERDICT=<YES or NO>; <one short sentence reason>\n\n"
    "If this just looks like ordinary run-of-play action (running, passing, tackling, standing "
    "around) with nothing notable happening, respond VERDICT=NO. Do not guess YES to be "
    "interesting -- only say YES if the moment is clearly and specifically notable."
)


def find_candidate_windows(
    track: Track, prefilter_cfg: dict, existing_windows: list[tuple[float, float]]
) -> list[tuple[float, float, float]]:
    """Candidate `(t_start, t_end, peak_speed)` windows in `track`'s own box sequence: a sustained
    speed segment (identical method to `src.events.sprints.detect_sprints`) that does not overlap
    any of `existing_windows` (that SAME player's own already-counted sprint/dribble/possession
    spans for this take) — pure/no I/O, unit-tested without a GPU/network call.

    Capped at `prefilter_cfg['max_candidates_per_take']`, keeping the highest-peak-speed windows
    when more qualify (the pre-filter's own strongest signal for "worth a Gemini look").
    """
    times, speeds = track_speed_series(track, prefilter_cfg["smoothing_window_frames"])
    segments = find_threshold_segments(
        times,
        speeds,
        prefilter_cfg["speed_threshold"],
        prefilter_cfg["min_duration_s"],
        prefilter_cfg["merge_gap_s"],
    )

    candidates: list[tuple[float, float, float]] = []
    for t0, t1 in segments:
        overlaps_existing = any(not (t1 <= w0 or t0 >= w1) for w0, w1 in existing_windows)
        if overlaps_existing:
            continue
        window_speeds = [s for t, s in zip(times, speeds, strict=True) if t0 <= t <= t1]
        peak = max(window_speeds) if window_speeds else 0.0
        candidates.append((t0, t1, peak))

    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates[: prefilter_cfg["max_candidates_per_take"]]


def _parse_verdict(text: str) -> tuple[bool, float, str]:
    """Parse Gemini's `VERDICT=...` line. Returns `(is_key_moment, confidence, raw_text)` — an
    unparseable response is treated as NO (never fabricated as a YES), logged as such via the
    returned raw text starting with `UNPARSEABLE:`."""
    match = _VERDICT_PATTERN.search(text or "")
    if match is None:
        return False, 0.0, f"UNPARSEABLE: {text!r}"
    is_yes = match.group(1).upper() == "YES"
    # Gemini gives no numeric confidence; a well-formed VERDICT=YES from a prompt explicitly
    # biased toward NO (never 1.0 -- still a VLM judgement call, not ground truth).
    return is_yes, (0.55 if is_yes else 0.0), text


def classify_candidate_windows(
    video_path,
    take_id: int | None,
    player_track_id: int,
    candidates: list[tuple[float, float, float]],
    gemini_api_key: str | None,
    vlm_cfg: dict,
    use_nvdec: bool = True,
) -> list[Event]:
    """Decode one representative native-res frame per candidate window (its midpoint) and ask
    Gemini to classify it. Returns `KEY_MOMENT` `Event`s ONLY for a Gemini "yes" clearing
    `vlm_cfg['min_confidence']` — everything else (a "no", an abstention-shaped unparseable reply,
    or a failed call after retries) emits nothing (Golden Rule 5).
    """
    if not candidates or not gemini_api_key:
        if candidates and not gemini_api_key:
            logger.info(
                "key_moments: %d candidate window(s) found but no GEMINI_API_KEY configured -- "
                "skipping VLM classification, emitting zero key-moment events (never guessed)",
                len(candidates),
            )
        return []

    events: list[Event] = []
    for t0, t1, peak_speed in candidates:
        mid_t = (t0 + t1) / 2.0
        frame = None
        for _idx, _t, decoded in decode_frames(
            video_path,
            fps=2,
            start=max(0.0, mid_t - 0.15),
            end=mid_t + 0.15,
            scale_width=None,
            use_nvdec=use_nvdec,
        ):
            frame = decoded
            break
        if frame is None:
            logger.warning("key_moments: could not decode a frame near t=%.2f, skipping", mid_t)
            continue

        text, error = call_gemini_vision(_PROMPT, frame, gemini_api_key, vlm_cfg)
        if error is not None:
            logger.warning("key_moments: gemini call failed at t=%.2f: %s", mid_t, error)
            continue
        is_yes, confidence, raw = _parse_verdict(text or "")
        if not is_yes or confidence < vlm_cfg["min_confidence"]:
            continue

        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.KEY_MOMENT,
                t_start=t0,
                t_end=t1,
                player_track_id=player_track_id,
                take_id=take_id,
                confidence=confidence,
                source="motion_prefilter_gemini_classified",
                evidence={
                    "peak_speed": peak_speed,
                    "unit": "bbox_heights_per_second",
                    "calibrated": False,
                    "gemini_raw_response": raw,
                    "classified_frame_t": mid_t,
                },
            )
        )
    return events
