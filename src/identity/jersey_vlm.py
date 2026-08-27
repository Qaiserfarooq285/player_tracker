"""ADR-15 — Gemini cross-check for jersey-number verification (CLAUDE.md §3.3/§5.1).

Plain `requests` POST (Apache-2.0, CLAUDE.md §7) to the Gemini `generateContent` endpoint — no
Google SDK dependency. Used ONLY to cross-check crops EasyOCR left ambiguous or silent on (see
`src/identity/verify.py`), never for player/ball detection (Golden Rule 1: RF-DETR stays the
detector; a VLM is one more specialized stage, same pattern as ADR-14's celebration classifier).

Model note (ADR-15): `gemini-2.5-flash` (ADR-14's implicit assumption) now 404s for new callers
("no longer available to new users") — resolved to `gemini-3.6-flash` (confirmed reachable
2026-08-27). A transient 503 "UNAVAILABLE" was also observed during testing, hence the retry with
exponential backoff below.

The prompt explicitly, repeatedly invites Gemini to answer UNKNOWN rather than guess (verified
2026-08-27: on a genuinely illegible test crop it correctly answered `NUMBER=UNKNOWN; ...` rather
than fabricating a digit) — this is the whole point of using a VLM here at all (CLAUDE.md ADR-15:
"never guess the jersey number... or mark it UNKNOWN/unverified").
"""

from __future__ import annotations

import base64
import re
import time

import cv2
import numpy as np
import requests

from src.common.logging import get_logger

logger = get_logger(__name__)

_NUMBER_PATTERN = re.compile(r"NUMBER\s*=\s*(UNKNOWN|\d{1,2})", re.IGNORECASE)

_PROMPT = (
    "You are looking at a cropped photo of ONE soccer/football player, taken from a video frame. "
    "Look ONLY at the number printed on the player's jersey (shirt/back/front), if one is clearly "
    "legible in this exact crop.\n\n"
    "Respond in EXACTLY this format, one line, nothing else:\n"
    "NUMBER=<the digit or two-digit number>; <one short sentence reason>\n\n"
    "If no jersey number is clearly and confidently legible in this crop -- because the player is "
    "turned away, the number is blurred, occluded, cropped off, too small, or simply not visible "
    "-- you MUST respond exactly:\n"
    "NUMBER=UNKNOWN; <one short sentence reason>\n\n"
    "Do not guess. It is much better and more useful to say UNKNOWN than to guess a number you "
    "are not genuinely confident about."
)

# Retried: transient server-side failure, worth another attempt. NOT retried (fails fast): a
# genuine client error (bad key, bad request) that a retry can never fix.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _extract_text(response_json: dict) -> str:
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _parse_response(text: str) -> tuple[str | None, float, str]:
    """Parse Gemini's `NUMBER=...` line. Returns `(digits, confidence, raw_text)`.

    `digits is None` here covers TWO distinct, both-legitimate outcomes the caller must not
    conflate: an explicit, honest `NUMBER=UNKNOWN` abstention (small nonzero confidence, since the
    model DID look and DID respond in the expected format), and an unparseable response (0.0
    confidence, `raw_text` prefixed `CALL_FAILED:` — treated as a failed call, never as a silent
    "no digit visible", per CLAUDE.md task spec).
    """
    match = _NUMBER_PATTERN.search(text or "")
    if match is None:
        return None, 0.0, f"CALL_FAILED: unparseable response (no NUMBER= found): {text!r}"
    value = match.group(1).upper()
    if value == "UNKNOWN":
        return None, 0.15, text
    # Gemini gives no numeric confidence of its own; a well-formed NUMBER=<digits> answer, from a
    # prompt that explicitly and repeatedly offers UNKNOWN as the easy/acceptable answer, is
    # treated as a fixed, documented confidence -- never 1.0 (a VLM read is still not ground
    # truth), comfortably above an ambiguous OCR read's own confidence band.
    return value, 0.75, text


def classify_jersey_number(
    crop_bgr: np.ndarray, api_key: str, vlm_cfg: dict
) -> tuple[str | None, float, str]:
    """Ask Gemini to read the jersey number in `crop_bgr`.

    Returns `(digits_or_None, confidence, raw_response)`. `raw_response` is prefixed
    `"CALL_FAILED:"` whenever the call never produced a usable answer (every retry exhausted, a
    non-retryable HTTP error, or an unparseable response) — the caller (`src/identity/verify.py`)
    must check for this prefix and log/count it DISTINCTLY from a genuine `NUMBER=UNKNOWN`
    abstention (CLAUDE.md task spec: "never treat a failed call as 'no digit visible'").
    """
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, vlm_cfg["jpeg_quality"]])
    if not ok:
        return None, 0.0, "CALL_FAILED: could not JPEG-encode crop"
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    url = vlm_cfg["api_url_template"].format(model=vlm_cfg["model"])
    body = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": vlm_cfg["temperature"]},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    max_retries = vlm_cfg["max_retries"]
    backoff_base = vlm_cfg["retry_backoff_base_s"]

    last_error = "no attempt made"
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=vlm_cfg["timeout_s"])
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "gemini call attempt %d/%d raised %s -- will retry",
                attempt + 1,
                max_retries,
                last_error,
            )
        else:
            if resp.status_code == 200:
                return _parse_response(_extract_text(resp.json()))
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                return None, 0.0, f"CALL_FAILED: {last_error}"
            logger.warning(
                "gemini call attempt %d/%d failed (%s) -- will retry",
                attempt + 1,
                max_retries,
                last_error,
            )
        if attempt < max_retries - 1:
            time.sleep(backoff_base * (2**attempt))

    return None, 0.0, f"CALL_FAILED: exhausted {max_retries} retries, last error: {last_error}"
