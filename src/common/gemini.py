"""Shared low-level Gemini `generateContent` HTTP caller (ADR-14/ADR-15).

Plain `requests` POST (Apache-2.0, CLAUDE.md §7) — no Google SDK dependency. Two callers build on
this: `src/identity/jersey_vlm.py` (jersey-number cross-check, ADR-15) and
`src/events/key_moments.py` (celebration/other-key-moment classification, ADR-14) — both need the
identical retry/backoff/JPEG-encode machinery with only the prompt and response-parsing differing,
so it lives once here rather than being copy-pasted per caller.

Golden Rule 1 (CLAUDE.md §2): Gemini is used ONLY to classify already-cheaply-filtered candidate
crops/windows — never for player/ball detection, which stays RF-DETR.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import requests

from src.common.logging import get_logger

logger = get_logger(__name__)

# Retried: transient server-side failure, worth another attempt. NOT retried (fails fast): a
# genuine client error (bad key, bad request) that a retry can never fix.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _extract_text(response_json: dict) -> str:
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def call_gemini_vision(
    prompt: str, image_bgr: np.ndarray, api_key: str, vlm_cfg: dict
) -> tuple[str | None, str | None]:
    """POST one prompt + one BGR image to Gemini `generateContent`.

    Returns `(text, error)`: `error is None` and `text` is the model's raw response text on
    success; on failure `text is None` and `error` is a human-readable reason (every retry
    exhausted, a non-retryable HTTP error, or a JPEG-encode failure) — callers must never treat a
    failure as a legitimate "nothing found" answer (CLAUDE.md task spec).
    """
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, vlm_cfg["jpeg_quality"]])
    if not ok:
        return None, "could not JPEG-encode image"
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    url = vlm_cfg["api_url_template"].format(model=vlm_cfg["model"])
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
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
                return _extract_text(resp.json()), None
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                return None, last_error
            logger.warning(
                "gemini call attempt %d/%d failed (%s) -- will retry",
                attempt + 1,
                max_retries,
                last_error,
            )
        if attempt < max_retries - 1:
            time.sleep(backoff_base * (2**attempt))

    return None, f"exhausted {max_retries} retries, last error: {last_error}"
