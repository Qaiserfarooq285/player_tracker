"""Shared low-level Gemini `generateContent` HTTP caller (ADR-14/ADR-15).

Plain `requests` POST (Apache-2.0, CLAUDE.md §7) — no Google SDK dependency. Two callers build on
this: `src/identity/jersey_vlm.py` (jersey-number cross-check, ADR-15) and
`src/events/key_moments.py` (celebration/other-key-moment classification, ADR-14) — both need the
identical retry/backoff/JPEG-encode/model-fallback machinery with only the prompt and
response-parsing differing, so it lives once here rather than being copy-pasted per caller.

Golden Rule 1 (CLAUDE.md §2): Gemini is used ONLY to classify already-cheaply-filtered candidate
crops/windows — never for player/ball detection, which stays RF-DETR.

⚠️ Model fallback (ADR-15, added 2026-08-27 after a real run hit it): a single hardcoded model
name is not viable -- `gemini-3.6-flash`'s free tier carries a hard 20-REQUESTS-PER-DAY quota
(`quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"`), confirmed by direct API probe
after take 6's identity verification burned through it mid-run. `vlm_cfg["models"]` is therefore
an ORDERED LIST, tried in sequence: a 429 whose JSON body's `error.status` is
`"RESOURCE_EXHAUSTED"` means THIS model's daily budget is gone for the day -- move to the next
model immediately (retrying the same model can never succeed until tomorrow, so don't burn the
backoff delay on it). A transient failure (429 with a DIFFERENT status, or 500/502/503/504) still
retries the SAME model with backoff first, exactly as before -- only after that model's retries
are exhausted (transient or otherwise) does the next model in the list get tried. A genuine
non-retryable client error (400/401/403/404 -- e.g. a model retired for new callers) also advances
to the next model rather than failing the whole call outright. Only when EVERY model in the list
has failed does this return an error (never a fabricated success).
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import requests

from src.common.logging import get_logger

logger = get_logger(__name__)

# Retried ON THE SAME MODEL: a transient server-side failure, worth another attempt before giving
# up on that model. NOT retried (advances to the next model in the fallback list immediately): a
# genuine client error (bad key, bad request, a retired model) that a retry can never fix, and a
# 429 that turns out to be RESOURCE_EXHAUSTED (see _daily_quota_exhausted).
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _extract_text(response_json: dict) -> str:
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _daily_quota_exhausted(resp: requests.Response) -> bool:
    """True for a 429 whose JSON body's `error.status` is `"RESOURCE_EXHAUSTED"` -- Gemini's own
    signal for "this model's quota is gone until it resets", as opposed to a transient overload
    (which also 429s sometimes, but with a different `status`, e.g. `"UNAVAILABLE"`)."""
    if resp.status_code != 429:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    return body.get("error", {}).get("status") == "RESOURCE_EXHAUSTED"


def call_gemini_vision(
    prompt: str, image_bgr: np.ndarray, api_key: str, vlm_cfg: dict
) -> tuple[str | None, str | None, str | None]:
    """POST one prompt + one BGR image to Gemini `generateContent`, trying each model in
    `vlm_cfg["models"]` in order (see module docstring for the fallback rules).

    Returns `(text, error, model_used)`: on success `error is None`, `text` is the model's raw
    response text, and `model_used` names which model in the list actually answered (recorded for
    reproducibility/debugging, CLAUDE.md task spec). On failure `text is None`, `model_used is
    None`, and `error` is a human-readable reason covering every model tried — callers must never
    treat a failure as a legitimate "nothing found" answer (Golden Rule 5).
    """
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, vlm_cfg["jpeg_quality"]])
    if not ok:
        return None, "could not JPEG-encode image", None
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

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
    models = vlm_cfg["models"]

    last_error = "no model attempted"
    for model in models:
        url = vlm_cfg["api_url_template"].format(model=model)
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=vlm_cfg["timeout_s"])
            except requests.RequestException as exc:
                last_error = f"model={model}: {type(exc).__name__}: {exc}"
                logger.warning(
                    "gemini[%s] attempt %d/%d raised %s -- retrying",
                    model,
                    attempt + 1,
                    max_retries,
                    last_error,
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff_base * (2**attempt))
                continue

            if resp.status_code == 200:
                return _extract_text(resp.json()), None, model

            if _daily_quota_exhausted(resp):
                last_error = f"model={model}: daily quota exhausted (429 RESOURCE_EXHAUSTED)"
                logger.warning(
                    "gemini[%s] daily quota exhausted -- falling back to the next model", model
                )
                break  # never retry the same model on a quota exhaustion -- won't recover today

            last_error = f"model={model}: HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                logger.warning(
                    "gemini[%s] non-retryable error (%s) -- falling back to the next model",
                    model,
                    last_error,
                )
                break  # e.g. 404 model retired, 401/403 auth -- retrying this model won't help

            logger.warning(
                "gemini[%s] attempt %d/%d failed (%s) -- retrying",
                model,
                attempt + 1,
                max_retries,
                last_error,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2**attempt))
        # this model's attempts are exhausted (transient retries used up, quota hit, or a
        # non-retryable error) -- the outer loop advances to the next model in the list.

    return None, f"all {len(models)} fallback model(s) failed, last error: {last_error}", None
