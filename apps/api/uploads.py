"""Chunked video upload -- the one implementation behind both `apps/api/main.py` (on the GPU pod)
and `apps/gateway/main.py` (on the always-on VPS). Same wire contract on both, so the gateway
can also *replay* a file to the pod with the very same chunk protocol it accepted it with.

Why chunks at all: a hosted deployment may sit behind a proxy that caps one request body (the
Cloudflare free plan at 100 MB), and a 4K clip is routinely several hundred MB (docs/DEPLOY.md).
The browser (`apps/web/js/app.js` UPLOAD_CHUNK_BYTES) slices; `ChunkedUploads.receive` reassembles
in order, refusing anything out of order or unknown rather than silently appending.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import HTTPException

ALLOWED_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# Partial uploads live under the input dir so they end up on the same (persistent) volume as the
# final file and the rename at the end is atomic; the leading dot keeps listings from showing them.
UPLOAD_PART_DIRNAME = ".uploads"
# What the browser sends per slice; the gateway reuses it when replaying to the pod.
UPLOAD_CHUNK_BYTES = 24 * 1024 * 1024


def safe_upload_name(filename: str | None) -> str:
    """Basename only, spaces -> underscores, and a video extension we can actually decode --
    anything else is a 400, not a silently-accepted junk file in `input/`."""
    name = Path(filename or f"upload_{int(time.time())}.mp4").name.replace(" ", "_")
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    if Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported video type {Path(name).suffix!r}; use one of {sorted(ALLOWED_UPLOAD_SUFFIXES)}",
        )
    return name


def upload_response(target_path: Path) -> dict[str, Any]:
    return {
        "status": "success",
        "filename": target_path.name,
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
        "path": str(target_path),
    }


class ChunkedUploads:
    """In-memory progress per upload id. The servers run as ONE process each (the pod's JOBS and
    in-flight guard already depend on that), so a dict behind a lock is the right store."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def receive(
        self,
        input_dir: Path,
        upload_id: str,
        index: int,
        total: int,
        filename: str,
        chunk: BinaryIO,
    ) -> dict[str, Any]:
        """Append one sequential slice. The final slice's response carries the same payload as a
        one-shot upload; earlier ones return `{"status": "partial", ...}`."""
        if not UPLOAD_ID_RE.match(upload_id):
            raise HTTPException(status_code=400, detail="invalid upload_id")
        if total < 1 or not 0 <= index < total:
            raise HTTPException(status_code=400, detail="index/total out of range")
        safe_name = safe_upload_name(filename)
        part_dir = input_dir / UPLOAD_PART_DIRNAME
        part_dir.mkdir(parents=True, exist_ok=True)
        part_path = part_dir / f"{upload_id}.part"

        with self._lock:
            state = self._state.get(upload_id)
            if index == 0:
                state = {"next": 0, "filename": safe_name, "total": total}
                self._state[upload_id] = state
            elif state is None:
                raise HTTPException(status_code=404, detail="unknown upload_id (chunk 0 never arrived)")
            if state["next"] != index:
                raise HTTPException(status_code=409, detail=f"expected chunk {state['next']}, got {index}")
            if state["total"] != total or state["filename"] != safe_name:
                raise HTTPException(status_code=409, detail="upload metadata changed mid-stream")

        with open(part_path, "wb" if index == 0 else "ab") as buffer:
            shutil.copyfileobj(chunk, buffer)

        with self._lock:
            state["next"] = index + 1
            complete = state["next"] == total
            if complete:
                self._state.pop(upload_id, None)

        if not complete:
            return {"status": "partial", "upload_id": upload_id, "received": index + 1, "total": total}

        target_path = input_dir / safe_name
        os.replace(part_path, target_path)
        return upload_response(target_path)
