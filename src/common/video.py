"""ffmpeg/ffprobe-based decode + encode helpers (CLAUDE.md §5 Stage 0, ADR-1: PyAV/ffmpeg, no
`decord`).

Bulk decode goes through an `ffmpeg` subprocess piping raw frames (NVDEC when available, CPU
fallback otherwise) rather than PyAV, matching the measured environment in CLAUDE.md §11.1
(`h264_cuvid` / `h264_nvenc` / `-hwaccel cuda` all confirmed present).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .logging import get_logger

logger = get_logger(__name__)

_BGR_CHANNELS = 3  # inherent to the bgr24 pixel format, not a tunable


def _parse_rate(rate: str) -> float:
    """Parse an ffprobe frame-rate string (``"30/1"`` or ``"29.97"``) to a float."""
    if "/" in rate:
        num, den = rate.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def probe(path: str | Path) -> dict:
    """Probe a video file's primary video stream via `ffprobe`.

    Returns a dict with ``width``, ``height``, ``fps``, ``duration`` (seconds), ``nb_frames``,
    and ``codec``.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream = data["streams"][0]
    fmt = data.get("format", {})

    fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    duration_raw = stream.get("duration") or fmt.get("duration")
    duration = float(duration_raw) if duration_raw not in (None, "N/A") else 0.0

    nb_frames_raw = stream.get("nb_frames")
    nb_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "N/A") else None
    if nb_frames is None and duration and fps:
        nb_frames = round(duration * fps)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
        "codec": stream.get("codec_name"),
    }


def decode_frames(
    path: str | Path,
    fps: float | None = None,
    start: float | None = None,
    end: float | None = None,
    scale_width: int | None = None,
    use_nvdec: bool = True,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode frames from `path` via an `ffmpeg` subprocess piping raw BGR24 frames.

    Yields ``(frame_index, t, frame)`` where ``frame`` is an ``HxWx3`` uint8 BGR numpy array.
    Tries NVDEC (``-hwaccel cuda -c:v h264_cuvid``) first when `use_nvdec` is set, and
    automatically falls back to CPU decode (logging the fallback) if the hardware path produces
    no frames.
    """
    meta = probe(path)
    width, height = meta["width"], meta["height"]
    out_width, out_height = width, height
    if scale_width and scale_width < width:
        out_width = scale_width
        out_height = (round(height * (scale_width / width)) // 2) * 2  # keep even for yuv

    def build_cmd(nvdec: bool) -> list[str]:
        cmd = ["ffmpeg", "-v", "error"]
        if nvdec:
            cmd += ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        if start is not None:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(path)]
        if end is not None:
            cmd += ["-t", str(end - (start or 0.0))]
        vf = []
        if fps:
            vf.append(f"fps={fps}")
        if out_width != width:
            vf.append(f"scale={out_width}:{out_height}")
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return cmd

    frame_bytes = out_width * out_height * _BGR_CHANNELS

    proc = subprocess.Popen(build_cmd(use_nvdec), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_chunk = proc.stdout.read(frame_bytes)

    if use_nvdec and not first_chunk:
        proc.stdout.close()
        proc.wait()
        stderr = proc.stderr.read().decode(errors="replace")
        logger.warning(
            "NVDEC decode produced no frames for %s, falling back to CPU decode: %s",
            path,
            stderr.strip()[:500],
        )
        proc = subprocess.Popen(build_cmd(False), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        first_chunk = proc.stdout.read(frame_bytes)

    effective_fps = fps or meta["fps"] or 1.0
    index = 0
    chunk = first_chunk
    try:
        while chunk and len(chunk) == frame_bytes:
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(out_height, out_width, 3)
            t = (start or 0.0) + index / effective_fps
            yield index, t, frame
            index += 1
            chunk = proc.stdout.read(frame_bytes)
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()


def extract_clip(
    src: str | Path,
    dst: str | Path,
    t_start: float,
    t_end: float,
    encoder: str = "h264_nvenc",
) -> Path:
    """Cut ``[t_start, t_end)`` from `src` into `dst` via `ffmpeg`.

    Uses the hardware encoder `encoder` (default ``h264_nvenc``, NVENC per CLAUDE.md §11.1) and
    falls back to the CPU encoder ``libx264`` (logging the fallback) if it fails.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = t_end - t_start

    def build_cmd(enc: str) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            str(t_start),
            "-i",
            str(src),
            "-t",
            str(duration),
            "-c:v",
            enc,
            "-c:a",
            "aac",
            str(dst),
        ]

    result = subprocess.run(build_cmd(encoder), capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "encoder %s failed for %s, falling back to libx264: %s",
            encoder,
            dst,
            result.stderr.strip()[:500],
        )
        result = subprocess.run(build_cmd("libx264"), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract clip {dst}: {result.stderr}")
    return dst


def write_video(
    frames_iter: Iterable[np.ndarray],
    dst: str | Path,
    fps: float,
    size: tuple[int, int],
) -> Path:
    """Write a sequence of BGR uint8 frames to `dst` (mp4, libx264) via an `ffmpeg` subprocess.

    `size` is ``(width, height)``; every frame in `frames_iter` must already match it. Used for
    annotated overlay video artifacts (CLAUDE.md §10: "annotated overlay video artifact" per stage).
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for frame in frames_iter:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            n += 1
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed writing {dst}: {stderr}")
    logger.info("wrote %d frame(s) -> %s", n, dst)
    return dst
