"""ADR-15 — full-resolution annotated original-video renderer (CLAUDE.md §13.1).

Generalizes `scripts/overlay_target.py` (magenta-target/dim-others/ball-marker/live-counters/
CUT-banner debug overlay for ONE arrow-selected target on a single-take-or-few-takes clip) into a
renderer that covers a WHOLE multi-take video with PER-TAKE verified identity gating: a take with
`identity_status == "verified"` gets a thick RED box on its own verified track-id set, labeled
with THAT take's own verified jersey number; every other detected player renders as a thin GREEN
box with its track id; a take with no verified identity gets no red box at all, plus a visible
"IDENTITY: unverified in this segment" note next to its CUT banner (CLAUDE.md §13.1).

Streaming, not buffering: unlike `src.common.video.write_video`'s hardware-encoder branch (which
buffers every frame in Python memory before encoding — fine for a short debug clip, but ~7000
native-4K frames here would be on the order of 150+ GB of raw uint8, an easy OOM), this module
pipes frames to `ffmpeg`/NVENC one at a time via stdin exactly like `write_video`'s own libx264
fast path already does — see `_stream_encode`.
"""

from __future__ import annotations

import subprocess
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from src.common.logging import get_logger
from src.common.types import BallDetection, Event, Take, Track
from src.common.video import decode_frames, probe
from src.common.viz import draw_ball
from src.identity.verify import TakeIdentityResult
from src.track.tracker import assign_take_id

logger = get_logger("annotated_video")

# Native-4K-scaled drawing constants (overlay_target.py's own constants were tuned for its 960px
# preview output; this renders at native resolution, ~4x larger, so line/font sizes scale up).
_RED = (0, 0, 255)  # BGR
_GREEN = (0, 200, 0)
_RED_THICKNESS = 6
_GREEN_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE_TARGET = 1.6
_LABEL_FONT_SCALE_OTHER = 0.9
_LABEL_THICKNESS_TARGET = 4
_LABEL_THICKNESS_OTHER = 2

_PANEL_MARGIN_PX = 24
_PANEL_PADDING_PX = 20
_PANEL_LINE_HEIGHT_PX = 46
_PANEL_BG_COLOR = (15, 15, 15)
_PANEL_ALPHA = 0.65
_PANEL_TEXT_COLOR = (255, 255, 255)
_PANEL_HEADER_COLOR = (0, 255, 255)
_PANEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_FONT_SCALE = 1.0
_PANEL_FONT_THICKNESS = 2

_BANNER_HEIGHT_PX = 90
_BANNER_BG_COLOR = (20, 20, 20)
_BANNER_TEXT_COLOR = (255, 255, 255)
_BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_FONT_SCALE = 1.4
_BANNER_FONT_THICKNESS = 3
_BANNER_SECONDS = 2.0  # how long the CUT banner stays on screen after a take starts

# Live-panel categories, matching statcard.md's own summary lines (CLAUDE.md §13.2) so the
# overlay never presents a number the stat card itself doesn't also report (Golden Rule 5).
_PANEL_EVENT_TYPES = ("touch", "pass", "sprint", "shot", "tackle", "save", "dribble")
_PANEL_LABELS = {
    "touch": "Touches",
    "pass": "Passes",
    "sprint": "Sprints",
    "shot": "Shots",
    "tackle": "Tackles",
    "save": "Saves",
    "dribble": "Dribbles",
}


def _nearest_by_time(items: list, t: float, tolerance: float, key):
    """Bisect-based nearest-in-time lookup (identical technique to
    `scripts/overlay_target.py::_nearest`) — O(log n) per call instead of a linear scan. Kept for
    one-off lookups and its existing unit tests; the render hot loop below uses
    `_SortedTimeIndex` instead, which sorts ONCE rather than on every one of ~7000 frame queries.
    """
    if not items:
        return None
    items_sorted = sorted(items, key=key)
    keys = [key(it) for it in items_sorted]
    pos = bisect_left(keys, t)
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(items_sorted)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(keys[i] - t))
    return items_sorted[best] if abs(keys[best] - t) <= tolerance else None


class _SortedTimeIndex:
    """Presorted-once nearest-in-time index — the SAME lookup as `_nearest_by_time`, but sorting
    happens once at construction instead of on every query. Matters here because the render loop
    below queries `ball_detections` (~7000 items for a whole 234s video) and every track's own box
    list once PER RENDERED FRAME (~7000 native frames) — re-sorting on every call, as a naive
    `_nearest_by_time(items, ...)` would, turns an O(log n) lookup into an O(n log n) one repeated
    thousands of times over, which measurably dominates render time at this frame count.
    """

    def __init__(self, items: list, key) -> None:
        self._items = sorted(items, key=key)
        self._keys = [key(it) for it in self._items]

    def nearest(self, t: float, tolerance: float):
        if not self._items:
            return None
        pos = bisect_left(self._keys, t)
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(self._items)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self._keys[i] - t))
        return self._items[best] if abs(self._keys[best] - t) <= tolerance else None


def red_box_track_ids(identity: TakeIdentityResult | None) -> set[int]:
    """The verified-identity GATE (CLAUDE.md §13.1): which raw track ids get the thick RED "target"
    box this take. Empty for `None`/an unverified take -- those takes get green boxes for every
    detected player and nothing else, never a fallback guess. Pure, so the render-time gate itself
    is unit-testable without decoding a single frame (see `tests/test_annotated_video.py`)."""
    if identity is None or identity.status != "verified":
        return set()
    return set(identity.location_track_ids)


class _NumberProgress:
    """Running per-jersey-number event counts as rendering advances through time — a lightweight,
    render-local pointer over that number's own time-sorted events (advances forward only, never
    recomputed from scratch per frame). Persists across non-contiguous takes verified to the SAME
    number, so a number that reappears later in the video keeps accumulating, not resetting."""

    def __init__(self, events: list[Event]) -> None:
        self.events_sorted = sorted(events, key=lambda e: e.t_start)
        self.pointer = 0
        self.counts: dict[str, int] = {}

    def advance_to(self, t: float) -> dict[str, int]:
        while (
            self.pointer < len(self.events_sorted) and self.events_sorted[self.pointer].t_start <= t
        ):
            ev = self.events_sorted[self.pointer]
            self.counts[ev.type.value] = self.counts.get(ev.type.value, 0) + 1
            self.pointer += 1
        return self.counts


def _draw_box_with_label(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: tuple[int, int, int],
    box_thickness: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(frame, p1, p2, color, box_thickness)
    (tw, th), baseline = cv2.getTextSize(label, _LABEL_FONT, font_scale, font_thickness)
    label_y = max(th + 10, p1[1] - 10)
    cv2.rectangle(
        frame, (p1[0], label_y - th - 10), (p1[0] + tw + 16, label_y + baseline), color, -1
    )
    cv2.putText(
        frame,
        label,
        (p1[0] + 8, label_y),
        _LABEL_FONT,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )


def _draw_live_panel(
    frame: np.ndarray,
    take_id: int | None,
    identity: TakeIdentityResult | None,
    counts: dict[str, int] | None,
) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if identity is not None and identity.status == "verified":
        lines.append((f"TARGET #{identity.jersey_number} -- VERIFIED", _PANEL_HEADER_COLOR))
        for etype in _PANEL_EVENT_TYPES:
            lines.append(
                (f"{_PANEL_LABELS[etype]}: {(counts or {}).get(etype, 0)}", _PANEL_TEXT_COLOR)
            )
        lines.append(("Goals: not available", _PANEL_TEXT_COLOR))
        lines.append(("Assists: not available", _PANEL_TEXT_COLOR))
    else:
        lines.append(("IDENTITY: unverified in this segment", _PANEL_HEADER_COLOR))
        lines.append(("No target player statistics for this segment", _PANEL_TEXT_COLOR))

    max_text_w = max(
        cv2.getTextSize(text, _PANEL_FONT, _PANEL_FONT_SCALE, _PANEL_FONT_THICKNESS)[0][0]
        for text, _color in lines
    )
    panel_w = max_text_w + 2 * _PANEL_PADDING_PX
    panel_h = _PANEL_PADDING_PX * 2 + len(lines) * _PANEL_LINE_HEIGHT_PX
    frame_h, frame_w = frame.shape[:2]
    x0 = frame_w - _PANEL_MARGIN_PX - panel_w
    y0 = _PANEL_MARGIN_PX

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0, dst=frame)

    for i, (text, color) in enumerate(lines):
        y = y0 + _PANEL_PADDING_PX + (i + 1) * _PANEL_LINE_HEIGHT_PX - 12
        cv2.putText(
            frame,
            text,
            (x0 + _PANEL_PADDING_PX, y),
            _PANEL_FONT,
            _PANEL_FONT_SCALE,
            color,
            _PANEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def _draw_cut_banner(frame: np.ndarray, take_id: int, verified: bool) -> None:
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, _BANNER_HEIGHT_PX), _BANNER_BG_COLOR, -1)
    text = f"CUT -- take {take_id}"
    if not verified:
        text += "   |   IDENTITY: unverified in this segment"
    cv2.putText(
        frame,
        text,
        (24, int(_BANNER_HEIGHT_PX * 0.65)),
        _BANNER_FONT,
        _BANNER_FONT_SCALE,
        _BANNER_TEXT_COLOR,
        _BANNER_FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _stream_encode(
    frames_iter, dst: Path, fps: float, size: tuple[int, int], encoder: str = "h264_nvenc"
) -> Path:
    """Stream BGR frames to `dst` via ffmpeg, never buffering the whole video in Python memory
    (see module docstring for why `src.common.video.write_video`'s hardware path is unsafe here).
    """
    width, height = size
    preset = ["-preset", "p4"] if encoder == "h264_nvenc" else ["-preset", "veryfast"]
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
        encoder,
        *preset,
        str(dst),
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
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
        if encoder != "libx264":
            logger.warning(
                "streaming encode with %s failed (%s) -- this will need a re-render with "
                "libx264, which is expensive; raising rather than silently redoing the whole "
                "decode+draw pass",
                encoder,
                stderr.strip()[:500],
            )
        raise RuntimeError(f"ffmpeg failed writing {dst} with encoder={encoder}: {stderr}")
    logger.info("streamed %d frame(s) -> %s (encoder=%s)", n, dst, encoder)
    return dst


def mux_original_audio(rendered_video_only: Path, original_video: Path, out_path: Path) -> Path:
    """Mux `original_video`'s own audio track onto `rendered_video_only`'s video stream (CLAUDE.md
    §13.1: "original audio preserved"). `?` on the audio map makes it optional so a source with no
    audio stream at all still produces a valid (silent) output rather than erroring."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(original_video),
        "-i",
        str(rendered_video_only),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed muxing audio into {out_path}: {result.stderr}")
    logger.info("muxed original audio -> %s", out_path)
    return out_path


def render_full_annotated_video(
    video_path: str | Path,
    work_dir: Path,
    final_out_path: Path,
    takes: list[Take],
    tracks: list[Track],
    ball_detections: list[BallDetection],
    identity_by_take: dict[int, TakeIdentityResult],
    events_by_number: dict[int, list[Event]],
    detect_frame_width: float,
    detect_frame_height: float,
    use_nvdec: bool = True,
) -> Path:
    """Render CLAUDE.md §13.1's primary output for one filename-less video: the WHOLE input, at
    native resolution/fps, red-boxed only during verified takes, green everywhere else, with a
    live stats panel and CUT banners -- then mux the original audio back in.
    """
    video_path = Path(video_path)
    native_meta = probe(video_path)
    native_w, native_h, native_fps = native_meta["width"], native_meta["height"], native_meta["fps"]
    scale_x = native_w / detect_frame_width if detect_frame_width else 1.0
    scale_y = native_h / detect_frame_height if detect_frame_height else 1.0

    tracks_by_take: dict[int, list[Track]] = {}
    for tr in tracks:
        tracks_by_take.setdefault(tr.take_id, []).append(tr)
    take_by_id = {tk.id: tk for tk in takes}

    # Presort ONCE (see _SortedTimeIndex docstring) -- the naive per-call `_nearest_by_time` would
    # otherwise re-sort ball_detections (~7000 items for a whole 234s video) and every track's own
    # box list on EVERY one of ~7000 rendered native frames.
    #
    # ⚠️ Keyed by (take_id, track_id), NOT track_id alone: raw Track.id RESETS per take (Golden
    # Rule 3), so a flat `{track_id: index}` dict silently collides two different takes' same-
    # numbered tracks -- whichever is built last in this dict comprehension wins, corrupting the
    # lookup for nearly every box in every earlier take. Caught for real on this exact video (a
    # full render produced ZERO green boxes anywhere -- every `.nearest()` call was silently
    # querying the wrong take's track).
    ball_index = _SortedTimeIndex(ball_detections, key=lambda b: b.t)
    box_index_by_track: dict[tuple[int, int], _SortedTimeIndex] = {
        (tr.take_id, tr.id): _SortedTimeIndex(tr.boxes, key=lambda b: b.t) for tr in tracks
    }

    progress_by_number: dict[int, _NumberProgress] = {
        number: _NumberProgress(events) for number, events in events_by_number.items()
    }

    video_only_path = work_dir / "debug" / "original_annotated_video_only.mp4"

    def frame_generator():
        for _idx, t, raw_frame in decode_frames(
            video_path, fps=None, scale_width=None, use_nvdec=use_nvdec
        ):
            # decode_frames yields frames built via np.frombuffer(chunk, ...) -- a READ-ONLY view
            # into the ffmpeg pipe's own bytes buffer. Every cv2 draw call below mutates in place,
            # so this MUST be a writable copy (same convention scripts/overlay_target.py already
            # follows via its own `annotated = frame.copy()`) or cv2.rectangle raises "img marked
            # as output argument, but provided NumPy array marked as readonly" (caught for real on
            # this exact video, CLAUDE.md task spec).
            frame = raw_frame.copy()
            take_id = assign_take_id(t, takes)
            take = take_by_id.get(take_id) if take_id is not None else None
            identity = identity_by_take.get(take_id) if take_id is not None else None
            take_tracks = tracks_by_take.get(take_id, [])
            target_ids = red_box_track_ids(identity)

            for tr in take_tracks:
                box = box_index_by_track[(tr.take_id, tr.id)].nearest(t, 0.3)
                if box is None:
                    continue
                x1, y1 = box.bbox.x1 * scale_x, box.bbox.y1 * scale_y
                x2, y2 = box.bbox.x2 * scale_x, box.bbox.y2 * scale_y
                if tr.id in target_ids:
                    _draw_box_with_label(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"#{identity.jersey_number} | TARGET",
                        _RED,
                        _RED_THICKNESS,
                        _LABEL_FONT_SCALE_TARGET,
                        _LABEL_THICKNESS_TARGET,
                    )
                else:
                    _draw_box_with_label(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"#{tr.id}",
                        _GREEN,
                        _GREEN_THICKNESS,
                        _LABEL_FONT_SCALE_OTHER,
                        _LABEL_THICKNESS_OTHER,
                    )

            ball = ball_index.nearest(t, 0.2)
            if ball is not None:
                scaled_bbox = ball.bbox.model_copy(
                    update={
                        "x1": ball.bbox.x1 * scale_x,
                        "y1": ball.bbox.y1 * scale_y,
                        "x2": ball.bbox.x2 * scale_x,
                        "y2": ball.bbox.y2 * scale_y,
                    }
                )
                draw_ball(frame, scaled_bbox, ball.conf, ball.interpolated)

            counts = None
            if identity is not None and identity.status == "verified":
                counts = progress_by_number[identity.jersey_number].advance_to(t)
            _draw_live_panel(frame, take_id, identity, counts)

            if take is not None and t < take.t_start + _BANNER_SECONDS:
                _draw_cut_banner(
                    frame, take.id, identity is not None and identity.status == "verified"
                )

            yield frame

    try:
        _stream_encode(
            frame_generator(),
            video_only_path,
            native_fps,
            (native_w, native_h),
            encoder="h264_nvenc",
        )
    except RuntimeError:
        logger.warning("falling back to libx264 for the full annotated render (nvenc failed)")
        _stream_encode(
            frame_generator(), video_only_path, native_fps, (native_w, native_h), encoder="libx264"
        )

    return mux_original_audio(video_only_path, video_path, final_out_path)
