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

import math
import subprocess
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np

from src.common.logging import get_logger
from src.common.types import BallDetection, BBox, Event, EventType, Take, Track
from src.common.video import decode_frames, probe
from src.common.viz import draw_ball
from src.goal.detect import GoalStructure, goal_bbox_at, take_motion_score
from src.identity.verify import TakeIdentityResult
from src.pipeline.player_output import _TIMELINE_EVENT_TYPES, _key_moment_label
from src.track.continuity import build_take_identities
from src.track.tracker import assign_take_id

logger = get_logger("annotated_video")

# Drawing constants below are all expressed at a 4K REFERENCE WIDTH and scaled per frame by
# `_overlay_scale` (see below) -- never used raw.
#
# Owner-reported bug, root-caused 2026-09-01: they used to be applied raw, with the note "this
# renders at native resolution, ~4x larger, so line/font sizes scale up". That silently assumed
# every input is 4K. It is not: the real broadcast inputs (`video2`/`video3`) are 1280x720, i.e.
# 3x narrower, so the whole overlay rendered ~3x oversized -- the red TARGET label alone spanned
# most of the frame width and occluded the play it was supposed to annotate (confirmed by
# extracting real frames at t=58s/59s). Scaling by the frame's own width fixes this generically
# for ANY input resolution rather than per-video, which is the standing requirement.
_REFERENCE_WIDTH_PX = 3840  # the width every size constant in this module was tuned at
_MIN_OVERLAY_SCALE = 0.30  # floor so overlays stay legible on small/SD inputs instead of
# collapsing to sub-pixel lines and unreadable 0.2-scale text.


def _overlay_scale(frame_width: int) -> float:
    """Per-frame multiplier for every size constant in this module, so the overlay is
    proportional to the actual video rather than assuming 4K."""
    return max(_MIN_OVERLAY_SCALE, frame_width / _REFERENCE_WIDTH_PX)


def _px(value: float, scale: float, minimum: int = 1) -> int:
    """Scale a pixel-valued constant, never below `minimum` (a 0px line draws nothing)."""
    return max(minimum, int(round(value * scale)))


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
_PANEL_EVENT_TYPES = (
    "touch",
    "pass",
    "sprint",
    "shot",
    "tackle",
    "save",
    "dribble",
    "goal",
    "assist",
)
_PANEL_LABELS = {
    "touch": "Touches",
    "pass": "Passes",
    "sprint": "Sprints",
    "shot": "Shots",
    "tackle": "Tackles",
    "save": "Saves",
    "dribble": "Dribbles",
    "goal": "Goals",
    "assist": "Assists",
}

# Event captions (ADR-19/20, CLAUDE.md §13.1): "a brief on-screen caption naming the event, the
# jersey number, and the timestamp" -- burned in at every event's own instant regardless of
# whether that event's player could be confidently red-boxed this frame (ADR-19's own
# "caption-only, no red box" rule for a manual-mode association that couldn't be made
# confidently), so the annotated video stays faithful to the underlying event stream even when
# tracking/association is weak.
_CAPTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAPTION_FONT_SCALE = 1.1
_CAPTION_FONT_THICKNESS = 3
_CAPTION_TEXT_COLOR = (255, 255, 255)
_CAPTION_BG_COLOR = (40, 40, 40)
_CAPTION_PADDING_PX = 12
_CAPTION_BOTTOM_MARGIN_PX = 30
_CAPTION_LINE_HEIGHT_PX = 56
_CAPTION_DURATION_S = 2.5  # how long ONE event's caption stays on screen after its own t_start --
# long enough to be readable at a glance, short enough that two close-together events (e.g. a
# touch immediately followed by a pass) don't visually smear into one incoherent caption.

# ADR-20/21: human-marked goal-mouth region overlay (`configs/goal_region.yaml`) — deliberately a
# THIRD colour, distinct from both the target's red and other-players' green, so a viewer never
# mistakes a goal-mouth polygon for a player box. Drawn on every frame (cheap: a handful of
# polylines/label calls, no per-frame recomputation of the polygon geometry itself).
_GOAL_REGION_COLOR = (255, 255, 0)  # BGR cyan
_GOAL_REGION_THICKNESS = 3
_GOAL_REGION_LABEL = "GOAL REGION"
_GOAL_REGION_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GOAL_REGION_LABEL_FONT_SCALE = 0.9
_GOAL_REGION_LABEL_FONT_THICKNESS = 2


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


class _ActiveEventIndex:
    """Pure, render-local index answering "which (jersey_number, Event) pairs have an active
    caption window at time `t`" — presorted ONCE by `Event.t_start` (same `_SortedTimeIndex`-style
    "sort once, bisect per query" discipline as the ball/track box indices above, needed for the
    same reason: this is queried once per rendered native frame, ~7000 times for a whole video).

    An event is "active" for `[t_start, t_start + caption_duration_s]` — captions are keyed off
    when the event STARTED, not `t_end` (a `TOUCH`/`GOAL`/... event's own `t_start`==`t_end` in the
    common case anyway; a longer interval event like `POSSESSION` isn't a template timeline row
    at all, see `src.pipeline.player_output._TIMELINE_EVENT_TYPES`'s own docstring).
    """

    def __init__(self, events_by_number: dict[int, list[Event]], caption_duration_s: float) -> None:
        entries = [
            (ev.t_start, number, ev) for number, events in events_by_number.items() for ev in events
        ]
        entries.sort(key=lambda e: e[0])
        self._starts = [e[0] for e in entries]
        self._entries = entries
        self._caption_duration_s = caption_duration_s

    def active_at(self, t: float) -> list[tuple[int, Event]]:
        """Every `(jersey_number, Event)` whose own `[t_start, t_start + caption_duration_s]`
        window contains `t`, oldest-started first."""
        lo = bisect_left(self._starts, t - self._caption_duration_s)
        hi = bisect_right(self._starts, t)
        return [(number, ev) for _start, number, ev in self._entries[lo:hi]]


def _format_caption_timestamp(t: float) -> str:
    """`M:SS` rendering — identical convention to
    `src.pipeline.player_output._format_timestamp`'s whole-minutes/seconds split, just without the
    tenths-of-a-second decimal (a caption is read at a glance, not audited to the frame)."""
    minutes = int(t // 60)
    seconds = int(t % 60)
    return f"{minutes}:{seconds:02d}"


def _event_caption_label(event: Event) -> str:
    """The SAME label a viewer would find on that event's `statcard.md` timeline row — reuses
    `src.pipeline.player_output`'s own `_TIMELINE_EVENT_TYPES`/`_key_moment_label` verbatim
    (Golden Rule 5: the overlay must never show a label the stat card itself wouldn't also use)."""
    if event.type == EventType.KEY_MOMENT:
        return _key_moment_label(event)
    return _TIMELINE_EVENT_TYPES.get(event.type, event.type.value.replace("_", " ").title())


def _draw_event_caption(frame: np.ndarray, active_events: list[tuple[int, Event]]) -> None:
    """Draw one bottom-centred caption line per currently-active `(jersey_number, Event)` pair
    (CLAUDE.md §13.1: "a brief on-screen caption naming the event, the jersey number, and the
    timestamp"), stacked upward so simultaneous captions (rare, but possible with two players
    annotated at nearly the same instant) never overlap. A no-op when nothing is active."""
    if not active_events:
        return
    frame_h, frame_w = frame.shape[:2]
    sc = _overlay_scale(frame_w)
    font_scale = _CAPTION_FONT_SCALE * sc
    thickness = _px(_CAPTION_FONT_THICKNESS, sc)
    pad = _px(_CAPTION_PADDING_PX, sc)
    for i, (number, ev) in enumerate(active_events):
        text = f"#{number} | {_event_caption_label(ev)} | {_format_caption_timestamp(ev.t_start)}"
        (tw, th), baseline = cv2.getTextSize(text, _CAPTION_FONT, font_scale, thickness)
        x = (frame_w - tw) // 2
        y = frame_h - _px(_CAPTION_BOTTOM_MARGIN_PX, sc) - i * _px(_CAPTION_LINE_HEIGHT_PX, sc)
        cv2.rectangle(
            frame,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad // 2),
            _CAPTION_BG_COLOR,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            _CAPTION_FONT,
            font_scale,
            _CAPTION_TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )


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
    scale = _overlay_scale(frame.shape[1])
    pad_y, pad_x, text_x = _px(10, scale), _px(16, scale), _px(8, scale)
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(frame, p1, p2, color, box_thickness)
    (tw, th), baseline = cv2.getTextSize(label, _LABEL_FONT, font_scale, font_thickness)
    label_y = max(th + pad_y, p1[1] - pad_y)
    cv2.rectangle(
        frame, (p1[0], label_y - th - pad_y), (p1[0] + tw + pad_x, label_y + baseline), color, -1
    )
    cv2.putText(
        frame,
        label,
        (p1[0] + text_x, label_y),
        _LABEL_FONT,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )


def _target_panel_header(identity: TakeIdentityResult) -> str:
    """The panel's own headline for a verified take -- owner-reported bug fix, 2026-08-31:
    "VERIFIED" used to be shown unconditionally, even when the drawn BOX itself was only a
    kit-colour pick among many identically-dressed teammates (no real digit read backing it) --
    confirmed on real footage to have silently boxed the WRONG player while still claiming
    "VERIFIED" (Golden Rule 5: never display more certainty than the evidence supports). Split out
    as its own pure function so this exact text decision is directly unit-testable without a real
    frame/cv2 draw call -- see `TakeIdentityResult.association_confirmed_by_jersey`'s own docstring
    for the full story.
    """
    if identity.association_confirmed_by_jersey:
        return f"TARGET #{identity.jersey_number} -- VERIFIED"
    return f"TARGET #{identity.jersey_number} -- COLOUR MATCH (jersey unconfirmed)"


def _draw_live_panel(
    frame: np.ndarray,
    take_id: int | None,
    identity: TakeIdentityResult | None,
    counts: dict[str, int] | None,
) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if identity is not None and identity.status == "verified":
        lines.append((_target_panel_header(identity), _PANEL_HEADER_COLOR))
        for etype in _PANEL_EVENT_TYPES:
            lines.append(
                (f"{_PANEL_LABELS[etype]}: {(counts or {}).get(etype, 0)}", _PANEL_TEXT_COLOR)
            )
    else:
        lines.append(("IDENTITY: unverified in this segment", _PANEL_HEADER_COLOR))
        lines.append(("No target player statistics for this segment", _PANEL_TEXT_COLOR))

    frame_h, frame_w = frame.shape[:2]
    sc = _overlay_scale(frame_w)
    font_scale = _PANEL_FONT_SCALE * sc
    thickness = _px(_PANEL_FONT_THICKNESS, sc)
    padding = _px(_PANEL_PADDING_PX, sc)
    line_height = _px(_PANEL_LINE_HEIGHT_PX, sc)
    margin = _px(_PANEL_MARGIN_PX, sc)

    max_text_w = max(
        cv2.getTextSize(text, _PANEL_FONT, font_scale, thickness)[0][0] for text, _color in lines
    )
    panel_w = max_text_w + 2 * padding
    panel_h = padding * 2 + len(lines) * line_height
    x0 = frame_w - margin - panel_w
    y0 = margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, _PANEL_ALPHA, frame, 1 - _PANEL_ALPHA, 0, dst=frame)

    for i, (text, color) in enumerate(lines):
        y = y0 + padding + (i + 1) * line_height - _px(12, sc)
        cv2.putText(
            frame,
            text,
            (x0 + padding, y),
            _PANEL_FONT,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _prepare_goal_region_polygons(
    goal_region_cfg: dict | None,
    slug: str,
    detect_frame_width: float,
    detect_frame_height: float,
    scale_x: float,
    scale_y: float,
) -> list[np.ndarray]:
    """Resolve `configs/goal_region.yaml`'s own `[0,1]`-fraction-of-detect-frame polygons (see
    that file's header comment — same coordinate space `src.events.goals.detect_goals_for_video`
    already scales into, ADR-20) into native-pixel `cv2.polylines`-ready `int32` arrays for THIS
    video's own slug. Empty (⇒ nothing drawn, no placeholder) when no config, no `regions` map, or
    no entry for this slug exists — the common/default case (ADR-20: `regions: {}` until the owner
    actually marks one). A malformed individual polygon is logged and skipped, never crashes the
    whole render (CLAUDE.md §10: one bad config entry must not take down a multi-minute render).
    """
    if not goal_region_cfg:
        return []
    raw_polygons = (goal_region_cfg.get("regions") or {}).get(slug)
    if not raw_polygons:
        return []

    polygons: list[np.ndarray] = []
    for raw_poly in raw_polygons:
        try:
            pts = [
                (
                    int(round(x * detect_frame_width * scale_x)),
                    int(round(y * detect_frame_height * scale_y)),
                )
                for x, y in raw_poly
            ]
            if len(pts) < 3:
                raise ValueError(f"polygon needs >= 3 vertices, got {len(pts)}")
        except (TypeError, ValueError) as exc:
            logger.warning(
                "goal_region: could not parse polygon %r for slug=%s -- skipping it (%s)",
                raw_poly,
                slug,
                exc,
            )
            continue
        polygons.append(np.array(pts, dtype=np.int32).reshape((-1, 1, 2)))
    return polygons


def _draw_goal_regions(frame: np.ndarray, polygons: list[np.ndarray]) -> None:
    """Draw every prepared goal-region polygon (ADR-20/21) — a no-op when `polygons` is empty
    (the default, no region configured for this slug), never a placeholder/"not configured" note
    (Golden Rule 5 spirit: don't manufacture visual noise for an absent feature)."""
    sc = _overlay_scale(frame.shape[1])
    for pts in polygons:
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=_GOAL_REGION_COLOR,
            thickness=_px(_GOAL_REGION_THICKNESS, sc),
        )
        label_x, label_y = int(pts[0][0][0]), int(pts[0][0][1])
        cv2.putText(
            frame,
            _GOAL_REGION_LABEL,
            (label_x, max(label_y - _px(10, sc), _px(20, sc))),
            _GOAL_REGION_LABEL_FONT,
            _GOAL_REGION_LABEL_FONT_SCALE * sc,
            _GOAL_REGION_COLOR,
            _px(_GOAL_REGION_LABEL_FONT_THICKNESS, sc),
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Stage E ("streamed-gathering-treehouse" plan) -- draw Stage C's own AUTO-TRACKED goal structure,
# not just the static manual configs/goal_region.yaml polygon (still drawn as a fallback, see
# render_full_annotated_video's own priority rule below).
# ---------------------------------------------------------------------------


def _tracked_goal_bboxes_for_take(
    goal_structure: GoalStructure,
    video_path: str | Path,
    take: Take,
    hardware_cfg: dict,
    profile_cfg: dict,
    goal_structure_cfg: dict,
    use_nvdec: bool = True,
) -> list[tuple[float, BBox]]:
    """`[(anchor_t, native_bbox), ...]` across `take` -- the SAME anchor-timestamp-generation +
    `src.goal.detect.goal_bbox_at` query pattern as `src.events.goals._tracked_polygons_for_take`,
    independently duplicated here per CLAUDE.md §10's own "stages independently runnable/
    debuggable" convention (same precedent as `_effective_frame_size`'s own verbatim copy between
    `src/pipeline/run.py` and `extended_output.py`). Returns NATIVE pixel boxes directly -- unlike
    `goals.py`'s own detect-pixel-fraction conversion, this renderer already decodes at native
    resolution (`scale_width=None` below), so no scaling is needed at all.
    """
    tracking_cfg = goal_structure_cfg["tracking"]
    motion_cfg = profile_cfg["motion"]
    motion_score = take_motion_score(video_path, take, hardware_cfg, profile_cfg, use_nvdec)
    interval = (
        tracking_cfg["min_reestimate_interval_s"]
        if motion_score > tracking_cfg["reestimate_motion_score_threshold"]
        else tracking_cfg["max_reestimate_interval_s"]
    )
    duration = max(take.t_end - take.t_start, 1e-3)
    n_anchors = max(1, min(10, math.ceil(duration / interval)))
    anchor_ts = [take.t_start + duration * (i + 0.5) / n_anchors for i in range(n_anchors)]
    return [
        (
            t,
            goal_bbox_at(
                goal_structure,
                video_path,
                t,
                take,
                motion_score,
                tracking_cfg,
                motion_cfg,
                use_nvdec,
            ),
        )
        for t in anchor_ts
    ]


def _nearest_tracked_bbox(anchors: list[tuple[float, BBox]], t: float) -> BBox | None:
    """Whichever precomputed anchor sits closest in time to `t` -- a cheap in-memory lookup (no
    I/O), never a fresh `goal_bbox_at` call per rendered frame (that would mean a fresh decode +
    affine fit for every one of a multi-minute render's thousands of frames)."""
    if not anchors:
        return None
    return min(anchors, key=lambda item: abs(item[0] - t))[1]


def _bbox_to_polygon(bbox: BBox) -> np.ndarray:
    """A `BBox` as a `cv2.polylines`-ready 4-corner `int32` array -- the same shape
    `_prepare_goal_region_polygons` already produces for the manual polygon path, so
    `_draw_goal_regions` draws either source identically."""
    pts = [
        (int(round(bbox.x1)), int(round(bbox.y1))),
        (int(round(bbox.x2)), int(round(bbox.y1))),
        (int(round(bbox.x2)), int(round(bbox.y2))),
        (int(round(bbox.x1)), int(round(bbox.y2))),
    ]
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _draw_cut_banner(frame: np.ndarray, take_id: int, verified: bool) -> None:
    width = frame.shape[1]
    sc = _overlay_scale(width)
    banner_h = _px(_BANNER_HEIGHT_PX, sc)
    cv2.rectangle(frame, (0, 0), (width, banner_h), _BANNER_BG_COLOR, -1)
    text = f"CUT -- take {take_id}"
    if not verified:
        text += "   |   IDENTITY: unverified in this segment"
    cv2.putText(
        frame,
        text,
        (_px(24, sc), int(banner_h * 0.65)),
        _BANNER_FONT,
        _BANNER_FONT_SCALE * sc,
        _BANNER_TEXT_COLOR,
        _px(_BANNER_FONT_THICKNESS, sc),
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
    goal_region_cfg: dict | None = None,
    goal_structures_by_take: dict[int, GoalStructure] | None = None,
    goal_structure_cfg: dict | None = None,
    hardware_cfg: dict | None = None,
    profile_cfg: dict | None = None,
    selection_cfg: dict | None = None,
) -> Path:
    """Render CLAUDE.md §13.1's primary output for one video: the WHOLE input, at native
    resolution/fps, red-boxed only during known-identity takes, green everywhere else, with a
    live stats panel and CUT banners -- then mux the original audio back in.

    `goal_region_cfg` (ADR-20/21, optional) is `configs/goal_region.yaml` verbatim -- when this
    video's own slug (`work_dir.name`) has one or more marked polygons, they're drawn as a thin
    cyan outline on every frame; absent for every one of this project's clips until the owner
    marks one (`_prepare_goal_region_polygons` returns `[]` and nothing is drawn -- Golden Rule 5:
    no placeholder for an unconfigured feature).

    `goal_structures_by_take`/`goal_structure_cfg`/`hardware_cfg`/`profile_cfg` (Stage C/D,
    "streamed-gathering-treehouse" plan) draw Stage C's own AUTO-TRACKED goal box instead, per
    take: a take with a confident `GoalStructure` gets ITS tracked box drawn (never the manual
    polygon at the same time -- one box on screen, matching whichever source
    `src/events/goals.py::detect_goals_for_take` actually used for that take's own goal/assist
    reasoning); a take with none falls back to the manual `goal_region_cfg` polygon exactly as
    before. All four default to `None` -- omitting any of them draws only the manual polygon (or
    nothing), the exact pre-Stage-C/D behaviour, never a crash.
    """
    video_path = Path(video_path)
    native_meta = probe(video_path)
    native_w, native_h, native_fps = native_meta["width"], native_meta["height"], native_meta["fps"]
    scale_x = native_w / detect_frame_width if detect_frame_width else 1.0
    scale_y = native_h / detect_frame_height if detect_frame_height else 1.0
    # Every drawing constant in this module is expressed at a 4K reference width; scale them to
    # THIS video's real size so the overlay is proportional on 720p broadcast as well as on 4K.
    overlay_scale = _overlay_scale(native_w)

    tracks_by_take: dict[int, list[Track]] = {}
    for tr in tracks:
        tracks_by_take.setdefault(tr.take_id, []).append(tr)
    take_by_id = {tk.id: tk for tk in takes}

    # Display the STITCHED identity, not the raw per-fragment Track.id. ByteTrack is motion/IoU
    # only (no appearance model), so a single real player is routinely split across many raw
    # fragments -- MEASURED on work/chelsea_burnley_target10 take 0: 231 raw fragments for the
    # ~13-18 people actually on screen, median fragment life 2.0s, 34% under 1s. Labelling boxes
    # with `Track.id` therefore makes a player's on-screen number visibly churn every couple of
    # seconds, which is exactly what the owner reported. `build_take_identities` already chains
    # those fragments into persistent per-take identities (231 -> 80 on that same take) and is
    # already the basis of the target's own identity lock, so reusing it here costs nothing new
    # and makes every OTHER player's label as stable as the target's already is.
    # NOT a full fix on its own: 80 chains is still ~5x the real head-count, so a label can still
    # change when the stitcher itself loses a player -- honest partial improvement, not a claim of
    # perfect identity (Golden Rule 5).
    display_identity_by_take: dict[int, dict[int, int]] = {}
    if selection_cfg is not None:
        for take_id_key, take_track_list in tracks_by_take.items():
            try:
                identity_of, _conf = build_take_identities(take_track_list, selection_cfg)
            except AssertionError:  # defensive: mixed-take list should be impossible here
                logger.warning("skipping display-identity stitching for take=%s", take_id_key)
                continue
            display_identity_by_take[take_id_key] = identity_of

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
    active_event_index = _ActiveEventIndex(events_by_number, _CAPTION_DURATION_S)

    goal_region_polygons = _prepare_goal_region_polygons(
        goal_region_cfg, work_dir.name, detect_frame_width, detect_frame_height, scale_x, scale_y
    )

    # Stage E: precompute each take's own tracked-goal anchors ONCE, up front -- never a fresh
    # goal_bbox_at call per rendered frame (see _tracked_goal_bboxes_for_take's own docstring for
    # the cost this avoids). A take absent from `goal_structures_by_take` (or missing any of the
    # four new params) simply has no entry here and falls back to the manual polygon at render
    # time below.
    tracked_anchors_by_take: dict[int, list[tuple[float, BBox]]] = {}
    if goal_structures_by_take and goal_structure_cfg and hardware_cfg and profile_cfg:
        for take in takes:
            structure = goal_structures_by_take.get(take.id)
            if structure is None:
                continue
            tracked_anchors_by_take[take.id] = _tracked_goal_bboxes_for_take(
                structure,
                video_path,
                take,
                hardware_cfg,
                profile_cfg,
                goal_structure_cfg,
                use_nvdec,
            )

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
                    # ADR-18 (3): track ID and jersey identity are DIFFERENT concepts (a Track.id
                    # is a within-take tracker artifact that resets at every cut; a jersey number
                    # is a verified identity) -- show BOTH explicitly so no viewer ever reads
                    # "Track ID 10" as if it meant "Jersey #10".
                    _draw_box_with_label(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"#{identity.jersey_number} | TARGET | ID: {tr.id}",
                        _RED,
                        _px(_RED_THICKNESS, overlay_scale),
                        _LABEL_FONT_SCALE_TARGET * overlay_scale,
                        _px(_LABEL_THICKNESS_TARGET, overlay_scale),
                    )
                else:
                    # ADR-18 (3): "ID: {n}" (not a bare "#{n}") -- a bare "#10" visually reads as a
                    # jersey number, which this is NOT (it is only ever a tracker-side id).
                    # The id shown is the STITCHED identity where available (see
                    # `display_identity_by_take` above), falling back to the raw fragment id only
                    # when stitching wasn't computed -- so a player's label stays put across the
                    # short fragment breaks ByteTrack produces constantly on crowded footage.
                    display_id = display_identity_by_take.get(take_id, {}).get(tr.id, tr.id)
                    _draw_box_with_label(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"ID: {display_id}",
                        _GREEN,
                        _px(_GREEN_THICKNESS, overlay_scale),
                        _LABEL_FONT_SCALE_OTHER * overlay_scale,
                        _px(_LABEL_THICKNESS_OTHER, overlay_scale),
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

            # ADR-19/20 (CLAUDE.md §13.1): event captions fire regardless of whether THIS frame's
            # take has a red box at all -- a manual-mode event whose track association couldn't be
            # made confidently still gets its caption (name + jersey + timestamp), never silently
            # dropped just because there's nothing to box.
            _draw_event_caption(frame, active_event_index.active_at(t))

            # Stage E: this take's own AUTO-TRACKED box (Stage C/D) wins over the static manual
            # polygon when one exists for it -- one box on screen, matching whichever source
            # actually fed this take's own goal/assist reasoning. No entry for this take ->
            # fall back to the manual polygon exactly as before Stage C/D existed.
            take_anchors = tracked_anchors_by_take.get(take_id) if take_id is not None else None
            if take_anchors:
                nearest_bbox = _nearest_tracked_bbox(take_anchors, t)
                frame_goal_polygons = [_bbox_to_polygon(nearest_bbox)] if nearest_bbox else []
            else:
                frame_goal_polygons = goal_region_polygons
            _draw_goal_regions(frame, frame_goal_polygons)

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
