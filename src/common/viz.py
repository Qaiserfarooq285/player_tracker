"""Thin drawing wrappers over `supervision` annotators for debug/overlay videos.

CLAUDE.md §10 requires an annotated overlay video artifact after every stage — this module is the
shared way stages produce one. Deliberately avoids importing torch (or anything from the
`detect`/`team` extras) so it stays importable in a core-only install.

Cosmetic constants (colours, line/font sizing) are named module-level constants rather than
inline literals; pipeline-meaningful tunables (e.g. pitch dimensions) are caller-supplied
parameters sourced from `configs/*.yaml`, never defaulted here.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from .types import BBox, DetectionClass

# Cosmetic annotator styling — not pipeline tunables, so not YAML-config material, but named
# rather than left as bare literals in the drawing calls below.
_TEAM_COLORS = ["#FF6B6B", "#4D96FF", "#FFD93D", "#95E06C"]
_TEAM_PALETTE = sv.ColorPalette.from_hex(_TEAM_COLORS)
_CLASS_ID_BY_DETECTION_CLASS = {c: i for i, c in enumerate(DetectionClass)}

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT_SCALE = 0.6
_HUD_FONT_THICKNESS = 1
_HUD_TEXT_COLOR = (255, 255, 255)
_HUD_BG_COLOR = (0, 0, 0)
_HUD_MARGIN_PX = 10
_HUD_LINE_HEIGHT_PX = 22

_MINIMAP_BORDER_COLOR = (255, 255, 255)
_MINIMAP_POINT_COLOR = (0, 215, 255)
_MINIMAP_POINT_RADIUS_PX = 3
_MINIMAP_BORDER_THICKNESS_PX = 1

_BALL_COLOR_OBSERVED = (0, 165, 255)  # orange, matches DetectionClass.BALL's viz.py convention
_BALL_COLOR_INTERPOLATED = (0, 215, 255)  # yellow-orange — visually distinct from an observed ball
_BALL_MARKER_RADIUS_PX = 6
_BALL_MARKER_THICKNESS_PX = 2

_ARROW_OUTLINE_COLOR = (60, 60, 220)  # red-ish, echoes the arrow graphic's own colour
_ARROW_OUTLINE_THICKNESS_PX = 2
_ARROW_TIP_COLOR = (0, 255, 0)
_ARROW_TIP_RADIUS_PX = 7

_BANNER_BG_COLOR = (20, 20, 20)
_BANNER_TEXT_COLOR = (255, 255, 255)
_BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_FONT_SCALE = 0.8
_BANNER_FONT_THICKNESS = 2
_BANNER_HEIGHT_PX = 40

_box_annotator = sv.BoxAnnotator()
_label_annotator = sv.LabelAnnotator()


def _detections_to_sv(bboxes: list[BBox], **fields: np.ndarray) -> sv.Detections:
    """Build an `sv.Detections` from our `BBox` list plus arbitrary extra numpy fields."""
    if not bboxes:
        return sv.Detections.empty()
    xyxy = np.array([b.to_xyxy() for b in bboxes], dtype=np.float32)
    return sv.Detections(xyxy=xyxy, **fields)


def draw_detections(
    frame: np.ndarray, detections: list[tuple[BBox, DetectionClass, float]]
) -> np.ndarray:
    """Draw Stage 2 detections (player/goalkeeper/referee/ball) with class + confidence labels.

    `detections` is a list of ``(bbox, cls, conf)``.
    """
    if not detections:
        return frame.copy()
    bboxes = [d[0] for d in detections]
    class_id = np.array([_CLASS_ID_BY_DETECTION_CLASS[d[1]] for d in detections], dtype=int)
    dets = _detections_to_sv(bboxes, class_id=class_id)
    labels = [f"{cls.value} {conf:.2f}" for _, cls, conf in detections]
    annotated = _box_annotator.annotate(scene=frame.copy(), detections=dets)
    return _label_annotator.annotate(scene=annotated, detections=dets, labels=labels)


def draw_tracks(frame: np.ndarray, tracks: list[tuple[int, BBox, int | None]]) -> np.ndarray:
    """Draw current-frame track boxes, coloured by team, labelled with the track ID.

    `tracks` is a list of ``(track_id, bbox, team)`` for the boxes active in this frame; `team`
    of ``None`` renders with the palette's first colour.
    """
    if not tracks:
        return frame.copy()
    bboxes = [t[1] for t in tracks]
    tracker_id = np.array([t[0] for t in tracks], dtype=int)
    team_id = np.array([t[2] if t[2] is not None else 0 for t in tracks], dtype=int)
    dets = _detections_to_sv(bboxes, tracker_id=tracker_id, class_id=team_id)
    team_annotator = sv.BoxAnnotator(color=_TEAM_PALETTE)
    labels = [f"#{tid}" + (f" T{team}" if team is not None else "") for tid, _, team in tracks]
    annotated = team_annotator.annotate(scene=frame.copy(), detections=dets)
    return _label_annotator.annotate(scene=annotated, detections=dets, labels=labels)


def draw_pitch_overlay(
    frame: np.ndarray,
    points_pitch_xy: list[tuple[float, float]],
    pitch_length_m: float,
    pitch_width_m: float,
    origin_px: tuple[int, int],
    size_px: tuple[int, int],
) -> np.ndarray:
    """Draw a small pitch-coordinate minimap (player dots) onto a corner of `frame`.

    `points_pitch_xy` are homography-mapped pitch positions in metres (see
    `PitchHomography.pixel_to_pitch`); `pitch_length_m`/`pitch_width_m` set the minimap's metric
    extent and `origin_px`/`size_px` its pixel placement — pass these from `configs/pitch.yaml`.
    """
    out = frame.copy()
    ox, oy = origin_px
    w, h = size_px
    cv2.rectangle(
        out, (ox, oy), (ox + w, oy + h), _MINIMAP_BORDER_COLOR, _MINIMAP_BORDER_THICKNESS_PX
    )
    for x_m, y_m in points_pitch_xy:
        px = ox + int((x_m / pitch_length_m) * w)
        py = oy + int((y_m / pitch_width_m) * h)
        if ox <= px <= ox + w and oy <= py <= oy + h:
            cv2.circle(out, (px, py), _MINIMAP_POINT_RADIUS_PX, _MINIMAP_POINT_COLOR, -1)
    return out


def draw_hud(frame: np.ndarray, frame_index: int, t: float, stage: str) -> np.ndarray:
    """Draw a small heads-up text block: frame index, timestamp, and pipeline stage name."""
    out = frame.copy()
    lines = [f"frame {frame_index}", f"t={t:.2f}s", f"stage={stage}"]
    for i, line in enumerate(lines):
        y = _HUD_MARGIN_PX + (i + 1) * _HUD_LINE_HEIGHT_PX
        cv2.putText(
            out,
            line,
            (_HUD_MARGIN_PX, y),
            _HUD_FONT,
            _HUD_FONT_SCALE,
            _HUD_BG_COLOR,
            _HUD_FONT_THICKNESS + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (_HUD_MARGIN_PX, y),
            _HUD_FONT,
            _HUD_FONT_SCALE,
            _HUD_TEXT_COLOR,
            _HUD_FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return out


def draw_ball(frame: np.ndarray, bbox: BBox, conf: float, interpolated: bool) -> np.ndarray:
    """Draw a Stage 2 ball detection as a small circle at its bbox centre.

    `interpolated` (CLAUDE.md Golden Rule 5: never present interpolated as observed) picks a
    visually distinct colour so a filled-gap ball position never looks identical to an observed
    one in the overlay.
    """
    out = frame.copy()
    color = _BALL_COLOR_INTERPOLATED if interpolated else _BALL_COLOR_OBSERVED
    center = (int(round(bbox.cx)), int(round(bbox.cy)))
    cv2.circle(out, center, _BALL_MARKER_RADIUS_PX, color, _BALL_MARKER_THICKNESS_PX)
    label = f"ball {conf:.2f}" + (" (interp)" if interpolated else "")
    cv2.putText(
        out,
        label,
        (center[0] + _BALL_MARKER_RADIUS_PX + 2, center[1]),
        _HUD_FONT,
        _HUD_FONT_SCALE,
        color,
        _HUD_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return out


def draw_arrow_outline(frame: np.ndarray, bbox: BBox, tip: tuple[float, float]) -> np.ndarray:
    """Outline a Stage 2 `ArrowHint`'s bbox and mark its tip (CLAUDE.md §3.2 consequence 1)."""
    out = frame.copy()
    p1 = (int(bbox.x1), int(bbox.y1))
    p2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(out, p1, p2, _ARROW_OUTLINE_COLOR, _ARROW_OUTLINE_THICKNESS_PX)
    cv2.circle(out, (int(tip[0]), int(tip[1])), _ARROW_TIP_RADIUS_PX, _ARROW_TIP_COLOR, -1)
    return out


def draw_banner(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a full-width top banner (e.g. "CUT -> take 2") across `frame`."""
    out = frame.copy()
    height, width = out.shape[:2]
    cv2.rectangle(out, (0, 0), (width, _BANNER_HEIGHT_PX), _BANNER_BG_COLOR, -1)
    cv2.putText(
        out,
        text,
        (_HUD_MARGIN_PX, int(_BANNER_HEIGHT_PX * 0.7)),
        _BANNER_FONT,
        _BANNER_FONT_SCALE,
        _BANNER_TEXT_COLOR,
        _BANNER_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return out
