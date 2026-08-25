"""Stage 5/6 debug artifact — target-player tracking + live event-counter overlay (CLAUDE.md §10:
"after each stage: an annotated overlay video"; a variant of `scripts/overlay_tracks.py`, not a
replacement — that script shows every tracked player, this one is about making the SELECTED
target unmistakable plus visualising Stage 6's own numbers as they'd appear live).

Renders `work/<slug>/debug/overlay_target.mp4`: the target player's stitched timeline (from
`work/<slug>/selection.json`, Stage 5) drawn in a thick, high-contrast box with a persistent
"TARGET #<jersey>" label; every other player drawn thin/dim so the target reads unambiguously;
the ball (`ball_detections.parquet`, distinct marker when interpolated, same honesty rule as
`common.viz.draw_ball`); a live counter panel for exactly the event types Phase 1 detects
(`Sprints`, `Shots`, `Goals: N/A`) that flashes the relevant counter while that event's window is
active, with a small progress bar across `[t_start, t_end]`; and a "CUT -> take N" banner at every
take boundary (same pattern as `overlay_tracks.py`).

The counter numbers are read STRAIGHT from `output/<slug>/stat_card.json`'s own `timeline` field
(added by this same task) rather than re-derived here — the overlay must show the same numbers the
stat card reports, never a second, possibly-divergent computation (Golden Rule 5).

Golden Rule 7 / CLAUDE.md task spec: NO touch/pass/tackle/assist/save counting anywhere in this
script — the one line acknowledging those categories exist is static text, never a number.

Usage: ``.venv/bin/python scripts/overlay_target.py "input/clip2 77.mp4"``

Import note: this module's pure logic (`running_counts_at`, `active_event_types_at`,
`active_event_progress`, `find_active_take`) is imported directly by
`tests/test_overlay_target.py` with NO GPU/video I/O — the heavy pipeline stack (torch/rfdetr via
`src.pipeline.run`) is therefore imported lazily, inside `render_overlay`, not at module level, so
importing this module for its pure functions stays cheap (mirrors `src/team/classifier.py`'s own
lazy-`import torch` pattern).
"""

from __future__ import annotations

import re
import sys
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import typer  # noqa: E402

from src.common.io import load_json, load_models_parquet, work_dir_for  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.types import BallDetection, Take, Track  # noqa: E402
from src.common.video import decode_frames, write_video  # noqa: E402
from src.common.viz import draw_ball, draw_banner, draw_hud  # noqa: E402
from src.highlights.selection import SelectionResult  # noqa: E402
from src.ingest.discovery import parse_filename  # noqa: E402
from src.track.tracker import assign_take_id  # noqa: E402

logger = get_logger("overlay_target")
app = typer.Typer(add_completion=False)

# Cosmetic/behavioural constants for this debug artifact only (not a pipeline tunable, per
# viz.py's own convention).
OUTPUT_WIDTH_PX = 960
BANNER_SECONDS = 1.0  # how long the "CUT -> take N" banner stays on screen after a take starts

# Exactly the event types Phase 1 actually detects (CLAUDE.md Golden Rule 7) — goals are reported
# separately as a static "N/A" line (§3.2(3): no scoreboard anywhere in this footage), and
# touches/passes/tackles/assists/saves are never counted at all, anywhere, by design.
COUNTED_EVENT_TYPES: tuple[str, ...] = ("sprint", "shot")
UNMEASURED_CATEGORIES_LINE = "Touches/Passes/Tackles/Assists/Saves: not detected in Phase 1"

_TARGET_BOX_COLOR = (255, 0, 255)  # magenta (BGR) — high-contrast against grass/kit colours
_TARGET_BOX_THICKNESS = 4
_TARGET_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TARGET_LABEL_FONT_SCALE = 0.7
_TARGET_LABEL_FONT_THICKNESS = 2
_TARGET_LABEL_PAD_PX = 4

_DIM_BOX_COLOR = (90, 90, 90)  # dim grey — every non-target player
_DIM_BOX_THICKNESS = 1
_DIM_LABEL_FONT_SCALE = 0.4

_PANEL_MARGIN_PX = 10
_PANEL_PADDING_PX = 8
_PANEL_WIDTH_PX = 380
_PANEL_LINE_HEIGHT_PX = 24
_PANEL_BG_COLOR = (20, 20, 20)
_PANEL_ALPHA = 0.6
_PANEL_TEXT_COLOR = (255, 255, 255)
_PANEL_FLASH_TEXT_COLOR = (0, 255, 255)  # bright cyan/yellow — an active event's counter line
_PANEL_FLASH_BG_COLOR = (0, 110, 110)
_PANEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_FONT_SCALE = 0.55
_PANEL_FONT_THICKNESS = 1
_PANEL_PROGRESS_BG_COLOR = (90, 90, 90)
_PANEL_PROGRESS_BAR_WIDTH_PX = 220
_PANEL_PROGRESS_BAR_HEIGHT_PX = 4

# Which counter panel line (index into the `lines` list built in `_draw_counter_panel`) each
# counted event type flashes — kept as one explicit mapping rather than positional guessing.
_LINE_INDEX_BY_EVENT_TYPE = {"sprint": 1, "shot": 2}


# ---------------------------------------------------------------------------
# pure logic (unit-tested in tests/test_overlay_target.py, no video I/O)
# ---------------------------------------------------------------------------


def running_counts_at(t: float, timeline: list[dict]) -> dict[str, int]:
    """Cumulative count of each Phase-1-countable event type (`sprint`, `shot` only — Golden
    Rule 7) whose window has STARTED by time `t` (`t_start <= t`) — what a live viewer watching
    the video would see the on-screen counter tick up to. `timeline` is `stat_card.json`'s own
    `timeline` field (or any list of dicts with `t_start`/`type` keys): the SAME numbers the stat
    card reports, never a separately re-derived figure (Golden Rule 5).
    """
    counts = {etype: 0 for etype in COUNTED_EVENT_TYPES}
    for row in timeline:
        etype = row["type"]
        if etype in counts and row["t_start"] <= t:
            counts[etype] += 1
    return counts


def active_event_types_at(t: float, timeline: list[dict]) -> set[str]:
    """Event types whose window (`t_start <= t <= t_end`) currently covers `t` — used to flash
    that type's counter line and draw its progress marker."""
    return {
        row["type"]
        for row in timeline
        if row["type"] in COUNTED_EVENT_TYPES and row["t_start"] <= t <= row["t_end"]
    }


def active_event_progress(t: float, timeline: list[dict]) -> dict[str, float]:
    """Fraction (0..1) elapsed through each currently-active event's `[t_start, t_end]` window —
    what the progress bar under a flashing counter line renders. A zero-duration window reads as
    fully (1.0) elapsed rather than dividing by zero. Empty when nothing is active at `t`."""
    progress: dict[str, float] = {}
    for row in timeline:
        etype = row["type"]
        if etype not in COUNTED_EVENT_TYPES:
            continue
        t0, t1 = row["t_start"], row["t_end"]
        if t0 <= t <= t1:
            span = t1 - t0
            progress[etype] = (t - t0) / span if span > 0 else 1.0
    return progress


def find_active_take(t: float, takes: list[Take]) -> Take | None:
    """The `Take` whose `[t_start, t_end)` covers timestamp `t`, or `None` — a thin, pure wrapper
    around `assign_take_id` that returns the `Take` object itself (what the overlay needs both to
    pick the right take's target-track-id set and to draw the take-boundary banner)."""
    take_id = assign_take_id(t, takes)
    if take_id is None:
        return None
    return next((tk for tk in takes if tk.id == take_id), None)


def _nearest(items: list, t: float, tolerance: float, key):
    """Return whichever item in `items` is closest to `t` by `key(item)`, within `tolerance`
    seconds, or `None` (same pattern as `scripts/overlay_tracks.py::_nearest`)."""
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


# ---------------------------------------------------------------------------
# drawing (uses common.viz where it already fits; local helpers for the new visual grammar this
# script needs — a thick/dim player-box distinction and a semi-transparent counter panel — that
# viz.py has no equivalent for yet)
# ---------------------------------------------------------------------------


def _draw_target_box(frame, bbox, jersey: int | None):
    out = frame
    p1 = (int(bbox.x1), int(bbox.y1))
    p2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(out, p1, p2, _TARGET_BOX_COLOR, _TARGET_BOX_THICKNESS)
    label = f"TARGET #{jersey}" if jersey is not None else "TARGET"
    (tw, th), baseline = cv2.getTextSize(
        label, _TARGET_LABEL_FONT, _TARGET_LABEL_FONT_SCALE, _TARGET_LABEL_FONT_THICKNESS
    )
    label_y = max(th + _TARGET_LABEL_PAD_PX, p1[1] - _TARGET_LABEL_PAD_PX)
    cv2.rectangle(
        out,
        (p1[0], label_y - th - _TARGET_LABEL_PAD_PX),
        (p1[0] + tw + 2 * _TARGET_LABEL_PAD_PX, label_y + baseline),
        _TARGET_BOX_COLOR,
        -1,
    )
    cv2.putText(
        out,
        label,
        (p1[0] + _TARGET_LABEL_PAD_PX, label_y),
        _TARGET_LABEL_FONT,
        _TARGET_LABEL_FONT_SCALE,
        (0, 0, 0),
        _TARGET_LABEL_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return out


def _draw_dim_box(frame, bbox, track_id: int):
    out = frame
    p1 = (int(bbox.x1), int(bbox.y1))
    p2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(out, p1, p2, _DIM_BOX_COLOR, _DIM_BOX_THICKNESS)
    cv2.putText(
        out,
        f"#{track_id}",
        (p1[0], max(10, p1[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        _DIM_LABEL_FONT_SCALE,
        _DIM_BOX_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_counter_panel(
    frame,
    counts: dict[str, int],
    active_types: set[str],
    progress: dict[str, float],
    jersey: int | None,
    selection_label: str,
):
    # Anchored top-RIGHT (task spec allows either corner) so it never collides with
    # `common.viz.draw_hud`'s own frame/t/stage text, which this script also draws top-left
    # (same pattern as scripts/overlay_tracks.py) -- the two must not overlap.
    out = frame
    frame_width = out.shape[1]
    header = f"TARGET #{jersey}" if jersey is not None else "TARGET (jersey unknown)"
    lines = [
        f"{header} -- {selection_label}",
        f"Sprints: {counts.get('sprint', 0)}",
        f"Shots: {counts.get('shot', 0)}",
        "Goals: N/A (no scoreboard)",
        UNMEASURED_CATEGORIES_LINE,
    ]
    panel_h = _PANEL_PADDING_PX * 2 + len(lines) * _PANEL_LINE_HEIGHT_PX
    # Measured, not assumed (the static "not detected in Phase 1" disclaimer line is the widest
    # one and does not fit `_PANEL_WIDTH_PX` at this font -- MEASURED 457px vs a 380px minimum):
    # size the panel to whichever is wider, the configured minimum or the longest actual line.
    max_text_w = max(
        cv2.getTextSize(line, _PANEL_FONT, _PANEL_FONT_SCALE, _PANEL_FONT_THICKNESS)[0][0]
        for line in lines
    )
    panel_w = max(_PANEL_WIDTH_PX, max_text_w + 2 * _PANEL_PADDING_PX)
    panel_x0 = frame_width - _PANEL_MARGIN_PX - panel_w
    panel_y0 = _PANEL_MARGIN_PX

    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (panel_x0, panel_y0),
        (panel_x0 + panel_w, panel_y0 + panel_h),
        _PANEL_BG_COLOR,
        -1,
    )
    out = cv2.addWeighted(overlay, _PANEL_ALPHA, out, 1 - _PANEL_ALPHA, 0)

    line_index_by_type = {v: k for k, v in _LINE_INDEX_BY_EVENT_TYPE.items()}
    for i, line in enumerate(lines):
        baseline_y = panel_y0 + _PANEL_PADDING_PX + (i + 1) * _PANEL_LINE_HEIGHT_PX - 8
        etype = line_index_by_type.get(i)
        flashing = etype is not None and etype in active_types
        if flashing:
            cv2.rectangle(
                out,
                (panel_x0 + 3, baseline_y - _PANEL_LINE_HEIGHT_PX + 10),
                (panel_x0 + panel_w - 3, baseline_y + 4),
                _PANEL_FLASH_BG_COLOR,
                -1,
            )
        color = _PANEL_FLASH_TEXT_COLOR if flashing else _PANEL_TEXT_COLOR
        cv2.putText(
            out,
            line,
            (panel_x0 + _PANEL_PADDING_PX, baseline_y),
            _PANEL_FONT,
            _PANEL_FONT_SCALE,
            color,
            _PANEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        if flashing and etype in progress:
            bar_x0 = panel_x0 + _PANEL_PADDING_PX
            bar_y0 = baseline_y + 5
            filled_w = int(_PANEL_PROGRESS_BAR_WIDTH_PX * progress[etype])
            cv2.rectangle(
                out,
                (bar_x0, bar_y0),
                (bar_x0 + _PANEL_PROGRESS_BAR_WIDTH_PX, bar_y0 + _PANEL_PROGRESS_BAR_HEIGHT_PX),
                _PANEL_PROGRESS_BG_COLOR,
                -1,
            )
            cv2.rectangle(
                out,
                (bar_x0, bar_y0),
                (bar_x0 + filled_w, bar_y0 + _PANEL_PROGRESS_BAR_HEIGHT_PX),
                _PANEL_FLASH_TEXT_COLOR,
                -1,
            )
    return out


# ---------------------------------------------------------------------------
# rendering entrypoint (heavy imports deferred to here — see module docstring)
# ---------------------------------------------------------------------------


def render_overlay(
    video_path: Path,
    work_root: str | Path = "work",
    output_root: str | Path = "output",
) -> Path:
    # Lazy heavy import: only this function needs the full Stage 0.5-6 stack (torch/rfdetr via
    # src.pipeline.run -> src.track.run -> src.detect.*). Keeping it out of module scope means
    # `import scripts.overlay_target` for the pure functions above (tests/test_overlay_target.py)
    # never pulls torch — same reasoning as src/team/classifier.py's own lazy `import torch`.
    from src.pipeline.run import _load_all_configs, run_pipeline_for_video

    video_path = Path(video_path)
    configs = _load_all_configs()
    pattern = re.compile(configs["run"]["filename_convention_regex"])
    _clip_index, target_jersey = parse_filename(video_path.stem, pattern)

    # Self-sufficient like overlay_tracks.py: (re)runs the full pipeline if any artifact this
    # script needs is missing/stale; a cheap no-op on a cache hit (CLAUDE.md §10).
    run_pipeline_for_video(
        video_path, configs, target_jersey, work_root=work_root, output_root=output_root
    )

    work_dir = work_dir_for(video_path, root=work_root)
    output_dir = Path(output_root) / work_dir.name

    takes = load_models_parquet(work_dir / "shots" / "takes.parquet", Take)
    tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    ball_detections = load_models_parquet(
        work_dir / "detect" / "ball_detections.parquet", BallDetection
    )
    selection = SelectionResult.model_validate(load_json(work_dir / "selection.json"))
    stat_card = load_json(output_dir / "stat_card.json")
    timeline = stat_card["timeline"]

    target_track_ids_by_take: dict[int, set[int]] = {
        ts.take_id: set(ts.track_ids) for ts in selection.takes
    }
    method = selection.takes[0].method if len(selection.takes) == 1 else "mixed_per_take"
    selection_label = f"{method} conf={selection.overall_confidence:.2f}"

    hardware_config = configs["hardware"]
    fps_sample = hardware_config["stages"]["track"]["fps_sample"]
    scale_width = hardware_config["decode"]["scale_width"]
    tolerance = 0.5 / fps_sample if fps_sample > 0 else 0.5

    out_width = OUTPUT_WIDTH_PX
    out_height = None
    frames_out = []
    n_frames = 0

    for _idx, t, frame in decode_frames(
        video_path, fps=fps_sample, scale_width=scale_width, use_nvdec=True
    ):
        take = find_active_take(t, takes)
        take_id = take.id if take is not None else None
        target_ids = target_track_ids_by_take.get(take_id, set())

        annotated = frame.copy()
        for tr in tracks:
            if tr.take_id != take_id:
                continue
            box = _nearest(tr.boxes, t, tolerance, key=lambda b: b.t)
            if box is None:
                continue
            if tr.id in target_ids:
                annotated = _draw_target_box(annotated, box.bbox, target_jersey)
            else:
                annotated = _draw_dim_box(annotated, box.bbox, tr.id)

        ball = _nearest(ball_detections, t, tolerance, key=lambda b: b.t)
        if ball is not None:
            annotated = draw_ball(annotated, ball.bbox, ball.conf, ball.interpolated)

        counts = running_counts_at(t, timeline)
        active_types = active_event_types_at(t, timeline)
        progress = active_event_progress(t, timeline)
        annotated = _draw_counter_panel(
            annotated, counts, active_types, progress, target_jersey, selection_label
        )

        if take is not None and take.id > 0 and take.t_start <= t < take.t_start + BANNER_SECONDS:
            annotated = draw_banner(annotated, f"CUT -> take {take.id}")

        annotated = draw_hud(annotated, _idx, t, stage="target")

        height, width = annotated.shape[:2]
        if out_height is None:
            out_height = (round(height * (out_width / width)) // 2) * 2

        resized = cv2.resize(annotated, (out_width, out_height), interpolation=cv2.INTER_AREA)
        frames_out.append(resized)
        n_frames += 1

    if not frames_out:
        raise RuntimeError(f"no frames decoded from {video_path}; cannot render overlay")

    out_path = work_dir / "debug" / "overlay_target.mp4"
    write_video(
        iter(frames_out),
        out_path,
        fps=fps_sample,
        size=(out_width, out_height),
        encoder="h264_nvenc",
    )
    logger.info(
        "wrote %d frame(s) -> %s (target jersey=%s, selection=%s)",
        n_frames,
        out_path,
        target_jersey,
        selection_label,
    )
    return out_path


@app.command()
def main(video: Path = typer.Argument(..., help="Path to a video in input/")) -> None:
    out_path = render_overlay(video)
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
