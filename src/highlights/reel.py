"""Stage 6 — reel assembly (CLAUDE.md §5 Stage 6): trims ranked clips to the export budget, then
concatenates them into `output/<slug>/reel.mp4` via ffmpeg's concat demuxer.

Caption burn-in (event type + confidence + "UNCALIBRATED" where applicable) is called out in
`configs/highlights.yaml`/the task spec as a nice-to-have, not a hard requirement. **It is
deliberately SKIPPED here** — the `drawtext` filter's per-clip variable text would need a bespoke
filter graph assembled per segment (variable clip count, variable caption strings), which is
real additional surface area for tonight's actual deliverable (a correct, playable reel); every
clip's own metadata is already fully traceable via `stat_card.json`/`run_report.json`
(Golden Rule 5), so no traceability is lost by skipping the burn-in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.common.logging import get_logger
from src.common.types import Clip

logger = get_logger(__name__)


def select_for_export(clips_with_scores: list[tuple[Clip, float]], export_cfg: dict) -> list[Clip]:
    """Best-first `clips_with_scores` -> the final reel's clip list, honouring
    `configs/highlights.yaml: export.max_clips`/`target_reel_seconds`.

    `target_reel_seconds` (when not `null`) trims the LOWEST-ranked tail clips: since the input is
    already best-first, this simply stops adding once the next clip would exceed the budget
    (never reorders to "better pack" the budget — a lower-ranked clip must never bump a
    higher-ranked one just because it fits more snugly).
    """
    ranked = sorted(clips_with_scores, key=lambda pair: pair[1], reverse=True)
    clips = [c for c, _ in ranked][: export_cfg["max_clips"]]

    target = export_cfg.get("target_reel_seconds")
    if not target:
        return clips

    kept: list[Clip] = []
    total = 0.0
    for c in clips:
        duration = c.t_end - c.t_start
        if kept and total + duration > target:
            break
        kept.append(c)
        total += duration
    return kept


def build_reel(clips: list[Clip], out_path: str | Path) -> Path:
    """Concatenate `clips` (in the order given — caller is responsible for best-first ordering)
    into `out_path` via ffmpeg's concat demuxer.

    Tries a fast `-c copy` remux first (valid since every clip already came from the same
    `src.common.video.extract_clip` encoding path); falls back to a full re-encode if the
    segments' codec parameters don't line up exactly (e.g. one clip fell back to libx264 while
    another used NVENC, per `extract_clip`'s own per-call fallback).
    """
    if not clips:
        raise ValueError("no clips to concatenate into a reel")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_path = out_path.parent / f"{out_path.stem}_concat_list.txt"
    with list_path.open("w") as f:
        for c in clips:
            if c.path is None:
                raise ValueError(f"Clip for event {c.event_id} has no cut file path")
            f.write(f"file '{Path(c.path).resolve()}'\n")

    def run_concat(extra_args: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            *extra_args,
            str(out_path),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    result = run_concat(["-c", "copy"])
    if result.returncode != 0:
        logger.warning(
            "reel concat -c copy failed for %s, falling back to re-encode: %s",
            out_path,
            result.stderr.strip()[:500],
        )
        result = run_concat(["-c:v", "libx264", "-c:a", "aac"])
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to build reel {out_path}: {result.stderr}")

    logger.info("wrote reel (%d clip(s)) -> %s", len(clips), out_path)
    return out_path
