"""Stage 0 — find input videos and parse the `clip<N> <jersey>.mp4` filename convention
(CLAUDE.md §3.1).

Filename parsing is pure/testable (:func:`parse_filename`); :func:`find_videos` adds the I/O
(directory scan + `ffprobe`) on top for the CLI/pipeline entrypoints.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.video import probe

logger = get_logger(__name__)


class VideoRef(BaseModel):
    """One discovered input video plus what we could parse/probe about it (CLAUDE.md §3.1)."""

    path: Path
    clip_index: int | None = None
    target_jersey: int | None = None
    probe_info: dict = {}


def parse_filename(stem: str, pattern: re.Pattern[str]) -> tuple[int | None, int | None]:
    """Parse a video filename stem against the `clip<N> <jersey>` convention.

    Returns ``(clip_index, target_jersey)``; both are ``None`` when `stem` doesn't match `pattern`
    (CLAUDE.md §3.1: the jersey number is run metadata only, never fabricated when absent/unparsed).
    """
    match = pattern.match(stem)
    if match is None:
        return None, None
    groups = match.groupdict()
    clip_index = int(groups["n"]) if "n" in groups else None
    target_jersey = int(groups["jersey"]) if "jersey" in groups else None
    return clip_index, target_jersey


def find_videos(
    input_dir: str | Path,
    filename_convention_regex: str,
    video_extensions: list[str],
) -> list[VideoRef]:
    """Scan `input_dir` for videos, parse each filename, and `ffprobe` each match.

    Returns every video found (CLAUDE.md §3.1: "if several videos exist and no explicit choice is
    given, return all — the CLI decides"), sorted by `clip_index` (unparsed filenames sort last,
    then alphabetically). Files that fail to probe are logged and skipped, not silently dropped.
    """
    input_dir = Path(input_dir)
    pattern = re.compile(filename_convention_regex)
    extensions = {ext.lower() for ext in video_extensions}

    if not input_dir.is_dir():
        logger.warning("input dir does not exist: %s", input_dir)
        return []

    refs: list[VideoRef] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        clip_index, target_jersey = parse_filename(path.stem, pattern)
        if clip_index is None:
            logger.warning(
                "%s does not match the filename convention %r; clip_index/target_jersey unset",
                path.name,
                filename_convention_regex,
            )
        try:
            probe_info = probe(path)
        except Exception:
            logger.exception("ffprobe failed for %s, skipping", path)
            continue
        refs.append(
            VideoRef(
                path=path,
                clip_index=clip_index,
                target_jersey=target_jersey,
                probe_info=probe_info,
            )
        )

    refs.sort(key=lambda r: (r.clip_index is None, r.clip_index or 0, str(r.path)))
    return refs
