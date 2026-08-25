"""Shared pure helpers for Stage 4's threshold-based event segmentation.

Both `src/events/sprints.py` (player speed) and `src/events/shots.py` (ball speed) reduce a
per-sample speed series to merged, minimum-duration time segments the same way — factored out
once here so the merge-gap / min-duration boundary semantics aren't duplicated (CLAUDE.md §10: no
duplicated magic-number logic) and so both are covered by one focused, GPU-free unit test file
(`tests/test_events.py`).
"""

from __future__ import annotations


def moving_average(values: list[float], window: int) -> list[float]:
    """Trailing moving average with `window` >= 1.

    Returns a list the SAME LENGTH as `values` (no NaNs to propagate downstream): the first
    `window - 1` output samples average however many input values are actually available so far,
    rather than being undefined. `window <= 1` (or an empty input) is a no-op copy.
    """
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    running_sum = 0.0
    for i, v in enumerate(values):
        running_sum += v
        lo = max(0, i - window + 1)
        if i >= window:
            running_sum -= values[i - window]
        out.append(running_sum / (i + 1 - lo))
    return out


def find_threshold_segments(
    times: list[float],
    values: list[float],
    threshold: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    """Turn a per-sample `(times[i], values[i])` series into merged, minimum-duration segments.

    Three pure steps, in order:

    1. **Raw runs.** Find maximal runs of consecutive samples with `value >= threshold` (a sample
       exactly AT the threshold qualifies — inclusive).
    2. **Merge.** Two runs are merged into one when the gap between the first run's end time and
       the second run's start time is STRICTLY LESS than `merge_gap_s` — a gap exactly EQUAL to
       `merge_gap_s` is NOT merged (this task's spec: "under" the gap threshold). Merging chains,
       so three runs each `merge_gap_s / 2` apart all end up in one segment.
    3. **Filter.** Keep only merged segments whose duration (`end - start`) is `>= min_duration_s`
       (a segment exactly AT `min_duration_s` DOES qualify — "at least", inclusive).

    Pure function over two equal-length parallel lists — no `Track`/`BallDetection` dependency,
    so callers and unit tests can both feed it synthetic series directly. `times` must be sorted
    ascending (both real callers already produce/consume time-ordered series).
    """
    if len(times) != len(values):
        raise ValueError("times and values must be the same length")
    if not times:
        return []

    raw_index_runs: list[list[int]] = []
    current: list[int] = []
    for i, v in enumerate(values):
        if v >= threshold:
            current.append(i)
        elif current:
            raw_index_runs.append(current)
            current = []
    if current:
        raw_index_runs.append(current)
    if not raw_index_runs:
        return []

    runs = [(times[idxs[0]], times[idxs[-1]]) for idxs in raw_index_runs]

    merged: list[list[float]] = [list(runs[0])]
    for start, end in runs[1:]:
        gap = start - merged[-1][1]
        if gap < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if (e - s) >= min_duration_s]
