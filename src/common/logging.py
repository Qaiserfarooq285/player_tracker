"""Console logging + drop-tracking (CLAUDE.md §10: "log everything dropped. Silent filtering =
hidden bugs.").
"""

from __future__ import annotations

import logging
from collections import Counter

from rich.logging import RichHandler

_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single Rich console handler to the root logger, once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger backed by a Rich console handler."""
    _configure_root()
    return logging.getLogger(name)


class DropCounter:
    """Tallies items dropped (replays, close-ups, low-confidence detections, ...) by reason.

    Every stage that filters items should route the drops through one of these so nothing is
    silently discarded (CLAUDE.md §10). ``report()`` logs a summary and feeds
    :class:`src.common.types.RunReport.dropped`.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._counts: Counter[str] = Counter()
        self._logger = get_logger(f"drops.{stage}")

    def drop(self, reason: str, n: int = 1) -> None:
        """Record ``n`` items dropped for ``reason``."""
        self._counts[reason] += n
        self._logger.debug("dropped %d item(s): %s", n, reason)

    @property
    def total(self) -> int:
        """Total number of items dropped across all reasons."""
        return sum(self._counts.values())

    def as_dict(self) -> dict[str, int]:
        """Return the drop tally as a plain ``{reason: count}`` dict."""
        return dict(self._counts)

    def report(self) -> dict[str, int]:
        """Log one summary line per drop reason (CLAUDE.md §10) and return the tally."""
        if not self._counts:
            self._logger.info("[%s] no items dropped", self.stage)
        else:
            for reason, count in self._counts.most_common():
                self._logger.info("[%s] dropped %d item(s): %s", self.stage, count, reason)
        return self.as_dict()
