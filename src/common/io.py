"""Resumable artifact cache + serialization helpers (CLAUDE.md §10: cache every stage's output).

Every stage writes its output to `work/<video-slug>/...` and reads it back on the next run instead
of recomputing, keyed by a hash of the config that produced it. Every cache hit/miss is logged.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import yaml
from pydantic import BaseModel

from .logging import get_logger

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _slugify(text: str) -> str:
    """Lowercase, replace runs of non-alphanumerics with ``_``, strip leading/trailing ``_``."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "video"


def work_dir_for(video_path: str | Path, root: str | Path = "work") -> Path:
    """Return (and create) the resumable working directory for ``video_path`` under ``root``.

    e.g. ``input/clip1 43.mp4`` -> ``work/clip1_43``.
    """
    slug = _slugify(Path(video_path).stem)
    d = Path(root) / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert pydantic models (and containers of them) to JSON-safe primitives."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    """Serialize ``obj`` (a pydantic model, or a dict/list of them) to ``path`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, default=str))
    logger.debug("saved json -> %s", path)


def load_json(path: str | Path) -> Any:
    """Load and parse a JSON file written by :func:`save_json`."""
    return json.loads(Path(path).read_text())


def load_yaml(path: str | Path) -> dict:
    """Load a `configs/*.yaml` file into a plain dict (CLAUDE.md §10: one config per stage).

    Every stage loads its config through this one helper so YAML parsing stays in a single
    place; callers pass the resulting dict into stage functions rather than each module
    re-reading/re-parsing YAML itself (keeps stage logic unit-testable with synthetic configs).
    """
    return yaml.safe_load(Path(path).read_text())


def save_models_parquet(models: list[BaseModel], path: str | Path) -> None:
    """Write a list of pydantic models to a parquet file (one row per model)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [m.model_dump(mode="json") for m in models]
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    df.to_parquet(path, index=False)
    logger.info("saved %d row(s) -> %s", len(models), path)


def load_models_parquet(path: str | Path, model_cls: type[ModelT]) -> list[ModelT]:
    """Read a parquet file written by :func:`save_models_parquet` back into pydantic models."""
    df = pd.read_parquet(path)
    records = df.to_dict(orient="records")
    return [model_cls.model_validate(r) for r in records]


def cache_exists(path: str | Path) -> bool:
    """True if an artifact file exists at ``path``."""
    return Path(path).exists()


def config_hash(config: dict) -> str:
    """A short, deterministic hash of a config dict (sha256 of canonical JSON, 12 hex chars)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class StageCache:
    """Resumable disk cache for one pipeline stage's output artifact.

    Recompute is skipped when the artifact at ``path`` exists AND its sidecar ``<name>.meta.json``
    records a config hash matching the current config (CLAUDE.md §10). Every hit/miss is logged.
    """

    def __init__(self, path: str | Path, config: dict, stage: str | None = None) -> None:
        self.path = Path(path)
        self.config = config
        self.stage = stage or self.path.stem
        self.hash = config_hash(config)
        self._logger = get_logger(f"cache.{self.stage}")

    @property
    def meta_path(self) -> Path:
        """Path to the sidecar metadata file recording this artifact's config hash."""
        return self.path.parent / f"{self.path.name}.meta.json"

    def is_valid(self) -> bool:
        """True if the artifact exists and its recorded config hash matches the current config."""
        if not cache_exists(self.path) or not self.meta_path.exists():
            return False
        try:
            meta = load_json(self.meta_path)
        except (json.JSONDecodeError, OSError):
            return False
        return meta.get("config_hash") == self.hash

    def hit(self) -> bool:
        """Check validity, logging a cache HIT or MISS, and return whether it was a hit."""
        valid = self.is_valid()
        if valid:
            self._logger.info("cache HIT  stage=%s path=%s", self.stage, self.path)
        else:
            self._logger.info("cache MISS stage=%s path=%s", self.stage, self.path)
        return valid

    def write_meta(self) -> None:
        """Record the current config hash alongside a freshly-written artifact."""
        save_json({"config_hash": self.hash, "stage": self.stage}, self.meta_path)

    def get_or_compute(
        self,
        compute_fn: Callable[[], Any],
        save_fn: Callable[[Any, Path], None],
        load_fn: Callable[[Path], Any],
    ) -> Any:
        """Return the cached artifact if valid, else compute, save, stamp metadata, and return."""
        if self.hit():
            return load_fn(self.path)
        result = compute_fn()
        save_fn(result, self.path)
        self.write_meta()
        return result


def stage_cache(
    path_fn: Callable[..., str | Path],
    config_fn: Callable[..., dict],
    save_fn: Callable[[Any, Path], None],
    load_fn: Callable[[Path], Any],
) -> Callable:
    """Decorator form of :class:`StageCache` for simple, single-artifact stage functions.

    ``path_fn``/``config_fn`` are called with the wrapped function's arguments to derive the
    artifact path and the config dict to hash; ``save_fn``/``load_fn`` do the (de)serialization.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = Path(path_fn(*args, **kwargs))
            config = config_fn(*args, **kwargs)
            cache = StageCache(path, config, stage=fn.__name__)
            return cache.get_or_compute(lambda: fn(*args, **kwargs), save_fn, load_fn)

        return wrapper

    return decorator
