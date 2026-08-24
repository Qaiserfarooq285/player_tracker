"""Tests for src/common/io.py: config_hash, parquet round-trip, StageCache hit/miss."""

from __future__ import annotations

from pathlib import Path

from src.common.io import (
    StageCache,
    cache_exists,
    config_hash,
    load_json,
    load_models_parquet,
    save_json,
    save_models_parquet,
    work_dir_for,
)
from src.common.types import BBox, Detection, DetectionClass

# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def test_config_hash_deterministic():
    cfg = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_order_independent():
    cfg_a = {"a": 1, "b": 2}
    cfg_b = {"b": 2, "a": 1}
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_sensitive_to_value_change():
    cfg_a = {"threshold": 0.5}
    cfg_b = {"threshold": 0.6}
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_length():
    assert len(config_hash({"x": 1})) == 12


# ---------------------------------------------------------------------------
# work_dir_for
# ---------------------------------------------------------------------------


def test_work_dir_for_slugifies_stem(tmp_path: Path):
    root = tmp_path / "work"
    d = work_dir_for("input/clip1 43.mp4", root=root)
    assert d == root / "clip1_43"
    assert d.is_dir()


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------


def test_save_load_json_round_trip(tmp_path: Path):
    path = tmp_path / "nested" / "obj.json"
    save_json({"a": 1, "b": [1, 2, 3]}, path)
    assert load_json(path) == {"a": 1, "b": [1, 2, 3]}


def test_save_json_serializes_pydantic_models(tmp_path: Path):
    det = Detection(
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        cls=DetectionClass.PLAYER,
        conf=0.8,
        frame_index=0,
    )
    path = tmp_path / "det.json"
    save_json(det, path)
    data = load_json(path)
    assert data["cls"] == "player"
    assert data["conf"] == 0.8


# ---------------------------------------------------------------------------
# parquet round-trip
# ---------------------------------------------------------------------------


def test_parquet_round_trip_detections(tmp_path: Path):
    detections = [
        Detection(
            bbox=BBox(x1=float(i), y1=0.0, x2=float(i) + 10.0, y2=10.0),
            cls=DetectionClass.PLAYER if i % 2 == 0 else DetectionClass.BALL,
            conf=0.5 + i * 0.01,
            frame_index=i,
        )
        for i in range(5)
    ]
    path = tmp_path / "detections.parquet"
    save_models_parquet(detections, path)
    assert cache_exists(path)

    restored = load_models_parquet(path, Detection)
    assert len(restored) == len(detections)
    for original, back in zip(detections, restored, strict=True):
        assert back.frame_index == original.frame_index
        assert back.cls == original.cls
        assert back.conf == original.conf
        assert back.bbox.x1 == original.bbox.x1


def test_cache_exists_false_for_missing_path(tmp_path: Path):
    assert not cache_exists(tmp_path / "does_not_exist.parquet")


# ---------------------------------------------------------------------------
# StageCache
# ---------------------------------------------------------------------------


def test_stage_cache_miss_when_artifact_missing(tmp_path: Path):
    cache = StageCache(tmp_path / "artifact.json", config={"threshold": 0.5})
    assert cache.hit() is False


def test_stage_cache_hit_after_write_with_same_config(tmp_path: Path):
    path = tmp_path / "artifact.json"
    config = {"threshold": 0.5, "fps": 10}

    cache = StageCache(path, config=config, stage="detect")
    assert cache.hit() is False
    save_json({"result": 42}, path)
    cache.write_meta()

    cache2 = StageCache(path, config=config, stage="detect")
    assert cache2.hit() is True


def test_stage_cache_miss_when_config_changes(tmp_path: Path):
    path = tmp_path / "artifact.json"

    cache = StageCache(path, config={"threshold": 0.5}, stage="detect")
    save_json({"result": 42}, path)
    cache.write_meta()

    cache_changed = StageCache(path, config={"threshold": 0.6}, stage="detect")
    assert cache_changed.hit() is False


def test_stage_cache_get_or_compute_skips_recompute(tmp_path: Path):
    path = tmp_path / "artifact.json"
    config = {"threshold": 0.5}
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"result": calls["n"]}

    cache = StageCache(path, config=config, stage="detect")
    first = cache.get_or_compute(compute, save_json, load_json)
    assert first == {"result": 1}
    assert calls["n"] == 1

    cache2 = StageCache(path, config=config, stage="detect")
    second = cache2.get_or_compute(compute, save_json, load_json)
    assert second == {"result": 1}  # not recomputed
    assert calls["n"] == 1
