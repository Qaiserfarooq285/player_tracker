"""Stage 3 — team assignment: torso-colour clustering (default) with SigLIP embeddings kept as a
config-selectable fallback (CLAUDE.md §5 Stage 3).

**Primary method switched to torso colour, 2026-08-24 — the second correction on this module,
against a second controlled empirical check.** The original spec was SigLIP embeddings -> UMAP ->
KMeans; a first correction (same day, see git history) replaced that with a 3-candidate
silhouette-based comparison that still leaned on SigLIP. The coordinator then ran an independent,
controlled experiment on the SAME cached tracks: torso band (`torso_top_frac=0.15` to
`torso_bottom_frac=0.50` of bbox height, `torso_x_inset_frac=0.18` off each side), boxes under
`min_box_height_px` skipped, crop converted to CIELAB, **mean within each crop, MEDIAN across a
track's sampled crops** (robust to a single occluded/motion-blurred sample), KMeans(k=2) directly
on the resulting one-Lab-vector-per-track. Result on `clip2`: a 24/15 split with L*=97 vs. L*=146
— a large, physically meaningful lightness gap matching dark-blue vs. white kits — decisively
better than anything the SigLIP-embedding path produced on the same data. Adopted as the default
(`configs/team.yaml: method: colour`); SigLIP is kept, not deleted, as `method: siglip` — it may
still be the better choice on real broadcast footage with larger, cleaner crops, and switching is
one config line, not a code change.

Pipeline (per take — `src/track/run.py` calls `assign_teams` once per take, not globally across a
whole multi-venue clip like `clip4`, since lighting differs by venue/take, CLAUDE.md §3.2): collect
torso crops per track (`collect_track_crops`), reduce each track to ONE colour vector
(`per_track_lab_median`), cluster those vectors directly with KMeans(k=2) — no majority-voting step
needed since clustering already happens at track granularity. `Track.team_confidence` is each
track's distance to the OTHER cluster's centroid relative to its total distance to both centroids
(`centroid_distance_confidence`) — 1.0 sitting exactly on its own centroid, 0.5 exactly midway
(CLAUDE.md Golden Rule 5: every stat carries a confidence). A lopsided resulting split (few
opposing-kit players actually visible, or a bad split) is flagged loudly and every track's
confidence for that take is capped low (`configs/team.yaml: cluster.min_team_balance_ratio`/
`low_confidence_cap`) rather than reading as confident either way (CLAUDE.md §10).

Referees/goalkeepers (`Track.dominant_class`, set by Stage 3's tracker) are excluded from
clustering entirely; a player track with no usable crop (too small/short-lived throughout) also
gets `team=None` — both counted and logged, never silently dropped.

Uses `SiglipVisionModel` + `SiglipImageProcessor` (the *vision-only* half of SigLIP), not the
combined `SiglipModel`/`SiglipProcessor` — measured 2026-08-24: the combined processor's tokenizer
requires the `sentencepiece` package, which is not installed and not needed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.logging import get_logger
from src.common.types import BBox, DetectionClass, Take, Track, TrackBox
from src.common.video import decode_frames

logger = get_logger(__name__)


def load_siglip(team_cfg: dict, hardware_config: dict) -> tuple[Any, Any, str]:
    """Load the SigLIP vision tower + image processor onto `hardware_config["device"]`."""
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    device = hardware_config["device"] if torch.cuda.is_available() else "cpu"
    model_name = team_cfg["embedding"]["model"]
    processor = SiglipImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()
    logger.info("loaded SigLIP vision tower %s on %s", model_name, device)
    return model, processor, device


def free_siglip(model: Any) -> None:
    """Explicitly free SigLIP's VRAM (CLAUDE.md §11: unload each model before the next)."""
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_crops(
    model: Any, processor: Any, crops_bgr: list[np.ndarray], device: str, batch_size: int
) -> np.ndarray:
    """Embed a list of BGR pixel crops with SigLIP's pooled vision output (`(N, hidden_size)`)."""
    import torch

    if not crops_bgr:
        hidden_size = model.config.hidden_size
        return np.zeros((0, hidden_size), dtype=np.float32)
    crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
    chunks: list[np.ndarray] = []
    for i in range(0, len(crops_rgb), batch_size):
        batch = crops_rgb[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        chunks.append(out.pooler_output.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _nearest_box(track: Track, t: float, tolerance: float) -> TrackBox | None:
    """The track's own box closest in time to `t`, if within `tolerance` seconds."""
    best: TrackBox | None = None
    best_dt = tolerance
    for box in track.boxes:
        dt = abs(box.t - t)
        if dt <= best_dt:
            best, best_dt = box, dt
    return best


def torso_region(bbox: BBox, top_frac: float, bottom_frac: float, x_inset_frac: float) -> BBox:
    """The jersey/torso sub-region of a full player bbox: a horizontal band between
    `top_frac`/`bottom_frac` of the bbox's own height, inset `x_inset_frac` off each side.

    Pure geometry, no image I/O — split out so it's directly unit-testable. See module docstring
    for why the torso band (not the full bbox) is what team assignment should use.
    """
    height = bbox.y2 - bbox.y1
    width = bbox.x2 - bbox.x1
    y1 = bbox.y1 + top_frac * height
    y2 = bbox.y1 + bottom_frac * height
    inset = x_inset_frac * width
    x1 = bbox.x1 + inset
    x2 = bbox.x2 - inset
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def collect_track_crops(
    video_path: str | Path,
    take: Take,
    tracks: list[Track],
    decode_cfg: dict,
    team_stage_cfg: dict,
    sampling_cfg: dict,
    use_nvdec: bool = True,
) -> dict[int, list[np.ndarray]]:
    """Decode `take`'s frame range ONCE at `team_stage_cfg['fps_sample']` and pull up to
    `sampling_cfg['crops_per_track']` player **torso** crops per track via nearest-timestamp box
    matching (see `torso_region`). A box shorter than `sampling_cfg['min_box_height_px']` is
    skipped outright — a far player's box (CLAUDE.md §3.2: as short as ~30-60px) carries
    essentially no reliable kit-colour signal once cropped down to just its torso band.

    A single continuous decode pass (not one `-ss` seek per crop) — `src/ingest/decode.py`'s own
    docstring documents that tiny per-crop seek windows measurably hang NVDEC on this footage;
    this mirrors that module's `_dump_debug_frames` fix (decode once, pick frames as they stream).
    `decode_cfg['scale_width']` must match whatever Stage 2 detected at, so a track's stored pixel
    bboxes stay valid against these frames.
    """
    fps_sample = team_stage_cfg["fps_sample"]
    tolerance = 0.5 / fps_sample if fps_sample > 0 else 0.5
    crops_per_track = sampling_cfg["crops_per_track"]
    min_box_height_px = sampling_cfg["min_box_height_px"]
    top_frac = sampling_cfg["torso_top_frac"]
    bottom_frac = sampling_cfg["torso_bottom_frac"]
    x_inset_frac = sampling_cfg["torso_x_inset_frac"]

    crops_by_track: dict[int, list[np.ndarray]] = {track.id: [] for track in tracks}
    if not tracks:
        return crops_by_track

    for _idx, t, frame in decode_frames(
        video_path,
        fps=fps_sample,
        start=take.t_start,
        end=take.t_end,
        scale_width=decode_cfg["scale_width"],
        use_nvdec=use_nvdec,
    ):
        height, width = frame.shape[:2]
        for track in tracks:
            if len(crops_by_track[track.id]) >= crops_per_track:
                continue
            box = _nearest_box(track, t, tolerance)
            if box is None:
                continue
            if box.bbox.height < min_box_height_px:
                continue
            torso = torso_region(box.bbox, top_frac, bottom_frac, x_inset_frac)
            x1, y1, x2, y2 = (int(round(v)) for v in torso.to_xyxy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:  # degenerate after clipping to frame bounds
                continue
            crops_by_track[track.id].append(frame[y1:y2, x1:x2].copy())

    return crops_by_track


def crop_lab_mean(crop_bgr: np.ndarray) -> np.ndarray:
    """Mean CIELAB colour of a crop ("mean within a crop" — see `per_track_lab_median` for the
    across-crops aggregation). CIELAB (not raw BGR/chromaticity) per the coordinator's controlled
    experiment: L* separated the clip2 kits cleanly (dark-blue vs. white, ~97 vs. ~146) where a
    768-D SigLIP embedding did not.
    """
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return lab.reshape(-1, 3).mean(axis=0)


def per_track_lab_median(crops: list[np.ndarray]) -> np.ndarray | None:
    """One Lab vector per track: mean-Lab of each crop, then MEDIAN across the track's crops.

    Median (not mean) across samples per the coordinator's spec — robust to a single occluded or
    motion-blurred crop skewing the track's colour estimate. `None` when `crops` is empty (a
    track with zero usable torso crops this take — reported by the caller as "too small to
    judge", not silently defaulted to any colour).
    """
    if not crops:
        return None
    samples = np.stack([crop_lab_mean(c) for c in crops])
    return np.median(samples, axis=0)


def centroid_distance_confidence(
    point: np.ndarray, own_centroid: np.ndarray, other_centroids: list[np.ndarray]
) -> float:
    """Confidence = distance-to-nearest-OTHER-centroid / (distance-to-own + distance-to-nearest-
    other). 1.0 when `point` sits exactly on its own centroid; 0.5 exactly midway between its own
    and the nearest other centroid (CLAUDE.md task spec: "a track sitting midway between
    centroids deserves low confidence"); below 0.5 if `point` is actually nearer another
    centroid than its own (can happen after KMeans converges with points reassigned mid-fit).
    Pure function, unit-testable without clustering.
    """
    d_own = float(np.linalg.norm(point - own_centroid))
    d_other = min(float(np.linalg.norm(point - oc)) for oc in other_centroids)
    total = d_own + d_other
    return d_other / total if total > 0 else 0.5


def balance_ratio(team_by_track: dict[int, int | None]) -> float:
    """`min(cluster size) / total assigned` across a track-level team assignment.

    0.0 when fewer than 2 tracks got a (non-`None`) team, or everything landed in one cluster —
    both read as "not a usable two-team split". Pure function, unit-testable without clustering.
    Used only as a post-hoc *confidence* signal (see `assign_teams`), not to pick between
    candidate methods: a correct-but-genuinely-lopsided split (few opposing-kit players visible
    in a short clip) scores as "unbalanced" too.
    """
    values = [v for v in team_by_track.values() if v is not None]
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    if len(counts) < 2:
        return 0.0
    return min(counts.values()) / sum(counts.values())


def _apply_balance_cap(
    team_by_track: dict[int, int],
    confidence_by_track: dict[int, float],
    cluster_cfg: dict,
    method: str,
) -> dict[int, float]:
    """Cap every confidence in `confidence_by_track` at `cluster_cfg['low_confidence_cap']`,
    logging loudly, if the split's `balance_ratio` is below `cluster_cfg['min_team_balance_ratio']`.
    """
    balance = balance_ratio(team_by_track)
    min_balance = cluster_cfg["min_team_balance_ratio"]
    if balance >= min_balance:
        return confidence_by_track
    cap = cluster_cfg["low_confidence_cap"]
    logger.warning(
        "LOPSIDED team split this take: minority/total=%.2f < min_team_balance_ratio=%.2f "
        "(method=%s, n_tracks=%d) — could be a genuinely lopsided clip (few opposing-kit "
        "players actually visible) or a bad split; capping every track's team_confidence at "
        "%.2f rather than letting it read as fully confident either way (CLAUDE.md §10).",
        balance,
        min_balance,
        method,
        len(team_by_track),
        cap,
    )
    return {tid: min(c, cap) for tid, c in confidence_by_track.items()}


def _assign_teams_colour(
    crops_by_track: dict[int, list[np.ndarray]], eligible_ids: list[int], cluster_cfg: dict
) -> tuple[dict[int, int | None], dict[int, float], int]:
    """Torso-colour clustering: one Lab vector per track (`per_track_lab_median`), KMeans(k=2)
    directly on those vectors, confidence via `centroid_distance_confidence`.

    Returns `(team_by_track, confidence_by_track, n_too_small)` for the `eligible_ids` only —
    `n_too_small` is how many had zero usable torso crops this take (reported by the caller).
    """
    lab_by_track: dict[int, np.ndarray | None] = {
        tid: per_track_lab_median(crops_by_track[tid]) for tid in eligible_ids
    }
    usable_ids = [tid for tid in eligible_ids if lab_by_track[tid] is not None]
    n_too_small = len(eligible_ids) - len(usable_ids)

    team_by_track: dict[int, int | None] = {
        tid: None for tid in eligible_ids if lab_by_track[tid] is None
    }
    confidence_by_track: dict[int, float] = dict.fromkeys(team_by_track, 0.0)

    n_clusters = min(cluster_cfg["n_clusters"], len(usable_ids))
    if n_clusters < 2:
        for tid in usable_ids:
            team_by_track[tid] = None
            confidence_by_track[tid] = 0.0
        return team_by_track, confidence_by_track, n_too_small

    from sklearn.cluster import KMeans

    X = np.stack([lab_by_track[tid] for tid in usable_ids])
    kmeans = KMeans(n_clusters=n_clusters, random_state=cluster_cfg["random_state"], n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for i, tid in enumerate(usable_ids):
        label = int(labels[i])
        team_by_track[tid] = label
        other_centroids = [centroids[j] for j in range(n_clusters) if j != label]
        confidence_by_track[tid] = centroid_distance_confidence(
            X[i], centroids[label], other_centroids
        )

    return team_by_track, confidence_by_track, n_too_small


def _cluster_and_vote(
    features: np.ndarray, crop_track_ids: list[int], n_clusters: int, cluster_cfg: dict
) -> tuple[dict[int, int], dict[int, float]]:
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=cluster_cfg["random_state"], n_init=10)
    labels = kmeans.fit_predict(features)
    return majority_vote_teams(crop_track_ids, labels.tolist())


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalise `embeddings` (safe against zero-norm rows)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def _assign_teams_siglip(
    crops_by_track: dict[int, list[np.ndarray]],
    eligible_ids: list[int],
    team_cfg: dict,
    hardware_config: dict,
) -> tuple[dict[int, int | None], dict[int, float], int]:
    """SigLIP-embedding fallback (`configs/team.yaml: method: siglip`) — KMeans on L2-normalised
    embeddings, team assigned per track by majority vote over its own crops' cluster labels
    (confidence = that majority's vote share). Kept for footage where torso colour may be less
    reliable than it was on `clip2` (e.g. same-coloured kits, very washed-out broadcast colour
    grading) — not used by default; see module docstring for why colour is the default here.
    """
    all_crops: list[np.ndarray] = []
    crop_track_ids: list[int] = []
    for tid in eligible_ids:
        for crop in crops_by_track[tid]:
            all_crops.append(crop)
            crop_track_ids.append(tid)
    n_too_small = sum(1 for tid in eligible_ids if not crops_by_track[tid])

    team_by_track: dict[int, int | None] = {
        tid: None for tid in eligible_ids if not crops_by_track[tid]
    }
    confidence_by_track: dict[int, float] = dict.fromkeys(team_by_track, 0.0)

    if not all_crops:
        return team_by_track, confidence_by_track, n_too_small

    model, processor, device = load_siglip(team_cfg, hardware_config)
    try:
        batch_size = hardware_config["stages"]["team"]["batch_size"]
        embeddings = embed_crops(model, processor, all_crops, device, batch_size)
    finally:
        free_siglip(model)

    cluster_cfg = team_cfg["cluster"]
    n_clusters = min(cluster_cfg["n_clusters"], len(set(crop_track_ids)))
    if n_clusters < 2:
        for tid in eligible_ids:
            if tid not in team_by_track:
                team_by_track[tid] = None
                confidence_by_track[tid] = 0.0
        return team_by_track, confidence_by_track, n_too_small

    normalized = _l2_normalize(embeddings)
    voted_team, voted_conf = _cluster_and_vote(normalized, crop_track_ids, n_clusters, cluster_cfg)
    team_by_track.update(voted_team)
    confidence_by_track.update(voted_conf)
    return team_by_track, confidence_by_track, n_too_small


def assign_teams(
    crops_by_track: dict[int, list[np.ndarray]],
    dominant_class_by_track: dict[int, DetectionClass | None],
    team_cfg: dict,
    hardware_config: dict,
) -> tuple[dict[int, int | None], dict[int, float], str]:
    """Assign each outfield-player track a team for one take (see module docstring for the
    torso-colour-by-default / SigLIP-fallback design and why).

    Dispatches on `team_cfg.get("method", "colour")` — `"colour"` (default): per-track median
    Lab colour, KMeans(k=2), `centroid_distance_confidence`. `"siglip"`: per-crop SigLIP
    embeddings, KMeans(k=2), majority vote. Either way, `balance_ratio` is then applied as a
    post-hoc confidence cap (see `_apply_balance_cap`) — never the method-selection criterion.

    Returns `(team_by_track_id, team_confidence_by_track_id, method)`. Referees/goalkeepers
    (`dominant_class_by_track != DetectionClass.PLAYER`) are excluded from clustering and get
    `team=None`, `team_confidence=0.0`; a player track with no usable torso crop this take also
    gets `team=None` (counted as "too small to judge" and logged, per CLAUDE.md §10).
    """
    method = team_cfg.get("method", "colour")
    eligible_ids = [
        tid
        for tid, cls in dominant_class_by_track.items()
        if cls == DetectionClass.PLAYER and tid in crops_by_track
    ]
    non_eligible_ids = [tid for tid in crops_by_track if tid not in eligible_ids]
    team_by_track: dict[int, int | None] = dict.fromkeys(non_eligible_ids)
    confidence_by_track: dict[int, float] = dict.fromkeys(non_eligible_ids, 0.0)

    if not eligible_ids:
        return team_by_track, confidence_by_track, method

    if method == "siglip":
        team_e, conf_e, n_too_small = _assign_teams_siglip(
            crops_by_track, eligible_ids, team_cfg, hardware_config
        )
    else:
        team_e, conf_e, n_too_small = _assign_teams_colour(
            crops_by_track, eligible_ids, team_cfg["cluster"]
        )

    conf_e = _apply_balance_cap(
        {tid: t for tid, t in team_e.items() if t is not None}, conf_e, team_cfg["cluster"], method
    )

    logger.info(
        "team clustering (method=%s, take): %d eligible track(s), %d too small to judge, "
        "balance=%.2f",
        method,
        len(eligible_ids),
        n_too_small,
        balance_ratio(team_e),
    )

    team_by_track.update(team_e)
    confidence_by_track.update(conf_e)
    return team_by_track, confidence_by_track, method


def majority_vote_teams(
    crop_track_ids: list[int], labels: list[int]
) -> tuple[dict[int, int], dict[int, float]]:
    """Pure majority-vote step (used by the `siglip` fallback method only — the default `colour`
    method clusters at track granularity directly, so it needs no vote): given each crop's track
    id and its cluster label (parallel lists, one entry per crop, across possibly many tracks),
    assign each track the most common label among its own crops, with confidence = that label's
    vote share for the track.
    """
    labels_by_track: dict[int, list[int]] = defaultdict(list)
    for tid, label in zip(crop_track_ids, labels, strict=True):
        labels_by_track[tid].append(int(label))

    team_by_track: dict[int, int] = {}
    confidence_by_track: dict[int, float] = {}
    for tid, track_labels in labels_by_track.items():
        votes = Counter(track_labels)
        team_label, n_votes = votes.most_common(1)[0]
        team_by_track[tid] = team_label
        confidence_by_track[tid] = n_votes / len(track_labels)
    return team_by_track, confidence_by_track
