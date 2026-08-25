"""Pydantic v2 data contracts shared across every pipeline stage (CLAUDE.md §4).

Stages compose through these models; intermediate artifacts cache to disk (parquet/JSON + video)
so stages run independently and resumably (CLAUDE.md §10). Rule: no stage invents a field it can't
justify — unknowns are explicit (``None`` + a confidence), never guessed (Golden Rule 5).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """An axis-aligned pixel bounding box, corners ``(x1, y1)``–``(x2, y2)``."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        """X coordinate of the box centre."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Y coordinate of the box centre."""
        return (self.y1 + self.y2) / 2.0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as an ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


class SourceProfile(str, Enum):  # noqa: UP042 -- (str, Enum) is the spec'd contract (CLAUDE.md §4)
    """Input video category produced by the Stage 0.5 source profiler (CLAUDE.md §5, Stage 0.5)."""

    BROADCAST = "broadcast"
    SINGLE_STATIC = "single-static"
    SINGLE_PANNING = "single-panning"


class DetectionClass(str, Enum):  # noqa: UP042 -- (str, Enum) is the spec'd contract (CLAUDE.md §4)
    """Object classes emitted by the Stage 2 detector."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class Frame(BaseModel):
    """A single decoded video frame reference (CLAUDE.md §4)."""

    index: int
    t: float
    path: Path | None = None


class Detection(BaseModel):
    """A single Stage 2 player/goalkeeper/referee detection in one frame.

    ``frame_index`` is the *sampled*-sequence index (0, 1, 2, ... in decode order at the stage's
    own `fps_sample`), not a native-video frame number — ``t`` (seconds, from the same decode
    call) is what downstream stages must use to relate a detection to a `Take`'s `t_start`/`t_end`
    (native-frame-numbered) or to another stage sampled at a different fps.
    """

    bbox: BBox
    cls: DetectionClass
    conf: float
    frame_index: int
    t: float


class BallDetection(BaseModel):
    """A single Stage 2 ball detection, possibly filled in by interpolation.

    See `Detection.frame_index`/`t` docstring — the same sampled-index-vs-timestamp distinction
    applies here.
    """

    bbox: BBox
    conf: float
    frame_index: int
    t: float
    interpolated: bool = False


class PitchHomography(BaseModel):
    """Pixel <-> pitch-metre transform for one take or frame (CLAUDE.md §5, Stage 2 / ADR-3)."""

    matrix: list[list[float]]
    keypoints: list[tuple[float, float]] = Field(default_factory=list)
    conf: float
    frame_index: int | None = None
    method: str

    @field_validator("matrix")
    @classmethod
    def _validate_3x3(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("PitchHomography.matrix must be a 3x3 matrix")
        return v

    def to_numpy(self) -> np.ndarray:
        """Return the homography matrix as a 3x3 ``float64`` numpy array."""
        return np.array(self.matrix, dtype=np.float64)

    def pixel_to_pitch(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Map pixel coordinates to pitch metres via ``cv2.perspectiveTransform``."""
        import cv2

        src = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.to_numpy())
        return [(float(p[0][0]), float(p[0][1])) for p in dst]


class Take(BaseModel):
    """A continuous camera take, the unit of within-take tracking (CLAUDE.md Golden Rule 3)."""

    id: int
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    kind: Literal["main", "closeup", "replay", "unknown"] = "unknown"
    dropped_reason: str | None = None


class TrackBox(BaseModel):
    """One frame's worth of a track's bounding box."""

    frame_index: int
    t: float
    bbox: BBox
    conf: float


class Track(BaseModel):
    """A within-take multi-object track (CLAUDE.md §4); IDs reset at every cut.

    `dominant_class` is the majority-vote `DetectionClass` across the track's own detections
    (Stage 3, src/track/tracker.py) — the only per-track signal Stage 3's team classifier has for
    excluding referees/goalkeepers from the 2-team outfield clustering (CLAUDE.md §5 Stage 3).
    `None` only for a track with zero boxes (should not occur in practice).
    """

    id: int
    take_id: int
    boxes: list[TrackBox]
    team: int | None = None
    team_confidence: float = 0.0
    dominant_class: DetectionClass | None = None
    jersey_number: int | None = None
    id_confidence: float = 0.0
    path_pitch_xy: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def t_start(self) -> float:
        """Timestamp (seconds) of the track's first box, or 0.0 if empty."""
        return self.boxes[0].t if self.boxes else 0.0

    @property
    def t_end(self) -> float:
        """Timestamp (seconds) of the track's last box, or 0.0 if empty."""
        return self.boxes[-1].t if self.boxes else 0.0

    @property
    def duration(self) -> float:
        """Track lifetime in seconds."""
        return self.t_end - self.t_start

    @property
    def n_boxes(self) -> int:
        """Number of boxes (frames) in the track."""
        return len(self.boxes)


class EventType(str, Enum):  # noqa: UP042 -- (str, Enum) is the spec'd contract (CLAUDE.md §4)
    """Kinds of events the pipeline can emit (CLAUDE.md §5, Stage 5)."""

    GOAL = "goal"
    SHOT = "shot"
    SPRINT = "sprint"
    PASS = "pass"
    TOUCH = "touch"
    TACKLE = "tackle"
    DRIBBLE = "dribble"
    KEY_MOMENT = "key_moment"


class Event(BaseModel):
    """A detected/inferred event, traceable to its evidence (CLAUDE.md Golden Rule 5).

    ``source`` documents HOW the event was derived (e.g. ``"scoreboard_delta"``,
    ``"speed_heuristic"``); ``evidence`` holds traceable references (frame indices, speeds,
    homography-derived positions, etc.) so no stat is ever fabricated.
    """

    id: str
    type: EventType
    t_start: float
    t_end: float
    player_track_id: int | None
    take_id: int | None
    confidence: float
    source: str
    evidence: dict = Field(default_factory=dict)


class Clip(BaseModel):
    """A ranked, cuttable highlight clip derived from one event."""

    event_id: str
    t_start: float
    t_end: float
    take_id: int | None
    rank_score: float
    path: Path | None = None


class PlayerStats(BaseModel):
    """Aggregated, confidence-carrying stats for one (manually or auto) selected player."""

    player_ref: str
    track_ids: list[int]
    counts: dict[str, int]
    confidences: dict[str, float]
    events: list[str]


class RunProfile(BaseModel):
    """Stage 0.5 source-profiler output: what kind of footage this is."""

    profile: SourceProfile
    n_cuts: int
    cuts_per_min: float
    motion_score: float
    width: int
    height: int
    fps: float
    duration: float
    quality_flag: str
    notes: list[str] = Field(default_factory=list)


class StageTiming(BaseModel):
    """Wall-clock + VRAM measurement for one pipeline stage (CLAUDE.md §11)."""

    stage: str
    wall_seconds: float
    vram_peak_mb: float | None
    notes: str = ""


class RunReport(BaseModel):
    """Per-run summary: config, profile, per-stage timings, and dropped-item counts."""

    run_id: str
    video: str
    config_hash: str
    profile: RunProfile | None
    timings: list[StageTiming]
    dropped: dict[str, int]
    created_at: datetime
