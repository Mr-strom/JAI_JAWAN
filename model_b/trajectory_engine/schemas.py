"""
schemas.py — Output event schema for the Trajectory Engine.
Pydantic v2. Validates every outgoing event before it hits the MQTT wire.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class TrajectoryMetadata(BaseModel):
    model_version: str
    engine_name: Literal["trajectory"]
    processing_time_ms: int
    tracker: Literal["bytetrack"]
    max_track_age: int
    kalman_enabled: bool
    trajectory_points: List[List[float]]        # [[x, y], ...]  normalized
    velocity: float                              # pixels/second
    direction_degrees: float                     # 0-360
    zone_transitions: List[str] = Field(default_factory=list)
    behavior_tags: List[str] = Field(default_factory=list)

    # ── New state features ────────────────────────────────────────────────────
    lifecycle_state: str = "NEW"                 # NEW | ACTIVE | LOST | REMOVED
    persistence_score: float = 0.0              # frame_count / stable_frames  [0,1]
    distance_travelled_px: float = 0.0          # cumulative Euclidean distance in pixels
    stationary_duration_s: float = 0.0          # accumulated seconds below vel threshold
    zone_history: List[str] = Field(default_factory=list)  # chronological zone entries


    @field_validator("velocity")
    @classmethod
    def velocity_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("velocity must be >= 0")
        return v

    @field_validator("direction_degrees")
    @classmethod
    def direction_in_range(cls, v: float) -> float:
        return v % 360.0

    @field_validator("behavior_tags")
    @classmethod
    def valid_behavior_tags(cls, v: List[str]) -> List[str]:
        allowed = {"loitering", "rapid_approach"}
        bad = [t for t in v if t not in allowed]
        if bad:
            raise ValueError(f"Unknown behavior tags: {bad}")
        return v


class TrajectoryEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["trajectory_update"] = "trajectory_update"
    severity: Literal["info"] = "info"
    timestamp: str                              # ISO-8601 UTC
    camera_id: str
    zone_tag: Literal["long_range"] = "long_range"
    entity_type: Literal["human", "vehicle"]
    engine_source: Literal["trajectory"] = "trajectory"
    entity_id: str                              # persistent track_id as string
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)  # [x1,y1,x2,y2] normalised
    evidence_ref: str
    metadata: TrajectoryMetadata
    hash: str                                   # SHA-256 of evidence file
    provisional: Literal[False] = False

    @field_validator("bbox")
    @classmethod
    def bbox_normalised(cls, v: List[float]) -> List[float]:
        for val in v:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"bbox value {val} out of [0,1] range")
        return v
