"""
schema_v1 — LOCKED JSON Event Schema
SIH26187 | Model A → Model B Integration Contract

CRITICAL:
  - Field names are LOCKED. Do not rename.
  - To extend, propose schema_v2 with owner approval.
  - Validation failures MUST fail loudly (raise/log). Never publish malformed events.
  - This is the single source of truth for both Model A publisher and Model B consumers.
"""

from __future__ import annotations

import hashlib
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums — LOCKED. Do not add values without owner approval.
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    motion          = "motion"
    trigger         = "trigger"
    system_health   = "system_health"
    camera_anomaly  = "camera_anomaly"
    animal_detected = "animal_detected"


class Severity(str, Enum):
    info        = "info"
    warning     = "warning"
    provisional = "provisional"
    confirmed   = "confirmed"
    critical    = "critical"


class ZoneTag(str, Enum):
    close_range = "close_range"
    long_range  = "long_range"


class Zone(str, Enum):
    warning_zone   = "warning_zone"
    intrusion_zone = "intrusion_zone"
    chokepoint     = "chokepoint"
    icp            = "icp"
    perimeter      = "perimeter"


class EntityType(str, Enum):
    human       = "human"
    vehicle     = "vehicle"
    animal      = "animal"
    animal_cart = "animal_cart"
    unknown     = "unknown"


class TriggerType(str, Enum):
    climbing       = "climbing"
    fence_cutting  = "fence_cutting"
    rapid_approach = "rapid_approach"
    zone_violation = "zone_violation"


# ---------------------------------------------------------------------------
# Nested metadata block — LOCKED field names
# ---------------------------------------------------------------------------

class EventMetadata(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model_version:       str   = Field(..., description="Semver string, e.g. '1.0.0'")
    processing_time_ms:  int   = Field(..., ge=0)
    frame_number:        int   = Field(..., ge=0)
    trigger_type:        Optional[TriggerType] = Field(None)
    confirmation_frames: int   = Field(..., ge=0, le=10,
                                       description="Frames confirming trigger. Must be >=3 for confirmed/critical.")
    spoofing_flags:      List[str] = Field(default_factory=list)
    fallback_active:     bool      = Field(default=False, description="True when running in fallback/degraded mode")
    blur_score:          Optional[float] = Field(default=None, description="Laplacian variance sharpness score")
    is_blurry:           bool      = Field(default=False, description="True if frame blur exceeds threshold")


# ---------------------------------------------------------------------------
# Root event model — schema_v1
# LOCKED: Every output of Model A MUST validate against this before publish.
# ---------------------------------------------------------------------------

class ModelAEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    event_id:     str         = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:   EventType
    severity:     Severity
    timestamp:    str         = Field(..., description="ISO 8601 UTC, e.g. 2025-01-01T00:00:00Z")
    camera_id:    str         = Field(..., min_length=1)
    zone_tag:     ZoneTag
    zone:         Zone
    entity_type:  EntityType
    entity_id:    Optional[str] = Field(None, description="track_id, global_fusion_id, or null")
    confidence:   float       = Field(..., ge=0.0, le=1.0)
    bbox:         List[float] = Field(..., min_length=4, max_length=4,
                                      description="[x1,y1,x2,y2] normalised 0-1")
    evidence_ref: str         = Field(..., description="Filepath or URL to evidence image/video")
    hash:         str         = Field(..., description="SHA-256 of evidence_ref file")
    engine_source:str         = Field("model_a", pattern=r"^model_a$")
    metadata:     EventMetadata


    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("bbox")
    @classmethod
    def bbox_normalised(cls, v: List[float]) -> List[float]:
        """All bbox coordinates must be in [0, 1]."""
        if not all(0.0 <= coord <= 1.0 for coord in v):
            raise ValueError(
                f"bbox coordinates must be normalised to [0,1]. Got: {v}"
            )
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"bbox must satisfy x2>x1 and y2>y1. Got: {v}"
            )
        return v

    # ------------------------------------------------------------------
    # Model-level validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def confirmation_frames_gate(self) -> "ModelAEvent":
        """
        Multi-frame confirmation rule (NON-NEGOTIABLE — Rule #1):
        Severity confirmed/critical REQUIRES metadata.confirmation_frames >= 3.
        This is the software equivalent of the hardware lock that CIBMS lacked.
        """
        if self.severity in (Severity.confirmed, Severity.critical):
            if self.metadata.confirmation_frames < 3:
                raise ValueError(
                    f"severity='{self.severity}' requires confirmation_frames >= 3. "
                    f"Got {self.metadata.confirmation_frames}. "
                    "Do NOT bypass this check. See Rule #1."
                )
        return self

    @model_validator(mode="after")
    def trigger_type_required_for_trigger_events(self) -> "ModelAEvent":
        """trigger events MUST carry a trigger_type in metadata."""
        if self.event_type == EventType.trigger:
            if self.metadata.trigger_type is None:
                raise ValueError(
                    "event_type='trigger' requires metadata.trigger_type to be set."
                )
        return self

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def to_mqtt_payload(self) -> bytes:
        """Serialise to UTF-8 JSON bytes for MQTT publish."""
        return self.model_dump_json(indent=None).encode("utf-8")

    @staticmethod
    def compute_hash(filepath: str) -> str:
        """
        Compute SHA-256 of an evidence file.
        Returns hex digest string.
        Call this AFTER the evidence file is fully written to disk.
        """
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            # Evidence file not yet flushed — return sentinel; caller must retry
            return "FILE_NOT_FOUND"
