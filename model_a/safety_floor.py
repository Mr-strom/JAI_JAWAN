"""
Safety Floor — Lightweight Fallback Pipeline
SIH26187 | Model A | Step 13 of pipeline

What this IS:
  The minimum viable detection pipeline that runs on a camera when its
  assigned Model B engine is dead (heartbeat stale >30s).
  Uses only Model A components: YOLO → AnimalFilter → BBoxConsistency →
  TriggerDetector. Publishes schema_v1 events as usual.

  This PREVENTS total blindness but is deliberately lightweight.
  Operators are alerted via the engine health events that they are in
  fallback mode and should restore Model B manually.

What this EXPLICITLY IS NOT (and will never be):
  - NOT Face recognition (close-range only, requires Model B Face engine)
  - NOT ANPR (requires Model B ANPR engine + chokepoint tag)
  - NOT Trajectory analysis (requires Model B Trajectory engine)
  - NOT full Posture classification (requires Model B Posture engine)
  - NOT a replacement for Model B — it is a safety net, not a feature set.

Rule: Do NOT auto-restart Model B engines. Manual intervention required.
      This pipeline just keeps cameras non-blind until the operator acts.

Schema compliance:
  All safety floor events are valid schema_v1 events.
  engine_source is always "model_a".
  metadata.spoofing_flags includes "SAFETY_FLOOR_ACTIVE" to signal to
  downstream consumers (dashboards, operators) that this event was
  generated during Model B fallback — without polluting spoofing_flags
  semantics (the flag is informational only).
"""

from __future__ import annotations

import datetime
import logging
from typing import List, Optional

import numpy as np

from model_a.animal_filter import AnimalFilter
from model_a.bbox_consistency import BBoxConsistencyChecker
from model_a.detector import Detector
from model_a.schema_v1 import (
    EntityType,
    EventMetadata,
    EventType,
    ModelAEvent,
    Severity,
    TriggerType,
    Zone,
    ZoneTag,
)
from model_a.trigger_detector import TriggerDetector

logger = logging.getLogger(__name__)

MODEL_VERSION     = "1.0.0"
_SAFETY_FLOOR_FLAG = "SAFETY_FLOOR_ACTIVE"   # informational flag for dashboards


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


class SafetyFloor:
    """
    Lightweight single-camera detection pipeline used during Model B fallback.

    Instantiate one SafetyFloor per camera that needs coverage.
    This is NOT a permanent fixture — it is activated by FallbackRouter
    when an engine goes dead and deactivated when it recovers.

    Usage::

        floor = SafetyFloor(camera_id="cam_01", detector=detector)
        events = floor.process(frame, frame_number=n, timestamp_utc=ts)
        for event in events:
            bus_client.publish_event(event)
    """

    def __init__(
        self,
        camera_id:   str,
        detector:    Detector,
        zone_tag:    ZoneTag = ZoneTag.long_range,
        zone:        Zone    = Zone.perimeter,
        trigger_detector: Optional[TriggerDetector] = None,
        animal_filter:    Optional[AnimalFilter]    = None,
        bbox_checker:     Optional[BBoxConsistencyChecker] = None,
    ) -> None:
        self.camera_id = camera_id
        self.zone_tag  = zone_tag
        self.zone      = zone

        # Reuse existing components — same rules apply
        self._detector  = detector
        self._trigger   = trigger_detector or TriggerDetector(confirmation_frames=3)
        self._afilt     = animal_filter    or AnimalFilter()
        self._bbox      = bbox_checker     or BBoxConsistencyChecker()

        logger.warning(
            "SafetyFloor ACTIVATED for camera '%s'. "
            "Model B capabilities (Face/ANPR/Posture/Trajectory) NOT available. "
            "Basic motion + trigger detection only. "
            "Restore Model B engine to resume full coverage.",
            camera_id,
        )

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(
        self,
        frame:         np.ndarray,
        frame_number:  int,
        timestamp_utc: str,
        trigger_type_hint: Optional[TriggerType] = None,
    ) -> List[ModelAEvent]:
        """
        Run the safety floor pipeline on one frame.

        Returns validated ModelAEvent objects ready for publishing.
        All events include the SAFETY_FLOOR_ACTIVE spoofing_flag to
        signal downstream consumers that Model B is unavailable.
        """
        published: List[ModelAEvent] = []
        spoofing_flags = [_SAFETY_FLOOR_FLAG]

        # --- YOLO detection ---
        detections = self._detector.detect(frame, frame_number)

        # --- Animal filtering ---
        animal_dets, trigger_candidates = self._afilt.classify(detections)

        # Animal info events
        for adet in animal_dets:
            event = self._build_event(
                event_type       = EventType.animal_detected,
                severity         = Severity.info,
                det              = adet,
                frame_number     = frame_number,
                timestamp_utc    = timestamp_utc,
                spoofing_flags   = spoofing_flags,
                trigger_type     = None,
                confirmation_frames = 0,
            )
            if event:
                published.append(event)

        # Motion events for human/vehicle detections
        for det in trigger_candidates:
            motion = self._build_event(
                event_type       = EventType.motion,
                severity         = Severity.info,
                det              = det,
                frame_number     = frame_number,
                timestamp_utc    = timestamp_utc,
                spoofing_flags   = spoofing_flags,
                trigger_type     = None,
                confirmation_frames = 0,
            )
            if motion:
                published.append(motion)

        # --- Trigger detection (if hint provided) ---
        if trigger_type_hint is not None:
            for det in trigger_candidates:
                if det.track_id is None:
                    continue

                consistent = self._bbox.check(det.track_id, det.bbox)
                if not consistent:
                    self._trigger.miss(det.track_id)
                    continue

                result = self._trigger.update(det.track_id, trigger_type_hint, frame_number)
                if result.should_publish:
                    trigger_event = self._build_event(
                        event_type       = EventType.trigger,
                        severity         = result.severity,
                        det              = det,
                        frame_number     = frame_number,
                        timestamp_utc    = timestamp_utc,
                        spoofing_flags   = spoofing_flags,
                        trigger_type     = trigger_type_hint,
                        confirmation_frames = result.confirmation_frames,
                    )
                    if trigger_event:
                        published.append(trigger_event)

        return published

    def deactivate(self) -> None:
        """Call when Model B recovers and the safety floor is no longer needed."""
        logger.info(
            "SafetyFloor DEACTIVATED for camera '%s'. "
            "Model B engine has recovered — resuming full pipeline.",
            self.camera_id,
        )

    # ------------------------------------------------------------------
    # Internal event builder
    # ------------------------------------------------------------------

    def _build_event(
        self,
        event_type:          EventType,
        severity:            Severity,
        det,
        frame_number:        int,
        timestamp_utc:       str,
        spoofing_flags:      List[str],
        trigger_type:        Optional[TriggerType],
        confirmation_frames: int,
    ) -> Optional[ModelAEvent]:
        try:
            return ModelAEvent(
                event_type   = event_type,
                severity     = severity,
                timestamp    = timestamp_utc,
                camera_id    = self.camera_id,
                zone_tag     = self.zone_tag,
                zone         = self.zone,
                entity_type  = det.entity_type,
                entity_id    = det.track_id,
                confidence   = det.confidence,
                bbox         = det.bbox,
                evidence_ref = "pending",
                hash         = "pending",
                metadata     = EventMetadata(
                    model_version      = MODEL_VERSION,
                    processing_time_ms = 0,
                    frame_number       = frame_number,
                    trigger_type       = trigger_type,
                    confirmation_frames = confirmation_frames,
                    spoofing_flags     = spoofing_flags,
                ),
            )
        except Exception as exc:
            logger.error(
                "SafetyFloor schema validation failed for camera=%s frame=%d: %s "
                "— event rejected.",
                self.camera_id, frame_number, exc,
            )
            return None
