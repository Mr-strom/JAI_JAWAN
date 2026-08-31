"""
Frame Pipeline — Per-Frame Orchestrator (Model A Core)
SIH26187 | Model A | Ties all steps together

Full pipeline per frame:
  1. Preprocess (low-light enhancement if dark)
  2. Time-sample (MSE dedup — reject redundant frames)
  3. MDF selection (best frame in 1s window)
  4. Zone tagging (close_range / long_range + zone label)
  5. YOLO detection (entity_type, confidence, bbox, track_id)
  6. Animal filtering (split animals → info events; humans → trigger pipeline)
  7. Anti-spoofing check (timestamp, FPS, frame continuity)
  8. BBox consistency check (spatial guard against shadows/foliage)
  9. Trigger detection (IDLE→PROV_1→PROV_2→CONFIRMED state machine)
  10. Multi-camera fusion (global_fusion_id assignment)
  11. Schema validation (Pydantic ModelAEvent)
  12. MQTT publish (via BusClient)

Note: Health monitoring runs as a parallel concern (not in this hot path).
Note: Fallback routing (dead Model B heartbeat) is managed by HealthMonitor.
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from typing import List, Optional

import numpy as np

from model_a.animal_filter import AnimalFilter
from model_a.anti_spoofing import AntiSpoofingChecker
from model_a.bbox_consistency import BBoxConsistencyChecker
from model_a.bus_client import BusClient
from model_a.detector import Detection, Detector
from model_a.fusion_engine import FusionEngine
from model_a.preprocessor import Preprocessor
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
from model_a.time_sampler import TimeSampler
from model_a.trigger_detector import TriggerDetector, TriggerState
from model_a.zone_tagger import ZoneTagger

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"
EVIDENCE_DIR  = os.environ.get("EVIDENCE_DIR", "./evidence")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class FramePipeline:
    """
    Per-camera frame processing pipeline.

    Instantiate one FramePipeline per camera feed.

    Usage::

        pipe = FramePipeline(
            camera_id="cam_01",
            zone_tagger=ZoneTagger("cam_01", frame_height_px=1080),
            detector=Detector("yolov8n.pt"),
            bus_client=bus,
        )
        for frame_bgr in camera_stream:
            pipe.process(frame_bgr, frame_number=n, timestamp_utc=ts)
    """

    def __init__(
        self,
        camera_id:     str,
        zone_tagger:   ZoneTagger,
        detector:      Detector,
        bus_client:    BusClient,
        preprocessor:  Optional[Preprocessor]       = None,
        time_sampler:  Optional[TimeSampler]         = None,
        trigger_detector: Optional[TriggerDetector]  = None,
        fusion_engine: Optional[FusionEngine]        = None,
        anti_spoofing: Optional[AntiSpoofingChecker] = None,
        bbox_checker:  Optional[BBoxConsistencyChecker] = None,
        animal_filter: Optional[AnimalFilter]         = None,
    ) -> None:
        self.camera_id  = camera_id
        self._zone      = zone_tagger
        self._detector  = detector
        self._bus       = bus_client

        # Components with safe defaults
        self._pre   = preprocessor   or Preprocessor()
        self._samp  = time_sampler   or TimeSampler()
        self._trig  = trigger_detector or TriggerDetector(confirmation_frames=3)
        self._fuse  = fusion_engine   or FusionEngine()
        self._spoof = anti_spoofing   or AntiSpoofingChecker(camera_id)
        self._bbox  = bbox_checker    or BBoxConsistencyChecker()
        self._afilt = animal_filter   or AnimalFilter()

        os.makedirs(EVIDENCE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point — called for every raw frame from the RTSP stream
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_utc: str,
        trigger_type_override: Optional[TriggerType] = None,
    ) -> List[ModelAEvent]:
        """
        Run the full pipeline on one frame.

        Returns:
            List of validated ModelAEvent objects that were published.
            Empty list if the frame was skipped or produced no publishable events.
        """
        published: List[ModelAEvent] = []

        # --- Step 1: Low-light preprocessing ---
        frame = self._pre.enhance(frame)

        # --- Step 2: Time sampling (MSE dedup) ---
        accepted, frame = self._samp.accept(frame)
        if not accepted:
            return published  # redundant frame — skip

        # --- Step 3: MDF selection (flush at 1s window boundary) ---
        if self._samp.should_flush_window():
            best_frame = self._samp.flush_mdf()
            if best_frame is not None:
                frame = best_frame

        # --- Step 4: Anti-spoofing ---
        spoof_report = self._spoof.check(timestamp_utc, frame_number)
        # NOTE: spoof_report.is_suspicious does NOT suppress the event.
        # Flags are stored and forwarded in metadata. (Rule #8)

        # --- Step 5: YOLO detection ---
        detections = self._detector.detect(frame, frame_number)

        # --- Step 6: Animal filtering ---
        animal_dets, trigger_candidates = self._afilt.classify(detections)

        # Publish animal_detected info events
        for adet in animal_dets:
            event = self._build_animal_event(adet, frame_number, timestamp_utc, spoof_report.flags)
            self._publish(event)
            published.append(event)

        # --- Step 7: Motion event if any human/vehicle detected (no trigger yet) ---
        if trigger_candidates and not trigger_type_override:
            # Publish a provisional motion event for Model B awareness
            for det in trigger_candidates:
                zone_tag, zone = self._zone.tag(det.bbox)
                global_id = self._fuse.assign_or_merge(
                    self.camera_id, det.track_id or str(uuid.uuid4()), det.bbox, det.entity_type.value
                )
                motion_event = self._build_motion_event(
                    det, zone_tag, zone, global_id, frame_number, timestamp_utc, spoof_report.flags
                )
                self._publish(motion_event)
                published.append(motion_event)

        # --- Step 8 + 9: BBox consistency + Trigger detection ---
        for det in trigger_candidates:
            if det.track_id is None:
                continue  # no track ID → can't maintain per-track state

            ttype = trigger_type_override or self._infer_trigger_type(det)
            if ttype is None:
                continue  # no trigger signal detected for this entity

            # Spatial consistency check (shadow/foliage suppressor)
            spatially_consistent = self._bbox.check(det.track_id, det.bbox)

            if not spatially_consistent:
                # BBox jumped — spatial discontinuity. Reset confirmation.
                self._trig.miss(det.track_id)
                logger.debug(
                    "cam=%s track=%s: spatial discontinuity → miss() called, confirmation reset.",
                    self.camera_id, det.track_id,
                )
                continue

            # Multi-frame confirmation (state machine)
            result = self._trig.update(det.track_id, ttype, frame_number)

            if not result.should_publish:
                continue  # still provisional or in cooldown

            # At this point we have severity >= provisional that should be published
            zone_tag, zone = self._zone.tag(det.bbox)
            global_id = self._fuse.assign_or_merge(
                self.camera_id,
                det.track_id,
                det.bbox,
                det.entity_type.value,
            )

            trigger_event = self._build_trigger_event(
                det         = det,
                trigger_type = ttype,
                result_severity = result.severity,
                confirmation_frames = result.confirmation_frames,
                zone_tag    = zone_tag,
                zone        = zone,
                global_id   = global_id,
                frame_number = frame_number,
                timestamp_utc = timestamp_utc,
                spoofing_flags = spoof_report.flags,
            )

            if trigger_event is not None:
                self._publish(trigger_event)
                published.append(trigger_event)

        # Periodic stale purge
        self._trig.purge_stale()
        self._fuse.purge_stale()

        return published

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    def _build_animal_event(
        self,
        det: Detection,
        frame_number: int,
        timestamp_utc: str,
        spoofing_flags: list[str],
    ) -> ModelAEvent:
        zone_tag, zone = self._zone.tag(det.bbox)
        return ModelAEvent(
            event_type   = EventType.animal_detected,
            severity     = Severity.info,
            timestamp    = timestamp_utc,
            camera_id    = self.camera_id,
            zone_tag     = zone_tag,
            zone         = zone,
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
                trigger_type       = None,
                confirmation_frames = 0,
                spoofing_flags     = spoofing_flags,
            ),
        )

    def _build_motion_event(
        self,
        det: Detection,
        zone_tag: ZoneTag,
        zone: Zone,
        entity_id: str,
        frame_number: int,
        timestamp_utc: str,
        spoofing_flags: list[str],
    ) -> ModelAEvent:
        return ModelAEvent(
            event_type   = EventType.motion,
            severity     = Severity.info,
            timestamp    = timestamp_utc,
            camera_id    = self.camera_id,
            zone_tag     = zone_tag,
            zone         = zone,
            entity_type  = det.entity_type,
            entity_id    = entity_id,
            confidence   = det.confidence,
            bbox         = det.bbox,
            evidence_ref = "pending",
            hash         = "pending",
            metadata     = EventMetadata(
                model_version      = MODEL_VERSION,
                processing_time_ms = 0,
                frame_number       = frame_number,
                trigger_type       = None,
                confirmation_frames = 0,
                spoofing_flags     = spoofing_flags,
            ),
        )

    def _build_trigger_event(
        self,
        det: Detection,
        trigger_type: TriggerType,
        result_severity,
        confirmation_frames: int,
        zone_tag: ZoneTag,
        zone: Zone,
        global_id: str,
        frame_number: int,
        timestamp_utc: str,
        spoofing_flags: list[str],
    ) -> Optional[ModelAEvent]:
        """
        Build and validate a trigger event.
        Returns None if Pydantic validation fails (malformed → logged, not published).
        """
        try:
            return ModelAEvent(
                event_type   = EventType.trigger,
                severity     = result_severity,
                timestamp    = timestamp_utc,
                camera_id    = self.camera_id,
                zone_tag     = zone_tag,
                zone         = zone,
                entity_type  = det.entity_type,
                entity_id    = global_id,
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
            # Schema validation failed — log and REJECT. Never publish malformed.
            logger.error(
                "SCHEMA VALIDATION FAILED for trigger event: %s — event REJECTED, not published.",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Publish helper
    # ------------------------------------------------------------------

    def _publish(self, event: ModelAEvent) -> None:
        """Publish to MQTT bus. Errors are logged; pipeline continues."""
        try:
            self._bus.publish_event(event)
        except Exception as exc:
            logger.error(
                "PUBLISH FAILED for event_id=%s: %s. Event lost.", event.event_id, exc
            )

    # ------------------------------------------------------------------
    # Trigger type inference (stub — real version uses posture model output)
    # ------------------------------------------------------------------

    def _infer_trigger_type(self, det: Detection) -> Optional[TriggerType]:
        """
        Placeholder trigger type inference.
        In production, this is fed by Model B posture signals or
        geometric analysis (rapid approach = bbox growing fast).
        For Phase Berlin, callers inject trigger_type_override directly.
        Returns None if no trigger signal is detected.
        """
        return None  # Phase Berlin: injected externally via trigger_type_override
