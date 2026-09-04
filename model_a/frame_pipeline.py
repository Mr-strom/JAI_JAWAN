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
import time
import uuid
from typing import List, Optional

import numpy as np
import cv2

from model_a.animal_filter import AnimalFilter
from model_a.animal_cart_fuser import AnimalCartFuser
from model_a.anti_spoofing import AntiSpoofingChecker
from model_a.fallback_router import FallbackRouter
from model_a.homography import HomographyCorrector
from model_a.bbox_consistency import BBoxConsistencyChecker
from model_a.bus_client import BusClient
from model_a.detector import Detection, Detector
from model_a.fusion_engine import FusionEngine
from model_a.preprocessor import Preprocessor
from model_a.safety_floor import SafetyFloor, _SAFETY_FLOOR_FLAG
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
        animal_cart_fuser: Optional[AnimalCartFuser]   = None,
        homography: Optional[HomographyCorrector]      = None,
        fallback_router: Optional[FallbackRouter]      = None,
        safety_floor: Optional[SafetyFloor]            = None,
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
        self._cart_fuse = animal_cart_fuser or AnimalCartFuser()
        # Homography is purely optional — None means no perspective correction
        self._homography: Optional[HomographyCorrector] = homography

        self._fallback_router: Optional[FallbackRouter] = fallback_router
        self._safety_floor: SafetyFloor = safety_floor or SafetyFloor(
            camera_id=camera_id,
            detector=detector,
            animal_filter=self._afilt,
            bbox_checker=self._bbox,
            trigger_detector=self._trig,
        )
        self._latest_motion_evidence: dict[str, tuple[str, str, float]] = {}

        os.makedirs(EVIDENCE_DIR, exist_ok=True)

    def _save_evidence(self, frame: Optional[np.ndarray], event_id: str) -> tuple[str, str]:
        """
        Save frame image to EVIDENCE_DIR/{camera_id}/{event_id}.jpg and compute SHA-256.
        Returns (evidence_ref, hash).
        """
        cam_dir = os.path.join(EVIDENCE_DIR, self.camera_id)
        os.makedirs(cam_dir, exist_ok=True)
        rel_path = os.path.join(EVIDENCE_DIR, self.camera_id, f"{event_id}.jpg").replace("\\", "/")
        try:
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                cv2.imwrite(rel_path, frame)
            else:
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(rel_path, dummy)
            file_hash = ModelAEvent.compute_hash(rel_path)
            return rel_path, file_hash
        except Exception as exc:
            logger.error("Failed to write evidence frame for event %s: %s", event_id, exc)
            return rel_path, "HASH_FAILED"

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
        t0_start = time.perf_counter()

        def _calc_latency_ms() -> int:
            return max(1, int(round((time.perf_counter() - t0_start) * 1000)))

        # Quality gating: Check motion blur via Laplacian variance
        is_blurry, blur_score = self._pre.check_blur(frame)

        # Fallback router check: evaluate staleness & check if camera is in fallback
        is_fallback = False
        if self._fallback_router is not None:
            self._fallback_router.evaluate()
            if self.camera_id in self._fallback_router.get_fallback_cameras():
                is_fallback = True

        # --- Step 1: Low-light preprocessing ---
        frame = self._pre.enhance(frame)

        # --- Step 1b: Homography perspective correction (chokepoint/ICP cameras) ---
        # Applied before YOLO so bbox coordinates are already in the corrected space.
        # No-op if this camera_id has no calibration entry.
        if self._homography is not None:
            frame = self._homography.correct_frame(self.camera_id, frame)

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
        if is_fallback and _SAFETY_FLOOR_FLAG not in spoof_report.flags:
            spoof_report.flags.append(_SAFETY_FLOOR_FLAG)
        if is_blurry and "FRAME_BLURRED" not in spoof_report.flags:
            spoof_report.flags.append("FRAME_BLURRED")

        # --- Step 5: Detection — dual-zone routing ---
        #
        # STRATEGY:
        #   close_range detections → YOLOv8n (fast; reliable for large bboxes)
        #   long_range  detections → YOLOv8s (better sensitivity for small/distant humans)
        #
        # We cannot do "upgrade only what v8n found" because v8n MISSES the person
        # entirely at low confidence — there is no bbox to upgrade from.
        # Instead:
        #   Pass 1: v8n full-frame  → keep all close_range hits.
        #   Pass 2: v8s full-frame  → keep all long_range hits.
        #   Final list: union of both, deduplicated by zone.
        #
        t0_v8n = time.perf_counter()
        v8n_dets = self._detector.detect(frame, frame_number)
        t_v8n_ms = (time.perf_counter() - t0_v8n) * 1000

        # Classify v8n results by zone; keep only close_range ones.
        close_range_dets = []
        has_any_long_range_v8n = False
        for det in v8n_dets:
            zone_tag_val, _ = self._zone.tag(det.bbox)
            if zone_tag_val.value == "close_range":
                close_range_dets.append(det)
            else:
                has_any_long_range_v8n = True  # v8n saw something distant (weak signal)

        # Pass 2: run v8s on the full frame and keep only long_range results.
        # We always run v8s here because v8n may have MISSED the distant person.
        t0_v8s = time.perf_counter()
        v8s_dets = self._detector.detect_full_frame_long_range(frame, frame_number)
        t_v8s_ms = (time.perf_counter() - t0_v8s) * 1000

        long_range_dets = []
        for det in v8s_dets:
            zone_tag_val, _ = self._zone.tag(det.bbox)
            if zone_tag_val.value == "long_range":
                long_range_dets.append(det)

        detections = close_range_dets + long_range_dets

        logger.debug(
            "cam=%s frame=%d | v8n=%.1fms(%d close) | v8s=%.1fms(%d long)",
            self.camera_id, frame_number,
            t_v8n_ms, len(close_range_dets),
            t_v8s_ms, len(long_range_dets),
        )

        # --- Step 6: Animal filtering ---
        animal_dets, trigger_candidates = self._afilt.classify(detections)

        # --- Step 6b: Animal-cart proximity fusion ---
        # Separate vehicles from trigger_candidates so the fuser can check
        # animal×vehicle proximity. Non-vehicle triggers are unaffected.
        vehicle_dets    = [d for d in trigger_candidates
                           if d.entity_type == EntityType.vehicle]
        non_veh_triggers = [d for d in trigger_candidates
                            if d.entity_type != EntityType.vehicle]

        animal_dets, vehicle_dets, cart_dets = self._cart_fuse.fuse(
            animal_dets, vehicle_dets, frame_number
        )

        # Merge vehicle dets (unfused) back into trigger_candidates
        trigger_candidates = non_veh_triggers + vehicle_dets

        # Publish animal_detected info events for plain animals
        for adet in animal_dets:
            event = self._build_animal_event(
                adet, frame_number, timestamp_utc, spoof_report.flags,
                frame=frame, processing_time_ms=_calc_latency_ms(),
                fallback_active=is_fallback, blur_score=blur_score, is_blurry=is_blurry
            )
            self._publish(event)
            published.append(event)

        # Publish animal_detected info events for confirmed animal_cart detections
        for cdet in cart_dets:
            event = self._build_animal_event(
                cdet, frame_number, timestamp_utc, spoof_report.flags,
                frame=frame, processing_time_ms=_calc_latency_ms(),
                fallback_active=is_fallback, blur_score=blur_score, is_blurry=is_blurry
            )
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
                    det, zone_tag, zone, global_id, frame_number, timestamp_utc, spoof_report.flags,
                    frame=frame, processing_time_ms=_calc_latency_ms(),
                    fallback_active=is_fallback, blur_score=blur_score, is_blurry=is_blurry
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
                det                 = det,
                trigger_type        = ttype,
                result_severity     = result.severity,
                confirmation_frames = result.confirmation_frames,
                zone_tag            = zone_tag,
                zone                = zone,
                global_id           = global_id,
                frame_number        = frame_number,
                timestamp_utc       = timestamp_utc,
                spoofing_flags      = spoof_report.flags,
                frame               = frame,
                processing_time_ms  = _calc_latency_ms(),
                fallback_active     = is_fallback,
                blur_score          = blur_score,
                is_blurry           = is_blurry,
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
        frame: Optional[np.ndarray] = None,
        processing_time_ms: int = 1,
        fallback_active: bool = False,
        blur_score: Optional[float] = None,
        is_blurry: bool = False,
    ) -> ModelAEvent:
        zone_tag, zone = self._zone.tag(det.bbox)
        event_id = str(uuid.uuid4())
        evidence_ref, file_hash = self._save_evidence(frame, event_id)
        return ModelAEvent(
            event_id     = event_id,
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
            evidence_ref = evidence_ref,
            hash         = file_hash,
            metadata     = EventMetadata(
                model_version       = MODEL_VERSION,
                processing_time_ms  = max(1, processing_time_ms),
                frame_number        = frame_number,
                trigger_type        = None,
                confirmation_frames = 0,
                spoofing_flags      = spoofing_flags,
                fallback_active     = fallback_active,
                blur_score          = blur_score,
                is_blurry           = is_blurry,
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
        frame: Optional[np.ndarray] = None,
        processing_time_ms: int = 1,
        fallback_active: bool = False,
        blur_score: Optional[float] = None,
        is_blurry: bool = False,
    ) -> ModelAEvent:
        event_id = str(uuid.uuid4())
        now_mono = time.monotonic()
        last_entry = self._latest_motion_evidence.get(entity_id)

        # Rate-limit motion image saving: 1 keyframe per second per entity
        if last_entry is None or (now_mono - last_entry[2]) >= 1.0:
            evidence_ref, file_hash = self._save_evidence(frame, event_id)
            self._latest_motion_evidence[entity_id] = (evidence_ref, file_hash, now_mono)
        else:
            evidence_ref, file_hash, _ = last_entry

        return ModelAEvent(
            event_id     = event_id,
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
            evidence_ref = evidence_ref,
            hash         = file_hash,
            metadata     = EventMetadata(
                model_version       = MODEL_VERSION,
                processing_time_ms  = max(1, processing_time_ms),
                frame_number        = frame_number,
                trigger_type        = None,
                confirmation_frames = 0,
                spoofing_flags      = spoofing_flags,
                fallback_active     = fallback_active,
                blur_score          = blur_score,
                is_blurry           = is_blurry,
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
        frame: Optional[np.ndarray] = None,
        processing_time_ms: int = 1,
        fallback_active: bool = False,
        blur_score: Optional[float] = None,
        is_blurry: bool = False,
    ) -> Optional[ModelAEvent]:
        """
        Build and validate a trigger event.
        Returns None if Pydantic validation fails (malformed → logged, not published).
        """
        try:
            event_id = str(uuid.uuid4())
            evidence_ref, file_hash = self._save_evidence(frame, event_id)
            return ModelAEvent(
                event_id     = event_id,
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
                evidence_ref = evidence_ref,
                hash         = file_hash,
                metadata     = EventMetadata(
                    model_version       = MODEL_VERSION,
                    processing_time_ms  = max(1, processing_time_ms),
                    frame_number        = frame_number,
                    trigger_type        = trigger_type,
                    confirmation_frames = min(10, confirmation_frames),
                    spoofing_flags      = spoofing_flags,
                    fallback_active     = fallback_active,
                    blur_score          = blur_score,
                    is_blurry           = is_blurry,
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
