"""
Live Gap Remediation Tests for Model A
SIH26187 | Model A Ingestion & Tagging Layer

Verifies:
  1. Real evidence saving to ./evidence/{camera_id}/{event_id}.jpg
  2. Real SHA-256 hash computation matching disk image file
  3. Real non-zero processing_time_ms measurement
  4. FallbackRouter live integration: heartbeat timeout -> fallback_active=True
  5. Blur quality gating: Laplacian variance calculation and flagging
  6. Real vehicle footage processing from video_test/vehicle_clip.mp4
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import List

import cv2
import numpy as np
import pytest

from model_a.detector import Detection, Detector
from model_a.fallback_router import FallbackRouter
from model_a.frame_pipeline import FramePipeline
from model_a.preprocessor import Preprocessor
from model_a.schema_v1 import (
    EntityType,
    EventType,
    ModelAEvent,
    TriggerType,
)
from model_a.zone_tagger import ZoneTagger


class _CapturingBusClient:
    """In-memory bus client capturing published events for test assertions."""

    def __init__(self) -> None:
        self.published: List[ModelAEvent] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def publish_event(self, event: ModelAEvent) -> None:
        self.published.append(event)


class _MockPersonDetector:
    """Mock detector returning a fixed human detection for pipeline tests."""

    def __init__(self, confidence: float = 0.88) -> None:
        self.confidence = confidence

    def detect(self, frame: np.ndarray, frame_number: int) -> List[Detection]:
        return [
            Detection(
                track_id="trk_person_gap",
                entity_type=EntityType.human,
                confidence=self.confidence,
                bbox=[0.40, 0.50, 0.60, 0.90],
                class_id=0,
                class_name="person",
                frame_number=frame_number,
            )
        ]

    def detect_full_frame_long_range(
        self, frame: np.ndarray, frame_number: int
    ) -> List[Detection]:
        return []


class TestModelALiveGaps:

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        self.bus = _CapturingBusClient()
        self.cam_id = "cam_gap_test"
        self.tagger = ZoneTagger(self.cam_id, frame_height_px=480)
        self.detector = _MockPersonDetector()
        yield

    def test_gap_01_real_evidence_ref_and_sha256_hash(self):
        """
        TASK 1: evidence_ref and hash must point to a real file on disk
        with a genuine SHA-256 hash matching the disk contents.
        """
        pipe = FramePipeline(
            camera_id=self.cam_id,
            zone_tagger=self.tagger,
            detector=self.detector,
            bus_client=self.bus,
        )

        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        events = pipe.process(
            test_frame,
            frame_number=1,
            timestamp_utc="2026-09-04T12:00:00.000Z",
        )

        assert len(events) >= 1, "Expected at least one published event"
        event = events[0]

        # Verify not "pending"
        assert event.evidence_ref != "pending", "evidence_ref cannot be 'pending'"
        assert event.hash != "pending", "hash cannot be 'pending'"

        # Verify file exists on disk
        assert os.path.isfile(event.evidence_ref), (
            f"Evidence file does not exist: {event.evidence_ref}"
        )

        # Verify path matches ./evidence/{camera_id}/{event_id}.jpg structure
        assert self.cam_id in event.evidence_ref
        assert event.evidence_ref.endswith(f"{event.event_id}.jpg")

        # Verify SHA-256 matches actual file on disk
        hasher = hashlib.sha256()
        with open(event.evidence_ref, "rb") as f:
            hasher.update(f.read())
        expected_hash = hasher.hexdigest()

        assert event.hash == expected_hash, "Event hash must match disk file SHA-256"
        assert len(event.hash) == 64, "SHA-256 must be a 64-char hex string"

    def test_gap_02_real_processing_time_ms_nonzero(self):
        """
        TASK 2: processing_time_ms must be measured and strictly non-zero.
        """
        pipe = FramePipeline(
            camera_id=self.cam_id,
            zone_tagger=self.tagger,
            detector=self.detector,
            bus_client=self.bus,
        )

        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        events = pipe.process(
            test_frame,
            frame_number=1,
            timestamp_utc="2026-09-04T12:00:00.000Z",
        )

        assert len(events) >= 1
        event = events[0]

        # Verify real non-zero timing
        assert isinstance(event.metadata.processing_time_ms, int)
        assert event.metadata.processing_time_ms > 0, (
            "processing_time_ms must be measured and > 0"
        )

    def test_gap_03_fallback_router_heartbeat_timeout_flips_fallback_active(self):
        """
        TASK 3: FallbackRouter heartbeat timeout causes fallback_active to flip to True
        and SAFETY_FLOOR_ACTIVE flag to be present in event metadata.
        """
        timeout_s = 0.05
        router = FallbackRouter(heartbeat_timeout_s=timeout_s)
        router.register_engine("face_engine", cameras=[self.cam_id])

        pipe = FramePipeline(
            camera_id=self.cam_id,
            zone_tagger=self.tagger,
            detector=self.detector,
            bus_client=self.bus,
            fallback_router=router,
        )

        # 1. Normal state: heartbeat is fresh
        router.update_heartbeat("face_engine")
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        events_normal = pipe.process(
            test_frame, frame_number=1, timestamp_utc="2026-09-04T12:00:00.000Z"
        )
        assert len(events_normal) >= 1
        assert events_normal[0].metadata.fallback_active is False
        assert "SAFETY_FLOOR_ACTIVE" not in events_normal[0].metadata.spoofing_flags

        # 2. Engine goes silent: wait beyond timeout
        time.sleep(timeout_s * 1.5)

        # 3. Next frame process with new frame data (so MSE dedup accepts it)
        test_frame2 = np.full((480, 640, 3), 160, dtype=np.uint8)
        events_fallback = pipe.process(
            test_frame2, frame_number=2, timestamp_utc="2026-09-04T12:00:01.000Z"
        )
        assert len(events_fallback) >= 1
        ev = events_fallback[0]
        assert ev.metadata.fallback_active is True, (
            "fallback_active must be True when engine heartbeat times out"
        )
        assert "SAFETY_FLOOR_ACTIVE" in ev.metadata.spoofing_flags, (
            "SAFETY_FLOOR_ACTIVE must be added to spoofing_flags"
        )

    def test_gap_04_motion_blur_gating_and_metadata(self):
        """
        TASK 4: Sharp frame has is_blurry=False; blurred frame has is_blurry=True
        and carries FRAME_BLURRED in spoofing_flags without dropping the frame.
        """
        pipe = FramePipeline(
            camera_id=self.cam_id,
            zone_tagger=self.tagger,
            detector=self.detector,
            bus_client=self.bus,
        )

        # Sharp frame with high-frequency edges
        sharp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sharp_frame[::8, :] = 255
        sharp_frame[:, ::8] = 255

        events_sharp = pipe.process(
            sharp_frame, frame_number=1, timestamp_utc="2026-09-04T12:00:00.000Z"
        )
        assert len(events_sharp) >= 1
        assert events_sharp[0].metadata.is_blurry is False
        assert events_sharp[0].metadata.blur_score is not None
        assert events_sharp[0].metadata.blur_score > 100.0
        assert "FRAME_BLURRED" not in events_sharp[0].metadata.spoofing_flags

        # Heavily blurred frame
        blurred_frame = cv2.GaussianBlur(sharp_frame, (51, 51), 0)
        events_blurred = pipe.process(
            blurred_frame, frame_number=2, timestamp_utc="2026-09-04T12:00:01.000Z"
        )
        assert len(events_blurred) >= 1
        assert events_blurred[0].metadata.is_blurry is True
        assert events_blurred[0].metadata.blur_score < 100.0
        assert "FRAME_BLURRED" in events_blurred[0].metadata.spoofing_flags

    def test_gap_05_real_vehicle_clip_ingestion(self):
        """
        TASK 5: Ingest real vehicle footage from video_test/vehicle_clip.mp4.
        """
        video_path = "video_test/vehicle_clip.mp4"
        assert os.path.isfile(video_path), f"Missing {video_path}"

        cap = cv2.VideoCapture(video_path)
        # Seek to frame 62 where vehicles are actively driving through the scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, 61)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Failed to read vehicle frame from vehicle_clip.mp4"

        h, w = frame.shape[:2]
        pipe = FramePipeline(
            camera_id="cam_vehicle_live",
            zone_tagger=ZoneTagger("cam_vehicle_live", frame_height_px=h),
            detector=Detector("yolov8n.pt"),
            bus_client=self.bus,
        )

        events = pipe.process(
            frame, frame_number=62, timestamp_utc="2026-09-04T12:00:00.000Z"
        )

        # Frame should process and publish schema-valid events with vehicle entity
        assert len(events) >= 1
        assert any(e.entity_type == EntityType.vehicle for e in events), (
            "Expected vehicle entity detected in vehicle_clip.mp4 frame 62"
        )
        ev = events[0]
        assert ev.evidence_ref != "pending"
        assert os.path.isfile(ev.evidence_ref)
        assert ev.hash != "pending"
        assert len(ev.hash) == 64
        assert ev.metadata.processing_time_ms > 0
        assert isinstance(ev.metadata.fallback_active, bool)
        assert isinstance(ev.metadata.is_blurry, bool)
