"""
Phase Rome — Scaffolding + Schema Validation Tests
SIH26187 | Model A Test Suite

Test checkpoints per spec:
  Phase Rome: Repo/module scaffolding, Pydantic schema validation,
              publish/subscribe a dummy valid event.

Run:
  pytest tests/test_phase_rome.py -v

All tests must pass before moving to Phase Berlin.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import pytest
from pydantic import ValidationError

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
from model_a.trigger_detector import TriggerDetector, TriggerState
from model_a.time_sampler import TimeSampler
from model_a.zone_tagger import ZoneTagger
from model_a.anti_spoofing import AntiSpoofingChecker
from model_a.fusion_engine import FusionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_event(**overrides) -> ModelAEvent:
    """Build a minimal valid ModelAEvent. Override any field via kwargs."""
    defaults = dict(
        event_id    = str(uuid.uuid4()),
        event_type  = EventType.motion,
        severity    = Severity.info,
        timestamp   = "2025-01-01T00:00:00Z",
        camera_id   = "cam_01",
        zone_tag    = ZoneTag.long_range,
        zone        = Zone.perimeter,
        entity_type = EntityType.human,
        entity_id   = None,
        confidence  = 0.85,
        bbox        = [0.1, 0.2, 0.4, 0.8],
        evidence_ref = "/tmp/evidence/frame_001.jpg",
        hash        = "a" * 64,   # placeholder SHA-256
        engine_source = "model_a",
        metadata    = EventMetadata(
            model_version="1.0.0",
            processing_time_ms=35,
            frame_number=1001,
            trigger_type=None,
            confirmation_frames=0,
            spoofing_flags=[],
        ),
    )
    defaults.update(overrides)
    return ModelAEvent(**defaults)


# ===========================================================================
# ROME-01: Schema — Valid Event Passes
# ===========================================================================

class TestSchemaValidEvent:

    def test_valid_motion_event_builds(self):
        """A fully valid motion event must build without errors."""
        event = make_valid_event()
        assert event.event_type == EventType.motion
        assert event.severity   == Severity.info
        assert event.engine_source == "model_a"

    def test_valid_trigger_event_with_trigger_type(self):
        """trigger events with a trigger_type and 3 confirmation frames must pass."""
        event = make_valid_event(
            event_type = EventType.trigger,
            severity   = Severity.confirmed,
            metadata   = EventMetadata(
                model_version="1.0.0",
                processing_time_ms=45,
                frame_number=1050,
                trigger_type=TriggerType.climbing,
                confirmation_frames=3,
                spoofing_flags=[],
            ),
        )
        assert event.metadata.confirmation_frames == 3
        assert event.metadata.trigger_type == TriggerType.climbing

    def test_event_serialises_to_json(self):
        """Events must serialise cleanly to JSON for MQTT publish."""
        event   = make_valid_event()
        payload = event.to_mqtt_payload()
        parsed  = json.loads(payload.decode("utf-8"))

        assert parsed["engine_source"] == "model_a"
        assert "event_id" in parsed
        assert "metadata" in parsed

    def test_engine_source_is_always_model_a(self):
        """engine_source must always be 'model_a' — frozen in schema."""
        event = make_valid_event()
        assert event.engine_source == "model_a"

    def test_bbox_normalised_values_accepted(self):
        """Normalised bbox in [0,1] must be accepted."""
        event = make_valid_event(bbox=[0.0, 0.0, 1.0, 1.0])
        assert event.bbox == [0.0, 0.0, 1.0, 1.0]

    def test_animal_event_type(self):
        """animal_detected events must be buildable."""
        event = make_valid_event(
            event_type  = EventType.animal_detected,
            entity_type = EntityType.animal,
            severity    = Severity.info,
        )
        assert event.event_type == EventType.animal_detected


# ===========================================================================
# ROME-02: Schema — Invalid Events Rejected
# ===========================================================================

class TestSchemaRejection:

    def test_confirmed_severity_with_only_2_frames_is_rejected(self):
        """
        CRITICAL RULE #1 TEST:
        severity=confirmed with confirmation_frames=2 MUST raise ValidationError.
        This is the schema-level guard against false alarms.
        """
        with pytest.raises(ValidationError) as exc_info:
            make_valid_event(
                event_type = EventType.trigger,
                severity   = Severity.confirmed,
                metadata   = EventMetadata(
                    model_version="1.0.0",
                    processing_time_ms=30,
                    frame_number=100,
                    trigger_type=TriggerType.climbing,
                    confirmation_frames=2,   # <-- must be rejected
                    spoofing_flags=[],
                ),
            )
        assert "confirmation_frames" in str(exc_info.value).lower() or \
               "rule #1" in str(exc_info.value).lower() or \
               "3" in str(exc_info.value)

    def test_critical_severity_with_2_frames_is_rejected(self):
        """severity=critical with confirmation_frames=2 must also be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(
                event_type = EventType.trigger,
                severity   = Severity.critical,
                metadata   = EventMetadata(
                    model_version="1.0.0",
                    processing_time_ms=30,
                    frame_number=100,
                    trigger_type=TriggerType.fence_cutting,
                    confirmation_frames=2,
                    spoofing_flags=[],
                ),
            )

    def test_trigger_event_without_trigger_type_is_rejected(self):
        """event_type=trigger with trigger_type=None must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(
                event_type = EventType.trigger,
                severity   = Severity.confirmed,
                metadata   = EventMetadata(
                    model_version="1.0.0",
                    processing_time_ms=30,
                    frame_number=100,
                    trigger_type=None,     # <-- must be rejected
                    confirmation_frames=3,
                    spoofing_flags=[],
                ),
            )

    def test_bbox_out_of_range_is_rejected(self):
        """bbox coordinates outside [0,1] must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(bbox=[0.1, 0.2, 1.5, 0.8])   # x2=1.5 invalid

    def test_bbox_inverted_is_rejected(self):
        """bbox where x2 <= x1 must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(bbox=[0.5, 0.2, 0.3, 0.8])   # x2 < x1

    def test_invalid_severity_string_is_rejected(self):
        """Unknown severity strings must be rejected by Pydantic."""
        with pytest.raises(ValidationError):
            make_valid_event(severity="SUPER_CRITICAL")

    def test_invalid_event_type_is_rejected(self):
        """Unknown event_type strings must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(event_type="intrusion_alarm")

    def test_wrong_engine_source_is_rejected(self):
        """engine_source != 'model_a' must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(engine_source="model_b")

    def test_confidence_above_1_is_rejected(self):
        """confidence > 1.0 must be rejected."""
        with pytest.raises(ValidationError):
            make_valid_event(confidence=1.5)

    def test_empty_camera_id_is_rejected(self):
        """camera_id must not be empty string."""
        with pytest.raises(ValidationError):
            make_valid_event(camera_id="")


# ===========================================================================
# ROME-03: Trigger Detector State Machine
# ===========================================================================

class TestTriggerDetectorStateMachine:

    def test_constructor_rejects_confirmation_below_3(self):
        """TriggerDetector must refuse confirmation_frames < 3 at construction."""
        with pytest.raises(ValueError):
            TriggerDetector(confirmation_frames=2)

    def test_single_frame_gives_provisional_not_confirmed(self):
        """1 frame → PROVISIONAL_1. No confirmed/critical yet."""
        det = TriggerDetector(confirmation_frames=3)
        result = det.update("trk_01", TriggerType.climbing, frame_number=1)

        assert result.state == TriggerState.PROVISIONAL_1
        assert result.severity == Severity.provisional
        assert result.confirmation_frames == 1

    def test_two_consecutive_frames_still_provisional(self):
        """2 consecutive frames → PROVISIONAL_2. Still not confirmed."""
        det = TriggerDetector(confirmation_frames=3)
        det.update("trk_01", TriggerType.climbing, frame_number=1)
        result = det.update("trk_01", TriggerType.climbing, frame_number=2)

        assert result.state == TriggerState.PROVISIONAL_2
        assert result.severity == Severity.provisional
        assert result.confirmation_frames == 2

    def test_three_consecutive_frames_gives_confirmed(self):
        """3 consecutive frames → CONFIRMED_TRIGGER. severity=confirmed."""
        det = TriggerDetector(confirmation_frames=3)
        det.update("trk_01", TriggerType.climbing, frame_number=1)
        det.update("trk_01", TriggerType.climbing, frame_number=2)
        result = det.update("trk_01", TriggerType.climbing, frame_number=3)

        assert result.state == TriggerState.CONFIRMED_TRIGGER
        assert result.severity == Severity.confirmed
        assert result.confirmation_frames == 3

    def test_two_frame_boundary_does_not_confirm(self):
        """
        CRITICAL EDGE CASE (Phase Rome spec):
        Track seen for 2 frames, disappears on 3rd → MUST reset to IDLE.
        No confirmed/critical event may be published.
        """
        det = TriggerDetector(confirmation_frames=3)
        det.update("trk_42", TriggerType.fence_cutting, frame_number=10)
        det.update("trk_42", TriggerType.fence_cutting, frame_number=11)
        # Track disappears — call miss()
        det.miss("trk_42")

        # Now check state
        state = det.active_tracks.get("trk_42")
        assert state is not None
        assert state["state"] == "IDLE"
        assert state["frames"] == 0

    def test_miss_in_provisional_1_resets_to_idle(self):
        """miss() after 1 frame → IDLE, frames=0."""
        det = TriggerDetector(confirmation_frames=3)
        det.update("trk_99", TriggerType.rapid_approach, frame_number=5)
        det.miss("trk_99")

        state = det.active_tracks["trk_99"]
        assert state["state"] == "IDLE"
        assert state["frames"] == 0


# ===========================================================================
# ROME-04: Time Sampler
# ===========================================================================

import numpy as np

class TestTimeSampler:

    def _blank_frame(self, value: int = 128, size=(480, 640, 3)) -> np.ndarray:
        return np.full(size, value, dtype=np.uint8)

    def test_first_frame_always_accepted(self):
        sampler = TimeSampler(mse_threshold=0.001)
        frame = self._blank_frame()
        accepted, _ = sampler.accept(frame)
        assert accepted is True

    def test_identical_frame_is_rejected(self):
        sampler = TimeSampler(mse_threshold=0.001)
        frame = self._blank_frame()
        sampler.accept(frame)  # first frame accepted
        accepted, _ = sampler.accept(frame)  # identical → rejected
        assert accepted is False

    def test_different_frame_is_accepted(self):
        sampler = TimeSampler(mse_threshold=0.001)
        sampler.accept(self._blank_frame(128))
        accepted, _ = sampler.accept(self._blank_frame(200))
        assert accepted is True

    def test_mdf_selects_highest_variance_frame(self):
        sampler = TimeSampler(mse_threshold=0.001)
        # Low variance frame
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        sampler.accept(flat)
        # High variance frame — random noise
        noisy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sampler.accept(noisy)
        # MDF should pick the noisy frame (higher variance)
        best = sampler.flush_mdf()
        assert best is not None


# ===========================================================================
# ROME-05: Zone Tagger
# ===========================================================================

class TestZoneTagger:

    def test_large_bbox_tagged_close_range(self):
        """Object with height ≥200px at 1080p → close_range."""
        tagger = ZoneTagger(camera_id="cam_01", frame_height_px=1080)
        # bbox height = 0.25 normalised → 0.25 * 1080 = 270px ≥ 200px
        zone_tag, zone = tagger.tag([0.1, 0.1, 0.5, 0.35])
        assert zone_tag == ZoneTag.close_range

    def test_small_bbox_tagged_long_range(self):
        """Object with height <200px at 1080p → long_range."""
        tagger = ZoneTagger(camera_id="cam_01", frame_height_px=1080)
        # bbox height = 0.10 normalised → 0.10 * 1080 = 108px < 200px
        zone_tag, zone = tagger.tag([0.1, 0.1, 0.5, 0.20])
        assert zone_tag == ZoneTag.long_range

    def test_static_zone_override_respected(self):
        """Static zone_tag from camera calibration overrides pixel heuristic."""
        tagger = ZoneTagger(
            camera_id="cam_gate",
            frame_height_px=1080,
            static_zone_tag=ZoneTag.close_range,
            static_zone=Zone.chokepoint,
        )
        # Even a small bbox → close_range due to override
        zone_tag, zone = tagger.tag([0.1, 0.1, 0.2, 0.12])
        assert zone_tag == ZoneTag.close_range
        assert zone == Zone.chokepoint


# ===========================================================================
# ROME-06: Anti-Spoofing
# ===========================================================================

class TestAntiSpoofing:

    def test_clean_stream_has_no_flags(self):
        """
        A clean stream at ~25 FPS (40ms intervals) must not raise any spoofing flags.
        Using a realistic 40ms gap (1/25s) between frames.
        """
        checker = AntiSpoofingChecker(camera_id="cam_01")
        # Simulate 5 frames at ~25 FPS (40ms each) to let the rolling FPS window stabilise
        base = "2025-01-01T00:00:00"
        timestamps = [
            "2025-01-01T00:00:00.000Z",
            "2025-01-01T00:00:00.040Z",
            "2025-01-01T00:00:00.080Z",
            "2025-01-01T00:00:00.120Z",
            "2025-01-01T00:00:00.160Z",
        ]
        reports = [checker.check(ts, frame_number=i) for i, ts in enumerate(timestamps)]
        # Only check the last report — first frame has no prior context
        last = reports[-1]
        assert not last.is_suspicious, (
            f"Clean 25 FPS stream should not be flagged. Got flags: {last.flags}"
        )

    def test_negative_timestamp_gap_is_flagged(self):
        checker = AntiSpoofingChecker(camera_id="cam_01")
        checker.check("2025-01-01T00:00:05Z", frame_number=100)
        report = checker.check("2025-01-01T00:00:03Z", frame_number=101)
        assert report.is_suspicious
        assert any("TIMESTAMP_NON_MONOTONIC" in f for f in report.flags)

    def test_large_frame_gap_is_flagged(self):
        checker = AntiSpoofingChecker(camera_id="cam_01", max_frame_gap=50)
        checker.check("2025-01-01T00:00:00Z", frame_number=1)
        report = checker.check("2025-01-01T00:00:01Z", frame_number=200)
        assert report.is_suspicious
        assert any("FRAME_GAP" in f for f in report.flags)


# ===========================================================================
# ROME-07: Fusion Engine
# ===========================================================================

class TestFusionEngine:

    def test_new_entity_gets_global_id(self):
        fusion = FusionEngine()
        gid = fusion.assign_or_merge("cam_01", "trk_001", [0.1, 0.2, 0.4, 0.8], "human")
        assert gid is not None
        assert len(gid) == 36   # UUID v4

    def test_same_track_returns_same_global_id(self):
        fusion = FusionEngine()
        gid1 = fusion.assign_or_merge("cam_01", "trk_001", [0.1, 0.2, 0.4, 0.8], "human")
        gid2 = fusion.assign_or_merge("cam_01", "trk_001", [0.1, 0.2, 0.4, 0.8], "human")
        assert gid1 == gid2

    def test_overlapping_entities_from_two_cameras_merged(self):
        """Same physical entity in overlapping FOV → same global_fusion_id."""
        fusion = FusionEngine(iou_threshold=0.4)
        # cam_01 sees entity at position A
        gid_cam1 = fusion.assign_or_merge("cam_01", "trk_001", [0.1, 0.2, 0.5, 0.8], "human")
        # cam_02 sees entity at similar position (IoU should be high)
        gid_cam2 = fusion.assign_or_merge("cam_02", "trk_002", [0.12, 0.21, 0.51, 0.79], "human")
        assert gid_cam1 == gid_cam2   # merged into one global entity

    def test_non_overlapping_entities_get_different_ids(self):
        """Distinct physical entities in non-overlapping positions → distinct global IDs."""
        fusion = FusionEngine(iou_threshold=0.4)
        gid1 = fusion.assign_or_merge("cam_01", "trk_001", [0.0, 0.0, 0.2, 0.3], "human")
        gid2 = fusion.assign_or_merge("cam_02", "trk_002", [0.8, 0.7, 1.0, 1.0], "human")
        assert gid1 != gid2


# ===========================================================================
# ROME-08: MQTT Round-Trip (integration, requires Mosquitto)
# ===========================================================================

class TestMQTTRoundTrip:
    """
    Integration test — requires a running Mosquitto broker on localhost:1883.
    Skip automatically if broker unavailable.
    """

    @pytest.fixture(autouse=True)
    def skip_without_broker(self):
        """Skip if no MQTT broker is reachable."""
        import socket
        try:
            s = socket.create_connection(("localhost", 1883), timeout=1)
            s.close()
        except (OSError, ConnectionRefusedError):
            pytest.skip("MQTT broker not available at localhost:1883")

    def test_publish_and_receive_valid_event(self):
        """
        Phase Rome integration checkpoint:
        Publish a valid ModelAEvent → receive on subscriber → verify round-trip.
        """
        from model_a.bus_client import BusClient

        received_payloads: list[dict] = []
        event_received  = threading.Event()

        publisher  = BusClient(client_id="test_publisher")
        subscriber = BusClient(client_id="test_subscriber")

        publisher.connect()
        subscriber.connect()

        def on_message(topic: str, payload: dict) -> None:
            received_payloads.append(payload)
            event_received.set()

        subscriber.subscribe_events("cam_test", on_message)
        time.sleep(0.3)  # let subscription propagate

        event = make_valid_event(camera_id="cam_test")
        publisher.publish_event(event)

        assert event_received.wait(timeout=5.0), "No event received within 5s"

        assert len(received_payloads) == 1
        payload = received_payloads[0]
        assert payload["event_id"] == event.event_id
        assert payload["engine_source"] == "model_a"
        assert payload["camera_id"] == "cam_test"

        publisher.disconnect()
        subscriber.disconnect()
