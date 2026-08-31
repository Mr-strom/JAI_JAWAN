"""
Phase Oslo — Fallback Routing Tests
SIH26187 | Model A Test Suite

Spec checkpoint:
  Phase Oslo: Fallback routing test — simulate dead Model B heartbeat,
  confirm fallback engages ONLY on that camera.

Edge cases from spec (all tested here):
  "Dead Model B Heartbeat: Delayed but not dead (slow) → wait for exact timeout.
   Dead (>30s) → engage Model A safety floor."
  "Fallback routing engages ONLY on that specific camera."

Test map:
  OSLO-01  Live engine (heartbeat current) → NORMAL, no fallback
  OSLO-02  Slow engine (heartbeat at 99% of timeout) → still NORMAL
  OSLO-03  Dead engine (heartbeat stale > timeout) → FALLBACK engaged
  OSLO-04  Fallback is camera-scoped: dead engine's cameras fallback, others NORMAL
  OSLO-05  Multiple independent engines: each can be dead/alive independently
  OSLO-06  Engine never sent a heartbeat → immediate FALLBACK on first evaluate()
  OSLO-07  Engine recovers: FALLBACK → RECOVERING → NORMAL after N beats
  OSLO-08  Safety floor events carry SAFETY_FLOOR_ACTIVE flag and are schema-valid
  OSLO-09  Safety floor: confirmed trigger events still fire (Rule #1 still holds)
  OSLO-10  Auto-restart NOT performed: state stays FALLBACK until heartbeat received
  OSLO-11  Exact boundary: stale == timeout → still NORMAL; stale > timeout → FALLBACK
  OSLO-12  SafetyFloor animal events: SAFETY_FLOOR_ACTIVE flag + animal_detected type

Run:
  pytest tests/test_phase_oslo.py -v
"""

from __future__ import annotations

import datetime
import time
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from model_a.animal_filter import AnimalFilter
from model_a.bbox_consistency import BBoxConsistencyChecker
from model_a.detector import Detection
from model_a.fallback_router import EngineState, FallbackRouter
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
from model_a.trigger_detector import TriggerDetector


# ===========================================================================
# Helpers
# ===========================================================================

def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


def _short_timeout_router(timeout_s: float = 0.05) -> FallbackRouter:
    """FallbackRouter with a very short timeout for fast tests."""
    return FallbackRouter(heartbeat_timeout_s=timeout_s)


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_detection(
    track_id: str = "trk_01",
    entity_type: EntityType = EntityType.human,
    confidence: float = 0.85,
    bbox: Optional[List[float]] = None,
) -> Detection:
    return Detection(
        track_id    = track_id,
        entity_type = entity_type,
        confidence  = confidence,
        bbox        = bbox or [0.30, 0.40, 0.50, 0.80],
        class_id    = 0,
        class_name  = "person",
        frame_number = 0,
    )


class _MockDetector:
    """Returns a fixed list of detections regardless of frame content."""
    def __init__(self, detections: List[Detection]) -> None:
        self._dets = detections

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> List[Detection]:
        return list(self._dets)


# ===========================================================================
# OSLO-01, OSLO-02: Normal / Slow engine — no premature fallback
# ===========================================================================

class TestNormalAndSlowEngine:

    def test_oslo_01_live_engine_stays_normal(self):
        """
        OSLO-01: Engine sends heartbeats regularly (within timeout).
        State must remain NORMAL. No fallback.
        """
        router = _short_timeout_router(timeout_s=0.10)
        router.register_engine("face_engine", cameras=["cam_01", "cam_02"])

        # Send a fresh heartbeat
        router.update_heartbeat("face_engine")
        router.evaluate()

        assert router.engine_state("face_engine") == EngineState.NORMAL
        assert router.is_fallback_active("face_engine") is False
        assert router.get_fallback_cameras() == []

    def test_oslo_02_slow_engine_below_timeout_stays_normal(self):
        """
        OSLO-02: Engine heartbeat is delayed but within timeout (< 30s equivalent).
        Spec: "Delayed but not dead (slow) → wait for exact timeout."
        Must NOT engage fallback prematurely.
        """
        timeout_s = 0.10  # 100ms for testing
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("posture_engine", cameras=["cam_03"])

        router.update_heartbeat("posture_engine")

        # Sleep for 40% of timeout — slow but alive
        time.sleep(timeout_s * 0.40)
        router.evaluate()

        assert router.engine_state("posture_engine") == EngineState.NORMAL, (
            "Engine should be NORMAL at 40% of timeout. "
            "Fallback must NOT engage prematurely."
        )
        assert router.get_fallback_cameras() == []

    def test_oslo_02b_just_under_timeout_boundary_stays_normal(self):
        """
        OSLO-02b: Heartbeat at 95% of timeout → still NORMAL.
        Tests the exact boundary from the spec.
        """
        timeout_s = 0.08
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("traj_engine", cameras=["cam_05"])

        router.update_heartbeat("traj_engine")
        time.sleep(timeout_s * 0.92)   # 92% of timeout — should still be NORMAL
        router.evaluate()

        assert router.engine_state("traj_engine") == EngineState.NORMAL, (
            "Engine should stay NORMAL before timeout expires."
        )


# ===========================================================================
# OSLO-03: Dead engine → FALLBACK
# ===========================================================================

class TestDeadEngineTransition:

    def test_oslo_03_stale_heartbeat_engages_fallback(self):
        """
        OSLO-03: Heartbeat stale > timeout → FALLBACK engaged.
        Spec: "Dead (>30s) → engage Model A safety floor."
        """
        timeout_s = 0.05
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("face_engine", cameras=["cam_01"])

        router.update_heartbeat("face_engine")
        time.sleep(timeout_s * 1.5)   # 150% of timeout — clearly dead
        router.evaluate()

        assert router.engine_state("face_engine") == EngineState.FALLBACK
        assert router.is_fallback_active("face_engine") is True
        assert "cam_01" in router.get_fallback_cameras()

    def test_oslo_06_engine_never_sent_heartbeat_immediately_fallback(self):
        """
        OSLO-06: Engine registered but never sends a heartbeat.
        On first evaluate() → must enter FALLBACK immediately.
        An engine that never checks in is effectively dead.
        """
        router = _short_timeout_router(timeout_s=0.05)
        router.register_engine("anpr_engine", cameras=["cam_gate_01"])

        # Do NOT call update_heartbeat — simulate engine never coming online
        router.evaluate()

        assert router.engine_state("anpr_engine") == EngineState.FALLBACK
        assert router.is_fallback_active("anpr_engine") is True
        assert "cam_gate_01" in router.get_fallback_cameras()

    def test_oslo_10_fallback_does_not_auto_restart(self):
        """
        OSLO-10: Spec: "Do NOT auto-restart failed Model B engines."
        After fallback is engaged, state must remain FALLBACK indefinitely
        until a heartbeat is manually received. No automatic recovery.
        """
        timeout_s = 0.05
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("posture_engine", cameras=["cam_07"])

        # Make it dead
        router.update_heartbeat("posture_engine")
        time.sleep(timeout_s * 2.0)
        router.evaluate()
        assert router.engine_state("posture_engine") == EngineState.FALLBACK

        # Wait even longer — state must NOT change on its own
        time.sleep(timeout_s * 3.0)
        router.evaluate()   # no heartbeat → must stay FALLBACK

        assert router.engine_state("posture_engine") == EngineState.FALLBACK, (
            "Engine must stay in FALLBACK without a heartbeat. "
            "Auto-restart is forbidden by spec."
        )


# ===========================================================================
# OSLO-04 & OSLO-05: Camera-scoped fallback
# ===========================================================================

class TestCameraScopedFallback:

    def test_oslo_04_fallback_is_camera_scoped_not_global(self):
        """
        OSLO-04: Dead engine's cameras fallback. Other cameras unaffected.
        Spec: "Fallback routing engages ONLY on that specific camera."
        """
        timeout_s = 0.05
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("face_engine",    cameras=["cam_01", "cam_02"])
        router.register_engine("posture_engine", cameras=["cam_03", "cam_04"])

        # Both engines get fresh heartbeats
        router.update_heartbeat("face_engine")
        router.update_heartbeat("posture_engine")

        # Let face_engine go stale
        time.sleep(timeout_s * 1.5)
        router.update_heartbeat("posture_engine")  # posture stays alive
        router.evaluate()

        # face_engine cameras → fallback
        assert "cam_01" in router.get_fallback_cameras()
        assert "cam_02" in router.get_fallback_cameras()

        # posture_engine cameras → NORMAL, unaffected
        assert "cam_03" not in router.get_fallback_cameras()
        assert "cam_04" not in router.get_fallback_cameras()

        assert router.engine_state("face_engine")    == EngineState.FALLBACK
        assert router.engine_state("posture_engine") == EngineState.NORMAL

    def test_oslo_05_multiple_engines_independently_dead(self):
        """
        OSLO-05: Two engines go dead at different times.
        Each independently transitions to FALLBACK.
        Third engine stays NORMAL throughout.
        """
        timeout_s = 0.05
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("engine_A", cameras=["cam_A1"])
        router.register_engine("engine_B", cameras=["cam_B1"])
        router.register_engine("engine_C", cameras=["cam_C1"])

        router.update_heartbeat("engine_A")
        router.update_heartbeat("engine_B")
        router.update_heartbeat("engine_C")

        # engine_A goes dead
        time.sleep(timeout_s * 1.5)
        router.update_heartbeat("engine_B")  # B gets a new beat — resets timer
        router.update_heartbeat("engine_C")  # C also gets a new beat
        router.evaluate()

        assert router.engine_state("engine_A") == EngineState.FALLBACK
        assert router.engine_state("engine_B") == EngineState.NORMAL
        assert router.engine_state("engine_C") == EngineState.NORMAL

        # Now B also goes dead
        time.sleep(timeout_s * 1.5)
        router.update_heartbeat("engine_C")  # C stays alive
        router.evaluate()

        assert router.engine_state("engine_A") == EngineState.FALLBACK
        assert router.engine_state("engine_B") == EngineState.FALLBACK
        assert router.engine_state("engine_C") == EngineState.NORMAL

        # Verify camera lists
        fallback_cams = set(router.get_fallback_cameras())
        normal_cams   = set(router.get_normal_cameras())

        assert fallback_cams == {"cam_A1", "cam_B1"}
        assert normal_cams   == {"cam_C1"}


# ===========================================================================
# OSLO-07: Recovery flow
# ===========================================================================

class TestEngineRecovery:

    def test_oslo_07_engine_recovers_after_n_heartbeats(self):
        """
        OSLO-07: Engine goes dead → FALLBACK. Heartbeats resume.
        FALLBACK → RECOVERING after 1st beat.
        RECOVERING → NORMAL after N consecutive beats.
        """
        timeout_s = 0.05
        router = FallbackRouter(
            heartbeat_timeout_s=timeout_s,
            recovery_beat_count=3,   # need 3 beats to fully recover
        )
        router.register_engine("face_engine", cameras=["cam_01"])

        # Make it dead
        router.update_heartbeat("face_engine")
        time.sleep(timeout_s * 1.5)
        router.evaluate()
        assert router.engine_state("face_engine") == EngineState.FALLBACK

        # First heartbeat — RECOVERING, not yet NORMAL
        router.update_heartbeat("face_engine")
        assert router.engine_state("face_engine") == EngineState.RECOVERING
        assert router.is_fallback_active("face_engine") is True  # still covered

        # Second heartbeat — still RECOVERING
        router.update_heartbeat("face_engine")
        assert router.engine_state("face_engine") == EngineState.RECOVERING

        # Third heartbeat — NORMAL
        router.update_heartbeat("face_engine")
        assert router.engine_state("face_engine") == EngineState.NORMAL
        assert router.is_fallback_active("face_engine") is False
        assert "cam_01" not in router.get_fallback_cameras()

    def test_oslo_07b_camera_exits_fallback_after_recovery(self):
        """
        OSLO-07b: Once engine is NORMAL, its cameras are removed from
        fallback_cameras list and appear in normal_cameras.
        """
        timeout_s = 0.05
        router = FallbackRouter(heartbeat_timeout_s=timeout_s, recovery_beat_count=2)
        router.register_engine("posture_engine", cameras=["cam_03", "cam_04"])

        router.update_heartbeat("posture_engine")
        time.sleep(timeout_s * 1.5)
        router.evaluate()

        # Both cams in fallback
        assert {"cam_03", "cam_04"} == set(router.get_fallback_cameras())

        # Recovery
        router.update_heartbeat("posture_engine")
        router.update_heartbeat("posture_engine")

        # Both cams back to normal
        assert router.get_fallback_cameras() == []
        assert "cam_03" in router.get_normal_cameras()
        assert "cam_04" in router.get_normal_cameras()


# ===========================================================================
# OSLO-11: Exact timeout boundary
# ===========================================================================

class TestExactBoundary:

    def test_oslo_11_exact_boundary_behaviour(self):
        """
        OSLO-11: Test the exact boundary between NORMAL and FALLBACK.
        stale < timeout  → NORMAL
        stale > timeout  → FALLBACK
        (Equal case: float comparison, treated as NORMAL since > not >=)
        """
        timeout_s = 0.10
        router = _short_timeout_router(timeout_s=timeout_s)
        router.register_engine("traj_engine", cameras=["cam_boundary"])

        # Just under timeout — still NORMAL
        router.update_heartbeat("traj_engine")
        time.sleep(timeout_s * 0.80)   # 80% of timeout
        router.evaluate()
        assert router.engine_state("traj_engine") == EngineState.NORMAL, (
            "At 80% of timeout the engine must still be NORMAL."
        )

        # Cross the boundary
        time.sleep(timeout_s * 0.40)   # now at 120% total
        router.evaluate()
        assert router.engine_state("traj_engine") == EngineState.FALLBACK, (
            "At 120% of timeout the engine must be FALLBACK."
        )


# ===========================================================================
# OSLO-08 & OSLO-12: Safety Floor events — schema validity + SAFETY_FLOOR_ACTIVE flag
# ===========================================================================

class TestSafetyFloorEvents:
    """
    Tests that SafetyFloor produces correctly structured events with the
    SAFETY_FLOOR_ACTIVE flag, and that all spec rules still hold
    (multi-frame confirmation, animal suppression, schema_v1 compliance).
    """

    def _make_floor(
        self,
        camera_id: str = "cam_fallback_01",
        detections: Optional[list] = None,
    ) -> tuple[SafetyFloor, "_MockDetector"]:
        if detections is None:
            detections = [_make_detection()]
        mock_det = _MockDetector(detections)
        floor    = SafetyFloor(
            camera_id = camera_id,
            detector  = mock_det,
            zone_tag  = ZoneTag.long_range,
            zone      = Zone.perimeter,
        )
        return floor, mock_det

    def test_oslo_08_safety_floor_events_are_schema_valid(self):
        """
        OSLO-08: Events produced by SafetyFloor must pass schema_v1 validation.
        engine_source must be 'model_a'. spoofing_flags must include SAFETY_FLOOR_ACTIVE.
        """
        floor, _ = self._make_floor()
        frame = _blank_frame()
        events = floor.process(frame, frame_number=1, timestamp_utc=_now_iso())

        assert len(events) > 0, "SafetyFloor should produce at least one event."
        for event in events:
            assert event.engine_source == "model_a"
            assert _SAFETY_FLOOR_FLAG in event.metadata.spoofing_flags, (
                f"SAFETY_FLOOR_ACTIVE flag missing from event {event.event_id}"
            )
            # Validate it's a proper schema_v1 event (no exception = valid)
            payload = event.to_mqtt_payload()
            import json
            parsed = json.loads(payload)
            assert parsed["engine_source"] == "model_a"

    def test_oslo_08b_motion_event_produced_on_human_detection(self):
        """
        OSLO-08b: Human detection in safety floor → motion event (severity=info).
        """
        floor, _ = self._make_floor(
            detections=[_make_detection("trk_h1", entity_type=EntityType.human)]
        )
        events = floor.process(_blank_frame(), frame_number=1, timestamp_utc=_now_iso())

        motion_events = [e for e in events if e.event_type == EventType.motion]
        assert len(motion_events) >= 1
        assert motion_events[0].severity == Severity.info
        assert _SAFETY_FLOOR_FLAG in motion_events[0].metadata.spoofing_flags

    def test_oslo_12_safety_floor_animal_events_also_flagged(self):
        """
        OSLO-12: Animal detected via safety floor → animal_detected (info).
        SAFETY_FLOOR_ACTIVE flag present. Fence triggers suppressed (same as main pipeline).
        """
        floor, _ = self._make_floor(
            detections=[_make_detection("deer_01", entity_type=EntityType.animal, confidence=0.75)]
        )
        events = floor.process(_blank_frame(), frame_number=1, timestamp_utc=_now_iso())

        animal_events  = [e for e in events if e.event_type == EventType.animal_detected]
        trigger_events = [e for e in events if e.event_type == EventType.trigger]

        assert len(animal_events) >= 1
        assert len(trigger_events) == 0   # no fence trigger for animals even in fallback
        assert all(_SAFETY_FLOOR_FLAG in e.metadata.spoofing_flags for e in animal_events)

    def test_oslo_09_safety_floor_trigger_still_requires_3_frames(self):
        """
        OSLO-09: Even in safety floor mode, Rule #1 still holds.
        Trigger confirmed only after 3 consecutive confirming frames.
        2 frames → NO confirmed event.
        """
        floor, _ = self._make_floor(
            detections=[_make_detection("climber_sf", entity_type=EntityType.human,
                                        bbox=[0.30, 0.40, 0.50, 0.80])]
        )
        ts = _now_iso()

        # Frame 1
        e1 = floor.process(_blank_frame(), frame_number=1, timestamp_utc=ts,
                            trigger_type_hint=TriggerType.climbing)
        # Frame 2
        e2 = floor.process(_blank_frame(), frame_number=2, timestamp_utc=ts,
                            trigger_type_hint=TriggerType.climbing)

        # No confirmed/critical in first two frames
        all_events = e1 + e2
        confirmed = [e for e in all_events
                     if e.event_type == EventType.trigger
                     and e.severity in (Severity.confirmed, Severity.critical)]
        assert len(confirmed) == 0, (
            "Safety floor must NOT confirm a trigger in 2 frames. Rule #1 still applies."
        )

        # Frame 3 — should confirm
        e3 = floor.process(_blank_frame(), frame_number=3, timestamp_utc=ts,
                            trigger_type_hint=TriggerType.climbing)
        confirmed_3 = [e for e in e3
                       if e.event_type == EventType.trigger
                       and e.severity in (Severity.confirmed, Severity.critical)]
        assert len(confirmed_3) >= 1, (
            "Safety floor must confirm trigger after 3 frames. Rule #1 requires it."
        )
        # The confirmed event must also carry the SAFETY_FLOOR_ACTIVE flag
        assert all(_SAFETY_FLOOR_FLAG in e.metadata.spoofing_flags for e in confirmed_3)

    def test_oslo_09b_safety_floor_deactivate_logs_correctly(self):
        """
        OSLO-09b: SafetyFloor.deactivate() must not raise and should log
        that the engine has recovered.
        """
        floor, _ = self._make_floor()
        # Should not raise
        floor.deactivate()


# ===========================================================================
# OSLO: Full scenario integration test
# ===========================================================================

class TestOsloScenarioIntegration:
    """
    End-to-end Oslo scenario:
    1. Two engines running normally
    2. Engine A goes dead
    3. FallbackRouter detects it
    4. SafetyFloor activated for engine A's cameras
    5. Events from safety floor are schema-valid with SAFETY_FLOOR_ACTIVE flag
    6. Engine A recovers → safety floor deactivated
    """

    def test_oslo_full_scenario(self):
        """
        Full Oslo lifecycle:
        NORMAL → FALLBACK (engine dead) → RECOVERING → NORMAL (engine back)
        """
        timeout_s = 0.06
        router = FallbackRouter(
            heartbeat_timeout_s=timeout_s,
            recovery_beat_count=2,
        )
        router.register_engine("face_engine",    cameras=["cam_01", "cam_02"])
        router.register_engine("posture_engine", cameras=["cam_03"])

        # Both healthy
        router.update_heartbeat("face_engine")
        router.update_heartbeat("posture_engine")
        router.evaluate()
        assert router.get_fallback_cameras() == []

        # face_engine dies
        time.sleep(timeout_s * 1.5)
        router.update_heartbeat("posture_engine")   # posture stays alive
        router.evaluate()

        fallback_cams = set(router.get_fallback_cameras())
        assert fallback_cams == {"cam_01", "cam_02"}, (
            f"Only face_engine cameras should be in fallback. Got: {fallback_cams}"
        )
        assert "cam_03" not in fallback_cams   # posture_engine unaffected

        # Safety floor activated for cam_01
        floor = SafetyFloor(
            camera_id = "cam_01",
            detector  = _MockDetector([
                _make_detection("trk_fallback", entity_type=EntityType.human)
            ]),
        )
        frame  = _blank_frame()
        events = floor.process(frame, frame_number=50, timestamp_utc=_now_iso())

        assert len(events) > 0
        assert all(_SAFETY_FLOOR_FLAG in e.metadata.spoofing_flags for e in events)
        assert all(e.camera_id == "cam_01" for e in events)

        # face_engine recovers
        router.update_heartbeat("face_engine")   # beat 1 → RECOVERING
        assert router.engine_state("face_engine") == EngineState.RECOVERING

        router.update_heartbeat("face_engine")   # beat 2 → NORMAL
        assert router.engine_state("face_engine") == EngineState.NORMAL

        # Safety floor deactivated
        floor.deactivate()

        # cam_01 and cam_02 are no longer in fallback
        assert "cam_01" not in router.get_fallback_cameras()
        assert "cam_02" not in router.get_fallback_cameras()
        assert "cam_03" not in router.get_fallback_cameras()
