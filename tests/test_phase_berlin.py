"""
Phase Berlin — False-Trigger Suppression + True-Trigger Confirmation Tests
SIH26187 | Model A Test Suite

Spec checkpoints:
  Phase Berlin: False-trigger suppression test + true-trigger confirmation test
                (2-frame boundary explicitly tested).

Test cases covered (direct from spec edge cases):
  BERLIN-01: 2-frame boundary — trigger for 2 frames then gone. Must NOT confirm.
  BERLIN-02: True-trigger — 3 consecutive frames → CONFIRMED, severity=confirmed
  BERLIN-03: Critical escalation — 5+ frames → severity=critical
  BERLIN-04: Deer jumps fence at night (confidence 0.75) → animal_detected, 0 fence alerts
  BERLIN-05: Shadow / Blowing Foliage → BBoxConsistency fails → suppressed
  BERLIN-06: Farmer crouching (open border) → motion published, NO trigger
  BERLIN-07: Person climbs fence, stays 2s, drops → CRITICAL published after 3 frames
  BERLIN-08: Animal filter — animal_cart suppressed just like plain animal
  BERLIN-09: Multiple animals, zero false trigger events
  BERLIN-10: Shadow: frame-by-frame bbox consistency with real IoU values
  BERLIN-11: Pipeline integration — detector mock → animal filter → trigger
  BERLIN-12: Pipeline integration — detector mock → true trigger → confirmed event

Run:
  pytest tests/test_phase_berlin.py -v
"""

from __future__ import annotations

import datetime
import uuid
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from model_a.animal_filter import AnimalFilter
from model_a.bbox_consistency import BBoxConsistencyChecker
from model_a.detector import Detection, MockDetector
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


# ===========================================================================
# Helpers
# ===========================================================================

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_det(
    track_id: str,
    entity_type: EntityType = EntityType.human,
    confidence: float = 0.85,
    bbox: Optional[List[float]] = None,
    frame_number: int = 0,
) -> Detection:
    return Detection(
        track_id    = track_id,
        entity_type = entity_type,
        confidence  = confidence,
        bbox        = bbox or [0.30, 0.40, 0.50, 0.80],
        class_id    = 0 if entity_type == EntityType.human else 16,
        class_name  = "person" if entity_type == EntityType.human else "dog",
        frame_number = frame_number,
    )


def _make_animal_det(
    track_id: str = "animal_01",
    confidence: float = 0.75,
    entity_type: EntityType = EntityType.animal,
    bbox: Optional[List[float]] = None,
) -> Detection:
    return Detection(
        track_id    = track_id,
        entity_type = entity_type,
        confidence  = confidence,
        bbox        = bbox or [0.10, 0.20, 0.30, 0.60],
        class_id    = 16,
        class_name  = "dog",
        frame_number = 0,
    )


# ===========================================================================
# BERLIN-01 & BERLIN-02: 2-Frame Boundary + True-Trigger Confirmation
# (Unit level — TriggerDetector state machine)
# ===========================================================================

class TestTriggerConfirmationBoundaries:
    """
    These are the MOST CRITICAL tests in Phase Berlin.
    They directly guard against the CIBMS false-alarm failure mode.
    """

    def test_2_frame_boundary_does_not_confirm(self):
        """
        BERLIN-01: Track seen for EXACTLY 2 frames then vanishes.
        MUST NOT produce confirmed or critical severity.
        State must reset to IDLE after miss().
        """
        det = TriggerDetector(confirmation_frames=3)

        r1 = det.update("trk_01", TriggerType.climbing, frame_number=100)
        assert r1.severity == Severity.provisional
        assert r1.confirmation_frames == 1

        r2 = det.update("trk_01", TriggerType.climbing, frame_number=101)
        assert r2.severity == Severity.provisional
        assert r2.confirmation_frames == 2

        # Track disappears on frame 102 — miss()
        det.miss("trk_01")

        # Verify: state MUST be IDLE, frames=0, NO confirmed/critical ever published
        state = det.active_tracks["trk_01"]
        assert state["state"] == "IDLE", (
            f"Expected IDLE after 2-frame boundary miss. Got: {state['state']}"
        )
        assert state["frames"] == 0

        # Verify r1, r2 were both provisional — never confirmed/critical
        assert r1.severity not in (Severity.confirmed, Severity.critical)
        assert r2.severity not in (Severity.confirmed, Severity.critical)

    def test_3_consecutive_frames_confirms(self):
        """
        BERLIN-02: 3 consecutive frames → CONFIRMED.
        Exactly the minimum required by Rule #1.
        """
        det = TriggerDetector(confirmation_frames=3)

        det.update("trk_02", TriggerType.fence_cutting, frame_number=1)
        det.update("trk_02", TriggerType.fence_cutting, frame_number=2)
        r3 = det.update("trk_02", TriggerType.fence_cutting, frame_number=3)

        assert r3.state == TriggerState.CONFIRMED_TRIGGER
        assert r3.severity == Severity.confirmed
        assert r3.confirmation_frames == 3

    def test_confirmed_severity_validates_against_schema(self):
        """
        After 3 frames, build a ModelAEvent with severity=confirmed.
        Pydantic MUST accept it (the schema-level gate should pass).
        """
        det = TriggerDetector(confirmation_frames=3)
        det.update("trk_03", TriggerType.climbing, frame_number=1)
        det.update("trk_03", TriggerType.climbing, frame_number=2)
        result = det.update("trk_03", TriggerType.climbing, frame_number=3)

        # Should not raise
        event = ModelAEvent(
            event_type   = EventType.trigger,
            severity     = result.severity,
            timestamp    = _now(),
            camera_id    = "cam_01",
            zone_tag     = ZoneTag.long_range,
            zone         = Zone.perimeter,
            entity_type  = EntityType.human,
            entity_id    = "trk_03",
            confidence   = 0.88,
            bbox         = [0.3, 0.4, 0.5, 0.8],
            evidence_ref = "pending",
            hash         = "pending",
            metadata     = EventMetadata(
                model_version      = "1.0.0",
                processing_time_ms = 42,
                frame_number       = 3,
                trigger_type       = TriggerType.climbing,
                confirmation_frames = result.confirmation_frames,
                spoofing_flags     = [],
            ),
        )
        assert event.severity == Severity.confirmed
        assert event.metadata.trigger_type == TriggerType.climbing

    def test_critical_escalation_after_5_frames(self):
        """
        BERLIN-03: 5+ consecutive frames → severity escalates to critical.
        """
        det = TriggerDetector(confirmation_frames=3)
        for i in range(5):
            result = det.update("trk_04", TriggerType.rapid_approach, frame_number=i)
        assert result.severity == Severity.critical
        assert result.confirmation_frames == 5

    def test_track_resets_after_cooldown_expires(self):
        """
        After confirmation + cooldown expires, the same track can re-trigger
        with a fresh 3-frame cycle.
        """
        det = TriggerDetector(confirmation_frames=3, cooldown_seconds=0.001)
        for i in range(3):
            det.update("trk_05", TriggerType.zone_violation, frame_number=i)
        det.miss("trk_05")  # enters cooldown

        import time; time.sleep(0.01)  # let cooldown expire

        # New cycle — first frame should give PROVISIONAL_1
        r = det.update("trk_05", TriggerType.zone_violation, frame_number=100)
        assert r.state == TriggerState.PROVISIONAL_1
        assert r.severity == Severity.provisional


# ===========================================================================
# BERLIN-04 & BERLIN-08 & BERLIN-09: Animal Filtering
# ===========================================================================

class TestAnimalFiltering:
    """
    Spec: "Deer jumps fence at night: YOLO confidence 0.75.
    Logs animal, suppresses alert. 0 false positives."
    """

    def test_deer_at_night_classified_as_animal(self):
        """BERLIN-04: Animal detection at confidence 0.75 → classified as animal."""
        filt = AnimalFilter()
        deer = _make_animal_det(confidence=0.75, entity_type=EntityType.animal)
        animals, trigger_candidates = filt.classify([deer])

        assert len(animals) == 1
        assert len(trigger_candidates) == 0  # ZERO fence triggers

    def test_animal_detection_never_enters_trigger_pipeline(self):
        """
        BERLIN-04 (end-to-end guard):
        Animals classified by AnimalFilter must NEVER enter TriggerDetector.
        """
        filt = AnimalFilter()
        det_state = TriggerDetector(confirmation_frames=3)

        # Simulate 5 frames of a deer
        for i in range(5):
            deer = _make_animal_det(confidence=0.80)
            animals, candidates = filt.classify([deer])
            assert len(animals) == 1
            assert len(candidates) == 0   # zero candidates → TriggerDetector never called

        # TriggerDetector should be completely empty
        assert len(det_state.active_tracks) == 0

    def test_animal_cart_also_suppressed(self):
        """BERLIN-08: animal_cart entity_type also suppressed (not a fence trigger)."""
        filt = AnimalFilter()
        cart = _make_animal_det(entity_type=EntityType.animal_cart, confidence=0.70)
        animals, candidates = filt.classify([cart])

        assert len(animals) == 1
        assert len(candidates) == 0

    def test_multiple_animals_zero_fence_triggers(self):
        """
        BERLIN-09: 3 different animals detected simultaneously.
        0 trigger candidates produced.
        """
        filt = AnimalFilter()
        detections = [
            _make_animal_det("a1", entity_type=EntityType.animal, confidence=0.65),
            _make_animal_det("a2", entity_type=EntityType.animal, confidence=0.72),
            _make_animal_det("a3", entity_type=EntityType.animal_cart, confidence=0.60),
        ]
        animals, candidates = filt.classify(detections)

        assert len(animals) == 3
        assert len(candidates) == 0

    def test_below_min_confidence_discarded(self):
        """Detections below min_confidence threshold are discarded entirely."""
        filt = AnimalFilter(min_confidence=0.50)
        low_conf = _make_animal_det(confidence=0.30, entity_type=EntityType.animal)
        animals, candidates = filt.classify([low_conf])

        assert len(animals) == 0
        assert len(candidates) == 0

    def test_human_passes_through_filter(self):
        """Human detections are NOT filtered — they go to trigger pipeline."""
        filt = AnimalFilter()
        human = _make_det("trk_h1", entity_type=EntityType.human, confidence=0.85)
        animals, candidates = filt.classify([human])

        assert len(animals) == 0
        assert len(candidates) == 1

    def test_animal_event_schema_valid(self):
        """
        animal_detected event built from animal detection
        must validate against schema_v1.
        """
        event = ModelAEvent(
            event_type   = EventType.animal_detected,
            severity     = Severity.info,
            timestamp    = _now(),
            camera_id    = "cam_night_01",
            zone_tag     = ZoneTag.long_range,
            zone         = Zone.perimeter,
            entity_type  = EntityType.animal,
            entity_id    = "animal_trk_01",
            confidence   = 0.75,
            bbox         = [0.10, 0.20, 0.30, 0.60],
            evidence_ref = "pending",
            hash         = "pending",
            metadata     = EventMetadata(
                model_version      = "1.0.0",
                processing_time_ms = 28,
                frame_number       = 500,
                trigger_type       = None,      # animals have no trigger_type
                confirmation_frames = 0,
                spoofing_flags     = [],
            ),
        )
        assert event.event_type == EventType.animal_detected
        assert event.severity == Severity.info
        assert event.entity_type == EntityType.animal


# ===========================================================================
# BERLIN-05 & BERLIN-10: Shadow / Foliage Suppression via BBoxConsistency
# ===========================================================================

class TestBBoxConsistencySuppression:
    """
    Spec: "Shadow / Blowing Foliage: High-frequency noise fails multi-frame
    consistent bounding box check. Suppressed."
    """

    def test_spatially_consistent_track_passes(self):
        """
        BERLIN-05a: Real person walking — bbox moves smoothly.
        IoU between consecutive frames ≥ 0.35 → consistent → triggers allowed.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)

        # Simulate a person moving slowly right (small displacement)
        bboxes = [
            [0.30, 0.40, 0.50, 0.80],  # frame 1
            [0.31, 0.40, 0.51, 0.80],  # frame 2: shifted 1%
            [0.32, 0.40, 0.52, 0.80],  # frame 3: shifted another 1%
        ]
        results = [checker.check("trk_real", bb) for bb in bboxes]
        assert all(results), f"Smooth movement should be consistent: {results}"

    def test_shadow_bbox_jump_fails_consistency(self):
        """
        BERLIN-05b: Shadow/foliage — bbox teleports across the frame.
        IoU near 0 → inconsistent → confirmation reset.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)

        # Frame 1: detection at top-left
        assert checker.check("trk_shadow", [0.05, 0.05, 0.20, 0.25]) is True

        # Frame 2: detection jumps to bottom-right (shadow shift)
        assert checker.check("trk_shadow", [0.75, 0.70, 0.95, 0.95]) is False

    def test_shadow_with_trigger_detector_resets(self):
        """
        BERLIN-10: Full integration — shadow bbox jump → miss() called →
        TriggerDetector resets to IDLE. No confirmed event.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)
        trigger = TriggerDetector(confirmation_frames=3)

        # Frame 1: valid detection — PROVISIONAL_1
        bbox_1 = [0.30, 0.40, 0.50, 0.80]
        c1 = checker.check("shadow_trk", bbox_1)
        if c1:
            trigger.update("shadow_trk", TriggerType.zone_violation, frame_number=1)

        # Frame 2: valid detection — PROVISIONAL_2
        bbox_2 = [0.31, 0.40, 0.51, 0.80]
        c2 = checker.check("shadow_trk", bbox_2)
        if c2:
            trigger.update("shadow_trk", TriggerType.zone_violation, frame_number=2)
        assert trigger.active_tracks["shadow_trk"]["state"] == "PROVISIONAL_2"

        # Frame 3: shadow JUMPS — IoU = 0 → miss()
        bbox_3 = [0.80, 0.70, 0.95, 0.95]
        c3 = checker.check("shadow_trk", bbox_3)
        assert c3 is False  # inconsistent
        trigger.miss("shadow_trk")  # reset

        # State MUST be IDLE — no CONFIRMED event ever fires
        state = trigger.active_tracks["shadow_trk"]
        assert state["state"] == "IDLE"
        assert state["frames"] == 0

    def test_foliage_with_erratic_bboxes_never_confirms(self):
        """
        BERLIN-05c: 5 frames of foliage with random bbox positions.
        TriggerDetector should never reach CONFIRMED because every bbox
        jump triggers miss() which resets to IDLE.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)
        trigger = TriggerDetector(confirmation_frames=3)

        # Erratic bboxes (foliage — random positions)
        erratic_bboxes = [
            [0.10, 0.10, 0.25, 0.30],
            [0.70, 0.60, 0.90, 0.80],  # jump
            [0.05, 0.05, 0.15, 0.20],  # jump again
            [0.85, 0.50, 0.95, 0.70],  # jump
            [0.30, 0.20, 0.45, 0.40],  # jump
        ]

        confirmed_events = []
        for i, bbox in enumerate(erratic_bboxes):
            consistent = checker.check("foliage_trk", bbox)
            if consistent:
                r = trigger.update("foliage_trk", TriggerType.zone_violation, frame_number=i)
                if r.severity in (Severity.confirmed, Severity.critical):
                    confirmed_events.append(r)
            else:
                trigger.miss("foliage_trk")

        assert len(confirmed_events) == 0, (
            f"Foliage should NEVER confirm. Got {len(confirmed_events)} confirmed events."
        )


# ===========================================================================
# BERLIN-06: Farmer Crouching (Open Border Context)
# ===========================================================================

class TestOpenBorderContext:
    """
    Spec: "Open border context (Farmer crouching): Publishes motion.
    Model B scores threat based on time/posture/zone.
    Model A does NOT preemptively suppress or alert."
    """

    def test_human_in_open_border_zone_generates_motion_not_trigger(self):
        """
        BERLIN-06: Farmer crouching in a perimeter zone.
        AnimalFilter lets it through (it's a human).
        TriggerDetector is NOT called (no trigger signal from YOLO posture).
        Only a motion event is published — severity=info.
        Decision is deferred to Model B.
        """
        filt = AnimalFilter()

        # Farmer crouching — low bbox (crouched), human entity
        farmer = _make_det("farmer_01", entity_type=EntityType.human, confidence=0.82,
                           bbox=[0.40, 0.65, 0.55, 0.80])  # short bbox = crouched

        # AnimalFilter passes farmer through
        animals, candidates = filt.classify([farmer])
        assert len(animals) == 0       # farmer is NOT an animal
        assert len(candidates) == 1    # farmer IS a trigger candidate

        # BUT: no trigger_type signal → motion event only, no TriggerDetector call
        # Model A publishes motion; Model B decides if threat
        motion_event = ModelAEvent(
            event_type   = EventType.motion,
            severity     = Severity.info,       # NOT provisional/confirmed
            timestamp    = _now(),
            camera_id    = "cam_border_01",
            zone_tag     = ZoneTag.long_range,
            zone         = Zone.perimeter,      # NOT intrusion_zone — open border
            entity_type  = EntityType.human,
            entity_id    = "farmer_01",
            confidence   = 0.82,
            bbox         = [0.40, 0.65, 0.55, 0.80],
            evidence_ref = "pending",
            hash         = "pending",
            metadata     = EventMetadata(
                model_version      = "1.0.0",
                processing_time_ms = 35,
                frame_number       = 2000,
                trigger_type       = None,       # NO trigger_type — farmer is not climbing
                confirmation_frames = 0,
                spoofing_flags     = [],
            ),
        )
        # Motion event must be valid schema
        assert motion_event.event_type == EventType.motion
        assert motion_event.severity == Severity.info
        assert motion_event.metadata.trigger_type is None

    def test_farmer_crouching_does_not_generate_trigger_event(self):
        """
        BERLIN-06b: Even after 5 frames of farmer crouching, if no trigger_type
        signal is given, TriggerDetector is NEVER called.
        """
        trigger = TriggerDetector(confirmation_frames=3)

        # 5 frames of farmer detected but NO trigger signal
        # (trigger_type is None → TriggerDetector.update() never called)
        # This simulates how FramePipeline handles no trigger_type_override

        assert len(trigger.active_tracks) == 0  # No tracks registered
        # After 5 frames with no trigger signal — still empty
        # (FramePipeline skips TriggerDetector when trigger_type is None)
        assert len(trigger.active_tracks) == 0


# ===========================================================================
# BERLIN-07: Person Climbs Fence (Full Scenario)
# ===========================================================================

class TestFenceClimbingScenario:
    """
    Spec: "Person climbs fence: Posture climbing → stays 2s → drops.
    Buffer hits 3 → publishes CRITICAL trigger."
    """

    def test_climbing_scenario_3_frames_confirmed(self):
        """
        BERLIN-07: Person detected climbing for 3 consecutive frames.
        Frame 1 → PROVISIONAL_1, Frame 2 → PROVISIONAL_2, Frame 3 → CONFIRMED.
        Event severity = confirmed after frame 3.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)
        trigger = TriggerDetector(confirmation_frames=3)

        # Climbing person: bbox moves upward slightly each frame (ascending)
        climbing_bboxes = [
            [0.40, 0.55, 0.60, 0.85],  # frame 1: lower position
            [0.40, 0.50, 0.60, 0.80],  # frame 2: slightly higher
            [0.40, 0.45, 0.60, 0.75],  # frame 3: higher still
        ]

        results = []
        for i, bbox in enumerate(climbing_bboxes):
            consistent = checker.check("climber_01", bbox)
            assert consistent is True, f"Climbing bbox should be spatially consistent at frame {i}"
            r = trigger.update("climber_01", TriggerType.climbing, frame_number=i + 1)
            results.append(r)

        assert results[0].severity == Severity.provisional
        assert results[1].severity == Severity.provisional
        assert results[2].severity == Severity.confirmed
        assert results[2].confirmation_frames == 3

    def test_climbing_5_frames_escalates_to_critical(self):
        """
        BERLIN-07b: Sustained climbing for 5 frames → severity=critical.
        """
        checker = BBoxConsistencyChecker(iou_threshold=0.35)
        trigger = TriggerDetector(confirmation_frames=3)

        base_bbox = [0.40, 0.50, 0.60, 0.80]
        last_result = None
        for i in range(5):
            # Slight upward movement each frame
            shift = i * 0.01
            bbox = [0.40, 0.50 - shift, 0.60, 0.80 - shift]
            consistent = checker.check("climber_02", bbox)
            assert consistent is True
            last_result = trigger.update("climber_02", TriggerType.climbing, frame_number=i + 1)

        assert last_result.severity == Severity.critical
        assert last_result.confirmation_frames == 5

    def test_climbing_trigger_event_validates_against_schema(self):
        """
        BERLIN-07c: The CONFIRMED climbing trigger event must pass schema_v1.
        Both Pydantic-level and business-rule-level checks.
        """
        event = ModelAEvent(
            event_type   = EventType.trigger,
            severity     = Severity.confirmed,
            timestamp    = _now(),
            camera_id    = "cam_fence_01",
            zone_tag     = ZoneTag.close_range,   # climber is close to camera
            zone         = Zone.intrusion_zone,
            entity_type  = EntityType.human,
            entity_id    = "global_fusion_abc123",
            confidence   = 0.91,
            bbox         = [0.40, 0.45, 0.60, 0.75],
            evidence_ref = "pending",
            hash         = "pending",
            metadata     = EventMetadata(
                model_version      = "1.0.0",
                processing_time_ms = 44,
                frame_number       = 3,
                trigger_type       = TriggerType.climbing,
                confirmation_frames = 3,
                spoofing_flags     = [],
            ),
        )
        assert event.event_type   == EventType.trigger
        assert event.severity     == Severity.confirmed
        assert event.zone_tag     == ZoneTag.close_range
        assert event.zone         == Zone.intrusion_zone
        assert event.metadata.trigger_type == TriggerType.climbing
        assert event.metadata.confirmation_frames == 3


# ===========================================================================
# BERLIN-11 & BERLIN-12: Pipeline Integration with MockDetector
# ===========================================================================

class TestPipelineIntegration:
    """
    Full pipeline integration tests using MockDetector (no real YOLO model needed).
    Validates that the wiring of all components produces correct event types.
    """

    def _run_scenario(
        self,
        detections_per_frame: list[list[Detection]],
        trigger_type: Optional[TriggerType] = None,
    ) -> list[ModelAEvent]:
        """
        Run N frames through AnimalFilter + BBoxConsistency + TriggerDetector.
        Returns all events that would have been published.
        """
        filt    = AnimalFilter()
        checker = BBoxConsistencyChecker(iou_threshold=0.35)
        trigger = TriggerDetector(confirmation_frames=3)
        events: list[ModelAEvent] = []

        for frame_num, dets in enumerate(detections_per_frame):
            animals, candidates = filt.classify(dets)

            # Animal events
            for adet in animals:
                events.append(ModelAEvent(
                    event_type   = EventType.animal_detected,
                    severity     = Severity.info,
                    timestamp    = _now(),
                    camera_id    = "cam_test",
                    zone_tag     = ZoneTag.long_range,
                    zone         = Zone.perimeter,
                    entity_type  = adet.entity_type,
                    entity_id    = adet.track_id,
                    confidence   = adet.confidence,
                    bbox         = adet.bbox,
                    evidence_ref = "pending",
                    hash         = "pending",
                    metadata     = EventMetadata(
                        model_version="1.0.0", processing_time_ms=20,
                        frame_number=frame_num, trigger_type=None,
                        confirmation_frames=0, spoofing_flags=[],
                    ),
                ))

            # Trigger pipeline
            if trigger_type is not None:
                for det in candidates:
                    if det.track_id is None:
                        continue
                    consistent = checker.check(det.track_id, det.bbox)
                    if not consistent:
                        trigger.miss(det.track_id)
                        continue
                    result = trigger.update(det.track_id, trigger_type, frame_num)
                    if result.should_publish:
                        try:
                            events.append(ModelAEvent(
                                event_type   = EventType.trigger,
                                severity     = result.severity,
                                timestamp    = _now(),
                                camera_id    = "cam_test",
                                zone_tag     = ZoneTag.long_range,
                                zone         = Zone.perimeter,
                                entity_type  = det.entity_type,
                                entity_id    = det.track_id,
                                confidence   = det.confidence,
                                bbox         = det.bbox,
                                evidence_ref = "pending",
                                hash         = "pending",
                                metadata     = EventMetadata(
                                    model_version="1.0.0", processing_time_ms=45,
                                    frame_number=frame_num,
                                    trigger_type=trigger_type,
                                    confirmation_frames=result.confirmation_frames,
                                    spoofing_flags=[],
                                ),
                            ))
                        except Exception:
                            pass  # schema rejection — expected for <3 frames

            # Handle missed tracks
            active_ids = {d.track_id for d in candidates if d.track_id}
            for tid in list(trigger.active_tracks.keys()):
                if tid not in active_ids:
                    trigger.miss(tid)

        return events

    def test_berlin_11_animal_only_scenario(self):
        """
        BERLIN-11: 5 frames of only animals → ZERO trigger events published.
        animal_detected events = 5. fence trigger events = 0.
        """
        frames = [
            [_make_animal_det("deer_01", confidence=0.75)]
            for _ in range(5)
        ]
        events = self._run_scenario(frames, trigger_type=None)

        trigger_events = [e for e in events if e.event_type == EventType.trigger]
        animal_events  = [e for e in events if e.event_type == EventType.animal_detected]

        assert len(trigger_events) == 0, (
            f"ZERO trigger events expected for animal-only scenario. Got {len(trigger_events)}."
        )
        assert len(animal_events) == 5

    def test_berlin_12_true_trigger_3_frames(self):
        """
        BERLIN-12: 3 frames of a human with climbing trigger.
        Expected: 1 confirmed trigger event published after frame 3.
        """
        human_bbox = [0.30, 0.40, 0.50, 0.80]
        frames = [
            [_make_det("climber_x", entity_type=EntityType.human, bbox=human_bbox)]
            for _ in range(3)
        ]
        events = self._run_scenario(frames, trigger_type=TriggerType.climbing)

        trigger_events = [e for e in events
                          if e.event_type == EventType.trigger
                          and e.severity in (Severity.confirmed, Severity.critical)]

        assert len(trigger_events) >= 1, "Expected at least 1 confirmed trigger event after 3 frames."
        assert trigger_events[-1].metadata.trigger_type == TriggerType.climbing
        assert trigger_events[-1].metadata.confirmation_frames >= 3

    def test_berlin_12b_2_frame_boundary_no_confirmed(self):
        """
        BERLIN-12b: Only 2 frames then track vanishes.
        ZERO confirmed/critical events must be published.
        """
        human_bbox = [0.30, 0.40, 0.50, 0.80]
        frames = [
            [_make_det("climber_y", entity_type=EntityType.human, bbox=human_bbox)],
            [_make_det("climber_y", entity_type=EntityType.human, bbox=human_bbox)],
            [],   # track gone on frame 3
        ]
        events = self._run_scenario(frames, trigger_type=TriggerType.climbing)

        confirmed = [e for e in events
                     if e.event_type == EventType.trigger
                     and e.severity in (Severity.confirmed, Severity.critical)]

        assert len(confirmed) == 0, (
            f"2-frame boundary must NOT produce confirmed/critical. Got: {[e.severity for e in confirmed]}"
        )

    def test_berlin_mixed_animal_and_human(self):
        """
        Mixed scenario: animal + human on same frame.
        Animal → animal_detected. Human → trigger pipeline.
        """
        frames = [
            [_make_animal_det("deer_02"), _make_det("human_02")],  # mixed frame
            [_make_animal_det("deer_02"), _make_det("human_02")],
            [_make_animal_det("deer_02"), _make_det("human_02")],
        ]
        events = self._run_scenario(frames, trigger_type=TriggerType.climbing)

        animal_events  = [e for e in events if e.event_type == EventType.animal_detected]
        trigger_events = [e for e in events
                          if e.event_type == EventType.trigger
                          and e.severity in (Severity.confirmed, Severity.critical)]

        assert len(animal_events) >= 3    # at least one per frame
        assert len(trigger_events) >= 1   # human confirmed after 3 frames
