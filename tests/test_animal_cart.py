"""
Animal-Cart Fuser — Unit Tests
SIH26187 | Phase 1 Extension

Tests required per implementation plan:
  CART-01: Animal alone (no vehicle) → stays entity_type: animal
  CART-02: Vehicle alone (no animal) → stays entity_type: vehicle, trigger pipeline
  CART-03: Animal + vehicle in proximity, 1 frame → NO fusion (must not confirm early)
  CART-04: Animal + vehicle in proximity, 2 frames → fused to animal_cart
  CART-05: Animal + vehicle adjacent, then vehicle moves away → fusion resets
  CART-06: Two animals, no vehicles → no fusion

Run:
  pytest tests/test_animal_cart.py -v
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from model_a.animal_cart_fuser import AnimalCartFuser, _DEFAULT_MIN_CONFIRMATION_FRAMES
from model_a.detector import Detection
from model_a.schema_v1 import EntityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_det(
    entity_type: EntityType,
    bbox: List[float],
    track_id: str,
    confidence: float = 0.75,
    class_name: str = "test",
) -> Detection:
    return Detection(
        track_id    = track_id,
        entity_type = entity_type,
        confidence  = confidence,
        bbox        = bbox,
        class_id    = 0,
        class_name  = class_name,
        frame_number = 1,
    )


def _animal(track_id: str = "a1", bbox: List[float] = None) -> Detection:
    bbox = bbox or [0.10, 0.70, 0.30, 0.90]   # lower-left area
    return _make_det(EntityType.animal, bbox, track_id, class_name="horse")


def _vehicle(track_id: str = "v1", bbox: List[float] = None) -> Detection:
    bbox = bbox or [0.25, 0.70, 0.45, 0.90]   # adjacent to animal bbox
    return _make_det(EntityType.vehicle, bbox, track_id, class_name="cart")


def _vehicle_far(track_id: str = "v1") -> Detection:
    """Vehicle far from any animal (top-right corner)."""
    return _make_det(EntityType.vehicle, [0.75, 0.05, 0.95, 0.25], track_id)


# ---------------------------------------------------------------------------
# CART-01: Animal alone → entity_type stays animal, nothing fused
# ---------------------------------------------------------------------------

class TestCart01AnimalAlone:
    def test_animal_alone_no_fusion(self):
        fuser = AnimalCartFuser()
        animal_dets  = [_animal()]
        vehicle_dets = []

        for frame in range(5):
            a_out, v_out, cart_out = fuser.fuse(animal_dets, vehicle_dets, frame)

        assert len(cart_out) == 0,   "No cart fused when no vehicle present"
        assert len(a_out) == 1,      "Animal detection still returned"
        assert a_out[0].entity_type == EntityType.animal

    def test_animal_entity_type_unchanged(self):
        fuser = AnimalCartFuser()
        a_out, _, cart_out = fuser.fuse([_animal()], [], frame_number=1)
        assert a_out[0].entity_type == EntityType.animal
        assert cart_out == []


# ---------------------------------------------------------------------------
# CART-02: Vehicle alone → entity_type stays vehicle, trigger pipeline untouched
# ---------------------------------------------------------------------------

class TestCart02VehicleAlone:
    def test_vehicle_alone_stays_in_trigger_list(self):
        fuser = AnimalCartFuser()
        vehicle = _vehicle()

        for frame in range(5):
            a_out, v_out, cart_out = fuser.fuse([], [vehicle], frame)

        assert len(cart_out) == 0,   "No cart fused when no animal present"
        assert len(v_out) == 1,      "Vehicle detection returned unfused"
        assert v_out[0].entity_type == EntityType.vehicle

    def test_vehicle_entity_type_unchanged(self):
        fuser = AnimalCartFuser()
        _, v_out, _ = fuser.fuse([], [_vehicle()], frame_number=1)
        assert v_out[0].entity_type == EntityType.vehicle


# ---------------------------------------------------------------------------
# CART-03: Animal + vehicle in proximity, exactly 1 frame → NO fusion yet
# ---------------------------------------------------------------------------

class TestCart03OnlyOneFrame:
    def test_single_frame_proximity_no_fusion(self):
        """
        Core multi-frame discipline test.
        1 frame of proximity must NOT produce an animal_cart event.
        Mirrors the 2-frame boundary test in BERLIN-01 for trigger_detector.
        """
        fuser = AnimalCartFuser(min_confirmation_frames=2)

        a_out, v_out, cart_out = fuser.fuse(
            [_animal()], [_vehicle()], frame_number=1
        )

        assert cart_out == [], (
            "FAIL: fused on a single frame — violates multi-frame discipline. "
            "Expected 0 cart_dets after 1 frame, got %d" % len(cart_out)
        )
        # Originals should still be present (not removed)
        assert len(a_out) == 1, "Animal should NOT be removed before fusion confirmed"
        assert len(v_out) == 1, "Vehicle should NOT be removed before fusion confirmed"

    def test_pair_consecutive_counter_is_1_after_first_frame(self):
        """Internal state check: consecutive_frames should be 1, not 0 or 2."""
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        fuser.fuse([_animal()], [_vehicle()], frame_number=1)

        assert len(fuser._pairs) == 1
        pair_rec = list(fuser._pairs.values())[0]
        assert pair_rec.consecutive_frames == 1


# ---------------------------------------------------------------------------
# CART-04: Animal + vehicle in proximity, 2 frames → fused to animal_cart
# ---------------------------------------------------------------------------

class TestCart04TwoFramesFusion:
    def test_two_frames_produces_animal_cart(self):
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal()
        vehicle = _vehicle()

        # Frame 1 — candidate only
        fuser.fuse([animal], [vehicle], frame_number=1)

        # Frame 2 — should confirm and fuse
        a_out, v_out, cart_out = fuser.fuse([animal], [vehicle], frame_number=2)

        assert len(cart_out) == 1, (
            "Expected 1 animal_cart detection after 2 frames, got %d" % len(cart_out)
        )
        assert cart_out[0].entity_type == EntityType.animal_cart

    def test_fused_detection_suppressed_from_trigger_candidates(self):
        """After fusion, both original animal and vehicle must be removed."""
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal(track_id="a1")
        vehicle = _vehicle(track_id="v1")

        fuser.fuse([animal], [vehicle], frame_number=1)
        a_out, v_out, cart_out = fuser.fuse([animal], [vehicle], frame_number=2)

        # Animal no longer in animal list (was fused)
        assert not any(d.track_id == "a1" for d in a_out), \
            "Fused animal should be removed from animal_dets_out"
        # Vehicle no longer in vehicle list (was fused)
        assert not any(d.track_id == "v1" for d in v_out), \
            "Fused vehicle should be removed from vehicle_dets_out"

    def test_fused_cart_bbox_is_union(self):
        """Union bbox must contain both original bboxes."""
        fuser   = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal(bbox=[0.10, 0.70, 0.30, 0.90])
        vehicle = _vehicle(bbox=[0.25, 0.70, 0.45, 0.90])

        fuser.fuse([animal], [vehicle], frame_number=1)
        _, _, cart_out = fuser.fuse([animal], [vehicle], frame_number=2)

        assert len(cart_out) == 1
        cx1, cy1, cx2, cy2 = cart_out[0].bbox
        # Union should span both bboxes
        assert cx1 == pytest.approx(0.10, abs=1e-6)
        assert cy1 == pytest.approx(0.70, abs=1e-6)
        assert cx2 == pytest.approx(0.45, abs=1e-6)
        assert cy2 == pytest.approx(0.90, abs=1e-6)

    def test_fused_confidence_is_max(self):
        """Fused confidence = max(animal.conf, vehicle.conf)."""
        fuser   = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal(); animal.confidence  = 0.62
        vehicle = _vehicle(); vehicle.confidence = 0.81

        fuser.fuse([animal], [vehicle], frame_number=1)
        _, _, cart_out = fuser.fuse([animal], [vehicle], frame_number=2)

        assert cart_out[0].confidence == pytest.approx(0.81, abs=1e-6)

    def test_fused_entity_type_is_animal_cart(self):
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        fuser.fuse([_animal()], [_vehicle()], frame_number=1)
        _, _, cart_out = fuser.fuse([_animal()], [_vehicle()], frame_number=2)
        assert cart_out[0].entity_type == EntityType.animal_cart


# ---------------------------------------------------------------------------
# CART-05: Animal + vehicle adjacent, then vehicle moves away → fusion resets
# ---------------------------------------------------------------------------

class TestCart05FusionResets:
    def test_pair_resets_when_not_in_proximity(self):
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        animal = _animal()

        # Frame 1: adjacent (would be candidate)
        fuser.fuse([animal], [_vehicle()], frame_number=1)

        # Frame 2: vehicle moves far away
        a_out, v_out, cart_out = fuser.fuse(
            [animal], [_vehicle_far()], frame_number=2
        )

        assert cart_out == [], "Pair separated — must NOT fuse"
        # Vehicle should remain in trigger candidates (unfused)
        assert len(v_out) == 1

    def test_counter_resets_to_zero_on_separation(self):
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal(track_id="a1")
        vehicle_near = _vehicle(track_id="v1")
        vehicle_far  = _vehicle_far(track_id="v1")

        fuser.fuse([animal], [vehicle_near], frame_number=1)   # counter = 1

        # Next frame: same track IDs but far apart
        fuser.fuse([animal], [vehicle_far], frame_number=2)    # counter resets

        # Pair key should now have consecutive_frames = 0
        pair_key = ("a1", "v1")
        if pair_key in fuser._pairs:
            assert fuser._pairs[pair_key].consecutive_frames == 0

    def test_re_approach_restarts_count_from_one(self):
        """After separation, re-approach requires full 2-frame confirmation again."""
        fuser   = AnimalCartFuser(min_confirmation_frames=2)
        animal  = _animal()
        vehicle = _vehicle()

        # Frames 1-2: near (would confirm on frame 2 under normal flow,
        # but then they separate)
        fuser.fuse([animal], [vehicle], frame_number=1)
        fuser.fuse([animal], [_vehicle_far()], frame_number=2)  # reset

        # Frame 3: back near, but counter is 1 (re-started)
        _, _, cart_out = fuser.fuse([animal], [vehicle], frame_number=3)
        assert cart_out == [], "Must NOT fuse on re-approach frame 1 (restarted count)"

        # Frame 4: second consecutive re-approach frame → should fuse now
        _, _, cart_out = fuser.fuse([animal], [vehicle], frame_number=4)
        assert len(cart_out) == 1, "Should fuse after 2 consecutive re-approach frames"


# ---------------------------------------------------------------------------
# CART-06: Two animals, no vehicles → no fusion
# ---------------------------------------------------------------------------

class TestCart06TwoAnimalsNoVehicles:
    def test_two_animals_no_fusion(self):
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        a1 = _animal(track_id="a1", bbox=[0.10, 0.70, 0.30, 0.90])
        a2 = _animal(track_id="a2", bbox=[0.28, 0.70, 0.48, 0.90])

        for frame in range(5):
            a_out, v_out, cart_out = fuser.fuse([a1, a2], [], frame)

        assert cart_out == [], "Two adjacent animals must NOT fuse into animal_cart"
        assert len(a_out) == 2, "Both animals still present"
        assert all(d.entity_type == EntityType.animal for d in a_out)

    def test_no_candidate_pairs_tracked(self):
        """No pairs should be tracked when there are no vehicles."""
        fuser = AnimalCartFuser()
        fuser.fuse([_animal("a1"), _animal("a2")], [], frame_number=1)
        assert len(fuser._pairs) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCartEdgeCases:
    def test_min_confirmation_frames_must_be_at_least_2(self):
        with pytest.raises(ValueError, match="min_confirmation_frames must be"):
            AnimalCartFuser(min_confirmation_frames=1)

    def test_empty_lists_no_crash(self):
        fuser = AnimalCartFuser()
        a, v, c = fuser.fuse([], [], frame_number=1)
        assert a == [] and v == [] and c == []

    def test_stale_pairs_purged(self):
        """Pairs not seen within timeout window are removed from memory."""
        fuser = AnimalCartFuser(min_confirmation_frames=2)
        fuser.fuse([_animal()], [_vehicle()], frame_number=1)
        assert len(fuser._pairs) == 1

        # Jump 10 frames ahead — stale timeout is 5
        fuser.fuse([], [], frame_number=10)
        assert len(fuser._pairs) == 0, "Stale pair should be purged"

    def test_animal_cart_entity_type_already_in_schema(self):
        """Confirm schema_v1 is not being changed — EntityType.animal_cart exists."""
        assert EntityType.animal_cart == EntityType.animal_cart
        assert EntityType.animal_cart.value == "animal_cart"
