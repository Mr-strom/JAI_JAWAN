"""
Animal-Cart Proximity Fuser — Multi-Frame Animal+Vehicle Fusion
SIH26187 | Model A | Extension (approved 2026-09-01)

PROBLEM:
  A horse-drawn cart (or similar animal-towed vehicle) appears as two
  separate YOLO detections in the same frame — one animal bbox and one
  vehicle bbox. The current AnimalFilter.classify() correctly suppresses
  the animal from the trigger pipeline, but:
    1. The vehicle detection may slip through to TriggerDetector and fire
       a false alarm.
    2. There is no "this is an animal_cart" classification — the two
       detections are treated as unrelated objects.

SOLUTION:
  AnimalCartFuser sits between AnimalFilter.classify() and the trigger
  pipeline in FramePipeline.process(). It inspects (animal_dets,
  vehicle_dets) from the SAME frame:
    - For every animal×vehicle pair, check proximity (IoU OR center-distance).
    - Track candidate pairs across consecutive frames using a
      ProximityRecord (mirrors trigger_detector.py's per-track dict pattern).
    - After min_confirmation_frames consecutive close proximity:
        * Synthesize ONE animal_cart Detection (merged bbox, entity_type
          overridden to EntityType.animal_cart, confidence = max of both).
        * Remove both originals from their lists so the vehicle does NOT
          reach TriggerDetector.
    - Result: an animal_cart event is published (severity=info, same as
      plain animal) and NO fence trigger fires.

MULTI-FRAME DISCIPLINE:
  animal_cart is confirmed only after 2 consecutive frames where the pair
  are in proximity. 1-frame proximity → candidate recorded, nothing fused.
  This mirrors the "don't confirm on a single frame" rule used throughout
  this codebase.

DOES NOT TOUCH:
  - schema_v1.py (EntityType.animal_cart already exists)
  - AnimalFilter.classify() signature or behavior
  - trigger_detector.py or bus_client.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from model_a.detector import Detection
from model_a.schema_v1 import EntityType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IoU between animal and vehicle bboxes at or above this → in proximity.
# Low threshold (0.05) because many real-world carts have bboxes that
# touch/slightly overlap rather than fully containing each other.
_DEFAULT_PROXIMITY_IOU = 0.05

# Normalised center-to-center distance at or below this → in proximity.
# 0.15 = 15% of frame width/height. A horse-drawn cart is typically ~0.1–0.2
# frame-widths long, so 0.15 catches adjacent bboxes without going too wide.
_DEFAULT_PROXIMITY_CENTER_DIST = 0.15

# Consecutive frames with proximity required before fusion fires.
# Per spec: "at least 2 consecutive frames" (matches trigger_detector minimum).
_DEFAULT_MIN_CONFIRMATION_FRAMES = 2

# Frames since last seen before a candidate pair is purged from memory.
_STALE_PAIR_TIMEOUT_FRAMES = 5


# ---------------------------------------------------------------------------
# Per-pair state
# ---------------------------------------------------------------------------

@dataclass
class ProximityRecord:
    """Tracks how many consecutive frames an animal+vehicle pair have been
    in close proximity. Mirrors trigger_detector.py's TrackRecord pattern."""
    consecutive_frames: int = 0
    last_seen_frame:    int = -1


# ---------------------------------------------------------------------------
# AnimalCartFuser
# ---------------------------------------------------------------------------

class AnimalCartFuser:
    """
    Fuses co-located animal + vehicle detections into a single animal_cart
    detection after min_confirmation_frames consecutive close frames.

    Usage (from FramePipeline.process())::

        animal_dets, trigger_candidates = self._afilt.classify(detections)

        # Separate vehicles from trigger_candidates for proximity check
        vehicle_dets = [d for d in trigger_candidates
                        if d.entity_type == EntityType.vehicle]
        non_vehicle_triggers = [d for d in trigger_candidates
                                if d.entity_type != EntityType.vehicle]

        animal_dets, vehicle_dets, cart_dets = self._cart_fuse.fuse(
            animal_dets, vehicle_dets, frame_number
        )

        trigger_candidates = non_vehicle_triggers + vehicle_dets
        # cart_dets → publish as animal_detected (info, suppressed from triggers)
    """

    def __init__(
        self,
        proximity_iou_threshold: float = _DEFAULT_PROXIMITY_IOU,
        proximity_center_dist:   float = _DEFAULT_PROXIMITY_CENTER_DIST,
        min_confirmation_frames: int   = _DEFAULT_MIN_CONFIRMATION_FRAMES,
    ) -> None:
        if min_confirmation_frames < 2:
            raise ValueError(
                "min_confirmation_frames must be >= 2. "
                "Fusing on a single frame violates the multi-frame discipline."
            )
        self.proximity_iou_threshold  = proximity_iou_threshold
        self.proximity_center_dist    = proximity_center_dist
        self.min_confirmation_frames  = min_confirmation_frames

        # Dict keyed by (animal_track_id, vehicle_track_id) → ProximityRecord
        self._pairs: Dict[Tuple[str, str], ProximityRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        animal_dets:  List[Detection],
        vehicle_dets: List[Detection],
        frame_number: int,
    ) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """
        Inspect all animal×vehicle pairs for proximity and fuse confirmed ones.

        Returns:
            animal_dets_out  — animal detections NOT fused this frame
            vehicle_dets_out — vehicle detections NOT fused this frame
            animal_cart_dets — newly synthesized animal_cart detections
                               (confirmed pairs only; 1-frame candidates NOT returned)
        """
        animal_cart_dets:   List[Detection] = []
        fused_animal_ids:   set[Optional[str]] = set()
        fused_vehicle_ids:  set[Optional[str]] = set()
        active_pair_keys:   set[Tuple[str, str]] = set()

        # --- Check every animal×vehicle pair ---
        for adet in animal_dets:
            for vdet in vehicle_dets:
                a_key = adet.track_id or id(adet)
                v_key = vdet.track_id or id(vdet)
                pair_key = (str(a_key), str(v_key))

                if self._in_proximity(adet.bbox, vdet.bbox):
                    active_pair_keys.add(pair_key)
                    rec = self._pairs.setdefault(pair_key, ProximityRecord())
                    rec.consecutive_frames += 1
                    rec.last_seen_frame = frame_number

                    logger.debug(
                        "Animal-cart candidate: animal_track=%s vehicle_track=%s "
                        "consecutive=%d/%d",
                        a_key, v_key, rec.consecutive_frames, self.min_confirmation_frames,
                    )

                    if rec.consecutive_frames >= self.min_confirmation_frames:
                        # --- Synthesize animal_cart detection ---
                        cart_det = self._synthesize_cart(adet, vdet)
                        animal_cart_dets.append(cart_det)
                        fused_animal_ids.add(adet.track_id)
                        fused_vehicle_ids.add(vdet.track_id)

                        logger.info(
                            "ANIMAL-CART fused: animal_track=%s vehicle_track=%s "
                            "consecutive=%d → entity_type=animal_cart (info). "
                            "Fence trigger SUPPRESSED.",
                            a_key, v_key, rec.consecutive_frames,
                        )
                else:
                    # Pair was tracked but is no longer in proximity — reset counter
                    if pair_key in self._pairs:
                        logger.debug(
                            "Animal-cart candidate RESET: animal_track=%s "
                            "vehicle_track=%s (no longer in proximity)",
                            a_key, v_key,
                        )
                        self._pairs[pair_key].consecutive_frames = 0

        # --- Purge stale pairs (not seen in _STALE_PAIR_TIMEOUT_FRAMES) ---
        self._purge_stale(frame_number)

        # --- Filter out fused detections from the output lists ---
        animal_dets_out  = [d for d in animal_dets
                            if d.track_id not in fused_animal_ids]
        vehicle_dets_out = [d for d in vehicle_dets
                            if d.track_id not in fused_vehicle_ids]

        return animal_dets_out, vehicle_dets_out, animal_cart_dets

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _in_proximity(self, bbox_a: List[float], bbox_b: List[float]) -> bool:
        """
        True if the two bboxes are in proximity by either metric:
          - IoU >= proximity_iou_threshold  (overlapping / touching bboxes)
          - normalised center-to-center distance <= proximity_center_dist
            (adjacent bboxes with a small gap)
        """
        iou = self._compute_iou(bbox_a, bbox_b)
        if iou >= self.proximity_iou_threshold:
            return True

        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

        return dist <= self.proximity_center_dist

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """IoU of two normalised bboxes [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union  = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _merged_bbox(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
        """Return the union bounding box of two normalised bboxes."""
        return [
            min(bbox_a[0], bbox_b[0]),
            min(bbox_a[1], bbox_b[1]),
            max(bbox_a[2], bbox_b[2]),
            max(bbox_a[3], bbox_b[3]),
        ]

    def _synthesize_cart(
        self, animal_det: Detection, vehicle_det: Detection
    ) -> Detection:
        """
        Merge an animal detection and a vehicle detection into a single
        entity_type=animal_cart detection.

        - bbox: union of both bboxes
        - confidence: max of both (higher of the two is the dominant signal)
        - track_id: animal's track_id (so this event links to the animal track)
        - class_name / class_id: kept from animal det (primary classifier)
        - model_used: annotated as 'animal_cart_fuser'
        """
        return Detection(
            track_id    = animal_det.track_id,
            entity_type = EntityType.animal_cart,
            confidence  = max(animal_det.confidence, vehicle_det.confidence),
            bbox        = self._merged_bbox(animal_det.bbox, vehicle_det.bbox),
            class_id    = animal_det.class_id,
            class_name  = f"{animal_det.class_name}+{vehicle_det.class_name}",
            frame_number = animal_det.frame_number,
            model_used  = "animal_cart_fuser",
        )

    def _purge_stale(self, current_frame: int) -> None:
        """Remove pairs not seen within the stale timeout window."""
        stale = [
            k for k, rec in self._pairs.items()
            if (current_frame - rec.last_seen_frame) > _STALE_PAIR_TIMEOUT_FRAMES
        ]
        for k in stale:
            logger.debug("Purging stale animal-cart pair: %s", k)
            del self._pairs[k]

    def stats(self) -> dict:
        """Return current fuser state for diagnostics."""
        return {
            "active_candidate_pairs": len(self._pairs),
            "pairs": {str(k): v.consecutive_frames for k, v in self._pairs.items()},
        }
