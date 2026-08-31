"""
Animal Filter — Separates Animal Detections from Trigger Pipeline
SIH26187 | Model A | Step 9 of pipeline

Rule (NON-NEGOTIABLE, per spec Step 9):
  Detect animals via YOLO.
  Log as `animal_detected` (severity=info).
  DO NOT trigger fence alerts for animals.
  Return 0 false positives for animal crossings.

Example: "Deer jumps fence at night. YOLO confidence 0.75."
  → animal_detected event published (severity=info)
  → NO climbing/fence_cutting trigger fires
  → 0 false positives

Borderline case: `animal_cart` (horse-drawn cart)
  → Classified as EntityType.animal_cart
  → Logged as animal_detected
  → NOT a fence trigger

This module returns two lists from classify():
  (animal_detections, human_vehicle_detections)

Caller should:
  1. Build animal_detected events from animal_detections (info severity)
  2. Feed ONLY human_vehicle_detections into TriggerDetector
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from model_a.detector import Detection
from model_a.schema_v1 import EntityType

logger = logging.getLogger(__name__)

# EntityTypes that are classified as animals and must NEVER trigger fence alerts
_ANIMAL_ENTITY_TYPES = {EntityType.animal, EntityType.animal_cart}


class AnimalFilter:
    """
    Splits a list of Detection objects into (animals, non-animals).

    Animals are logged for awareness but excluded from the trigger pipeline.
    Non-animals (human, vehicle, unknown) proceed to TriggerDetector.

    Usage::

        filt = AnimalFilter()
        animals, humans = filt.classify(detections)
        # Build animal_detected events from `animals`
        # Feed `humans` to TriggerDetector
    """

    def __init__(self, min_confidence: float = 0.35) -> None:
        """
        Args:
            min_confidence: Detections below this threshold are discarded entirely
                            (too uncertain to classify as anything meaningful).
        """
        self.min_confidence = min_confidence
        self._total_animals_filtered = 0

    def classify(
        self,
        detections: List[Detection],
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Partition detections into (animal_detections, trigger_candidates).

        Returns:
            animal_detections  — publish as animal_detected (severity=info)
            trigger_candidates — safe to feed into TriggerDetector
        """
        animal_detections:  List[Detection] = []
        trigger_candidates: List[Detection] = []

        for det in detections:
            if det.confidence < self.min_confidence:
                logger.debug(
                    "Detection discarded (conf=%.2f < min=%.2f class=%s)",
                    det.confidence, self.min_confidence, det.class_name,
                )
                continue

            if det.entity_type in _ANIMAL_ENTITY_TYPES:
                animal_detections.append(det)
                self._total_animals_filtered += 1
                logger.info(
                    "ANIMAL detected — track=%s class=%s conf=%.2f "
                    "→ animal_detected (info). Fence trigger SUPPRESSED.",
                    det.track_id, det.class_name, det.confidence,
                )
            else:
                trigger_candidates.append(det)

        return animal_detections, trigger_candidates

    def stats(self) -> dict:
        """Return cumulative filtering statistics."""
        return {"total_animals_filtered": self._total_animals_filtered}
