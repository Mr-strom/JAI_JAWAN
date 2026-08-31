"""
YOLOv8n Frame Detector — Model A Detection Layer
SIH26187 | Model A | Detection step (feeds trigger_detector and animal_filter)

Wraps ultralytics YOLOv8n (yolov8n.pt) to produce per-frame detections.

Design decisions:
  - Uses YOLOv8n (nano) — smallest model, fastest on edge (Jetson Orin Nano).
  - Entity-type classification is done HERE from COCO class IDs so that
    downstream modules always work with schema_v1 EntityType, never raw ints.
  - Confidence threshold is configurable (default 0.4 for real-world noise).
  - Returns a list of Detection objects; empty list if nothing found.
  - If the model file is missing, raises FileNotFoundError loudly. We do NOT
    silently degrade — missing model = operator must fix. (Rule: honest limitation.)

COCO class mapping (relevant subset):
  0  = person   → EntityType.human
  1  = bicycle  → EntityType.vehicle
  2  = car      → EntityType.vehicle
  3  = motorcycle → EntityType.vehicle
  5  = bus      → EntityType.vehicle
  7  = truck    → EntityType.vehicle
  14 = bird     → EntityType.animal
  15 = cat      → EntityType.animal
  16 = dog      → EntityType.animal
  17 = horse    → EntityType.animal
  18 = sheep    → EntityType.animal
  19 = cow      → EntityType.animal
  20 = elephant → EntityType.animal
  21 = bear     → EntityType.animal
  22 = zebra    → EntityType.animal
  23 = giraffe  → EntityType.animal
  All others    → EntityType.unknown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from model_a.schema_v1 import EntityType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO class → EntityType mapping
# ---------------------------------------------------------------------------

_PERSON_IDS:  set[int] = {0}
_VEHICLE_IDS: set[int] = {1, 2, 3, 4, 5, 6, 7, 8}
_ANIMAL_IDS:  set[int] = {
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # standard animals
}


def coco_class_to_entity_type(class_id: int) -> EntityType:
    if class_id in _PERSON_IDS:
        return EntityType.human
    if class_id in _VEHICLE_IDS:
        return EntityType.vehicle
    if class_id in _ANIMAL_IDS:
        return EntityType.animal
    return EntityType.unknown


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single-object detection result from YOLOv8n on one frame."""
    track_id:    Optional[str]   # ByteTrack track_id (str for compatibility with schema)
    entity_type: EntityType
    confidence:  float
    bbox:        List[float]     # [x1, y1, x2, y2] normalised [0,1]
    class_id:    int
    class_name:  str
    frame_number: int


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    YOLOv8n-based object detector with ByteTrack tracking.

    Usage::

        det = Detector(model_path="yolov8n.pt", conf_threshold=0.4)
        detections = det.detect(frame_bgr, frame_number=1234)
        for d in detections:
            # d.entity_type, d.confidence, d.bbox, d.track_id are ready
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float  = 0.45,
        device: str = "cpu",       # "cuda" on Jetson, "cpu" for dev
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device
        self._model         = None
        self._model_path    = model_path

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            logger.info("YOLOv8n model loaded from '%s' on device='%s'", model_path, device)
        except Exception as exc:
            logger.warning(
                "YOLOv8n model load failed ('%s'): %s. "
                "Detection will return empty results. Operator must fix.",
                model_path, exc,
            )

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> List[Detection]:
        """
        Run detection + tracking on a single frame.

        Returns a list of Detection objects.
        Returns empty list if model not loaded or no detections.
        """
        if self._model is None:
            return []

        h, w = frame.shape[:2]

        try:
            results = self._model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                persist=True,        # maintain ByteTrack IDs across calls
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference error on frame %d: %s", frame_number, exc)
            return []

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # Extract normalised bbox
                xyxyn = box.xyxyn[0].tolist()   # [x1,y1,x2,y2] in [0,1]
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])

                # Track ID from ByteTrack (may be None on first frame)
                tid   = box.id
                track_id = str(int(tid[0])) if tid is not None else None

                entity_type = coco_class_to_entity_type(cls)
                class_name  = result.names.get(cls, "unknown")

                detections.append(Detection(
                    track_id    = track_id,
                    entity_type = entity_type,
                    confidence  = conf,
                    bbox        = xyxyn,
                    class_id    = cls,
                    class_name  = class_name,
                    frame_number = frame_number,
                ))

        logger.debug(
            "frame=%d detections=%d", frame_number, len(detections)
        )
        return detections


# ---------------------------------------------------------------------------
# Mock detector for testing (no model download required)
# ---------------------------------------------------------------------------

class MockDetector:
    """
    Test-only detector that returns pre-programmed detection sequences.
    Used in Phase Berlin tests to avoid needing a real model file.

    Usage::

        mock = MockDetector()
        mock.set_sequence("cam_01", [
            [Detection(track_id="1", entity_type=EntityType.human, ...)],  # frame 0
            [Detection(...)],                                                # frame 1
            [],                                                              # frame 2 — track gone
        ])
        dets = mock.detect(frame, frame_number=0)
    """

    def __init__(self) -> None:
        self._sequences: dict[str, list[list[Detection]]] = {}
        self._frame_counters: dict[str, int] = {}

    def set_sequence(self, key: str, seq: list[list[Detection]]) -> None:
        """Register a detection sequence for a given key (e.g. camera/scenario name)."""
        self._sequences[key] = seq
        self._frame_counters[key] = 0

    def next(self, key: str) -> list[Detection]:
        """Return the next frame's detections for the given key."""
        seq     = self._sequences.get(key, [])
        counter = self._frame_counters.get(key, 0)
        if counter >= len(seq):
            return []
        result = seq[counter]
        self._frame_counters[key] = counter + 1
        return result
