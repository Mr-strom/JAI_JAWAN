"""
Dual-Model Frame Detector — Model A Detection Layer
SIH26187 | Model A | Detection step (feeds trigger_detector and animal_filter)

Design decisions:
  - PRIMARY model: YOLOv8n (nano) — fast (~90ms CPU), used for all close_range detections
    and as the first-pass full-frame scan that produces bboxes for zone routing.
  - LONG-RANGE model: YOLOv8s (small) — lazy-loaded on first use, used ONLY to
    upgrade detections in the long_range zone (small/distant humans).
  - Two-pass strategy for long_range:
      1. v8n detects the full frame and returns preliminary bboxes.
      2. For each bbox classified as long_range by ZoneTagger, caller invokes
         detect_by_zone(frame, "long_range", source_det) which crops the ROI and
         runs v8s on the crop for a higher-confidence result.
      3. The v8n ByteTrack ID is *inherited* by the upgraded detection — v8s runs
         in predict() mode (no tracking) to avoid ByteTrack state collision.
  - detect() is kept for backward compatibility and always uses v8n.
  - Entity-type classification is done HERE from COCO class IDs so that
    downstream modules always work with schema_v1 EntityType, never raw ints.
  - Confidence threshold is configurable (default 0.4 for real-world noise).
  - Returns a list of Detection objects; empty list if nothing found.
  - If the primary model file is missing, logs loudly. Operator must fix.
    If the long-range model is missing, logs WARNING and falls back to v8n.

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
from typing import List, Optional, Tuple

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

# Padding fraction around bbox when cropping for long-range v8s upgrade.
# 10% on each side gives context without pulling in too much background.
_CROP_PADDING = 0.10


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
    """Single-object detection result from YOLOv8 on one frame."""
    track_id:    Optional[str]   # ByteTrack track_id (str for compatibility with schema)
    entity_type: EntityType
    confidence:  float
    bbox:        List[float]     # [x1, y1, x2, y2] normalised [0,1]
    class_id:    int
    class_name:  str
    frame_number: int
    # Which model produced (or upgraded) this detection — for diagnostics
    model_used:  str = field(default="yolov8n")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Dual-model object detector: YOLOv8n (primary) + YOLOv8s (long-range upgrade).

    Typical usage::

        det = Detector()   # loads yolov8n.pt eagerly, yolov8s.pt lazy

        # Full-frame detection with v8n (fast path)
        detections = det.detect(frame_bgr, frame_number=1234)

        # Upgrade a specific bbox with v8s (long-range crop)
        upgraded = det.detect_by_zone(frame_bgr, "long_range", det_obj, frame_number=1234)

    Zone routing is driven by the caller (FramePipeline.process).
    """

    PRIMARY_MODEL_PATH     = "yolov8n.pt"
    LONG_RANGE_MODEL_PATH  = "yolov8s.pt"

    def __init__(
        self,
        model_path:            str   = PRIMARY_MODEL_PATH,
        conf_threshold:        float = 0.40,
        iou_threshold:         float = 0.45,
        device:                str   = "cpu",
        long_range_model_path: str   = LONG_RANGE_MODEL_PATH,
    ) -> None:
        self.conf_threshold         = conf_threshold
        self.iou_threshold          = iou_threshold
        self.device                 = device
        self._model_path            = model_path
        self._long_range_model_path = long_range_model_path
        self._model                 = None   # v8n — loaded eagerly
        self._model_s               = None   # v8s — loaded lazily
        self._model_s_available     = True   # set False if load fails

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            logger.info(
                "YOLOv8n model loaded from '%s' on device='%s' (primary / close-range)",
                model_path, device,
            )
        except Exception as exc:
            logger.warning(
                "YOLOv8n model load failed ('%s'): %s. "
                "Detection will return empty results. Operator must fix.",
                model_path, exc,
            )

    # ------------------------------------------------------------------
    # Lazy-loader for the long-range (v8s) model
    # ------------------------------------------------------------------

    def _load_long_range_model(self) -> bool:
        """
        Lazy-load YOLOv8s on first long-range detection request.
        Returns True if model is ready, False if unavailable (fallback to v8n).
        """
        if self._model_s is not None:
            return True
        if not self._model_s_available:
            return False  # already failed once — don't retry every frame

        try:
            from ultralytics import YOLO
            self._model_s = YOLO(self._long_range_model_path)
            logger.info(
                "YOLOv8s model lazy-loaded from '%s' (long-range upgrade model)",
                self._long_range_model_path,
            )
            return True
        except Exception as exc:
            logger.warning(
                "YOLOv8s model load failed ('%s'): %s. "
                "Long-range detections will fall back to YOLOv8n.",
                self._long_range_model_path, exc,
            )
            self._model_s_available = False
            return False

    # ------------------------------------------------------------------
    # Primary detect() — v8n full-frame, backward-compatible
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> List[Detection]:
        """
        Run YOLOv8n detection + ByteTrack tracking on a single full frame.
        This is the close-range / backward-compat path.
        Returns a list of Detection objects (model_used='yolov8n').
        """
        return self._run_inference(
            model=self._model,
            frame=frame,
            frame_number=frame_number,
            model_name="yolov8n",
            use_tracking=True,
        )

    # ------------------------------------------------------------------
    # detect_by_zone() — routes to v8n (no-op) or v8s crop
    # ------------------------------------------------------------------

    def detect_by_zone(
        self,
        frame:        np.ndarray,
        zone_tag:     str,          # "close_range" | "long_range"
        source_det:   Detection,    # the v8n detection to potentially upgrade
        frame_number: int = 0,
    ) -> Detection:
        """
        Upgrade a single Detection's confidence using the zone-appropriate model.

        - close_range: returns source_det unchanged (v8n result is good).
        - long_range:  crops the ROI from frame, runs YOLOv8s on the crop,
                       and returns an upgraded Detection if a matching entity is
                       found. Falls back to source_det if v8s finds nothing or
                       is unavailable.

        The ByteTrack track_id from source_det is ALWAYS preserved — v8s runs
        in predict() mode (stateless) to avoid stomping ByteTrack's state.
        """
        if zone_tag != "long_range":
            return source_det  # close_range: v8n result is sufficient

        if not self._load_long_range_model():
            logger.debug(
                "frame=%d track=%s: v8s unavailable, keeping v8n (conf=%.3f)",
                frame_number, source_det.track_id, source_det.confidence,
            )
            return source_det

        # --- Crop the ROI with padding ---
        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = source_det.bbox

        pad_x = _CROP_PADDING * (x2n - x1n)
        pad_y = _CROP_PADDING * (y2n - y1n)
        x1c = max(0.0, x1n - pad_x)
        y1c = max(0.0, y1n - pad_y)
        x2c = min(1.0, x2n + pad_x)
        y2c = min(1.0, y2n + pad_y)

        px1, py1, px2, py2 = int(x1c * w), int(y1c * h), int(x2c * w), int(y2c * h)
        if px2 <= px1 or py2 <= py1:
            return source_det  # degenerate crop

        crop = frame[py1:py2, px1:px2]

        # --- Run v8s on crop (predict, no tracking) ---
        try:
            results = self._model_s.predict(
                source=crop,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error(
                "YOLOv8s crop inference error on frame %d track %s: %s",
                frame_number, source_det.track_id, exc,
            )
            return source_det

        # --- Find best matching entity in crop results ---
        best_conf = source_det.confidence  # only upgrade if v8s is strictly better
        best_det: Optional[Detection] = None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                crop_conf   = float(box.conf[0])
                crop_cls    = int(box.cls[0])
                crop_entity = coco_class_to_entity_type(crop_cls)

                if crop_entity != source_det.entity_type:
                    continue  # wrong entity type — skip

                if crop_conf > best_conf:
                    best_conf = crop_conf
                    xyxyn = box.xyxyn[0].tolist()
                    # Remap crop-relative bbox back to full-frame normalised coords
                    full_x1 = x1c + xyxyn[0] * (x2c - x1c)
                    full_y1 = y1c + xyxyn[1] * (y2c - y1c)
                    full_x2 = x1c + xyxyn[2] * (x2c - x1c)
                    full_y2 = y1c + xyxyn[3] * (y2c - y1c)
                    best_det = Detection(
                        track_id     = source_det.track_id,  # inherit ByteTrack ID
                        entity_type  = crop_entity,
                        confidence   = crop_conf,
                        bbox         = [full_x1, full_y1, full_x2, full_y2],
                        class_id     = crop_cls,
                        class_name   = result.names.get(crop_cls, "unknown"),
                        frame_number = frame_number,
                        model_used   = "yolov8s",
                    )

        if best_det is not None:
            logger.debug(
                "frame=%d track=%s long_range UPGRADED: v8n=%.3f → v8s=%.3f",
                frame_number, source_det.track_id,
                source_det.confidence, best_det.confidence,
            )
            return best_det

        logger.debug(
            "frame=%d track=%s long_range: no v8s upgrade (v8n_conf=%.3f kept)",
            frame_number, source_det.track_id, source_det.confidence,
        )
        return source_det

    # ------------------------------------------------------------------
    # detect_full_frame_long_range() — v8s full-frame pass (independent)
    # ------------------------------------------------------------------

    def detect_full_frame_long_range(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> List[Detection]:
        """
        Run YOLOv8s on the full frame with ByteTrack tracking, as an independent
        long-range detection pass.

        This is used by FramePipeline when it needs long_range detections that v8n
        missed entirely (v8n confidence too low → no bbox → no crop to upgrade from).

        Returns a list of Detection objects (model_used='yolov8s').
        The caller filters the returned list to keep only long_range-zoned detections.

        ByteTrack state for v8s is maintained separately from v8n's ByteTrack state
        because they are different model instances — their track IDs occupy a disjoint
        namespace. FramePipeline handles this by using v8s results only for long_range.
        """
        if not self._load_long_range_model():
            logger.debug("frame=%d: v8s unavailable for full-frame long-range pass", frame_number)
            return []

        return self._run_inference(
            model=self._model_s,
            frame=frame,
            frame_number=frame_number,
            model_name="yolov8s",
            use_tracking=True,
        )



    def _run_inference(
        self,
        model,
        frame:        np.ndarray,
        frame_number: int,
        model_name:   str,
        use_tracking: bool = True,
    ) -> List[Detection]:
        """Run YOLO inference and parse results into Detection objects."""
        if model is None:
            return []

        try:
            if use_tracking:
                results = model.track(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
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
                xyxyn    = box.xyxyn[0].tolist()
                conf     = float(box.conf[0])
                cls      = int(box.cls[0])
                tid      = box.id
                track_id = str(int(tid[0])) if tid is not None else None

                detections.append(Detection(
                    track_id     = track_id,
                    entity_type  = coco_class_to_entity_type(cls),
                    confidence   = conf,
                    bbox         = xyxyn,
                    class_id     = cls,
                    class_name   = result.names.get(cls, "unknown"),
                    frame_number = frame_number,
                    model_used   = model_name,
                ))

        logger.debug("frame=%d model=%s detections=%d", frame_number, model_name, len(detections))
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

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> list[Detection]:
        """Compat shim so MockDetector can be dropped in wherever Detector is used."""
        return self.next("default")

    def detect_by_zone(
        self,
        frame:        np.ndarray,
        zone_tag:     str,
        source_det:   Detection,
        frame_number: int = 0,
    ) -> Detection:
        """MockDetector passthrough — always returns source_det unchanged."""
        return source_det
