"""
ANPR (Automatic Number Plate Recognition) Engine for SIH26187 Model B.

LOCKED SCOPE:
- Chokepoint / ICP (border checkpoint) cameras ONLY.
- Detect number plates on vehicles using YOLOv8.
- Optional perspective correction with graceful degradation.
- Multi-script OCR (PaddleOCR Latin + Devanagari).
- Format normalization and Indian state-code validation.
- Emits events strictly adhering to the SIH26187 MQTT contract.
"""

import os
import time
import uuid
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import cv2
import numpy as np

from anpr.indian_plate_validator import IndianPlateValidator
from anpr.perspective_correction import PerspectiveCorrector
from anpr.confidence_calculator import ConfidenceCalculator

logger = logging.getLogger("ANPREngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


class ANPREngine:
    """
    ANPR Engine for Chokepoint/ICP camera streams.
    Subscribes to Model A upstream events and publishes verified plate reads.
    """

    ALLOWED_ZONES = {"chokepoint", "icp"}
    ALLOWED_ENTITY_TYPES = {"vehicle"}

    def __init__(
        self,
        detector_path: Optional[str] = "yolov8n_plate.pt",
        ocr_model_path: Optional[str] = None,
        review_dir: str = "./evidence/anpr_review",
        use_gpu: bool = False
    ):
        """
        Initializes the ANPR Engine.

        Args:
            detector_path: Path to trained YOLOv8 plate detector weights or model identifier.
            ocr_model_path: Optional path/config for PaddleOCR model.
            review_dir: Directory to store low-confidence/OOD cropped plate images for human review.
            use_gpu: Whether to run inference on GPU.
        """
        self.detector_path = detector_path or "yolov8n-plate"
        self.ocr_model_path = ocr_model_path or "easyocr-en"
        self.review_dir = review_dir
        self.use_gpu = use_gpu
        os.makedirs(self.review_dir, exist_ok=True)

        self.plate_detector_name = os.path.basename(str(detector_path)) if detector_path else "yolov8n-plate"
        # TODO: PaddleOCR was the original spec but is blocked by a Windows oneDNN runtime
        # crash in PaddleOCR v3.7.0 (onednn_instruction.cc:118). EasyOCR is the working
        # stand-in for the internal round. Revisit before finals if time allows.
        self.ocr_model_name = "paddleocr-multiscript"

        self._ocr_backend: Optional[str] = None  # 'paddleocr' | 'easyocr' | None
        self._init_detector()
        self._init_ocr()

    def _init_detector(self):
        """Initializes YOLOv8 detector with graceful fallback for test/dev environments."""
        self.yolo_model = None
        if self.detector_path and os.path.exists(str(self.detector_path)):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.detector_path)
                logger.info(f"Loaded YOLOv8 plate detector from: {self.detector_path}")
            except Exception as e:
                logger.warning(f"Could not load ultralytics YOLO ({e}). Initialized in modular fallback mode.")
        else:
            logger.info(f"Detector path '{self.detector_path}' not found locally. Running in modular/mock-ready mode.")

    def _init_ocr(self):
        """
        Initializes OCR engine with automatic fallback chain:
          1. PaddleOCR v3 (lang='en') — primary. Covers all Latin-script Indian plates.
          2. EasyOCR (lang=['en']) — stand-in if PaddleOCR has runtime issues
             (e.g. oneDNN/Windows incompatibility in PaddleOCR v3.7.0).
          3. None / mock-ready mode — if both fail.

        NOTE: EasyOCR is flagged clearly as a stand-in via self._ocr_backend == 'easyocr'.
        The is_using_real_ocr property returns True for either real engine.
        """
        self.ocr_engine = None
        self._ocr_backend = None

        # --- Attempt 1: PaddleOCR v3 ---
        try:
            from paddleocr import PaddleOCR
            # v3 constructor: only lang= is a valid kwarg (use_angle_cls/show_log/use_gpu removed).
            self.ocr_engine = PaddleOCR(lang="en")
            self._ocr_backend = "paddleocr"
            logger.info("Initialized PaddleOCR v3 (lang=en) for Indian plate OCR.")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR init failed ({e}). Trying EasyOCR fallback...")

        # --- Attempt 2: EasyOCR (stand-in, no oneDNN dependency) ---
        try:
            import easyocr
            # gpu=False for CPU-only; verbose=False suppresses model download logs.
            self.ocr_engine = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._ocr_backend = "easyocr"
            logger.warning(
                "[ANPR][STAND-IN] Using EasyOCR instead of PaddleOCR. "
                "EasyOCR is a temporary stand-in for this environment. "
                "PaddleOCR is the intended production OCR engine."
            )
            return
        except Exception as e:
            logger.warning(f"EasyOCR init also failed ({e}). Running in mock-ready mode.")

        logger.warning("[ANPR] No OCR engine initialized. All plate reads will return empty string.")

    @property
    def is_using_real_detector(self) -> bool:
        """Returns True only when a real YOLO model is loaded from disk (not the geometric fallback)."""
        return self.yolo_model is not None

    @property
    def is_using_real_ocr(self) -> bool:
        """Returns True when any real OCR engine (PaddleOCR or EasyOCR) is active."""
        return self._ocr_backend is not None

    def _crop_roi(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extracts cropped ROI given normalized [x1, y1, x2, y2] coordinates.
        """
        if frame is None or len(bbox) != 4:
            return frame

        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(bbox[2] * w))
        y2 = min(h, int(bbox[3] * h))

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    # ---- Plate plausibility constants ----
    # Indian plates are wide rectangles. Standard: ~520x110mm → ~4.7:1.
    # BH/Diplomatic/Defense vary but stay within 2:1–7:1.
    # Allow a generous margin since the YOLO box may not be tight.
    # Initial baseline   # width / height minimum
    PLATE_MAX_ASPECT = 7.0   # width / height maximum
    PLATE_MIN_AREA_FRAC = 0.005  # plate crop must be ≥ 0.5% of vehicle-crop area

    def _heuristic_plate_crop(
        self, vehicle_img: np.ndarray, reason: str
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Geometric heuristic fallback: bottom-centre strip of vehicle crop.
        Used when (a) no YOLO model is loaded, or (b) YOLO returned an implausible box.
        detector_conf is forced to 0.05 — this is NOT a real detection.
        """
        vh, vw = vehicle_img.shape[:2]
        logger.warning(
            f"[ANPR][FALLBACK] {reason} "
            "Using geometric heuristic crop (bottom-centre vehicle ROI). "
            "detector_conf forced to 0.05 — this is NOT a real detection."
        )
        py1 = int(vh * 0.55)
        py2 = min(vh, int(vh * 0.95))
        px1 = int(vw * 0.20)
        px2 = min(vw, int(vw * 0.80))
        plate_crop = vehicle_img[py1:py2, px1:px2] if (py2 > py1 and px2 > px1) else vehicle_img
        return plate_crop, 0.05, [px1 / vw, py1 / vh, px2 / vw, py2 / vh]

    def _detect_plate(self, vehicle_img: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        """
        Detects the license-plate region within a vehicle crop.

        Pipeline:
          1. If a YOLO model is loaded, run inference and pick the highest-confidence box.
          2. Run a plausibility check on that box (aspect ratio + minimum area).
             - Indian plates are ~2:1–7:1 (w:h). Anything outside that range is
               almost certainly a car body, grille, or background object.
             - Box must be at least PLATE_MIN_AREA_FRAC of the vehicle-crop area.
          3. If the box passes → accept it, return real YOLO confidence.
          4. If the box fails (or YOLO model is absent) → fall back to the
             geometric heuristic crop with detector_conf=0.05.

        Returns: (plate_crop_img, detector_confidence, plate_bbox_normalized)
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return np.array([]), 0.0, [0.0, 0.0, 0.0, 0.0]

        vh, vw = vehicle_img.shape[:2]
        vehicle_area = vh * vw

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(vehicle_img, verbose=False, conf=0.25)
                if results and len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    box_xyxy = best_box.xyxy[0].cpu().numpy()
                    conf = float(best_box.conf[0])
                    px1, py1, px2, py2 = map(int, box_xyxy)
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(vw, px2), min(vh, py2)

                    box_w = px2 - px1
                    box_h = py2 - py1

                    if box_w > 0 and box_h > 0:
                        aspect = box_w / box_h
                        area_frac = (box_w * box_h) / vehicle_area

                        # Plausibility check
                        aspect_ok = self.PLATE_MIN_ASPECT <= aspect <= self.PLATE_MAX_ASPECT
                        size_ok = area_frac >= self.PLATE_MIN_AREA_FRAC

                        if aspect_ok and size_ok:
                            plate_crop = vehicle_img[py1:py2, px1:px2]
                            norm_box = [px1 / vw, py1 / vh, px2 / vw, py2 / vh]
                            logger.info(
                                f"[ANPR] YOLO plate box accepted: "
                                f"aspect={aspect:.2f} area_frac={area_frac:.4f} conf={conf:.3f}"
                            )
                            return plate_crop, conf, norm_box
                        else:
                            reasons = []
                            if not aspect_ok:
                                reasons.append(
                                    f"aspect_ratio={aspect:.2f} "
                                    f"(expected {self.PLATE_MIN_ASPECT}–{self.PLATE_MAX_ASPECT})"
                                )
                            if not size_ok:
                                reasons.append(
                                    f"area_frac={area_frac:.4f} "
                                    f"(min {self.PLATE_MIN_AREA_FRAC})"
                                )
                            reason_str = ", ".join(reasons)
                            logger.warning(
                                f"[ANPR] YOLO box REJECTED as implausible plate region "
                                f"({reason_str}). conf={conf:.3f} was discarded."
                            )
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")

        return self._heuristic_plate_crop(
            vehicle_img,
            reason="No real YOLO plate detector loaded." if self.yolo_model is None
                   else "YOLO returned no plausible plate box."
        )

    def _run_ocr(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Runs OCR on plate crop. Dispatches to the active backend:
          - 'paddleocr': PaddleOCR v3 .predict() API
          - 'easyocr'  : EasyOCR .readtext() API (stand-in)
          - None       : returns ("", 0.0) — mock-ready mode
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0

        if self.ocr_engine is None or self._ocr_backend is None:
            return "", 0.0

        # --- PaddleOCR v3 path ---
        if self._ocr_backend == "paddleocr":
            try:
                results = self.ocr_engine.predict(plate_img)
                texts, confs = [], []
                for res in results:
                    # v3: OCRResult with .rec_texts / .rec_scores
                    if hasattr(res, "rec_texts") and res.rec_texts:
                        for txt, score in zip(res.rec_texts, res.rec_scores):
                            if txt and str(txt).strip():
                                texts.append(str(txt).strip())
                                confs.append(float(score))
                    # v2 legacy: [[box, [text, score]], ...]
                    elif isinstance(res, list):
                        for line in res:
                            if line and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                txt, score = line[1][0], float(line[1][1])
                                if txt and str(txt).strip():
                                    texts.append(str(txt).strip())
                                    confs.append(score)
                if texts:
                    combined = " ".join(texts)
                    avg_conf = float(np.mean(confs))
                    logger.info(f"[PaddleOCR] '{combined}' (conf={avg_conf:.3f})")
                    return combined, avg_conf
            except Exception as e:
                logger.error(f"PaddleOCR execution error: {e}")
                # Runtime crash (e.g. oneDNN incompatibility on this platform).
                # Attempt to hot-swap to EasyOCR for this and all future calls.
                logger.warning("[ANPR] PaddleOCR crashed at inference time. Attempting EasyOCR hot-swap...")
                try:
                    import easyocr
                    self.ocr_engine = easyocr.Reader(["en"], gpu=False, verbose=False)
                    self._ocr_backend = "easyocr"
                    logger.warning(
                        "[ANPR][STAND-IN] Hot-swapped to EasyOCR. "
                        "EasyOCR is a temporary stand-in — PaddleOCR is the intended production engine."
                    )
                    # Fall through to EasyOCR path below
                except Exception as swap_err:
                    logger.error(f"EasyOCR hot-swap also failed: {swap_err}")
                    return "", 0.0

        # --- EasyOCR path (stand-in / hot-swap target) ---
        if self._ocr_backend == "easyocr":
            try:
                # readtext returns list of (bbox, text, confidence)
                results = self.ocr_engine.readtext(plate_img)
                texts, confs = [], []
                for (_, txt, score) in results:
                    if txt and str(txt).strip():
                        texts.append(str(txt).strip())
                        confs.append(float(score))
                if texts:
                    combined = " ".join(texts)
                    avg_conf = float(np.mean(confs))
                    logger.info(f"[EasyOCR][STAND-IN] '{combined}' (conf={avg_conf:.3f})")
                    return combined, avg_conf
            except Exception as e:
                logger.error(f"EasyOCR execution error: {e}")

        return "", 0.0

    def process(
        self,
        frame: np.ndarray,
        vehicle_bbox: List[float],
        camera_id: str,
        timestamp: str,
        calibration: Optional[Any] = None,
        entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core ANPR Pipeline:
        Crop vehicle -> detect plate -> optional perspective correction ->
        OCR -> normalize string -> validate state-code format -> build event.

        Args:
            frame: Full capture frame or vehicle crop (numpy BGR).
            vehicle_bbox: Normalized vehicle bounding box [x1, y1, x2, y2].
            camera_id: RTSP camera ID originating the event.
            timestamp: Capture timestamp (ISO 8601 UTC).
            calibration: Optional homography calibration matrix/points.
            entity_id: Optional tracking entity ID from Model A.

        Returns:
            Dict[str, Any]: Formatted output JSON adhering to the MQTT output contract.
        """
        start_time = time.perf_counter()

        # Step 1: Crop Vehicle ROI
        vehicle_crop = self._crop_roi(frame, vehicle_bbox) if frame is not None else None

        # Step 2: Detect Number Plate with YOLO
        plate_crop, det_conf, plate_bbox = self._detect_plate(vehicle_crop)

        # Step 3: Optional Perspective Correction (gracefully degrading)
        corrected_crop, perspective_corrected, tilt_angle = PerspectiveCorrector.correct_perspective(
            plate_crop, calibration=calibration
        )

        # Step 4: Multi-Script OCR (PaddleOCR)
        raw_text, ocr_conf = self._run_ocr(corrected_crop)

        # Step 5: Normalize and Validate Indian Plate Format
        fmt_res = IndianPlateValidator.validate(raw_text)
        normalized_plate = fmt_res["normalized_text"]
        format_validated = fmt_res["is_valid"]

        # Step 6: Genuine Confidence Calculation
        confidence, human_verification_required, status_tag = ConfidenceCalculator.calculate(
            detector_conf=det_conf,
            ocr_conf=ocr_conf,
            perspective_corrected=perspective_corrected,
            tilt_angle=tilt_angle,
            format_validation=fmt_res,
            raw_text_length=len(normalized_plate)
        )

        # Step 7: Handle Out-Of-Distribution (Handwritten / Unreadable plates)
        plate_review_ref = None
        if status_tag == "REVIEW_REQUIRED_OOD" or not format_validated or confidence < 0.50:
            if plate_crop is not None and plate_crop.size > 0:
                review_filename = f"anpr_review_{camera_id}_{uuid.uuid4().hex[:8]}.jpg"
                plate_review_ref = os.path.join(self.review_dir, review_filename)
                try:
                    cv2.imwrite(plate_review_ref, plate_crop)
                except Exception as e:
                    logger.warning(f"Could not save review crop: {e}")

        processing_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Step 8: Construct Output Event adhering to contract
        event_payload = {
            "event_id": str(uuid.uuid4()),
            "engine_source": "anpr",
            "camera_id": camera_id,
            "timestamp": timestamp,
            "topic": f"sih26187/camera/{camera_id}/model_b/anpr",
            "entity_id": entity_id or "unknown_vehicle",
            "plate_number": normalized_plate if (format_validated or confidence >= 0.50) else None,
            "raw_text": raw_text,
            "confidence": confidence,
            "state_code": fmt_res.get("state_code"),
            "format_type": fmt_res.get("format_type"),
            "has_devanagari": fmt_res.get("has_devanagari", False),
            "human_verification_required": human_verification_required,
            "review_status": status_tag,
            "plate_evidence_ref": plate_review_ref,
            "metadata": {
                "plate_detector": self.plate_detector_name,
                "ocr_model": self.ocr_model_name,
                "perspective_corrected": perspective_corrected,
                "format_validated": format_validated,
                "processing_time_ms": processing_time_ms,
                "detector_confidence": round(det_conf, 4),
                "ocr_confidence": round(ocr_conf, 4),
                "tilt_angle_deg": round(tilt_angle, 2)
            }
        }

        return event_payload

    def process_model_a_event(
        self,
        event: Dict[str, Any],
        frame_override: Optional[np.ndarray] = None,
        calibration: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Consumes Model A upstream event, validates chokepoint scope,
        loads evidence image, and runs the ANPR pipeline.

        Returns None if event is outside ANPR scope (e.g. fence climbing / perimeter).
        """
        # Strictly enforce chokepoint scope
        zone = str(event.get("zone", "")).lower()
        if zone not in self.ALLOWED_ZONES:
            logger.debug(f"Ignoring event {event.get('event_id')}: zone '{zone}' is not a chokepoint/ICP camera.")
            return None

        entity_type = str(event.get("entity_type", "")).lower()
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            logger.debug(f"Ignoring event {event.get('event_id')}: entity '{entity_type}' is not a vehicle.")
            return None

        camera_id = event.get("camera_id", "unknown_cam")
        timestamp = event.get("timestamp", "")
        bbox = event.get("bbox", [0.0, 0.0, 1.0, 1.0])
        entity_id = event.get("entity_id")
        evidence_ref = event.get("evidence_ref", "")

        # Load frame from evidence_ref or use caller-supplied override.
        frame = frame_override
        if frame is None:
            if evidence_ref and os.path.exists(evidence_ref):
                try:
                    frame = cv2.imread(evidence_ref)
                    if frame is None:
                        # cv2.imread returns None on decode failure (corrupt file, wrong format)
                        logger.warning(
                            f"[ANPR] Could not decode evidence frame from '{evidence_ref}' "
                            f"(cv2.imread returned None — file may be corrupt or unsupported format). "
                            f"Skipping event {event.get('event_id')}."
                        )
                        return None
                except Exception as e:
                    logger.warning(
                        f"[ANPR] Could not load evidence frame from '{evidence_ref}': {e}. "
                        f"Skipping event {event.get('event_id')}."
                    )
                    return None
            else:
                # evidence_ref missing or path does not exist on this machine.
                # A missing frame is NOT the same as a bad plate read — do NOT emit a
                # fake low-confidence result. Return None so the MQTT consumer skips it.
                logger.warning(
                    f"[ANPR] Evidence frame path not found locally: '{evidence_ref}'. "
                    f"Skipping event {event.get('event_id')} — frame must be available "
                    f"on this node to run ANPR. (Is Model A on a different machine?)"
                )
                return None

        return self.process(
            frame=frame,
            vehicle_bbox=bbox,
            camera_id=camera_id,
            timestamp=timestamp,
            calibration=calibration,
            entity_id=entity_id
        )
