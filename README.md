# SIH26187 Model B — ANPR Engine (Automatic Number Plate Recognition)

This repository contains the standalone **ANPR Engine** for Model B in the **SIH26187 AI Border-Surveillance Platform** (Ministry of Home Affairs / SSB).

---

## 1. Locked Scope & Architecture
- **Camera Scope**: Strictly Chokepoint & ICP (Integrated Checkpost) cameras (`zone in ["chokepoint", "icp"]`). All perimeter/fence/riverine cameras and non-vehicle entities are filtered out immediately.
- **Tech Stack**:
  - **YOLOv8** for number plate localization within the vehicle ROI.
  - **PaddleOCR** (`use_angle_cls=True`, multi-script support for Latin & Devanagari numerals/characters).
- **Runtime**: Pure MQTT pub/sub event pipeline (no FastAPI/HTTP wrappers).

---

## 2. Module Breakdown

| Module | File | Purpose |
|---|---|---|
| **Core ANPR Engine** | [`anpr/engine.py`](file:///c:/Users/pkuma/projects/Jai-jawan(bhumesh)-modelb/anpr/engine.py) | Full pipeline orchestrator matching the `ANPREngine` interface. |
| **Indian Plate Validator** | [`anpr/indian_plate_validator.py`](file:///c:/Users/pkuma/projects/Jai-jawan(bhumesh)-modelb/anpr/indian_plate_validator.py) | Validates State codes, Bharat (BH) series, Defense/Military, Diplomatic, and handles Devanagari transliteration & OCR character disambiguation. |
| **Perspective Correction** | [`anpr/perspective_correction.py`](file:///c:/Users/pkuma/projects/Jai-jawan(bhumesh)-modelb/anpr/perspective_correction.py) | **Provisional** homography & contour rectification. Gracefully degrades to uncorrected crop with `perspective_corrected=False`. |
| **Confidence Calculator** | [`anpr/confidence_calculator.py`](file:///c:/Users/pkuma/projects/Jai-jawan(bhumesh)-modelb/anpr/confidence_calculator.py) | Computes genuine dynamic confidence based on detector score, OCR certainty, tilt angle, and syntax purity. |
| **MQTT Consumer** | [`anpr/mqtt_consumer.py`](file:///c:/Users/pkuma/projects/Jai-jawan(bhumesh)-modelb/anpr/mqtt_consumer.py) | Subscribes to upstream Model A topics and publishes results. |

---

## 3. MQTT Output Contract Compliance

- **Topic**: `sih26187/camera/{cam_id}/model_b/anpr`
- **Output JSON Structure**:
```json
{
  "event_id": "b347e6d2-dca1-416c-bbfc-3fd394042873",
  "engine_source": "anpr",
  "camera_id": "cam_chokepoint_01",
  "timestamp": "2026-09-02T11:24:00.120Z",
  "topic": "sih26187/camera/cam_chokepoint_01/model_b/anpr",
  "entity_id": "trk_truck_08",
  "plate_number": "DL01AB1234",
  "raw_text": "DL 01 AB 1234",
  "confidence": 0.9620,
  "state_code": "DL",
  "format_type": "STANDARD",
  "has_devanagari": false,
  "human_verification_required": false,
  "review_status": "HIGH_CONFIDENCE",
  "plate_evidence_ref": null,
  "metadata": {
    "plate_detector": "yolov8n-plate",
    "ocr_model": "paddleocr-multiscript",
    "perspective_corrected": false,
    "format_validated": true,
    "processing_time_ms": 14.5,
    "detector_confidence": 0.95,
    "ocr_confidence": 0.94,
    "tilt_angle_deg": 1.2
  }
}
```

---

## 4. Verification & Testing

Run all unit tests:
```bash
python -m unittest discover -s tests -v
```

Run demo against Model A dummy events:
```bash
python main.py
```
