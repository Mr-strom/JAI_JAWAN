"""
Task 4 End-to-End Real Inference Test.
Ground truth plate: MH20DV2366 (Maharashtra, Skoda Superb, clean front-facing)

Usage: .\venv_anpr\Scripts\python.exe test_images\run_real_inference.py <image_path>
"""
import sys
import os
import json
import time
import importlib

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_TRUTH_PLATE = "MH20DV2366"

def run_inference(image_path: str):
    import cv2
    import numpy as np
    from anpr.engine import ANPREngine

    print("=" * 65)
    print("SIH26187 ANPR Engine — Real Inference Test")
    print(f"Image:        {image_path}")
    print(f"Ground Truth: {GROUND_TRUTH_PLATE}")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at: {image_path}")
        print("Please save the plate image to that path and retry.")
        sys.exit(1)

    # Load engine with real plate-specific detector weights (MIT license).
    # Source: Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
    # OCR: EasyOCR (stand-in for PaddleOCR, blocked by Windows oneDNN crash in v3.7.0)
    print("\n[1/5] Initialising ANPREngine...")
    engine = ANPREngine(detector_path="models/license_plate_detector.pt", ocr_model_path=None)
    print(f"      is_using_real_detector : {engine.is_using_real_detector}")
    print(f"      is_using_real_ocr      : {engine.is_using_real_ocr}")

    if not engine.is_using_real_ocr:
        print("\n  WARNING: PaddleOCR not loaded — OCR will return empty string.")
        print("  Install paddleocr in this venv to get real text recognition.")

    # Load image
    print(f"\n[2/5] Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: OpenCV could not read image at {image_path}")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"      Loaded {w}x{h} frame.")

    # Use full frame as vehicle bbox (whole image is the car)
    vehicle_bbox = [0.0, 0.0, 1.0, 1.0]
    camera_id = "cam_chokepoint_real_test"
    timestamp = "2026-09-02T11:24:00.000Z"

    print("\n[3/5] Running full ANPR pipeline (process())...")
    t0 = time.perf_counter()
    output = engine.process(
        frame=frame,
        vehicle_bbox=vehicle_bbox,
        camera_id=camera_id,
        timestamp=timestamp,
        calibration=None  # No calibration — perspective_corrected will be False
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"      Pipeline completed in {elapsed_ms:.1f}ms")

    print("\n[4/5] Raw output JSON:")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n[5/5] Accuracy assessment vs ground truth:")
    plate_result = output.get("plate_number") or ""
    raw_text     = output.get("raw_text") or ""
    conf         = output.get("confidence", 0.0)
    det_conf     = output.get("metadata", {}).get("detector_confidence", 0.0)
    ocr_conf     = output.get("metadata", {}).get("ocr_confidence", 0.0)
    fmt_valid    = output.get("metadata", {}).get("format_validated", False)
    persp_corr   = output.get("metadata", {}).get("perspective_corrected", False)
    proc_ms      = output.get("metadata", {}).get("processing_time_ms", 0.0)

    print(f"  Ground truth plate : {GROUND_TRUTH_PLATE}")
    print(f"  Detected plate     : {plate_result!r}")
    print(f"  Raw OCR text       : {raw_text!r}")
    print(f"  Confidence         : {conf:.4f}")
    print(f"  Detector conf      : {det_conf:.4f}")
    print(f"  OCR conf           : {ocr_conf:.4f}")
    print(f"  Format validated   : {fmt_valid}")
    print(f"  Perspective corr.  : {persp_corr}")
    print(f"  Processing time    : {proc_ms:.2f}ms")

    exact_match = plate_result.replace(" ", "") == GROUND_TRUTH_PLATE
    print(f"\n  Exact match: {'YES ✓' if exact_match else 'NO ✗'}")

    if not exact_match and raw_text:
        # Character-level overlap as rough accuracy
        matched = sum(1 for a, b in zip(plate_result, GROUND_TRUTH_PLATE) if a == b)
        char_acc = matched / len(GROUND_TRUTH_PLATE) * 100
        print(f"  Char-level accuracy: {char_acc:.0f}% ({matched}/{len(GROUND_TRUTH_PLATE)} chars)")

    if not engine.is_using_real_ocr:
        print("\n  NOTE: OCR was not active — result is expected to be empty.")
        print("  Install paddleocr in venv_anpr and re-run for real OCR output.")

    if engine.is_using_real_detector:
        print("\n  NOTE: yolov8n.pt is COCO-pretrained (80 generic classes).")
        print("  License plates are NOT in its class list.")
        print("  A plate-specific YOLO checkpoint is needed for real detection accuracy.")

    print("=" * 65)


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/mh20dv2366.jpg"
    run_inference(image_path)
