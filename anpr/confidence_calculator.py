"""
Dynamic Genuine Confidence Calculator for ANPR Engine.

Calculates composite confidence scores based on:
1. Plate detection bounding box clarity / detector confidence ($C_{det}$)
2. OCR character/word recognition certainty ($C_{ocr}$)
3. Plate geometry / perspective tilt penalty ($P_{geom}$)
4. Indian registration syntax & state-code format validation match ($P_{fmt}$)
5. Multi-script / Devanagari uncertainty adjustment ($P_{script}$)
"""

import math
from typing import Dict, Any, Tuple


class ConfidenceCalculator:
    """Computes realistic, dynamic confidence metrics for license plate reads."""

    @staticmethod
    def calculate(
        detector_conf: float,
        ocr_conf: float,
        perspective_corrected: bool,
        tilt_angle: float,
        format_validation: Dict[str, Any],
        raw_text_length: int
    ) -> Tuple[float, bool, str]:
        """
        Calculates composite confidence and flags if human verification is required.

        Args:
            detector_conf: YOLO plate detector confidence [0.0 - 1.0].
            ocr_conf: PaddleOCR recognition confidence [0.0 - 1.0].
            perspective_corrected: Whether perspective correction was applied.
            tilt_angle: Absolute tilt angle in degrees.
            format_validation: Output dict from IndianPlateValidator.
            raw_text_length: Length of extracted characters.

        Returns:
            Tuple[float, bool, str]:
            - composite_confidence: float in [0.0, 1.0], rounded to 4 decimals.
            - human_verification_required: bool.
            - status_tag: str ("HIGH_CONFIDENCE", "REQUIRES_VERIFICATION", "REVIEW_REQUIRED_OOD").
        """
        # Base confidence from detector and OCR models
        c_det = max(0.0, min(1.0, float(detector_conf)))
        c_ocr = max(0.0, min(1.0, float(ocr_conf)))

        # Format validation factor [0.0 - 1.0]
        fmt_score = float(format_validation.get("validation_score", 0.0))
        is_valid_format = bool(format_validation.get("is_valid", False))
        has_devanagari = bool(format_validation.get("has_devanagari", False))

        # Angle penalty: plates tilted > 15 degrees suffer degradation
        abs_angle = abs(float(tilt_angle))
        if abs_angle > 45.0:
            angle_penalty = 0.35
        elif abs_angle > 25.0:
            angle_penalty = 0.18
        elif abs_angle > 10.0:
            angle_penalty = 0.08
        else:
            angle_penalty = 0.0

        # Perspective bonus/penalty
        if not perspective_corrected and abs_angle > 15.0:
            perspective_penalty = 0.12
        else:
            perspective_penalty = 0.0

        # Text length penalty (Indian plates are usually 8-10 characters)
        if raw_text_length < 5:
            len_penalty = 0.40
        elif raw_text_length < 8:
            len_penalty = 0.15
        elif raw_text_length > 12:
            len_penalty = 0.20
        else:
            len_penalty = 0.0

        # Multi-script adjustment: Devanagari or mixed script has higher variance
        script_factor = 0.94 if has_devanagari else 1.0

        # Weighted calculation
        # Detector: 25%, OCR: 45%, Format Match: 30%
        base_score = (0.25 * c_det) + (0.45 * c_ocr) + (0.30 * fmt_score)

        # Apply cumulative penalties
        penalties = angle_penalty + perspective_penalty + len_penalty
        adjusted_score = max(0.05, (base_score - penalties) * script_factor)

        # Boost perfectly matched clean plates
        if is_valid_format and c_ocr >= 0.92 and c_det >= 0.90 and abs_angle <= 10.0:
            composite_confidence = min(0.99, max(0.95, adjusted_score))
        else:
            composite_confidence = min(0.98, adjusted_score)

        composite_confidence = round(composite_confidence, 4)

        # Determine review flags
        # 1. Clean, straight-angle plate -> high confidence (>95%), no flags.
        if composite_confidence >= 0.92 and is_valid_format and not has_devanagari:
            return composite_confidence, False, "HIGH_CONFIDENCE"

        # 2. Angled plate (~30°) with Devanagari script or medium confidence -> flag for human verification
        if 0.50 <= composite_confidence < 0.92 or (has_devanagari and composite_confidence < 0.95) or abs_angle >= 20.0:
            return composite_confidence, True, "REQUIRES_VERIFICATION"

        # 3. Handwritten / temporary plate / low OCR (out of distribution) -> save crop for human review
        return composite_confidence, True, "REVIEW_REQUIRED_OOD"
