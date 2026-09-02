"""
Unit Tests for SIH26187 Model B ANPR Engine.

Schema authority: Real Model A schema_v1 (provided 2026-09-02).
All negative-test events now carry the complete real-schema fields.
Event D is the first true happy-path (vehicle + chokepoint).

Flagged fields (project-owner decision required, not decided here):
- `hash`     : Model A emits SHA-256 for evidence tamper-detection.
               ANPR output currently does NOT forward or re-compute this.
- `severity` : Model A schema includes 'provisional'/'confirmed'/'critical'.
               ANPR output currently does NOT include a severity field.
               Two-stage severity tagging is an open project question.
"""

import unittest
import numpy as np
from anpr.indian_plate_validator import IndianPlateValidator
from anpr.perspective_correction import PerspectiveCorrector
from anpr.confidence_calculator import ConfidenceCalculator
from anpr.engine import ANPREngine


class TestIndianPlateValidator(unittest.TestCase):

    def test_standard_plate(self):
        res = IndianPlateValidator.validate("DL 01 AB 1234")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["normalized_text"], "DL01AB1234")
        self.assertEqual(res["state_code"], "DL")
        self.assertEqual(res["format_type"], "STANDARD")

    def test_bharat_series(self):
        res = IndianPlateValidator.validate("22 BH 1234 AA")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["normalized_text"], "22BH1234AA")
        self.assertEqual(res["format_type"], "BHARAT_SERIES")

    def test_defense_plate(self):
        res = IndianPlateValidator.validate("↑18D123456A")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["format_type"], "DEFENSE")

    def test_devanagari_transliteration(self):
        # Mixed Devanagari digits: MH 12 DE १४३३
        res = IndianPlateValidator.validate("MH 12 DE १४३३")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["normalized_text"], "MH12DE1433")
        self.assertTrue(res["has_devanagari"])

    def test_ocr_confusion_fix(self):
        # 'O' instead of '0', 'I' instead of '1', 'Z' instead of '2'
        res = IndianPlateValidator.validate("DLOIABIZ34")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["normalized_text"], "DL01AB1234")

    def test_invalid_plate(self):
        res = IndianPlateValidator.validate("XYZ INVALID 999999")
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["format_type"], "UNKNOWN")


class TestPerspectiveCorrector(unittest.TestCase):

    def test_graceful_degrade_on_none_calibration(self):
        dummy_img = np.zeros((50, 150, 3), dtype=np.uint8)
        corrected, is_corrected, angle = PerspectiveCorrector.correct_perspective(dummy_img, calibration=None)
        self.assertFalse(is_corrected)
        self.assertEqual(corrected.shape, dummy_img.shape)

    def test_calibration_matrix_warping(self):
        dummy_img = np.zeros((50, 150, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float32)
        corrected, is_corrected, angle = PerspectiveCorrector.correct_perspective(dummy_img, calibration=H)
        self.assertTrue(is_corrected)


class TestConfidenceCalculator(unittest.TestCase):

    def test_clean_straight_plate_high_confidence(self):
        fmt_res = {
            "normalized_text": "DL01AB1234",
            "is_valid": True,
            "validation_score": 1.0,
            "has_devanagari": False
        }
        conf, req_human, tag = ConfidenceCalculator.calculate(
            detector_conf=0.96,
            ocr_conf=0.95,
            perspective_corrected=False,
            tilt_angle=2.0,
            format_validation=fmt_res,
            raw_text_length=10
        )
        self.assertGreaterEqual(conf, 0.95)
        self.assertFalse(req_human)
        self.assertEqual(tag, "HIGH_CONFIDENCE")

    def test_angled_devanagari_plate_verification_flag(self):
        fmt_res = {
            "normalized_text": "MH12DE1433",
            "is_valid": True,
            "validation_score": 1.0,
            "has_devanagari": True
        }
        conf, req_human, tag = ConfidenceCalculator.calculate(
            detector_conf=0.85,
            ocr_conf=0.78,
            perspective_corrected=False,
            tilt_angle=30.0,
            format_validation=fmt_res,
            raw_text_length=10
        )
        self.assertTrue(req_human)
        self.assertEqual(tag, "REQUIRES_VERIFICATION")

    def test_ood_handwritten_plate(self):
        fmt_res = {
            "normalized_text": "RANDOM",
            "is_valid": False,
            "validation_score": 0.1,
            "has_devanagari": False
        }
        conf, req_human, tag = ConfidenceCalculator.calculate(
            detector_conf=0.60,
            ocr_conf=0.30,
            perspective_corrected=False,
            tilt_angle=10.0,
            format_validation=fmt_res,
            raw_text_length=6
        )
        self.assertTrue(req_human)
        self.assertEqual(tag, "REVIEW_REQUIRED_OOD")


class TestANPREngine(unittest.TestCase):

    def setUp(self):
        self.engine = ANPREngine(detector_path=None, ocr_model_path=None)

    def test_scope_rejection_for_non_chokepoints(self):
        """
        Verifies that all three real Model A negative-test events are discarded.
        Confirms filter is on `zone` (specific area) NOT `zone_tag` (broad spatial bucket).

        zone_tag='close_range' on Event A is NOT sufficient to reject it —
        the filter must check zone='fence' != {'chokepoint','icp'}.
        """
        # -----------------------------------------------------------------------
        # Event A: Real schema — Fence climbing, zone='fence', entity_type='human'
        # Expected: discarded because zone='fence' not in ALLOWED_ZONES
        # -----------------------------------------------------------------------
        event_a = {
            "event_id": "a8f3b612-4c2e-49b8-93d1-e5d871a2c001",
            "engine_source": "model_a",
            "event_type": "trigger",
            "severity": "critical",
            "timestamp": "2026-09-02T11:20:15.340Z",
            "camera_id": "cam_perimeter_04",
            "zone_tag": "close_range",   # close_range — would wrongly pass if filtered on zone_tag alone
            "zone": "fence",             # ← correct filter field; must reject
            "entity_type": "human",
            "entity_id": "global_fusion_8f90c1",
            "confidence": 0.89,
            "bbox": [0.35, 0.42, 0.58, 0.91],
            "evidence_ref": "./evidence/cam_perimeter_04_f1042_a8f3b612.jpg",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 18.4,
                "frame_number": 1042,
                "trigger_type": "climbing",
                "confirmation_frames": 5,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }
        res_a = self.engine.process_model_a_event(event_a)
        self.assertIsNone(res_a,
            "Event A (fence climbing / zone='fence') must be discarded by zone filter")

        # -----------------------------------------------------------------------
        # Event B: Real schema — Animal, zone='open_border'
        # Expected: discarded because zone='open_border' not in ALLOWED_ZONES
        # -----------------------------------------------------------------------
        event_b = {
            "event_id": "5c12d48a-112e-4f56-8a02-b349071c8902",
            "engine_source": "model_a",
            "event_type": "animal_detected",
            "severity": "info",
            "timestamp": "2026-09-02T11:21:04.112Z",
            "camera_id": "cam_perimeter_02",
            "zone_tag": "long_range",
            "zone": "open_border",
            "entity_type": "animal",
            "entity_id": "trk_deer_01",
            "confidence": 0.76,
            "bbox": [0.65, 0.12, 0.78, 0.28],
            "evidence_ref": "./evidence/cam_perimeter_02_f1105_5c12d48a.jpg",
            "hash": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 14.1,
                "frame_number": 1105,
                "trigger_type": None,
                "confirmation_frames": 0,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }
        res_b = self.engine.process_model_a_event(event_b)
        self.assertIsNone(res_b,
            "Event B (animal / zone='open_border') must be discarded by zone filter")

        # -----------------------------------------------------------------------
        # Event C: Real schema — zone='chokepoint' BUT entity_type='animal_cart'
        # Expected: passes zone filter, discarded at entity_type filter
        # -----------------------------------------------------------------------
        event_c = {
            "event_id": "9b3c4d1e-8e7a-4290-b1cf-098712345678",
            "engine_source": "model_a",
            "event_type": "animal_detected",
            "severity": "info",
            "timestamp": "2026-09-02T11:22:45.000Z",
            "camera_id": "cam_chokepoint_01",
            "zone_tag": "close_range",
            "zone": "chokepoint",         # ← passes zone filter
            "entity_type": "animal_cart", # ← but fails entity_type filter
            "entity_id": "trk_horse_01",
            "confidence": 0.81,
            "bbox": [0.10, 0.70, 0.45, 0.90],
            "evidence_ref": "./evidence/cam_chokepoint_01_f1240_9b3c4d1e.jpg",
            "hash": "b2f5c7198e3b1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b112",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 22.3,
                "frame_number": 1240,
                "trigger_type": None,
                "confirmation_frames": 0,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }
        res_c = self.engine.process_model_a_event(event_c)
        self.assertIsNone(res_c,
            "Event C (animal_cart on chokepoint) must be discarded by entity_type filter")

    def test_chokepoint_vehicle_contract_compliance(self):
        """
        Happy-path / Event D: First event with zone='chokepoint' AND entity_type='vehicle'.
        Uses the complete real Model A schema_v1 for all fields.
        Verifies full output contract compliance.
        """
        # Event D — constructed to match real schema_v1, vehicle at chokepoint
        event_d = {
            "event_id": "fe1023a1-1234-4567-89ab-cdef01234567",
            "engine_source": "model_a",
            "event_type": "motion",
            "severity": "info",
            "timestamp": "2026-09-02T11:25:00.000Z",
            "camera_id": "cam_chokepoint_02",
            "zone_tag": "close_range",
            "zone": "chokepoint",
            "entity_type": "vehicle",
            "entity_id": "trk_car_42",
            "confidence": 0.92,
            "bbox": [0.20, 0.30, 0.70, 0.80],
            "evidence_ref": "./evidence/cam_chokepoint_02_f1500_fe1023a1.jpg",
            "hash": "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 20.1,
                "frame_number": 1500,
                "trigger_type": None,
                "confirmation_frames": 0,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }

        # Inject a mock frame since evidence_ref doesn't exist on disk in test env
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = self.engine.process_model_a_event(event_d, frame_override=mock_frame)

        # Must produce output (not None)
        self.assertIsNotNone(output, "Event D (vehicle/chokepoint) must produce an ANPR output")

        # engine_source must be 'anpr' (overrides upstream 'model_a')
        self.assertEqual(output["engine_source"], "anpr")

        # MQTT topic must follow contract pattern
        self.assertEqual(output["topic"], "sih26187/camera/cam_chokepoint_02/model_b/anpr")

        # entity_id must be forwarded from upstream event
        self.assertEqual(output["entity_id"], "trk_car_42")

        # Confidence must be a genuine float in [0.0, 1.0]
        self.assertIsInstance(output["confidence"], float)
        self.assertGreaterEqual(output["confidence"], 0.0)
        self.assertLessEqual(output["confidence"], 1.0)

        # Metadata must contain all 4 required contract fields
        self.assertIn("metadata", output)
        meta = output["metadata"]
        self.assertIn("plate_detector",       meta, "metadata.plate_detector required")
        self.assertIn("ocr_model",            meta, "metadata.ocr_model required")
        self.assertIn("perspective_corrected", meta, "metadata.perspective_corrected required")
        self.assertIn("format_validated",      meta, "metadata.format_validated required")

        # perspective_corrected must be boolean
        self.assertIsInstance(meta["perspective_corrected"], bool)
        # format_validated must be boolean
        self.assertIsInstance(meta["format_validated"], bool)

        # Output event must carry its own new UUID (not the upstream event_id)
        self.assertNotEqual(output.get("event_id"), event_d["event_id"],
            "ANPR output event_id must be a fresh UUID, not the upstream event_id")


class TestSchemaFieldFlags(unittest.TestCase):
    """
    Flags two fields from Model A's schema that ANPR output currently does NOT include.
    These are NOT asserted as required — that is a project-owner decision, not an
    engine decision. This test class exists solely as a living audit trail.
    """

    def test_flag_hash_field_not_forwarded(self):
        """
        Model A includes `hash` (SHA-256 of evidence frame) for tamper-detection.
        ANPR output currently does NOT forward or re-compute `hash`.

        PROJECT OWNER QUESTION: Should ANPR output include `hash` for consistency
        with Model A's schema and the project's evidence integrity chain?
        This is a question for the project owner, not something the engine decides.
        """
        engine = ANPREngine(detector_path=None, ocr_model_path=None)
        event_d = {
            "event_id": "aaaabbbb-cccc-dddd-eeee-ffffffffffff",
            "engine_source": "model_a",
            "event_type": "motion",
            "severity": "info",
            "timestamp": "2026-09-02T12:00:00.000Z",
            "camera_id": "cam_icp_01",
            "zone_tag": "close_range",
            "zone": "icp",
            "entity_type": "vehicle",
            "entity_id": "trk_bus_99",
            "confidence": 0.90,
            "bbox": [0.10, 0.20, 0.80, 0.90],
            "evidence_ref": "./evidence/cam_icp_01_f9000_aaaa.jpg",
            "hash": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 19.0,
                "frame_number": 9000,
                "trigger_type": None,
                "confirmation_frames": 0,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = engine.process_model_a_event(event_d, frame_override=mock_frame)
        self.assertIsNotNone(output)
        # Audit: `hash` is intentionally absent from ANPR output — flag for project owner
        self.assertNotIn("hash", output,
            "[AUDIT] `hash` not in ANPR output. Project owner to decide if it should be.")

    def test_flag_severity_field_not_forwarded(self):
        """
        Model A includes `severity` (enum: info/provisional/confirmed/critical).
        ANPR output currently does NOT include a `severity` field.

        PROJECT OWNER QUESTION: Should ANPR output include `severity`? The 'provisional'
        value may already address the open two-stage severity tagging question.
        This is a question for the project owner, not something the engine decides.
        """
        engine = ANPREngine(detector_path=None, ocr_model_path=None)
        event_d = {
            "event_id": "11112222-3333-4444-5555-666677778888",
            "engine_source": "model_a",
            "event_type": "motion",
            "severity": "confirmed",   # ← real schema field, not currently forwarded
            "timestamp": "2026-09-02T12:01:00.000Z",
            "camera_id": "cam_chokepoint_03",
            "zone_tag": "close_range",
            "zone": "chokepoint",
            "entity_type": "vehicle",
            "entity_id": "trk_suv_07",
            "confidence": 0.91,
            "bbox": [0.15, 0.25, 0.75, 0.85],
            "evidence_ref": "./evidence/cam_chokepoint_03_f2000_1111.jpg",
            "hash": "pending",
            "metadata": {
                "model_version": "1.0.0",
                "processing_time_ms": 21.0,
                "frame_number": 2000,
                "trigger_type": None,
                "confirmation_frames": 0,
                "spoofing_flags": [],
                "fallback_active": False
            }
        }
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = engine.process_model_a_event(event_d, frame_override=mock_frame)
        self.assertIsNotNone(output)
        # Audit: `severity` is intentionally absent from ANPR output — flag for project owner
        self.assertNotIn("severity", output,
            "[AUDIT] `severity` not in ANPR output. Project owner to decide if it should be.")


if __name__ == "__main__":
    unittest.main()
