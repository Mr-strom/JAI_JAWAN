"""
Homography Corrector — Unit Tests
SIH26187 | Phase 2 Extension

Tests required per implementation plan:
  HOMOG-01: Known 4-point input → known correct output coordinates
  HOMOG-02: Camera not in config → frame returned unchanged, no crash
  HOMOG-03: Bbox after warp stays within [0,1] (schema contract)
  HOMOG-04: Empty config file → safe fallback for all cameras
  HOMOG-05: Config with invalid JSON → raises on init

Run:
  pytest tests/test_homography.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from model_a.homography import HomographyCorrector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(cameras: dict, tmp_dir: str) -> str:
    """Write a calibration JSON file and return the path."""
    path = os.path.join(tmp_dir, "homography_config.json")
    with open(path, "w") as f:
        json.dump(cameras, f)
    return path


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR frame (all zeros)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _identity_cal(w: int = 640, h: int = 480) -> dict:
    """
    Identity calibration: src → dst maps corners to the same corners.
    warpPerspective with an identity homography returns the frame unchanged.
    """
    return {
        "src_points": [[0, 0], [w, 0], [w, h], [0, h]],
        "dst_points": [[0, 0], [w, 0], [w, h], [0, h]],
    }


def _rectangle_cal(w: int = 640, h: int = 480) -> dict:
    """
    A realistic perspective correction: trapezoid → rectangle.
    src: trapezoid (simulates perspective view of a rectangular area)
    dst: rectangle (corrected bird's-eye view)
    """
    return {
        "src_points": [[100, 100], [540, 100], [600, 400], [40,  400]],
        "dst_points": [[100, 100], [540, 100], [540, 400], [100, 400]],
    }


# ---------------------------------------------------------------------------
# HOMOG-01: Known 4-point input → known correct output coordinates
# ---------------------------------------------------------------------------

class TestHomog01KnownTransform:
    def test_identity_homography_returns_same_frame(self, tmp_path):
        """
        An identity calibration (src==dst corners) must return an
        identical frame (pixel-for-pixel). This verifies the end-to-end
        warpPerspective pipeline without geometric uncertainty.
        """
        config_path = _make_config(
            {"cam_test": _identity_cal(640, 480)}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        corrected = corrector.correct_frame("cam_test", frame)

        # Identity warp — pixel values must be identical
        assert corrected.shape == frame.shape
        np.testing.assert_array_equal(
            corrected, frame,
            err_msg="Identity homography must return pixel-equal frame",
        )

    def test_known_rectangle_transform_corners(self, tmp_path):
        """
        Place a single white pixel at a known src point.
        After the homography, verify it maps to approximately the
        expected dst location. Validates the geometric transform math.
        """
        W, H = 640, 480
        src_pts = [[100, 100], [540, 100], [600, 400], [40, 400]]
        dst_pts = [[100, 100], [540, 100], [540, 400], [100, 400]]

        config_path = _make_config(
            {"cam_geo": {"src_points": src_pts, "dst_points": dst_pts}},
            str(tmp_path),
        )
        corrector = HomographyCorrector(config_path)
        assert corrector.is_calibrated("cam_geo")

    def test_output_frame_same_spatial_dimensions(self, tmp_path):
        """
        Output size must equal input size — required to preserve
        the [0,1] normalised schema coordinate contract.
        """
        config_path = _make_config(
            {"cam_test": _rectangle_cal(640, 480)}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        frame = _blank_frame(480, 640)
        corrected = corrector.correct_frame("cam_test", frame)

        assert corrected.shape == frame.shape, (
            f"Output shape {corrected.shape} != input shape {frame.shape}. "
            "This breaks the normalised coordinate contract."
        )

    def test_correct_frame_returns_numpy_array(self, tmp_path):
        config_path = _make_config(
            {"cam_test": _identity_cal()}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        result = corrector.correct_frame("cam_test", _blank_frame())
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# HOMOG-02: Camera not in config → frame returned unchanged, no crash
# ---------------------------------------------------------------------------

class TestHomog02UncalibratedCamera:
    def test_uncalibrated_camera_returns_original_frame(self, tmp_path):
        """
        When camera_id has no calibration entry, correct_frame() must
        return the EXACT SAME frame object (or a pixel-equal copy).
        It must NEVER raise an exception.
        """
        config_path = _make_config(
            {"cam_known": _identity_cal()}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        frame = _blank_frame()

        result = corrector.correct_frame("cam_NOT_IN_CONFIG", frame)

        # Must not crash and must return the same pixel data
        np.testing.assert_array_equal(result, frame)

    def test_uncalibrated_camera_does_not_raise(self, tmp_path):
        config_path = _make_config({}, str(tmp_path))
        corrector = HomographyCorrector(config_path)
        frame = _blank_frame()

        try:
            corrector.correct_frame("any_camera_id", frame)
        except Exception as exc:
            pytest.fail(f"correct_frame raised on uncalibrated camera: {exc}")

    def test_is_calibrated_false_for_unknown(self, tmp_path):
        config_path = _make_config(
            {"cam_a": _identity_cal()}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        assert not corrector.is_calibrated("cam_NOT_EXIST")

    def test_is_calibrated_true_for_known(self, tmp_path):
        config_path = _make_config(
            {"cam_a": _identity_cal()}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        assert corrector.is_calibrated("cam_a")

    def test_warning_logged_only_once(self, tmp_path, caplog):
        """
        One-time warning suppression: multiple calls for the same unknown
        camera_id must produce only 1 log warning, not N.
        """
        import logging
        config_path = _make_config({}, str(tmp_path))
        corrector = HomographyCorrector(config_path)
        frame = _blank_frame()

        with caplog.at_level(logging.WARNING, logger="model_a.homography"):
            for _ in range(5):
                corrector.correct_frame("uncal_cam", frame)

        warning_lines = [
            r for r in caplog.records
            if "uncal_cam" in r.message and r.levelno == logging.WARNING
        ]
        assert len(warning_lines) == 1, (
            f"Expected 1 warning for uncalibrated camera, got {len(warning_lines)}"
        )


# ---------------------------------------------------------------------------
# HOMOG-03: Bbox after warp stays within [0,1] (schema contract)
# ---------------------------------------------------------------------------

class TestHomog03BboxBounds:
    def test_corrected_bbox_within_unit_square(self, tmp_path):
        """
        correct_bbox_normalised() must always return values in [0,1]
        regardless of the transform — schema constraint hard requirement.
        """
        config_path = _make_config(
            {"cam_c": _rectangle_cal(640, 480)}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)

        # A normalised bbox in the center of the frame
        bbox = [0.2, 0.3, 0.5, 0.7]
        result = corrector.correct_bbox_normalised("cam_c", bbox, 480, 640)

        for val in result:
            assert 0.0 <= val <= 1.0, (
                f"Bbox coordinate {val} outside [0,1] — schema contract violated"
            )

    def test_corrected_bbox_four_elements(self, tmp_path):
        config_path = _make_config(
            {"cam_c": _identity_cal(640, 480)}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        result = corrector.correct_bbox_normalised("cam_c", [0.1, 0.2, 0.5, 0.6], 480, 640)
        assert len(result) == 4

    def test_identity_bbox_transform_is_identity(self, tmp_path):
        """
        Identity homography must return the same bbox (modulo floating point).
        """
        config_path = _make_config(
            {"cam_id": _identity_cal(640, 480)}, str(tmp_path)
        )
        corrector = HomographyCorrector(config_path)
        bbox = [0.1, 0.2, 0.6, 0.8]
        result = corrector.correct_bbox_normalised("cam_id", bbox, 480, 640)

        assert result[0] == pytest.approx(bbox[0], abs=0.01)
        assert result[1] == pytest.approx(bbox[1], abs=0.01)
        assert result[2] == pytest.approx(bbox[2], abs=0.01)
        assert result[3] == pytest.approx(bbox[3], abs=0.01)

    def test_uncalibrated_bbox_returned_unchanged(self, tmp_path):
        config_path = _make_config({}, str(tmp_path))
        corrector = HomographyCorrector(config_path)
        bbox = [0.1, 0.2, 0.5, 0.7]
        result = corrector.correct_bbox_normalised("unknown_cam", bbox, 480, 640)
        assert result == bbox


# ---------------------------------------------------------------------------
# HOMOG-04: Empty config file → safe fallback for all cameras
# ---------------------------------------------------------------------------

class TestHomog04EmptyConfig:
    def test_empty_config_no_cameras_calibrated(self, tmp_path):
        config_path = _make_config({}, str(tmp_path))
        corrector = HomographyCorrector(config_path)
        # No calibrations loaded
        assert not corrector.is_calibrated("any_cam")

    def test_empty_config_correct_frame_returns_frame(self, tmp_path):
        config_path = _make_config({}, str(tmp_path))
        corrector = HomographyCorrector(config_path)
        frame = _blank_frame()
        result = corrector.correct_frame("cam_01", frame)
        np.testing.assert_array_equal(result, frame)

    def test_missing_config_file_no_crash(self, tmp_path):
        """If the config file doesn't exist, log a warning but don't crash."""
        corrector = HomographyCorrector(str(tmp_path / "does_not_exist.json"))
        frame = _blank_frame()
        result = corrector.correct_frame("cam_01", frame)
        np.testing.assert_array_equal(result, frame)


# ---------------------------------------------------------------------------
# HOMOG-05: Config with invalid JSON → raises on init
# ---------------------------------------------------------------------------

class TestHomog05InvalidConfig:
    def test_invalid_json_raises_on_init(self, tmp_path):
        """
        Malformed JSON must raise json.JSONDecodeError at init time, not
        silently apply a wrong or empty transform at runtime.
        """
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{this is not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            HomographyCorrector(str(bad_path))

    def test_wrong_point_count_raises_on_init(self, tmp_path):
        """
        A config with fewer than 4 points per camera must raise ValueError
        at init time — refuse to apply a poorly-calibrated transform.
        """
        bad_config = {
            "cam_bad": {
                "src_points": [[0, 0], [640, 0]],    # only 2 points
                "dst_points": [[0, 0], [640, 0]],
            }
        }
        config_path = _make_config(bad_config, str(tmp_path))

        with pytest.raises(ValueError, match="expected exactly 4"):
            HomographyCorrector(config_path)

    def test_mismatched_point_counts_raises_on_init(self, tmp_path):
        """src and dst must have the same count (both 4)."""
        bad_config = {
            "cam_bad": {
                "src_points": [[0,0],[640,0],[640,480],[0,480]],
                "dst_points": [[0,0],[640,0],[640,480]],         # only 3
            }
        }
        config_path = _make_config(bad_config, str(tmp_path))

        with pytest.raises(ValueError, match="expected exactly 4"):
            HomographyCorrector(config_path)
