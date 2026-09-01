"""
Homography Perspective Corrector — Model A
SIH26187 | Model A | Phase 2 Extension (approved 2026-09-01)

PURPOSE:
  CCTV cameras mounted at oblique angles (chokepoints, ICPs) produce
  perspective distortion: objects close to the camera appear much larger
  than identical objects farther away. This breaks the assumption that
  bbox size is a reliable zone-proximity signal.

  cv2.warpPerspective applies a projective transform that maps the raw
  camera view to a rectified bird's-eye equivalent. After correction:
    - Objects at the same physical distance appear at the same scale.
    - ZoneTagger's normalised bbox-height thresholds become physically
      meaningful (size reflects actual distance, not just camera angle).

DESIGN:
  - HomographyCorrector is OPTIONAL in FramePipeline. If instantiated with
    a config file, it reads per-camera 4-point calibrations and computes
    the full 3×3 homography matrix at init time (not per-frame).
  - On correct_frame(camera_id, frame): applies warpPerspective if the
    camera has a calibration entry; returns frame unchanged if not.
  - Missing camera_id → logs a one-time warning, returns frame unmodified.
    NEVER crashes the pipeline on an uncalibrated camera.
  - Output size = input size (warpPerspective dsize = (frame.shape[1], frame.shape[0])).
    This preserves the [0,1] normalised-coordinate schema contract — YOLO
    normalises against the corrected frame's pixel dimensions which equal
    the original frame's dimensions.

CONFIG FORMAT (model_a/homography_config.json):
  {
    "cam_chokepoint_01": {
      "src_points": [[x,y], [x,y], [x,y], [x,y]],   // raw frame pixels
      "dst_points": [[x,y], [x,y], [x,y], [x,y]]    // corrected frame pixels
    }
  }

  Points are in absolute pixel coordinates for the camera's native resolution.
  src/dst must be ordered consistently (e.g. top-left, top-right, bottom-right,
  bottom-left). 4 points required per camera.

DOES NOT TOUCH:
  - schema_v1.py, bus_client.py, trigger_detector.py
  - ZoneTagger or AnimalFilter behavior
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyCorrector:
    """
    Applies per-camera perspective correction via cv2.warpPerspective.

    Instantiate once at startup with the calibration config path.
    Call correct_frame() on every raw frame in FramePipeline.process().

    Usage::

        corrector = HomographyCorrector("model_a/homography_config.json")

        # In FramePipeline.process():
        frame = corrector.correct_frame(self.camera_id, frame)
        # Then proceed to YOLO detection as normal.
    """

    def __init__(self, config_path: str) -> None:
        """
        Load calibration config and pre-compute homography matrices.

        Args:
            config_path: Path to the JSON calibration file.

        Raises:
            json.JSONDecodeError: If the config file is not valid JSON.
            ValueError: If a camera entry has invalid point counts.
        """
        self._matrices: Dict[str, np.ndarray] = {}   # camera_id → 3×3 H matrix
        self._warned_uncalibrated: Set[str] = set()  # suppress repeat warnings

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(
                "HomographyCorrector: config file not found at '%s'. "
                "No cameras will be corrected.",
                config_path,
            )
            return

        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict = json.load(f)  # raises json.JSONDecodeError on bad JSON

        for camera_id, cal in raw.items():
            src_pts = cal.get("src_points", [])
            dst_pts = cal.get("dst_points", [])

            if len(src_pts) != 4 or len(dst_pts) != 4:
                raise ValueError(
                    f"Camera '{camera_id}': expected exactly 4 src_points and 4 dst_points, "
                    f"got src={len(src_pts)} dst={len(dst_pts)}. "
                    "Refusing to apply a poorly-calibrated homography."
                )

            src_np = np.array(src_pts, dtype=np.float32)
            dst_np = np.array(dst_pts, dtype=np.float32)

            H, status = cv2.findHomography(src_np, dst_np, method=0)
            if H is None:
                raise ValueError(
                    f"Camera '{camera_id}': cv2.findHomography returned None. "
                    "Points may be collinear — check calibration."
                )

            self._matrices[camera_id] = H
            logger.info(
                "HomographyCorrector: loaded calibration for camera '%s'.", camera_id
            )

        logger.info(
            "HomographyCorrector: %d camera(s) calibrated: %s",
            len(self._matrices),
            list(self._matrices.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Apply perspective correction to a raw camera frame.

        If camera_id has no calibration entry, returns frame unchanged
        and logs a one-time warning. NEVER raises.

        Args:
            camera_id: Camera identifier matching the config key.
            frame:     BGR frame from RTSP stream (np.ndarray).

        Returns:
            Corrected frame (same dtype, same spatial dimensions as input).
        """
        H = self._matrices.get(camera_id)
        if H is None:
            self._warn_once(camera_id)
            return frame

        h, w = frame.shape[:2]
        corrected = cv2.warpPerspective(frame, H, (w, h))
        return corrected

    def correct_bbox_normalised(
        self,
        camera_id: str,
        bbox: List[float],
        frame_h: int,
        frame_w: int,
    ) -> List[float]:
        """
        Apply the homography to a normalised bbox [x1,y1,x2,y2].

        Useful for correcting coordinates-only (e.g. when passing events
        to Model B / ANPR pipeline without re-running YOLO on a warped frame).

        If camera_id has no calibration, returns bbox unchanged.

        Args:
            camera_id: Camera identifier.
            bbox:      Normalised [x1,y1,x2,y2] in [0,1].
            frame_h:   Frame height in pixels (for denormalisation).
            frame_w:   Frame width  in pixels (for denormalisation).

        Returns:
            Corrected normalised bbox [x1,y1,x2,y2], clipped to [0,1].
        """
        H = self._matrices.get(camera_id)
        if H is None:
            return bbox

        x1n, y1n, x2n, y2n = bbox

        # Denormalise corners to absolute pixels
        pts_abs = np.array([
            [[x1n * frame_w, y1n * frame_h]],
            [[x2n * frame_w, y1n * frame_h]],
            [[x2n * frame_w, y2n * frame_h]],
            [[x1n * frame_w, y2n * frame_h]],
        ], dtype=np.float32)

        # Apply homography to all 4 corners
        pts_warped = cv2.perspectiveTransform(pts_abs, H)
        pts_warped = pts_warped.reshape(-1, 2)  # shape (4, 2)

        # Compute new axis-aligned bounding box from warped corners
        wx1 = float(np.min(pts_warped[:, 0]))
        wy1 = float(np.min(pts_warped[:, 1]))
        wx2 = float(np.max(pts_warped[:, 0]))
        wy2 = float(np.max(pts_warped[:, 1]))

        # Normalise back and clip to [0,1] (schema contract)
        result = [
            max(0.0, min(1.0, wx1 / frame_w)),
            max(0.0, min(1.0, wy1 / frame_h)),
            max(0.0, min(1.0, wx2 / frame_w)),
            max(0.0, min(1.0, wy2 / frame_h)),
        ]
        return result

    def is_calibrated(self, camera_id: str) -> bool:
        """Return True if this camera has a homography calibration loaded."""
        return camera_id in self._matrices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warn_once(self, camera_id: str) -> None:
        """Log a warning for an uncalibrated camera, once per camera_id."""
        if camera_id not in self._warned_uncalibrated:
            logger.warning(
                "HomographyCorrector: no calibration for camera '%s'. "
                "Frame returned uncorrected. "
                "Add a calibration entry to homography_config.json to enable correction.",
                camera_id,
            )
            self._warned_uncalibrated.add(camera_id)
