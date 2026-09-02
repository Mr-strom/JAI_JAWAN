"""
Provisional Perspective Correction & Homography Module for ANPR.

NOTE / PROVISIONAL NOTICE:
Whether the team has full homography/camera-calibration parameters per chokepoint
camera is an open, unresolved question. This module is implemented as strictly
OPTIONAL and gracefully degrading:
- If `calibration` is provided (e.g. 3x3 homography matrix or 4 source points),
  it applies cv2.warpPerspective.
- If `calibration` is None, it optionally attempts provisional contour-based
  quadrilateral rectification.
- If any error or degenerate geometry is encountered, it gracefully returns
  the original uncorrected crop with `perspective_corrected=False`.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Any, Dict


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    return rect


class PerspectiveCorrector:
    """Provisional Homography & Perspective Correction."""

    @staticmethod
    def correct_perspective(
        plate_img: np.ndarray,
        calibration: Optional[Any] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Attempts perspective correction on a license plate crop.

        Args:
            plate_img: BGR uint8 image of the detected plate crop.
            calibration: Optional camera homography matrix (3x3 ndarray) or 
                         calibration dict {'homography': np.ndarray, 'target_size': (w, h)}
                         or 4 calibration source points.

        Returns:
            Tuple[np.ndarray, bool, float]:
            - corrected_image (or original if failed)
            - perspective_corrected (bool: True if warped successfully, False otherwise)
            - angle_estimate (estimated tilt angle in degrees, or 0.0)
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img, False, 0.0

        h, w = plate_img.shape[:2]
        if h < 10 or w < 20:
            # Degenerate size, avoid processing
            return plate_img, False, 0.0

        # --- PROVISIONAL PATH 1: Calibration matrix supplied ---
        if calibration is not None:
            try:
                if isinstance(calibration, dict) and "homography" in calibration:
                    H = np.array(calibration["homography"], dtype="float32")
                    target_w, target_h = calibration.get("target_size", (max(w, 200), max(h, 60)))
                    warped = cv2.warpPerspective(plate_img, H, (target_w, target_h), flags=cv2.INTER_LINEAR)
                    return warped, True, 0.0
                elif isinstance(calibration, np.ndarray) and calibration.shape == (3, 3):
                    warped = cv2.warpPerspective(plate_img, calibration.astype("float32"), (w, h), flags=cv2.INTER_LINEAR)
                    return warped, True, 0.0
                elif isinstance(calibration, (list, np.ndarray)) and len(calibration) == 4:
                    # 4 points supplied
                    src_pts = order_quad_points(np.array(calibration, dtype="float32"))
                    dst_w = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
                    dst_h = int(max(np.linalg.norm(src_pts[1] - src_pts[2]), np.linalg.norm(src_pts[0] - src_pts[3])))
                    dst_w = max(dst_w, 120)
                    dst_h = max(dst_h, 40)
                    dst_pts = np.array([
                        [0, 0],
                        [dst_w - 1, 0],
                        [dst_w - 1, dst_h - 1],
                        [0, dst_h - 1]
                    ], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(plate_img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
                    return warped, True, 0.0
            except Exception:
                # Provisional fallback: gracefully degrade
                pass

        # --- PROVISIONAL PATH 2: Automatic edge & contour detection fallback ---
        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200)

            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)

                # Plate is quadrilateral
                if len(approx) == 4 and cv2.contourArea(approx) > (0.30 * w * h):
                    pts = approx.reshape(4, 2)
                    rect = order_quad_points(pts)

                    (tl, tr, br, bl) = rect
                    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    max_width = max(int(width_a), int(width_b))

                    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    max_height = max(int(height_a), int(height_b))

                    # Aspect ratio check (Indian plates are roughly 2.0 to 5.0 aspect ratio)
                    if max_height > 0 and (1.5 <= (max_width / max_height) <= 6.0):
                        dst = np.array([
                            [0, 0],
                            [max_width - 1, 0],
                            [max_width - 1, max_height - 1],
                            [0, max_height - 1]
                        ], dtype="float32")

                        M = cv2.getPerspectiveTransform(rect, dst)
                        warped = cv2.warpPerspective(plate_img, M, (max_width, max_height), flags=cv2.INTER_CUBIC)

                        # Estimate tilt angle from top vector
                        dx = tr[0] - tl[0]
                        dy = tr[1] - tl[1]
                        angle = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 else 0.0

                        return warped, True, angle
        except Exception:
            pass

        # Default fallback: return original crop, perspective_corrected=False
        return plate_img, False, 0.0
