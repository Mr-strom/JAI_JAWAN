"""
utils/geometry.py — Pure geometry helpers for the Trajectory Engine.
No model imports here — keeps this testable without YOLO/MQTT present.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ─── Velocity & Direction ────────────────────────────────────────────────────

def calculate_velocity_direction(
    points: List[Tuple[float, float]],
    fps: float,
    window: int = 5,
) -> Tuple[float, float]:
    """
    Given a trajectory (list of (x, y) pixel coords, newest last),
    return (velocity_px_per_sec, direction_degrees).

    Uses the last `window` points for smoothing.
    Returns (0.0, 0.0) when fewer than 2 points exist.
    """
    if len(points) < 2:
        return 0.0, 0.0

    pts = points[-window:]  # take last N points
    if len(pts) < 2:
        return 0.0, 0.0

    # Displacement vectors between consecutive points
    dx_list, dy_list = [], []
    for i in range(1, len(pts)):
        dx_list.append(pts[i][0] - pts[i - 1][0])
        dy_list.append(pts[i][1] - pts[i - 1][1])

    # Mean displacement per frame → scale by fps
    mean_dx = float(np.mean(dx_list))
    mean_dy = float(np.mean(dy_list))

    velocity = math.hypot(mean_dx, mean_dy) * fps  # pixels/second

    # Direction: 0° = right (+x), 90° = down (+y), measured clockwise
    # atan2 gives angle from +x axis; we convert to clockwise-from-north later
    direction_rad = math.atan2(mean_dy, mean_dx)
    direction_deg = math.degrees(direction_rad) % 360.0

    return velocity, direction_deg


# ─── Smoothness ──────────────────────────────────────────────────────────────

def movement_smoothness(
    points: List[Tuple[float, float]],
    max_expected_delta_px: float = 80.0,
    window: int = 5,
) -> float:
    """
    Returns a 0-1 score; 1 = perfectly smooth, 0 = maximally erratic.

    Algorithm: compute std of consecutive step distances, normalise by
    max_expected_delta_px, then subtract from 1.
    """
    if len(points) < 3:
        return 1.0  # not enough history — give benefit of doubt

    pts = points[-window:]
    deltas = [
        math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    ]
    if len(deltas) < 2:
        return 1.0

    std_delta = float(np.std(deltas))
    roughness = min(std_delta / max_expected_delta_px, 1.0)
    return 1.0 - roughness


# ─── Zone polygon hit-test ───────────────────────────────────────────────────

def is_point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Ray-casting polygon hit-test.
    `point` and `polygon` vertices can be in any consistent coordinate space
    (pixel or normalised — just keep them consistent).
    """
    if len(polygon) < 3:
        return False

    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i

    return inside


def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return the arithmetic centroid of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return float(np.mean(xs)), float(np.mean(ys))


def is_moving_toward_polygon(
    trajectory: List[Tuple[float, float]],
    polygon: List[Tuple[float, float]],
    window: int = 5,
) -> bool:
    """
    Returns True if the entity's recent motion vector points toward the
    polygon's centroid (dot product > 0).
    """
    if len(trajectory) < 2:
        return False

    pts = trajectory[-window:]
    # Mean velocity vector
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]

    cx, cy = polygon_centroid(polygon)
    # Vector from current position to centroid
    tx = cx - pts[-1][0]
    ty = cy - pts[-1][1]

    dot = dx * tx + dy * ty
    return dot > 0


# ─── Position variance ────────────────────────────────────────────────────────

def position_variance(points: List[Tuple[float, float]], window: int = 30) -> float:
    """
    Mean of per-axis variance over the last `window` points.
    Low variance = entity is sitting still (not loitering in a wide area).
    Moderate-high variance = entity is milling around a spot (loitering).
    """
    if len(points) < 2:
        return 0.0

    pts = points[-window:]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.var(xs) + np.var(ys))
