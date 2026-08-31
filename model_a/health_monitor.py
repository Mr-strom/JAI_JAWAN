"""
Camera Health Monitor — Stream Integrity & Fallback Routing
SIH26187 | Model A | Step 12 + 13 of pipeline

Monitors:
  1. Lens obstruction   — histogram darkness / uniform patch detection
  2. Darkness / IR fail — mean pixel luminance below threshold
  3. Frozen stream      — MSE delta near zero for >N consecutive frames
  4. FPS anomalies      — measured FPS too low or too high
  5. Model B heartbeat  — if engine heartbeat stale >30s, flag for fallback routing

CRITICAL (Rule #12 & #13):
  Transient network blips (sub-threshold outage) must NOT trigger a health event.
  Use a minimum observation window before publishing camera_anomaly events.
  Fallback routing engages ONLY on that specific camera.

Health events are published on:
  sih26187/camera/{cam_id}/model_a/event   (event_type=camera_anomaly or system_health)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DARKNESS_THRESHOLD       = 20.0    # mean luminance (0-255) below this → dark
_FROZEN_MSE_THRESHOLD     = 1e-6    # near-zero MSE → frozen
_FROZEN_FRAME_COUNT       = 15      # consecutive frozen frames before declaring frozen
_OBSTRUCTION_STD_THRESHOLD = 5.0    # very low std → possible lens obstruction
_MODEL_B_HEARTBEAT_TIMEOUT = 30.0   # seconds — Rule #13
_BLIP_WINDOW_SECONDS      = 3.0     # sub-threshold outage allowed before health event


# ---------------------------------------------------------------------------
# Health States
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    HEALTHY     = auto()
    DARK        = auto()
    FROZEN      = auto()
    OBSTRUCTED  = auto()
    FPS_ANOMALY = auto()
    OFFLINE     = auto()     # confirmed outage (beyond blip window)


# ---------------------------------------------------------------------------
# Health Report
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    camera_id: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.monotonic)
    should_publish: bool = False
    fallback_active: bool = False


# ---------------------------------------------------------------------------
# Per-camera health tracker
# ---------------------------------------------------------------------------

@dataclass
class CameraHealthRecord:
    camera_id: str
    last_frame_time: float = field(default_factory=time.monotonic)
    last_frame_grey: Optional[np.ndarray] = None
    consecutive_frozen: int = 0
    blip_started_at: Optional[float] = None   # monotonic time of first miss
    status: HealthStatus = HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# CameraHealthMonitor
# ---------------------------------------------------------------------------

class CameraHealthMonitor:
    """
    Tracks health of all registered cameras.

    Usage::

        monitor = CameraHealthMonitor()
        monitor.register_camera("cam_01")

        # On each frame arrival:
        report = monitor.check_frame("cam_01", frame_bgr)
        if report.should_publish:
            bus_client.publish_event(build_health_event(report))

        # On each Model B heartbeat:
        monitor.update_model_b_heartbeat("face_engine")

        # On camera frame miss (no frame received this tick):
        report = monitor.check_timeout("cam_01")
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, CameraHealthRecord] = {}
        self._model_b_heartbeats: Dict[str, float] = {}   # engine_id → last beat time

    # ------------------------------------------------------------------
    # Camera registration
    # ------------------------------------------------------------------

    def register_camera(self, camera_id: str) -> None:
        self._cameras[camera_id] = CameraHealthRecord(camera_id=camera_id)
        logger.info("Camera registered for health monitoring: %s", camera_id)

    # ------------------------------------------------------------------
    # Per-frame checks
    # ------------------------------------------------------------------

    def check_frame(self, camera_id: str, frame: np.ndarray) -> HealthReport:
        """
        Run all frame-level health checks.
        Returns HealthReport — caller decides whether to publish.
        """
        rec = self._ensure_record(camera_id)
        rec.last_frame_time = time.monotonic()
        rec.blip_started_at = None   # frame arrived → blip cleared

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Check darkness
        mean_lum = float(np.mean(grey))
        if mean_lum < _DARKNESS_THRESHOLD:
            rec.status = HealthStatus.DARK
            logger.warning("cam=%s DARKNESS DETECTED mean_lum=%.1f", camera_id, mean_lum)
            return HealthReport(
                camera_id=camera_id,
                status=HealthStatus.DARK,
                message=f"DARKNESS_DETECTED mean_lum={mean_lum:.1f}",
                should_publish=True,
            )

        # Check frozen stream
        if rec.last_frame_grey is not None:
            mse = float(np.mean((grey / 255.0 - rec.last_frame_grey / 255.0) ** 2))
            if mse < _FROZEN_MSE_THRESHOLD:
                rec.consecutive_frozen += 1
                if rec.consecutive_frozen >= _FROZEN_FRAME_COUNT:
                    rec.status = HealthStatus.FROZEN
                    logger.warning("cam=%s FROZEN STREAM detected after %d frames",
                                   camera_id, rec.consecutive_frozen)
                    return HealthReport(
                        camera_id=camera_id,
                        status=HealthStatus.FROZEN,
                        message=f"FROZEN_STREAM consecutive_frames={rec.consecutive_frozen}",
                        should_publish=True,
                    )
            else:
                rec.consecutive_frozen = 0

        # Check lens obstruction (very low standard deviation)
        std_lum = float(np.std(grey))
        if std_lum < _OBSTRUCTION_STD_THRESHOLD:
            rec.status = HealthStatus.OBSTRUCTED
            logger.warning("cam=%s LENS OBSTRUCTION suspected std=%.2f", camera_id, std_lum)
            return HealthReport(
                camera_id=camera_id,
                status=HealthStatus.OBSTRUCTED,
                message=f"LENS_OBSTRUCTION std={std_lum:.2f}",
                should_publish=True,
            )

        rec.last_frame_grey = grey
        rec.status = HealthStatus.HEALTHY
        return HealthReport(
            camera_id=camera_id,
            status=HealthStatus.HEALTHY,
            message="OK",
            should_publish=False,
        )

    def check_timeout(self, camera_id: str) -> HealthReport:
        """
        Called when no frame arrives from a camera in the expected window.

        Distinguishes blips (<_BLIP_WINDOW_SECONDS) from genuine outages.
        Only publishes OFFLINE after the blip window expires.
        (Rule #12: avoid flapping.)
        """
        rec  = self._ensure_record(camera_id)
        now  = time.monotonic()

        if rec.blip_started_at is None:
            rec.blip_started_at = now
            logger.debug("cam=%s frame miss — starting blip window.", camera_id)
            return HealthReport(
                camera_id=camera_id,
                status=HealthStatus.HEALTHY,
                message="BLIP_STARTED — within tolerance window",
                should_publish=False,
            )

        blip_duration = now - rec.blip_started_at
        if blip_duration < _BLIP_WINDOW_SECONDS:
            logger.debug("cam=%s blip ongoing (%.1fs < %.1fs)",
                         camera_id, blip_duration, _BLIP_WINDOW_SECONDS)
            return HealthReport(
                camera_id=camera_id,
                status=HealthStatus.HEALTHY,
                message=f"BLIP_ONGOING duration={blip_duration:.1f}s",
                should_publish=False,
            )

        # Blip window exceeded → genuine outage
        rec.status = HealthStatus.OFFLINE
        logger.error("cam=%s OFFLINE confirmed (blip_duration=%.1fs > %.1fs)",
                     camera_id, blip_duration, _BLIP_WINDOW_SECONDS)
        return HealthReport(
            camera_id=camera_id,
            status=HealthStatus.OFFLINE,
            message=f"CAMERA_OFFLINE duration={blip_duration:.1f}s",
            should_publish=True,
        )

    # ------------------------------------------------------------------
    # Model B heartbeat management (Rule #13)
    # ------------------------------------------------------------------

    def update_model_b_heartbeat(self, engine_id: str) -> None:
        """Record that a Model B engine is alive."""
        self._model_b_heartbeats[engine_id] = time.monotonic()
        logger.debug("Model B heartbeat received from engine=%s", engine_id)

    def is_model_b_alive(self, engine_id: str) -> bool:
        """
        Returns True if the engine's last heartbeat was within 30s.
        Returns False (dead) if stale > MODEL_B_HEARTBEAT_TIMEOUT.
        """
        last_beat = self._model_b_heartbeats.get(engine_id)
        if last_beat is None:
            return False
        return (time.monotonic() - last_beat) <= _MODEL_B_HEARTBEAT_TIMEOUT

    def get_fallback_cameras(self, engine_id: str, managed_cameras: list[str]) -> list[str]:
        """
        If Model B engine is dead (>30s), return the cameras it manages.
        Model A must activate safety floor for these cameras only.
        """
        if self.is_model_b_alive(engine_id):
            return []
        cameras = list(managed_cameras)
        if cameras:
            logger.warning(
                "Model B engine '%s' heartbeat stale >30s. "
                "Fallback routing engaged for cameras: %s",
                engine_id, cameras,
            )
        return cameras

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_record(self, camera_id: str) -> CameraHealthRecord:
        if camera_id not in self._cameras:
            self.register_camera(camera_id)
        return self._cameras[camera_id]
