"""
camera_stream.py — Per-camera RTSP stream manager.

Each camera gets exactly one CameraStream instance (stored in a dict by cam_id
in the MQTT bridge). The stream is lazily opened on the first process() call.
Frames are grabbed in a background thread so the main engine thread never
blocks on VideoCapture.read().
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import FRAME_CAPTURE_TIMEOUT_S

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe wrapper around cv2.VideoCapture.

    Usage:
        stream = CameraStream("rtsp://...")
        frame = stream.get_frame()   # returns latest numpy frame or None
        stream.release()
    """

    def __init__(self, source: str, cam_id: str) -> None:
        self.source = source
        self.cam_id = cam_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._fps: float = 25.0          # fallback until we read from cap
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the capture and start the background grab thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error("[CameraStream] Cannot open source for cam %s: %s", self.cam_id, self.source)
            return False

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self._fps = fps

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name=f"cam-{self.cam_id}")
        self._thread.start()
        logger.info("[CameraStream] Started stream for cam %s @ %.1f fps", self.cam_id, self._fps)
        return True

    def get_frame(self, timeout: float = FRAME_CAPTURE_TIMEOUT_S) -> Optional[np.ndarray]:
        """Return the latest grabbed frame, or None if none available within timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.005)
        logger.warning("[CameraStream] Timed out waiting for frame from cam %s", self.cam_id)
        return None

    @property
    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        """Stop the background thread and release the capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info("[CameraStream] Released stream for cam %s", self.cam_id)

    def is_alive(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ── Internal ────────────────────────────────────────────────────────────

    def _grab_loop(self) -> None:
        """Continuously grabs the most recent frame (drops stale buffered frames)."""
        while self._running:
            if not self._cap or not self._cap.isOpened():
                logger.error("[CameraStream] Capture lost for cam %s, stopping thread.", self.cam_id)
                self._running = False
                break

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                # Brief pause before retry to avoid busy-spin on a dead stream
                time.sleep(0.033)
