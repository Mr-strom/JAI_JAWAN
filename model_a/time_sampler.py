"""
Time Sampler — reduces raw 25 FPS to effective 1-5 FPS
SIH26187 | Model A | Step 3 of pipeline

Algorithm:
  MSE frame differencing with threshold (default 0.001).
  Frames whose MSE delta from the last accepted frame is below threshold
  are classified as REDUNDANT and skipped.

  Additionally implements Most-Differentiated-Frame (MDF) selection
  within a 1-second window (Step 4): keeps the frame with maximum
  pixel variance within each window.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TimeSampler:
    """
    Filters redundant frames using MSE threshold comparison.

    Usage::

        sampler = TimeSampler(mse_threshold=0.001, fps_target=5)
        for raw_frame in stream:
            kept, frame = sampler.accept(raw_frame)
            if kept:
                pipeline.process(frame)
    """

    def __init__(
        self,
        mse_threshold: float = 0.001,
        fps_target: int = 5,
        window_seconds: float = 1.0,
    ) -> None:
        if mse_threshold <= 0:
            raise ValueError("mse_threshold must be positive.")
        self.mse_threshold  = mse_threshold
        self.fps_target     = fps_target
        self.window_seconds = window_seconds

        self._last_accepted: Optional[np.ndarray] = None
        self._frames_accepted: int = 0
        self._frames_skipped: int  = 0

        # MDF window buffer: list of (variance, frame)
        self._mdf_window: list[Tuple[float, np.ndarray]] = []
        self._window_start: float = time.monotonic()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def accept(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Decide whether a frame should be passed downstream.

        Returns:
            (True, frame)  — frame accepted, pass to pipeline.
            (False, None)  — frame rejected as redundant.
        """
        if self._last_accepted is None:
            # First frame is always accepted
            self._last_accepted = self._to_grey(frame)
            self._frames_accepted += 1
            self._add_to_mdf_window(frame)
            return True, frame

        grey    = self._to_grey(frame)
        mse     = self._compute_mse(self._last_accepted, grey)

        if mse < self.mse_threshold:
            self._frames_skipped += 1
            logger.debug("Frame SKIPPED (MSE=%.6f < threshold=%.6f)", mse, self.mse_threshold)
            return False, None

        self._last_accepted = grey
        self._frames_accepted += 1
        self._add_to_mdf_window(frame)
        logger.debug("Frame ACCEPTED (MSE=%.6f)", mse)
        return True, frame

    def flush_mdf(self) -> Optional[np.ndarray]:
        """
        Return the Most-Differentiated Frame from the current window,
        then reset the window. Call this at the end of each 1-second window.

        Returns None if no frames have been buffered yet.
        """
        if not self._mdf_window:
            return None
        best_frame = max(self._mdf_window, key=lambda t: t[0])[1]
        self._mdf_window.clear()
        self._window_start = time.monotonic()
        logger.debug("MDF window flushed — best variance frame selected.")
        return best_frame

    def should_flush_window(self) -> bool:
        """True if the current 1-second MDF window has expired."""
        return (time.monotonic() - self._window_start) >= self.window_seconds

    def stats(self) -> dict:
        """Return sampling statistics."""
        total = self._frames_accepted + self._frames_skipped
        ratio = self._frames_accepted / total if total else 0
        return {
            "frames_accepted": self._frames_accepted,
            "frames_skipped":  self._frames_skipped,
            "acceptance_ratio": round(ratio, 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_grey(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to greyscale float32 normalised [0,1]."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return grey.astype(np.float32) / 255.0

    def _compute_mse(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Mean Squared Error between two greyscale frames."""
        diff = a - b
        return float(np.mean(diff * diff))

    def _add_to_mdf_window(self, frame: np.ndarray) -> None:
        """Buffer frame with its pixel variance for MDF selection."""
        variance = float(np.var(frame))
        self._mdf_window.append((variance, frame))
