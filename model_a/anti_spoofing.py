"""
Anti-Spoofing Checker — Stream Integrity Validation
SIH26187 | Model A | Step 8 of pipeline

Checks performed:
  1. Timestamp monotonicity   — no negative gaps between frames.
  2. FPS consistency          — frame rate within expected band.
  3. Frame continuity         — no large unexplained frame-number jumps.

Rule (NON-NEGOTIABLE):
  Spoofing flags are LOGGED and appended to the event's metadata.
  Events with spoofing flags are NOT suppressed.
  Spoofing alerts are published as severity=warning on the same bus.
  (Rule #8: DO NOT suppress the event.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_EXPECTED_FPS   = 25.0
_FPS_TOLERANCE          = 0.30   # 30% deviation allowed
_MAX_FRAME_GAP          = 50     # frames — beyond this is suspicious
_MIN_TIMESTAMP_GAP_S    = 0.0    # negative gaps are always suspicious


@dataclass
class SpoofingReport:
    """Result of anti-spoofing analysis for one frame."""
    flags: List[str] = field(default_factory=list)
    is_suspicious: bool = False

    def add_flag(self, flag: str) -> None:
        self.flags.append(flag)
        self.is_suspicious = True
        logger.warning("ANTI-SPOOFING flag raised: %s", flag)


class AntiSpoofingChecker:
    """
    Stateful per-camera stream integrity checker.

    Usage::

        checker = AntiSpoofingChecker(camera_id="cam_01")
        report = checker.check(
            timestamp_utc="2025-01-01T00:00:01Z",
            frame_number=101,
        )
        if report.is_suspicious:
            # Append report.flags to event metadata — DO NOT suppress event.
    """

    def __init__(
        self,
        camera_id: str,
        expected_fps: float = _DEFAULT_EXPECTED_FPS,
        fps_tolerance: float = _FPS_TOLERANCE,
        max_frame_gap: int   = _MAX_FRAME_GAP,
    ) -> None:
        self.camera_id     = camera_id
        self.expected_fps  = expected_fps
        self.fps_tolerance = fps_tolerance
        self.max_frame_gap = max_frame_gap

        self._last_timestamp_s: Optional[float] = None
        self._last_frame_number: Optional[int]  = None
        self._frame_intervals: list[float]      = []   # rolling window for FPS check

    def check(
        self,
        timestamp_utc: str,
        frame_number: int,
    ) -> SpoofingReport:
        """
        Run all anti-spoofing checks for a single frame.

        Args:
            timestamp_utc: ISO 8601 UTC string from the frame/stream.
            frame_number:  Monotonically increasing frame index from the camera.

        Returns:
            SpoofingReport with flags (may be empty if no anomalies).
        """
        report = SpoofingReport()
        ts_s   = self._parse_timestamp(timestamp_utc)

        if ts_s is None:
            report.add_flag(f"UNPARSEABLE_TIMESTAMP:{timestamp_utc}")
        else:
            self._check_monotonicity(ts_s, report)
            self._check_fps_consistency(ts_s, report)
            self._last_timestamp_s = ts_s

        self._check_frame_continuity(frame_number, report)
        self._last_frame_number = frame_number

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_monotonicity(self, ts_s: float, report: SpoofingReport) -> None:
        """Negative timestamp gap = clock manipulation or replay attack."""
        if self._last_timestamp_s is not None:
            gap = ts_s - self._last_timestamp_s
            if gap < _MIN_TIMESTAMP_GAP_S:
                report.add_flag(
                    f"TIMESTAMP_NON_MONOTONIC:gap={gap:.6f}s "
                    f"(prev={self._last_timestamp_s:.3f} cur={ts_s:.3f})"
                )
            elif gap == 0.0:
                report.add_flag(f"TIMESTAMP_DUPLICATE:ts={ts_s:.3f}")

    def _check_fps_consistency(self, ts_s: float, report: SpoofingReport) -> None:
        """
        Track rolling frame interval. If measured FPS deviates > tolerance
        from expected, flag it.
        """
        if self._last_timestamp_s is not None:
            gap = ts_s - self._last_timestamp_s
            if gap > 0:
                self._frame_intervals.append(gap)
                # Keep rolling window of last 25 frames
                if len(self._frame_intervals) > 25:
                    self._frame_intervals.pop(0)

                avg_interval = sum(self._frame_intervals) / len(self._frame_intervals)
                measured_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                deviation    = abs(measured_fps - self.expected_fps) / self.expected_fps

                if deviation > self.fps_tolerance:
                    report.add_flag(
                        f"FPS_ANOMALY:measured={measured_fps:.1f} "
                        f"expected={self.expected_fps:.1f} "
                        f"deviation={deviation:.1%}"
                    )

    def _check_frame_continuity(self, frame_number: int, report: SpoofingReport) -> None:
        """Large frame-number jumps may indicate spliced/replaced footage."""
        if self._last_frame_number is not None:
            gap = frame_number - self._last_frame_number
            if gap < 0:
                report.add_flag(
                    f"FRAME_NUMBER_REGRESSION:prev={self._last_frame_number} cur={frame_number}"
                )
            elif gap > self.max_frame_gap:
                report.add_flag(
                    f"FRAME_GAP_TOO_LARGE:gap={gap} max={self.max_frame_gap}"
                )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _parse_timestamp(self, ts: str) -> Optional[float]:
        """Parse ISO 8601 UTC string to POSIX float. Returns None on failure."""
        import datetime
        try:
            # Python 3.11+ fromisoformat handles Z; earlier needs replacement
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return None
