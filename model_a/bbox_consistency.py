"""
BBox Consistency Checker — Shadow and Foliage False-Positive Suppressor
SIH26187 | Model A | Phase Berlin — Spatial Guard

Problem:
  Shadows, blowing foliage, and lighting transients produce detections that
  move erratically across frames. The multi-frame counter counts 3 hits, but
  the bboxes have no spatial continuity — they jump all over the frame.
  This is a separate failure mode from the time-domain multi-frame gate.

Solution:
  Track the bbox history per track_id.
  For a confirmation to be valid, the bboxes across the confirmation window
  must be spatially consistent: IoU(frame_n, frame_n-1) >= threshold.
  If the bbox jumps (IoU drops below threshold), the track is considered
  a spurious noise detection and the confirmation window is reset.

From spec (Edge Cases):
  "Shadow / Blowing Foliage: High-frequency noise fails multi-frame consistent
   bounding box check. Suppressed."

Threshold calibration:
  A real person climbing a fence moves ~10-15% of frame height per second.
  At 5 effective FPS (200ms intervals), displacement is ~2-3% of frame height.
  IoU for a 0.15-tall bbox shifted 3% vertically ≈ 0.80.
  Shadow detections can jump 30-50% of the frame → IoU ≈ 0.0.
  Threshold = 0.35 gives a wide safety margin.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum IoU between consecutive bboxes in the confirmation window.
# Below this → treat as spatial discontinuity → reset confirmation.
_IOU_CONSISTENCY_THRESHOLD = 0.35

# How many bboxes to keep in history per track (must cover confirmation window)
_BBOX_HISTORY_SIZE = 5


@dataclass
class ConsistencyRecord:
    bbox_history: Deque[List[float]] = field(
        default_factory=lambda: deque(maxlen=_BBOX_HISTORY_SIZE)
    )
    consecutive_consistent: int = 0
    consecutive_inconsistent: int = 0


class BBoxConsistencyChecker:
    """
    Validates that trigger-candidate bboxes are spatially consistent across
    consecutive frames. Used to eliminate shadow/foliage false positives.

    Integration with TriggerDetector:
      Call check() BEFORE passing the update to TriggerDetector.
      If check() returns False → call miss() on TriggerDetector instead.

    Usage::

        checker = BBoxConsistencyChecker()

        for each detection in frame:
            is_consistent = checker.check(track_id, bbox)
            if is_consistent:
                trigger_detector.update(track_id, trigger_type, frame_number)
            else:
                trigger_detector.miss(track_id)  # spatial jump → reset
    """

    def __init__(self, iou_threshold: float = _IOU_CONSISTENCY_THRESHOLD) -> None:
        self.iou_threshold = iou_threshold
        self._records: Dict[str, ConsistencyRecord] = {}

    def check(self, track_id: str, bbox: List[float]) -> bool:
        """
        Check if this bbox is spatially consistent with the track's history.

        Returns:
            True  — bbox is consistent; allow trigger confirmation to proceed.
            False — bbox jumped too far; treat as spatial discontinuity.
        """
        rec = self._get_or_create(track_id)

        if not rec.bbox_history:
            # No history yet — first detection is always consistent
            rec.bbox_history.append(bbox)
            rec.consecutive_consistent = 1
            return True

        prev_bbox = rec.bbox_history[-1]
        iou = self._compute_iou(prev_bbox, bbox)

        rec.bbox_history.append(bbox)

        if iou >= self.iou_threshold:
            rec.consecutive_consistent += 1
            rec.consecutive_inconsistent = 0
            logger.debug(
                "track=%s bbox CONSISTENT IoU=%.3f >= threshold=%.3f",
                track_id, iou, self.iou_threshold,
            )
            return True
        else:
            rec.consecutive_consistent = 0
            rec.consecutive_inconsistent += 1
            logger.warning(
                "track=%s bbox INCONSISTENT IoU=%.3f < threshold=%.3f "
                "— SPATIAL DISCONTINUITY. Resetting confirmation. "
                "(Shadow/foliage suppression active.)",
                track_id, iou, self.iou_threshold,
            )
            return False

    def reset(self, track_id: str) -> None:
        """Reset a track's history (e.g. after it re-appears after a gap)."""
        if track_id in self._records:
            self._records[track_id] = ConsistencyRecord()

    def purge_stale(self, active_track_ids: List[str]) -> None:
        """Remove records for tracks no longer active."""
        stale = [tid for tid in self._records if tid not in active_track_ids]
        for tid in stale:
            del self._records[tid]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id: str) -> ConsistencyRecord:
        if track_id not in self._records:
            self._records[track_id] = ConsistencyRecord()
        return self._records[track_id]

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """IoU of two normalised bboxes [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union  = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
