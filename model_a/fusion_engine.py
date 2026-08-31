"""
Multi-Camera Fusion — Cross-Camera Identity Merger
SIH26187 | Model A | Step 7 of pipeline

Purpose:
  When two cameras see the same physical entity (overlapping FOVs),
  merge their detections into a single global_fusion_id before handing
  off to Model B. This prevents double-tracking and double-counting.

Strategy:
  1. Maintain a registry of active global entities keyed by global_fusion_id.
  2. For each new detection, compute IoU between its projected position and
     existing tracked entities from other cameras.
  3. If IoU > threshold → same physical entity → merge into existing global ID.
  4. If no match → assign new global_fusion_id.

Limitation (honest, per spec):
  Without a shared coordinate system or camera overlap map, positional
  projection is approximate. For Phase Rome we implement the data structures
  and ID-assignment logic; geometric calibration can be added in a later phase.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

_IOU_MERGE_THRESHOLD   = 0.40   # IoU above this → same entity
_ENTITY_STALE_TIMEOUT  = 30.0   # seconds — purge entities not seen this long


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GlobalEntity:
    """Represents a physical entity tracked across multiple cameras."""
    global_fusion_id: str
    entity_type:      str
    first_seen:       float = field(default_factory=time.monotonic)
    last_seen:        float = field(default_factory=time.monotonic)
    # Per-camera last-known normalised bbox [x1,y1,x2,y2]
    camera_bboxes:    Dict[str, List[float]] = field(default_factory=dict)
    contributing_cameras: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Merges per-camera detections into global entity identities.

    Usage::

        fusion = FusionEngine()

        # For each processed detection:
        global_id = fusion.assign_or_merge(
            camera_id  = "cam_01",
            local_track_id = "trk_017",
            bbox_normalised = [0.3, 0.4, 0.5, 0.8],
            entity_type = "human",
        )
        # Use global_id as entity_id in the ModelAEvent.
    """

    def __init__(
        self,
        iou_threshold: float = _IOU_MERGE_THRESHOLD,
        stale_timeout: float = _ENTITY_STALE_TIMEOUT,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.stale_timeout = stale_timeout

        # local_key → global_fusion_id  (for fast lookup)
        self._local_to_global: Dict[Tuple[str, str], str] = {}
        # global_fusion_id → GlobalEntity
        self._entities: Dict[str, GlobalEntity] = {}

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def assign_or_merge(
        self,
        camera_id: str,
        local_track_id: str,
        bbox_normalised: List[float],
        entity_type: str,
    ) -> str:
        """
        Assign or find the global_fusion_id for a detection.

        Returns:
            global_fusion_id (str) — stable across cameras for the same entity.
        """
        local_key = (camera_id, local_track_id)

        # 1. Already mapped from a previous frame
        if local_key in self._local_to_global:
            gid = self._local_to_global[local_key]
            self._update_entity(gid, camera_id, bbox_normalised)
            return gid

        # 2. Try to merge with an existing global entity from another camera
        best_gid  = None
        best_iou  = 0.0
        for gid, entity in self._entities.items():
            if camera_id in entity.camera_bboxes:
                # Same camera already has this entity — skip IoU check
                continue
            for other_cam_id, other_bbox in entity.camera_bboxes.items():
                if other_cam_id == camera_id:
                    continue
                iou = self._compute_iou(bbox_normalised, other_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gid = gid

        if best_gid and best_iou >= self.iou_threshold:
            # Merge into existing entity
            self._local_to_global[local_key] = best_gid
            self._update_entity(best_gid, camera_id, bbox_normalised)
            logger.info(
                "FUSION MERGE: cam=%s track=%s → global_id=%s (IoU=%.2f)",
                camera_id, local_track_id, best_gid, best_iou,
            )
            return best_gid

        # 3. New entity — assign fresh global_fusion_id
        new_gid = str(uuid.uuid4())
        entity  = GlobalEntity(
            global_fusion_id=new_gid,
            entity_type=entity_type,
            camera_bboxes={camera_id: bbox_normalised},
            contributing_cameras=[camera_id],
        )
        self._entities[new_gid] = entity
        self._local_to_global[local_key] = new_gid
        logger.debug(
            "FUSION NEW entity: global_id=%s cam=%s track=%s",
            new_gid, camera_id, local_track_id,
        )
        return new_gid

    def purge_stale(self) -> int:
        """Remove entities not updated for stale_timeout seconds."""
        now   = time.monotonic()
        stale = [
            gid for gid, ent in self._entities.items()
            if (now - ent.last_seen) > self.stale_timeout
        ]
        for gid in stale:
            # Remove local-key mappings
            self._local_to_global = {
                k: v for k, v in self._local_to_global.items() if v != gid
            }
            del self._entities[gid]
        if stale:
            logger.debug("Fusion: purged %d stale global entities.", len(stale))
        return len(stale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_entity(
        self, gid: str, camera_id: str, bbox: List[float]
    ) -> None:
        ent = self._entities[gid]
        ent.last_seen = time.monotonic()
        ent.camera_bboxes[camera_id] = bbox
        if camera_id not in ent.contributing_cameras:
            ent.contributing_cameras.append(camera_id)

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Intersection-over-Union for two normalised bboxes [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
