"""
Zone Tagger — classifies frames as close_range or long_range
SIH26187 | Model A | Step 5 of pipeline

Rule (from spec):
  close_range : object height >= 200px at 1080p reference
  long_range  : object height <  200px at 1080p reference

  Heights are normalised relative to 1080p before comparison,
  so the same threshold works for cameras of different resolutions.

  Camera metadata (from calibration config) can override the pixel
  heuristic entirely via a static zone_tag assignment.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from model_a.schema_v1 import Zone, ZoneTag

logger = logging.getLogger(__name__)

# Reference resolution for height normalisation
_REF_HEIGHT_PX = 1080
_CLOSE_RANGE_THRESHOLD_PX = 200  # default at 1080p (spec SIH26187)


class ZoneTagger:
    """
    Tags each frame (and its primary detection bbox) with a ZoneTag
    and a Zone label.

    Camera calibration metadata may supply:
      - static_zone_tag  : forced "close_range" or "long_range"
      - static_zone      : forced Zone enum value
      - frame_height_px  : actual camera resolution height (for normalisation)

    If no calibration override is set, the tagger derives the tag from
    the tallest bounding box in the frame.
    """

    def __init__(
        self,
        camera_id: str,
        frame_height_px: int = 1080,
        static_zone_tag: Optional[ZoneTag] = None,
        static_zone: Optional[Zone] = None,
        zone_boundary_px: int = _CLOSE_RANGE_THRESHOLD_PX,
    ) -> None:
        """
        Args:
            camera_id:        Unique camera identifier.
            frame_height_px:  Actual camera resolution height for normalisation.
            static_zone_tag:  If set, all detections are forced to this tag.
            static_zone:      If set, all detections are forced to this Zone.
            zone_boundary_px: Minimum bbox height (at 1080p equivalent) that
                              classifies a detection as close_range.
                              Default 200 per SIH26187 spec. Adjustable without
                              code changes for threshold calibration.
        """
        self.camera_id        = camera_id
        self.frame_height_px  = frame_height_px
        self.static_zone_tag  = static_zone_tag
        self.static_zone      = static_zone
        self._zone_boundary_px = zone_boundary_px

        # Scale factor to normalise to 1080p
        self._scale = _REF_HEIGHT_PX / frame_height_px

    def tag(
        self,
        bbox_normalised: list[float],   # [x1, y1, x2, y2] in [0,1]
        override_zone: Optional[Zone] = None,
    ) -> tuple[ZoneTag, Zone]:
        """
        Classify a detection into (ZoneTag, Zone).

        Args:
            bbox_normalised: [x1, y1, x2, y2] all in [0.0, 1.0].
            override_zone:   If provided, overrides the zone label
                             (caller has stronger context, e.g. polygon map).

        Returns:
            (ZoneTag, Zone) tuple.
        """
        zone_tag = self._resolve_zone_tag(bbox_normalised)
        zone     = override_zone or self.static_zone or self._default_zone(zone_tag)

        logger.debug(
            "cam=%s zone_tag=%s zone=%s bbox=%s",
            self.camera_id, zone_tag.value, zone.value, bbox_normalised,
        )
        return zone_tag, zone

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def zone_boundary_px(self) -> int:
        """Threshold (at 1080p equivalent) in pixels. Exposed for visualization."""
        return self._zone_boundary_px

    @property
    def zone_boundary_native_px(self) -> int:
        """
        Threshold converted back to this camera's native resolution.
        A detection bbox taller than this (in native pixels) is close_range.
        Use this for drawing the boundary on frames at native resolution.
        """
        return int(self._zone_boundary_px / self._scale)

    def _resolve_zone_tag(self, bbox: list[float]) -> ZoneTag:
        """Use calibration override if set, otherwise compute from bbox height."""
        if self.static_zone_tag is not None:
            return self.static_zone_tag

        _, y1, _, y2 = bbox
        # Pixel height in the camera's native resolution
        native_height_px = (y2 - y1) * self.frame_height_px
        # Normalise to 1080p equivalent
        normalised_px = native_height_px * self._scale

        if normalised_px >= self._zone_boundary_px:
            return ZoneTag.close_range
        return ZoneTag.long_range

    def _default_zone(self, zone_tag: ZoneTag) -> Zone:
        """
        Conservative default zone assignment when no polygon map is available.
        Close-range defaults to intrusion_zone; long-range to perimeter.
        """
        if zone_tag == ZoneTag.close_range:
            return Zone.intrusion_zone
        return Zone.perimeter
