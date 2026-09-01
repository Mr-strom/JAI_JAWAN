"""
trajectory_engine.py — Core Trajectory Engine.

Responsibilities:
  - Maintain per-camera YOLO+ByteTrack instances.
  - Process a long_range event: grab latest frame, run tracking, update track state.
  - Compute blended confidence, velocity/direction, zone transitions, behavior tags.
  - Build and return a validated TrajectoryEvent (ready to publish).

Does NOT touch MQTT, camera streams, or file I/O for evidence — those live in
mqtt_bridge.py and camera_stream.py respectively.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from schemas import TrajectoryEvent, TrajectoryMetadata
from utils.geometry import (
    calculate_velocity_direction,
    is_moving_toward_polygon,
    is_point_in_polygon,
    movement_smoothness,
    position_variance,
)
from utils.hash_util import sha256_file

logger = logging.getLogger(__name__)


# ─── Track State ─────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Stores all state for one tracked entity."""
    track_id: int
    entity_type: str                            # "human" | "vehicle"
    camera_id: str

    # Trajectory: list of (cx_px, cy_px) tuples, newest appended last
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    first_seen: float = field(default_factory=time.monotonic)  # monotonic seconds
    last_seen: float = field(default_factory=time.monotonic)
    frame_count: int = 0                        # frames this track has been active
    last_bbox_norm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_detection_conf: float = 0.0
    last_zone: Optional[str] = None             # last known zone_tag name
    zone_transitions: List[str] = field(default_factory=list)

    # Behavior state
    loitering_start: Optional[float] = None     # monotonic time when loiter started


# ─── Per-Camera Engine Instance ──────────────────────────────────────────────

@dataclass
class _CameraEngineState:
    """Holds the YOLO model + track dict for a single camera."""
    model: object                   # ultralytics YOLO instance
    tracks: Dict[int, Track]        # bytetrack_id -> Track


# ─── Trajectory Engine ────────────────────────────────────────────────────────

class TrajectoryEngine:
    """
    One instance shared across all cameras.
    Per-camera YOLO models are created lazily on first event for that cam.
    """

    def __init__(self) -> None:
        from ultralytics import YOLO  # deferred so tests can mock

        self._YOLO = YOLO
        self._cameras: Dict[str, _CameraEngineState] = {}
        self._evidence_dir = Path(cfg.EVIDENCE_OUTPUT_DIR)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[TrajectoryEngine] Initialised. Evidence dir: %s", self._evidence_dir)

    # ── Public entry point ───────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: str,
        entity_type: str,
        evidence_ref_in: str,
        camera_metadata: dict,
    ) -> Optional[TrajectoryEvent]:
        """
        Process one long_range event.

        Args:
            frame:            Latest frame from the camera stream (BGR numpy).
            camera_id:        Camera identifier string.
            timestamp:        ISO-8601 UTC string from the triggering Model A event.
            entity_type:      "human" | "vehicle" (from Model A, used as hint).
            evidence_ref_in:  Path to the frame saved by Model A (for SHA-256).
            camera_metadata:  Dict with at least {"zone_polygons": {"zone_name": [[x,y],...]}}

        Returns:
            A validated TrajectoryEvent, or None if no tracks were active this frame.
        """
        t_start = time.perf_counter()

        state = self._get_or_create_camera_state(camera_id)
        zone_polygons: Dict[str, List[Tuple[float, float]]] = self._parse_zone_polygons(camera_metadata)

        h, w = frame.shape[:2]

        # ── Run YOLOv8n + ByteTrack ────────────────────────────────────────
        results = state.model.track(
            frame,
            tracker=cfg.BYTETRACK_TRACKER_FILE,
            conf=cfg.YOLO_CONF_THRESHOLD,
            iou=cfg.YOLO_IOU_THRESHOLD,
            classes=cfg.YOLO_CLASSES,
            persist=True,   # keeps ByteTrack state between calls on the same model
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes

        # Pick the best (highest conf) relevant detection to build the event from.
        # All tracks are still updated internally.
        best_event: Optional[TrajectoryEvent] = None
        best_conf = -1.0

        for box in boxes:
            # ByteTrack assigns id via box.id; skip unconfirmed tracks
            if box.id is None:
                continue

            track_id_int = int(box.id.item())
            det_conf = float(box.conf.item())
            cls_id = int(box.cls.item())

            etype = self._class_to_entity_type(cls_id)

            # Normalised bbox [x1,y1,x2,y2]
            xyxy = box.xyxy[0].cpu().numpy()  # pixel coords
            x1n = float(xyxy[0]) / w
            y1n = float(xyxy[1]) / h
            x2n = float(xyxy[2]) / w
            y2n = float(xyxy[3]) / h
            bbox_norm = [x1n, y1n, x2n, y2n]

            # Centre in normalised coords (for trajectory history)
            cx_norm = (x1n + x2n) / 2.0
            cy_norm = (y1n + y2n) / 2.0

            # ── Update / create Track ─────────────────────────────────────
            track = self._update_track(
                state, track_id_int, etype, camera_id,
                (cx_norm, cy_norm), bbox_norm, det_conf,
            )

            # ── Zone hit-test ─────────────────────────────────────────────
            current_zone = self._detect_zone(cx_norm, cy_norm, zone_polygons)
            if current_zone and current_zone != track.last_zone:
                if track.last_zone:
                    transition = f"{track.last_zone}→{current_zone}"
                    track.zone_transitions.append(transition)
                    logger.debug("[TE] Track %d zone transition: %s", track_id_int, transition)
                track.last_zone = current_zone

            # ── Velocity / direction ──────────────────────────────────────
            fps = 25.0  # fallback; mqtt_bridge passes real fps when available
            vel, direction = calculate_velocity_direction(
                track.trajectory, fps, window=cfg.TRAJECTORY_SMOOTH_WINDOW
            )

            # ── Behavior ──────────────────────────────────────────────────
            behavior_tags = self._analyze_behavior(track, vel, zone_polygons)

            # ── Blended confidence ────────────────────────────────────────
            blended_conf = self._blended_confidence(
                det_conf, track.frame_count, track.trajectory
            )

            # ── Keep track of the best detection this frame ───────────────
            if blended_conf > best_conf:
                best_conf = blended_conf
                best_event = self._build_event(
                    camera_id=camera_id,
                    timestamp=timestamp,
                    entity_type=etype,
                    track=track,
                    track_id_int=track_id_int,
                    bbox_norm=bbox_norm,
                    blended_conf=blended_conf,
                    vel=vel,
                    direction=direction,
                    behavior_tags=behavior_tags,
                    frame=frame,
                    evidence_ref_in=evidence_ref_in,
                    t_start=t_start,
                )

        return best_event

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_or_create_camera_state(self, camera_id: str) -> _CameraEngineState:
        if camera_id not in self._cameras:
            model = self._YOLO(cfg.YOLO_MODEL_PATH)
            self._cameras[camera_id] = _CameraEngineState(model=model, tracks={})
            logger.info("[TE] Created YOLO model instance for cam %s", camera_id)
        return self._cameras[camera_id]

    def _update_track(
        self,
        state: _CameraEngineState,
        track_id: int,
        entity_type: str,
        camera_id: str,
        center_norm: Tuple[float, float],
        bbox_norm: List[float],
        det_conf: float,
    ) -> Track:
        now = time.monotonic()
        if track_id not in state.tracks:
            state.tracks[track_id] = Track(
                track_id=track_id,
                entity_type=entity_type,
                camera_id=camera_id,
            )

        track = state.tracks[track_id]
        track.last_seen = now
        track.frame_count += 1
        track.last_bbox_norm = bbox_norm
        track.last_detection_conf = det_conf

        # Append to trajectory, cap length
        track.trajectory.append(center_norm)
        if len(track.trajectory) > cfg.TRAJECTORY_MAX_HISTORY:
            track.trajectory = track.trajectory[-cfg.TRAJECTORY_MAX_HISTORY:]

        return track

    def _detect_zone(
        self,
        cx: float,
        cy: float,
        zone_polygons: Dict[str, List[Tuple[float, float]]],
    ) -> Optional[str]:
        """Return the name of the zone that contains (cx, cy), else None."""
        for zone_name, polygon in zone_polygons.items():
            if is_point_in_polygon((cx, cy), polygon):
                return zone_name
        return None

    def _analyze_behavior(
        self,
        track: Track,
        velocity: float,
        zone_polygons: Dict[str, List[Tuple[float, float]]],
    ) -> List[str]:
        tags: List[str] = []
        now = time.monotonic()
        duration = now - track.first_seen

        # ── Loitering ────────────────────────────────────────────────────
        is_slow = velocity < cfg.LOITER_VELOCITY_THRESHOLD_PX_S
        variance = position_variance(track.trajectory)
        is_milling = variance > cfg.LOITER_VARIANCE_THRESHOLD

        if is_slow and is_milling:
            if track.loitering_start is None:
                track.loitering_start = now
            loiter_duration = now - track.loitering_start
            if loiter_duration >= cfg.LOITER_DURATION_S:
                tags.append("loitering")
        else:
            # Reset loiter clock if they speed up or stop milling
            track.loitering_start = None

        # ── Rapid approach ────────────────────────────────────────────────
        intrusion_polygon = zone_polygons.get("intrusion_zone")
        if (
            velocity >= cfg.RAPID_APPROACH_VELOCITY_THRESHOLD_PX_S
            and intrusion_polygon
            and is_moving_toward_polygon(track.trajectory, intrusion_polygon)
        ):
            tags.append("rapid_approach")

        return tags

    def _blended_confidence(
        self,
        detection_conf: float,
        frame_count: int,
        trajectory: List[Tuple[float, float]],
    ) -> float:
        """
        confidence = detection_conf × track_age_ratio × movement_smoothness
        All factors in [0, 1]; result clamped to [0, 1].
        """
        track_age_ratio = min(frame_count / cfg.TRACK_AGE_STABLE_FRAMES, 1.0)
        smoothness = movement_smoothness(trajectory, cfg.MAX_EXPECTED_DELTA_PX)
        raw = detection_conf * track_age_ratio * smoothness
        return float(min(max(raw, 0.0), 1.0))

    def _build_event(
        self,
        *,
        camera_id: str,
        timestamp: str,
        entity_type: str,
        track: Track,
        track_id_int: int,
        bbox_norm: List[float],
        blended_conf: float,
        vel: float,
        direction: float,
        behavior_tags: List[str],
        frame: np.ndarray,
        evidence_ref_in: str,
        t_start: float,
    ) -> TrajectoryEvent:
        """Save the processed frame, compute SHA-256, and assemble the event."""

        # Save annotated evidence frame
        evidence_path = self._save_evidence_frame(frame, camera_id, track_id_int)

        # SHA-256 of the model A evidence file (not our own frame)
        evidence_hash = sha256_file(evidence_ref_in)

        processing_ms = int((time.perf_counter() - t_start) * 1000)

        # Trajectory points: normalised (x, y), last 5 shown
        traj_points = [[p[0], p[1]] for p in track.trajectory[-5:]]

        metadata = TrajectoryMetadata(
            model_version=cfg.MODEL_VERSION,
            engine_name="trajectory",
            processing_time_ms=processing_ms,
            tracker=cfg.TRACKER_NAME,
            max_track_age=cfg.MAX_TRACK_AGE,
            kalman_enabled=cfg.KALMAN_ENABLED,
            trajectory_points=traj_points,
            velocity=round(vel, 4),
            direction_degrees=round(direction, 4),
            zone_transitions=list(track.zone_transitions),
            behavior_tags=behavior_tags,
        )

        event = TrajectoryEvent(
            timestamp=timestamp,
            camera_id=camera_id,
            entity_type=entity_type if entity_type in ("human", "vehicle") else "human",
            entity_id=str(track_id_int),
            confidence=round(blended_conf, 6),
            bbox=bbox_norm,
            evidence_ref=str(evidence_path),
            metadata=metadata,
            hash=evidence_hash,
        )

        return event

    def _save_evidence_frame(
        self, frame: np.ndarray, camera_id: str, track_id: int
    ) -> Path:
        """Write the processed frame to disk, return its path."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        filename = f"{camera_id}_track{track_id}_{ts}.jpg"
        path = self._evidence_dir / filename
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    @staticmethod
    def _class_to_entity_type(cls_id: int) -> str:
        """Map YOLO class id to entity_type. COCO: 0=person, rest=vehicle."""
        return "human" if cls_id == 0 else "vehicle"

    @staticmethod
    def _parse_zone_polygons(
        camera_metadata: dict,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract zone_polygons from camera_metadata.
        Expected format: {"zone_polygons": {"zone_name": [[x,y], ...]}}
        Returns empty dict gracefully if the key is missing.
        """
        raw = camera_metadata.get("zone_polygons", {})
        parsed: Dict[str, List[Tuple[float, float]]] = {}
        for name, pts in raw.items():
            try:
                parsed[name] = [(float(p[0]), float(p[1])) for p in pts]
            except (TypeError, IndexError, ValueError):
                logger.warning("[TE] Skipping malformed polygon for zone '%s'", name)
        return parsed

    def get_active_track_count(self) -> int:
        """Return total number of live tracks across all cameras."""
        return sum(len(state.tracks) for state in self._cameras.values())

    def cleanup(self) -> None:
        """Release all model resources. Call on shutdown."""
        self._cameras.clear()
        logger.info("[TrajectoryEngine] Cleaned up all camera states.")
