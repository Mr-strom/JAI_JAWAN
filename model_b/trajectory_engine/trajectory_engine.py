"""
trajectory_engine.py — Core Trajectory Engine.

Responsibilities:
  - Maintain per-camera YOLO+ByteTrack instances.
  - Process a long_range event: grab latest frame, run tracking, update track state.
  - Compute blended confidence, velocity/direction, zone transitions, behavior tags.
  - Build and return a list of validated TrajectoryEvents (one per active track).

Does NOT touch MQTT, camera streams, or file I/O for evidence — those live in
mqtt_bridge.py and camera_stream.py respectively.

Coordinate convention (internal):
  - Track.trajectory stores pixel coordinates (cx_px, cy_px).
  - Velocity / direction / behaviour thresholds all operate in pixel space.
  - Output bbox and trajectory_points are normalized [0,1] only at the schema boundary.

Features implemented in Track:
  - Lifecycle state: NEW → ACTIVE → LOST → REMOVED
  - Persistence score: frame_count / TRACK_AGE_STABLE_FRAMES  (standard MOT metric)
  - Distance travelled: cumulative Euclidean distance in pixels
  - Stationary duration: accumulated seconds below velocity threshold
  - Zone history: chronological list of zones entered (distinct from zone_transitions)
"""

from __future__ import annotations

import logging
import math
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

# Lifecycle states (standard MOT convention)
TRACK_NEW = "NEW"         # fewer than MIN_HITS confirmed frames
TRACK_ACTIVE = "ACTIVE"   # confirmed, currently detected
TRACK_LOST = "LOST"       # missed in current frame, within max_age
TRACK_REMOVED = "REMOVED" # expired — max_age exceeded


@dataclass
class Track:
    """Stores all state for one tracked entity."""
    track_id: int
    entity_type: str                            # "human" | "vehicle"
    camera_id: str

    # Trajectory: list of (cx_px, cy_px) in PIXEL coordinates, newest appended last.
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    frame_count: int = 0
    last_bbox_norm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # normalised
    last_detection_conf: float = 0.0
    last_zone: Optional[str] = None
    zone_transitions: List[str] = field(default_factory=list)

    # Behavior state
    loitering_start: Optional[float] = None

    # ── New features ──────────────────────────────────────────────────────────

    # 1. Lifecycle state
    lifecycle_state: str = TRACK_NEW

    # 2. Distance travelled — cumulative Euclidean distance in pixels
    distance_travelled_px: float = 0.0

    # 3. Stationary duration — accumulated seconds below velocity threshold
    stationary_duration_s: float = 0.0
    _last_update_time: float = field(default_factory=time.monotonic)   # internal clock

    # 4. Zone history — chronological list of distinct zones entered
    zone_history: List[str] = field(default_factory=list)

    # 5. EMA-smoothed detection confidence — reduces per-frame flicker.
    #    Initialised to -1.0 to signal "first observation" (set directly on first hit).
    conf_ema: float = -1.0

    # 6. Hit streak (extracted from OC-SORT / Norfair pattern).
    #    Counts consecutive frames where this track was matched to a detection.
    #    Resets to 0 when a frame is missed (track was predicted but not observed).
    #    More robust than frame_count for ghost suppression: a track seen 3 frames
    #    ago but missed 5 frames has frame_count=3 but hit_streak=0.
    hit_streak: int = 0

    # 7. Last real observation centroid in pixels (OC-SORT insight).
    #    Only updated when the track is actually matched to a detection this frame.
    #    Used for direction calculation: prevents Kalman-drift from poisoning
    #    the direction vector when the tracker is coasting on predictions.
    #    None until the second detection (need two real points for a direction).
    last_real_obs_px: Optional[Tuple[float, float]] = None

    # 8. Direction EMA as a unit vector (BoT-SORT smooth_feat pattern).
    #    Stored as (dx, dy) components rather than scalar degrees to avoid
    #    0°/360° wraparound discontinuity in the EMA update.
    #    Seeded with (0.0, 0.0) = unknown; first non-zero direction initialises it.
    dir_ema_dx: float = 0.0
    dir_ema_dy: float = 0.0

    # 9. Centroid EMA — smooth stored positions before appending to trajectory.
    #    Reduces detection jitter without touching ByteTrack's Kalman state.
    #    None until first observation.
    centroid_ema: Optional[Tuple[float, float]] = None

    # 10. Velocity EMA — timestamp-accurate, spike-damped px/s estimate.
    #     Computed from wall-clock dt × centroid displacement inside _update_track.
    #     -1.0 = not yet initialised (first frame).
    vel_ema: float = -1.0

    # 11. Zone enter times — maps zone_name → wall-clock time of first entry.
    #     Used for hysteresis: entry/exit only recorded after ZONE_DWELL_S seconds.
    zone_enter_times: Dict[str, float] = field(default_factory=dict)

    # 12. Track quality score [0.0, 1.0] — synthesises hit_streak, conf_ema, and
    #     trajectory smoothness into a single reliability signal.
    #     Used to gate analytics confidence without changing event schema.
    #     Updated every active frame in process().
    track_quality: float = 0.0


# ─── Per-Camera Engine Instance ────────────────────────────────────────────────────

@dataclass
class _CameraEngineState:
    """Holds the YOLO model + track dict for a single camera.

    When SAHI is enabled, bytetracker is a BYTETracker instance that receives
    merged tile detections directly. When SAHI is disabled, bytetracker is None
    and ByteTrack runs internally via model.track() as before.
    """
    model: object                       # ultralytics YOLO instance
    tracks: Dict[int, Track]            # bytetrack_id → Track
    bytetracker: Optional[object] = None  # BYTETracker instance, only when SAHI enabled
    # Trajectory archive: tracks moved here by GC instead of deleted outright.
    # When ByteTrack reassigns the same ID after occlusion, we restore from here
    # so trajectory history and velocity/direction EMA continue uninterrupted.
    track_archive: Dict[int, Track] = field(default_factory=dict)


# ─── SAHI path helpers ────────────────────────────────────────────────────────

class _SahiTrackedBox:
    """Wraps one row of BYTETracker._format_output() into the box interface.

    BYTETracker._format_output() returns rows of:
        [x1, y1, x2, y2, track_id, conf, cls, idx]

    The per-box loop in process() reads:
        box.id       → tensor wrapping track_id (or None to skip)
        box.conf     → tensor [1]
        box.cls      → tensor [1]
        box.xyxy[0]  → tensor [4] in full-frame pixels

    This class satisfies that interface without importing ultralytics Results.
    """
    def __init__(self, row: "np.ndarray", w: int, h: int) -> None:
        import torch
        x1, y1, x2, y2, track_id, conf, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4]), float(row[5]), float(row[6])
        self.id   = torch.tensor([track_id])
        self.conf = torch.tensor([conf])
        self.cls  = torch.tensor([cls])
        # shape [1,4] so xyxy[0] works like model.track() output
        self.xyxy = torch.tensor([[x1, y1, x2, y2]])


class _EmptyDetections:
    """Fed to BYTETracker when no SAHI detections exist in a frame.

    BYTETracker._split_detections() reads .conf and .xywh and checks len().
    Returning empty numpy arrays lets the tracker correctly age and retire
    lost tracks even when a frame has no new detections.
    """
    def __init__(self) -> None:
        import numpy as np
        self.conf = np.empty((0,), dtype=np.float32)
        self.xywh = np.empty((0, 4), dtype=np.float32)
        self.cls  = np.empty((0,), dtype=np.float32)

    def __len__(self) -> int:
        return 0

    def __getitem__(self, mask) -> "_EmptyDetections":
        return self



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

        # SAHI import: deferred here so the module can still be imported even if
        # sahi_inference.py has an issue — fail is surfaced at first use, not import.
        self._sahi_enabled = cfg.ENABLE_SAHI
        if self._sahi_enabled:
            from sahi_inference import run_sahi_inference as _sahi_fn
            self._run_sahi = _sahi_fn
            logger.info("[TrajectoryEngine] SAHI sliced inference ENABLED "
                        "(tiles=%dx%d, overlap=%.1f/%.1f, conf=%.2f)",
                        cfg.SAHI_SLICE_WIDTH, cfg.SAHI_SLICE_HEIGHT,
                        cfg.SAHI_OVERLAP_WIDTH_RATIO, cfg.SAHI_OVERLAP_HEIGHT_RATIO,
                        cfg.SAHI_CONF_THRESHOLD)
        else:
            logger.info("[TrajectoryEngine] Initialised (SAHI disabled — baseline mode). "
                        "Evidence dir: %s", self._evidence_dir)

        # Night / low-light enhancement (RetinexFormer).
        # Loaded eagerly so weight errors surface at startup, not mid-stream.
        # When disabled: _night_enhancer is None — zero overhead, zero code path change.
        self._night_enhancer = None
        self._night_brightness_threshold = cfg.NIGHT_BRIGHTNESS_THRESHOLD
        if cfg.ENABLE_NIGHT_ENHANCEMENT:
            import torch as _torch
            _night_device = "cuda" if _torch.cuda.is_available() else "cpu"
            from night_enhancement import NightEnhancer
            self._night_enhancer = NightEnhancer(
                weights_path=cfg.NIGHT_WEIGHTS_PATH,
                processing_size=cfg.NIGHT_PROCESSING_SIZE,
                device=_night_device,
                n_feat=cfg.NIGHT_N_FEAT,
                stage=cfg.NIGHT_STAGE,
                num_blocks=cfg.NIGHT_NUM_BLOCKS,
            )
            logger.info(
                "[TrajectoryEngine] Night enhancement ENABLED "
                "(proc_size=%s, brightness_threshold=%.0f, device=%s)",
                cfg.NIGHT_PROCESSING_SIZE, cfg.NIGHT_BRIGHTNESS_THRESHOLD, _night_device,
            )

    # ── Public entry point ───────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: str,
        entity_type: str,
        evidence_ref_in: str,
        camera_metadata: dict,
    ) -> List[TrajectoryEvent]:
        """
        Process one frame and return a TrajectoryEvent for EVERY active track.

        Returns an empty list if no tracks are active this frame.
        Callers that previously expected Optional[TrajectoryEvent] should take
        the first element of the list (or None if empty) — see mqtt_bridge.py.

        Args:
            frame:            Latest frame from the camera stream (BGR numpy).
            camera_id:        Camera identifier string.
            timestamp:        ISO-8601 UTC string from the triggering Model A event.
            entity_type:      "human" | "vehicle" (from Model A, used as hint).
            evidence_ref_in:  Path to the frame saved by Model A (for SHA-256).
            camera_metadata:  Dict with at least {"zone_polygons": {"zone_name": [[x,y],...]}}
        """
        t_start = time.perf_counter()
        now = time.monotonic()   # single wall-clock snapshot for this entire frame

        state = self._get_or_create_camera_state(camera_id)
        zone_polygons: Dict[str, List[Tuple[float, float]]] = self._parse_zone_polygons(camera_metadata)

        h, w = frame.shape[:2]

        # ── Night / low-light enhancement (optional) ─────────────────────────
        # Applied BEFORE detection so YOLO sees a brightness-corrected frame.
        # Only runs when ENABLE_NIGHT_ENHANCEMENT=True AND the frame is dark enough.
        # When disabled (_night_enhancer is None): this block is skipped entirely.
        if self._night_enhancer is not None:
            mean_brightness = float(frame.mean())
            if mean_brightness < self._night_brightness_threshold:
                frame = self._night_enhancer.enhance(frame)
                # h, w unchanged — enhance() returns same spatial size

        # ── Detection: SAHI tiled path OR baseline model.track() ──────────────
        if self._sahi_enabled:
            boxes = self._run_sahi_boxes(state, frame, h, w)
        else:
            # ── Baseline: YOLO + ByteTrack (unchanged from original) ─────
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
                return []
            boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # Still reset hit_streak for all live tracks — no box matched any of them
            for t in state.tracks.values():
                t.hit_streak = 0
                if t.lifecycle_state == TRACK_ACTIVE:
                    t.lifecycle_state = TRACK_LOST
            return []

        # Collect which track IDs are active this frame so we can reset streaks
        # for missed tracks after the loop.
        active_ids_this_frame: set = set()

        events: List[TrajectoryEvent] = []

        for box in boxes:
            # ByteTrack assigns id via box.id; skip unconfirmed tracks
            if box.id is None:
                continue

            track_id_int = int(box.id.item())
            det_conf = float(box.conf.item())
            cls_id = int(box.cls.item())

            etype = self._class_to_entity_type(cls_id)

            # Normalised bbox [x1,y1,x2,y2] — used for schema output only.
            # Clamp pixel coords to valid frame bounds BEFORE dividing.
            # Rationale: BYTETracker's Kalman filter can extrapolate a track's
            # predicted position slightly outside the frame (e.g. x1 = -0.0015px)
            # when a target is near an edge. This is expected Kalman behaviour, not
            # a tracker bug. Without clamping, the resulting normalised value is
            # fractionally negative and fails Pydantic's [0,1] bbox validation.
            # The clamp is applied here (at the schema boundary) so all internal
            # tracking calculations continue to use the raw Kalman coordinates.
            xyxy = box.xyxy[0].cpu().numpy()  # pixel coords (may be slightly OOB)
            x1_px = float(max(0.0, min(xyxy[0], w)))
            y1_px = float(max(0.0, min(xyxy[1], h)))
            x2_px = float(max(0.0, min(xyxy[2], w)))
            y2_px = float(max(0.0, min(xyxy[3], h)))
            x1n = x1_px / w
            y1n = y1_px / h
            x2n = x2_px / w
            y2n = y2_px / h
            bbox_norm = [x1n, y1n, x2n, y2n]

            # ── Minimum detection area filter ─────────────────────────────
            # Skip tiny bounding boxes that are sensor noise or birds.
            # Border deployment: a human at 100m occupies ~400-600px² at 1080p.
            # Set to 0 in config to disable (pass all detections through).
            box_area_px = (x2_px - x1_px) * (y2_px - y1_px)
            if box_area_px < cfg.MIN_DETECTION_AREA_PX:
                continue

            # Centre in PIXEL coordinates — used for all internal calculations
            cx_px = (float(xyxy[0]) + float(xyxy[2])) / 2.0
            cy_px = (float(xyxy[1]) + float(xyxy[3])) / 2.0

            # ── Update / create Track ──────────────────────────────────────────
            # velocity is now computed INSIDE _update_track from wall-clock dt.
            track = self._update_track(
                state, track_id_int, etype, camera_id,
                (cx_px, cy_px), bbox_norm, det_conf,
            )
            active_ids_this_frame.add(track_id_int)

            # ── Zone hit-test with hysteresis (ZONE_DWELL_S) ────────────────
            # Prevents rapid enter/exit flicker for objects on zone boundaries.
            # An object must remain inside a zone for ZONE_DWELL_S seconds
            # before the entry is recorded. If it leaves before that, the
            # pending entry is cancelled (zone_enter_times entry removed).
            cx_norm = (x1n + x2n) / 2.0
            cy_norm = (y1n + y2n) / 2.0
            current_zone = self._detect_zone(cx_norm, cy_norm, zone_polygons)

            if current_zone:
                if current_zone not in track.zone_enter_times:
                    track.zone_enter_times[current_zone] = now
                dwell = now - track.zone_enter_times[current_zone]
                if dwell >= cfg.ZONE_DWELL_S and current_zone != track.last_zone:
                    if track.last_zone:
                        transition = f"{track.last_zone}→{current_zone}"
                        track.zone_transitions.append(transition)
                        logger.debug("[TE] Track %d zone: %s", track_id_int, transition)
                    track.zone_history.append(current_zone)
                    track.last_zone = current_zone
            else:
                # Object not in any zone — clear pending entry timers
                # (keeps only the timer for the current confirmed zone, if any)
                zones_to_clear = [z for z in track.zone_enter_times if z != track.last_zone]
                for z in zones_to_clear:
                    del track.zone_enter_times[z]

            # ── Velocity / direction (pixel space) ──────────────────────────
            # vel_px: from track.vel_ema (timestamp-accurate, EMA-smoothed)
            # direction: from endpoint-span trajectory + direction EMA unit vector
            vel_px = max(track.vel_ema, 0.0)

            # Apply noise floor dead zone (do NOT zero direction)
            if vel_px < cfg.VELOCITY_NOISE_FLOOR_PX_S:
                vel_px = 0.0

            # ── Direction EMA as unit vector ───────────────────────────────
            # Heading is computed from the endpoint-span of the SMOOTHED trajectory
            # (centroid EMA already removed jitter from stored points).
            # Applied as (dx, dy) unit vector to avoid 0°/360° wraparound.
            # Only updated when velocity is above dead zone (heading is meaningful).
            # Below the dead zone, last known EMA heading is held — no zeroing.
            _, direction_raw = calculate_velocity_direction(
                track.trajectory, cfg.CAMERA_FPS, window=cfg.TRAJECTORY_SMOOTH_WINDOW
            )
            if vel_px >= cfg.VELOCITY_NOISE_FLOOR_PX_S or (
                track.dir_ema_dx == 0.0 and track.dir_ema_dy == 0.0 and direction_raw != 0.0
            ):
                dir_rad = math.radians(direction_raw)
                raw_dx = math.cos(dir_rad)
                raw_dy = math.sin(dir_rad)
                if track.dir_ema_dx == 0.0 and track.dir_ema_dy == 0.0:
                    track.dir_ema_dx, track.dir_ema_dy = raw_dx, raw_dy
                else:
                    a = cfg.DIRECTION_EMA_ALPHA
                    track.dir_ema_dx = a * raw_dx + (1.0 - a) * track.dir_ema_dx
                    track.dir_ema_dy = a * raw_dy + (1.0 - a) * track.dir_ema_dy
                mag = math.hypot(track.dir_ema_dx, track.dir_ema_dy)
                if mag > 1e-9:
                    track.dir_ema_dx /= mag
                    track.dir_ema_dy /= mag

            if track.dir_ema_dx == 0.0 and track.dir_ema_dy == 0.0:
                direction = 0.0
            else:
                direction = math.degrees(
                    math.atan2(track.dir_ema_dy, track.dir_ema_dx)
                ) % 360.0


            # ── EMA confidence smoothing (Phase 3) ────────────────────────
            # Smooth det_conf with an exponential moving average to reduce
            # frame-to-frame flicker caused by single low-confidence detections.
            if track.conf_ema < 0.0:
                track.conf_ema = det_conf          # first observation: seed directly
            else:
                track.conf_ema = (
                    cfg.CONF_EMA_ALPHA * det_conf
                    + (1.0 - cfg.CONF_EMA_ALPHA) * track.conf_ema
                )
            smoothed_conf = track.conf_ema

            # ── Ghost track suppression (Phase 3) ─────────────────────────
            # Do not emit events for tracks that ByteTrack has not yet confirmed
            # (lifecycle_state=NEW). A NEW track has fewer than MIN_HITS frames
            # and may still be a spurious detection.
            if cfg.SUPPRESS_NEW_TRACK_EVENTS and track.lifecycle_state == TRACK_NEW:
                continue

            # ── Convert pixel zones to pixel space for behaviour checks ───
            # Zone polygons from camera_metadata are in normalised [0,1].
            # Scale them to pixels so they share the same space as trajectories.
            zone_polygons_px = self._scale_zones_to_pixels(zone_polygons, w, h)

            # ── Behavior ──────────────────────────────────────────────────
            behavior_tags = self._analyze_behavior(track, vel_px, zone_polygons_px)

            # ── Detection confidence (clean, not multiplied down) ─────────
            blended_conf = self._compute_confidence(smoothed_conf, track.frame_count, track.trajectory)

            # ── Track quality score [0, 1] ────────────────────────────────
            # Synthesises hit_streak (continuity), conf_ema (detector certainty),
            # and trajectory smoothness (motion stability) into one signal.
            # Weights: streak 40%, confidence 40%, smoothness 20%.
            # Streak saturates at TRACK_AGE_STABLE_FRAMES (30) = quality 1.0.
            streak_score = min(track.hit_streak / float(cfg.TRACK_AGE_STABLE_FRAMES), 1.0)
            conf_score = max(track.conf_ema, 0.0)
            smooth_score = movement_smoothness(track.trajectory, cfg.MAX_EXPECTED_DELTA_PX)
            track.track_quality = round(
                0.40 * streak_score + 0.40 * conf_score + 0.20 * smooth_score, 3
            )

            # ── Build and collect event ───────────────────────────────────
            event = self._build_event(
                camera_id=camera_id,
                timestamp=timestamp,
                entity_type=etype,
                track=track,
                track_id_int=track_id_int,
                bbox_norm=bbox_norm,
                blended_conf=blended_conf,
                vel=vel_px,
                direction=direction,
                behavior_tags=behavior_tags,
                frame=frame,
                evidence_ref_in=evidence_ref_in,
                t_start=t_start,
                w=w,
                h=h,
            )
            events.append(event)

        # ── Hit streak reset for missed tracks (OC-SORT / Norfair pattern) ─
        # Any track in state.tracks that was NOT returned by ByteTrack this frame
        # (i.e. not in active_ids_this_frame) missed this detection cycle.
        # Reset hit_streak to 0 — the track can only re-reach ACTIVE by getting
        # HIT_STREAK_MIN consecutive hits again after re-detection.
        for tid, t in state.tracks.items():
            if tid not in active_ids_this_frame:
                t.hit_streak = 0
                if t.lifecycle_state == TRACK_ACTIVE:
                    t.lifecycle_state = TRACK_LOST

        # ── Stale track GC (archive-based, v3) ───────────────────────────────
        # Instead of hard-deleting, move stale live tracks to state.track_archive.
        # ByteTrack often reuses the same numeric ID after short occlusions.
        # When it does, _update_track finds the archived Track and restores it —
        # preserving trajectory, vel_ema, dir_ema, and zone history.
        stale_ids = [
            tid for tid, t in state.tracks.items()
            if (now - t.last_seen) > cfg.TRACK_MAX_STALE_S
        ]
        for tid in stale_ids:
            t = state.tracks.pop(tid)
            t.lifecycle_state = TRACK_LOST
            state.track_archive[tid] = t
            logger.debug("[TE] GC: archived track id=%d cam %s", tid, camera_id)

        # Expire archive entries older than TRAJECTORY_ARCHIVE_TTL_S
        expired_archive = [
            tid for tid, t in state.track_archive.items()
            if (now - t.last_seen) > cfg.TRAJECTORY_ARCHIVE_TTL_S
        ]
        for tid in expired_archive:
            del state.track_archive[tid]
            logger.debug("[TE] GC: expired archive id=%d cam %s", tid, camera_id)

        return events


    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_or_create_camera_state(self, camera_id: str) -> _CameraEngineState:
        if camera_id not in self._cameras:
            model = self._YOLO(cfg.YOLO_MODEL_PATH)

            # ── Device selection ─────────────────────────────────────────────
            # Resolve "auto" at runtime so config.py never imports torch at
            # module level (keeps import fast and avoids circular-import issues).
            import torch as _torch
            if cfg.YOLO_DEVICE == "auto":
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            else:
                device = cfg.YOLO_DEVICE
            model.to(device)
            logger.info("[TE] YOLO model on device=%s for cam %s", device, camera_id)

            bytetracker = None
            if self._sahi_enabled:
                # Create a dedicated BYTETracker for SAHI path.
                # This tracker holds state between frames for this camera (same role as
                # the internal tracker inside model.track() in the baseline path).
                from ultralytics.trackers import BYTETracker
                from ultralytics.utils import IterableSimpleNamespace
                import importlib.resources as _ir
                try:
                    import yaml as _yaml
                    _pkg = _ir.files("ultralytics") / "cfg" / "trackers" / "bytetrack.yaml"
                    bt_cfg = _yaml.safe_load(_pkg.read_text())
                except Exception:
                    bt_cfg = {
                        "tracker_type": "bytetrack",
                        "track_high_thresh": 0.25,
                        "track_low_thresh": 0.1,
                        "new_track_thresh": 0.25,
                        "track_buffer": cfg.MAX_TRACK_AGE,
                        "match_thresh": 0.8,
                        "fuse_score": True,
                    }
                bytetracker = BYTETracker(IterableSimpleNamespace(**bt_cfg))
                logger.info("[TE] Created YOLO+BYTETracker for cam %s", camera_id)
            else:
                logger.info("[TE] Created YOLO model instance for cam %s", camera_id)

            self._cameras[camera_id] = _CameraEngineState(
                model=model, tracks={}, bytetracker=bytetracker
            )
        return self._cameras[camera_id]

    def _run_sahi_boxes(
        self,
        state: _CameraEngineState,
        frame: np.ndarray,
        h: int,
        w: int,
    ) -> Optional[List["_SahiTrackedBox"]]:
        """Run SAHI tiled inference and return ByteTrack-assigned tracked boxes.

        Called only when cfg.ENABLE_SAHI is True.
        Returns a list of _SahiTrackedBox objects that expose the same interface
        (.id .conf .cls .xyxy) as the ultralytics Boxes objects the per-box loop uses.
        Returns None if no tracked outputs this frame.
        """
        # 1. Tiled YOLO inference → merged detections (ByteTracker-compatible)
        detections = self._run_sahi(
            model=state.model,
            frame=frame,
            slice_height=cfg.SAHI_SLICE_HEIGHT,
            slice_width=cfg.SAHI_SLICE_WIDTH,
            overlap_height_ratio=cfg.SAHI_OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=cfg.SAHI_OVERLAP_WIDTH_RATIO,
            conf_threshold=cfg.SAHI_CONF_THRESHOLD,
            iou_threshold=cfg.YOLO_IOU_THRESHOLD,
            classes=cfg.YOLO_CLASSES,
            nms_iou_threshold=cfg.SAHI_NMS_IOU_THRESHOLD,
        )

        if detections is None or len(detections) == 0:
            # Still update tracker with empty detections so it ages lost tracks
            state.bytetracker.update(
                _EmptyDetections(), frame
            )
            return None

        # 2. Feed merged detections into BYTETracker → get tracked output
        # BYTETracker.update() returns np.ndarray: [x1,y1,x2,y2,track_id,conf,cls,...]
        tracked = state.bytetracker.update(detections, frame)

        if tracked is None or len(tracked) == 0:
            return None

        # 3. Wrap tracked rows as _SahiTrackedBox so the loop below stays unchanged
        return [_SahiTrackedBox(row, w, h) for row in tracked]


    def _update_track(
        self,
        state: _CameraEngineState,
        track_id: int,
        entity_type: str,
        camera_id: str,
        center_px: Tuple[float, float],   # raw pixel centroid from ByteTrack
        bbox_norm: List[float],
        det_conf: float,
    ) -> Track:
        now = time.monotonic()
        if track_id not in state.tracks:
            if track_id in state.track_archive:
                # ByteTrack reused this ID (track re-appeared after occlusion).
                # Restore from archive — trajectory, vel_ema, dir_ema, zone_history
                # all preserved. Reset hit_streak so lifecycle re-confirms correctly.
                state.tracks[track_id] = state.track_archive.pop(track_id)
                state.tracks[track_id].hit_streak = 0
                logger.debug("[TE] Restored archived trajectory for track id=%d", track_id)
            else:
                state.tracks[track_id] = Track(
                    track_id=track_id,
                    entity_type=entity_type,
                    camera_id=camera_id,
                    _last_update_time=now,
                )

        track = state.tracks[track_id]
        dt = now - track._last_update_time   # real wall-clock seconds since last hit
        track._last_update_time = now
        track.last_seen = now
        track.frame_count += 1
        track.last_bbox_norm = bbox_norm
        track.last_detection_conf = det_conf

        # ── Centroid EMA smoothing ────────────────────────────────────────────
        # Smooth the raw centroid before storing in trajectory.
        # This removes jitter from detection noise without touching ByteTrack.
        # Only the stored trajectory is smoothed; ByteTrack's Kalman is unaffected.
        a_c = cfg.CENTROID_EMA_ALPHA
        if track.centroid_ema is None:
            track.centroid_ema = center_px
        else:
            track.centroid_ema = (
                a_c * center_px[0] + (1.0 - a_c) * track.centroid_ema[0],
                a_c * center_px[1] + (1.0 - a_c) * track.centroid_ema[1],
            )
        smooth_cx, smooth_cy = track.centroid_ema

        # ── Velocity (timestamp-accurate, EMA-smoothed) ───────────────────────
        # Compute velocity from wall-clock dt so it is FPS-independent.
        # Apply EMA with VELOCITY_EMA_ALPHA to suppress per-frame spikes.
        if track.trajectory and dt > 1e-6:
            prev = track.trajectory[-1]
            step_px = math.hypot(smooth_cx - prev[0], smooth_cy - prev[1])
            inst_vel = step_px / dt   # px/s, timestamp-accurate
            if track.vel_ema < 0.0:
                track.vel_ema = inst_vel          # seed on first real observation
            else:
                a_v = cfg.VELOCITY_EMA_ALPHA
                track.vel_ema = a_v * inst_vel + (1.0 - a_v) * track.vel_ema
        elif not track.trajectory:
            track.vel_ema = 0.0                   # first frame: no displacement

        # ── Cumulative distance (uses smoothed centroid) ───────────────────────
        if track.trajectory:
            prev = track.trajectory[-1]
            step = math.hypot(smooth_cx - prev[0], smooth_cy - prev[1])
            track.distance_travelled_px += step

        # ── Append smoothed centroid to trajectory, cap history ───────────────
        track.trajectory.append((smooth_cx, smooth_cy))
        if len(track.trajectory) > cfg.TRAJECTORY_MAX_HISTORY:
            track.trajectory = track.trajectory[-cfg.TRAJECTORY_MAX_HISTORY:]

        # ── Hit streak ────────────────────────────────────────────────────────
        track.hit_streak += 1
        track.last_real_obs_px = (smooth_cx, smooth_cy)

        # ── Lifecycle ─────────────────────────────────────────────────────────
        if track.hit_streak >= cfg.HIT_STREAK_MIN:
            track.lifecycle_state = TRACK_ACTIVE
        else:
            track.lifecycle_state = TRACK_NEW

        # ── Stationary duration (uses vel_ema) ────────────────────────────────
        vel = max(track.vel_ema, 0.0)
        if vel < cfg.LOITER_VELOCITY_THRESHOLD_PX_S and dt > 0:
            track.stationary_duration_s += dt
        elif vel >= cfg.LOITER_VELOCITY_THRESHOLD_PX_S:
            if cfg.STATIONARY_RESET_ON_MOVE:
                track.stationary_duration_s = 0.0

        return track


    def _detect_zone(
        self,
        cx: float,
        cy: float,
        zone_polygons: Dict[str, List[Tuple[float, float]]],
    ) -> Optional[str]:
        """Return the zone name containing (cx, cy). Coordinates must match polygon space."""
        for zone_name, polygon in zone_polygons.items():
            if is_point_in_polygon((cx, cy), polygon):
                return zone_name
        return None

    @staticmethod
    def _scale_zones_to_pixels(
        zone_polygons: Dict[str, List[Tuple[float, float]]],
        w: int,
        h: int,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Scale normalised zone polygons to pixel space for behaviour calculations."""
        return {
            name: [(x * w, y * h) for x, y in pts]
            for name, pts in zone_polygons.items()
        }

    def _analyze_behavior(
        self,
        track: Track,
        velocity_px: float,
        zone_polygons_px: Dict[str, List[Tuple[float, float]]],
    ) -> List[str]:
        tags: List[str] = []
        now = time.monotonic()

        # ── Loitering ─────────────────────────────────────────────────────
        # A loitering entity has: low velocity + position variance within a bounded
        # area (moderate variance means milling; zero variance means frozen/stuck).
        # Both thresholds operate in pixel space (same space as track.trajectory).
        is_slow = velocity_px < cfg.LOITER_VELOCITY_THRESHOLD_PX_S
        variance_px = position_variance(track.trajectory)

        # Loitering = slow AND staying in a bounded area (variance not too small
        # meaning genuinely frozen/artifact, not too large meaning traversing).
        # LOITER_VARIANCE_MIN_PX: small motion within a small area (milling)
        # LOITER_VARIANCE_MAX_PX: upper bound so traversal doesn't count
        is_in_area = cfg.LOITER_VARIANCE_MIN_PX <= variance_px <= cfg.LOITER_VARIANCE_MAX_PX

        if is_slow and is_in_area:
            if track.loitering_start is None:
                track.loitering_start = now
            loiter_duration = now - track.loitering_start
            if loiter_duration >= cfg.LOITER_DURATION_S:
                tags.append("loitering")
        else:
            track.loitering_start = None

        # ── Rapid approach ─────────────────────────────────────────────────
        intrusion_polygon_px = zone_polygons_px.get("intrusion_zone")
        if (
            velocity_px >= cfg.RAPID_APPROACH_VELOCITY_THRESHOLD_PX_S
            and intrusion_polygon_px
            and is_moving_toward_polygon(track.trajectory, intrusion_polygon_px)
        ):
            tags.append("rapid_approach")

        return tags

    def _compute_confidence(
        self,
        detection_conf: float,
        frame_count: int,
        trajectory: List[Tuple[float, float]],
    ) -> float:
        """
        Blended confidence = weighted combination of detection confidence,
        track stability (age), and movement smoothness.

        Uses weighted average rather than multiplication to avoid near-zero
        values on new tracks:
          confidence = 0.6 * det_conf + 0.2 * age_factor + 0.2 * smoothness

        All three components are [0,1]. Result clamped to [0,1].

        - detection_conf: raw YOLO detection score.
        - age_factor:     how long the track has been alive (0 new → 1 stable).
        - smoothness:     how consistent the motion has been (1 smooth, 0 erratic).
        """
        age_factor = min(frame_count / cfg.TRACK_AGE_STABLE_FRAMES, 1.0)
        smoothness = movement_smoothness(trajectory, cfg.MAX_EXPECTED_DELTA_PX)
        blended = 0.6 * detection_conf + 0.2 * age_factor + 0.2 * smoothness
        return float(min(max(blended, 0.0), 1.0))

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
        w: int,
        h: int,
    ) -> TrajectoryEvent:
        """Save the processed frame, compute SHA-256, and assemble the event."""

        # Save annotated evidence frame
        evidence_path = self._save_evidence_frame(frame, camera_id, track_id_int)

        # SHA-256 of the model A evidence file (not our own frame)
        evidence_hash = sha256_file(evidence_ref_in)

        processing_ms = int((time.perf_counter() - t_start) * 1000)

        # Trajectory points exported as normalised [x, y] (last 5 points).
        # Trajectory is stored in pixels internally; normalise here at the boundary.
        traj_points_norm = [
            [p[0] / w, p[1] / h]
            for p in track.trajectory[-5:]
        ]

        # 5. Persistence score = current_age / stable_age threshold (standard MOT metric)
        persistence_score = min(track.frame_count / cfg.TRACK_AGE_STABLE_FRAMES, 1.0)

        metadata = TrajectoryMetadata(
            model_version=cfg.MODEL_VERSION,
            engine_name="trajectory",
            processing_time_ms=processing_ms,
            tracker=cfg.TRACKER_NAME,
            max_track_age=cfg.MAX_TRACK_AGE,
            kalman_enabled=cfg.KALMAN_ENABLED,
            trajectory_points=traj_points_norm,
            velocity=round(vel, 4),
            direction_degrees=round(direction, 4),
            zone_transitions=list(track.zone_transitions),
            behavior_tags=behavior_tags,
            # New features
            lifecycle_state=track.lifecycle_state,
            persistence_score=round(persistence_score, 4),
            distance_travelled_px=round(track.distance_travelled_px, 2),
            stationary_duration_s=round(track.stationary_duration_s, 2),
            zone_history=list(track.zone_history),
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
        Returns normalised [0,1] polygons. Zone detection uses normalised centroid.
        Behaviour checks scale these to pixels via _scale_zones_to_pixels().
        """
        raw = camera_metadata.get("zone_polygons", {})
        parsed: Dict[str, List[Tuple[float, float]]] = {}
        for name, pts in raw.items():
            try:
                parsed[name] = [(float(p[0]), float(p[1])) for p in pts]
            except (TypeError, IndexError, ValueError):
                logger.warning("[TE] Skipping malformed polygon for zone '%s'", name)
        return parsed

    def cleanup(self) -> None:
        """Release all model resources. Call on shutdown."""
        self._cameras.clear()
        logger.info("[TrajectoryEngine] Cleaned up all camera states.")
