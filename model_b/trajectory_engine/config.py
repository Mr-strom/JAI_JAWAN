"""
config.py — Trajectory Engine Configuration
All tuneable knobs live here. Do not scatter magic numbers in engine code.

Coordinate convention note:
  - All velocity and spatial thresholds are in PIXEL space.
  - The trajectory buffer stores (cx_px, cy_px); so thresholds here must match.
  - Zone polygons from camera_metadata are normalised [0,1] but are scaled to
    pixels before any behaviour threshold comparisons (see trajectory_engine.py).
"""

# ─── MQTT ────────────────────────────────────────────────────────────────────
MQTT_BROKER_HOST: str = "localhost"
MQTT_BROKER_PORT: int = 1883
MQTT_KEEPALIVE: int = 60
MQTT_CLIENT_ID: str = "trajectory_engine_v1"

# Topics
TOPIC_SUBSCRIBE: str = "sih26187/camera/+/model_a/raw"
TOPIC_PUBLISH_TEMPLATE: str = "sih26187/camera/{cam_id}/model_b/trajectory"
TOPIC_HEALTH: str = "sih26187/orchestrator/health"

HEARTBEAT_INTERVAL_S: int = 10          # publish health every N seconds

# ─── MODEL ───────────────────────────────────────────────────────────────────
# YOLO11x: strongest stable model for this environment (RTX 4060 Laptop, CUDA 12.1).
# YOLO12x has 1.2pt better COCO mAP but 2× the GFLOPs — not worth it on a laptop GPU.
YOLO_MODEL_PATH: str = "yolo11x.pt"     # downloaded on first run if not present
# Device: "cuda" uses the GPU when available; "cpu" forces CPU.
# "auto" is resolved at runtime — CUDA if torch.cuda.is_available(), else CPU.
YOLO_DEVICE: str = "auto"
YOLO_CONF_THRESHOLD: float = 0.30
YOLO_IOU_THRESHOLD: float = 0.45
YOLO_CLASSES: list = [0, 2, 3, 5, 7]   # person, car, motorcycle, bus, truck


# ─── SAHI — Sliced Inference for Small/Distant Targets ───────────────────────
# When ENABLE_SAHI is False the engine runs exactly as before (model.track() per frame).
# When True, each frame is tiled and YOLO runs on each tile before ByteTrack.
#
# Reference implementation:  repos/03_small_object/sahi/
# Algorithm extracted from:  sahi/slicing.py :: get_slice_bboxes()
#
# Recommended starting values for 1080p border surveillance footage:
#   slice_height/width: 640  (match YOLO training input size)
#   overlap: 0.2             (20% overlap prevents boundary misses)
#   nms_iou_threshold: 0.5   (cross-tile duplicate suppression)
ENABLE_SAHI: bool = False           # master switch — False = zero behaviour change

SAHI_SLICE_HEIGHT: int = 640           # tile height in pixels
SAHI_SLICE_WIDTH: int = 640            # tile width in pixels
SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2 # fractional vertical overlap between tiles
SAHI_OVERLAP_WIDTH_RATIO: float = 0.2  # fractional horizontal overlap between tiles
SAHI_CONF_THRESHOLD: float = 0.25      # per-tile confidence (lower than full-frame to catch small targets)
SAHI_NMS_IOU_THRESHOLD: float = 0.5    # IoU threshold for cross-tile NMS merge


# ─── NIGHT / LOW-LIGHT ENHANCEMENT ───────────────────────────────────────────
# RetinexFormer (ICCV 2023) — applied BEFORE YOLO detection on dark frames.
# Source: repos/04_night_vision/RetinexFormer  (MIT License)
#
# When ENABLE_NIGHT_ENHANCEMENT is False: zero overhead, pipeline unchanged.
# When True: each frame is enhanced before being passed to YOLO/SAHI.
#
# Pretrained weights (LOL-v1, recommended for outdoor low-light surveillance):
#   Download: https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2
#   Filename: LOLv1.pth
#   Place at: sih/model_b/trajectory_engine/weights/LOLv1.pth
ENABLE_NIGHT_ENHANCEMENT: bool = True   # master switch — False = zero behaviour change

# Path to pretrained RetinexFormer weights (.pth file)
NIGHT_WEIGHTS_PATH = "model_b/trajectory_engine/weights/LOL_v1.pth"

# Processing resolution for enhancement.
# Running RetinexFormer at full 1920x1080 costs ~1400ms — impractical.
# Quarter-res (480x270) costs ~51ms and produces visually equivalent results
# because the frame is bicubic-upsampled back to full res afterwards.
# Set to None to process at full resolution (useful if frame is already small).
#   (480, 270)  → ~51ms  — recommended for 1080p input
#   (640, 360)  → ~97ms  — higher quality if latency budget allows
#   None        → full res (only for sub-480p frames)
NIGHT_PROCESSING_SIZE: tuple = (480, 270)   # (width, height) to scale down to

# Brightness threshold — enhance only if mean pixel value is below this.
# Range [0, 255]. Set to 255 to always enhance regardless of scene brightness.
# A value of 80 catches dark/night scenes while skipping well-lit frames.
NIGHT_BRIGHTNESS_THRESHOLD: float = 80.0

# RetinexFormer architecture (must match the pretrained weights)
# LOL-v1 weights: n_feat=40, stage=1, num_blocks=[1,2,2]  ← default
# Do not change unless using a different checkpoint.
NIGHT_N_FEAT: int = 40
NIGHT_STAGE: int = 1
NIGHT_NUM_BLOCKS: list = [1, 2, 2]


# ─── BYTETRACK ───────────────────────────────────────────────────────────────
# bytetrack_border.yaml is our tuned version (lower thresholds for small/far
# targets, higher track_buffer for occlusion recovery, stricter match_thresh).
import os as _os
BYTETRACK_TRACKER_FILE: str = _os.path.join(
    _os.path.dirname(__file__), "bytetrack_border.yaml"
)
MAX_TRACK_AGE: int = 60                  # frames — matches track_buffer in yaml
MIN_HITS: int = 2                        # frames before a track is confirmed

# ─── TRAJECTORY ──────────────────────────────────────────────────────────────
TRAJECTORY_MAX_HISTORY: int = 100        # max stored points per track (pixels)
TRAJECTORY_SMOOTH_WINDOW: int = 10       # endpoint-span window for velocity/direction
                                          # (was 5 — increased because endpoint-span
                                          #  divides jitter by window, not amplifies it)

# ─── CONFIDENCE BLENDING ─────────────────────────────────────────────────────
# Weighted average: 0.6*det_conf + 0.2*age_factor + 0.2*smoothness
# avoids multiplicative near-zero suppression on new tracks.
TRACK_AGE_STABLE_FRAMES: int = 30       # age at which age_factor reaches 1.0
MAX_EXPECTED_DELTA_PX: float = 80.0     # pixels — jump ceiling for smoothness score

# ─── BEHAVIOR DETECTION (all thresholds in PIXEL space) ──────────────────────
# Velocity thresholds — pixels per second
LOITER_VELOCITY_THRESHOLD_PX_S: float = 15.0   # below this = "slow"
RAPID_APPROACH_VELOCITY_THRESHOLD_PX_S: float = 80.0   # above this = "fast"

# Loitering position variance — pixels² (trajectory stores pixel coords).
# A stationary or slightly milling person has variance in the hundreds of px².
# Setting a minimum avoids flagging a completely frozen artifact.
# Setting a maximum avoids flagging someone walking a large loop.
#   Example: milling within ~15px radius → variance ≈ 15² / 2 ≈ 112 px²
LOITER_VARIANCE_MIN_PX: float = 10.0    # px² — minimum to exclude frozen artifacts
LOITER_VARIANCE_MAX_PX: float = 4000.0  # px² — maximum to exclude wide traversal

LOITER_DURATION_S: float = 60.0         # seconds the slow+bounded condition must hold

# ─── TRACK LIFECYCLE ─────────────────────────────────────────────────────────
# Stationary duration resets to 0 when the entity resumes movement.
# Set False to accumulate total stationary time across the track lifetime.
STATIONARY_RESET_ON_MOVE: bool = True

# ─── TRACKING STABILITY (Border Surveillance v2) ─────────────────────────────

# Camera FPS — used only as fallback when timestamp dt is unreliable (e.g. first frame).
# Velocity is now computed from wall-clock timestamps inside _update_track,
# so this value affects only the trajectory-based direction magnitude, not vel_ema.
CAMERA_FPS: float = 30.0

# ── Centroid EMA (trajectory smoothing) ──────────────────────────────────────
# Smooth the raw centroid from ByteTrack before storing in trajectory.
# Eliminates jitter from frame-to-frame detection noise without affecting
# the tracker's internal Kalman state. Only the stored trajectory is smoothed.
# α=0.5: equal weight to new and previous — strong but not laggy.
# Set to 1.0 to disable (raw centroid).
CENTROID_EMA_ALPHA: float = 0.5

# ── Velocity EMA (spike damping) ─────────────────────────────────────────────
# Velocity is computed from wall-clock dt × centroid displacement (timestamp-
# accurate, not FPS-dependent). Then an EMA is applied to suppress spikes.
# α=0.25: heavily smoothed — spike of 380 px/s decays to baseline in ~4 frames.
# Set to 1.0 to disable EMA (raw instantaneous velocity).
VELOCITY_EMA_ALPHA: float = 0.25

# ── Direction EMA ─────────────────────────────────────────────────────────────
# Applied as unit vector (dx, dy) to avoid 0°/360° wraparound artefacts.
# α=0.25: responds to real heading change in ~4 frames, ignores single-frame jitter.
DIRECTION_EMA_ALPHA: float = 0.25

# ── Velocity noise floor (dead zone) ─────────────────────────────────────────
# Below this threshold, vel_ema is clamped to 0 for stationary classification.
# Border surveillance: slow-moving infiltrators at long range may have 5-8 px/s.
# Keep threshold low enough to catch genuine slow movement.
VELOCITY_NOISE_FLOOR_PX_S: float = 5.0

# ── Stale track GC ────────────────────────────────────────────────────────────
TRACK_MAX_STALE_S: float = 6.0  # slightly > MAX_TRACK_AGE/CAMERA_FPS (60/30=2s)

# ── Confidence EMA ───────────────────────────────────────────────────────────
CONF_EMA_ALPHA: float = 0.5

# ── Ghost track suppression ──────────────────────────────────────────────────
SUPPRESS_NEW_TRACK_EVENTS: bool = True

# ── Hit streak (consecutive frames required for ACTIVE) ──────────────────────
# Border use: 2 consecutive frames is enough for long-range small targets.
# Reducing from 3 to 2 improves detection of fast-crossing infiltrators.
HIT_STREAK_MIN: int = 2

# ── ByteTrack border tuning ───────────────────────────────────────────────────
# These override the ultralytics bytetrack.yaml defaults for border surveillance.
# Higher track_buffer = longer occlusion recovery window (person behind vehicle).
# Lower high_thresh = detect small/far targets that YOLO scores at 0.20-0.25.
BYTETRACK_TRACK_BUFFER: int = 60      # frames: 2s @ 30fps occlusion tolerance
BYTETRACK_HIGH_THRESH: float = 0.20   # lower → catch small far humans
BYTETRACK_LOW_THRESH: float = 0.07    # second-stage low-confidence matches
BYTETRACK_NEW_TRACK_THRESH: float = 0.20
BYTETRACK_MATCH_THRESH: float = 0.85  # higher → stricter association (fewer ID switches)

# ── Zone hysteresis ───────────────────────────────────────────────────────────
# An object must remain inside a zone for at least this many seconds before
# a zone entry/exit event is recorded. Prevents rapid enter/exit flicker for
# objects moving along zone boundaries.
ZONE_DWELL_S: float = 1.5

# ── Trajectory archive TTL ────────────────────────────────────────────────────
# How long (seconds) a GC-archived track is kept before permanent deletion.
# Should be longer than TRACK_MAX_STALE_S to allow ByteTrack ID reuse recovery.
# 10s gives enough time for a person to walk behind a building and reappear.
TRAJECTORY_ARCHIVE_TTL_S: float = 10.0

# ── Minimum detection area filter ────────────────────────────────────────────
# Bounding boxes smaller than this (px²) are discarded before tracking.
# Suppresses sensor noise and birds without discarding genuine small targets.
# Human at 100m ≈ 20×30 = 600 px² at 1080p. Set to 0 to disable.
MIN_DETECTION_AREA_PX: float = 200.0

# ── Track quality score parameters ───────────────────────────────────────────
# hit_streak required for quality streak score to saturate at 1.0.
# 30 frames at 30fps = 1 second of continuous detection = fully trusted track.
TRACK_AGE_STABLE_FRAMES: int = 30

# Maximum expected displacement (px) between two frames for a tracked object.
# Used to normalise trajectory smoothness score. 40px/frame ≈ fast vehicle.
MAX_EXPECTED_DELTA_PX: float = 40.0

# ── Camera motion compensation ────────────────────────────────────────────────
# Not needed for fixed cameras. When PTZ/moving cameras are deployed, enable
# this and implement GMC (Global Motion Compensation) from BoT-SORT.
# Currently a stub — setting True has no effect until the module is implemented.
ENABLE_CAMERA_MOTION_COMP: bool = False

# ─── TRAJECTORY VISUALIZATION ────────────────────────────────────────────────
# Number of tail points drawn as a polyline on the test overlay.
# Configurable so you can tune visual density without touching engine logic.
TRAJ_VIZ_POINTS: int = 30              # draw last N points as tail


# ─── CAMERA ──────────────────────────────────────────────────────────────────
CAMERA_CONFIG_PATH: str = "camera_config.json"  # zone polygons per cam_id
FRAME_CAPTURE_TIMEOUT_S: float = 5.0            # how long to wait for a frame
EVIDENCE_OUTPUT_DIR: str = "evidence_frames"    # where processed frames are saved

# ─── ENGINE META ─────────────────────────────────────────────────────────────
MODEL_VERSION: str = "TrajectoryEngine-v1.1-yolo11x"
ENGINE_NAME: str = "trajectory"
TRACKER_NAME: str = "bytetrack"
KALMAN_ENABLED: bool = True
