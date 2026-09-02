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
ENABLE_SAHI: bool = True              # master switch — False = zero behaviour change

SAHI_SLICE_HEIGHT: int = 640           # tile height in pixels
SAHI_SLICE_WIDTH: int = 640            # tile width in pixels
SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2 # fractional vertical overlap between tiles
SAHI_OVERLAP_WIDTH_RATIO: float = 0.2  # fractional horizontal overlap between tiles
SAHI_CONF_THRESHOLD: float = 0.25      # per-tile confidence (lower than full-frame to catch small targets)
SAHI_NMS_IOU_THRESHOLD: float = 0.5    # IoU threshold for cross-tile NMS merge


# ─── BYTETRACK ───────────────────────────────────────────────────────────────
BYTETRACK_TRACKER_FILE: str = "bytetrack.yaml"   # ultralytics built-in
MAX_TRACK_AGE: int = 30                  # frames before a lost track is dropped
MIN_HITS: int = 3                        # frames before a track is confirmed

# ─── TRAJECTORY ──────────────────────────────────────────────────────────────
TRAJECTORY_MAX_HISTORY: int = 100        # max stored points per track (pixels)
TRAJECTORY_SMOOTH_WINDOW: int = 5        # last N points for velocity/direction

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
