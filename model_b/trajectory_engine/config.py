"""
config.py — Trajectory Engine Configuration
All tuneable knobs live here. Do not scatter magic numbers in engine code.
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
YOLO_MODEL_PATH: str = "yolov8n.pt"     # downloaded on first run if not present
YOLO_CONF_THRESHOLD: float = 0.35
YOLO_IOU_THRESHOLD: float = 0.45
YOLO_CLASSES: list = [0, 2, 3, 5, 7]   # person, car, motorcycle, bus, truck

# ─── BYTETRACK ───────────────────────────────────────────────────────────────
BYTETRACK_TRACKER_FILE: str = "bytetrack.yaml"   # ultralytics built-in
MAX_TRACK_AGE: int = 30                  # frames before a lost track is dropped
MIN_HITS: int = 3                        # frames before a track is confirmed

# ─── TRAJECTORY ──────────────────────────────────────────────────────────────
TRAJECTORY_MAX_HISTORY: int = 100        # max stored points per track
TRAJECTORY_SMOOTH_WINDOW: int = 5        # last N points for velocity/direction

# ─── CONFIDENCE BLENDING ─────────────────────────────────────────────────────
TRACK_AGE_STABLE_FRAMES: int = 30       # age at which track_age_ratio reaches 1.0
MAX_EXPECTED_DELTA_PX: float = 80.0     # pixels — erratic jump ceiling for smoothness

# ─── BEHAVIOR DETECTION ──────────────────────────────────────────────────────
LOITER_VELOCITY_THRESHOLD_PX_S: float = 15.0   # px/s — below = "low velocity"
LOITER_VARIANCE_THRESHOLD: float = 20.0         # px   — variance of last N positions
LOITER_DURATION_S: float = 60.0                 # seconds a track must persist at low vel

RAPID_APPROACH_VELOCITY_THRESHOLD_PX_S: float = 80.0  # px/s — above = fast

# ─── CAMERA ──────────────────────────────────────────────────────────────────
CAMERA_CONFIG_PATH: str = "camera_config.json"  # zone polygons per cam_id
FRAME_CAPTURE_TIMEOUT_S: float = 5.0            # how long to wait for a frame
EVIDENCE_OUTPUT_DIR: str = "evidence_frames"    # where processed frames are saved

# ─── ENGINE META ─────────────────────────────────────────────────────────────
MODEL_VERSION: str = "TrajectoryEngine-v1.0"
ENGINE_NAME: str = "trajectory"
TRACKER_NAME: str = "bytetrack"
KALMAN_ENABLED: bool = True
