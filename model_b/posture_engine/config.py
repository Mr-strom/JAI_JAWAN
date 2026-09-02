"""
config.py — Posture Engine Configuration
All tuneable knobs in one place.
"""

# ─── MQTT ─────────────────────────────────────────────────────────────────────
MQTT_BROKER_HOST: str = "localhost"
MQTT_BROKER_PORT: int = 1883
MQTT_KEEPALIVE: int = 60
MQTT_CLIENT_ID: str = "posture_engine_v1"

TOPIC_SUBSCRIBE: str = "sih26187/camera/+/model_a/raw"
TOPIC_PUBLISH_TEMPLATE: str = "sih26187/camera/{cam_id}/model_b/posture"
TOPIC_HEALTH: str = "sih26187/orchestrator/health"

HEARTBEAT_INTERVAL_S: int = 10

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
# static_image_mode=True because we process single saved frames, not video streams
MP_STATIC_IMAGE_MODE: bool = True
MP_MODEL_COMPLEXITY: int = 1          # 0=lite, 1=full, 2=heavy — 1 is best balance
MP_MIN_DETECTION_CONFIDENCE: float = 0.5
MP_MIN_TRACKING_CONFIDENCE: float = 0.5   # unused in static mode, kept for completeness

# ─── POSTURE RULE THRESHOLDS ──────────────────────────────────────────────────
# All ratios are derived from normalised landmark coordinates (0-1 range).

# height_ratio = person bounding-box height / width
# A tall narrow box = upright (standing/walking/running)
# A short wide box  = horizontal (crawling) or crouched

HEIGHT_RATIO_UPRIGHT_MIN: float = 1.2   # height/width must be > this to be upright
HEIGHT_RATIO_CROUCHED_MAX: float = 1.0  # height/width < this suggests crouching
HEIGHT_RATIO_CRAWL_MAX: float = 0.7     # height/width < this strongly suggests crawling

# nose_y and hip_y are normalised vertical coords (0=top, 1=bottom of frame crop)
# In a crawling pose, nose is near or below hip level in the crop
NOSE_BELOW_HIP_THRESHOLD: float = 0.05  # nose_y > hip_avg_y - threshold => crawling indicator

# Torso angle: angle between shoulders and hips midpoints
# Near 0° = upright torso, >30° = leaning/crouching, >60° = crawling
TORSO_ANGLE_UPRIGHT_MAX: float = 20.0   # degrees
TORSO_ANGLE_CROUCHED_MIN: float = 25.0
TORSO_ANGLE_CRAWL_MIN: float = 55.0

# Wrist-carry heuristic: if wrist is significantly below hip, person may be carrying something
WRIST_BELOW_HIP_THRESHOLD: float = 0.15  # normalised y delta

# Speed indicators (used as secondary signals only — we don't have inter-frame speed here,
# so we use limb spread as a proxy)
# Running: legs spread wide (left/right ankle distance > threshold)
ANKLE_SPREAD_RUNNING_MIN: float = 0.18   # normalised x distance between ankles
# Walking: moderate spread
ANKLE_SPREAD_WALKING_MIN: float = 0.06

# ─── ANOMALY SCORING WEIGHTS (raw signal only, NOT threat score) ───────────────
# These define the base anomaly_score per class.
# NOTE: anomaly_score is a raw posture signal for the Orchestrator — it is NOT a
# threat score. The Orchestrator's Border Context Profile decides what's actually
# suspicious based on zone, time, and trajectory context.
ANOMALY_WEIGHTS: dict = {
    "standing":  0.05,
    "walking":   0.10,
    "running":   0.35,
    "crouching": 0.60,
    "crawling":  0.85,
    "carrying":  0.45,
}

# ─── ENGINE META ──────────────────────────────────────────────────────────────
MODEL_VERSION: str = "PostureEngine-v1.0"
ENGINE_NAME: str = "posture"
POSE_MODEL: str = "mediapipe_pose"
CLASSIFIER: str = "rule_based"
LANDMARK_COUNT: int = 33
