"""
posture_engine.py — Core Posture Engine.

Responsibilities:
  - Read a saved frame from evidence_ref (single-frame mode, no RTSP stream needed).
  - Crop the person region using the input bbox.
  - Run MediaPipe Pose to extract 33 body landmarks.
  - Classify posture into one of 6 categories using rule-based landmark ratios.
  - Calculate a confidence score reflecting how clearly the landmarks fit the chosen class.
  - Calculate an anomaly_score — a raw posture signal.
    NOTE: anomaly_score is NOT a threat score. It is only a raw per-posture signal.
    The Orchestrator's Border Context Profile converts this into threat context later.
  - Build and return a validated output dict (ready for mqtt_bridge to publish).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid

import cv2
import mediapipe as mp
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

# ── MediaPipe landmark indices (from MediaPipe Pose 33-point model) ────────────
_L = mp.solutions.pose.PoseLandmark
NOSE        = _L.NOSE
LEFT_SHOULDER  = _L.LEFT_SHOULDER
RIGHT_SHOULDER = _L.RIGHT_SHOULDER
LEFT_HIP       = _L.LEFT_HIP
RIGHT_HIP      = _L.RIGHT_HIP
LEFT_KNEE      = _L.LEFT_KNEE
RIGHT_KNEE     = _L.RIGHT_KNEE
LEFT_ANKLE     = _L.LEFT_ANKLE
RIGHT_ANKLE    = _L.RIGHT_ANKLE
LEFT_WRIST     = _L.LEFT_WRIST
RIGHT_WRIST    = _L.RIGHT_WRIST


# ─── Posture Engine ───────────────────────────────────────────────────────────

class PostureEngine:
    """
    Processes a single human event: reads the saved frame, crops the person,
    runs MediaPipe Pose, classifies posture, computes confidence + anomaly_score,
    and returns the output event dict.
    """

    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=cfg.MP_STATIC_IMAGE_MODE,
            model_complexity=cfg.MP_MODEL_COMPLEXITY,
            min_detection_confidence=cfg.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MP_MIN_TRACKING_CONFIDENCE,
        )
        logger.info("[PostureEngine] Initialised (static_image_mode=%s, complexity=%d)",
                    cfg.MP_STATIC_IMAGE_MODE, cfg.MP_MODEL_COMPLEXITY)

    # ── Public entry point ────────────────────────────────────────────────────

    def process(
        self,
        camera_id: str,
        zone_tag: str,
        entity_id: Optional[str],
        bbox_norm: list,         # [x1, y1, x2, y2] normalised
        evidence_ref: str,       # path to saved frame from Model A
        timestamp: str,          # ISO-8601 UTC from input event
    ) -> Optional[dict]:
        """
        Process one human event.

        Returns a dict matching the locked output schema, or None if the frame
        cannot be read or MediaPipe finds no pose.
        """
        t_start = time.perf_counter()

        # ── Load frame ────────────────────────────────────────────────────
        frame = self._read_frame(evidence_ref)
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # ── Crop person bbox ──────────────────────────────────────────────
        crop = self._crop_bbox(frame, bbox_norm, w, h)
        if crop is None or crop.size == 0:
            logger.warning("[PE] Empty crop for evidence_ref=%s", evidence_ref)
            return None

        # ── Run MediaPipe Pose ────────────────────────────────────────────
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            logger.debug("[PE] No pose detected in crop from %s", evidence_ref)
            return None

        lm = result.pose_landmarks.landmark   # list of 33 NormalizedLandmark

        # ── Classify posture ──────────────────────────────────────────────
        posture_class, confidence = self._classify_posture(lm, crop)

        # ── Anomaly score (raw posture signal — NOT a threat score) ───────
        # NOTE: The Orchestrator's Border Context Profile will decide what's
        # actually threatening based on zone, time, trajectory context, etc.
        # This score only reflects how unusual the posture looks in isolation.
        anomaly_score = cfg.ANOMALY_WEIGHTS.get(posture_class, 0.1)

        processing_ms = int((time.perf_counter() - t_start) * 1000)

        # ── SHA-256 of evidence file ──────────────────────────────────────
        evidence_hash = _sha256_file(evidence_ref)

        # ── Build output event ────────────────────────────────────────────
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "posture_anomaly",
            "severity": "info",
            "timestamp": timestamp,
            "camera_id": camera_id,
            "zone_tag": zone_tag,
            "entity_type": "human",
            "engine_source": "posture",
            "entity_id": entity_id,   # pass through — may be null; we do NOT assign track_ids
            "confidence": round(confidence, 6),
            "bbox": bbox_norm,
            "evidence_ref": evidence_ref,
            "metadata": {
                "model_version": cfg.MODEL_VERSION,
                "engine_name": cfg.ENGINE_NAME,
                "processing_time_ms": processing_ms,
                "pose_model": cfg.POSE_MODEL,
                "classifier": cfg.CLASSIFIER,
                "landmark_count": cfg.LANDMARK_COUNT,
                "posture_class": posture_class,
                "anomaly_score": round(anomaly_score, 4),
            },
            "hash": evidence_hash,
            "provisional": False,
        }

        logger.debug("[PE] cam=%s posture=%s conf=%.3f anomaly=%.3f ms=%d",
                     camera_id, posture_class, confidence, anomaly_score, processing_ms)
        return event

    # ── Classifier ───────────────────────────────────────────────────────────

    def _classify_posture(
        self,
        lm: list,
        crop: np.ndarray,
    ) -> tuple[str, float]:
        """
        Rule-based posture classifier using normalised landmark ratios.

        Returns (posture_class, confidence) where confidence reflects how
        clearly the landmark geometry fits the chosen class's thresholds.

        Rules are applied in priority order from most-distinctive to least:
          crawling → crouching → carrying → running → walking → standing
        """
        ch, cw = crop.shape[:2]

        # ── Derived metrics ───────────────────────────────────────────────

        # 1. Bounding-box height/width ratio of the crop itself
        height_ratio = ch / (cw + 1e-6)

        # 2. Nose vertical position relative to hips
        nose_y = lm[NOSE].y
        hip_avg_y = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0
        nose_below_hip = nose_y > (hip_avg_y - cfg.NOSE_BELOW_HIP_THRESHOLD)

        # 3. Torso angle: angle between shoulder-midpoint and hip-midpoint vectors
        torso_angle = _torso_angle_degrees(lm)

        # 4. Ankle spread (normalised x distance between ankles)
        ankle_spread = abs(lm[LEFT_ANKLE].x - lm[RIGHT_ANKLE].x)

        # 5. Wrist-below-hip: carry indicator
        wrist_min_y = min(lm[LEFT_WRIST].y, lm[RIGHT_WRIST].y)
        wrist_below_hip = wrist_min_y > (hip_avg_y + cfg.WRIST_BELOW_HIP_THRESHOLD)

        # ── Rule evaluation (priority order) ──────────────────────────────

        # CRAWLING: very flat box OR nose near/below hip level AND torso very horizontal
        if height_ratio < cfg.HEIGHT_RATIO_CRAWL_MAX or (nose_below_hip and torso_angle >= cfg.TORSO_ANGLE_CRAWL_MIN):
            # Confidence: how strongly the geometry deviates from upright
            conf = _confidence_from_margins([
                (cfg.HEIGHT_RATIO_CRAWL_MAX - height_ratio) / cfg.HEIGHT_RATIO_CRAWL_MAX,
                (torso_angle - cfg.TORSO_ANGLE_CRAWL_MIN) / 90.0,
            ])
            return "crawling", conf

        # CROUCHING: moderately flat box AND large torso angle
        if height_ratio < cfg.HEIGHT_RATIO_CROUCHED_MAX and torso_angle >= cfg.TORSO_ANGLE_CROUCHED_MIN:
            conf = _confidence_from_margins([
                (cfg.HEIGHT_RATIO_CROUCHED_MAX - height_ratio) / cfg.HEIGHT_RATIO_CROUCHED_MAX,
                (torso_angle - cfg.TORSO_ANGLE_CROUCHED_MIN) / 90.0,
            ])
            return "crouching", conf

        # CARRYING: upright box, wrist held low (below hip level — as if holding a bag/object)
        if height_ratio >= cfg.HEIGHT_RATIO_UPRIGHT_MIN and wrist_below_hip:
            conf = _confidence_from_margins([
                (wrist_min_y - hip_avg_y - cfg.WRIST_BELOW_HIP_THRESHOLD) / 0.3,
                (height_ratio - cfg.HEIGHT_RATIO_UPRIGHT_MIN) / cfg.HEIGHT_RATIO_UPRIGHT_MIN,
            ])
            return "carrying", conf

        # RUNNING: upright box AND wide ankle spread
        if height_ratio >= cfg.HEIGHT_RATIO_UPRIGHT_MIN and ankle_spread >= cfg.ANKLE_SPREAD_RUNNING_MIN:
            conf = _confidence_from_margins([
                (ankle_spread - cfg.ANKLE_SPREAD_RUNNING_MIN) / 0.3,
                (height_ratio - cfg.HEIGHT_RATIO_UPRIGHT_MIN) / cfg.HEIGHT_RATIO_UPRIGHT_MIN,
            ])
            return "running", conf

        # WALKING: upright box AND moderate ankle spread
        if height_ratio >= cfg.HEIGHT_RATIO_UPRIGHT_MIN and ankle_spread >= cfg.ANKLE_SPREAD_WALKING_MIN:
            conf = _confidence_from_margins([
                (ankle_spread - cfg.ANKLE_SPREAD_WALKING_MIN) /
                (cfg.ANKLE_SPREAD_RUNNING_MIN - cfg.ANKLE_SPREAD_WALKING_MIN + 1e-6),
                (height_ratio - cfg.HEIGHT_RATIO_UPRIGHT_MIN) / cfg.HEIGHT_RATIO_UPRIGHT_MIN,
            ])
            return "walking", conf

        # STANDING: default upright case
        conf = _confidence_from_margins([
            (height_ratio - cfg.HEIGHT_RATIO_UPRIGHT_MIN) / cfg.HEIGHT_RATIO_UPRIGHT_MIN,
            1.0 - ankle_spread / cfg.ANKLE_SPREAD_WALKING_MIN,  # very close feet = standing
        ])
        return "standing", max(conf, 0.4)   # floor at 0.4 — standing is the safe default

    # ── Frame helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_frame(evidence_ref: str) -> Optional[np.ndarray]:
        """Load a BGR frame from the saved evidence file path."""
        p = Path(evidence_ref)
        if not p.exists():
            logger.warning("[PE] evidence_ref not found: %s", evidence_ref)
            return None
        frame = cv2.imread(str(p))
        if frame is None:
            logger.warning("[PE] cv2.imread returned None for: %s", evidence_ref)
        return frame

    @staticmethod
    def _crop_bbox(
        frame: np.ndarray,
        bbox_norm: list,
        w: int,
        h: int,
    ) -> Optional[np.ndarray]:
        """
        Crop a person from the frame using a normalised bbox.
        Adds a small padding and clamps to frame boundaries.
        """
        x1, y1, x2, y2 = bbox_norm
        pad = 0.02   # 2% padding on each side

        px1 = max(0, int((x1 - pad) * w))
        py1 = max(0, int((y1 - pad) * h))
        px2 = min(w, int((x2 + pad) * w))
        py2 = min(h, int((y2 + pad) * h))

        if px2 <= px1 or py2 <= py1:
            return None
        return frame[py1:py2, px1:px2]

    def cleanup(self) -> None:
        """Release MediaPipe resources. Call on shutdown."""
        self._pose.close()
        logger.info("[PostureEngine] Closed MediaPipe Pose.")


# ─── Pure helpers (no engine state) ──────────────────────────────────────────

def _torso_angle_degrees(lm: list) -> float:
    """
    Angle between the vertical axis and the line from hip-midpoint to
    shoulder-midpoint. 0° = perfectly upright torso, 90° = horizontal.
    """
    sh_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0
    sh_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
    hp_x = (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2.0
    hp_y = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0

    dx = sh_x - hp_x
    dy = sh_y - hp_y   # positive = shoulders above hips (normal upright)

    # Angle from vertical (dy axis)
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return angle


def _confidence_from_margins(margins: list) -> float:
    """
    Convert a list of 0-1 margin values (how far each signal is beyond its
    threshold) into a single confidence score, clamped to [0.3, 0.97].

    Mean of positive margins; returns 0.3 if all margins are zero or negative.
    """
    positive = [min(m, 1.0) for m in margins if m > 0]
    if not positive:
        return 0.3
    raw = sum(positive) / len(positive)
    return float(min(max(raw, 0.3), 0.97))


def _sha256_file(path: str, chunk_size: int = 65536) -> str:
    """SHA-256 of a file. Returns empty string if file missing."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
