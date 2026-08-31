# Model A Developer Guide & Work Delegation — SIH26187

**Title:** Module Creator / Lightweight Filter-Classifier (Gatekeeper)

**Version:** 1.0 | 2026-08-31

**Classification:** SIH Internal Round — Top 50 Qualifier

**Status:** **LOCKED SPECIFICATION**

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[03_Architecture_Map]] · [[06_Version_Control_Testing_Strategy]] · [[07B_Model_B_Delegation]]
> 
> **Purpose:** Complete implementation guide for Model A — the gatekeeper and plumbing layer that ingests raw frames, filters redundancy, tags zones, detects absolute triggers with multi-frame confirmation, checks anti-spoofing, and normalizes everything into the unified JSON event schema. You own the infrastructure everything else depends on.

## Section 1: Scope & Responsibilities (Locked)

You own the gatekeeper layer. Your explicit responsibilities include:

- **Ingestion Handling:** RTSP/ONVIF connection to CCTV feeds, handling camera-agnostic input.
    
- **Time-Sampling:** Skip redundant/near-identical frames to reduce compute load.
    
- **Most-Differentiated-Frame Selection:** Within a sampling window, pick the frame(s) with maximum information content to pass downstream.
    
- **Range/Zone Tagging:** Classify each camera feed/frame as Close Range or Long Range based on apparent object size and camera calibration.
    
- **Absolute-Trigger Detection:** Climbing, fence-cutting, rapid motion toward fence. **Highest Priority:** Must include multi-frame confirmation (2-3 consecutive frames) before flagging to fix the documented CIBMS failure mode.
    
- **Anti-Spoofing Check:** Verify temporal continuity; flag gaps >2s as possible replay/injection.
    
- **JSON Normalization:** Convert all events into the unified JSON event schema (`schema_v1`).
    
- **Event Bus Publishing:** MQTT (Mosquitto), publishing independently (no waiting on Model B engines).
    
- **Camera Health Monitor:** Basic ping/FPS check, flags lens obstruction, darkness, frozen stream, FPS anomalies.
    
- **Fallback Routing (Safety Floor):** If a Model B engine fails/heartbeat drops, route that camera's traffic back through Model A's basic motion+trigger detection at reduced capability to prevent total blindness.
    

### 🚫 What You Must NOT Do

- **DO NOT** build a separate/bypass bus channel for CRITICAL triggers. This is a locked security decision; use severity-tagging on the _same_ channel only.
    
- **DO NOT** touch Face/ANPR/Posture/Trajectory engine internals — those belong strictly to Model B.
    
- **DO NOT** mutate `schema_v1` in place if you need new fields. Version it (`schema_v2`) to prevent silently breaking your partner's build.
    

## Section 2: Input & Output Specifications

### 2.1 Input Data

Target Input Rate: Typical IP CCTV = 25 FPS. Model A targets 1-5 FPS effective processing after sampling.

|**Field**|**Type**|**Description**|
|---|---|---|
|`source`|string|RTSP URL or file path|
|`camera_id`|string|Unique camera identifier (e.g., `BOP_01_CAM_NORTH`)|
|`frame`|numpy.ndarray|BGR image, shape (H, W, 3)|
|`timestamp`|float|Unix timestamp with microsecond precision|
|`frame_number`|int|Sequential frame counter since stream start|
|`camera_metadata`|dict|Calibration data: mounting_height, tilt_angle, focal_length, zone_polygons|

### 2.2 Output Destinations

All outputs MUST conform to the locked JSON event schema.

|**Output Type**|**Destination**|**Description**|
|---|---|---|
|`model_a/raw` events|MQTT topic: `sih26187/camera/{cam_id}/model_a/raw`|Real-time events for Model B engines and Orchestrator|
|`training_dataset`|Local filesystem: `/data/training/{cam_id}/`|Cropped detections + labels for Model B retraining|
|`health_status`|MQTT topic: `sih26187/orchestrator/health`|Camera health: FPS, frame drops, darkness, obstruction|

## Section 3: Data Contract & JSON Schema

**Integration Boundary:** Your partner's Model B engines subscribe to your `zone_tag` and `entity_type` outputs to know which camera feed to process and how. Every event you publish to the bus must be schema-valid and include, at minimum:

JSON

```
{
  "event_id": "string (uuid-v4)",
  "event_type": "enum [motion, trigger, animal_detected, system_health, camera_anomaly]",
  "severity": "enum [info, warning, critical, provisional, confirmed]",
  "timestamp": "string (ISO-8601-UTC)",
  "camera_id": "string",
  "zone_tag": "enum [close_range, long_range, warning_zone, intrusion_zone, chokepoint, icp, perimeter]",
  "entity_type": "enum [human, vehicle, animal, animal_cart, unknown]",
  "engine_source": "model_a",
  "entity_id": "string (track_id or null)",
  "confidence": "float (0.0-1.0)",
  "bbox": "[x1, y1, x2, y2] (normalized)",
  "evidence_ref": "string (filepath to saved frame)",
  "metadata": {
    "model_version": "ModelA-v1.0",
    "processing_time_ms": "int",
    "frame_number": "int",
    "trigger_type": "null | climbing | fence_cutting | rapid_approach | zone_violation",
    "confirmation_frames": "int",
    "spoofing_flags": "[string]",
    "zone_tag_method": "pixel_height | calibration"
  },
  "hash": "string (SHA-256 of evidence file)",
  "provisional": "boolean"
}
```

## Section 4: Core Algorithms

### 4.1 Time-Sampling & Differentiated Frame Selection

**Purpose:** Skip frames with minimal change (<0.1% pixel change); select the highest variance frame within a window (e.g., 25 frames / 1s) to maximize detail.

Python

```
def should_process_frame(frame_t, frame_t_minus_1, threshold=0.001):
    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray_t.astype(float) - gray_prev.astype(float)) ** 2) / 255.0
    normalized_mse = mse / (255.0 * 255.0) 
    return normalized_mse > threshold

def select_best_frame(frames_window):
    best_frame, best_ts, best_score = None, None, -1
    for frame, ts in frames_window:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = np.var(gray) # Variance = sharpness/detail
        if score > best_score:
            best_score, best_frame, best_ts = score, frame, ts
    return best_frame, best_ts, best_score
```

### 4.2 Absolute Trigger Detection (Multi-Frame Confirmation)

**Purpose:** Suppress single-frame noise. Trigger MUST persist across 3 frames (600ms to 3s depending on FPS sampling).

Python

```
class TriggerDetector:
    def __init__(self, confirmation_frames=3):
        self.confirmation_frames = confirmation_frames
        self.trigger_buffer = {}  
        
    def check_trigger(self, cam_id, frame, detections, zone_polygons):
        # Implementation details inside full skeleton...
        # 1. Identify current frame triggers (CLIMBING, RAPID_APPROACH, ZONE_VIOLATION)
        # 2. Append to trigger_buffer
        # 3. IF buffer count >= self.confirmation_frames -> Fire Confirmed Trigger
        # 4. Clean buffer > 5 seconds old
        pass
```

### 4.3 Anti-Spoofing Check

**Purpose:** Detect replayed or injected footage based on timestamp/FPS monotonicity.

- **NEGATIVE_GAP:** Possible replay.
    
- **LARGE_GAP (>2s):** Possible injection.
    
- **FPS_ANOMALY:** Timing inconsistent (>20% drift).
    
- _Action:_ Publish event with `warning` severity and `spoofing_flags`. Do NOT suppress the event.
    

### 4.4 Zone Tagging & Animal Filtering

- **Close Range:** Object height >= 200px at 1080p (<=15m). Face/ANPR viable.
    
- **Long Range:** Object height < 200px at 1080p. Trajectory/Posture only.
    
- **Animal Filter:** Identify wildlife (`confidence > 0.6`); log the event but do NOT trigger fence alerts.
    

## Section 5: Testing & Integration Strategy

### 5.1 Your Phase-Specific Test Responsibilities

_(Map to [[06_Version_Control_Testing_Strategy]])_

- **Phase Rome:** Ensure strict JSON schema validation for all outputs.
    
- **Phase Berlin:** Prove multi-frame confirmation successfully suppresses false positive triggers (animal/shadow).
    
- **Phase The Hague:** Event bus integration; verify severity-tag priority under load.
    
- **Phase Oslo:** Fallback routing verification (deliberately kill a Model B engine and confirm your fallback logic engages).
    

### 5.2 Integration Checkpoints with Model B Partner

- **Checkpoint 1 (After Rome):** Confirm both partners can publish/subscribe a dummy JSON event successfully.
    
- **Checkpoint 2 (After Berlin/Paris):** Joint test — Verify your `zone_tag` output correctly routes traffic to their Close Range (Face/ANPR) or Long Range (Trajectory/Posture) modules.
    
- **Checkpoint 3 (After The Hague):** Joint load test on the shared MQTT bus.
    

### 5.3 Performance Targets

Process 1000 frames from test video and measure:

- **Speed:** <50ms per frame on target hardware (Jetson Orin Nano).
    
- **Memory:** <2GB footprint.
    
- **Latency:** MQTT event publish latency < 1 second.
    

## Section 6: Known Limitations & Extension Points

**Known Limitations (Documented for Judges):**

- Model A is deliberately lightweight; it prevents total blindness during fallback but CANNOT fully replace Model B's capabilities.
    
- Zone tagging relies on camera calibration metadata; bad metadata = bad classification.
    
- Multi-frame confirmation adds latency (600ms - 3s). Acceptable for borders, but not for high-speed tracking.
    
- Anti-spoofing checks are basic. Frame-perfect sophisticated spoofing will bypass this.
    

**Extension Points (PROPOSED — Needs Owner Sign-off):**

- **Animal-Cart Tagging:** Animal + wheeled object in proximity = animal-cart tag.
    
- **Perspective Correction:** Homography transform for chokepoint cameras (pre-ANPR).
    
- **Weather-Aware Sampling:** Increase sampling rate during rain/fog when baseline variance naturally rises.
    
- **Adaptive Thresholds:** Learn optimal MSE threshold per camera based on historical variance.
    

## Section 7: Code Skeleton (`model_a.py`)

Python

```
# model_a.py
# SIH26187 — Model A: Module Creator
# Locked specification — do not modify architecture without owner approval

import cv2
import numpy as np
import time
import uuid
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import paho.mqtt.client as mqtt

class ModelA:
    def __init__(self, config):
        self.config = config
        self.camera_states = {}  # cam_id -> state dict
        self.trigger_detector = TriggerDetector(confirmation_frames=3)
        self.anti_spoofing = AntiSpoofing(expected_fps=25)
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(config['mqtt_broker'], config['mqtt_port'])
        
    def process_frame(self, cam_id, frame, timestamp, frame_number, camera_metadata):
        # Initialize camera state if new
        if cam_id not in self.camera_states:
            self.camera_states[cam_id] = {
                'last_frame': None,
                'last_timestamp': 0,
                'frame_buffer': [],
                'sampling_window_size': self.config.get('sampling_window', 25)
            }
        
        state = self.camera_states[cam_id]
        
        # 1. Anti-spoofing check
        is_valid, spoof_issues = self.anti_spoofing.check_frame(cam_id, timestamp, frame_number)
        if not is_valid:
            self.publish_spoofing_alert(cam_id, spoof_issues, timestamp)
        
        # 2. Time-sampling
        if state['last_frame'] is not None:
            if not should_process_frame(frame, state['last_frame'], threshold=0.001):
                state['last_frame'] = frame
                return None  # Skip redundant frame
        
        # 3. Buffer frames for most-differentiated selection
        state['frame_buffer'].append((frame, timestamp))
        if len(state['frame_buffer']) < state['sampling_window_size']:
            state['last_frame'] = frame
            return None  # Wait for window to fill
        
        # 4. Select best frame from window
        best_frame, best_ts, best_score = select_best_frame(state['frame_buffer'])
        state['frame_buffer'] = []  # Clear buffer
        state['last_frame'] = best_frame
        
        # 5. Run lightweight detection (YOLOv8 nano)
        detections = self.run_detection(best_frame)
        
        # 6. Filter animals
        detections = filter_animals(detections)
        
        # 7. Zone tagging and trigger detection
        events = []
        for det in detections:
            zone = tag_zone(det['bbox'], camera_metadata)
            det['zone'] = zone
            
            # Check absolute triggers
            triggers = self.trigger_detector.check_trigger(
                cam_id, best_frame, [det], camera_metadata.get('zone_polygons', {})
            )
            
            for trig in triggers:
                event = self.create_event(
                    cam_id=cam_id,
                    event_type='trigger',
                    severity='critical',
                    zone=zone,
                    entity_type=det['class'],
                    entity_id=det.get('track_id'),
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    trigger_type=trig['type'],
                    evidence_frame=best_frame,
                    metadata={
                        'confirmation_frames': trig['duration_frames'],
                        'spoofing_flags': spoof_issues,
                        'processing_time_ms': int((time.time() - timestamp) * 1000)
                    }
                )
                events.append(event)
                self.publish_event(event)
            
            # Always publish motion event for humans/vehicles
            if det['class'] in ['human', 'vehicle']:
                motion_event = self.create_event(
                    cam_id=cam_id,
                    event_type='motion',
                    severity='info',
                    zone=zone,
                    entity_type=det['class'],
                    entity_id=det.get('track_id'),
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    trigger_type=None,
                    evidence_frame=best_frame,
                    metadata={
                        'spoofing_flags': spoof_issues,
                        'processing_time_ms': int((time.time() - timestamp) * 1000)
                    }
                )
                events.append(motion_event)
                self.publish_event(motion_event)
        
        # 8. Save training data
        self.save_training_data(cam_id, best_frame, detections)
        
        # 9. Publish health status
        self.publish_health(cam_id, timestamp)
        
        return events
    
    def run_detection(self, frame):
        # Placeholder: integrate YOLOv8 nano
        # Returns list of detections: [{class, confidence, bbox, track_id}]
        pass

    def create_event(self, cam_id, event_type, severity, zone, entity_type,
                     entity_id, confidence, bbox, trigger_type, evidence_frame, metadata):
        evidence_path = self.save_evidence(cam_id, evidence_frame)
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'camera_id': cam_id,
            'zone': zone,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'confidence': confidence,
            'bbox': bbox,
            'evidence_ref': evidence_path,
            'metadata': {
                'model_version': 'ModelA-v1.0',
                'engine_name': 'ModelA',
                **metadata
            },
            'hash': self.compute_hash(evidence_path),
            'provisional': trigger_type is not None
        }
        return event

    def publish_event(self, event):
        topic = f"sih26187/camera/{event['camera_id']}/model_a/raw"
        self.mqtt_client.publish(topic, json.dumps(event))

    def publish_spoofing_alert(self, cam_id, issues, timestamp):
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'camera_anomaly',
            'severity': 'warning',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'camera_id': cam_id,
            'zone': 'unknown',
            'entity_type': 'unknown',
            'entity_id': None,
            'confidence': 1.0,
            'bbox': [0, 0, 1, 1],
            'evidence_ref': None,
            'metadata': {
                'model_version': 'ModelA-v1.0',
                'engine_name': 'ModelA',
                'spoofing_flags': issues
            },
            'hash': None,
            'provisional': False
        }
        self.publish_event(event)

    def save_evidence(self, cam_id, frame):
        path = Path(f"/data/evidence/{cam_id}/{datetime.now():%Y%m%d%H%M%S%f}.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return str(path)

    def save_training_data(self, cam_id, frame, detections):
        # Save cropped detections with labels for Model B training
        for det in detections:
            x1, y1, x2, y2 = [int(v * dim) for v, dim in zip(det['bbox'], frame.shape[:2][::-1] * 2)]
            crop = frame[y1:y2, x1:x2]
            label = det['class']
            path = Path(f"/data/training/{cam_id}/{label}/{datetime.now():%Y%m%d%H%M%S_%f}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), crop)

    def compute_hash(self, filepath):
        if filepath is None:
            return None
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def publish_health(self, cam_id, timestamp):
        # Simplified: publish FPS and status
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'system_health',
            'severity': 'info',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'camera_id': cam_id,
            'zone': 'unknown',
            'entity_type': 'system',
            'entity_id': 'model_a',
            'confidence': 1.0,
            'bbox': [0, 0, 1, 1],
            'evidence_ref': None,
            'metadata': {
                'model_version': 'ModelA-v1.0',
                'engine_name': 'ModelA',
                'status': 'healthy'
            },
            'hash': None,
            'provisional': False
        }
        self.mqtt_client.publish(f"sih26187/orchestrator/health", json.dumps(event))

# Helper functions 
def should_process_frame(frame_t, frame_t_minus_1, threshold=0.001):
    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray_t.astype(float) - gray_prev.astype(float)) ** 2)
    normalized_mse = mse / (255.0 * 255.0)
    return normalized_mse > threshold

def select_best_frame(frames_window):
    best_frame, best_ts, best_score = None, None, -1
    for frame, ts in frames_window:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = np.var(gray)
        if score > best_score:
            best_score = score
            best_frame = frame
            best_ts = ts
    return best_frame, best_ts, best_score

def tag_zone(bbox, camera_metadata):
    # Simplified: use pixel height threshold
    obj_height = bbox[3] - bbox[1]
    threshold = camera_metadata.get('close_range_threshold', 0.185)  # ~200px/1080p
    return 'close_range' if obj_height >= threshold else 'long_range'

def filter_animals(detections, threshold=0.6):
    return [d for d in detections if not (d['class'] == 'animal' and d['confidence'] > threshold)]

def point_in_polygon(point, polygon):
    # polygon: list of (x, y) normalized coordinates
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0
```

**Document Control**

**Owner:** Trajectory/Posture Track Owner.

**Reviewers:** Technical Lead, Integration Owner.

**Approval Required For:** Any change to core algorithms, any change to JSON schema, any change to trigger definitions.

**Change Log:** v1.0 — 2026-08-31 — Initial Model A Developer Guide and Work Delegation merged.