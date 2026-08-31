# Model B Developer Guide & Work Delegation — SIH26187

**Title:** Engine Suite (Face, ANPR, Posture, Trajectory) & Orchestration Layer **Version:** 1.0 | 2026-08-31 **Classification:** SIH Internal Round — Top 50 Qualifier **Status:** **LOCKED SPECIFICATION**

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[03_Architecture_Map]] · [[06_Version_Control_Testing_Strategy]] · [[07A_Model_A_Delegation]] **Purpose:** Complete implementation guide for Model B's partner delegation: the four independent AI engines (Face, ANPR, Posture, Trajectory), the Single Orchestration Line (threat-scoring), and the Human Orchestrator Dashboard.

## Section 1: Scope & Responsibilities (Locked)

You own the **Model B Engine Suite** and the **Orchestrator Layer**. Your explicit responsibilities include:

- **Close Range Module:** Face Engine (InsightFace/ArcFace) and ANPR Engine (PaddleOCR).
    
- **Long Range Module:** Trajectory Engine (YOLO+ByteTrack) and Posture Engine (MediaPipe Pose). _Note: Vehicle detection/classification lives here, feeding off Trajectory._
    
- **Single Orchestration Line:** Threat-scoring (combining posture + zone + time + trajectory) and the Border Context Profile.
    
- **Orchestrator Dashboard:** Left panel (live video/metadata) and Right panel (alerts/health/fallback). Must assume low technical literacy.
    
- **Design Principle:** All engines are INDEPENDENT. No engine waits for another. Each subscribes to `model_a/raw` events relevant to its scope, processes asynchronously, and publishes results.
    

### 🚫 What You Must NOT Do

- **DO NOT** touch Model A's ingestion, time-sampling, or bus-publishing logic.
    
- **DO NOT** build a "hidden" extra engine. 5 models total (Model A + 4 Model B engines) is a locked, protected claim.
    
- **DO NOT** run ANPR on non-chokepoint cameras. This is an expected scope limitation, not a bug to "fix" by ignoring Model A's zone classification.
    

## Section 2: Data Contracts & Event Bus Integration

### 2.1 The Integration Boundary (Incoming)

You subscribe to Model A's bus output (`engine_source: "model_a"`) and route traffic based on:

- `zone_tag`: Route `close_range` → Face/ANPR. Route `long_range` → Trajectory/Posture/Vehicle.
    
- `entity_type`: Skip irrelevant processing (e.g., do not run Face Engine on an `entity_type: vehicle` event).
    

### 2.2 The Integration Boundary (Outgoing)

Every engine's output must be schema-valid (`engine_source` set to your engine name), so the Orchestration Line can combine them. **Confidence scores must be genuinely calibrated** per engine — do not default everything to a flat value, since threat-scoring depends on relative confidence across engines.

### 2.3 Event Bus Architecture (MQTT)

Plaintext

```
Unified Event Bus (MQTT)
    │
    ├── Topic: sih26187/camera/{cam_id}/model_b/face       ← Face Engine
    ├── Topic: sih26187/camera/{cam_id}/model_b/anpr       ← ANPR Engine
    ├── Topic: sih26187/camera/{cam_id}/model_b/posture    ← Posture Engine
    └── Topic: sih26187/camera/{cam_id}/model_b/trajectory ← Trajectory Engine
```

## Section 3: Trajectory Engine (Build This First)

**Priority:** Highest-leverage component. Build this first within your track, since it feeds vehicle-detection context and proposed extensions (package-drop, group-coordination).

- **Purpose:** Track entities across frames, maintain persistent IDs, calculate velocity/direction, detect zone transitions, and identify behaviors (loitering, rapid approach).
    
- **Tech Stack:** YOLOv8 (shared with Model A), ByteTrack (primary) or DeepSORT (fallback). Kalman Filter for occlusion prediction.
    
- **Target:** <50ms per track update on Jetson Orin Nano. ID switch rate <5%.
    
- **Known Limitations:** Occlusions >10s likely lose track ID. Crowded scenes may cause Hungarian algorithm mismatches.
    

Python

```
class TrajectoryEngine:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        from ultralytics import YOLO
        from byte_tracker import BYTETracker 
        self.detector = YOLO('yolov8n.pt')
        self.tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        self.tracks = {}  # track_id -> Track object
        self.max_trajectory_length = 100
    
    def process(self, frame, detections, camera_id, timestamp, camera_metadata):
        # Format detections for ByteTrack: [x1, y1, x2, y2, confidence, class]
        # Update tracker -> Calculate velocity/direction -> Check zone transitions
        # Analyze behavior (loitering, rapid approach) -> Publish trajectory_update
        pass

    def calculate_velocity_direction(self, trajectory):
        # Use last 5 points for smoothing to calculate pixels per second & direction (degrees)
        pass

    def analyze_behavior(self, track, camera_metadata):
        # Loitering: low velocity + high position variance + duration > 60s
        # Rapid approach: high velocity moving toward intrusion zone
        pass
```

## Section 4: Posture Engine

- **Purpose:** Estimate human body pose (33 landmarks), classify into 6 categories (standing, walking, running, crouching, crawling, carrying).
    
- **Scope Note:** Posture _alone_ only detects "this looks like crawling" — it is NOT complete suspicious-activity detection until combined with the Border Context Profile in the Orchestrator.
    
- **Tech Stack:** MediaPipe Pose (10x lighter than OpenPose, edge-capable). Rule-based or lightweight MLP classifier.
    
- **Target:** <80ms per person.
    

Python

```
class PostureEngine:
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    
    def process(self, frame, bbox, track_id, camera_id, timestamp):
        # Crop person -> Extract 33 landmarks -> Classify posture 
        # Calculate base anomaly score -> Save skeleton visualization
        pass

    def classify_posture(self, landmarks):
        # Torso angle vs Body height ratios
        # e.g., height_ratio < 0.3 and nose_y > hip_y => crawling (0.9 conf)
        pass
```

## Section 5: Face Engine

- **Purpose:** Detect faces, extract 512-dim embedding vectors, match against watchlist.
    
- **Scope (Locked):** Close-range cameras ONLY. Long-range face ID explicitly OUT OF SCOPE.
    
- **Tech Stack:** InsightFace RetinaFace (detector) + ArcFace `iresnet100` or `mobilefacenet` (embedding). Cosine similarity against SQLite/vector DB.
    
- **Target:** <100ms per face on edge hardware. Watchlist search <50ms.
    
- **Privacy Rules:** Non-matched embeddings stored for 30 days then purged. Matches retained indefinitely for legal audit.
    

Python

```
class FaceEngine:
    def __init__(self, model_path, watchlist_db_path, match_threshold=0.6):
        import insightface
        self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.recognizer = insightface.model_zoo.get_model('arcface_r100_v1')
        self.watchlist = self.load_watchlist(watchlist_db_path)
    
    def process(self, frame, bbox, camera_id, timestamp, parent_event_id):
        # Expand bbox -> crop face -> detect landmarks -> extract embedding
        # Cosine similarity against watchlist -> generate event
        pass
```

## Section 6: ANPR Engine

- **Purpose:** Detect plates, correct perspective, recognize multi-script text (Latin + Devanagari).
    
- **Scope (Locked):** Chokepoint / ICP cameras ONLY. Do not process high-mounted perimeter feeds.
    
- **Tech Stack:** YOLOv8 (plate detector) + PaddleOCR (multi-script) or custom CRNN.
    
- **Target:** <150ms per plate read. OCR accuracy >70% on Indian multi-format plates.
    

Python

```
class ANPREngine:
    def __init__(self, detector_path, ocr_model_path):
        from ultralytics import YOLO
        import paddleocr
        self.plate_detector = YOLO(detector_path)
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    
    def process(self, frame, vehicle_bbox, camera_id, timestamp, calibration=None):
        # Crop vehicle -> detect plate -> apply homography (perspective correction)
        # PaddleOCR extract text -> normalize string -> validate state code format
        pass
```

## Section 7: Single Orchestration Line & Multi-Camera Fusion

### 7.1 Threat-Scoring & Border Context Profile

- **Logic:** Combine posture + zone + time + trajectory into a final Threat Score (0-100).
    
- **Border Context Profile:** Prioritize building 1-2 profiles (BSF & SSB) once engines are stable. _This is the most important unbuilt piece in the project._ It changes score thresholds (e.g., presence on SSB border is tolerated; presence on BSF border is flagged).
    
- **Proposed Extensions:** Package-drop detection (object separates from person-track) and Group-coordination (3+ tracks, same direction). Both reuse Trajectory output.
    

### 7.2 Multi-Camera Fusion

- **Purpose:** Prevent same entity tracking as two separate weaker entities across cameras.
    
- **Algorithm:** Greedy merge by highest score combining Temporal gap (exit/entry time match) + Spatial proximity (exit/entry zone overlap) + Appearance similarity.
    

Python

```
class MultiCameraFusion:
    def fuse(self, camera_events):
        # Find candidates across overlapping cameras
        # Greedy merge by highest score (Spatial + Temporal check)
        # Yield unified global trajectory_update event
        pass
```

## Section 8: Dashboard & Engine Fallback

### 8.1 Orchestrator Dashboard

- **Left Panel:** Live detected video RT, metadata, pipeline-flow status.
    
- **Right Panel:** Flagging/control, pipeline health monitor, engine health monitor + FALLBACK indicator.
    
- **Constraint:** Operable by non-technical jawans. Usability-test on a non-teammate before demo day. Single-click dismiss for alerts.
    

### 8.2 Health Monitoring & Fallback Logic

- **Heartbeat:** Every Model B engine publishes a heartbeat every 10s to `sih26187/orchestrator/health`.
    
- **Fallback Trigger:** If heartbeat missing >30s:
    
    1. Orchestrator tags engine as `DEGRADED`.
        
    2. Camera traffic routes through Model A's safety floor (motion+trigger only).
        
    3. Dashboard shows **FALLBACK ACTIVE** indicator.
        
    4. No auto-restart (prevents loops).
        

## Section 9: JSON Engine Metadata Additions

All engines use the base schema (`schema_v1`), inserting engine-specific metrics into the `metadata` object:

- **Face:** `face_detector`, `embedding_model`, `match_threshold`, `watchlist_size`.
    
- **ANPR:** `plate_detector`, `ocr_model`, `perspective_corrected`, `format_validated`.
    
- **Posture:** `pose_model`, `classifier`, `landmark_count`.
    
- **Trajectory:** `tracker`, `max_track_age`, `kalman_enabled`.
    

## Section 10: Testing & Integration Strategy

### 10.1 Your Phase-Specific Test Responsibilities

_(Map to [[06_Version_Control_Testing_Strategy]])_

- **Phase Paris:** Per-engine precision/recall spot-check. Verify ANPR is scoped _strictly_ to chokepoints.
    
- **Phase Oslo:** Dashboard fallback-status display and health monitor accuracy.
    
- **Phase Tokyo:** Border Context Profile testing — prove that the same exact input yields a different threat score depending on the active profile.
    

### 10.2 Integration Checkpoints with Model A Partner

- **Checkpoint 1 (After Rome):** Confirm you can subscribe to a dummy Model A JSON event.
    
- **Checkpoint 2 (After Berlin/Paris):** Joint test — Model A's `zone_tag` correctly routes traffic to your Close Range or Long Range modules.
    
- **Checkpoint 3 (After The Hague):** Joint load test on the shared bus (4 engines running concurrently).
    
- **Checkpoint 4 (After Oslo):** Joint fallback test — deliberately kill one of your engines, confirm dashboard + Model A safety floor both react correctly.
    

**Document Control** **Owner:** Face/ANPR Track Owner & Trajectory/Posture Track Owner. **Reviewers:** Technical Lead, Integration Owner. **Approval Required For:** Any change to engine algorithms, model weights, or scope boundaries (e.g., adding new engines). **Change Log:** v1.0 — 2026-08-31 — Initial Model B Developer Guide and Work Delegation merged.