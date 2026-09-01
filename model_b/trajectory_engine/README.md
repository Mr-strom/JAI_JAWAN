# Trajectory Engine — SIH26187

Part of **Model B Engine Suite** | Long-Range Module.

## What it does

Subscribes to Model A's raw events, filters to `zone_tag == "long_range"`, runs its own
YOLOv8n detection + ByteTrack multi-object tracking on the live RTSP camera feed, and
publishes a `trajectory_update` event per tracked entity.

**Assigns persistent `entity_id`** (track_id) — Model A never assigns one for long_range events.

## File Map

```
trajectory_engine/
├── config.py               ← All tuneable knobs (MQTT, model paths, thresholds)
├── schemas.py              ← Pydantic v2 output schema — validated before publish
├── trajectory_engine.py    ← Core: YOLO + ByteTrack + behavior + confidence
├── mqtt_bridge.py          ← MQTT subscribe/filter/publish + heartbeat
├── camera_stream.py        ← Per-cam lazy RTSP reader (background grab thread)
├── api.py                  ← FastAPI wrapper — GET /health endpoint
├── camera_config.json      ← Zone polygons + RTSP URLs per cam (edit this!)
├── utils/
│   ├── geometry.py         ← Velocity, direction, zone hit-test, smoothness
│   └── hash_util.py        ← SHA-256 for evidence file integrity
└── requirements.txt
```

## Quick Start

```bash
# 1. Create and activate the virtual environment (scoped to this engine)
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install deps into the venv
pip install -r requirements.txt

# 3. Edit camera_config.json — add your cam RTSP URLs and zone polygons

# 4. Make sure Mosquitto (or any MQTT broker) is running on localhost:1883

# 5a. Run as plain MQTT engine (no HTTP)
python mqtt_bridge.py

# 5b. Run as FastAPI microservice (MQTT runs in background, HTTP on :8000)
uvicorn api:app --host 0.0.0.0 --port 8000
# Health check: http://localhost:8000/health
```

For demo mode without a real camera, set `rtsp_url` in `camera_config.json` to a local
video file path (e.g. `"test_footage.mp4"`).

## Configuration

All knobs live in `config.py`. Key ones:

| Key | Default | Meaning |
|-----|---------|---------|
| `LOITER_VELOCITY_THRESHOLD_PX_S` | 15.0 | px/s below which entity is "slow" |
| `LOITER_VARIANCE_THRESHOLD` | 20.0 | position variance threshold for loiter |
| `LOITER_DURATION_S` | 60.0 | seconds before loitering tag fires |
| `RAPID_APPROACH_VELOCITY_THRESHOLD_PX_S` | 80.0 | px/s for rapid approach |
| `TRACK_AGE_STABLE_FRAMES` | 30 | frames for track_age_ratio to reach 1.0 |
| `MAX_TRACK_AGE` | 30 | frames before ByteTrack drops a lost track |

## Blended Confidence Formula

```
confidence = detection_conf × track_age_ratio × movement_smoothness
```

- `detection_conf` — raw YOLO detection confidence
- `track_age_ratio` = min(frames_seen / TRACK_AGE_STABLE_FRAMES, 1.0)
- `movement_smoothness` = 1 - std(step_distances) / MAX_EXPECTED_DELTA_PX

## Output Schema

Published to `sih26187/camera/{cam_id}/model_b/trajectory`. Schema is in `schemas.py`.

## Known Limitations (from spec)

- Occlusions > 10s may lose track ID — accepted, not worked around.
- Crowded scenes may cause Hungarian algorithm mismatches in ByteTrack.
- No close_range processing — this engine ignores those events by design.

## Tests

Run geometry helpers standalone (no YOLO needed):

```bash
python -c "
from utils.geometry import calculate_velocity_direction, movement_smoothness, is_point_in_polygon
pts = [(0,0),(1,1),(2,2),(3,3),(4,4)]
print(calculate_velocity_direction(pts, fps=25))
print(movement_smoothness(pts))
print(is_point_in_polygon((2,2), [(0,0),(5,0),(5,5),(0,5)]))
"
```
