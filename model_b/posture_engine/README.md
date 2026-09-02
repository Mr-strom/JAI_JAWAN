# Posture Engine — SIH26187

Part of **Model B Engine Suite** | Both Close-Range and Long-Range.

## What it does

Subscribes to Model A's raw events, filters to `entity_type == "human"` (any zone_tag),
reads the saved frame from `evidence_ref`, crops the person using the provided bbox,
runs **MediaPipe Pose** to extract 33 body landmarks, classifies the posture into one
of 6 categories using a rule-based classifier, and publishes a `posture_anomaly` event.

**Scope note:** `anomaly_score` is a raw posture signal only — it is NOT a threat score.
The Orchestrator's Border Context Profile decides what's suspicious based on zone, time,
and trajectory context. This engine just says "this person is crawling", not "this is a threat".

## File Map

```
posture_engine/
├── config.py           ← All tuneable knobs (MQTT, MediaPipe settings, rule thresholds)
├── posture_engine.py   ← Core: MediaPipe Pose + rule classifier + event builder
├── mqtt_bridge.py      ← MQTT subscribe/filter/publish + 10s heartbeat
└── requirements.txt
```

## Quick Start

```bash
# 1. Create and activate venv
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Make sure Mosquitto is running on localhost:1883

# 4. Run
python mqtt_bridge.py
```

## Posture Classes & Rule Logic

| Class | Key Rule |
|-------|----------|
| `crawling` | height/width ratio < 0.7 OR (nose below hips AND torso angle > 55°) |
| `crouching` | height/width ratio < 1.0 AND torso angle > 25° |
| `carrying` | upright box AND wrist held well below hip level |
| `running` | upright box AND ankle spread > 0.18 (wide leg stride) |
| `walking` | upright box AND ankle spread > 0.06 |
| `standing` | default — upright box, close feet |

Rules apply in this priority order (most-distinctive first).

## Anomaly Scores (base values, configurable in config.py)

| Class | anomaly_score |
|-------|--------------|
| standing | 0.05 |
| walking | 0.10 |
| running | 0.35 |
| carrying | 0.45 |
| crouching | 0.60 |
| crawling | 0.85 |

These are raw signals. The Orchestrator combines them with zone, time, and trajectory.

## Configuration

All thresholds are in `config.py`. Key ones:

| Key | Default | Meaning |
|-----|---------|---------|
| `HEIGHT_RATIO_CRAWL_MAX` | 0.7 | crop h/w below this → likely crawling |
| `TORSO_ANGLE_CRAWL_MIN` | 55° | torso angle above this → horizontal torso |
| `ANKLE_SPREAD_RUNNING_MIN` | 0.18 | normalised ankle x-distance for running stride |
| `MP_MODEL_COMPLEXITY` | 1 | 0=lite, 1=full, 2=heavy |

## Known Limitations

- Single-frame mode: no inter-frame velocity — ankle spread used as a stride proxy for running/walking.
- If `evidence_ref` file doesn't exist (e.g. Model A hasn't written it yet), event is silently skipped.
- MediaPipe may miss pose on very small or occluded figures at long range — expected.
