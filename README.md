# SIH26187 — Jai Jawan: AI-Based Intelligent Video Analytics Platform
## Ministry of Home Affairs | Sashastra Seema Bal (SSB) | Police II Division

> **Software-only fix** for the infrastructure that ₹86Cr (BOLD-QIT) and CIBMS couldn't deliver.
> Runs on existing IP CCTV + edge compute. No dedicated FRS/ANPR/smart-camera hardware.

---

## Project Structure

```
Jai-jawan/
├── model_a/                   ← Gatekeeper Engine (Model A)
│   ├── __init__.py
│   ├── schema_v1.py           ← Locked JSON event schema (Pydantic v2)
│   ├── bus_client.py          ← MQTT pub/sub (single topic, QoS-by-severity, LWT)
│   ├── time_sampler.py        ← MSE frame dedup + MDF selection (25→1-5 FPS)
│   ├── zone_tagger.py         ← close_range / long_range classification
│   ├── trigger_detector.py    ← Multi-frame state machine (IDLE→PROV→CONFIRMED)
│   ├── anti_spoofing.py       ← Stream integrity: timestamp, FPS, frame continuity
│   ├── health_monitor.py      ← Darkness, frozen, obstruction, Model B heartbeat
│   ├── fusion_engine.py       ← Multi-camera IoU entity merger
│   ├── preprocessor.py        ← Zero-DCE ONNX + CLAHE/Retinex low-light
│   ├── detector.py            ← YOLOv8n + ByteTrack wrapper + MockDetector
│   ├── animal_filter.py       ← Animal→info events; human/vehicle→trigger pipeline
│   ├── bbox_consistency.py    ← Shadow/foliage suppressor (IoU drift per track)
│   └── frame_pipeline.py      ← Per-frame orchestrator (all 12 steps wired)
│
├── tests/
│   ├── test_phase_rome.py     ← 36 tests: schema, state machine, sampling, zones
│   └── test_phase_berlin.py   ← 25 tests: false-trigger suppression, animal filter
│
├── docs/
└── pyproject.toml
```

---

## Non-Negotiable Architecture Rules

| Rule | Enforcement |
|------|-------------|
| **Multi-frame confirmation ≥3** | `TriggerDetector.__init__` raises `ValueError` if `confirmation_frames < 3`. `ModelAEvent` model validator rejects `severity=confirmed/critical` if `confirmation_frames < 3`. |
| **One event bus, no bypass** | `BusClient.publish_event()` is the only egress path. Topic pattern is hardcoded: `sih26187/camera/{cam_id}/model_a/event` |
| **schema_v1 frozen** | Pydantic `ModelAEvent` validates before `to_mqtt_payload()`. Invalid → `ValidationError` raised and logged. Never hits the wire. |
| **4 engines in Model B** | `model_a/` touches none of them. Integration contract is one-way: publish → they consume. |

---

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| **Rome** | Scaffolding + Schema validation + Bus round-trip | ✅ **36/36 PASSED** |
| **Berlin** | False-trigger suppression + Animal filter + 2-frame boundary | ✅ **25/25 PASSED** |
| **The Hague** | Bus load test — CRITICAL surfaces under routine-event flood | ✅ **19/19 PASSED** |
| Oslo | Fallback routing (dead Model B heartbeat) | 🔜 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Run unit tests (no broker needed)
pytest tests/test_phase_rome.py -v -k "not TestMQTTRoundTrip"

# 3. Run integration tests (requires Mosquitto on localhost:1883)
docker run -d -p 1883:1883 eclipse-mosquitto
pytest tests/test_phase_rome.py -v
```

---

## MQTT Topic Map

```
sih26187/camera/{cam_id}/model_a/event   ← Model A publishes ALL events here
sih26187/system/model_a/health           ← LWT fires here if Model A dies
sih26187/camera/{cam_id}/model_b/face    ← Model B Face engine (subscribe: close_range)
sih26187/camera/{cam_id}/model_b/anpr    ← Model B ANPR (subscribe: chokepoint/icp)
sih26187/camera/{cam_id}/model_b/posture ← Model B Posture (subscribe: long_range)
sih26187/camera/{cam_id}/model_b/traj   ← Model B Trajectory (subscribe: long_range)
```

---

## Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| Per-frame latency | <50ms | YOLOv8n, ONNX/TRT for Zero-DCE |
| RAM usage | <2GB | Queue-based I/O decoupling |
| Effective FPS | 1-5 | MSE time-sampler |
| End-to-end critical latency | <5s | QoS 2 on confirmed/critical |
