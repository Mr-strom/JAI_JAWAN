# Phase Plan & Development Roadmap — SIH26187

**Title:** Development Roadmap (Money Heist Codenames) **Version:** 1.0 | 2026-08-31 **Classification:** SIH Internal Round — Top 50 Qualifier **Status:** **ACTIVE PLAN — TRACK PROGRESS AFTER EACH PHASE**

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[03_Architecture_Map]] · [[04_Continuity_Protocol]] · [[05_Project_Handbook]] **Purpose:** Transform abstract architecture into concrete, time-boxed development phases. Each phase is named after a Money Heist city to make progress memorable and team morale high. Phases are ordered by dependency — **do not skip phases**.

## Phase 0: The Professor — Planning & Setup (Pre-Phase)

- **Duration:** 3 days | **Owner:** Project Owner + Technical Lead
    
- **Deliverables:**
    
    - Master Handoff document finalized and distributed.
        
    - Team roles assigned: Face/ANPR track owner, Trajectory/Posture track owner, Dashboard/Integration owner, Pitch lead.
        
    - Development environment set up: Ubuntu 22.04, Python 3.10, CUDA, Docker, Git repo.
        
    - Hardware decision: Jetson Orin Nano ordered OR laptop-with-GPU fallback confirmed.
        
    - Dataset inventory: Verify licensing for UNIRI-TID, PDIWS, SCface, NightOwls, CAMEL, CVC-14.
        
    - Demo strategy decided: Plan A/B/C/D selected based on hardware availability.
        
- **Success Criteria:** Every team member can run `git clone` and `docker-compose up` without errors.
    

## Phase 1: Tokyo — Foundation & Infrastructure

- **Duration:** 5 days | **Owner:** Dashboard/Integration Owner | **Dependencies:** Phase 0 complete
    
- **Theme:** _Tokyo is the heartbeat of the heist — reliable, fast, keeps everyone connected. This phase builds the infrastructure that everything else depends on._
    

**Tasks:**

1. **JSON Event Schema:** Final version locked with all field names, enums, validation rules.
    
2. **MQTT Broker:** Mosquitto installed, configured, topics created, QoS levels set.
    
3. **RTSP Ingestion:** `ffmpeg` or OpenCV-based RTSP capture from 1 test camera/stream.
    
4. **Project Skeleton:** Directory structure, Docker Compose, `requirements.txt`, config files.
    
5. **Basic Dashboard:** Plain HTML/JS page with WebSocket connection to MQTT; can display raw JSON events.
    
6. **Camera Health Monitor:** Basic ping/FPS check, publishes health events.
    
7. **Git Workflow:** Branching strategy, commit message format, PR review process.
    

**Deliverables:**

- `docker-compose up` starts MQTT broker + RTSP simulator + basic dashboard.
    
- Dashboard shows raw MQTT messages in real-time.
    
- Camera health status visible (green/red indicator).
    
- **Success Criteria:** Team can publish a test event from the command line and see it on the dashboard within 1 second.
    

## Phase 2: Nairobi — Core Detection & Tracking

- **Duration:** 7 days | **Owner:** Trajectory/Posture Track Owner | **Dependencies:** Phase 1 complete
    
- **Theme:** _Nairobi is the forger — creates identities, tracks people, makes the fake real. This phase builds the core AI engines that detect and track humans._
    

**Tasks:**

1. **YOLOv8 Setup:** Install `ultralytics`, download pretrained weights (`yolov8n.pt` for speed).
    
2. **ByteTrack Integration:** Tracking pipeline: detect → track → assign ID → persist.
    
3. **Trajectory Engine v1:** Kalman filter tracking, velocity/direction calculation, zone transition detection.
    
4. **MediaPipe Pose Setup:** Install `mediapipe`, test on sample images.
    
5. **Posture Engine v1:** 6-class classifier: standing, walking, running, crouching, crawling, carrying.
    
6. **Model A Skeleton:** Frame ingestion, time-sampling, basic motion detection.
    
7. **Virtual Fence v1:** Two-zone geofence, line-crossing detection, basic alert.
    

**Deliverables:**

- Single-camera human tracking: person walks across frame, consistent track ID.
    
- Posture classification: crawling person correctly labeled.
    
- Virtual fence alert: line crossing triggers dashboard alert.
    
- **Success Criteria:** >80% detection rate on test footage, <10% ID switch rate, virtual fence alert latency <5s.
    

## Phase 3: Berlin — Integration & Context Reasoning

- **Duration:** 7 days | **Owner:** Dashboard/Integration Owner | **Dependencies:** Phase 2 complete
    
- **Theme:** _Berlin is the planner — connects everything, makes the pieces work together, thinks 10 steps ahead. This phase integrates engines and adds the reasoning layer._
    

**Tasks:**

1. **Model A Complete:** Multi-frame confirmation logic, zone tagging, absolute trigger detection, anti-spoofing check.
    
2. **Event Bus Integration:** All engines publish to correct MQTT topics with valid JSON schema.
    
3. **Orchestrator v1:** Consumes all engine events, basic threat scoring (posture + zone + time).
    
4. **Border Context Profile v1:** 2 sample profiles (BSF strict and SSB tolerant). Time-of-day + zone lookup table.
    
5. **Multi-Camera Fusion:** Track identity resolution across 2 cameras (spatial overlap + temporal consistency).
    
6. **Dashboard v2:** LEFT panel (live video + metadata), RIGHT panel (alert list + health monitor + fallback indicator).
    
7. **Alert System:** Severity-based coloring, acknowledgment flow, evidence image display.
    

**Deliverables:**

- 2-camera setup: Person tracked across Camera A and Camera B as a single identity.
    
- BSF profile: Crawling at night near fence = critical alert.
    
- SSB profile: Walking at morning on farm path = no alert, logged only.
    
- Dashboard shows live video, alerts, health status.
    
- **Success Criteria:** End-to-end latency <10s. Multi-camera fusion prevents duplicate tracks. Context profile changes alert behavior demonstrably.
    

## Phase 4: Moscow — Specialized Engines & Testing

- **Duration:** 7 days | **Owner:** Face/ANPR Track Owner | **Dependencies:** Phase 3 complete
    
- **Theme:** _Moscow is the muscle — heavy lifting, specialized skills, gets the hard jobs done. This phase builds the specialized close-range engines and tests everything._
    

**Tasks:**

1. **InsightFace Setup:** Install `insightface`, download ArcFace model, test detection + embedding.
    
2. **Face Engine v1:** Detect face, extract 512-dim embedding, match against SQLite watchlist.
    
3. **Watchlist Management:** Add/remove faces, match threshold tuning, unknown face handling.
    
4. **ANPR Pipeline:** YOLO plate detector + PaddleOCR recognition + format validation.
    
5. **Perspective Correction:** Basic homography transform for chokepoint cameras (if team has skills).
    
6. **Load Testing:** 4 camera feeds sustained for 1 hour, memory leak check, thermal throttling.
    
7. **Scenario Testing:** Daylight, night, rain, fog, occlusion test footage. Metrics: detection rate, false positive rate, latency.
    

**Deliverables:**

- Face detection + watchlist match working on close-range test footage.
    
- ANPR reading plates on chokepoint test footage.
    
- Load test report: Memory stable, no crashes, latency within targets.
    
- Scenario test matrix with actual numbers (even if below target, document honestly).
    
- **Success Criteria:** Face detection >85% on close-range. ANPR >70% on clean plates. System stable for 1 hour under load.
    

## Phase 5: Denver — Demo Preparation & Polish

- **Duration:** 5 days | **Owner:** Pitch Lead + All Team | **Dependencies:** Phase 4 complete
    
- **Theme:** _Denver is the wildcard — unpredictable, high energy, makes the impossible happen. This phase makes the demo shine and handles edge cases._
    

**Tasks:**

1. **Demo Script:** 7-minute pitch structure finalized, speaker notes, transition cues.
    
2. **Demo Footage:** Record/stage all required scenarios: daylight tracking, night crawling, vehicle chokepoint, multi-camera fusion, fallback mode.
    
3. **Dashboard Polish:** Color scheme (red=critical, yellow=warning, green=normal), large buttons, minimal text, icon-driven.
    
4. **Fallback Demo:** Kill one engine during demo, show Model A safety floor activating.
    
5. **Health Monitor Demo:** Simulate camera obstruction, show health alert.
    
6. **Pitch Deck:** 10-15 slides (Problem, Solution, Architecture, Screenshots, Differentiation, Roadmap, Team).
    
7. **Judge Q&A Rehearsal:** Team practices all 10 prepared Q&A responses + emergency response protocol.
    
8. **Bug Bash:** 2 days of finding and fixing demo-breaking bugs only. No new features.
    

**Deliverables:**

- Smooth 7-minute demo with no crashes.
    
- Pitch deck complete and reviewed.
    
- Team can answer all 10 Q&A without hesitation.
    
- Known bugs documented, but the demo path avoids them.
    
- **Success Criteria:** Demo runs 5 times in a row without failure. Pitch timing strictly within 7+3 minutes.
    

## Phase 6: Rio — Finals Preparation (If Qualified)

- **Duration:** 14 days | **Owner:** All Team | **Dependencies:** Internal round qualified
    
- **Theme:** _Rio is the escape — the final push to freedom. This phase scales the prototype to a finals-worthy system._
    

**Tasks:**

1. **Hardware Benchmark:** Jetson Orin Nano arrives, benchmark all 5 models in parallel.
    
2. **Border Context Profile Full:** Implement all 4 profiles (BSF, SSB, ITBP, Assam Rifles).
    
3. **Animal-Cart Detection:** Labeled dataset collected, model trained, integrated.
    
4. **Package-Drop Detection:** Object separation + static detection logic.
    
5. **Group-Coordination Detection:** 3+ entities, same zone, same direction, time window.
    
6. **Thermal Camera Pathway:** Integration design for optional thermal input.
    
7. **Privacy Policy:** DPDP Act 2023 compliance framework, retention policy, consent mechanism.
    
8. **Model Retraining Pipeline:** Automated collection of flagged events, annotation interface, retrain, A/B test, deploy.
    
9. **Multi-Site Simulation:** 10+ camera feeds, sector HQ dashboard, alert escalation chain.
    
10. **Mobile App (Optional):** Field commander alert receiver.
    

**Deliverables:**

- Finals demo with 10+ cameras, all 4 border profiles, animal-cart detection.
    
- Benchmark report: Actual numbers on target hardware.
    
- Privacy compliance documentation & Retraining pipeline demonstration.
    
- **Success Criteria:** All 8 official capabilities covered. Benchmark numbers validate or honestly revise claims. System demo'd at scale.
    

## Phase 7: Stockholm — Post-Hackathon (Real Deployment)

- **Duration:** Ongoing | **Owner:** Project Owner + Future Team | **Dependencies:** Finals complete
    
- **Theme:** _Stockholm is the peace — the heist is over, now build something that lasts._
    

**Tasks:**

1. **Pilot Deployment:** 1 BOP with SSB or BSF, real cameras, real conditions.
    
2. **Field Feedback:** Jawans use system for 30 days, feedback collected, issues logged.
    
3. **Iteration:** Fix field-discovered issues, retrain models on real border footage.
    
4. **Scale:** Expand to sector level (10-20 BOPs), central command dashboard.
    
5. **Integration:** NCRB/CCTNS watchlist sync, external alert channels (satellite SMS, WhatsApp).
    
6. **AI Camera Placement:** Recommend optimal camera angles and cheap IR illuminator positions.
    
7. **Publication:** Paper or case study on border-specific AI analytics.
    

**Deliverables:**

- Operational system at 1+ real BOP.
    
- Published case study or technical paper.
    
- Sustainable maintenance and retraining process.
    
- **Success Criteria:** Jawans independently operate system. Alert rate <2/night. Contraband seizure attributed to system alert.
    

## Progress Tracker

_Update this table after every team meeting. Mark statuses as NOT STARTED, IN PROGRESS, BLOCKED, or COMPLETE._

|Phase|Codename|Status|Start|End|Owner|Blockers|
|---|---|---|---|---|---|---|
|**0**|Professor|NOT STARTED|—|—|Project Owner|—|
|**1**|Tokyo|NOT STARTED|—|—|Integration Owner|—|
|**2**|Nairobi|NOT STARTED|—|—|Trajectory/Posture Owner|—|
|**3**|Berlin|NOT STARTED|—|—|Integration Owner|—|
|**4**|Moscow|NOT STARTED|—|—|Face/ANPR Owner|—|
|**5**|Denver|NOT STARTED|—|—|Pitch Lead|—|
|**6**|Rio|NOT STARTED|—|—|All Team|Internal round qualification|
|**7**|Stockholm|NOT STARTED|—|—|Project Owner|Finals qualification + MHA interest|

## Risk Mitigation per Phase

|Phase|Risk|Mitigation|
|---|---|---|
|**Tokyo**|MQTT setup fails|Use plain HTTP polling fallback for demo.|
|**Tokyo**|RTSP stream unstable|Use pre-recorded video files as RTSP source.|
|**Nairobi**|YOLO too slow on target hardware|Use `yolov8n` (nano) variant, TensorRT optimization.|
|**Nairobi**|ByteTrack loses tracks frequently|Tune detection threshold, use DeepSORT fallback.|
|**Berlin**|Orchestrator logic too complex|Start with rule-based scoring, add ML later.|
|**Berlin**|Multi-camera fusion fails|Demo with 1 camera if needed, mention fusion as WIP.|
|**Moscow**|InsightFace too large for Jetson|Use lightweight `mobilefacenet` variant.|
|**Moscow**|ANPR fails on Indian plates|Use PaddleOCR Indian model, custom training if needed.|
|**Denver**|Demo crashes|Have Plan B/C/D ready, rehearse 10+ times.|
|**Denver**|Judge asks unanswerable question|Use emergency response protocol, honesty over BS.|
|**Rio**|Jetson Orin Nano not available|Use cloud GPU for demo, acknowledge edge testing pending.|
|**Rio**|Animal-cart dataset unavailable|Use synthetic data or staged footage, acknowledge gap.|

**Document Control** **Owner:** Project Owner. | **Reviewers:** All team members. | **Update:** After every phase completion, team meeting, or blocker resolution. | **Change Log:** v1.0 — 2026-08-31 — Initial phase plan from Master Handoff.