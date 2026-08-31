# Architecture Diagram & Mapping Document — SIH26187

**Title:** AI-Based Intelligent Video Analytics Platform for Border Surveillance

**Version:** 1.0 | 2026-08-31

**Classification:** SIH Internal Round — Top 50 Qualifier

**Status:** **LOCKED BASELINE**

**Format:** Obsidian-compatible Markdown with Mermaid diagrams

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[07A_Model_A_Delegation]] · [[07B_Model_B_Delegation]] · [[05_Project_Handbook]]
> 
> **Purpose:** Visual reference and diagnostic map for the full pipeline, data flows, component interactions, fallback logic, and security layer. This document exists so that when something breaks, you trace it in seconds, not minutes. Every node below is a linkable Obsidian heading.

## 1. Full System Pipeline (Top-Level) & Data Flow

### Diagram: End-to-End Pipeline

Code snippet

```
flowchart TB
subgraph CAMERAS [Existing IP CCTV Infrastructure]
    C1[Camera 1<br/>RTSP H.264]
    C2[Camera 2<br/>RTSP H.264]
    C3[Camera N<br/>RTSP H.264]
end

subgraph EDGE [Edge Compute Node<br/>Jetson Orin Nano / BOP]
    subgraph MA [Model A — Module Creator<br/>Lightweight Filter/Classifier]
        MA1[Time-Sampling<br/>Skip Redundant Frames]
        MA2[Zone Tagging<br/>Close Range / Long Range]
        MA3[Absolute Trigger Detection<br/>Climbing / Fence-Cutting<br/>Multi-Frame Confirmation]
        MA4[Anti-Spoofing Check<br/>Temporal Continuity]
        MA5[JSON Normalization<br/>Unified Event Schema]
    end

    subgraph BUS [Unified Event Bus<br/>MQTT / Kafka-Style]
        B1[Topic: model_a/raw]
        B2[Topic: model_b/face]
        B3[Topic: model_b/anpr]
        B4[Topic: model_b/posture]
        B5[Topic: model_b/trajectory]
    end

    subgraph MB [Model B — Engine Suite]
        subgraph CR [Close Range Module]
            FE[Face Engine<br/>InsightFace / ArcFace]
            AE[ANPR Engine<br/>YOLO + PaddleOCR]
        end

        subgraph LR [Long Range Module]
            TE[Trajectory Engine<br/>YOLO + ByteTrack]
            PE[Posture Engine<br/>MediaPipe Pose]
        end
    end

    subgraph ORCH [Orchestrator<br/>Context-Reasoning Layer]
        OR1[Threat Scoring<br/>Posture + Zone + Time + Trajectory]
        OR2[Package-Drop Detection<br/>PROPOSED]
        OR3[Group-Coordination Detection<br/>PROPOSED]
        OR4[Border Context Profile<br/>BSF / SSB / ITBP / Assam Rifles]
    end
end

subgraph CMD [Command Centre]
    DASH[Human Orchestrator Dashboard<br/>LEFT: Live Video + Metadata<br/>RIGHT: Flagging + Health Monitor]
    DB[(Event Log DB<br/>SQLite / PostgreSQL)]
    STORE[(Evidence Store<br/>Local FS + SHA-256)]
end

C1 --> MA
C2 --> MA
C3 --> MA

MA --> MA1 --> MA2 --> MA3 --> MA4 --> MA5

MA5 --> B1

B1 --> FE
B1 --> AE
B1 --> TE
B1 --> PE

FE --> B2
AE --> B3
PE --> B4
TE --> B5

B2 --> ORCH
B3 --> ORCH
B4 --> ORCH
B5 --> ORCH

ORCH --> DASH
ORCH --> DB
ORCH --> STORE

DASH --> ORCH
```

## 2. Phase-by-Phase Breakdown & Diagnostics

### Phase 0 — Ingestion

- **Input:** Multiple IP CCTV feeds (existing infra, camera-agnostic, VMS-agnostic — integrates via RTSP/ONVIF + API).
    
- **If broken:** Check camera connection string first, then protocol handshake (RTSP/ONVIF), before assuming a model bug downstream.
    

### Phase 1 — Model A (Module Creator)

- **Confirmed Functions:** Time-sampling, most-differentiated-frame selection, range/zone tagging (close vs. long), absolute-trigger detection with multi-frame confirmation (2–3 frames), JSON schema normalization.
    
- **Proposed Extensions (not built):** Animal-cart tagging, homography perspective-correction for chokepoint cameras.
    
- **Outputs:** (a) Training dataset for Model B, (b) Real-time flagged modules + metadata.
    
- **If broken:** False triggers reappearing → check multi-frame confirmation threshold first (this is the #1 regression risk — it's the exact bug that killed CIBMS). Missing zone tags → check range/zone tagging logic before touching any Model B engine.
    
- **Owner:** [[07A_Model_A_Delegation]]
    

Code snippet

```
flowchart LR
subgraph INPUT [Raw Frame Input]
    F1[Frame t]
    F2[Frame t+1]
    F3[Frame t+2]
end

F1 --> DIFF[Frame Difference<br/>MSE Calculation]
F2 --> DIFF

DIFF --> DEC{MSE < Threshold?}

DEC -->|YES| SKIP[Skip Frame]
DEC -->|NO| SELECT[Select Most<br/>Differentiated Frame]

SELECT --> ZONE[Zone Tagging<br/>Close Range vs Long Range]

ZONE --> TRIGGER[Absolute Trigger Detection<br/>Climbing / Fence-Cutting / Rapid Approach]

TRIGGER --> CONF{Trigger Persists<br/>2-3 Consecutive Frames?}

CONF -->|NO| DISCARD[Discard as Noise]
CONF -->|YES| SPOOF[Anti-Spoofing Check<br/>Temporal Continuity / FPS Consistency]

SPOOF --> VALID{Valid?}

VALID -->|NO| FLAG[Flag as Possible<br/>Replay / Injection]
VALID -->|YES| JSON[Normalize to<br/>JSON Event Schema]

JSON --> OUTPUT[Publish to Bus<br/>Topic: model_a/raw]
FLAG --> OUTPUT
```

### Phase 2 — Single-Line Event Bus

- **Pattern:** Event-bus (MQTT/Kafka-style). Every engine publishes independently, no engine waits on another.
    
- **LOCKED Security Decision:** No separate/bypass channel for absolute triggers. CRITICAL-severity events go through the SAME bus, just severity-tagged. This was deliberately rejected as a security risk (unmonitored bypass = spoofing target). Do not "optimize" this by adding a fast lane.
    
- **If broken:** If CRITICAL events are delayed, the fix is severity-tag prioritization logic on the SAME channel, never a new channel.
    

### Phase 3 & 4 — Model B Engine Suite (Close & Long Range Modules)

- **Phase 3 (Close Range):**
    
    - **Face Engine:** Close range only, no long-range face ID promised (resolution physics limitation).
        
    - **ANPR Engine:** PROPOSED SCOPE is chokepoint/ICP cameras only. High-mounted BOP cameras can't reliably read plates.
        
    - **If broken:** ANPR false negatives on non-chokepoint cameras are _expected behavior_. Check camera classification before debugging OCR.
        
- **Phase 4 (Long Range):**
    
    - **Trajectory Engine:** Highest-leverage component; powers human tracking, feeds vehicle detection, feeds proposed extensions (package-drop, group-coordination).
        
    - **Posture Engine:** Tightly coupled to Border Context Profile. Posture alone only says "this looks like crawling," not whether it's suspicious.
        
    - **If broken:** If Suspicious Activity flags are noisy, check whether it's a Posture Engine bug or a missing Context Profile (Phase 5b). Don't conflate them.
        
- **Owner:** [[07B_Model_B_Delegation]]
    

Code snippet

```
flowchart TB
subgraph BUS_IN [From Unified Bus<br/>Topic: model_a/raw]
    E1[Event: Close Range<br/>Human Detected]
    E2[Event: Close Range<br/>Vehicle Detected]
    E3[Event: Long Range<br/>Human Detected]
    E4[Event: Long Range<br/>Vehicle Detected]
end

E1 --> FE[Face Engine<br/>Input: Close Range Frame + BBox]
E1 --> PE1[Posture Engine<br/>Input: Any Frame + Human BBox]

E2 --> AE[ANPR Engine<br/>Input: Chokepoint Frame + Vehicle BBox]
E2 --> PE2[Posture Engine<br/>Input: Any Frame + Human BBox]

E3 --> TE1[Trajectory Engine<br/>Input: Long Range Frame + Track ID]
E3 --> PE3[Posture Engine<br/>Input: Any Frame + Human BBox]

E4 --> TE2[Trajectory Engine<br/>Input: Long Range Frame + Track ID]
E4 --> PE4[Posture Engine<br/>Input: Any Frame + Human BBox]

FE --> OUT1[Publish: model_b/face<br/>Face ID / Embedding / Confidence]
AE --> OUT2[Publish: model_b/anpr<br/>Plate Text / Confidence / Image]
PE1 & PE2 & PE3 & PE4 --> OUT3[Publish: model_b/posture<br/>Posture Class / Anomaly Score]
TE1 & TE2 --> OUT4[Publish: model_b/trajectory<br/>Track ID / Path / Velocity / Behavior]

subgraph NOTE [Key Design Principle]
    N1[All engines publish INDEPENDENTLY<br/>No engine waits for another<br/>No sequential dependency]
end
```

### Multi-Camera Fusion (Identity Resolution)

- **Logic:** Each feed goes through Model A individually, then normalizes/merges into one identity picture _before_ Model B. Prevents the same person from being double-tracked as two weak entities across two cameras.
    
- **If broken:** Duplicate-identity bugs live in the fusion/merge step, between Model A output and Model B input — check there first.
    

Code snippet

```
flowchart TB
subgraph CAM_A [Camera A<br/>Zone: Perimeter North]
    A1[Track ID: T1<br/>Human / Bounding Box]
    A2[Track ID: T2<br/>Human / Bounding Box]
end

subgraph CAM_B [Camera B<br/>Zone: Perimeter South]
    B1[Track ID: T3<br/>Human / Bounding Box]
    B2[Track ID: T4<br/>Human / Bounding Box]
end

subgraph FUSION [Multi-Camera Fusion Layer<br/>Inside Orchestrator]
    F1[Spatial Overlap Check<br/>Camera FOV overlap zones]
    F2[Temporal Consistency Check<br/>Exit Camera A ≈ Enter Camera B]
    F3[Appearance Match<br/>Embedding similarity / Color histogram]
    F4[Identity Merge Decision]
end

A1 & A2 --> F1
B1 & B2 --> F1

F1 --> F2 --> F3 --> F4

F4 --> MERGE{Same Entity?}

MERGE -->|YES| UNIFIED[Unified Identity<br/>Global Track ID: G1<br/>Merged Trajectory Across Cameras]
MERGE -->|NO| SEPARATE[Keep Separate<br/>T1, T2, T3, T4]

UNIFIED --> ORCH[Orchestrator<br/>Single Stronger Entity<br/>vs Multiple Weaker Entities]
SEPARATE --> ORCH

subgraph PREVENT [Prevents]
    P1[Same person tracked as<br/>two separate weaker entities<br/>across two cameras]
end
```

### Phase 5a & 5b — Orchestration & Border Context Profile

- **Phase 5a (Orchestration Line):** Context-reasoning rules run HERE, reusing engine output. Threat-scoring combines posture + zone + time + trajectory. _PROPOSED:_ Package-drop detection and Group-coordination detection.
    
- **Phase 5b (Border Context Profile - BIGGEST UNBUILT GAP):**
    
    - _Purpose:_ Tune threat-score thresholds/engine weighting per deployment context. Same 5 engines everywhere, different sensitivity dials.
        
    - _Status:_ **NOT BUILT.** Treat any change here as high-risk, high-visibility.
        
    - _Strategy:_ If you build anything here for the demo, scope explicitly to 1–2 profiles (e.g., SSB vs. BSF) as a PoC and present the rest as roadmap.
        

Code snippet

```
flowchart TB
subgraph INPUT [Orchestrator Input]
    I1[Posture: Crawling]
    I2[Zone: Intrusion Zone]
    I3[Time: 02:00 AM]
    I4[Trajectory: Approaching Fence]
end

I1 & I2 & I3 & I4 --> PROFILE[Border Context Profile<br/>Lookup / Rule Engine]

subgraph PROFILES [Profile-Specific Scoring]
    P_BSF[BSF Profile<br/>Pakistan/Bangladesh Border<br/>Strict Thresholds<br/>Fast Alert<br/>Face+ANPR Weighted Heavy]
    P_SSB[SSB Profile<br/>Nepal/Bhutan Border<br/>High-Tolerance Thresholds<br/>Posture+Trajectory+Time Weighted Heavy<br/>Presence Alone NEVER Triggers]
    P_ITBP[ITBP Profile<br/>China/Tibet Border<br/>Group/Patrol Pattern Detection<br/>Cold-Weather Validation]
    P_AR[Assam Rifles Profile<br/>Myanmar/Northeast<br/>Foliage Occlusion Handling<br/>Thermal Near-Mandatory]
end

PROFILE --> SELECT{Which Profile<br/>Active?}

SELECT -->|BSF| P_BSF
SELECT -->|SSB| P_SSB
SELECT -->|ITBP| P_ITBP
SELECT -->|AR| P_AR

P_BSF --> SCORE_BSF[Threat Score: 95<br/>Alert: IMMEDIATE]
P_SSB --> SCORE_SSB[Threat Score: 45<br/>Alert: LOG ONLY<br/>Context: Farmer Early Morning]
P_ITBP --> SCORE_ITBP[Threat Score: 70<br/>Alert: WATCH<br/>Pattern: Possible Patrol]
P_AR --> SCORE_AR[Threat Score: 80<br/>Alert: CHECK<br/>Occlusion: Heavy Foliage]

SCORE_BSF & SCORE_SSB & SCORE_ITBP & SCORE_AR --> OUT[Final Alert Decision<br/>Published to Dashboard]

subgraph GAP [Open Questions]
    G1[Full 4-profile implementation<br/>feasible before internal round?]
    G2[Who updates profile in field?<br/>Jawan-friendly UI or backend config?]
end
```

### Phase 6 — Human Orchestrator Dashboard

- **Left Panel:** Live detected video (RT), metadata (time, cam ID, etc.), pipeline-flow status.
    
- **Right Panel:** Flagging/control panel, pipeline health monitor, engine health monitor + fallback status.
    
- **If broken:** A "blank" dashboard almost always means a bus subscription issue, not a rendering bug — check MQTT/event subscription before touching frontend code.
    

## 3. Core System Behaviors (Locked)

### Fallback Logic (Engine Failure Handling)

- **Logic:** If any Model B engine fails, that camera's traffic routes back through Model A's basic motion+trigger detection.
    
- **Rule:** This is a safety floor. Model A _cannot_ replace Model B's specialized detection, it only prevents total blindness.
    
- **Debug Entry Point:** If you see degraded-but-not-zero detection, check Model B's health-check/heartbeat. Fallback mode is working as intended.
    

Code snippet

```
flowchart TB
subgraph HEALTH [Health Monitor]
    H1[Watch Heartbeat<br/>Each engine publishes<br/>health every 10s]
    H2{Heartbeat Missing<br/>>30s?}
end

H1 --> H2

H2 -->|NO| NORMAL[Normal Operation<br/>All Engines Active]
H2 -->|YES| FAIL[Engine Failure Detected<br/>e.g., Face Engine Down]

FAIL --> DEGRADE[Tag Engine as DEGRADED<br/>Log Incident]

DEGRADE --> ROUTE[Route Camera Traffic<br/>Back to Model A Safety Floor]

ROUTE --> MA_FALLBACK[Model A Basic Mode<br/>Motion Detection + Absolute Triggers Only<br/>NO Face / NO ANPR / NO Posture / NO Trajectory]

MA_FALLBACK --> BUS_FALLBACK[Publish to Bus<br/>Severity: DEGRADED<br/>Tag: fallback_active]

BUS_FALLBACK --> DASH[Dashboard Shows<br/>FALLBACK ACTIVE Indicator<br/>for Affected Camera]

DASH --> MANUAL[Manual Intervention Required<br/>No Auto-Restart]

subgraph WARNING [Critical Warning]
    W1[Model A is deliberately lightweight<br/>CANNOT fully replace Model B<br/>Prevents total blindness ONLY]
end
```

### Security Layer (Locked Decisions)

Code snippet

```
flowchart TB
subgraph SEC [Security Architecture]
    S1[Single Validated Bus<br/>NO Bypass Channel]
    S2[Severity Tagging<br/>CRITICAL on same bus<br/>as everything else]
    S3[Anti-Spoofing<br/>Temporal Continuity Check<br/>Inside Model A]
    S4[Zero Trust Network<br/>Camera LAN Isolated]
    S5[Mutual Auth<br/>TLS Client Certs / PSK]
    S6[SRT Encryption<br/>AES-128/256 for<br/>Unreliable Links]
    S7[Evidence Integrity<br/>SHA-256 + Model Version<br/>+ Timestamp]
    S8[Camera Health Monitor<br/>Obstruction / Darkness /<br/>Frozen Stream / FPS Anomaly]
end

S1 --> S2 --> S3
S4 --> S5 --> S6
S7 --> S8

subgraph REJECTED [Explicitly Rejected]
    R1[~~Separate Bus Line for Triggers~~<br/>REJECTED: Unmonitored bypass<br/>= easy spoofing target]
end
```

### Night/Low-Light Pipeline

Code snippet

```
flowchart LR
subgraph NIGHT [Night-Time Input]
    N1[IR-Illuminated Frame<br/>Low Light / High Noise]
end

N1 --> ENHANCE[Zero-DCE Enhancement<br/>Lightweight Preprocessing]

ENHANCE --> DETECT[YOLOv8 Detection<br/>Same Model, Better Input]

DETECT --> TRACK[ByteTrack / DeepSORT<br/>Same Tracking Pipeline]

TRACK --> OUTPUT[Standard Event Flow<br/>Model A → Bus → Model B]

subgraph LIMIT [Hard Limit]
    L1[Cannot extend physical<br/>IR camera range<br/>Software improves what<br/>sensor captures, not<br/>hardware reach]
end
```

### Demo Architecture (Internal Round)

Code snippet

```
flowchart TB
subgraph DEMO [Hackathon Demo Setup]
    subgraph SOURCE [Video Sources]
        S1[Laptop Webcam 1<br/>Simulates Camera A<br/>Daylight Scene]
        S2[Pre-Recorded Footage<br/>Simulates Camera B<br/>Night/Crawling Scene]
        S3[Pre-Recorded Footage<br/>Simulates Camera C<br/>Vehicle/Chokepoint Scene]
    end

    subgraph EDGE [Edge Node Simulation]
        E1[Laptop / Desktop<br/>GPU: RTX 3060 or better<br/>Ubuntu 22.04]
        E2[Docker Compose<br/>All Services Containerized]
    end

    subgraph DISPLAY [Dashboard Display]
        D1[React Dashboard<br/>Localhost:3000]
        D2[MQTT Broker<br/>Mosquitto :1883]
    end
end

S1 --> E1
S2 --> E1
S3 --> E1

E1 --> D1
E1 --> D2

subgraph SCOPE [Demo Scope]
    SC1[Human Tracking<br/>Across 2-3 feeds]
    SC2[Virtual Fence<br/>Line Crossing Alert]
    SC3[Basic Posture<br/>Crawling / Standing]
    SC4[Real-Time Alerts<br/><5s Latency Target]
    SC5[Dashboard<br/>Live View + Event Log]
end
```

## 4. Strategic Constraints & Quick Debugging

**Total Model Count (Protect This Claim):**

5 models by design = Model A (1) + Face/ANPR/Posture/Trajectory (4). Explicitly NOT one giant model. _Caveat:_ "5 lightweight models running in parallel" is UNVERIFIED until benchmarked on target hardware — do not present as confirmed efficiency without a benchmark run. (See `02_TRD > 4-locked-json-event-schema` for the schema that keeps these 5 models decoupled).

### Quick Debug Index

|**Symptom**|**Check First (Entry Point)**|
|---|---|
|**Camera feed silent**|Phase 0 Ingestion — RTSP/ONVIF handshake|
|**False fence alerts reappearing**|Phase 1 Model A — Multi-frame confirmation threshold|
|**CRITICAL alert delayed**|Phase 2 Bus — Severity-tag priority (do NOT create new channel)|
|**ANPR missing plates**|Phase 3 Close Range — Is this camera classified as a chokepoint?|
|**Duplicate person tracked across cameras**|Multi-Camera Fusion — Identity merge step|
|**Suspicious-activity false positive/negative**|Phase 5b Context Profile vs. Phase 4 Posture — Isolate the layer|
|**Dashboard blank/frozen**|Phase 6 Dashboard — Bus/MQTT subscription before frontend|

## Legend & Document Control

- `>>` : Data flow direction
    
- `par` : Parallel processing (no dependency)
    
- `DEGRADED` : Fallback mode active
    
- `PROPOSED` : Not yet built, pending approval
    
- `REJECTED` : Explicitly considered and rejected
    
- `GAP` : Known unbuilt dependency
    
- `LOCKED` : Decision finalized, do not change without owner approval
    

**Owner:** Technical Lead.

**Reviewers:** Project Owner, Pitch Lead.

**Approval Required For:** Any structural change to diagrams, any new component, any change to locked security decisions.

**Change Log:** v1.0 — 2026-08-31 — Initial architecture diagrams and mapping integrated from Master Handoff.