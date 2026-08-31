# SIH26187 — MASTER HANDOFF DOCUMENT
## Complete Project Context for AI-Assisted Continuation (Claude Code / Any Agent)

**READ THIS FIRST — RULES FOR WHOEVER PICKS THIS UP:**
1. This document is the locked baseline. Do not redesign, restructure, rename, or "improve" the architecture below without explicit approval from the project owner.
2. Where something is marked "PROPOSED" or "OPEN," it is not built/decided — treat it as a discussion point, not a fact.
3. New solutions should extend existing components (add a class, add a rule, add a preprocessing step) rather than introduce new top-level engines/models, unless the owner explicitly approves a structural change.
4. Never present internal technical estimates as verified external facts (see §8 — several "facts" used in early pitching were actually our own estimates).
5. This is a Smart India Hackathon (SIH) college-level competition project — currently preparing for the **internal college round (top 50 qualifier)**, not final submission. Scope decisions accordingly.

---

## 1. Problem Statement Identity

- **PS ID:** SIH26187
- **Ministry:** Ministry of Home Affairs
- **Requesting body:** Sashastra Seema Bal (SSB), Police II Division
- **Title:** AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure
- **Theme:** Blockchain & Cybersecurity | **Category:** Software
- **Core constraint (verbatim intent):** Software-only platform that transforms *existing* IP CCTV into intelligent surveillance — **explicitly without requiring dedicated FRS, ANPR, or smart-camera hardware.**

### Official required capabilities (confirmed from actual PS notes)
1. Human detection and tracking
2. Vehicle detection and classification
3. Face detection
4. ANPR (Automatic Number Plate Recognition)
5. Suspicious activity detection
6. Virtual fence intrusion detection
7. Night-time movement detection
8. Real-time generation and event logging

### Official solution requirements
No dependence on expensive hardware · smart AI-powered analytics · real-time alerts · supports face/vehicle/behavioral analytics via software only · good response time for BSF/SSB · human command centre · cost-effective, scalable, suitable for all locations.

---

## 2. Why This Approach (Government Context — Critical for Pitch)

MHA has already tried the hardware-heavy route twice, and both failed — this is the entire reason the PS demands software-only:

- **BOLD-QIT** (India-Bangladesh riverine border, Assam): ₹86 crore spent, equipment unused/junked, sub-standard gear, network collapsed.
- **CIBMS**: Confirmed failure causes (from IDSA research) — (1) sensor-fusion software integration was the real bottleneck, not hardware deployment; (2) false-alarm epidemic from wildlife/weather/vegetation triggering constant alerts; (3) operator competence gap — jawans lacked technical expertise to run sophisticated equipment; (4) maintenance collapse — high electronics cost, no spare parts, weather degradation; (5) vague procurement requirements led to over-engineering.

**Positioning:** "Don't sell MHA new hardware boxes — give them software that makes existing cameras smart, survives remote conditions, and is operable by non-technical jawans."

---

## 3. THE LOCKED ARCHITECTURE

### Full pipeline (top to bottom)

```
MULTIPLE CCTV CAMERAS (existing IP infrastructure, no new hardware)
        │
        ▼
┌─────────────────────────────┐
│  MODEL A — Module Creator      │
│  CONFIRMED FUNCTIONS:
│  • time-sampling (skip redundant/identical frames)
│  • most-differentiated-frame selection
│  • range/zone tagging (Close Range vs Long Range)
│  • absolute-trigger detection (climbing, fence-cutting) with
│    MULTI-FRAME CONFIRMATION — trigger must persist across
│    2-3 consecutive frames before flagging. This directly
│    fixes the documented real failure: "50 false alerts/night,
│    guards disabled the system."
│  • normalizes both outputs into one JSON event schema
│  • outputs: (a) training dataset for Model B engines
│             (b) real-time flagged modules + metadata
│
│  PROPOSED EXTENSIONS (not yet built, pending approval):
│  • animal-class filtering extended to detect "animal + wheeled
│    object in proximity" = animal-cart tag (reuses existing
│    animal-detection needed to prevent false fence triggers
│    from wildlife)
│  • perspective-correction preprocessing (homography transform)
│    for chokepoint/ICP cameras specifically, to partially
│    compensate for high camera mounting before ANPR runs
└──────────────┬────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │   SINGLE-LINE BUS     │  Event-bus pattern (Kafka/MQTT-style).
     │   (UNIFIED — NO       │  Every engine publishes independently —
     │    BYPASS CHANNEL)    │  no engine waits on another.
     └──────────┬─────────────┘
                │             SECURITY DECISION (locked): absolute
                │             triggers do NOT get a separate/dedicated
                │             bus line — that was considered and
                │             REJECTED as a security risk (an
                │             unmonitored bypass channel is an easy
                │             spoofing target). Instead: trigger event
                │             is tagged CRITICAL severity on the SAME
                │             bus everything else uses, so there is
                │             only one validated channel into the
                │             Orchestrator, not two.
    ┌───────────┴────────────┐
    ▼                         ▼
┌─────────────────┐  ┌─────────────────┐
│ CLOSE RANGE       │  │ LONG RANGE        │
│ MODULE             │  │ MODULE             │
│ ┌───────────────┐ │  │ ┌───────────────┐ │
│ │ Face Engine     │ │  │ │ Trajectory      │ │
│ └───────────────┘ │  │ │ Engine          │ │
│ ┌───────────────┐ │  │ └───────────────┘ │
│ │ ANPR Engine     │ │  │ ┌───────────────┐ │
│ │ (PROPOSED SCOPE:│ │  │ │ Posture Engine  │ │
│ │  chokepoint/ICP │ │  │ └───────────────┘ │
│ │  cameras only,  │ │  │ Vehicle detection/  │
│ │  NOT every       │ │  │ classification also │
│ │  camera)        │ │  │ lives here          │
│ └───────────────┘ │  └──────────┬─────────┘
└──────────┬─────────┘             │
           │                       │
           └───────────┬───────────┘
                        ▼
           ┌─────────────────────────┐
           │  SINGLE ORCHESTRATION      │
           │  LINE                      │
           │  (Context-reasoning rules   │
           │  run HERE, reusing engine    │
           │  output — not new engines):  │
           │  • threat-scoring (posture+  │
           │    zone+time+trajectory)     │
           │  • PROPOSED: package-drop     │
           │    detection (object         │
           │    separates from person-    │
           │    track and stays static)   │
           │  • PROPOSED: group-           │
           │    coordination detection    │
           │    (3+ tracked entities,      │
           │    same zone, same direction, │
           │    short time window)         │
           └──────────────┬──────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │           HUMAN ORCHESTRATOR             │
        │  ┌────────────────┐ ┌──────────────────┐│
        │  │ LEFT PANEL       │ │ RIGHT PANEL        ││
        │  │ • live detected  │ │ • flagging/control  ││
        │  │   video (RT)     │ │   panel              ││
        │  │ • metadata (time,│ │ • pipeline health    ││
        │  │   cam ID, etc)   │ │   monitor            ││
        │  │ • pipeline-flow  │ │ • engine health       ││
        │  │   status         │ │   monitor + FALLBACK  ││
        │  └────────────────┘ └──────────────────┘│
        └──────────────────────────────────────────┘
```

### Fallback logic (locked)
If any Model B engine fails, that camera's traffic routes back through Model A's basic motion+trigger detection — a safety floor, NOT a literal "role swap." Model A is deliberately lightweight and cannot fully replace Model B's specialized detection; it only prevents total blindness during an outage.

### Total model count (locked claim, protect this in pitch)
**5 models by design:** Model A (1 lightweight filter/classifier) + Model B's four engines (Face, ANPR, Posture, Trajectory). Explicitly NOT one giant model — this is a stated architectural strength, not a limitation. **Caveat for the pitch:** this "5 lightweight models running in parallel" claim is UNVERIFIED until benchmarked on actual target hardware — do not present as confirmed efficiency without testing.

### Multi-Camera Fusion
Multiple camera feeds each go through Model A individually, then get normalized/merged into one identity picture before reaching Model B — this prevents the same person being tracked as two separate, weaker entities across two cameras.

---

## 4. Capability-to-Architecture Coverage Audit

| # | Capability | Status | Notes |
|---|---|---|---|
| 1 | Human detection & tracking | **Fully covered** | Trajectory Engine + Multi-Camera Fusion |
| 2 | Vehicle detection & classification | **Covered for motorized; gap for animal carts** | Needs labeled dataset for animal-cart class — data source not yet confirmed |
| 3 | Face detection | **Covered within range limits** | Close Range only — long-range face ID not promised (resolution physics, not fixable by better AI) |
| 4 | ANPR | **Partial — chokepoint-only** | High-mounted BOP cameras cannot reliably read plates; scope explicitly limited to ICP/chokepoint cameras |
| 5 | Virtual fence intrusion | **Fully covered — strongest capability** | Two-tier geofence + multi-frame confirmation directly fixes documented false-alarm failure |
| 6 | Suspicious activity | **Core covered; context layer NOT built** | Posture Engine works; Border Context Profile (threshold tuning per border/time/zone) is the biggest unbuilt dependency — see §6 |
| 7 | Night-time movement | **Covered within sensor limits** | Cannot extend physical IR camera range — software improves what's captured, not hardware reach |
| 8 | Real-time alerts + logging | **Fully covered** | Directly engineered against "50 false alerts/night, guards disabled it" real failure mode |

**Simple summary: 5 of 8 solidly covered, 2 honestly scoped down (ANPR, animal-carts), 1 has a real unbuilt gap (Suspicious Activity's context layer).**

---

## 5. Capability Dependency Map (build order matters)

- **Trajectory Engine = highest-leverage component.** Powers Human Tracking directly, feeds Vehicle Detection context, and is the reused data source for both proposed extensions (package-drop, group-coordination). Build this well first.
- **Model A's multi-frame confirmation logic** is shared infrastructure for BOTH Virtual Fence (Cap 5) and Real-Time Alerting (Cap 8) — same anti-false-positive logic, applied twice. Fixing it once improves both capabilities simultaneously.
- **JSON event schema** is a dependency for everything downstream — get field names/structure right early (camera ID, timestamp, zone, entity type, confidence, evidence ref), or you'll retrofit it into 5 engines later.
- **Posture Engine + Border Context Profile are tightly coupled** — Posture alone only gives "this looks like crawling"; Suspicious Activity Detection isn't actually complete until Context Profile adds "crawling at 2am here = flag, same posture at 6am on a farm path = ignore." These need to land together.
- **Face Engine and ANPR Engine are independent of Trajectory/Posture** (not fully independent of the whole pipeline — they still depend on Model A's zone-tagging and the shared JSON schema). This means Face/ANPR work CAN be parallelized on a different team member's track from Trajectory/Posture work.

---

## 6. The Border Context Profile (Biggest Open Gap — Not Built)

**What it needs to do:** tune threat-score thresholds and engine weighting differently per deployment context. Same 5 engines everywhere, different sensitivity dials.

| Border Force | Territory | Threat Reality | Profile Needs |
|---|---|---|---|
| BSF | Pakistan, Bangladesh | Fenced, hostile — almost anyone crossing = threat | Strict thresholds, fast alert, Face+ANPR weighted heavy |
| SSB | Nepal, Bhutan | Open border (1950 Treaty), thousands of legitimate daily crossings (farmers, traders, pilgrims — ~38% marriage-related migration) | High-tolerance thresholds, Posture+Trajectory+time-pattern weighted heavy — presence alone must never trigger an alert |
| ITBP | China (Tibet) | High-altitude, threat = military patrols/espionage, not lone smugglers | Needs group/patrol-pattern detection (not yet designed), cold-weather hardware validation unconfirmed |
| Assam Rifles | Myanmar, Northeast | Dense forest, insurgents/arms smuggling | Heavy foliage occlusion breaks visual line-of-sight assumption; thermal becomes near-mandatory here, not optional |

**Open questions for whoever builds this:** Is full 4-profile implementation feasible before the internal round deadline, or should the team build/demo just 1-2 sample profiles (e.g., a simple time-of-day + zone lookup table) and present the rest as roadmap? Who updates a profile in the field — does it need a UI a non-technical jawan can use, or is it a backend config only the dev team touches? (This matters because "operable by non-technical jawans" is a stated positioning pillar — a profile that needs an engineer to edit JSON contradicts that pillar unless explicitly scoped as "out of hackathon scope.")

---

## 7. Competitive Research Summary

| Company/System | Technique Studied | Where It's Used In Our Design | Why We Beat Them |
|---|---|---|---|
| **Palantir (Gotham/AIP)** | Ontology — normalize all sources into one object model | Our JSON event schema design | No enterprise cost, no proprietary hardware |
| **Anduril (Lattice)** | Edge-first fusion, detect→classify→decide→act loop | Our edge-processing design (Model A/B run locally) | Runs on existing cameras, not proprietary towers |
| **Bosch/Hikvision/Axis/Milestone** | Rule/zone-trigger analytics, false-trigger suppression | Baseline for our Model A motion pre-filter | We add context reasoning (posture+zone+time) on top |
| **Staqu (JARVIS)** | AI video analytics on existing CCTV, works with 8 state police + Indian Army | Closest real Indian competitor — general policing tool | No border-specific virtual fencing, no offline-first design |
| **Innefu Labs** | 512-dim face vector + nearest-neighbor match, used by Delhi Police | Directly adopted for our Face Engine | Extended to work on genuinely low-res long-range CCTV via reconstruction |
| **Tonbo Imaging** | Thermal/electro-optic hardware for Indian Army | Validates our optional thermal extension direction | We deliver as software, not hardware |
| **BSF Tripura pilot (2024)** | Real deployed AI cameras + FRS, credited with contraband seizures | Proof-of-concept the government already believes in | Single pilot stretch, hardware-heavy — we scale nationwide, cheaply |
| **Elbit/IAI-ELTA/L3Harris/Teledyne FLIR** | Defense-tier integrated C4I (radar+EO/IR+UAV fusion) | Command-view concept → our Orchestrator dashboard design | Defense-procurement cost/timeline vs. our software-only speed |

**Honest positioning:** Not "world's best" vs. actual defense-grade platforms (Elbit/IAI-ELTA already have configurability at their scale) — but for a software-only, existing-CCTV, hackathon-scale solution, most contextually intelligent realistic entry in this competition.

---

## 8. Unconfirmed Ground-Truth Items (DO NOT PRESENT AS FACT)

- **"6 out of 10 videos miss movement" is OUR OWN technical estimate, NOT an official SSB/BSF finding.** Never cite as fact to judges. Cite SRT protocol research (independently verifiable) instead.
- Exact SSB camera make/model/resolution: not publicly available — not required since system is camera-agnostic by design.
- SSB's current command-and-control/VMS software: not publicly specified — system designed VMS-agnostic (integrates via RTSP/ONVIF + API).
- SSB BOP power/internet: partially confirmed (generators where no grid power; satellite phone use implies patchy terrestrial internet) — full picture unconfirmed.
- Real-time matching vs. forensic logging preference: best-guess is "forensic logging matters more given open-border volume," not officially confirmed.
- CIBMS failure causes: confirmed via IDSA research (see §2) — this one IS solid, citable.
- SRT protocol precedent: confirmed used in US federal/defense contexts (EOCs, GSOCs) — no confirmed Indian defense precedent yet, but it is NOT an untested protocol globally.
- India-Nepal open border legal status: confirmed — 1950 Treaty of Peace and Friendship, free movement, use of force is politically sensitive (any force incident can affect bilateral relations) — relevant if scoping Nepal-specific behavior.
- SSB jawan computer literacy: confirmed low baseline (UK-funded digital literacy program for Joint Border Task Force personnel implies this gap existed) — dashboard must be extremely simple.

---

## 9. Recommended Open-Source Tech Stack (corrected)

| Component | Recommended | NOT Recommended | Why |
|---|---|---|---|
| Human/vehicle detection + tracking | YOLOv8/v11 + ByteTrack or DeepSORT | — | Current standard, well-documented, matches Trajectory Engine needs |
| Face detection + embeddings | InsightFace (ArcFace models) | — | Open-source, same embedding-vector approach as Innefu's technique |
| Posture/pose estimation | **MediaPipe Pose** | ~~OpenPose~~ | OpenPose too heavy for real-time inference on lightweight edge hardware at a remote BOP; MediaPipe built for exactly this constraint |
| ANPR | **PaddleOCR-based pipeline, or custom YOLO (plate detector) + CRNN (text recognition)** | ~~OpenALPR~~ | OpenALPR historically underperforms on non-US/multi-format plates — weak fit for Indian multi-state, multi-script plates |
| Event bus | MQTT (via Mosquitto) for hackathon scale; Kafka if scaling to production | — | MQTT lighter to set up, sufficient to prove the concept |
| Night/low-light preprocessing | Zero-DCE | — | Lightweight enhancement, cheaper than training a separate night-only model |
| Datasets (verify licensing before use) | UNIRI-TID, PDIWS, SCface, NightOwls, CAMEL, CVC-14 | — | **UNCHECKED: verify these are freely usable/redistributable for a student hackathon prototype before assuming drop-in access** |

**Pitch framing:** "We're not reinventing face recognition or object detection — we're integrating proven open-source building blocks into a border-specific orchestration and context-reasoning layer, which is where our actual original engineering is." This is a stronger, more credible claim than "built everything from scratch."

---

## 10. Security Design (locked decisions)

- **No separate/bypass bus line for triggers** — ruled out as a security risk (unmonitored channel = spoofing target). All traffic through one validated bus, severity-tagged instead.
- **Anti-spoofing/replay-attack detection**: footage lacking trajectory/posture/temporal continuity gets flagged as possibly injected/replayed old footage — done inside Model A.
- **Camera-link security (proposed, not yet built):** Zero Trust network segmentation (camera network isolated from rest of infrastructure), mutual authentication between camera/edge box/orchestrator, no default credentials, SRT's built-in AES encryption for links crossing unreliable networks.
- **Evidence integrity (proposed):** every event carries a hash + model version + timestamp for traceability/tamper-detection.
- **Camera Health Monitor** covers lens obstruction/darkness/frozen stream/FPS anomalies — NOT yet explicitly connected to a deliberate-attack framing (someone spray-painting or lasering a lens) — worth a one-line mention if asked.

---

## 11. Known Loopholes Not Yet Addressed Anywhere

Priority order if time-constrained before internal round:

1. **[URGENT] No demo plan for internal round** — no real BOP footage access. Need to decide: pre-recorded test footage, live webcam simulation, or staged footage of someone crawling/climbing.
2. **[URGENT] Compute/hardware budget not decided** — Jetson Orin Nano (~$249) was estimated early on but never confirmed as the actual plan. Without this, the "5 lightweight models in parallel" claim can't be tested.
3. **[Likely judge question] Privacy/legal exposure of facial recognition on civilians** — India's DPDP Act 2023 governs biometric data processing; no retention/consent policy defined for non-flagged faces captured at an open border with thousands of daily legitimate crossers.
4. Model drift / retraining plan post-deployment — not addressed, low priority for hackathon but good to acknowledge if asked.
5. Physical attacks on the camera itself (spray paint, laser, deliberate obstruction) — partially covered by Camera Health Monitor but not framed as adversarial defense.
6. Dataset licensing — unchecked whether UNIRI-TID/PDIWS/SCface/NightOwls are freely usable for a hackathon prototype.
7. Team role assignment — mentioned once (Face/ANPR parallelizable from Trajectory/Posture) but no actual person-to-task mapping exists.
8. Evaluation metrics for the pitch — a scenario-matrix table (daylight/night/fog/rain/occlusion × detection/tracking/false-alarm-rate/latency) was proposed early on but never finalized — needed for "how do you know it works" question.
9. Camera placement recommendation module — proposed by teammate as a way to turn "can't fix physics" into "we tell you how to get more from what you have" (e.g., software guidance on re-angling an existing camera, or where a cheap IR illuminator would help) — not built, good differentiator if time allows.

---

## 12. Open Decisions Awaiting Owner's Call

- **Fast-path severity:** when Model A catches an absolute trigger, does it (a) tag CRITICAL and go through normal bus+engine flow, or (b) get a "provisional" alert immediately with Model B upgrading to "confirmed" a few seconds later? Suggested direction: two-stage severity (provisional → confirmed) in the JSON schema — not yet decided by project owner.
- **Border Context Profile scope for internal round:** full 4-force implementation vs. 1-2 demo profiles + roadmap slide.
- **ANPR perspective-correction feasibility:** does the team actually have homography/camera-calibration experience, or is this aspirational scope creep? Needs an honest team skills check.

---

## 13. Summary for Quick Orientation

Five small, swappable AI models (not one giant black box) — Face, ANPR, Posture, Trajectory engines plus a lightweight Model A gatekeeper — connected by one unified event bus, feeding a human-operated command dashboard with built-in fallback logic. Software-only, runs on existing IP CCTV, no new hardware. Directly engineered against two documented real government failures (CIBMS's false-alarm epidemic and sensor-fusion collapse). Biggest unbuilt piece: the Border Context Profile that would let the same engines behave appropriately differently at a hostile fenced border vs. an open friendly one. Biggest near-term risk: no demo plan and no confirmed hardware budget for the internal round.
