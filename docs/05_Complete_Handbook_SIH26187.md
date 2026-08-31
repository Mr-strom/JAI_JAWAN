# Comprehensive Project Handbook — SIH26187

**Title:** A-Z Scenarios, Loopholes, Judge Q&A, Competitive Positioning, and Disaster Recovery **Version:** 1.0 | 2026-08-31 **Classification:** SIH Internal Round — Top 50 Qualifier **Status:** **LIVING DOCUMENT — UPDATE AFTER EVERY PRACTICE PITCH**

> **Document Links:** [[01_PRD]] · [[02_TRD]] · [[03_Architecture_Map]] · [[04_Continuity_Protocol]] · [[06_Version_Control_Testing_Strategy]] · [[07A_Model_A_Delegation]] · [[07B_Model_B_Delegation]] **Purpose:** Single reference for every scenario the team might face — technical questions, judge objections, competitor comparisons, failure mode responses, and demo contingencies. Read before every pitch. Update after every feedback session.

## Section 1: Best-Case Workflow & Mid-Build Protocol

### 1.1 Ideal Development Workflow

1. **Schema Frozen:** `schema_v1` is locked (see [[02_TRD]]). Model A and Model B branches start in parallel on the same day.
    
2. **Model A Priorities:** Multi-frame confirmation logic finished first (shared infrastructure for Virtual Fence and Real-Time Alerting — de-risks two capabilities at once).
    
3. **Model B Priorities:** Trajectory Engine built early (highest-leverage, feeds everything downstream). Face/ANPR built in parallel by the second developer (schema-dependent but not Trajectory/Posture-dependent).
    
4. **Integration:** Bus + Orchestrator integration happens once both sides publish valid schema events.
    
5. **Context Profiles:** 1–2 Border Context Profiles demoed live (SSB vs. BSF), the rest presented as a roadmap.
    
6. **Demo Rehearsal:** Use staged/pre-recorded footage with a known false-trigger scenario built in, to visibly prove multi-frame suppression.
    

### 1.2 Protocol for Adding New Features Mid-Development

- **Check the PRD:** Does this feature map to one of the 8 official required capabilities, or is it scope creep?
    
- **Extend, Don't Build New:** Can it extend an existing engine ("add a class, add a rule, add a preprocessing step") rather than requiring a new top-level engine?
    
- **Owner Approval:** If it requires a new top-level engine or structural change to the locked architecture, it needs explicit owner approval — do not let an AI agent silently introduce one.
    
- **Schema Safety:** Update the JSON schema version if new fields are needed (`schema_v2`) — _never_ mutate the current version in place.
    

## Section 2: Capability Deep-Dive (A-Z Scenarios)

### 2.1 Human Detection & Tracking

- **Scenario A: Daylight, clear weather, single person walking perimeter.**
    
    - _Expected:_ Trajectory Engine detects at >100m, assigns track ID.
        
    - _Response:_ Standard operation. Event logged as info. No alert unless zone=intrusion.
        
- **Scenario B: Night, IR illumination, person crawling near fence.**
    
    - _Expected:_ Zero-DCE enhances visibility, Posture Engine classifies crawling, Model A flags absolute trigger, multi-frame confirmation fires.
        
    - _Response:_ Critical alert on dashboard within 3s. Evidence package auto-generated.
        
- **Scenario C: Heavy rain, person with umbrella crossing warning zone.**
    
    - _Expected:_ YOLO may miss/classify as unknown. Posture may misclassify due to umbrella. Trajectory maintains track if partial detection.
        
    - _Response:_ Degraded confidence event. Orchestrator lowers threat score. Camera Health Monitor notes weather condition.
        
- **Scenario D: 3 people walking together, farm path at 6am (SSB border).**
    
    - _Expected:_ Tracks all 3. Posture: walking. Zone: farm_path. Profile: SSB.
        
    - _Response:_ SSB Context Profile — high tolerance. Presence alone does NOT trigger. Logged, no alert. (Correct behavior for open border).
        
- **Scenario E: Same 3 people, 2am, carrying large backpacks, approaching fence.**
    
    - _Expected:_ Trajectory: approach pattern. Posture: carrying. Time: 2am.
        
    - _Response:_ SSB Profile — time+posture+trajectory weighted. Threat score rises. Alert: WATCH. Group-coordination detection flags 3+ entities.
        
- **Scenario F: Person disappears behind tree, reappears 5 seconds later.**
    
    - _Expected:_ ByteTrack Kalman filter predicts position. If reappearance matches, same track ID maintained.
        
    - _Response:_ Continuity preserved. (If >10s, new track ID assigned, Orchestrator may re-merge).
        
- **Scenario G: Person wears camouflage matching background.**
    
    - _Expected:_ YOLO detection confidence drops.
        
    - _Response:_ Lower confidence threshold in border zones. Motion-based fallback may flag movement. Documented limitation: Camouflage defeats visual AI (thermal extension proposed).
        

### 2.2 Vehicle Detection & Classification

- **Scenario H: Truck approaches ICP, stops at gate, plate visible.**
    
    - _Expected:_ YOLO detects heavy vehicle. ANPR reads plate. Face Engine captures driver.
        
    - _Response:_ Standard ICP processing. Plate logged, face matched against watchlist.
        
- **Scenario I: Motorcycle at night, no headlight, plate dirty.**
    
    - _Expected:_ YOLO detects 2-wheeler. ANPR detects plate but OCR confidence low.
        
    - _Response:_ Logged with low-confidence plate. Dashboard shows image for manual verification. System admits uncertainty.
        
- **Scenario J: Bullock cart crossing perimeter (animal-cart gap).**
    
    - _Expected:_ YOLO detects as vehicle (false) or animal (true) or unknown. No animal-cart class exists yet.
        
    - _Response:_ Logged as unknown. Gap acknowledged: requires labeled dataset, not promised for internal round.
        

### 2.3 Face Detection

- **Scenario K: Person walks through ICP gate, fully visible, good lighting.**
    
    - _Expected:_ Detects face, extracts 512-dim embedding, matches in <100ms.
        
    - _Response:_ If match > threshold: critical alert. If no match: log embedding for future search.
        
- **Scenario L: Person at 50m, face 20×20 pixels, side profile.**
    
    - _Expected:_ Face detection fails or confidence <0.3.
        
    - _Response:_ No face event generated. Trajectory still tracks. System does NOT promise long-range face ID (physics limit, honest scope boundary).
        
- **Scenario M: Person wearing mask/burqa.**
    
    - _Expected:_ Face detection fails (out of distribution).
        
    - _Response:_ Alert based on behavior (Posture/Trajectory), not identity. System respects biometric limits.
        

### 2.4 ANPR

- **Scenario N: Car with Delhi plate, clean, straight angle.**
    
    - _Expected/Response:_ >95% confidence. Standard event logged.
        
- **Scenario O: Truck with Rajasthan plate, Devanagari script, angled 30 degrees.**
    
    - _Expected/Response:_ OCR captures mixed Latin-Devanagari. Medium confidence. Human verification recommended. Multi-script support is a stated strength.
        
- **Scenario P: Bike with temporary handwritten plate.**
    
    - _Expected/Response:_ Plate detected, OCR fails (out of distribution). Image saved for human review.
        

### 2.5 Suspicious Activity & Virtual Fence

- **Scenario Q: Person crawling under fence at 3am (BSF border).**
    
    - _Response:_ Threat score: 95. Critical alert. Immediate escalation based on strict BSF Profile.
        
- **Scenario R: Farmer crouching to tie shoelace at 7am near farm path (SSB border).**
    
    - _Response:_ Threat score: 15. No alert, logged only. High-tolerance SSB Profile working as intended.
        
- **Scenario T (Virtual Fence): Deer jumps over fence at night.**
    
    - _Response:_ Motion detected, but shape inconsistent across 2-3 frames. Filtered as animal. **Wildlife false positive SUPPRESSED (fixes CIBMS failure).**
        
- **Scenario U (Virtual Fence): Shadow of tree branch moves in wind.**
    
    - _Response:_ No persistent human-shaped blob across multi-frame confirmation. **Weather false positive SUPPRESSED.**
        

### 2.6 Edge Cases & Hardware Limits

- **Scenario X: Complete darkness — IR illuminator failed.**
    
    - _Response:_ Model A flags DARKNESS anomaly. Health alert sent to dashboard. System does NOT hallucinate detections in black frames.
        
- **Scenario Z: System generates 50 alerts in 1 hour during thunderstorm.**
    
    - _Response:_ Orchestrator throttles similar alerts. Dashboard shows: ALERT STORM — CHECK WEATHER.
        

## Section 3: Judge Q&A & Prepared Responses

**Q1: How is this different from Staqu JARVIS?**

> **A:** Staqu is a great general policing tool, but it lacks border-specific virtual fencing with multi-frame confirmation, offline-first design for remote BOPs, and a Border Context Profile. We aren't competing on generic face recognition; we are competing on border-specific context intelligence.

**Q2: Why 5 small models instead of one big model?**

> **A:** Modularity (if Face fails, Trajectory keeps running), resource efficiency (load only the engines needed per camera on a 8GB Jetson), and maintainability (retrain posture without breaking ANPR). _Caveat: We will benchmark this on target hardware before claiming confirmed efficiency._

**Q3: What about false alarms? CIBMS had 50 per night.**

> **A:** That is exactly why we exist. Our multi-frame confirmation requires a trigger to persist across 2-3 consecutive frames, eliminating noise. Combined with the Context Profile, we target <2 false alerts per night per camera — a 25x improvement over the documented failure.

**Q4: You say software-only, but what hardware do you need / what is the budget?**

> **A:** The cameras are existing IP CCTV. We only need a general-purpose edge computer at the BOP (e.g., Jetson Orin Nano ~$249, or Pi 4 + Coral TPU ~$150). No proprietary hardware boxes. For this demo, we run on standard dev hardware; production sizing is a post-hackathon step.

**Q5: How do you handle the open India-Nepal border where thousands cross legitimately?**

> **A:** The Border Context Profile. For SSB, presence alone NEVER triggers an alert. Posture, trajectory, and time are weighted heavily. A farmer walking at 6am is logged; crawling at 2am is alerted. This reasoning layer is our core differentiator.

**Q6: What about privacy and the DPDP Act 2023?**

> **A:** Honest answer: we have not fully defined a retention/consent policy for non-flagged faces at open borders. Currently, embeddings for non-matches purge in 30 days; only flagged events are kept. Biometric data doesn't leave the edge node. We acknowledge complete legal compliance as a post-deployment requirement.

**Q7: How do you know it actually works? What are your numbers?**

> **A:** We will NOT present unverified/fake benchmarks. We will demonstrate human tracking, virtual fencing, and real-time alerts live/staged today. Our quantitative claims are based on published YOLO/Mediapipe benchmarks, which we will validate against our own evaluation matrix (daylight/night/weather x latency/accuracy) on target hardware before finals.

**Q8: What if the internet goes down at a remote BOP?**

> **A:** It is designed offline-first. Models run on the edge node, event bus is local, dashboard is local. Internet is only for Sector HQ escalation, watchlist syncs, and evidence upload. Local surveillance never stops.

**Q9: Why should MHA trust a college project over Palantir or Anduril?**

> **A:** We don't claim to beat Palantir at enterprise scale. We claim to be the most realistic _software-only, existing-CCTV_ solution here. Palantir costs millions; Anduril requires proprietary towers. We cost <$50k per BOP, run on existing cameras, and are operable by non-technical jawans. We are the pragmatic middle ground.

**Q10: Are your stats (e.g., "6 out of 10 videos miss movement") official?**

> **A:** No, that is our own technical estimate based on bandwidth drops. However, our use of the SRT protocol (which handles high-packet-loss networks) is an independently verifiable fix for that specific video-drop issue.

## Section 4: Competitive Positioning

|Competitor|Their Strength|Our Response|Why We Win|
|---|---|---|---|
|**Palantir (AIP)**|Enterprise ontology, massive scale|No enterprise cost, no proprietary hardware|Cost + deployability|
|**Anduril (Lattice)**|Edge-first fusion, proprietary towers|Runs on existing cameras|Hardware independence|
|**Bosch / Hikvision**|Built-in rule/zone-trigger analytics|We add context reasoning (posture+zone+time)|Context intelligence|
|**Staqu (JARVIS)**|Analytics on existing CCTV (Police/Army)|No border fencing, no offline-first|Border specificity|
|**Innefu Labs**|Face vectors + NN match (Delhi Police)|Extended to low-res long-range CCTV|Low-res adaptation|
|**Tonbo Imaging**|Thermal/electro-optic hardware|We deliver as software, not hardware|Scalability|

_Honest Positioning: We are not the world's best defense-grade C4I platform. We are the most contextually intelligent, pragmatic, software-only solution for existing CCTV at hackathon scale._

## Section 5: Known Loopholes, TODOs & Open Decisions

### 5.1 Loophole Priority Response Plan

|Priority|Loophole / Gap|Prepared Response|
|---|---|---|
|**URGENT**|No demo plan for internal round|Pre-recorded test footage + live webcam simulation + staged crawling/climbing footage.|
|**URGENT**|Compute budget unconfirmed|Jetson Orin Nano is an estimate. Demo runs on dev laptop. Will benchmark before finals.|
|**LIKELY**|Privacy / DPDP Act 2023|Acknowledge gap. Non-flagged purged in 30 days. Full audit trail. Post-deployment task.|
|**LIKELY**|Evaluation metrics not final|Scenario matrix proposed. Will finalize and test rigorously before finals.|
|**MEDIUM**|Model drift / retraining|Model versioning exists in hashes. Health monitor flags degradation. Retraining is Phase 2 roadmap.|
|**MEDIUM**|Dataset licensing|Using open academic datasets for prototype. Will verify commercial licenses before finals.|
|**MEDIUM**|Physical attacks on camera|Health Monitor detects spray paint/lasers (DARKNESS/OBSTRUCTION flag). Manual response required.|
|**RESOLVED**|Team role assignment|Roles mapped via `07A_Model_A` and `07B_Model_B` delegations.|

### 5.2 Blocking TODOs

- [ ] Verify dataset licenses (UNIRI-TID, PDIWS, SCface, NightOwls, CAMEL, CVC-14). Switch to COCO/WIDER FACE if blocked.
    
- [ ] Decide demo footage plan (staged vs. pre-recorded vs. live webcam).
    
- [ ] Freeze `schema_v1` before parallel builds start.
    
- [ ] Usability-test the Dashboard on a non-technical person.
    

### 5.3 Open Decisions Awaiting Owner's Call

1. **Fast-path severity:** Two-stage (provisional → confirmed) vs. single-stage CRITICAL tagging. _(Note: schema supports both)._
    
2. **Context Profile Scope:** Build full 4-force profiles, or build 1-2 for demo and roadmap the rest? _(Recommendation: Build BSF & SSB for sharpest contrast)._
    
3. **ANPR Perspective Correction:** Does the team actually have homography/camera-calibration skills, or is this scope creep?
    

## Section 6: Disaster Recovery & Demo Contingencies

### 6.1 Disaster Recovery Protocols

|Disaster Scenario|Recovery Action|
|---|---|
|**Schema changed mid-build, engines incompatible**|Roll back to last frozen version (`schema_v_N-1`). NEVER patch live. Use strict versioning.|
|**One developer's half breaks the night before**|Rely on Fallback Logic ([[03_Architecture_Map]]). Model A alone can demo basic motion + trigger detection.|
|**No demo footage exists by T-minus-3 days**|**Plan B:** Staged footage. Have a teammate walk/crawl toward a webcam on a taped "fence line". Disclose as staged.|
|**GPU / compute unavailable on demo day**|Switch to smaller YOLOv8n at reduced resolution, OR run a pre-recorded inference pass and play it back as video.|
|**Dataset licensing turns out to be blocked**|Swap to publicly licensed alternatives (COCO, WIDER FACE) immediately.|

### 6.2 Demo Execution Plans

- **Plan A (Full Live):** 3 webcams + 1 edge laptop. Show perimeter walk, crawling trigger, and chokepoint ANPR live.
    
- **Plan B (Hybrid - EXPECTED):** 1 live webcam for interaction + 2 pre-recorded RTSP streams for night/crawling scenarios that can't be staged live.
    
- **Plan C (Pre-Recorded):** All footage pre-recorded, injected via RTSP simulator (`ffmpeg -re -stream_loop`). Dashboard processes as if live.
    
- **Plan D (Screenshots Only):** Walk through Mermaid flows. Emphasize that internal round is about idea viability and integration is in progress.
    

### 6.3 Demo Checklist

- [ ] RTSP stream ingestion working.
    
- [ ] Model A running without crashes.
    
- [ ] At least 1 Model B engine producing events.
    
- [ ] MQTT bus publishing and subscribing.
    
- [ ] Dashboard receiving/displaying events.
    
- [ ] Alert fires within 5 seconds of trigger.
    
- [ ] Fallback mode demonstrable (kill one engine, show Model A taking over).
    
- [ ] Border Context Profile switchable (even if only 2 profiles exist).
    

## Section 7: Pitch Structure & Emergency Responses

### 7.1 Recommended 7-Minute Pitch Structure

1. **Hook (30s):** MHA spent Rs.86 crore on BOLD-QIT. Equipment junked. CIBMS: 50 false alerts/night. The problem is not sensors — it is intelligence.
    
2. **Problem (60s):** Hardware-heavy approach fails in remote conditions. Jawans can't operate it.
    
3. **Solution (90s):** Software-only layer on existing CCTV. 5 lightweight models, unified bus, offline-first.
    
4. **Demo (120s):** Show tracking, virtual fence, alert, dashboard.
    
5. **Differentiation (60s):** Border Context Profile. Multi-frame confirmation. 25x false alarm reduction. <$50K per BOP.
    
6. **Roadmap (30s):** Internal round → Finals → Real deployment.
    
7. **Ask (30s):** We need qualification to proceed to finals where we will benchmark on target hardware.
    

### 7.2 Emergency Responses (When Caught Off-Guard)

1. **Pause. Breathe.**
    
2. **If you know the answer:** Answer in 2 sentences, then ask: _"Would you like me to elaborate on [specific aspect]?"_
    
3. **If you don't know:** Say: _"That is an excellent question we have not yet fully explored. Our current thinking is [best guess]. We will research this before finals."_ Honesty beats BS.
    
4. **NEVER** make up numbers, cite unverified facts, or claim a PROPOSED feature is already built.
    
5. **If challenged on a competitor:** Acknowledge their strength, pivot to our niche. We aren't better at everything; we are better at _this specific problem_.