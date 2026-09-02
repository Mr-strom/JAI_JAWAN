As of: 2026-09-02 · Test suite: 161 passed, 3 skipped, 0 failed (164 collected; skips are live-broker-only tests) · Source: verified against live codebase and git diff, not written from memory.

Summary Verdict

Model A's required scope (07A) is fully built and tested. Both owner-approved proposed extensions (animal-cart, homography) are built and unit-tested, not yet real-footage-verified. All three locked files are confirmed byte-for-byte untouched via git diff. The system has not yet been integration-tested against real Model B code — only against a mock subscriber harness. Three decisions are now waiting on you.

1. Ingestion
Item	Status
Real video file ingestion (cv2.VideoCapture, not synthetic)	✅ Built, and confirmed run against real clips (walking_clip.mp4, climbing_clip.mp4, bird_clip.mp4, combined_clip.mp4, static_clip.mp4)
Multi-camera handling (per-camera_id pipeline instances)	✅ Built
Multi-camera fusion (fusion_engine.py, IoU-based merge)	✅ Built, 4/4 tests passing
2. Model A Core Functions (07A required scope)
#	Function	Status	Tests
1	Time-sampling (MSE dedup)	✅ Present	4/4 passing
2	MDF selection	✅ Present	1/1 passing
3	Zone tagging (close/long range)	✅ Present	3/3 passing
4	Multi-frame trigger confirmation	✅ Present	18/18 passing
5	Schema normalization	✅ Present	21/21 passing
6	Dual outputs — real-time events	✅ Present, fully tested	MQTT stream 100% built
6b	Dual outputs — training dataset export	⚠️ Partial — evidence saving hooks exist (evidence_ref, hash, disk dir), but no automated bundler/export script packages them into a usable training dataset	
3. Supporting Infrastructure — all present and enforced

bus_client.py (single topic, no bypass, QoS-by-severity) · fallback_router.py (camera-scoped fallback, 17/17 tests) · safety_floor.py (3-frame floor even in fallback) · anti_spoofing.py (timestamp/continuity checks) · health_monitor.py (CPU/memory/FPS tracking) — no gaps found.

4. Integration Readiness
Item	Status
Mock Model B subscriber harness	✅ Built, 22/23 passing (1 skipped, needs live broker)
ZURICH-18 IoU divergence	🟡 Still open — documented, not silently patched, awaiting your call (see §Decisions below)
Real Model B integration	❌ Has not happened. Only the mock harness has been exercised. This is the single biggest remaining unknown in the whole project.
5. Proposed Extensions (owner-approved, built this session)
Extension	Built	Real-footage verified	Notes
Animal-cart fusion	✅ 25/25 tests	❌ No — no animal+cart clip exists yet in video_test/	2-frame proximity fusion logic
Homography correction	✅ 19/19 tests	❌ No — synthetic/unit tests only	homography_config.json currently holds placeholder coordinates only — real site-survey calibration still needed before this can be trusted on live cameras
6. Locked Files — Integrity Confirmed

schema_v1.py, bus_client.py, trigger_detector.py — git diff against each returned zero changes. No drift.

7. Genuinely Not Done (no soft-pedaling)
Automated training-dataset export/bundler script
Real Model B integration (only mock-tested so far)
Animal-cart logic never run against real footage
Homography calibration is placeholder data, not real camera measurements
combined_clip/climbing_clip are now producing more trigger events (13 vs. 8 baseline) since YOLOv8s picks up distant targets YOLOv8n missed — not yet classified as true positives vs. false positives
Decisions Waiting on You
ZURICH-18 IoU threshold — pick one: (A) lower threshold 0.35→0.25, (B) switch to EMA bbox tracking, (C) accept occasional resets since confirmation still fires once the target reaches the fence.
Extra trigger events on combined/climbing clips — are the 5 additional detections real long-range catches worth keeping, or noise that needs re-tuning?
Paho MQTT v2 deprecation warning — approve a one-line update to bus_client.py (a locked file) to silence it, or leave as-is since it's a warning, not a failure?

None of these are blocking the demo today. All three should be resolved before Rio (final rehearsal), not left for the night before.
