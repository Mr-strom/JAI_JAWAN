# Phase Zurich Report — Model A Manual Verification (Real Footage)

**Date:** 2026-09-01
**Tool used:** `video_test/manual_verify.py`
**Scope:** End-to-end run of the real pipeline (YOLOv8n detector + real TriggerDetector state machine)
against real/synthetic CCTV-style footage, bypassing all synthetic mocks used in the earlier unit-level audit.

**Note on test conditions:** `trigger_type_override=TriggerType.climbing` is hardcoded in the manual
verification script for every frame, since Model B (posture classification) is not yet built. This means
these runs verify persistence/tracking-based confirmation logic only — they do not verify that the system
can distinguish climbing from other sustained motion (e.g. walking). See Divergence #2 below.

## Summary by clip

| Clip | Frames | Trigger events | Tracks | Animal suppression | Schema failures | Result |
|---|---|---|---|---|---|---|
| static | 240 | 0 | 0 | 0 | 0 | PASS — clean negative control |
| bird/glitch | 240 | 0 | 0 | 8 | 0 | PASS — animal filter correctly suppressed before reaching trigger logic |
| climbing | 240 | 34 | 3 | 0 | 0 | PASS — persistence-based confirmation fires within expected frame windows (e.g. frame 6, 37, 148) |
| walking | 240 | 12 | 1 | 0 | 0 | DIVERGENCE — see #2 below |
| combined | 1200 | 47 | 4 | 9 | 0 | PASS on reset behavior — animal suppression correctly re-engaged for the bird segment (frames 1038–1050) mid-way through the combined timeline; no evidence of stale state carrying over between segments |

Schema validation failures: 0 across all runs.
CLAHE (low-light preprocessing) engagement: 0 across all runs — expected, since all test clips were
daytime/well-lit; not yet verified on a genuine low-light clip.

## Divergences Found

### Divergence #1 — Trigger events re-fire repeatedly instead of firing once per event

Observed in every clip with a confirmed trigger. Example (climbing clip, track `7a180662...`):

```
Frame 211: CONFIRMED
Frame 216: CONFIRMED
Frame 220: CRITICAL
Frame 223: CRITICAL
Frame 226: CRITICAL
Frame 229: CRITICAL
Frame 232: CRITICAL
Frame 235: CRITICAL
Frame 237: CRITICAL
Frame 239: CRITICAL
```

Once a track reaches CONFIRMED/CRITICAL, the pipeline continues re-publishing an event every few frames
for as long as the track stays alive, rather than firing once (or firing once per severity escalation and
then going quiet). At real camera frame rates this would produce dozens of duplicate alerts for a single
ongoing event — the same operator-facing failure mode (alert flooding) that the multi-frame confirmation
logic was originally built to prevent, just occurring after confirmation instead of before it.

**Action needed:** add a "fire once per state, then suppress until state changes or track ends" guard in
the trigger publishing logic. Per project rule, this is a logic gap, not a threshold to retune.

### Divergence #2 — Walking clip produced a CONFIRMED/CRITICAL trigger sequence

The walking clip (no climbing or fence-cutting present) still produced 12 trigger events, confirming at
frame 211. This is a direct consequence of `trigger_type_override=TriggerType.climbing` being hardcoded
for all frames in the manual verification harness — the current pipeline has no way to distinguish
climbing from any other sustained, trackable motion, because Model B (posture/behavior classification)
does not exist yet.

**Action needed:** not a Model A bug — flag as an explicit known limitation until Model B is integrated.
Do not use walking-clip results as evidence against the persistence/confirmation logic itself, which is
working as designed (see climbing clip result).

## Not yet verified

- Low-light / night footage (CLAHE preprocessing engagement never triggered in these runs)
- Long-range vs close-range zone tagging accuracy on real detections (tagging exists per unit audit, not
  re-verified visually against these runs)
- Visual confirmation of bounding-box tracking stability and state-text transitions (per script's own
  guidance) — output `.mp4` files should be reviewed frame-by-frame before this is marked complete

## Diagnostic Findings

### Issue A — Missed detection on visible person (walking clip, Frame 42)
Investigation of raw YOLOv8n output on `walking_clip.mp4` reveals that the person is NOT completely missed by the model. The model does produce a bounding box for the person, but the confidence drops drastically as they move further away:
- Frame 40: `conf = 0.482`
- Frame 41: `conf = 0.051`
- Frame 42: `conf = 0.104`
- Frame 43: `conf = 0.070`
- Frame 46: `conf = 0.496`

Because the production `Detector` threshold is hardcoded to `0.40`, these low-confidence detections (frames 41-44) are discarded before reaching the tracker. This indicates YOLOv8n struggles with this specific scale/distance of human subjects. Lowering the threshold or using a larger model (`YOLOv8s`) / higher input resolution may be required.

### Issue B — Ghost detection (combined clip, Frame 542)
Visual inspection of the `manual_verify.py` output video showed a high-confidence bounding box ("human 0.92") on frame 542 with no person inside it. 

Diagnostic logging confirms this is **NOT** a detector or tracker hallucination, but a rendering artifact in the `manual_verify.py` test harness:
1. `manual_verify.py` intercepts `detector.detect()` to save `last_detections` for rendering bounding boxes on the output video.
2. The pipeline's `TimeSampler` correctly identifies frames in this segment as redundant/static and drops them, returning early.
3. Because the frame is dropped, `detector.detect()` is never called, and `last_detections` is not updated.
4. The test script proceeds to draw the stale bounding boxes from the *last un-skipped frame* onto the current frame.

The underlying model is correctly outputting `0` detections on these frames. The harness needs to clear `last_detections` when a frame is skipped by the sampler.