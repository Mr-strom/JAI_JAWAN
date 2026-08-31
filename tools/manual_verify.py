import argparse
import logging
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

# Adjust path so we can import model_a
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_a.frame_pipeline import FramePipeline
from model_a.zone_tagger import ZoneTagger
from model_a.detector import Detector, Detection
from model_a.schema_v1 import TriggerType, EventType, Severity
from model_a.trigger_detector import TriggerState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DummyBusClient:
    """Mock bus client to catch events without needing a real MQTT broker."""
    def __init__(self):
        self.published_events = []

    def connect(self): pass
    def disconnect(self): pass
    
    def publish_event(self, event):
        self.published_events.append(event)


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def run_verification(input_path: str, output_path: str):
    logger.info(f"Starting manual verification. Input: {input_path}")
    
    # Check if input is a webcam ID
    try:
        source = int(input_path)
    except ValueError:
        source = input_path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: # handle NaN or 0
        fps = 25.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video format: {width}x{height} @ {fps}fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # ---------------------------------------------------------
    # Initialize the REAL pipeline
    # We bypass synthetic generators and use the actual classes
    # ---------------------------------------------------------
    camera_id = "cam_manual_verify"
    zone_tagger = ZoneTagger(camera_id, frame_height_px=height)
    
    logger.info("Loading YOLOv8n detector...")
    try:
        detector = Detector("yolov8n.pt")
    except Exception as e:
        logger.error(f"Failed to load Detector: {e}")
        return

    bus_client = DummyBusClient()
    
    pipeline = FramePipeline(
        camera_id=camera_id,
        zone_tagger=zone_tagger,
        detector=detector,
        bus_client=bus_client
    )
    
    # Intercept detector to capture detections for drawing, since the pipeline 
    # process() method encapsulates them and may not return them if no event fires.
    last_detections: List[Detection] = []
    original_detect = pipeline._detector.detect
    
    def intercept_detect(frame_bgr, frame_number):
        nonlocal last_detections
        dets = original_detect(frame_bgr, frame_number)
        last_detections = dets
        return dets
        
    pipeline._detector.detect = intercept_detect

    # Intercept preprocessor to track if it engaged
    preprocessor_engaged_frames = []
    original_enhance = pipeline._pre.enhance
    def intercept_enhance(frame_bgr):
        # The preprocessor returns a new array if it enhanced it
        enhanced = original_enhance(frame_bgr)
        # simplistic check if enhanced (in real code, we could check luminance)
        if enhanced is not frame_bgr and not np.array_equal(enhanced, frame_bgr):
            preprocessor_engaged_frames.append(frame_num)
        return enhanced
    pipeline._pre.enhance = intercept_enhance
    
    # Track animal filter
    animal_suppressed_frames = []

    frame_num = 0
    trigger_events_fired = []
    schema_failures = 0

    logger.info(f"Writing annotated output to {output_path}")

    base_time_s = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        frame_time_s = base_time_s + (frame_num / fps)
        # Format timestamp with milliseconds precision ISO 8601 UTC
        gm = time.gmtime(frame_time_s)
        millis = int((frame_time_s % 1.0) * 1000)
        timestamp_utc = f"{time.strftime('%Y-%m-%dT%H:%M:%S', gm)}.{millis:03d}Z"
        
        # Keep track of events before
        events_before = len(bus_client.published_events)
        
        # ---------------------------------------------------------
        # Run the real pipeline
        # (trigger_type_override=TriggerType.climbing to simulate Model B posture)
        # ---------------------------------------------------------
        try:
            published_events = pipeline.process(
                frame=frame.copy(), 
                frame_number=frame_num,
                timestamp_utc=timestamp_utc,
                trigger_type_override=TriggerType.climbing 
            )
        except Exception as e:
            logger.error(f"Pipeline error on frame {frame_num}: {e}")
            schema_failures += 1
            published_events = []

        # Analyze events for tracking
        new_events = bus_client.published_events[events_before:]
        for ev in new_events:
            sev_str = ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity)
            ev_type_str = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
            if ev_type_str == "trigger" or ev.event_type == EventType.trigger:
                trigger_events_fired.append((frame_num, sev_str, ev.entity_id))
            elif ev_type_str == "animal_detected" or ev.event_type == EventType.animal_detected:
                animal_suppressed_frames.append(frame_num)
                
        # ---------------------------------------------------------
        # Render visualizations
        # ---------------------------------------------------------
        display_frame = frame.copy()
        
        # State colors mapping
        state_colors = {
            "IDLE": (0, 255, 0),         # Green
            "PROVISIONAL_1": (0, 255, 255), # Yellow
            "PROVISIONAL_2": (0, 165, 255), # Orange
            "CONFIRMED_TRIGGER": (0, 0, 255), # Red
            "COOLDOWN": (255, 0, 0)      # Blue
        }

        # Draw detections
        for det in last_detections:
            x1 = int(det.bbox[0] * width)
            y1 = int(det.bbox[1] * height)
            x2 = int(det.bbox[2] * width)
            y2 = int(det.bbox[3] * height)
            
            # Get track state
            state_name = "IDLE"
            if det.track_id and det.track_id in pipeline._trig._tracks:
                state_name = pipeline._trig._tracks[det.track_id].state.name
                
            color = state_colors.get(state_name, (200, 200, 200))
            
            # BBox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Zone Tagging
            zone_tag, _ = pipeline._zone.tag(det.bbox)
            
            # Labels
            label1 = f"{det.entity_type.value} {det.confidence:.2f}"
            label2 = f"State: {state_name}"
            label3 = f"Zone: {zone_tag.value}"
            
            draw_text(display_frame, label1, (x1, max(20, y1 - 40)), color)
            draw_text(display_frame, label2, (x1, max(20, y1 - 20)), color)
            draw_text(display_frame, label3, (x1, max(20, y1 - 5)), color)

        # Draw frame info
        draw_text(display_frame, f"Frame: {frame_num}", (20, 30))
        draw_text(display_frame, f"Time: {timestamp_utc}", (20, 60))
        draw_text(display_frame, f"Active Tracks: {len(pipeline._trig.active_tracks)}", (20, 90))

        # Console log (every 10 frames or if trigger state changes - simplified to every 15 for less spam)
        if frame_num % 15 == 0 or new_events:
            tracks_str = ", ".join([f"{tid[:6]}={info['state']}" for tid, info in pipeline._trig.active_tracks.items()])
            logger.info(f"Frame {frame_num:04d} | Detections: {len(last_detections)} | Tracks: [{tracks_str}] | Events published: {len(new_events)}")

        out.write(display_frame)
        
        # Optional: cv2.imshow("Verification", display_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    
    # ---------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🎥 MANUAL VERIFICATION SUMMARY")
    print("="*50)
    print(f"Total frames processed: {frame_num}")
    print(f"Total trigger events fired: {len(trigger_events_fired)}")
    for f_num, sev, eid in trigger_events_fired:
        print(f"  - Frame {f_num}: {sev.upper()} (track {eid})")
        
    print(f"\nFrames with Preprocessing (CLAHE) engaged: {len(preprocessor_engaged_frames)}")
    if preprocessor_engaged_frames:
        print(f"  First few: {preprocessor_engaged_frames[:5]}")
        
    print(f"Frames with Animal Filter suppression: {len(animal_suppressed_frames)}")
    if animal_suppressed_frames:
        print(f"  First few: {animal_suppressed_frames[:5]}")
        
    print(f"Schema validation failures: {schema_failures}")
    if schema_failures > 0:
        print("  ⚠️ FLAG: Schema validation failed on real data. Check logs.")
        
    # Check for divergences
    # A divergence could be: trigger confirmed, but then reset while still visible.
    # We can advise the user to watch the output video.
    print("\n🔍 DIVERGENCE CHECK")
    print("This tool processes real footage bypassing all synthetic generators.")
    print("Watch the output video to visually confirm:")
    print("  1. Does the bounding box reliably track the person?")
    print("  2. Does the state text transition smoothly (IDLE -> PROV_1 -> PROV_2 -> CONFIRM)?")
    print("  3. If it flickers back to IDLE frequently during approach, the IoU threshold (0.35)")
    print("     is being breached by the real bbox growth/jitter.")
    print("\nDO NOT SILENTLY RETUNE. If you see erratic resets, report it as a divergence in")
    print("docs/phase_zurich_report.md under 'Divergence Found'.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual verification tool for real video.")
    parser.add_argument("--input", required=True, help="Path to input video file or 0 for webcam.")
    parser.add_argument("--output", required=True, help="Path to output annotated .mp4 file.")
    args = parser.parse_args()
    
    run_verification(args.input, args.output)
