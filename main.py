"""
SIH26187 Model B - ANPR Engine Demo & Verification Script.

Executes dummy event consumption against Model A outputs and prints the contract payloads.
"""

import json
import numpy as np
from anpr.engine import ANPREngine


def run_demo():
    print("=" * 70)
    print("SIH26187 Model B - ANPR Engine Pipeline Demo")
    print("=" * 70)

    engine = ANPREngine(detector_path=None, ocr_model_path=None)

    # 1. Dummy Events from Model A
    dummy_events = [
        {
            "name": "Dummy Event A (Fence Climbing - Perimeter)",
            "payload": {
                "event_id": "a8f3b612-4c2e-49b8-93d1-e5d871a2c001",
                "engine_source": "model_a",
                "event_type": "trigger",
                "severity": "critical",
                "timestamp": "2026-09-02T11:20:15.340Z",
                "camera_id": "cam_perimeter_04",
                "zone_tag": "close_range",
                "zone": "fence",
                "entity_type": "human",
                "entity_id": "global_fusion_8f90c1",
                "confidence": 0.89,
                "bbox": [0.35, 0.42, 0.58, 0.91]
            }
        },
        {
            "name": "Dummy Event B (Animal Detection - Open Border)",
            "payload": {
                "event_id": "5c12d48a-112e-4f56-8a02-b349071c8902",
                "engine_source": "model_a",
                "event_type": "animal_detected",
                "severity": "info",
                "timestamp": "2026-09-02T11:21:04.112Z",
                "camera_id": "cam_perimeter_02",
                "zone_tag": "long_range",
                "zone": "open_border",
                "entity_type": "animal",
                "entity_id": "trk_deer_01",
                "confidence": 0.76,
                "bbox": [0.65, 0.12, 0.78, 0.28]
            }
        },
        {
            "name": "Dummy Event C (Animal-Cart - Chokepoint)",
            "payload": {
                "event_id": "9b3c4d1e-8e7a-4290-b1cf-098712345678",
                "engine_source": "model_a",
                "event_type": "animal_detected",
                "severity": "info",
                "timestamp": "2026-09-02T11:22:45.000Z",
                "camera_id": "cam_chokepoint_01",
                "zone_tag": "close_range",
                "zone": "chokepoint",
                "entity_type": "animal_cart",
                "entity_id": "trk_horse_01",
                "confidence": 0.81,
                "bbox": [0.10, 0.70, 0.45, 0.90]
            }
        },
        {
            "name": "Dummy Event D (Vehicle - Chokepoint Straight Plate)",
            "payload": {
                "event_id": "e4a2b1c0-1111-2222-3333-444455556666",
                "engine_source": "model_a",
                "event_type": "motion",
                "severity": "info",
                "timestamp": "2026-09-02T11:24:00.120Z",
                "camera_id": "cam_chokepoint_01",
                "zone_tag": "close_range",
                "zone": "chokepoint",
                "entity_type": "vehicle",
                "entity_id": "trk_truck_08",
                "confidence": 0.95,
                "bbox": [0.20, 0.25, 0.80, 0.85]
            }
        }
    ]

    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for item in dummy_events:
        print(f"\n--- Testing: {item['name']} ---")
        event = item["payload"]
        output = engine.process_model_a_event(event, frame_override=mock_frame)
        if output is None:
            print(f"-> Filtered OUT (Out of ANPR scope: zone='{event.get('zone')}', entity='{event.get('entity_type')}')")
        else:
            print("-> Successfully PROCESSED. Emitted MQTT Payload:")
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_demo()
