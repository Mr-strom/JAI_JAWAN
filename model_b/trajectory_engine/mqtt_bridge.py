"""
mqtt_bridge.py — MQTT wiring for the Trajectory Engine.

Responsibilities:
  - Connect to broker, subscribe to sih26187/camera/+/model_a/raw
  - Filter: only process zone_tag == "long_range"
  - Manage per-cam CameraStream instances
  - Load camera_config.json (zone polygons) keyed by cam_id
  - On each qualifying event: get frame → call TrajectoryEngine.process() → publish result
  - Publish a health heartbeat every 10s to sih26187/orchestrator/health
  - Graceful shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import paho.mqtt.client as mqtt

import config as cfg
from camera_stream import CameraStream
from trajectory_engine import TrajectoryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mqtt_bridge")

# Shared state — read by api.py for the /health endpoint.
# Written only by _heartbeat_loop; read-only everywhere else.
ENGINE_STATE: dict = {
    "status": "starting",
    "last_heartbeat": None,
    "active_tracks": 0,
}


# ─── Camera config loader ─────────────────────────────────────────────────────

def _load_camera_config(path: str) -> Dict[str, dict]:
    """
    Load camera_config.json.
    Format: { "cam_id": { "rtsp_url": "...", "zone_polygons": {...} } }
    Returns empty dict if file missing (engine still works, just no zone data).
    """
    p = Path(path)
    if not p.exists():
        logger.warning("camera_config.json not found at %s — zone polygons unavailable", path)
        return {}
    with p.open() as f:
        return json.load(f)


# ─── MQTT Bridge ─────────────────────────────────────────────────────────────

class MQTTBridge:
    def __init__(self) -> None:
        self._engine = TrajectoryEngine()
        self._camera_config = _load_camera_config(cfg.CAMERA_CONFIG_PATH)
        self._streams: Dict[str, CameraStream] = {}   # cam_id → CameraStream
        self._lock = threading.Lock()                  # guards _streams

        self._client = mqtt.Client(client_id=cfg.MQTT_CLIENT_ID)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("Connecting to MQTT broker %s:%d", cfg.MQTT_BROKER_HOST, cfg.MQTT_BROKER_PORT)
        self._client.connect(cfg.MQTT_BROKER_HOST, cfg.MQTT_BROKER_PORT, cfg.MQTT_KEEPALIVE)

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

        # Blocks until stop() is called (runs the MQTT network loop)
        self._client.loop_forever()

    def stop(self) -> None:
        logger.info("Shutting down MQTTBridge...")
        self._running = False
        self._client.disconnect()
        self._client.loop_stop()

        # Release all camera streams
        with self._lock:
            for cam_id, stream in self._streams.items():
                stream.release()
                logger.info("Released stream for cam %s", cam_id)
            self._streams.clear()

        self._engine.cleanup()
        logger.info("MQTTBridge shut down.")

    # ── MQTT Callbacks ────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            logger.info("Connected to broker. Subscribing to %s", cfg.TOPIC_SUBSCRIBE)
            client.subscribe(cfg.TOPIC_SUBSCRIBE, qos=1)
        else:
            logger.error("Failed to connect, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            logger.warning("Unexpected disconnect (rc=%d). Paho will auto-reconnect.", rc)

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Non-JSON payload on %s — skipped", msg.topic)
            return

        # ── Filter: long_range only ────────────────────────────────────
        if payload.get("zone_tag") != "long_range":
            return

        cam_id: str = payload.get("camera_id", "")
        if not cam_id:
            logger.warning("Event missing camera_id — skipped")
            return

        timestamp: str = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
        entity_type: str = payload.get("entity_type", "unknown")
        evidence_ref_in: str = payload.get("evidence_ref", "")

        # ── Get (or start) camera stream ───────────────────────────────
        stream = self._get_or_start_stream(cam_id)
        if stream is None:
            logger.warning("No stream available for cam %s — skipped", cam_id)
            return

        frame = stream.get_frame()
        if frame is None:
            logger.warning("No frame from cam %s — skipped this event", cam_id)
            return

        # ── Camera metadata (zone polygons) ────────────────────────────
        camera_metadata = self._camera_config.get(cam_id, {})

        # ── Run engine ─────────────────────────────────────────────────
        try:
            event = self._engine.process(
                frame=frame,
                camera_id=cam_id,
                timestamp=timestamp,
                entity_type=entity_type,
                evidence_ref_in=evidence_ref_in,
                camera_metadata=camera_metadata,
            )
        except Exception:
            logger.exception("TrajectoryEngine.process() raised for cam %s", cam_id)
            return

        if event is None:
            return  # No active tracks this frame — nothing to publish

        # ── Publish ────────────────────────────────────────────────────
        topic = cfg.TOPIC_PUBLISH_TEMPLATE.format(cam_id=cam_id)
        payload_out = event.model_dump_json()
        self._client.publish(topic, payload_out, qos=1)
        logger.debug("Published trajectory_update for cam %s track %s", cam_id, event.entity_id)

    # ── Camera stream management ──────────────────────────────────────────

    def _get_or_start_stream(self, cam_id: str) -> Optional[CameraStream]:
        with self._lock:
            if cam_id in self._streams and self._streams[cam_id].is_alive():
                return self._streams[cam_id]

            # Look up RTSP URL from config
            cam_cfg = self._camera_config.get(cam_id, {})
            rtsp_url = cam_cfg.get("rtsp_url", "")
            if not rtsp_url:
                logger.error("No rtsp_url configured for cam %s in camera_config.json", cam_id)
                return None

            stream = CameraStream(source=rtsp_url, cam_id=cam_id)
            if not stream.start():
                return None

            self._streams[cam_id] = stream
            return stream

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Publish engine health every HEARTBEAT_INTERVAL_S seconds."""
        while self._running:
            now = datetime.now(timezone.utc).isoformat()
            active_tracks = self._engine.get_active_track_count()

            # Update shared state for the /health API endpoint
            ENGINE_STATE["status"] = "healthy"
            ENGINE_STATE["last_heartbeat"] = now
            ENGINE_STATE["active_tracks"] = active_tracks

            payload = json.dumps({
                "engine": cfg.ENGINE_NAME,
                "model_version": cfg.MODEL_VERSION,
                "status": "healthy",
                "timestamp": now,
                "active_cameras": list(self._streams.keys()),
                "active_tracks": active_tracks,
            })
            try:
                self._client.publish(cfg.TOPIC_HEALTH, payload, qos=0)
            except Exception:
                logger.warning("Heartbeat publish failed (broker may be disconnected)")
            time.sleep(cfg.HEARTBEAT_INTERVAL_S)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    bridge = MQTTBridge()

    def _shutdown(sig, frame):
        logger.info("Signal %s received — stopping.", sig)
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bridge.start()


if __name__ == "__main__":
    main()
