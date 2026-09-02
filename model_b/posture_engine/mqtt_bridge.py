"""
mqtt_bridge.py — MQTT wiring for the Posture Engine.

Responsibilities:
  - Subscribe to sih26187/camera/+/model_a/raw
  - Filter: entity_type == "human" only (any zone_tag — both close_range and long_range)
  - Call PostureEngine.process() per qualifying event
  - Publish result to sih26187/camera/{cam_id}/model_b/posture
  - Publish 10s heartbeat to sih26187/orchestrator/health
  - Graceful shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

import config as cfg
from posture_engine import PostureEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mqtt_bridge")


class MQTTBridge:
    def __init__(self) -> None:
        self._engine = PostureEngine()

        self._client = mqtt.Client(client_id=cfg.MQTT_CLIENT_ID)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._running = False
        self._heartbeat_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("Connecting to MQTT broker %s:%d", cfg.MQTT_BROKER_HOST, cfg.MQTT_BROKER_PORT)
        self._client.connect(cfg.MQTT_BROKER_HOST, cfg.MQTT_BROKER_PORT, cfg.MQTT_KEEPALIVE)

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

        self._client.loop_forever()   # blocks until stop()

    def stop(self) -> None:
        logger.info("Shutting down MQTTBridge...")
        self._running = False
        self._client.disconnect()
        self._client.loop_stop()
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

        # ── Filter: human only (any zone_tag) ─────────────────────────────
        if payload.get("entity_type") != "human":
            return

        cam_id: str = payload.get("camera_id", "")
        if not cam_id:
            logger.warning("Event missing camera_id — skipped")
            return

        zone_tag: str = payload.get("zone_tag", "")
        entity_id = payload.get("entity_id")   # may be null — pass through as-is
        bbox = payload.get("bbox", [0.0, 0.0, 1.0, 1.0])
        evidence_ref: str = payload.get("evidence_ref", "")
        timestamp: str = payload.get("timestamp", datetime.now(timezone.utc).isoformat())

        if not evidence_ref:
            logger.warning("Event missing evidence_ref for cam %s — skipped", cam_id)
            return

        # ── Run engine ─────────────────────────────────────────────────────
        try:
            event = self._engine.process(
                camera_id=cam_id,
                zone_tag=zone_tag,
                entity_id=entity_id,
                bbox_norm=bbox,
                evidence_ref=evidence_ref,
                timestamp=timestamp,
            )
        except Exception:
            logger.exception("PostureEngine.process() raised for cam %s", cam_id)
            return

        if event is None:
            return  # No pose detected — nothing to publish

        # ── Publish ────────────────────────────────────────────────────────
        topic = cfg.TOPIC_PUBLISH_TEMPLATE.format(cam_id=cam_id)
        self._client.publish(topic, json.dumps(event), qos=1)
        logger.debug("Published posture_anomaly for cam %s entity %s posture=%s",
                     cam_id, entity_id, event["metadata"]["posture_class"])

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Publish engine health every HEARTBEAT_INTERVAL_S seconds."""
        while self._running:
            payload = json.dumps({
                "engine": cfg.ENGINE_NAME,
                "model_version": cfg.MODEL_VERSION,
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
