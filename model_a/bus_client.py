"""
MQTT Bus Client — Model A Publisher
SIH26187 | Rule #2 enforced: ONE topic structure, no bypass channel.

Topic pattern:  sih26187/camera/{cam_id}/model_a/event
QoS policy:     QoS 1 for info/warning/provisional
                QoS 2 for confirmed/critical   (Rule #11)

LWT (Last Will and Testament):
  If Model A disconnects unexpectedly the broker auto-publishes:
  topic  : sih26187/system/model_a/health
  payload: {"status": "DEAD", "engine": "model_a"}
  This lets Model B and operators detect Model A failure without polling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from model_a.schema_v1 import ModelAEvent, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QoS policy — derived from severity (Rule #11)
# ---------------------------------------------------------------------------

_QOS_MAP = {
    Severity.info:        1,
    Severity.warning:     1,
    Severity.provisional: 1,
    Severity.confirmed:   2,
    Severity.critical:    2,
}


def severity_to_qos(severity: str) -> int:
    """Return MQTT QoS level for a given severity string."""
    return _QOS_MAP.get(Severity(severity), 1)


# ---------------------------------------------------------------------------
# LWT payload
# ---------------------------------------------------------------------------

_LWT_TOPIC   = "sih26187/system/model_a/health"
_LWT_PAYLOAD = json.dumps({"status": "DEAD", "engine": "model_a"}).encode()


# ---------------------------------------------------------------------------
# BusClient
# ---------------------------------------------------------------------------

class BusClient:
    """
    Thin wrapper around paho-mqtt that:
      1. Enforces the single-topic publish contract (Rule #2).
      2. Selects QoS automatically from event severity (Rule #11).
      3. Registers LWT so operators detect unexpected disconnects.
      4. Validates the ModelAEvent schema before touching the wire (Rule #3).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int  = 1883,
        client_id: str    = "model_a_publisher",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
    ) -> None:
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._client_id   = client_id
        self._keepalive   = keepalive

        self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

        if username:
            self._client.username_pw_set(username, password)

        # Last Will and Testament (Rule #12 smart suggestion)
        self._client.will_set(
            topic=_LWT_TOPIC,
            payload=_LWT_PAYLOAD,
            qos=1,
            retain=True,
        )

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish    = self._on_publish

        self._connected = threading.Event()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 10.0) -> None:
        """Connect to the MQTT broker and start the network loop."""
        self._client.connect(self._broker_host, self._broker_port, self._keepalive)
        self._client.loop_start()
        if not self._connected.wait(timeout=timeout):
            raise TimeoutError(
                f"Could not connect to MQTT broker at "
                f"{self._broker_host}:{self._broker_port} within {timeout}s"
            )
        logger.info("BusClient connected to %s:%s", self._broker_host, self._broker_port)

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("BusClient disconnected.")

    # ------------------------------------------------------------------
    # Publish — the ONLY way events leave Model A
    # ------------------------------------------------------------------

    def publish_event(self, event: ModelAEvent) -> mqtt.MQTTMessageInfo:
        """
        Validate → serialise → publish one ModelAEvent.

        The topic is ALWAYS:
            sih26187/camera/{cam_id}/model_a/event
        There is no other topic. No shortcuts. (Rule #2)

        Raises ValidationError if the event is malformed — caller must catch,
        log the rejection, and NEVER attempt to publish raw/un-validated dicts.
        """
        # Schema is already validated by Pydantic at construction time,
        # but we re-serialise here to catch any mutation after construction.
        payload = event.to_mqtt_payload()

        topic = f"sih26187/camera/{event.camera_id}/model_a/event"
        qos   = severity_to_qos(event.severity)

        info = self._client.publish(topic=topic, payload=payload, qos=qos)

        logger.debug(
            "Published event_id=%s severity=%s qos=%d topic=%s mid=%d",
            event.event_id, event.severity, qos, topic, info.mid,
        )
        return info

    # ------------------------------------------------------------------
    # Subscribe — for round-trip testing and health monitoring
    # ------------------------------------------------------------------

    def subscribe_events(
        self,
        cam_id: str,
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        Subscribe to a camera's Model A event stream.
        callback(topic, payload_dict) is called for each received message.
        """
        topic = f"sih26187/camera/{cam_id}/model_a/event"

        def _on_message(_client, _userdata, msg: mqtt.MQTTMessage) -> None:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                callback(msg.topic, data)
            except json.JSONDecodeError as exc:
                logger.error("Received malformed JSON on %s: %s", msg.topic, exc)

        self._client.subscribe(topic, qos=1)
        self._client.on_message = _on_message
        logger.info("Subscribed to %s", topic)

    # ------------------------------------------------------------------
    # Paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected.set()
            logger.info("MQTT connection established (rc=0).")
        else:
            logger.error("MQTT connection failed with rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()
        if rc != 0:
            logger.warning("Unexpected MQTT disconnect (rc=%d). LWT will fire.", rc)

    def _on_publish(self, client, userdata, mid) -> None:
        logger.debug("MQTT ACK received for mid=%d", mid)
