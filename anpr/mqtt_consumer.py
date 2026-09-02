"""
MQTT Consumer and Dispatcher for ANPR Engine (Model B).

Subscribes to Model A event topic (e.g. `sih26187/model_a/events` or `sih26187/camera/+/model_a/#`),
filters for chokepoint vehicle detections, runs ANPREngine, and publishes
verified plate events to `sih26187/camera/{cam_id}/model_b/anpr`.
"""

import json
import logging
from typing import Optional, Dict, Any, Callable
from anpr.engine import ANPREngine

logger = logging.getLogger("ANPRMQTTConsumer")


class ANPRMQTTConsumer:
    """
    Subscribes to Model A events over MQTT, executes ANPREngine, and emits Model B ANPR events.
    """

    DEFAULT_SUB_TOPIC = "sih26187/camera/+/model_a/#"
    FALLBACK_SUB_TOPIC = "sih26187/model_a/events"

    def __init__(
        self,
        engine: ANPREngine,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "model_b_anpr_engine",
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.engine = engine
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.client = None

    def handle_message_payload(self, payload_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes single raw Model A event payload and returns generated ANPR event payload (if applicable).
        """
        # Engine validates zone (chokepoint/icp) and entity_type (vehicle)
        output_event = self.engine.process_model_a_event(payload_dict)
        return output_event

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            client.subscribe(self.DEFAULT_SUB_TOPIC)
            client.subscribe(self.FALLBACK_SUB_TOPIC)
            logger.info(f"Subscribed to '{self.DEFAULT_SUB_TOPIC}' and '{self.FALLBACK_SUB_TOPIC}'")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8")
            payload = json.loads(payload_str)
            logger.debug(f"Received event from {msg.topic}: {payload.get('event_id')}")

            output_event = self.handle_message_payload(payload)
            if output_event:
                cam_id = output_event.get("camera_id", "default")
                pub_topic = f"sih26187/camera/{cam_id}/model_b/anpr"
                client.publish(pub_topic, json.dumps(output_event), qos=1)
                logger.info(f"Published ANPR event {output_event['event_id']} to {pub_topic} [Plate: {output_event.get('plate_number')}, Conf: {output_event.get('confidence')}]")
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}", exc_info=True)

    def start(self, blocking: bool = True):
        """Starts the MQTT loop."""
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(client_id=self.client_id)
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message

            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            if blocking:
                self.client.loop_forever()
            else:
                self.client.loop_start()
        except ImportError:
            logger.warning("paho-mqtt library not installed. MQTT runner running in test/mock mode.")
        except Exception as e:
            logger.error(f"Failed to start MQTT consumer: {e}")

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
