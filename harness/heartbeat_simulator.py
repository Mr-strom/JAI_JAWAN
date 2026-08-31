"""
Heartbeat Simulator — Mock Model B Engine Heartbeat Publisher
SIH26187 | Phase Zurich | Part 1, Item 5

PURPOSE
  Simulates a Model B engine publishing regular heartbeat messages so that
  FallbackRouter.update_heartbeat() can be driven from a REALISTIC source
  (actual MQTT publish + subscribe cycle) instead of the unit-test mock
  (direct method call with no network path).

  This lets you independently verify NORMAL → FALLBACK → RECOVERING behaviour
  against a real heartbeat source by starting/pausing/stopping the simulator.

HEARTBEAT TOPIC
  sih26187/engine/{engine_id}/heartbeat

HEARTBEAT PAYLOAD
  {
    "engine_id":  "face_engine",
    "status":     "alive",
    "timestamp":  "2026-08-31T11:00:00.000Z",
    "interval_s": 10.0
  }

USAGE — as a script (terminal 1):
  python harness/heartbeat_simulator.py --engine face_engine --interval 5

USAGE — from Python (in harness or tests):
  sim = HeartbeatSimulator(engine_id="face_engine", interval_s=5.0)
  sim.start()
  ...
  sim.pause(duration_s=35.0)   # simulate 35s gap → triggers fallback
  ...
  sim.resume()
  sim.stop()

USAGE — HeartbeatListener (pairs with FallbackRouter):
  listener = HeartbeatListener(fallback_router, broker_host="localhost")
  listener.connect()
  # Now FallbackRouter.update_heartbeat() is called automatically on each beat.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import threading
import time
import uuid
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# Heartbeat topic template
_HEARTBEAT_TOPIC = "sih26187/engine/{engine_id}/heartbeat"


# ---------------------------------------------------------------------------
# HeartbeatSimulator — publishes beats
# ---------------------------------------------------------------------------

class HeartbeatSimulator:
    """
    Publishes periodic heartbeat messages on behalf of a named Model B engine.

    Can be paused to simulate a stale heartbeat (triggering FallbackRouter's
    NORMAL → FALLBACK transition) and resumed to test RECOVERING → NORMAL.

    Thread-safe: start()/stop()/pause()/resume() may be called from any thread.
    """

    def __init__(
        self,
        engine_id:   str,
        interval_s:  float = 10.0,
        broker_host: str   = "localhost",
        broker_port: int   = 1883,
        client_id:   str   = "",
    ) -> None:
        self.engine_id  = engine_id
        self.interval_s = interval_s
        self._topic     = _HEARTBEAT_TOPIC.format(engine_id=engine_id)

        client_id = client_id or f"hb_sim_{engine_id}_{uuid.uuid4().hex[:6]}"
        self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._broker_host = broker_host
        self._broker_port = broker_port

        self._connected = threading.Event()
        self._stop_event   = threading.Event()
        self._paused_event = threading.Event()   # set = paused, clear = running
        self._thread: Optional[threading.Thread] = None

        self._beats_sent   = 0
        self._paused_until: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to broker and begin publishing heartbeats."""
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._connected.wait(timeout=5.0):
            raise TimeoutError(
                f"HeartbeatSimulator could not connect to broker "
                f"{self._broker_host}:{self._broker_port}"
            )

        self._stop_event.clear()
        self._paused_event.clear()
        self._thread = threading.Thread(
            target=self._beat_loop,
            name=f"HBSim-{self.engine_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "HeartbeatSimulator started: engine=%s interval=%.1fs topic=%s",
            self.engine_id, self.interval_s, self._topic,
        )

    def stop(self) -> None:
        """Stop publishing and disconnect."""
        self._stop_event.set()
        self._paused_event.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._client.loop_stop()
        self._client.disconnect()
        logger.info(
            "HeartbeatSimulator stopped: engine=%s beats_sent=%d",
            self.engine_id, self._beats_sent,
        )

    def pause(self, duration_s: Optional[float] = None) -> None:
        """
        Pause heartbeat publishing to simulate a stale heartbeat.
        If duration_s is given, auto-resume after that many seconds.
        """
        self._paused_until = (
            time.monotonic() + duration_s if duration_s else None
        )
        self._paused_event.set()
        logger.warning(
            "HeartbeatSimulator PAUSED: engine=%s duration=%s "
            "— FallbackRouter should enter FALLBACK after %.1fs",
            self.engine_id,
            f"{duration_s:.1f}s" if duration_s else "indefinite",
            duration_s or float("inf"),
        )

    def resume(self) -> None:
        """Resume heartbeat publishing."""
        self._paused_until = None
        self._paused_event.clear()
        logger.info("HeartbeatSimulator RESUMED: engine=%s", self.engine_id)

    def is_paused(self) -> bool:
        return self._paused_event.is_set()

    @property
    def beats_sent(self) -> int:
        return self._beats_sent

    # ------------------------------------------------------------------
    # Beat loop
    # ------------------------------------------------------------------

    def _beat_loop(self) -> None:
        while not self._stop_event.is_set():
            # Check auto-resume
            if (
                self._paused_event.is_set()
                and self._paused_until is not None
                and time.monotonic() >= self._paused_until
            ):
                self.resume()

            if not self._paused_event.is_set():
                self._publish_beat()

            # Sleep in small increments to respond quickly to stop/resume
            for _ in range(int(self.interval_s * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def _publish_beat(self) -> None:
        payload = json.dumps({
            "engine_id":  self.engine_id,
            "status":     "alive",
            "timestamp":  datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "interval_s": self.interval_s,
            "beats_sent": self._beats_sent,
        }).encode("utf-8")

        self._client.publish(self._topic, payload=payload, qos=1)
        self._beats_sent += 1
        logger.debug(
            "Heartbeat published: engine=%s beat=%d topic=%s",
            self.engine_id, self._beats_sent, self._topic,
        )

    # ------------------------------------------------------------------
    # Paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected.set()
            logger.info(
                "HeartbeatSimulator connected to broker (engine=%s)", self.engine_id
            )
        else:
            logger.error(
                "HeartbeatSimulator connection failed rc=%d (engine=%s)",
                rc, self.engine_id,
            )

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()
        if rc != 0:
            logger.warning(
                "HeartbeatSimulator unexpected disconnect rc=%d (engine=%s)",
                rc, self.engine_id,
            )


# ---------------------------------------------------------------------------
# HeartbeatListener — receives beats and calls FallbackRouter.update_heartbeat
# ---------------------------------------------------------------------------

class HeartbeatListener:
    """
    Subscribes to all Model B engine heartbeat topics and calls
    FallbackRouter.update_heartbeat(engine_id) on each received beat.

    Pairs with HeartbeatSimulator to create a real MQTT-backed heartbeat
    loop for integration testing of the fallback system.

    Usage::

        router   = FallbackRouter(heartbeat_timeout_s=30.0)
        router.register_engine("face_engine", cameras=["cam_01"])

        listener = HeartbeatListener(fallback_router=router)
        listener.connect()
        # FallbackRouter now updated automatically from MQTT heartbeats
    """

    def __init__(
        self,
        fallback_router,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id:   str = "",
    ) -> None:
        client_id = client_id or f"hb_listener_{uuid.uuid4().hex[:6]}"
        self._router     = fallback_router
        self._client     = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._connected   = threading.Event()
        self._beats_received: dict = {}

        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

    def connect(self) -> None:
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._connected.wait(timeout=5.0):
            raise TimeoutError("HeartbeatListener could not connect to broker")
        # Subscribe to all engine heartbeats
        self._client.subscribe("sih26187/engine/+/heartbeat", qos=1)
        logger.info("HeartbeatListener connected and subscribed.")

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected.set()

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            engine_id = data.get("engine_id", "unknown")
            self._router.update_heartbeat(engine_id)
            self._beats_received[engine_id] = (
                self._beats_received.get(engine_id, 0) + 1
            )
            logger.debug(
                "HeartbeatListener: engine=%s beat#%d",
                engine_id, self._beats_received[engine_id],
            )
        except Exception as exc:
            logger.error("HeartbeatListener parse error: %s", exc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()

    @property
    def beats_received(self) -> dict:
        return dict(self._beats_received)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Model B heartbeat simulator — Phase Zurich"
    )
    parser.add_argument("--engine",   default="face_engine",  help="Engine ID")
    parser.add_argument("--interval", default=5.0, type=float, help="Heartbeat interval (s)")
    parser.add_argument("--broker",   default="localhost",     help="MQTT broker host")
    parser.add_argument("--port",     default=1883, type=int,  help="MQTT broker port")
    parser.add_argument(
        "--pause-after", default=None, type=float,
        help="Pause after N seconds (to trigger fallback testing)",
    )
    parser.add_argument(
        "--pause-duration", default=40.0, type=float,
        help="How long to pause for (seconds)",
    )
    args = parser.parse_args()

    sim = HeartbeatSimulator(
        engine_id   = args.engine,
        interval_s  = args.interval,
        broker_host = args.broker,
        broker_port = args.port,
    )

    try:
        sim.start()
        logger.info("Heartbeat simulator running. Press Ctrl-C to stop.")

        if args.pause_after:
            logger.info("Will pause after %.1fs for %.1fs", args.pause_after, args.pause_duration)
            time.sleep(args.pause_after)
            sim.pause(duration_s=args.pause_duration)
            time.sleep(args.pause_duration + 5)
            sim.resume()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        sim.stop()
        logger.info("Total beats sent: %d", sim.beats_sent)
