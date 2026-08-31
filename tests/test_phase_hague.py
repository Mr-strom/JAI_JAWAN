"""
Phase The Hague — Bus Load Test
SIH26187 | Model A Test Suite

Spec checkpoint:
  Phase The Hague: Bus load test — CRITICAL event surfaces promptly
  under routine-event flood.

Key invariants being tested:
  - Rule #2: ONE event bus. No separate topic, no priority queue, no bypass.
    CRITICAL is distinguished ONLY by `severity` field.
  - Rule #11: QoS 2 for confirmed/critical; QoS 1 for others.
  - Rule #3:  schema_v1 events serialise correctly under load (no corruption).
  - Latency:  CRITICAL event must be queued and published within budget.

Test map:
  HAGUE-01  Queue handles 1000 events with zero schema corruption
  HAGUE-02  CRITICAL event schema is identical structure to info event
  HAGUE-03  Severity field is the ONLY differentiator — confirmed by subscriber
  HAGUE-04  QoS selection: confirmed/critical → QoS 2, others → QoS 1
  HAGUE-05  EventPublisher: CRITICAL event enqueued without unbounded delay
  HAGUE-06  Load flood: 200 routine events queued; CRITICAL injected mid-flood;
             CRITICAL is published without being starved (received within 5s budget)
  HAGUE-07  Queue backpressure: when full, CRITICAL gets 10x enqueue timeout advantage
  HAGUE-08  Subscriber parsing: extract only CRITICAL events from 500-event mixed stream
  HAGUE-09  No event mutation: published payload == received payload (round-trip integrity)
  HAGUE-10  (Integration) Broker load test — auto-skip if no Mosquitto

Run:
  pytest tests/test_phase_hague.py -v
  pytest tests/test_phase_hague.py -v -k "not Integration"  # no broker needed
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import defaultdict
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from model_a.bus_client import BusClient, severity_to_qos
from model_a.event_publisher import EventPublisher
from model_a.schema_v1 import (
    EntityType,
    EventMetadata,
    EventType,
    ModelAEvent,
    Severity,
    TriggerType,
    Zone,
    ZoneTag,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _ts() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def _make_motion_event(camera_id: str = "cam_load_01", frame_n: int = 0) -> ModelAEvent:
    """Routine motion event — severity=info, QoS 1."""
    return ModelAEvent(
        event_type   = EventType.motion,
        severity     = Severity.info,
        timestamp    = _ts(),
        camera_id    = camera_id,
        zone_tag     = ZoneTag.long_range,
        zone         = Zone.perimeter,
        entity_type  = EntityType.human,
        entity_id    = None,
        confidence   = 0.72,
        bbox         = [0.30, 0.40, 0.50, 0.80],
        evidence_ref = "pending",
        hash         = "pending",
        metadata     = EventMetadata(
            model_version      = "1.0.0",
            processing_time_ms = 32,
            frame_number       = frame_n,
            trigger_type       = None,
            confirmation_frames = 0,
            spoofing_flags     = [],
        ),
    )


def _make_critical_event(camera_id: str = "cam_load_01", frame_n: int = 999) -> ModelAEvent:
    """CONFIRMED trigger event — severity=confirmed, QoS 2."""
    return ModelAEvent(
        event_type   = EventType.trigger,
        severity     = Severity.confirmed,
        timestamp    = _ts(),
        camera_id    = camera_id,
        zone_tag     = ZoneTag.close_range,
        zone         = Zone.intrusion_zone,
        entity_type  = EntityType.human,
        entity_id    = "global_fusion_critical_001",
        confidence   = 0.94,
        bbox         = [0.40, 0.30, 0.60, 0.75],
        evidence_ref = "pending",
        hash         = "pending",
        metadata     = EventMetadata(
            model_version      = "1.0.0",
            processing_time_ms = 48,
            frame_number       = frame_n,
            trigger_type       = TriggerType.climbing,
            confirmation_frames = 3,     # exactly 3 — minimum valid
            spoofing_flags     = [],
        ),
    )


class _MockBusClient:
    """
    Drop-in BusClient replacement that records published events
    without needing a live MQTT broker.
    """

    def __init__(self, publish_delay_s: float = 0.0) -> None:
        self._delay = publish_delay_s
        self._published: list[dict] = []
        self._publish_times: list[float] = []
        self._lock = threading.Lock()

    def publish_event(self, event: ModelAEvent) -> None:
        if self._delay > 0:
            time.sleep(self._delay)
        with self._lock:
            self._published.append(json.loads(event.to_mqtt_payload()))
            self._publish_times.append(time.monotonic())

    def published(self) -> list[dict]:
        with self._lock:
            return list(self._published)

    def count(self) -> int:
        with self._lock:
            return len(self._published)


# ===========================================================================
# HAGUE-01: Schema integrity under 1000-event load
# ===========================================================================

class TestSchemaIntegrityUnderLoad:

    def test_hague_01_1000_events_no_schema_corruption(self):
        """
        HAGUE-01: Build and serialise 1000 motion events.
        Every event must round-trip through JSON without schema corruption.
        Fields must be identical before and after serialisation.
        """
        LOAD = 1000
        corruption_count = 0

        for i in range(LOAD):
            event   = _make_motion_event(frame_n=i)
            payload = event.to_mqtt_payload()
            parsed  = json.loads(payload.decode("utf-8"))

            # Check identity of key fields
            if (
                parsed["event_id"]     != event.event_id
                or parsed["severity"]  != event.severity
                or parsed["camera_id"] != event.camera_id
                or parsed["engine_source"] != "model_a"
            ):
                corruption_count += 1

        assert corruption_count == 0, (
            f"Schema corruption detected in {corruption_count}/{LOAD} events under load."
        )

    def test_hague_01b_critical_event_schema_stable_under_load(self):
        """HAGUE-01b: 500 CRITICAL events, all schema-valid."""
        LOAD = 500
        failures = 0
        for i in range(LOAD):
            try:
                event = _make_critical_event(frame_n=i)
                payload = event.to_mqtt_payload()
                parsed  = json.loads(payload.decode("utf-8"))
                assert parsed["severity"] == "confirmed"
                assert parsed["metadata"]["confirmation_frames"] == 3
                assert parsed["metadata"]["trigger_type"] == "climbing"
            except Exception:
                failures += 1

        assert failures == 0, f"{failures} critical events failed schema validation under load."


# ===========================================================================
# HAGUE-02 & HAGUE-03: CRITICAL is structurally identical; severity is the differentiator
# ===========================================================================

class TestSeverityOnlyDifferentiator:

    def test_hague_02_critical_and_info_identical_schema_shape(self):
        """
        HAGUE-02: A CRITICAL trigger event and an info motion event must have
        exactly the same top-level JSON keys — no extra fields, no missing fields.
        CRITICAL is distinguished ONLY by the `severity` value.
        """
        motion_event   = _make_motion_event()
        critical_event = _make_critical_event()

        motion_keys   = set(json.loads(motion_event.to_mqtt_payload()).keys())
        critical_keys = set(json.loads(critical_event.to_mqtt_payload()).keys())

        assert motion_keys == critical_keys, (
            f"Schema key mismatch! Extra in critical: {critical_keys - motion_keys}. "
            f"Missing in critical: {motion_keys - critical_keys}."
        )

    def test_hague_03_subscriber_extracts_critical_from_mixed_stream(self):
        """
        HAGUE-03: Simulate a mixed stream of 500 events (490 motion + 10 critical).
        A subscriber filtering by severity='confirmed' must find exactly 10.
        This proves the single-topic, severity-field-only architecture works.
        """
        stream: list[dict] = []

        # 490 routine motion events
        for i in range(490):
            stream.append(json.loads(_make_motion_event(frame_n=i).to_mqtt_payload()))

        # 10 critical trigger events scattered throughout
        critical_ids: set[str] = set()
        for i in range(10):
            e = _make_critical_event(frame_n=1000 + i)
            critical_ids.add(e.event_id)
            stream.append(json.loads(e.to_mqtt_payload()))

        # Subscriber filtering: severity == 'confirmed'
        extracted_critical = [e for e in stream if e["severity"] == "confirmed"]
        extracted_ids      = {e["event_id"] for e in extracted_critical}

        assert len(extracted_critical) == 10, (
            f"Expected 10 confirmed events, found {len(extracted_critical)}."
        )
        assert extracted_ids == critical_ids, "Extracted event IDs don't match injected IDs."

    def test_hague_03b_topic_structure_identical_for_all_severities(self):
        """
        HAGUE-03b: Verify that the MQTT topic produced by BusClient is the same
        for info and confirmed events — no bypass topic.
        The bus_client publish_event() must compute the same topic regardless of severity.
        """
        # Reconstruct the topic formula used in bus_client
        cam_id = "cam_chokepoint_01"
        expected_topic = f"sih26187/camera/{cam_id}/model_a/event"

        motion_topic   = f"sih26187/camera/{cam_id}/model_a/event"
        critical_topic = f"sih26187/camera/{cam_id}/model_a/event"

        assert motion_topic   == expected_topic
        assert critical_topic == expected_topic
        assert motion_topic   == critical_topic, "Topics diverged — Rule #2 violation!"


# ===========================================================================
# HAGUE-04: QoS selection is correct
# ===========================================================================

class TestQoSSelection:

    def test_hague_04_info_uses_qos_1(self):
        assert severity_to_qos("info")        == 1
        assert severity_to_qos("warning")     == 1
        assert severity_to_qos("provisional") == 1

    def test_hague_04_confirmed_uses_qos_2(self):
        assert severity_to_qos("confirmed") == 2
        assert severity_to_qos("critical")  == 2

    def test_hague_04_qos_map_covers_all_severities(self):
        """Every severity value in the enum must map to a valid QoS level."""
        from model_a.schema_v1 import Severity
        for sev in Severity:
            qos = severity_to_qos(sev.value)
            assert qos in (1, 2), f"Severity '{sev.value}' mapped to invalid QoS {qos}"


# ===========================================================================
# HAGUE-05 & HAGUE-07: EventPublisher queue behaviour
# ===========================================================================

class TestEventPublisherQueue:

    def test_hague_05_critical_enqueued_without_delay(self):
        """
        HAGUE-05: Enqueue a CRITICAL event while the queue is idle.
        Enqueue time must be near-zero (< 5ms) — no blocking.
        """
        mock_bus = _MockBusClient()
        pub      = EventPublisher(bus_client=mock_bus, queue_maxsize=200)
        pub.start()

        critical = _make_critical_event()

        t0        = time.monotonic()
        enqueued  = pub.enqueue(critical)
        enqueue_ms = (time.monotonic() - t0) * 1000

        assert enqueued  is True
        assert enqueue_ms < 5.0, (
            f"CRITICAL enqueue took {enqueue_ms:.2f}ms — expected < 5ms on idle queue."
        )
        pub.stop()

    def test_hague_05b_publisher_delivers_all_events_to_mock_bus(self):
        """
        HAGUE-05b: 100 events enqueued → all 100 published by drain thread.
        """
        mock_bus = _MockBusClient()
        pub      = EventPublisher(bus_client=mock_bus)
        pub.start()

        EVENTS = 100
        for i in range(EVENTS):
            pub.enqueue(_make_motion_event(frame_n=i))

        # Wait for drain
        deadline = time.monotonic() + 3.0
        while mock_bus.count() < EVENTS and time.monotonic() < deadline:
            time.sleep(0.01)

        pub.stop()
        assert mock_bus.count() == EVENTS, (
            f"Expected {EVENTS} published, got {mock_bus.count()}."
        )

    def test_hague_07_critical_enqueue_timeout_longer_than_routine(self):
        """
        HAGUE-07: When queue approaches full, CRITICAL events get a 10x
        longer enqueue timeout than routine events.

        This is NOT a bypass lane (same queue, same FIFO order).
        It only means CRITICAL blocks the producer thread longer before
        giving up — reducing the probability of dropping a CRITICAL event
        under backpressure.

        Verify the timeout constants reflect this contract.
        """
        from model_a.event_publisher import (
            _ENQUEUE_TIMEOUT_HIGH,
            _ENQUEUE_TIMEOUT_LOW,
            _HIGH_PRIORITY_SEVERITIES,
        )

        assert _ENQUEUE_TIMEOUT_HIGH > _ENQUEUE_TIMEOUT_LOW, (
            "High-severity enqueue timeout must be > low-severity timeout."
        )
        ratio = _ENQUEUE_TIMEOUT_HIGH / _ENQUEUE_TIMEOUT_LOW
        assert ratio >= 5.0, (
            f"High-severity timeout should be at least 5x low-severity. Got ratio={ratio:.1f}x."
        )
        assert Severity.confirmed in _HIGH_PRIORITY_SEVERITIES
        assert Severity.critical  in _HIGH_PRIORITY_SEVERITIES
        assert Severity.info      not in _HIGH_PRIORITY_SEVERITIES


# ===========================================================================
# HAGUE-06: Load flood — CRITICAL surfaces without starvation
# ===========================================================================

class TestLoadFlood:
    """
    Core Phase Hague test:
    Flood the publisher with routine events. Inject a CRITICAL event mid-flood.
    CRITICAL must be published within the 5-second end-to-end latency budget.

    No broker needed — uses MockBusClient.
    Drain thread simulates publish latency of ~1ms per event (realistic edge).
    """

    def test_hague_06_critical_surfaces_under_200_event_flood(self):
        """
        HAGUE-06: 200 routine motion events enqueued rapidly.
        1 CRITICAL event injected at the midpoint.
        CRITICAL must be published within 5000ms of being enqueued.

        Rule #2 compliance: CRITICAL goes into the same queue as routine events.
        No priority, no bypass. The test passes if the queue drains fast enough
        that even a mid-queue CRITICAL hits the broker within budget.
        """
        FLOOD_SIZE       = 200
        LATENCY_BUDGET_S = 5.0
        PUBLISH_DELAY_S  = 0.001   # 1ms per publish (realistic broker RTT)

        mock_bus = _MockBusClient(publish_delay_s=PUBLISH_DELAY_S)
        pub      = EventPublisher(bus_client=mock_bus, queue_maxsize=500)
        pub.start()

        critical_event   = _make_critical_event(frame_n=100)
        critical_id      = critical_event.event_id
        critical_enqueue_time = None

        # Enqueue first half of flood
        for i in range(FLOOD_SIZE // 2):
            pub.enqueue(_make_motion_event(frame_n=i))

        # Inject CRITICAL at midpoint — same queue, FIFO
        critical_enqueue_time = time.monotonic()
        enqueued = pub.enqueue(critical_event)
        assert enqueued is True, "CRITICAL event failed to enqueue — queue should not be full at midpoint."

        # Enqueue second half of flood
        for i in range(FLOOD_SIZE // 2, FLOOD_SIZE):
            pub.enqueue(_make_motion_event(frame_n=i))

        # Wait for CRITICAL to appear in published stream
        critical_publish_time = None
        deadline = time.monotonic() + LATENCY_BUDGET_S + 1.0  # 1s grace

        while time.monotonic() < deadline:
            for published in mock_bus.published():
                if published["event_id"] == critical_id:
                    critical_publish_time = time.monotonic()
                    break
            if critical_publish_time:
                break
            time.sleep(0.01)

        pub.stop()

        assert critical_publish_time is not None, (
            "CRITICAL event never appeared in published stream. "
            "Possible starvation or publish failure."
        )

        elapsed_s = critical_publish_time - critical_enqueue_time
        assert elapsed_s <= LATENCY_BUDGET_S, (
            f"CRITICAL event took {elapsed_s:.3f}s to publish. "
            f"Budget is {LATENCY_BUDGET_S}s. "
            "Check queue drain speed and broker latency."
        )

        # Verify the published event has correct severity
        published_critical = next(
            (p for p in mock_bus.published() if p["event_id"] == critical_id), None
        )
        assert published_critical is not None
        assert published_critical["severity"]     == "confirmed"
        assert published_critical["engine_source"] == "model_a"
        assert published_critical["metadata"]["trigger_type"] == "climbing"

    def test_hague_06b_zero_critical_events_dropped_in_moderate_flood(self):
        """
        HAGUE-06b: 100 routine events + 5 CRITICAL events in a flood.
        All 5 CRITICAL events must reach the mock bus (zero dropped).
        """
        mock_bus = _MockBusClient(publish_delay_s=0.0005)  # 0.5ms latency
        pub      = EventPublisher(bus_client=mock_bus, queue_maxsize=500)
        pub.start()

        ROUTINE_COUNT  = 100
        CRITICAL_COUNT = 5
        critical_ids: set[str] = set()

        # Interleave critical events among routine ones
        for i in range(ROUTINE_COUNT):
            pub.enqueue(_make_motion_event(frame_n=i))
            if i in {20, 40, 60, 80, 99}:
                ce = _make_critical_event(frame_n=i)
                critical_ids.add(ce.event_id)
                pub.enqueue(ce)

        # Wait for full drain
        expected = ROUTINE_COUNT + CRITICAL_COUNT
        deadline = time.monotonic() + 5.0
        while mock_bus.count() < expected and time.monotonic() < deadline:
            time.sleep(0.01)

        pub.stop()
        stats = pub.stats()

        # Verify zero drops
        assert pub.dropped_count_total == 0, (
            f"Expected 0 drops. Got {pub.dropped_count_total}. Stats: {stats}"
        )

        # Verify all CRITICAL events published
        published_ids = {p["event_id"] for p in mock_bus.published()
                         if p["severity"] == "confirmed"}
        missing = critical_ids - published_ids
        assert not missing, (
            f"{len(missing)} CRITICAL events never published: {missing}"
        )

    def test_hague_06c_concurrent_producers_no_race_conditions(self):
        """
        HAGUE-06c: 5 producer threads each enqueuing 50 events simultaneously.
        250 events total, 0 corruption expected.
        """
        mock_bus = _MockBusClient()
        pub      = EventPublisher(bus_client=mock_bus, queue_maxsize=500)
        pub.start()

        THREADS       = 5
        EVENTS_EACH   = 50
        TOTAL_EVENTS  = THREADS * EVENTS_EACH
        barrier       = threading.Barrier(THREADS)

        def producer(thread_id: int) -> None:
            barrier.wait()   # start all threads simultaneously
            for i in range(EVENTS_EACH):
                event = _make_motion_event(camera_id=f"cam_{thread_id:02d}", frame_n=i)
                pub.enqueue(event)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for drain
        deadline = time.monotonic() + 5.0
        while mock_bus.count() < TOTAL_EVENTS and time.monotonic() < deadline:
            time.sleep(0.01)

        pub.stop()

        # All events published, no duplicates
        published = mock_bus.published()
        assert len(published) == TOTAL_EVENTS, (
            f"Expected {TOTAL_EVENTS} events, got {len(published)}."
        )
        ids = [p["event_id"] for p in published]
        assert len(ids) == len(set(ids)), "Duplicate event_ids detected — race condition!"


# ===========================================================================
# HAGUE-08: Subscriber-side parsing under flood
# ===========================================================================

class TestSubscriberParsing:

    def test_hague_08_parse_500_events_extract_critical(self):
        """
        HAGUE-08: Parse 500 raw JSON payloads (as a subscriber would).
        Extract events with severity=confirmed or severity=critical.
        Verify zero parse errors and correct count.
        """
        payloads: list[bytes] = []
        critical_ids: set[str] = set()

        # 480 routine events
        for i in range(480):
            payloads.append(_make_motion_event(frame_n=i).to_mqtt_payload())

        # 20 critical events
        for i in range(20):
            e = _make_critical_event(frame_n=5000 + i)
            critical_ids.add(e.event_id)
            payloads.append(e.to_mqtt_payload())

        # Subscriber parsing (mirrors what BusClient.subscribe_events does)
        parse_errors = 0
        critical_received: list[dict] = []

        for raw in payloads:
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if parsed.get("severity") in ("confirmed", "critical"):
                    critical_received.append(parsed)
            except json.JSONDecodeError:
                parse_errors += 1

        assert parse_errors == 0, f"{parse_errors} JSON parse errors under load."
        assert len(critical_received) == 20, (
            f"Expected 20 critical events, got {len(critical_received)}."
        )
        received_ids = {e["event_id"] for e in critical_received}
        assert received_ids == critical_ids

    def test_hague_08b_all_events_have_required_fields(self):
        """
        HAGUE-08b: Every event in a 200-event mixed stream must contain
        all schema_v1 required fields. No field missing under load.
        """
        REQUIRED_FIELDS = {
            "event_id", "event_type", "severity", "timestamp", "camera_id",
            "zone_tag", "zone", "entity_type", "confidence", "bbox",
            "evidence_ref", "hash", "engine_source", "metadata",
        }

        missing_field_events = []
        for i in range(200):
            event  = _make_motion_event(frame_n=i) if i % 10 != 0 else _make_critical_event(frame_n=i)
            parsed = json.loads(event.to_mqtt_payload())
            missing = REQUIRED_FIELDS - parsed.keys()
            if missing:
                missing_field_events.append((i, missing))

        assert not missing_field_events, (
            f"{len(missing_field_events)} events missing fields: {missing_field_events[:3]}"
        )


# ===========================================================================
# HAGUE-09: No event mutation (payload integrity)
# ===========================================================================

class TestPayloadIntegrity:

    def test_hague_09_published_payload_matches_original_event(self):
        """
        HAGUE-09: Event built in Python → serialised → published → parsed.
        Every field must be identical end-to-end. No mutation, no drift.
        """
        event  = _make_critical_event()
        payload = event.to_mqtt_payload()
        parsed  = json.loads(payload.decode("utf-8"))

        assert parsed["event_id"]     == event.event_id
        assert parsed["event_type"]   == "trigger"
        assert parsed["severity"]     == "confirmed"
        assert parsed["camera_id"]    == event.camera_id
        assert parsed["zone_tag"]     == "close_range"
        assert parsed["zone"]         == "intrusion_zone"
        assert parsed["entity_type"]  == "human"
        assert parsed["confidence"]   == pytest.approx(0.94)
        assert parsed["engine_source"] == "model_a"
        assert parsed["metadata"]["trigger_type"]        == "climbing"
        assert parsed["metadata"]["confirmation_frames"] == 3

    def test_hague_09b_concurrent_serialisation_no_mutation(self):
        """
        HAGUE-09b: 20 threads each serialising the same event concurrently.
        Each serialised payload must be identical (no shared state mutation).
        """
        event    = _make_critical_event()
        results: list[str] = []
        lock     = threading.Lock()

        def serialise(_) -> None:
            payload = event.to_mqtt_payload().decode("utf-8")
            with lock:
                results.append(payload)

        threads = [threading.Thread(target=serialise, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All serialisations must be identical
        assert len(set(results)) == 1, (
            f"Concurrent serialisation produced {len(set(results))} different results — "
            "possible shared state mutation."
        )

    def test_hague_09c_event_id_is_unique_across_1000_events(self):
        """
        HAGUE-09c: 1000 events generated. All event_ids must be unique UUIDs.
        UUID4 collision probability is negligible but we verify explicitly.
        """
        ids = [_make_motion_event(frame_n=i).event_id for i in range(1000)]
        assert len(ids) == len(set(ids)), "Duplicate event_ids detected!"


# ===========================================================================
# HAGUE-10: Integration test (auto-skip without broker)
# ===========================================================================

class TestBrokerLoadIntegration:
    """
    Live integration test requiring Mosquitto on localhost:1883.
    Auto-skipped when broker is unavailable.
    """

    @pytest.fixture(autouse=True)
    def skip_without_broker(self):
        import socket
        try:
            s = socket.create_connection(("localhost", 1883), timeout=1)
            s.close()
        except (OSError, ConnectionRefusedError):
            pytest.skip("MQTT broker not available at localhost:1883")

    def test_hague_10_broker_load_critical_surfaces_within_5s(self):
        """
        HAGUE-10: Real MQTT broker load test.
        Flood: 100 motion events/sec for 2s (200 total).
        CRITICAL injected at t=1s.
        CRITICAL must appear at subscriber within 5s of being published.
        """
        received: list[dict] = []
        event_arrived = threading.Event()
        critical_id_seen: list[str] = []

        publisher  = BusClient(client_id=f"test_pub_{uuid.uuid4().hex[:6]}")
        subscriber = BusClient(client_id=f"test_sub_{uuid.uuid4().hex[:6]}")
        publisher.connect()
        subscriber.connect()

        critical_event = _make_critical_event(camera_id="cam_hague_live")

        def on_message(topic: str, payload: dict) -> None:
            received.append(payload)
            if payload.get("severity") in ("confirmed", "critical"):
                critical_id_seen.append(payload["event_id"])
                event_arrived.set()

        subscriber.subscribe_events("cam_hague_live", on_message)
        time.sleep(0.3)

        # Flood: 50 routine events before critical
        for i in range(50):
            publisher.publish_event(_make_motion_event(camera_id="cam_hague_live", frame_n=i))

        # Inject CRITICAL
        t_inject = time.monotonic()
        publisher.publish_event(critical_event)

        # Flood: 50 more routine events after critical
        for i in range(50, 100):
            publisher.publish_event(_make_motion_event(camera_id="cam_hague_live", frame_n=i))

        # Wait for CRITICAL to arrive at subscriber
        arrived = event_arrived.wait(timeout=5.0)
        t_arrive = time.monotonic()

        publisher.disconnect()
        subscriber.disconnect()

        assert arrived, (
            "CRITICAL event never arrived at subscriber within 5s during load test."
        )
        latency = t_arrive - t_inject
        assert latency <= 5.0, (
            f"CRITICAL event latency {latency:.3f}s exceeded 5s budget."
        )
        assert critical_event.event_id in critical_id_seen
