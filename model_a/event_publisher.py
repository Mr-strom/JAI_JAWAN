"""
EventPublisher — Thread-Safe Bounded-Queue MQTT Publisher
SIH26187 | Model A | Bus I/O Decoupler

Problem:
  If frame processing and MQTT publish happen in the same thread, a slow
  broker (or QoS-2 handshake) will stall the entire detection pipeline.
  This causes frame drops and ruins the <50ms per-frame latency budget.

Solution (from spec smart-coding suggestion):
  Decouple frame processing (producer) from MQTT publish (consumer) using
  a bounded queue. A background drain thread pulls events and publishes them.

Critical rule (Rule #2 / NON-NEGOTIABLE):
  ALL events — including CRITICAL severity — go through the SAME queue.
  There is no separate lane, no priority insertion, no bypass path.
  The queue is ordered FIFO. CRITICAL events are distinguished ONLY by the
  `severity` field when the subscriber reads them.

Why this is safe:
  - QoS 2 (used for confirmed/critical) provides stronger delivery guarantees
    from the broker side once the message is handed off.
  - The queue is bounded (maxsize=200). Under normal load the queue depth
    stays near-zero since the drain thread runs at MQTT link speed.
  - Under pathological overload, low-severity events are dropped first
    via timeout — but CRITICAL events are still enqueued with a longer
    timeout (10x) to give them priority within the same queue without
    creating a bypass channel.

Metrics tracked:
  - published_count     : total events successfully handed to paho
  - dropped_count       : events discarded due to full queue (by severity)
  - publish_latency_ms  : rolling p50/p95/p99 of queue-to-publish time
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from model_a.schema_v1 import ModelAEvent, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_QUEUE_MAXSIZE         = 200     # bounded — prevents OOM under broker disconnect
_ENQUEUE_TIMEOUT_LOW   = 0.01    # 10ms — info/warning/provisional enqueue timeout
_ENQUEUE_TIMEOUT_HIGH  = 0.10    # 100ms — confirmed/critical enqueue timeout
_DRAIN_THREAD_SLEEP    = 0.001   # 1ms poll interval when queue is empty

# Severities that get the longer enqueue timeout (gives them better queue access)
_HIGH_PRIORITY_SEVERITIES = {Severity.confirmed, Severity.critical}


# ---------------------------------------------------------------------------
# Publish record (for latency tracking)
# ---------------------------------------------------------------------------

@dataclass
class _PublishRecord:
    event_id:    str
    severity:    str
    enqueued_at: float   # monotonic time
    published_at: Optional[float] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.published_at is None:
            return None
        return (self.published_at - self.enqueued_at) * 1000.0


# ---------------------------------------------------------------------------
# EventPublisher
# ---------------------------------------------------------------------------

class EventPublisher:
    """
    Decouples event production (frame pipeline) from MQTT I/O.

    The pipeline thread calls enqueue(); a background drain thread
    picks events off the queue and calls bus_client.publish_event().

    Same queue for all severities — Rule #2 compliance.

    Usage::

        publisher = EventPublisher(bus_client=bus)
        publisher.start()

        # From pipeline thread:
        publisher.enqueue(event)

        # On shutdown:
        publisher.stop()
        stats = publisher.stats()
    """

    def __init__(
        self,
        bus_client,              # BusClient instance (or any with publish_event())
        queue_maxsize: int = _QUEUE_MAXSIZE,
        on_publish: Optional[Callable[[ModelAEvent, float], None]] = None,
    ) -> None:
        """
        Args:
            bus_client:   BusClient or a mock/stub implementing publish_event().
            queue_maxsize: Bounded queue size. Prevents OOM under broker disconnect.
            on_publish:   Optional callback(event, latency_ms) called after each
                          successful publish. Used for testing and monitoring.
        """
        self._bus            = bus_client
        self._on_publish     = on_publish
        self._queue: queue.Queue[_PublishRecord] = queue.Queue(maxsize=queue_maxsize)

        # Counters
        self._published_count: int = 0
        self._dropped_count: Dict[str, int] = defaultdict(int)
        self._latencies_ms: deque[float] = deque(maxlen=1000)  # rolling window

        # Threading
        self._drain_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._lock         = threading.Lock()

        # Store events temporarily for latency tracking
        self._pending: Dict[str, _PublishRecord] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background drain thread."""
        self._stop_event.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name="EventPublisher-drain",
            daemon=True,
        )
        self._drain_thread.start()
        logger.info("EventPublisher drain thread started.")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal drain thread to stop and wait for it.
        Drains remaining queue items before stopping.
        """
        self._stop_event.set()
        if self._drain_thread:
            self._drain_thread.join(timeout=timeout)
        logger.info(
            "EventPublisher stopped. published=%d dropped=%s",
            self._published_count,
            dict(self._dropped_count),
        )

    # ------------------------------------------------------------------
    # Producer API — called from frame pipeline threads
    # ------------------------------------------------------------------

    def enqueue(self, event: ModelAEvent) -> bool:
        """
        Enqueue an event for publishing.

        Returns True if enqueued, False if dropped (queue full after timeout).
        NEVER blocks indefinitely — the pipeline must not stall.

        Rule #2 compliance: All severities go into the same queue.
        High-severity events get a 10x longer enqueue timeout to reduce
        their drop probability under backpressure.
        """
        severity = Severity(event.severity)
        timeout  = (
            _ENQUEUE_TIMEOUT_HIGH
            if severity in _HIGH_PRIORITY_SEVERITIES
            else _ENQUEUE_TIMEOUT_LOW
        )

        record = _PublishRecord(
            event_id    = event.event_id,
            severity    = event.severity,
            enqueued_at = time.monotonic(),
        )

        try:
            self._queue.put((record, event), timeout=timeout)
            logger.debug(
                "Enqueued event_id=%s severity=%s queue_depth=%d",
                event.event_id, event.severity, self._queue.qsize(),
            )
            return True

        except queue.Full:
            with self._lock:
                self._dropped_count[event.severity] += 1
            logger.error(
                "QUEUE FULL — event DROPPED: event_id=%s severity=%s. "
                "queue_depth=%d. Is the broker reachable?",
                event.event_id, event.severity, self._queue.qsize(),
            )
            return False

    # ------------------------------------------------------------------
    # Drain loop — runs in background thread
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        """Continuously drain the queue and publish events."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                record, event = self._queue.get(timeout=_DRAIN_THREAD_SLEEP)
            except queue.Empty:
                continue

            try:
                self._bus.publish_event(event)
                record.published_at = time.monotonic()
                latency_ms = record.latency_ms or 0.0

                with self._lock:
                    self._published_count += 1
                    self._latencies_ms.append(latency_ms)

                logger.debug(
                    "Published event_id=%s severity=%s latency=%.1fms",
                    event.event_id, event.severity, latency_ms,
                )

                if self._on_publish:
                    self._on_publish(event, latency_ms)

            except Exception as exc:
                logger.error(
                    "Publish error for event_id=%s: %s", record.event_id, exc
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Metrics API
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return publish statistics snapshot."""
        with self._lock:
            lats = list(self._latencies_ms)
            lats_sorted = sorted(lats)
            n = len(lats_sorted)

            def percentile(p: float) -> Optional[float]:
                if not lats_sorted:
                    return None
                idx = max(0, int(n * p / 100) - 1)
                return round(lats_sorted[idx], 2)

            return {
                "published_count": self._published_count,
                "dropped_count":   dict(self._dropped_count),
                "queue_depth":     self._queue.qsize(),
                "latency_p50_ms":  percentile(50),
                "latency_p95_ms":  percentile(95),
                "latency_p99_ms":  percentile(99),
                "latency_max_ms":  round(max(lats_sorted), 2) if lats_sorted else None,
            }

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def dropped_count_total(self) -> int:
        return sum(self._dropped_count.values())

    def queue_depth(self) -> int:
        return self._queue.qsize()
