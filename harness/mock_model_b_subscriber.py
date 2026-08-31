"""
Mock Model B Subscriber Harness — Phase Zurich Integration Readiness
SIH26187 | Phase Zurich | Part 1

PURPOSE
  Validates the schema_v1 wire contract BEFORE Model B's code exists.
  If this harness finds a schema violation, it is cheap to fix now.
  If Model B's developer discovers it during joint integration, it is expensive.

WHAT IT DOES
  1. Subscribes to the real MQTT bus (sih26187/camera/+/model_a/event)
     — not a mock bus, the actual bus_client.py wiring.
  2. Re-validates every received event against schema_v1 using a FRESH
     Pydantic parse (not Model A's internal object — from raw JSON bytes).
     This proves the WIRE FORMAT is correct, not just that our code
     agrees with itself.
  3. Routes events exactly as Model B would:
       close_range + human/animal/unknown → face_handler
       close_range + vehicle              → anpr_handler (chokepoint check)
       long_range                         → trajectory_posture_handler
       entity_type == vehicle             → NEVER calls face_handler
       entity_type == unknown             → warning flag + rate tracked
  4. ANPR chokepoint allowlist: rejects/flags ANPR routing for any camera_id
     not in the allowlist (mirrors real constraint: ANPR must not run everywhere).
  5. Outputs a plain-language integration report: total events, schema
     violations (must be 0), routing breakdown, unknown-entity rate,
     chokepoint violations.

ARCHITECTURE
  ModelBRouter     — pure routing/validation logic (no MQTT, fully testable)
  MockModelBSubscriber — wraps ModelBRouter with real MQTT subscription

STANDALONE USAGE
  python harness/mock_model_b_subscriber.py

  Press Ctrl-C to stop and print the integration report.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from pydantic import ValidationError

from model_a.schema_v1 import EntityType, ModelAEvent, ZoneTag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANPR Chokepoint Allowlist
# (mirrors real constraint: ANPR only at checkpoints, not every camera)
# ---------------------------------------------------------------------------

DEFAULT_CHOKEPOINT_ALLOWLIST = {
    "cam_gate_north",
    "cam_gate_south",
    "cam_gate_east",
    "cam_gate_west",
    "cam_chokepoint_01",
    "cam_chokepoint_02",
}


# ---------------------------------------------------------------------------
# Routing outcomes
# ---------------------------------------------------------------------------

class RoutingOutcome(Enum):
    FACE_HANDLER             = auto()   # close_range, non-vehicle
    ANPR_HANDLER             = auto()   # close_range, vehicle, in allowlist
    TRAJECTORY_POSTURE       = auto()   # long_range (any entity)
    ANPR_CHOKEPOINT_VIOLATION = auto()  # close_range, vehicle, NOT in allowlist
    SCHEMA_VIOLATION         = auto()   # Pydantic validation failed
    UNROUTABLE               = auto()   # unexpected zone_tag value


# ---------------------------------------------------------------------------
# Routing record (one per received message)
# ---------------------------------------------------------------------------

@dataclass
class RoutingRecord:
    raw_topic:    str
    camera_id:    str
    event_id:     str
    event_type:   str
    severity:     str
    zone_tag:     str
    entity_type:  str
    outcome:      RoutingOutcome
    schema_error: Optional[str] = None
    warning:      Optional[str] = None


# ---------------------------------------------------------------------------
# Integration report
# ---------------------------------------------------------------------------

@dataclass
class IntegrationReport:
    total_events:        int                     = 0
    schema_violations:   int                     = 0
    routing_breakdown:   Dict[str, int]          = field(default_factory=lambda: defaultdict(int))
    entity_breakdown:    Dict[str, int]          = field(default_factory=lambda: defaultdict(int))
    zone_breakdown:      Dict[str, int]          = field(default_factory=lambda: defaultdict(int))
    unknown_entity_count: int                    = 0
    unknown_entity_rate: float                   = 0.0
    chokepoint_violations: List[dict]            = field(default_factory=list)
    schema_violation_details: List[str]          = field(default_factory=list)
    run_duration_s:      float                   = 0.0

    def to_text(self) -> str:
        """Plain-language integration report for operator display."""
        sep = "=" * 60
        lines = [
            sep,
            "PHASE ZURICH — MODEL B SUBSCRIBER INTEGRATION REPORT",
            sep,
            f"  Total events received       : {self.total_events}",
            f"  Schema violations           : {self.schema_violations}"
              + (" ← MUST BE ZERO" if self.schema_violations > 0 else " ✓"),
            f"  Run duration                : {self.run_duration_s:.1f}s",
            "",
            "ROUTING BREAKDOWN",
        ]
        for outcome, count in sorted(self.routing_breakdown.items()):
            lines.append(f"  {outcome:<35} : {count}")

        lines += [
            "",
            "ENTITY TYPE BREAKDOWN",
        ]
        for etype, count in sorted(self.entity_breakdown.items()):
            lines.append(f"  {etype:<35} : {count}")

        lines += [
            "",
            "ZONE TAG BREAKDOWN",
        ]
        for ztag, count in sorted(self.zone_breakdown.items()):
            lines.append(f"  {ztag:<35} : {count}")

        lines += [
            "",
            f"UNKNOWN ENTITY RATE         : {self.unknown_entity_rate:.1%}"
              + (" ← INVESTIGATE" if self.unknown_entity_rate > 0.05 else " ✓"),
        ]

        if self.chokepoint_violations:
            lines += ["", "ANPR CHOKEPOINT VIOLATIONS (must be 0 in production):"]
            for v in self.chokepoint_violations:
                lines.append(f"  camera={v['camera_id']} event_id={v['event_id']}")

        if self.schema_violation_details:
            lines += ["", "SCHEMA VIOLATION DETAILS:"]
            for d in self.schema_violation_details:
                lines.append(f"  {d}")

        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_events":              self.total_events,
            "schema_violations":         self.schema_violations,
            "routing_breakdown":         dict(self.routing_breakdown),
            "entity_breakdown":          dict(self.entity_breakdown),
            "zone_breakdown":            dict(self.zone_breakdown),
            "unknown_entity_count":      self.unknown_entity_count,
            "unknown_entity_rate":       round(self.unknown_entity_rate, 4),
            "chokepoint_violations":     self.chokepoint_violations,
            "schema_violation_details":  self.schema_violation_details,
            "run_duration_s":            round(self.run_duration_s, 2),
        }


# ---------------------------------------------------------------------------
# ModelBRouter — pure routing/validation logic (no MQTT, unit-testable)
# ---------------------------------------------------------------------------

class ModelBRouter:
    """
    Simulates Model B's event routing logic.

    Testable without a broker: call process_raw(payload_bytes, topic) directly.

    All routing decisions are recorded and accessible via generate_report().
    """

    def __init__(
        self,
        chokepoint_allowlist: Optional[set] = None,
        on_face_route:         Optional[Callable] = None,
        on_trajectory_route:   Optional[Callable] = None,
        on_anpr_route:         Optional[Callable] = None,
    ) -> None:
        self.chokepoint_allowlist = chokepoint_allowlist or DEFAULT_CHOKEPOINT_ALLOWLIST
        self._records: List[RoutingRecord] = []
        self._start_time = time.monotonic()

        # Stub handlers — record calls, perform no actual detection
        self._face_handler       = on_face_route       or self._stub_face_handler
        self._trajectory_handler = on_trajectory_route or self._stub_trajectory_handler
        self._anpr_handler       = on_anpr_route       or self._stub_anpr_handler

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_raw(self, payload_bytes: bytes, topic: str) -> RoutingRecord:
        """
        Process one raw MQTT payload (bytes from wire).

        Steps:
          1. JSON parse
          2. Schema re-validate (FRESH Pydantic — independent of Model A's
             own in-memory state. This is what Model B's SDK call looks like.)
          3. Route based on zone_tag and entity_type
          4. ANPR chokepoint check
          5. Unknown entity_type rate tracking
          6. Record and return routing decision
        """
        # Step 1: JSON parse
        try:
            payload_dict = json.loads(payload_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            rec = RoutingRecord(
                raw_topic   = topic,
                camera_id   = "PARSE_ERROR",
                event_id    = "PARSE_ERROR",
                event_type  = "PARSE_ERROR",
                severity    = "PARSE_ERROR",
                zone_tag    = "PARSE_ERROR",
                entity_type = "PARSE_ERROR",
                outcome     = RoutingOutcome.SCHEMA_VIOLATION,
                schema_error = str(exc),
            )
            self._records.append(rec)
            return rec

        # Step 2: FRESH schema re-validation
        # This is exactly what Model B's SDK call looks like:
        #   event = ModelAEvent(**payload_dict)
        # If this raises, the wire contract is broken.
        try:
            event = ModelAEvent(**payload_dict)
        except (ValidationError, Exception) as exc:
            logger.error(
                "SCHEMA VIOLATION on topic=%s: %s — "
                "Wire contract broken. Fix schema_v1 before Model B integration.",
                topic, exc,
            )
            rec = RoutingRecord(
                raw_topic   = topic,
                camera_id   = payload_dict.get("camera_id", "UNKNOWN"),
                event_id    = payload_dict.get("event_id", "UNKNOWN"),
                event_type  = payload_dict.get("event_type", "UNKNOWN"),
                severity    = payload_dict.get("severity", "UNKNOWN"),
                zone_tag    = payload_dict.get("zone_tag", "UNKNOWN"),
                entity_type = payload_dict.get("entity_type", "UNKNOWN"),
                outcome     = RoutingOutcome.SCHEMA_VIOLATION,
                schema_error = str(exc),
            )
            self._records.append(rec)
            return rec

        # Step 3 & 4: Route
        rec = self._route(event, topic)
        self._records.append(rec)
        return rec

    def process_dict(self, payload_dict: dict, topic: str = "synthetic") -> RoutingRecord:
        """
        Process a payload dict directly (convenience for tests that already
        have a parsed dict — re-serialise to bytes then call process_raw).
        """
        raw = json.dumps(payload_dict).encode("utf-8")
        return self.process_raw(raw, topic)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route(self, event: ModelAEvent, topic: str) -> RoutingRecord:
        """Apply Model B's routing rules to a validated event."""
        warning: Optional[str] = None

        # Unknown entity_type — flag but still route by zone
        if event.entity_type == EntityType.unknown:
            warning = (
                f"entity_type=unknown for event_id={event.event_id} "
                f"camera={event.camera_id}. "
                f"Model B loses optimisation value (cannot select specialised handler). "
                f"Investigate YOLO class mapping."
            )
            logger.warning(warning)

        outcome = self._determine_outcome(event)

        # Call the appropriate stub handler (except for violations/errors)
        if outcome == RoutingOutcome.FACE_HANDLER:
            # CRITICAL RULE: vehicle must NEVER reach face_handler
            assert event.entity_type != EntityType.vehicle, (
                f"BUG: vehicle routed to face_handler! event_id={event.event_id}"
            )
            self._face_handler(event)

        elif outcome == RoutingOutcome.ANPR_HANDLER:
            self._anpr_handler(event)

        elif outcome == RoutingOutcome.TRAJECTORY_POSTURE:
            self._trajectory_handler(event)

        elif outcome == RoutingOutcome.ANPR_CHOKEPOINT_VIOLATION:
            logger.error(
                "ANPR CHOKEPOINT VIOLATION: camera=%s is NOT in allowlist. "
                "Event event_id=%s would have been routed to ANPR but is rejected. "
                "Allowlist: %s",
                event.camera_id, event.event_id, self.chokepoint_allowlist,
            )

        return RoutingRecord(
            raw_topic   = topic,
            camera_id   = event.camera_id,
            event_id    = event.event_id,
            event_type  = event.event_type,
            severity    = event.severity,
            zone_tag    = event.zone_tag,
            entity_type = event.entity_type,
            outcome     = outcome,
            warning     = warning,
        )

    def _determine_outcome(self, event: ModelAEvent) -> RoutingOutcome:
        """Pure routing decision — deterministic, no side effects."""
        zone_tag    = event.zone_tag
        entity_type = event.entity_type

        if zone_tag == ZoneTag.close_range:
            if entity_type == EntityType.vehicle:
                # Vehicle at close range → ANPR (check chokepoint allowlist)
                if event.camera_id in self.chokepoint_allowlist:
                    return RoutingOutcome.ANPR_HANDLER
                else:
                    return RoutingOutcome.ANPR_CHOKEPOINT_VIOLATION
            else:
                # Human / animal / unknown / animal_cart at close range → face handler
                # vehicle specifically excluded (never reaches here)
                return RoutingOutcome.FACE_HANDLER

        elif zone_tag == ZoneTag.long_range:
            return RoutingOutcome.TRAJECTORY_POSTURE

        else:
            return RoutingOutcome.UNROUTABLE

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> IntegrationReport:
        """Generate plain-language integration report from all records."""
        report = IntegrationReport(
            run_duration_s = time.monotonic() - self._start_time,
        )

        for rec in self._records:
            report.total_events += 1
            report.routing_breakdown[rec.outcome.name] = (
                report.routing_breakdown.get(rec.outcome.name, 0) + 1
            )
            report.entity_breakdown[rec.entity_type] = (
                report.entity_breakdown.get(rec.entity_type, 0) + 1
            )
            report.zone_breakdown[rec.zone_tag] = (
                report.zone_breakdown.get(rec.zone_tag, 0) + 1
            )

            if rec.outcome == RoutingOutcome.SCHEMA_VIOLATION:
                report.schema_violations += 1
                report.schema_violation_details.append(
                    f"topic={rec.raw_topic} error={rec.schema_error}"
                )

            if rec.entity_type in ("unknown", EntityType.unknown.value
                                   if hasattr(EntityType.unknown, "value") else "unknown"):
                report.unknown_entity_count += 1

            if rec.outcome == RoutingOutcome.ANPR_CHOKEPOINT_VIOLATION:
                report.chokepoint_violations.append({
                    "camera_id": rec.camera_id,
                    "event_id":  rec.event_id,
                    "topic":     rec.raw_topic,
                })

            if rec.warning:
                logger.warning(rec.warning)

        if report.total_events > 0:
            report.unknown_entity_rate = (
                report.unknown_entity_count / report.total_events
            )

        return report

    def record_count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Reset all records (for reuse across test runs)."""
        self._records.clear()
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Stub handlers — record calls, perform no actual Model B logic
    # ------------------------------------------------------------------

    def _stub_face_handler(self, event: ModelAEvent) -> None:
        logger.debug(
            "[STUB face_handler] event_id=%s camera=%s entity=%s severity=%s",
            event.event_id, event.camera_id, event.entity_type, event.severity,
        )

    def _stub_trajectory_handler(self, event: ModelAEvent) -> None:
        logger.debug(
            "[STUB trajectory_posture_handler] event_id=%s camera=%s zone=%s",
            event.event_id, event.camera_id, event.zone_tag,
        )

    def _stub_anpr_handler(self, event: ModelAEvent) -> None:
        logger.debug(
            "[STUB anpr_handler] event_id=%s camera=%s (chokepoint OK)",
            event.event_id, event.camera_id,
        )


# ---------------------------------------------------------------------------
# MockModelBSubscriber — real MQTT wiring around ModelBRouter
# ---------------------------------------------------------------------------

class MockModelBSubscriber:
    """
    Subscribes to the real MQTT bus and routes all Model A events through
    ModelBRouter for validation, routing, and report generation.

    Uses bus_client.py's subscribe_events() with camera_id="+" (wildcard)
    to receive events from all cameras on a single subscription.

    Usage::

        sub = MockModelBSubscriber(
            chokepoint_allowlist={"cam_gate_01"},
            broker_host="localhost",
        )
        sub.connect()

        # ... events published elsewhere ...

        report = sub.generate_report()
        print(report.to_text())
        sub.disconnect()
    """

    def __init__(
        self,
        chokepoint_allowlist: Optional[set]  = None,
        broker_host:          str            = "localhost",
        broker_port:          int            = 1883,
        client_id:            str            = "",
    ) -> None:
        from model_a.bus_client import BusClient
        client_id = client_id or f"mock_model_b_{uuid.uuid4().hex[:8]}"
        self._bus    = BusClient(
            broker_host=broker_host,
            broker_port=broker_port,
            client_id=client_id,
        )
        self._router = ModelBRouter(chokepoint_allowlist=chokepoint_allowlist)

    def connect(self) -> None:
        """Connect to broker and subscribe to all camera events."""
        self._bus.connect()
        # "+" is MQTT single-level wildcard → receives all camera events
        self._bus.subscribe_events("+", self._on_message)
        logger.info("MockModelBSubscriber connected and subscribed to all cameras.")

    def disconnect(self) -> None:
        self._bus.disconnect()

    def _on_message(self, topic: str, payload_dict: dict) -> None:
        """Called by BusClient for every received message."""
        # Re-serialise dict → bytes to simulate cross-process wire format
        raw_bytes = json.dumps(payload_dict).encode("utf-8")
        rec = self._router.process_raw(raw_bytes, topic)
        if rec.outcome == RoutingOutcome.SCHEMA_VIOLATION:
            logger.error(
                "SCHEMA VIOLATION DETECTED. This must be fixed before Model B integration."
            )

    def generate_report(self) -> IntegrationReport:
        return self._router.generate_report()

    @property
    def router(self) -> ModelBRouter:
        """Direct access to the router for test assertions."""
        return self._router


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
        description="Mock Model B subscriber harness — Phase Zurich"
    )
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port",   default=1883, type=int, help="MQTT broker port")
    parser.add_argument(
        "--chokepoints",
        nargs="*",
        default=list(DEFAULT_CHOKEPOINT_ALLOWLIST),
        help="ANPR chokepoint camera IDs",
    )
    parser.add_argument("--duration", default=30, type=int, help="Run for N seconds")
    args = parser.parse_args()

    sub = MockModelBSubscriber(
        chokepoint_allowlist=set(args.chokepoints),
        broker_host=args.broker,
        broker_port=args.port,
    )

    try:
        sub.connect()
        logger.info("Listening for %ds. Press Ctrl-C to stop early.", args.duration)
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        sub.disconnect()
        report = sub.generate_report()
        print("\n" + report.to_text())
