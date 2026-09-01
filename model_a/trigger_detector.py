"""
Trigger Detector — Multi-Frame Confirmation Engine
SIH26187 | Model A | Step 6 of pipeline

NON-NEGOTIABLE RULE #1:
  No trigger fires on a single frame.
  Severity CONFIRMED or CRITICAL requires exactly 3 consecutive
  confirming frames for the same track_id.
  If the track_id disappears before the 3rd confirmation, the
  state machine resets to IDLE — no event is published.

State Machine per track_id:
  IDLE → PROVISIONAL_1 → PROVISIONAL_2 → CONFIRMED_TRIGGER → COOLDOWN

  IDLE            : No trigger signal seen.
  PROVISIONAL_1   : 1 frame with trigger signal. Severity = provisional.
  PROVISIONAL_2   : 2 consecutive frames. Severity = provisional.
  CONFIRMED_TRIGGER: 3+ consecutive frames. Severity = confirmed/critical.
  COOLDOWN        : Post-trigger suppression window (default 10s).

Key edge cases handled:
  - 2-frame boundary (test case in Phase Berlin): track seen for 2 frames,
    gone on 3rd → state resets to IDLE, NO confirmed/critical published.
  - Track re-appears after cooldown → fresh state machine cycle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

from model_a.schema_v1 import Severity, TriggerType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine States
# ---------------------------------------------------------------------------

class TriggerState(Enum):
    IDLE              = auto()
    PROVISIONAL_1     = auto()
    PROVISIONAL_2     = auto()
    CONFIRMED_TRIGGER = auto()
    COOLDOWN          = auto()


# ---------------------------------------------------------------------------
# Per-track state record
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    state:             TriggerState = TriggerState.IDLE
    trigger_type:      Optional[TriggerType] = None
    confirmation_frames: int = 0
    last_seen_time:    float = field(default_factory=time.monotonic)
    confirmed_at:      Optional[float] = None   # monotonic time of confirmation
    cooldown_until:    float = 0.0              # monotonic time
    last_published_severity: Optional[Severity] = None  # dedup guard: last severity that was published


# ---------------------------------------------------------------------------
# TriggerDetector
# ---------------------------------------------------------------------------

class TriggerDetector:
    """
    Manages per-track trigger state machines to enforce the 3-frame
    confirmation rule.

    Usage::

        detector = TriggerDetector(confirmation_frames=3, cooldown_seconds=10)

        # On each processed frame, call update() for every track:
        result = detector.update(
            track_id="trk_042",
            trigger_type=TriggerType.climbing,
            frame_number=1234,
        )
        if result.severity in (Severity.confirmed, Severity.critical):
            bus_client.publish_event(build_event(result))

        # If a track disappears from the detection, call miss():
        detector.miss(track_id="trk_042")
    """

    def __init__(
        self,
        confirmation_frames: int = 3,    # Minimum per Rule #1
        cooldown_seconds: float = 10.0,
        stale_timeout_seconds: float = 5.0,
    ) -> None:
        if confirmation_frames < 3:
            raise ValueError(
                "confirmation_frames must be >= 3 per Rule #1. "
                f"Received {confirmation_frames}. Do NOT lower this."
            )
        self.confirmation_frames  = confirmation_frames
        self.cooldown_seconds     = cooldown_seconds
        self.stale_timeout        = stale_timeout_seconds
        self._tracks: Dict[str, TrackRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: str,
        trigger_type: TriggerType,
        frame_number: int,
    ) -> "TriggerResult":
        """
        Register a trigger signal for a track on this frame.

        Returns a TriggerResult describing the current state and
        what severity (if any) to publish.
        """
        now = time.monotonic()
        rec = self._get_or_create(track_id)

        # --- Cooldown guard ---
        if rec.state == TriggerState.COOLDOWN:
            if now < rec.cooldown_until:
                logger.debug("track=%s in COOLDOWN. Skipping.", track_id)
                return TriggerResult(
                    track_id=track_id,
                    state=TriggerState.COOLDOWN,
                    severity=None,
                    confirmation_frames=rec.confirmation_frames,
                    trigger_type=trigger_type,
                    frame_number=frame_number,
                )
            else:
                # Cooldown expired — reset
                rec.state = TriggerState.IDLE
                rec.confirmation_frames = 0
                rec.last_published_severity = None

        # --- Advance state machine ---
        rec.trigger_type  = trigger_type
        rec.last_seen_time = now

        if rec.state == TriggerState.IDLE:
            rec.state = TriggerState.PROVISIONAL_1
            rec.confirmation_frames = 1
            severity = Severity.provisional

        elif rec.state == TriggerState.PROVISIONAL_1:
            rec.state = TriggerState.PROVISIONAL_2
            rec.confirmation_frames = 2
            severity = Severity.provisional

        elif rec.state == TriggerState.PROVISIONAL_2:
            rec.state = TriggerState.CONFIRMED_TRIGGER
            rec.confirmation_frames = 3
            rec.confirmed_at = now
            severity = Severity.confirmed
            logger.info(
                "TRIGGER CONFIRMED — track=%s type=%s frames=%d",
                track_id, trigger_type.value, rec.confirmation_frames,
            )

        elif rec.state == TriggerState.CONFIRMED_TRIGGER:
            # Keep accumulating — can escalate to CRITICAL on additional config
            rec.confirmation_frames += 1
            severity = Severity.critical if rec.confirmation_frames >= 5 else Severity.confirmed
            logger.info(
                "TRIGGER SUSTAINED — track=%s type=%s frames=%d severity=%s",
                track_id, trigger_type.value, rec.confirmation_frames, severity.value,
            )

        else:
            severity = Severity.provisional  # fallback

        # --- Dedup guard: fire-once-per-severity-transition ---
        # Publish only when severity transitions to a new (higher) level.
        # Repeated frames at the same severity are suppressed to prevent
        # alert flooding (Divergence #1 in Phase Zurich report).
        is_transition = (severity != rec.last_published_severity)
        if is_transition and severity in (Severity.confirmed, Severity.critical):
            rec.last_published_severity = severity

        return TriggerResult(
            track_id=track_id,
            state=rec.state,
            severity=severity,
            confirmation_frames=rec.confirmation_frames,
            trigger_type=trigger_type,
            frame_number=frame_number,
            is_transition=is_transition,
        )

    def miss(self, track_id: str) -> None:
        """
        Signal that track_id was NOT detected on this frame.

        CRITICAL (2-frame boundary rule):
          If the track was in PROVISIONAL_1 or PROVISIONAL_2 when it
          disappears, we reset immediately to IDLE. No event is published.
          This is the guard against foliage/shadow false positives.
        """
        rec = self._tracks.get(track_id)
        if rec is None:
            return

        if rec.state in (TriggerState.PROVISIONAL_1, TriggerState.PROVISIONAL_2):
            logger.debug(
                "track=%s missed in state=%s — RESET to IDLE. "
                "2-frame boundary rule: no event published.",
                track_id, rec.state.name,
            )
            rec.state = TriggerState.IDLE
            rec.confirmation_frames = 0
            rec.last_published_severity = None

        elif rec.state == TriggerState.CONFIRMED_TRIGGER:
            # Track lost after confirmation — enter cooldown
            rec.state = TriggerState.COOLDOWN
            rec.cooldown_until = time.monotonic() + self.cooldown_seconds
            logger.info("track=%s CONFIRMED trigger lost — entering COOLDOWN.", track_id)

    def enter_cooldown(self, track_id: str) -> None:
        """Manually put a track into cooldown (e.g. after operator ack)."""
        rec = self._get_or_create(track_id)
        rec.state = TriggerState.COOLDOWN
        rec.cooldown_until = time.monotonic() + self.cooldown_seconds

    def purge_stale(self) -> int:
        """
        Remove tracks not seen for stale_timeout_seconds.
        Call periodically (e.g. every 5s) to prevent unbounded growth.
        Returns count of purged tracks.
        """
        now = time.monotonic()
        stale = [
            tid for tid, rec in self._tracks.items()
            if (now - rec.last_seen_time) > self.stale_timeout
            and rec.state not in (TriggerState.CONFIRMED_TRIGGER, TriggerState.COOLDOWN)
        ]
        for tid in stale:
            del self._tracks[tid]
        if stale:
            logger.debug("Purged %d stale tracks: %s", len(stale), stale)
        return len(stale)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id: str) -> TrackRecord:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackRecord()
        return self._tracks[track_id]

    @property
    def active_tracks(self) -> dict:
        """Snapshot of current track states (for diagnostics)."""
        return {
            tid: {"state": rec.state.name, "frames": rec.confirmation_frames}
            for tid, rec in self._tracks.items()
        }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriggerResult:
    track_id:            str
    state:               TriggerState
    severity:            Optional[Severity]   # None means: do not publish
    confirmation_frames: int
    trigger_type:        TriggerType
    frame_number:        int
    is_transition:       bool = True          # True only on severity transitions

    @property
    def should_publish(self) -> bool:
        """Publish only on severity transitions to confirmed/critical.
        
        Prevents duplicate alert firing for the same track at the same
        severity level (Divergence #1 fix). Provisional events are still
        suppressed by pipeline logic (severity != confirmed/critical).
        """
        return self.severity is not None and self.is_transition
