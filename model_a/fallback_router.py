"""
Fallback Router — Model B Heartbeat Monitor & Camera Fallback Manager
SIH26187 | Model A | Step 13 of pipeline

Spec (Rule #13 / Step 13):
  If a Model B engine's heartbeat is missing/stale >30s, route that camera's
  traffic through Model A's basic motion+trigger detection.
  This prevents total blindness but is deliberately lightweight.
  Fallback routing engages ONLY on that specific camera's set.

Rules:
  - Do NOT auto-restart failed Model B engines. Manual intervention only.
  - Slow/delayed heartbeat (< timeout) → wait. Do not engage fallback prematurely.
  - Dead heartbeat (> timeout) → engage safety floor for affected cameras.
  - When heartbeat resumes, record recovery time. Caller decides when to
    re-hand-off to Model B (conservative: wait for 3 consecutive heartbeats).

Fallback is per-engine, per-camera:
  Engine A dead → cameras [cam_01, cam_02] fallback. Cameras [cam_03] unaffected.
  Engine B dead → cameras [cam_03] fallback. Cameras [cam_01, cam_02] unaffected.

States per engine:
  NORMAL     → heartbeat current (within timeout)
  FALLBACK   → heartbeat stale > timeout. Safety floor active.
  RECOVERING → heartbeat resumed. Waiting for stability before handing back.

The timeout is configurable (default 30s). Tests use short timeouts (0.05s).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default timeout — matches spec Rule #13
# ---------------------------------------------------------------------------

_DEFAULT_HEARTBEAT_TIMEOUT_S  = 30.0
_RECOVERY_HEARTBEAT_COUNT     = 3      # consecutive beats before leaving RECOVERING


# ---------------------------------------------------------------------------
# Engine states
# ---------------------------------------------------------------------------

class EngineState(Enum):
    NORMAL     = auto()
    FALLBACK   = auto()
    RECOVERING = auto()


# ---------------------------------------------------------------------------
# Per-engine record
# ---------------------------------------------------------------------------

@dataclass
class EngineRecord:
    engine_id:            str
    managed_cameras:      List[str]          = field(default_factory=list)
    state:                EngineState        = EngineState.NORMAL
    last_heartbeat_time:  Optional[float]    = None   # monotonic
    fallback_engaged_at:  Optional[float]    = None   # monotonic, when fallback started
    recovery_beat_count:  int                = 0      # consecutive beats in RECOVERING


# ---------------------------------------------------------------------------
# FallbackRouter
# ---------------------------------------------------------------------------

class FallbackRouter:
    """
    Tracks Model B engine heartbeats and decides which cameras need
    Model A safety floor coverage.

    Usage::

        router = FallbackRouter(heartbeat_timeout_s=30.0)

        # Register which cameras each engine handles
        router.register_engine("face_engine",     cameras=["cam_01", "cam_02"])
        router.register_engine("posture_engine",  cameras=["cam_03", "cam_04"])

        # On each Model B heartbeat MQTT message:
        router.update_heartbeat("face_engine")

        # On each pipeline tick (e.g. every 1s):
        fallback_cams = router.get_fallback_cameras()
        for cam_id in fallback_cams:
            safety_floor.process(cam_id, frame)

        # To check a specific engine:
        if router.is_fallback_active("face_engine"):
            # run safety floor for that engine's cameras
    """

    def __init__(
        self,
        heartbeat_timeout_s: float = _DEFAULT_HEARTBEAT_TIMEOUT_S,
        recovery_beat_count: int   = _RECOVERY_HEARTBEAT_COUNT,
    ) -> None:
        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive.")
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.recovery_beat_count = recovery_beat_count
        self._engines: Dict[str, EngineRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_engine(self, engine_id: str, cameras: List[str]) -> None:
        """
        Register a Model B engine and the cameras it manages.
        Each camera should belong to exactly one engine.
        """
        self._engines[engine_id] = EngineRecord(
            engine_id       = engine_id,
            managed_cameras = list(cameras),
            last_heartbeat_time = None,   # no beats received yet
        )
        logger.info(
            "Engine registered: %s → cameras %s", engine_id, cameras
        )

    # ------------------------------------------------------------------
    # Heartbeat update (called when MQTT heartbeat arrives from Model B)
    # ------------------------------------------------------------------

    def update_heartbeat(self, engine_id: str) -> None:
        """
        Record a heartbeat for the given engine.
        If engine was in FALLBACK, moves it to RECOVERING.
        If engine was in RECOVERING, increments recovery counter.
        """
        rec = self._get_engine(engine_id)
        rec.last_heartbeat_time = time.monotonic()

        if rec.state == EngineState.FALLBACK:
            rec.state = EngineState.RECOVERING
            rec.recovery_beat_count = 1
            logger.warning(
                "Engine '%s' RECOVERING — first heartbeat after fallback. "
                "Need %d more beats before handing back.",
                engine_id, self.recovery_beat_count - 1,
            )

        elif rec.state == EngineState.RECOVERING:
            rec.recovery_beat_count += 1
            logger.info(
                "Engine '%s' recovery beat %d/%d.",
                engine_id, rec.recovery_beat_count, self.recovery_beat_count,
            )
            if rec.recovery_beat_count >= self.recovery_beat_count:
                rec.state = EngineState.NORMAL
                rec.fallback_engaged_at = None
                rec.recovery_beat_count = 0
                logger.info(
                    "Engine '%s' NORMAL — %d consecutive heartbeats received. "
                    "Fallback disengaged.",
                    engine_id, self.recovery_beat_count,
                )

        elif rec.state == EngineState.NORMAL:
            logger.debug("Heartbeat received from engine '%s'.", engine_id)

    # ------------------------------------------------------------------
    # Fallback state queries (called on each pipeline tick)
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        """
        Evaluate all engines. Transition NORMAL → FALLBACK for any engine
        whose heartbeat is stale beyond the timeout.

        Call this periodically (e.g. every 1s) from the main pipeline loop.
        """
        now = time.monotonic()
        for rec in self._engines.values():
            if rec.state != EngineState.NORMAL:
                continue  # already in fallback or recovering — don't re-evaluate

            if rec.last_heartbeat_time is None:
                # No heartbeat ever received — treat as dead if timeout has passed
                # since registration (conservative: use engine registration time)
                # For simplicity: if never received, immediately treat as dead
                rec.state = EngineState.FALLBACK
                rec.fallback_engaged_at = now
                logger.error(
                    "Engine '%s' has NEVER sent a heartbeat. "
                    "Fallback routing engaged for cameras: %s",
                    rec.engine_id, rec.managed_cameras,
                )
            else:
                staleness = now - rec.last_heartbeat_time
                if staleness > self.heartbeat_timeout_s:
                    rec.state = EngineState.FALLBACK
                    rec.fallback_engaged_at = now
                    logger.error(
                        "Engine '%s' heartbeat STALE (%.1fs > %.1fs timeout). "
                        "Fallback routing engaged for cameras: %s",
                        rec.engine_id, staleness, self.heartbeat_timeout_s,
                        rec.managed_cameras,
                    )

    def is_fallback_active(self, engine_id: str) -> bool:
        """True if the engine is in FALLBACK or RECOVERING state."""
        rec = self._engines.get(engine_id)
        if rec is None:
            return False
        return rec.state in (EngineState.FALLBACK, EngineState.RECOVERING)

    def get_fallback_cameras(self) -> List[str]:
        """
        Return the list of cameras currently needing safety floor coverage.
        Cameras managed by NORMAL engines are NOT included.
        """
        result: List[str] = []
        for rec in self._engines.values():
            if rec.state in (EngineState.FALLBACK, EngineState.RECOVERING):
                result.extend(rec.managed_cameras)
        return result

    def get_normal_cameras(self) -> List[str]:
        """Return cameras whose engines are NORMAL (full Model B available)."""
        result: List[str] = []
        for rec in self._engines.values():
            if rec.state == EngineState.NORMAL:
                result.extend(rec.managed_cameras)
        return result

    def engine_state(self, engine_id: str) -> Optional[EngineState]:
        """Return the current EngineState for an engine, or None if not registered."""
        rec = self._engines.get(engine_id)
        return rec.state if rec else None

    def staleness_seconds(self, engine_id: str) -> Optional[float]:
        """
        Return seconds since last heartbeat for an engine.
        Returns None if engine not registered or never received a heartbeat.
        """
        rec = self._engines.get(engine_id)
        if rec is None or rec.last_heartbeat_time is None:
            return None
        return time.monotonic() - rec.last_heartbeat_time

    def snapshot(self) -> Dict[str, dict]:
        """Diagnostic snapshot of all engine states."""
        return {
            engine_id: {
                "state":   rec.state.name,
                "cameras": rec.managed_cameras,
                "stale_s": self.staleness_seconds(engine_id),
            }
            for engine_id, rec in self._engines.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_engine(self, engine_id: str) -> EngineRecord:
        if engine_id not in self._engines:
            # Auto-register unknown engines with empty camera list
            logger.warning(
                "Heartbeat from unregistered engine '%s'. "
                "Auto-registering with no cameras.",
                engine_id,
            )
            self._engines[engine_id] = EngineRecord(engine_id=engine_id)
        return self._engines[engine_id]
