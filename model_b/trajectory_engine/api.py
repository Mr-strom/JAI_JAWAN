"""
api.py — FastAPI health wrapper for the Trajectory Engine.

The MQTT subscribe/publish loop runs in a background thread (same MQTTBridge
as before). FastAPI adds a /health endpoint on top — it does NOT replace MQTT.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import signal
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mqtt_bridge import ENGINE_STATE, MQTTBridge

# ── One shared bridge instance ────────────────────────────────────────────────
_bridge = MQTTBridge()
_bridge_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the MQTT bridge in a background thread; stop it on shutdown."""
    global _bridge_thread
    _bridge_thread = threading.Thread(
        target=_bridge.start, daemon=True, name="mqtt-bridge"
    )
    _bridge_thread.start()
    yield
    _bridge.stop()


app = FastAPI(
    title="Trajectory Engine",
    description="Model B — Long-Range Trajectory Tracking Microservice",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """
    Returns current engine health.

    - status: 'healthy' once the first heartbeat fires, 'starting' before that.
    - last_heartbeat: ISO-8601 UTC timestamp of the last MQTT heartbeat, or null.
    - active_tracks: total live ByteTrack IDs across all cameras right now.
    """
    return {
        "status": ENGINE_STATE["status"],
        "engine": "trajectory",
        "last_heartbeat": ENGINE_STATE["last_heartbeat"],
        "active_tracks": ENGINE_STATE["active_tracks"],
    }
