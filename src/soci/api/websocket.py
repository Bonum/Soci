"""WebSocket — real-time event stream for live clients and future game UI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        await websocket.send_json(message)


manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    return manager


@ws_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time event stream.

    Clients receive JSON messages with:
    - type: "tick" — new simulation tick with summary
    - type: "event" — world event occurred
    - type: "conversation" — new conversation turn
    - type: "action" — agent performed an action
    """
    await manager.connect(websocket)

    try:
        # Set up event forwarding from the simulation
        from soci.api.server import get_simulation
        sim = get_simulation()

        # Send the full current state immediately so the client is in sync
        # before the next tick fires (avoids "Day 1 6:00" on fresh connects).
        state = sim.get_state_summary()
        await manager.send_personal(websocket, {
            "type": "tick",
            "tick": sim.clock.total_ticks,
            "time": sim.clock.datetime_str,
            "state": state,
        })

        last_tick = sim.clock.total_ticks
        while True:
            try:
                # Wait for client messages (ping/pong or player input)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0,
                    )
                    # Handle client input
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ping":
                            await manager.send_personal(websocket, {"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass

                # Send state updates if tick advanced
                current_tick = sim.clock.total_ticks
                if current_tick > last_tick:
                    state = sim.get_state_summary()
                    await manager.send_personal(websocket, {
                        "type": "tick",
                        "tick": current_tick,
                        "time": sim.clock.datetime_str,
                        "state": state,
                    })
                    last_tick = current_tick

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
