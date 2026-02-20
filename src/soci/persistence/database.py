"""Database — SQLite persistence for simulation state."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# SOCI_DATA_DIR env var lets you point at a persistent disk (e.g. /var/data on Render).
DB_DIR = Path(os.environ.get("SOCI_DATA_DIR", "data"))
DEFAULT_DB = DB_DIR / "soci.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tick INTEGER NOT NULL,
    day INTEGER NOT NULL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER NOT NULL,
    day INTEGER NOT NULL,
    time_str TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent_id TEXT,
    location TEXT,
    description TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT NOT NULL,
    tick INTEGER NOT NULL,
    day INTEGER NOT NULL,
    location TEXT NOT NULL,
    participants_json TEXT NOT NULL,
    topic TEXT,
    turns_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_tick ON event_log(tick);
CREATE INDEX IF NOT EXISTS idx_event_agent ON event_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_conv_tick ON conversations(tick);
"""


class Database:
    """Async SQLite database for simulation persistence."""

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to the database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info(f"Database connected: {self.db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_snapshot(self, name: str, tick: int, day: int, state: dict) -> int:
        """Save a full simulation state snapshot."""
        assert self._db is not None
        cursor = await self._db.execute(
            "INSERT INTO snapshots (name, tick, day, state_json) VALUES (?, ?, ?, ?)",
            (name, tick, day, json.dumps(state)),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def load_snapshot(self, name: Optional[str] = None) -> Optional[dict]:
        """Load the latest snapshot, or a specific named one."""
        assert self._db is not None
        if name:
            cursor = await self._db.execute(
                "SELECT state_json FROM snapshots WHERE name = ? ORDER BY id DESC LIMIT 1",
                (name,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT state_json FROM snapshots ORDER BY id DESC LIMIT 1",
            )
        row = await cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupt snapshot in DB, ignoring: {e}")
        return None

    async def list_snapshots(self) -> list[dict]:
        """List all saved snapshots."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, name, created_at, tick, day FROM snapshots ORDER BY id DESC"
        )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2], "tick": r[3], "day": r[4]}
            for r in rows
        ]

    async def log_event(
        self,
        tick: int,
        day: int,
        time_str: str,
        event_type: str,
        description: str,
        agent_id: str = "",
        location: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a simulation event."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO event_log (tick, day, time_str, event_type, agent_id, location, description, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tick, day, time_str, event_type, agent_id, location, description,
             json.dumps(metadata) if metadata else None),
        )
        await self._db.commit()

    async def save_conversation(self, conv_data: dict) -> None:
        """Save a completed conversation."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO conversations (conv_id, tick, day, location, participants_json, topic, turns_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conv_data["id"],
                conv_data["turns"][-1]["tick"] if conv_data["turns"] else 0,
                0,  # Day would be tracked from clock
                conv_data["location"],
                json.dumps(conv_data["participants"]),
                conv_data.get("topic", ""),
                json.dumps(conv_data["turns"]),
            ),
        )
        await self._db.commit()

    async def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get recent events from the log."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT tick, day, time_str, event_type, agent_id, location, description "
            "FROM event_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "tick": r[0], "day": r[1], "time_str": r[2],
                "event_type": r[3], "agent_id": r[4],
                "location": r[5], "description": r[6],
            }
            for r in rows
        ]
