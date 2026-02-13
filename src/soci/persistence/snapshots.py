"""Snapshots — save and load full simulation state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from soci.engine.simulation import Simulation
    from soci.engine.llm import ClaudeClient
    from soci.persistence.database import Database

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = Path("data") / "snapshots"


async def save_simulation(
    sim: Simulation,
    db: Database,
    name: str = "autosave",
) -> None:
    """Save the full simulation state to database and JSON file."""
    state = sim.to_dict()

    # Save to database
    await db.save_snapshot(
        name=name,
        tick=sim.clock.total_ticks,
        day=sim.clock.day,
        state=state,
    )

    # Also save as JSON file for easy inspection
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info(f"Simulation saved: {name} (tick {sim.clock.total_ticks}, day {sim.clock.day})")


async def load_simulation(
    db: Database,
    llm: ClaudeClient,
    name: Optional[str] = None,
) -> Optional[Simulation]:
    """Load a simulation from the database."""
    from soci.engine.simulation import Simulation

    state = await db.load_snapshot(name)
    if not state:
        # Try loading from JSON file
        if name:
            path = SNAPSHOTS_DIR / f"{name}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)

    if not state:
        logger.warning(f"No snapshot found: {name or 'latest'}")
        return None

    sim = Simulation.from_dict(state, llm)
    logger.info(f"Simulation loaded: tick {sim.clock.total_ticks}, day {sim.clock.day}")
    return sim
