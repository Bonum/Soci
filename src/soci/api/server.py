"""FastAPI server — serves the simulation state and handles player input."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from soci.engine.llm import create_llm_client
from soci.engine.simulation import Simulation
from soci.persistence.database import Database
from soci.persistence.snapshots import load_simulation, save_simulation
from soci.world.city import City
from soci.world.clock import SimClock
from soci.api.routes import router
from soci.api.websocket import ws_router

logger = logging.getLogger(__name__)

# Global simulation instance (shared across requests)
_simulation: Optional[Simulation] = None
_database: Optional[Database] = None
_sim_task: Optional[asyncio.Task] = None
_sim_paused: bool = False
_sim_speed: float = 1.0  # 1.0 = normal, 0.5 = fast, 2.0 = slow


def get_simulation() -> Simulation:
    assert _simulation is not None, "Simulation not initialized"
    return _simulation


def get_database() -> Database:
    assert _database is not None, "Database not initialized"
    return _database


async def simulation_loop(sim: Simulation, db: Database, tick_delay: float = 2.0) -> None:
    """Background task that runs the simulation continuously."""
    global _sim_paused, _sim_speed
    while True:
        try:
            if _sim_paused:
                await asyncio.sleep(0.5)
                continue
            await sim.tick()
            # Auto-save every 24 ticks
            if sim.clock.total_ticks % 24 == 0:
                await save_simulation(sim, db, "autosave")
            await asyncio.sleep(tick_delay * _sim_speed)
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
            await save_simulation(sim, db, "autosave")
            break
        except Exception as e:
            logger.error(f"Simulation tick error: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait before retrying


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage simulation lifecycle."""
    global _simulation, _database, _sim_task

    # Start up
    logger.info("Starting Soci API server...")
    llm = create_llm_client()
    db = Database()
    await db.connect()
    _database = db

    # Try to resume
    sim = await load_simulation(db, llm)
    if sim is None:
        # Create new
        config_dir = Path(__file__).parents[3] / "config"
        city = City.from_yaml(str(config_dir / "city.yaml"))
        clock = SimClock(tick_minutes=15, hour=6, minute=0)
        sim = Simulation(city=city, clock=clock, llm=llm)
        sim.load_agents_from_yaml(str(config_dir / "personas.yaml"))
        logger.info(f"Created new simulation with {len(sim.agents)} agents")

    _simulation = sim

    # Start background simulation
    _sim_task = asyncio.create_task(simulation_loop(sim, db, tick_delay=2.0))

    yield

    # Shut down
    if _sim_task:
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    await save_simulation(sim, db, "shutdown_save")
    await db.close()
    logger.info("Soci API server stopped.")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Soci — City Population Simulator",
        description="API for the LLM-powered city population simulation",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")
    app.include_router(ws_router)

    # Serve web UI
    web_dir = Path(__file__).parents[3] / "web"
    if web_dir.exists():
        @app.get("/")
        async def serve_index():
            return FileResponse(web_dir / "index.html")

        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    return app


app = create_app()
