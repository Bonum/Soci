"""FastAPI server — serves the simulation state and handles player input."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from soci.engine.llm import create_llm_client, PROVIDER_GROQ, PROVIDER_GEMINI, PROVIDER_HF
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
_llm_provider: str = ""  # Track which provider is active


def get_simulation() -> Simulation:
    assert _simulation is not None, "Simulation not initialized"
    return _simulation


def get_database() -> Database:
    assert _database is not None, "Database not initialized"
    return _database


def get_llm_provider() -> str:
    return _llm_provider


async def switch_llm_provider(provider: str, model: Optional[str] = None) -> None:
    """Hot-swap the LLM client on the running simulation."""
    global _llm_provider, _simulation
    assert _simulation is not None, "Simulation not initialized"
    new_llm = create_llm_client(provider=provider, model=model)
    _simulation.llm = new_llm
    _llm_provider = provider
    logger.info(f"LLM provider switched to: {provider}/{model or 'default'} ({new_llm.__class__.__name__})")


async def simulation_loop(sim: Simulation, db: Database, tick_delay: float = 2.0) -> None:
    """Background task that runs the simulation continuously."""
    global _sim_paused, _sim_speed
    # Groq: 30 req/min, Gemini free: 15 RPM, HF serverless: also slow
    is_rate_limited = _llm_provider in (PROVIDER_GROQ, PROVIDER_GEMINI, PROVIDER_HF)
    if is_rate_limited:
        tick_delay = 4.0  # Longer ticks to stay under rate limit

    while True:
        try:
            if _sim_paused:
                await asyncio.sleep(0.5)
                continue

            # At high speeds, limit LLM calls to keep ticks fast
            if _sim_speed <= 0.05:
                # 50x: skip LLM entirely, pure routine mode
                sim._skip_llm_this_tick = True
                sim._max_llm_calls_this_tick = 0
            elif _sim_speed <= 0.15:
                # 10x: max 1 conversation per tick
                sim._max_convos_this_tick = 1
                sim._max_llm_calls_this_tick = 2 if is_rate_limited else 0
            elif _sim_speed <= 0.35:
                # 5x: max 2 conversations per tick
                sim._max_convos_this_tick = 2
                sim._max_llm_calls_this_tick = 3 if is_rate_limited else 0
            else:
                sim._skip_llm_this_tick = False
                if is_rate_limited:
                    # Rate-limited providers (Groq 30 RPM, Gemini 15 RPM, HF) — budget 4 calls/tick
                    sim._max_convos_this_tick = 1
                    sim._max_llm_calls_this_tick = 4
                else:
                    # Ollama / Claude: soft cap to keep ticks responsive
                    sim._max_convos_this_tick = 3
                    sim._max_llm_calls_this_tick = 10

            await sim.tick()

            # Auto-save every 24 ticks (~6 sim-hours)
            if sim.clock.total_ticks % 24 == 0:
                await save_simulation(sim, db, "autosave")

            # At high speeds, skip the delay entirely
            delay = tick_delay * _sim_speed
            if delay > 0.05:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # Yield to event loop
        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
            await save_simulation(sim, db, "autosave")
            break
        except Exception as e:
            logger.error(f"Simulation tick error: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait before retrying


async def load_state_from_github(data_dir: Path) -> bool:
    """Fetch autosave.json from the simulation-state branch on GitHub.

    Reads from GITHUB_STATE_BRANCH (default: "simulation-state") so pushes
    never touch the master branch and never trigger Render auto-deploys.

    Env vars:
        GITHUB_TOKEN       — personal access token with repo read/write
        GITHUB_OWNER       — repo owner e.g. "alice"   (preferred, no slash)
        GITHUB_REPO_NAME   — repo name  e.g. "soci"    (preferred, no slash)
        GITHUB_REPO        — "owner/repo" fallback for existing setups
        GITHUB_STATE_BRANCH — branch name (default: "simulation-state")
        GITHUB_STATE_FILE  — path inside repo (default: "state/autosave.json")
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    owner = os.environ.get("GITHUB_OWNER", "")
    repo_name = os.environ.get("GITHUB_REPO_NAME", "")
    repo = f"{owner}/{repo_name}" if owner and repo_name else os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        return False
    path = os.environ.get("GITHUB_STATE_FILE", "state/autosave.json")
    branch = os.environ.get("GITHUB_STATE_BRANCH", "simulation-state")
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{repo}/contents/{path}",
                params={"ref": branch},
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code == 404:
                logger.info(f"No GitHub state on branch '{branch}' — starting fresh")
                return False
            resp.raise_for_status()
            content = base64.b64decode(resp.json()["content"]).decode("utf-8").strip()
            if not content:
                logger.warning("GitHub state file is empty — starting fresh")
                return False
            local_path = data_dir / "snapshots" / "autosave.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding="utf-8")
            logger.info(f"Loaded state from GitHub branch '{branch}' ({len(content):,} bytes)")
            return True
    except Exception as e:
        logger.warning(f"Could not load state from GitHub: {e}")
        return False


async def save_state_to_github(data_dir: Path) -> bool:
    """Push autosave.json to the simulation-state branch (never touches master)."""
    token = os.environ.get("GITHUB_TOKEN", "")
    owner = os.environ.get("GITHUB_OWNER", "")
    repo_name = os.environ.get("GITHUB_REPO_NAME", "")
    repo = f"{owner}/{repo_name}" if owner and repo_name else os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        return False
    path = os.environ.get("GITHUB_STATE_FILE", "state/autosave.json")
    branch = os.environ.get("GITHUB_STATE_BRANCH", "simulation-state")
    local_path = data_dir / "snapshots" / "autosave.json"
    if not local_path.exists():
        logger.warning("No autosave.json to push to GitHub")
        return False
    try:
        content_bytes = local_path.read_bytes()
        encoded = base64.b64encode(content_bytes).decode("ascii")
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        async with httpx.AsyncClient() as client:
            # Fetch current file SHA on the state branch (needed to update)
            sha: Optional[str] = None
            get_resp = await client.get(
                f"https://api.github.com/repos/{repo}/contents/{path}",
                params={"ref": branch},
                headers=headers,
                timeout=30.0,
            )
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")
            elif get_resp.status_code == 404:
                # Branch or file doesn't exist yet — create the branch from master
                ref_resp = await client.get(
                    f"https://api.github.com/repos/{repo}/git/ref/heads/master",
                    headers=headers,
                    timeout=15.0,
                )
                if ref_resp.status_code == 200:
                    master_sha = ref_resp.json()["object"]["sha"]
                    await client.post(
                        f"https://api.github.com/repos/{repo}/git/refs",
                        headers=headers,
                        json={"ref": f"refs/heads/{branch}", "sha": master_sha},
                        timeout=15.0,
                    )
                    logger.info(f"Created GitHub branch '{branch}' for state storage")

            body: dict = {
                "message": "chore: save simulation state",
                "content": encoded,
                "branch": branch,
            }
            if sha:
                body["sha"] = sha

            put_resp = await client.put(
                f"https://api.github.com/repos/{repo}/contents/{path}",
                headers=headers,
                json=body,
                timeout=60.0,
            )
            put_resp.raise_for_status()
            logger.info(f"Saved state to GitHub branch '{branch}' ({len(content_bytes):,} bytes)")
            return True
    except Exception as e:
        logger.warning(f"Could not save state to GitHub: {e}")
        return False


def _choose_provider() -> str:
    """Let the user choose an LLM provider on startup.

    Priority: SOCI_PROVIDER env var > LLM_PROVIDER env var > interactive prompt.
    """
    # Check explicit env vars first
    provider = os.environ.get("SOCI_PROVIDER", "").lower() or os.environ.get("LLM_PROVIDER", "").lower()
    if provider in ("claude", "groq", "gemini", "hf", "ollama"):
        return provider

    # Check which keys are available
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_hf = bool(os.environ.get("HF_TOKEN"))

    # Free providers only; Claude can be forced via SOCI_PROVIDER / LLM_PROVIDER.
    options = []
    if has_groq:
        options.append(("groq", "Groq (free tier, 30 req/min)"))
    if has_gemini:
        options.append(("gemini", "Gemini (free tier, 15 req/min via AI Studio)"))
    if has_hf:
        options.append(("hf", "HF Inference (requires HF PRO / credits)"))
    options.append(("ollama", "Ollama (local, free, no rate limit)"))

    # If only one option, use it
    if len(options) == 1:
        chosen = options[0][0]
        print(f"  LLM Provider: {options[0][1]}")
        return chosen

    # Interactive selection
    print("\n  Choose LLM provider:")
    for i, (key, desc) in enumerate(options, 1):
        print(f"    {i}. {desc}")

    try:
        choice = input(f"  Enter choice [1-{len(options)}] (default: 1): ").strip()
        idx = int(choice) - 1 if choice else 0
        if 0 <= idx < len(options):
            chosen = options[idx][0]
        else:
            chosen = options[0][0]
    except (ValueError, EOFError):
        chosen = options[0][0]

    print(f"  -> Using {chosen}")
    return chosen


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage simulation lifecycle."""
    global _simulation, _database, _sim_task, _llm_provider

    # Start up
    logger.info("Starting Soci API server...")

    # Choose provider
    _llm_provider = _choose_provider()
    llm = create_llm_client(provider=_llm_provider)
    logger.info(f"LLM provider: {_llm_provider} ({llm.__class__.__name__})")

    db = Database()
    await db.connect()
    _database = db

    # Pull saved state from GitHub before trying to load locally
    data_dir = Path(os.environ.get("SOCI_DATA_DIR", "data"))
    await load_state_from_github(data_dir)

    # Try to resume — any failure falls back to a fresh simulation
    sim = None
    try:
        sim = await load_simulation(db, llm)
    except Exception as e:
        logger.warning(f"Failed to load saved simulation, starting fresh: {e}")

    if sim is None:
        # Create new
        config_dir = Path(__file__).parents[3] / "config"
        city = City.from_yaml(str(config_dir / "city.yaml"))
        clock = SimClock(tick_minutes=15, hour=6, minute=0)
        sim = Simulation(city=city, clock=clock, llm=llm)
        sim.load_agents_from_yaml(str(config_dir / "personas.yaml"))
        # Scale to target agent count with procedural generation
        target_agents = int(os.environ.get("SOCI_AGENTS", "50"))
        if len(sim.agents) < target_agents:
            sim.generate_agents(target_agents - len(sim.agents))
        logger.info(f"Created new simulation with {len(sim.agents)} agents")

    _simulation = sim

    # Start background simulation
    # SOCI_TICK_DELAY: seconds to sleep between ticks (default 0.5).
    # Set to 0 to let LLM latency pace the simulation naturally.
    tick_delay = float(os.environ.get("SOCI_TICK_DELAY", "0.5"))
    _sim_task = asyncio.create_task(simulation_loop(sim, db, tick_delay=tick_delay))

    yield

    # Shut down
    if _sim_task:
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    await save_simulation(sim, db, "shutdown_save")
    # Push state to GitHub so it survives the next redeploy
    await save_state_to_github(data_dir)
    await db.close()
    logger.info("Soci API server stopped.")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Soci — City Population Simulator",
        description="API for the LLM-powered city population simulation",
        version="0.2.0",
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
