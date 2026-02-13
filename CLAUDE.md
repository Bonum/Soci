# Soci — LLM-Powered City Population Simulator

## Project Overview
Simulates a diverse population of AI people living in a city using Claude as the reasoning engine. Each agent has a unique persona, memory stream, needs, and relationships. Inspired by Simile AI / Stanford Generative Agents research.

## Tech Stack
- Python 3.10+ (Anaconda ml-env)
- Anthropic Claude API (Sonnet for novel situations, Haiku for routine)
- FastAPI + WebSocket (API server)
- SQLite via aiosqlite (persistence)
- Rich (terminal dashboard)
- YAML (city/persona config)

## Key Commands
```bash
# Run with Anaconda ml-env:
"C:/Users/xabon/.conda/envs/ml-env/python.exe" main.py --ticks 20 --agents 5
"C:/Users/xabon/.conda/envs/ml-env/python.exe" test_simulation.py

# API server:
"C:/Users/xabon/.conda/envs/ml-env/python.exe" -m uvicorn soci.api.server:app --host 0.0.0.0 --port 8000
```

## Architecture
- `src/soci/world/` — City map, simulation clock, world events
- `src/soci/agents/` — Agent cognition: persona, memory, needs, relationships
- `src/soci/actions/` — Action types: movement, activities, conversation, social
- `src/soci/engine/` — Simulation loop, scheduler, entropy management, LLM client
- `src/soci/persistence/` — SQLite database, save/load snapshots
- `src/soci/api/` — FastAPI REST + WebSocket server
- `config/` — City layout (12 locations) and personas (20 characters)

## Agent Cognition Loop
Each tick per agent: OBSERVE → REFLECT → PLAN → ACT → REMEMBER

## Conventions
- All async (LLM calls are I/O-bound)
- Dataclasses with `to_dict()` / `from_dict()` for serialization
- YAML config for city layout and personas
- Cost tracking built into LLM client
