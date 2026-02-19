# Soci — LLM-Powered City Population Simulator

Simulates a diverse population of AI people living in a city using Claude as the reasoning engine. Each agent has a unique persona, memory stream, needs, and relationships.

Inspired by [Stanford Generative Agents (Joon Park et al.)](https://arxiv.org/abs/2304.03442), CitySim, AgentSociety, and a16z ai-town.

---

## Features

- AI agents with unique personas, goals, and memories
- Maslow-inspired needs system (hunger, energy, social, purpose, comfort, fun)
- Relationship graph with familiarity, trust, and sentiment
- Agent cognition loop: **OBSERVE → REFLECT → PLAN → ACT → REMEMBER**
- Web UI with animated city map, zoom, pan, and agent inspector
- Romance and social dynamics
- Speed controls and real-time WebSocket updates
- Cost-efficient model routing (Haiku for routine, Sonnet for novel situations)

---

## Tech Stack

- Python 3.10+ (Anaconda ml-env)
- Anthropic Claude API
- FastAPI + WebSocket
- SQLite via aiosqlite
- Rich (terminal dashboard)
- YAML config

---

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Bonum/Soci.git
   cd Soci
   ```

2. **Install dependencies** (inside your Python environment)
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Anthropic API key**
   ```bash
   export ANTHROPIC_API_KEY=sk-...
   ```

---

## Running

### Terminal simulation
```bash
python main.py --ticks 20 --agents 5
```

### Web UI (local only)
```bash
python -m uvicorn soci.api.server:app --host 127.0.0.1 --port 8000
```
Then open `http://localhost:8000` in your browser.

### Web UI (accessible from LAN)
```bash
python -m uvicorn soci.api.server:app --host 0.0.0.0 --port 8000
```
Then from any device on your network, open `http://<host-ip>:8000`.

---

## Accessing from Another Computer on LAN

1. Start the server with `--host 0.0.0.0` (see above).

2. Find your host machine's local IP:
   ```bash
   # Windows
   ipconfig
   # Look for IPv4 Address, e.g. 192.168.1.42
   ```

3. Allow port 8000 through Windows Firewall (if needed):
   ```bash
   netsh advfirewall firewall add rule name="Soci Port 8000" dir=in action=allow protocol=TCP localport=8000
   ```

4. From any device on the same network, open:
   ```
   http://192.168.1.42:8000
   ```

---

## Architecture

```
src/soci/
  world/        — City map, simulation clock, world events
  agents/       — Agent cognition: persona, memory, needs, relationships
  actions/      — Movement, activities, conversation, social actions
  engine/       — Simulation loop, scheduler, entropy management, LLM client
  persistence/  — SQLite database, save/load snapshots
  api/          — FastAPI REST + WebSocket server
config/
  city.yaml     — City layout (12 locations)
  personas.yaml — Character definitions (20 agents)
```

---

## Web UI Controls

| Action | How |
|---|---|
| Zoom | Scroll wheel or +/- buttons |
| Fit view | Fit button |
| Pan | Drag canvas or use sliders |
| Rectangle select | Click ⬚ button, then drag |

---

## Agent Cognition

Each simulation tick, every agent runs:

```
OBSERVE  — perceive nearby agents, objects, events
REFLECT  — update beliefs and emotional state
PLAN     — decide what to do next
ACT      — execute action (move, talk, work, rest…)
REMEMBER — store important events to memory stream
```

Memory entries are scored by importance (1–10) with recency decay, retrieved by relevance score.

---

## License

MIT
