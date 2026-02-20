# Soci — LLM-Powered City Population Simulator

Simulates a diverse population of AI people living in a city using an LLM as the reasoning engine. Each agent has a unique persona, memory stream, needs, and relationships.

Inspired by [Stanford Generative Agents (Joon Park et al.)](https://arxiv.org/abs/2304.03442), CitySim, AgentSociety, and a16z ai-town.

**Live demo:** https://soci-tl3c.onrender.com

---

## Features

- AI agents with unique personas, goals, and memories
- Maslow-inspired needs system (hunger, energy, social, purpose, comfort, fun)
- Relationship graph with familiarity, trust, sentiment, and romance
- Agent cognition loop: **OBSERVE → REFLECT → PLAN → ACT → REMEMBER**
- Web UI with animated city map, zoom, pan, and agent inspector
- Road-based movement with L-shaped routing (agents walk along streets)
- Agent animations: walking (profile/back view), sleeping on bed
- Speed controls (1x → 50x) and real-time WebSocket sync across browsers
- **Player login** — register an account, get your own agent on the map, chat with NPCs
- Multi-LLM support: Groq (free tier), Anthropic Claude, Ollama (local)
- GitHub-based state persistence (survives server reboots and redeploys)
- Cost-efficient model routing (Haiku for routine, Sonnet for novel situations)

---

## Tech Stack

- Python 3.10+
- Anthropic Claude API / Groq / Ollama
- FastAPI + WebSocket
- SQLite via aiosqlite
- YAML config

---

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Bonum/Soci.git
   cd Soci
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key** (choose one provider)
   ```bash
   # Groq (free tier — recommended for cloud)
   export GROQ_API_KEY=gsk_...

   # Anthropic Claude
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

---

## Running

### Web UI (local)
```bash
python -m uvicorn soci.api.server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

### Terminal simulation (no UI)
```bash
python main.py --ticks 20 --agents 5
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOCI_PROVIDER` | auto-detect | LLM provider: `groq`, `claude`, `ollama` |
| `GROQ_API_KEY` | — | Groq API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `SOCI_AGENTS` | `50` | Starting agent count |
| `SOCI_TICK_DELAY` | `0.5` | Seconds between simulation ticks |
| `SOCI_DATA_DIR` | `data` | Directory for SQLite DB and snapshots |
| `GITHUB_TOKEN` | — | GitHub PAT for state persistence across deploys |
| `GITHUB_REPO` | — | `owner/repo` for state persistence |
| `GITHUB_STATE_BRANCH` | `simulation-state` | Branch used for state (never touches main) |

---

## Deploying to Render (free tier)

1. Connect your GitHub repo in Render.
2. Set **Start Command**: `python -m uvicorn soci.api.server:app --host 0.0.0.0 --port $PORT`
3. Set env vars: `SOCI_PROVIDER`, `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`), `GITHUB_TOKEN`, `GITHUB_REPO`
4. Add an **Ignore Command** to prevent state-file commits from triggering redeploys:
   ```
   [ "$(git diff --name-only HEAD~1 HEAD | grep -v '^state/' | wc -l)" = "0" ]
   ```

Simulation state is automatically saved to a `simulation-state` branch on shutdown and restored on the next startup — no persistent disk required.

---

## Architecture

```
src/soci/
  world/        — City map, simulation clock, world events
  agents/       — Agent cognition: persona, memory, needs, relationships
  actions/      — Movement, activities, conversation, social actions
  engine/       — Simulation loop, scheduler, entropy, LLM client
  persistence/  — SQLite database, save/load snapshots
  api/          — FastAPI REST + WebSocket server
config/
  city.yaml     — City layout and building positions
  personas.yaml — Named character definitions
web/
  index.html    — Single-file web UI
```

---

## Web UI

| Action | How |
|--------|-----|
| Zoom | Scroll wheel or +/− buttons |
| Fit view | Fit button |
| Pan | Drag canvas or use sliders |
| Rectangle zoom | Click ⬚, then drag |
| Inspect agent | Click agent on map or in list |
| **Login / play** | Register → your agent appears on the map |
| **Talk to NPC** | Select any agent → "Talk to [Name]" button |
| **Move** | Player panel → location dropdown → Go |
| **Edit profile** | Player panel → Edit Profile |
| **Add plans** | Player panel → My Plans |

---

## Player Mode

Register an account to join the simulation as a participant:

1. Click **Register** in the login modal (or skip to observe only).
2. Your agent appears immediately on the map with a **gold ring** to identify you.
3. Click **Edit Profile** to set your name, age, occupation, background, and personality traits.
4. Click any NPC → **Talk to [Name]** to start a conversation — they reply in character via LLM.
5. Use **My Plans** to add goals (e.g. *"Go to the park and meet new people"*).

Multiple users can be logged in simultaneously — each controls their own agent.

---

## Agent Cognition

Each simulation tick, every NPC agent runs:

```
OBSERVE  — perceive nearby agents, events, environment
REFLECT  — update beliefs and emotional state
PLAN     — decide what to do next
ACT      — execute action (move, talk, work, rest, sleep…)
REMEMBER — store important events to memory stream
```

Memory entries are scored by importance (1–10) with recency decay, retrieved by relevance score.

---

## License

MIT
