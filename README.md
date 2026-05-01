---
title: Soci
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Soci — LLM-Powered City Population Simulator

Simulates a diverse population of AI people living in a city using an LLM as the reasoning engine. Each agent has a unique persona, memory stream, needs, and relationships.

Inspired by [Stanford Generative Agents (Joon Park et al.)](https://arxiv.org/abs/2304.03442), CitySim, AgentSociety, and a16z ai-town.

**Live demo:** https://huggingface.co/spaces/RayMelius/soci

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
- **LLM probability slider** — tune AI usage from 0–100% to stay within free-tier quotas
- **Player login** — register an account, get your own agent on the map, chat with NPCs
- Multi-LLM support: Gemini (free tier), Groq (free tier), Anthropic Claude, Ollama (local)
- GitHub-based state persistence (survives server reboots and redeploys)
- Cost-efficient model routing (Haiku for routine, Sonnet for novel situations)
- Daily quota circuit-breaker with warnings at 50 / 70 / 90 / 99% usage

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser  (web/index.html — single-file Vue-less UI)    │
│  • Canvas city map  • Agent inspector  • Chat panel     │
│  • Speed / LLM-probability sliders  • Login modal       │
└────────┬──────────────────────────────┬─────────────────┘
         │ REST  GET /api/*             │ WebSocket /ws
         │ POST /api/controls/*         │ push events
         ▼                              ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI Server  (soci/api/server.py)                    │
│  • lifespan: load state → start sim loop                │
│  • routes.py  — REST endpoints                          │
│  • websocket.py — broadcast tick events                 │
└────────────────────────┬─────────────────────────────────┘
                         │ asyncio.create_task
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Simulation Loop  (background task)                      │
│  tick every N sec  →  sim.tick()  →  sleep              │
│  respects: _sim_paused / _sim_speed / llm_call_prob      │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Simulation.tick()  (engine/simulation.py)               │
│                                                          │
│  1. Entropy / world events                               │
│  2. Daily plan generation  ──► LLM (if prob gate ✓)     │
│  3. Agent needs + routine actions  (no LLM)             │
│  4. LLM action decisions  ─────► LLM (if prob gate ✓)  │
│  5. Conversation turns  ───────► LLM (if prob gate ✓)  │
│  6. New conversation starts  ──► LLM (if prob gate ✓)  │
│  7. Reflections  ──────────────► LLM (if prob gate ✓)  │
│  8. Romance / relationship updates  (no LLM)            │
│  9. Clock advance                                        │
└────────────────────────┬─────────────────────────────────┘
                         │ await llm.complete_json()
                         ▼
┌──────────────────────────────────────────────────────────┐
│  LLM Client  (engine/llm.py)                             │
│  • Rate limiter (asyncio.Lock, min interval per RPM)     │
│  • Daily usage counter → warns at 50/70/90/99%          │
│  • Quota circuit-breaker (expires midnight Pacific)      │
│  • Providers: GeminiClient / GroqClient /                │
│               ClaudeClient / HFInferenceClient /         │
│               OllamaClient                               │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP (httpx async)
                         ▼
              External LLM API  (Gemini / Groq / …)
```

---

## Message Flow — One Simulation Tick

```
Browser poll (3s)          Simulation background loop
      │                              │
      │  GET /api/city              tick_delay (4s Gemini, 0.5s Ollama)
      │◄────────────────────────────┤
      │                             │ sim.tick()
      │                             │
      │                   ┌─────────┴──────────┐
      │                   │  For each agent:   │
      │                   │  tick_needs()      │
      │                   │  check routine     │──► execute routine action (no LLM)
      │                   │  roll prob gate    │
      │                   │  _decide_action()  │──► LLM call ──► AgentAction JSON
      │                   │  _execute_action() │
      │                   └─────────┬──────────┘
      │                             │
      │                   ┌─────────┴──────────┐
      │                   │  Conversations:    │
      │                   │  continue_conv()   │──► LLM call ──► dialogue turn
      │                   │  new conv start    │──► LLM call ──► opening line
      │                   └─────────┬──────────┘
      │                             │
      │                   ┌─────────┴──────────┐
      │                   │  Reflections:      │
      │                   │  should_reflect()?  │──► LLM call ──► memory insight
      │                   └─────────┬──────────┘
      │                             │
      │                        clock.tick()
      │                             │
      │  WebSocket push             │
      │◄── events/state ────────────┘
      │
[browser updates map, event log, agent inspector]
```

---

## Agent Cognition Loop

```
Every tick, each NPC agent runs:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   OBSERVE ──► perceive nearby agents, events,        │
  │               location, time of day                  │
  │      │                                               │
  │      ▼                                               │
  │   REFLECT ──► check memory.should_reflect()          │
  │               LLM synthesises insight from recent    │
  │               memories  →  stored as reflection      │
  │      │                                               │
  │      ▼                                               │
  │   PLAN  ───► if no daily plan: LLM generates         │
  │               ordered list of goals for the day      │
  │               (or routine fills the plan — no LLM)   │
  │      │                                               │
  │      ▼                                               │
  │   ACT  ────► routine slot?  → execute directly       │
  │               no slot?      → LLM picks action       │
  │               action types: move / work / eat /      │
  │               sleep / socialise / leisure / rest     │
  │      │                                               │
  │      ▼                                               │
  │   REMEMBER ► add_observation() to memory stream      │
  │               importance 1–10, recency decay,        │
  │               retrieved by relevance score           │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  LLM budget per tick (rate-limited providers):
    max_llm_calls_this_tick = 1   (Gemini/Groq/HF)
    llm_call_probability    = 0.45 (Gemini default → ~10h/day)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| API server | FastAPI + Uvicorn |
| Real-time | WebSocket (FastAPI) |
| Database | SQLite via aiosqlite |
| LLM providers | Gemini · Groq · Anthropic Claude · HF Inference · Ollama |
| Config | YAML (city layout, agent personas) |
| State persistence | GitHub API (simulation-state branch) |
| Container | Docker (HF Spaces / Render) |

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- At least one LLM API key — or [Ollama](https://ollama.ai) installed locally (free, no key needed)

### Install

```bash
git clone https://github.com/Bonum/Soci.git
cd Soci
pip install -e .
```

### Configure

Copy the example env file and edit it:

```bash
cp .env.example .env
```

By default the **NN provider** is active — a local ONNX model, free and fast, no API key needed.
To switch providers, uncomment the relevant lines in `.env`:

```bash
# Ollama (local, no key):
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1

# Gemini (free tier — generous quota):
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...

# Groq (free tier — fast):
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...

# Anthropic Claude (paid — highest quality):
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
# Web UI (recommended)
python -m uvicorn soci.api.server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000

# Terminal only
python main.py --ticks 20 --agents 5
```

---

## Deploying to the Internet

### Option 1 — Hugging Face Spaces (free, recommended)

HF Spaces runs the Docker container for free with automatic HTTPS.

1. **Create a Space** at https://huggingface.co/new-space
   - SDK: **Docker**
   - Visibility: Public

2. **Add the HF remote** and push:
   ```bash
   git remote add hf https://YOUR_HF_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_HF_USERNAME/soci
   git push hf master:main
   ```
   Get a write token at https://huggingface.co/settings/tokens (select *Write* + *Inference Providers* permissions).

3. **Add Space secrets** (Settings → Variables and Secrets):

   | Secret | Value |
   |--------|-------|
   | `SOCI_PROVIDER` | `gemini` |
   | `GEMINI_API_KEY` | your AI Studio key |
   | `GITHUB_TOKEN` | GitHub PAT (repo read/write) |
   | `GITHUB_OWNER` | your GitHub username |
   | `GITHUB_REPO_NAME` | `Soci` |

4. Your Space rebuilds automatically on every push. Visit
   `https://YOUR_HF_USERNAME-soci.hf.space`

> **Free-tier tip:** Gemini free tier = 5 RPM, ~1500 requests/day (resets midnight Pacific).
> The default LLM probability is set to **45%** which gives ~10 hours of AI-driven simulation per day.
> Use the 🧠 slider in the toolbar to adjust at runtime.

---

### Option 2 — Render (free tier)

1. Connect your GitHub repo at https://render.com/new
2. Choose **Web Service** → Docker
3. Set **Start Command**:
   ```
   python -m uvicorn soci.api.server:app --host 0.0.0.0 --port $PORT
   ```
4. Set environment variables in the Render dashboard:

   | Variable | Value |
   |----------|-------|
   | `SOCI_PROVIDER` | `gemini` or `groq` |
   | `GEMINI_API_KEY` | your key |
   | `GITHUB_TOKEN` | GitHub PAT |
   | `GITHUB_OWNER` | your GitHub username |
   | `GITHUB_REPO_NAME` | `Soci` |

5. To prevent state-file commits from triggering redeploys, set **Ignore Command**:
   ```
   [ "$(git diff --name-only HEAD~1 HEAD | grep -v '^state/' | wc -l)" = "0" ]
   ```

> **Note:** Render free tier spins down after 15 min of inactivity. Simulation state is saved to GitHub on shutdown and restored on the next boot — no data is lost.

---

### Option 3 — Railway

1. Go to https://railway.app → **New Project** → **Deploy from GitHub repo**
2. Railway auto-detects the Dockerfile
3. Add environment variables in the Railway dashboard (same as Render above)
4. Railway assigns a public URL automatically

---

### Option 4 — Local + Ngrok (quick public URL for testing)

```bash
# Start the server
python -m uvicorn soci.api.server:app --host 0.0.0.0 --port 8000 &

# Expose it publicly (install ngrok first: https://ngrok.com)
ngrok http 8000
# Copy the https://xxxx.ngrok.io URL and share it
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOCI_PROVIDER` | auto-detect | LLM provider: `gemini` · `groq` · `claude` · `hf` · `ollama` |
| `GEMINI_API_KEY` | — | Google AI Studio key (free tier: 5 RPM, ~1500 RPD) |
| `GROQ_API_KEY` | — | Groq API key (free tier: 30 RPM) |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `SOCI_LLM_PROB` | per-provider | LLM call probability 0–1 (`0.45` Gemini · `0.7` Groq · `1.0` Ollama) |
| `GEMINI_DAILY_LIMIT` | `1500` | Override Gemini daily request quota for warning thresholds |
| `SOCI_AGENTS` | `50` | Starting agent count |
| `SOCI_TICK_DELAY` | `0.5` | Seconds between simulation ticks (overridden to 4.0 for rate-limited providers) |
| `SOCI_DATA_DIR` | `data` | Directory for SQLite DB and snapshots |
| `GITHUB_TOKEN` | — | GitHub PAT for state persistence across deploys |
| `GITHUB_OWNER` | — | GitHub repo owner (e.g. `alice`) |
| `GITHUB_REPO_NAME` | — | GitHub repo name (e.g. `Soci`) |
| `GITHUB_STATE_BRANCH` | `simulation-state` | Branch used for state snapshots (never touches main) |
| `GITHUB_STATE_FILE` | `state/autosave.json` | Path inside repo for state file |
| `PORT` | `8000` | HTTP port (set to `7860` on HF Spaces automatically) |

---

## Web UI Controls

| Control | How |
|---------|-----|
| Zoom | Scroll wheel or **＋ / －** buttons |
| Fit view | **Fit** button |
| Pan | Drag canvas or use sliders |
| Rectangle zoom | Click **⬚**, then drag |
| Inspect agent | Click agent on map or in sidebar list |
| Speed | **🐢 1x 2x 5x 10x 50x** buttons |
| LLM usage | **🧠** slider (0–100%) — tune AI call frequency |
| Switch LLM | Click the provider badge (e.g. **✦ Gemini 2.0 Flash**) |
| **Login / play** | Register → your agent appears with a gold ring |
| **Talk to NPC** | Select agent → **Talk to [Name]** button |
| **Move** | Player panel → location dropdown → **Go** |
| **Edit profile** | Player panel → **Edit Profile** |
| **Add plans** | Player panel → **My Plans** |

---

## LLM Provider Comparison

| Provider | Free tier | RPM | Daily limit | Best for |
|----------|-----------|-----|-------------|----------|
| **Gemini 2.0 Flash** | ✅ Yes | 5 | ~1500 req | Cloud demos (default) |
| **Groq Llama 3.1 8B** | ✅ Yes | 30 | ~14k tokens/min | Fast responses |
| **Ollama** | ✅ Local | ∞ | ∞ | Local dev, no quota |
| **Anthropic Claude** | ❌ Paid | — | — | Highest quality |
| **HF Inference** | ⚠️ PRO only | 5 | varies | Experimenting |

---

## Project Structure

```
Soci/
├── src/soci/
│   ├── world/          City map, simulation clock, world events
│   ├── agents/         Agent cognition: persona, memory, needs, relationships
│   ├── actions/        Movement, activities, conversation, social actions
│   ├── engine/         Simulation loop, scheduler, entropy, LLM clients
│   ├── persistence/    SQLite database, save/load snapshots
│   └── api/            FastAPI REST + WebSocket server
├── config/
│   ├── city.yaml       City layout, building positions, zones
│   └── personas.yaml   Named character definitions (20 hand-crafted agents)
├── web/
│   └── index.html      Single-file web UI (no framework)
├── Dockerfile          For HF Spaces / Render / Railway deployment
├── render.yaml         Render deployment config
└── main.py             Terminal runner (no UI)
```

---

## License

MIT
