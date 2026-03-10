"""REST API routes — city state, agents, history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class PlayerActionRequest(BaseModel):
    action: str  # move, talk, work, eat, etc.
    target: str = ""
    detail: str = ""


class PlayerJoinRequest(BaseModel):
    name: str
    background: str = "A newcomer to Soci City."


# ── Auth models ───────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str


class PlayerCreateRequest(BaseModel):
    token: str
    name: str
    age: int = 30
    occupation: str = "Newcomer"
    background: str = "A newcomer to Soci City."
    gender: str = "unknown"
    extraversion: int = 5
    agreeableness: int = 7
    openness: int = 6


class PlayerUpdateRequest(BaseModel):
    token: str
    name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    background: Optional[str] = None
    gender: Optional[str] = None
    extraversion: Optional[int] = None
    agreeableness: Optional[int] = None
    openness: Optional[int] = None


class PlayerTalkRequest(BaseModel):
    token: str
    target_id: str
    message: str


class PlayerMoveRequest(BaseModel):
    token: str
    location: str


class PlayerPlanRequest(BaseModel):
    token: str
    plan_item: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _find_player_apartment(sim) -> str:
    """Find an available residential location for a new player."""
    preferred = [
        "apartment_block_1", "apartment_block_2", "apartment_block_3",
        "apt_northeast", "apt_northwest", "apt_southeast", "apt_southwest",
    ]
    for loc_id in preferred:
        loc = sim.city.get_location(loc_id)
        if loc and not loc.is_full:
            return loc_id
    for loc in sim.city.get_locations_in_zone("residential"):
        if not loc.is_full:
            return loc.id
    return "town_square"


async def _get_player_from_token(token: str):
    """Validate token and return (user, agent). Raises 401/404 on failure."""
    from soci.api.server import get_simulation, get_database
    db = get_database()
    sim = get_simulation()
    user = await db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    agent_id = user.get("agent_id")
    agent = sim.agents.get(agent_id) if agent_id else None
    return user, agent


@router.get("/city")
async def get_city():
    """Get the full city state — locations, agents, time, weather."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    return sim.get_state_summary()


@router.get("/city/locations")
async def get_locations():
    """Get all city locations and who's there."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    return {
        lid: {
            "name": loc.name,
            "zone": loc.zone,
            "description": loc.description,
            "occupants": [
                {"id": aid, "name": sim.agents[aid].name, "state": sim.agents[aid].state.value}
                for aid in loc.occupants if aid in sim.agents
            ],
            "connected_to": loc.connected_to,
        }
        for lid, loc in sim.city.locations.items()
    }


@router.get("/agents")
async def get_agents():
    """Get summary of all agents."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    return {
        aid: {
            "name": a.name,
            "age": a.persona.age,
            "gender": a.persona.gender,
            "occupation": a.persona.occupation,
            "location": a.location,
            "state": a.state.value,
            "mood": round(a.mood, 2),
            "action": a.current_action.detail if a.current_action else "idle",
            "partner_id": a.partner_id,
            "is_player": a.is_player,
        }
        for aid, a in sim.agents.items()
    }


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get detailed info about a specific agent."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    agent = sim.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    loc = sim.city.get_location(agent.location)
    p = agent.persona
    return {
        "id": agent.id,
        "name": agent.name,
        "age": p.age,
        "gender": p.gender,
        "occupation": p.occupation,
        "background": p.background,
        "traits": p.trait_summary,
        "personality": {
            "openness": getattr(p, "openness", 5),
            "conscientiousness": getattr(p, "conscientiousness", 5),
            "extraversion": getattr(p, "extraversion", 5),
            "agreeableness": getattr(p, "agreeableness", 5),
            "neuroticism": getattr(p, "neuroticism", 5),
        },
        "home_location": getattr(p, "home_location", ""),
        "work_location": getattr(p, "work_location", ""),
        "location": {"id": agent.location, "name": loc.name if loc else "unknown"},
        "state": agent.state.value,
        "mood": round(agent.mood, 2),
        "needs": agent.needs.to_dict(),
        "needs_description": agent.needs.describe(),
        "action": agent.current_action.detail if agent.current_action else "idle",
        "daily_plan": agent.daily_plan,
        "partner_id": agent.partner_id,
        "relationships": [
            {
                "agent_id": rel.agent_id,
                "name": rel.agent_name,
                "closeness": round(rel.closeness, 2),
                "romantic_interest": round(rel.romantic_interest, 2),
                "relationship_status": rel.relationship_status,
                "trust": round(rel.trust, 2),
                "sentiment": round(rel.sentiment, 2),
                "familiarity": round(rel.familiarity, 2),
                "interaction_count": rel.interaction_count,
                "description": rel.describe(),
            }
            for rel in agent.relationships.get_closest(10)
        ],
        "recent_memories": [
            {
                "time": f"Day {m.day} {m.time_str}",
                "type": m.type.value,
                "content": m.content,
                "importance": m.importance,
            }
            for m in agent.memory.get_recent(10)
        ],
        "life_events": agent.life_events[-20:],
        "goals": agent.goals,
        "pregnant": agent.pregnant,
        "children": agent.children,
    }


@router.get("/agents/{agent_id}/memories")
async def get_agent_memories(agent_id: str, limit: int = 20):
    """Get an agent's memory stream."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    agent = sim.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return [
        {
            "id": m.id,
            "time": f"Day {m.day} {m.time_str}",
            "type": m.type.value,
            "content": m.content,
            "importance": m.importance,
            "involved_agents": m.involved_agents,
        }
        for m in agent.memory.memories[-limit:]
    ]


@router.get("/conversations")
async def get_conversations(include_history: bool = True, limit: int = 20):
    """Get active and recent conversations with full dialogue."""
    from soci.api.server import get_simulation
    sim = get_simulation()

    def format_conv(conv_data, active=False):
        """Format a conversation (dict or Conversation object)."""
        if hasattr(conv_data, 'to_dict'):
            d = conv_data.to_dict()
        else:
            d = conv_data
        participant_names = [
            sim.agents[p].name for p in d.get("participants", []) if p in sim.agents
        ]
        return {
            "id": d.get("id", ""),
            "participants": d.get("participants", []),
            "participant_names": participant_names,
            "topic": d.get("topic", ""),
            "location": d.get("location", ""),
            "turns": d.get("turns", []),
            "is_active": active,
        }

    result = {
        "active": [
            format_conv(c, active=True)
            for c in sim.active_conversations.values()
        ],
        "recent": [],
    }
    if include_history:
        result["recent"] = [
            format_conv(c) for c in sim.conversation_history[-limit:]
        ][::-1]  # Most recent first
    return result


class SwitchProviderRequest(BaseModel):
    provider: str
    model: Optional[str] = None


@router.get("/llm/providers")
async def get_llm_providers():
    """Return available LLM providers (those with API keys set) and the current one."""
    import os
    from soci.api.server import get_llm_provider, get_simulation
    current = get_llm_provider()
    current_model = getattr(get_simulation().llm, "default_model", "")
    providers = []
    # NN is always available — local ONNX model, no API key needed
    providers.append({"id": "nn",     "label": "Soci Agent NN",  "icon": "🧠", "model": ""})
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append({"id": "claude",  "label": "Claude Haiku",        "icon": "◆", "model": ""})
    if os.environ.get("GROQ_API_KEY"):
        providers.append({"id": "groq",    "label": "Groq Llama 8B",       "icon": "⚡", "model": ""})
    if os.environ.get("GEMINI_API_KEY"):
        providers.append({"id": "gemini",  "label": "Gemini 2.0 Flash Lite", "icon": "✦", "model": ""})
    providers.append({"id": "ollama", "label": "Ollama (local LLM)",     "icon": "🦙", "model": ""})
    return {"current": current, "current_model": current_model, "providers": providers}


@router.get("/llm/test")
async def test_llm():
    """Make a minimal LLM call and return the raw response — for diagnosing provider issues."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    try:
        raw = await sim.llm.complete(
            system="You are a test assistant.",
            user_message='Reply with exactly: {"ok": true}',
            max_tokens=32,
        )
        error_detail = getattr(sim.llm, "_auth_error", "") or getattr(sim.llm, "_last_error", "")
        return {"ok": bool(raw), "raw": raw,
                "provider": getattr(sim.llm, "provider", "?"),
                "model": getattr(sim.llm, "default_model", "?"),
                "error": error_detail}
    except Exception as e:
        return {"ok": False, "raw": "", "error": str(e)}


@router.post("/llm/provider")
async def set_llm_provider(req: SwitchProviderRequest):
    """Hot-swap the active LLM provider."""
    from soci.api.server import switch_llm_provider
    valid = {"claude", "groq", "gemini", "nn", "ollama"}
    if req.provider not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{req.provider}'")
    try:
        await switch_llm_provider(req.provider, model=req.model or None)
        return {"ok": True, "provider": req.provider, "model": req.model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/quota")
async def get_llm_quota():
    """Return remaining daily quota and usage stats for budget planning.

    Supports Gemini (tracked internally) and Groq (estimated from RPM × hours remaining).
    """
    import os
    from soci.api.server import get_simulation, get_llm_provider, _sim_speed
    sim = get_simulation()
    llm = sim.llm
    provider = get_llm_provider()

    # Per-provider quota info
    # Gemini: has internal daily tracking (_daily_limit, _daily_requests)
    # Groq: 30 RPM free tier, ~14,400 RPD; no internal daily counter
    daily_limit = getattr(llm, "_daily_limit", 0)
    daily_requests = getattr(llm, "_daily_requests", 0)

    # Build quota for all rate-limited providers (even if not currently active)
    providers_quota = {}

    # Gemini quota
    if os.environ.get("GEMINI_API_KEY"):
        g_limit = daily_limit if provider == "gemini" else 1500
        g_used = daily_requests if provider == "gemini" else 0
        providers_quota["gemini"] = {
            "daily_limit": g_limit,
            "daily_requests": g_used,
            "remaining": max(0, g_limit - g_used),
        }

    # Groq quota (estimated: 30 RPM × 60 min × 24h = 14,400 RPD free tier)
    if os.environ.get("GROQ_API_KEY"):
        groq_limit = int(os.environ.get("GROQ_DAILY_LIMIT", "14400"))
        groq_used = getattr(llm, "_daily_requests", 0) if provider == "groq" else 0
        providers_quota["groq"] = {
            "daily_limit": groq_limit,
            "daily_requests": groq_used,
            "remaining": max(0, groq_limit - groq_used),
        }

    # Current provider's quota (for backward compat with nn_selfimprove)
    cur = providers_quota.get(provider, {"daily_limit": 0, "daily_requests": 0, "remaining": 0})

    # Estimate ticks per hour — rate-limited providers use 4s tick delay
    tick_delay_current = 4.0 if provider in ("gemini", "groq") else 2.0
    ticks_per_hour = 3600.0 / (tick_delay_current * max(_sim_speed, 0.01))
    max_calls_per_tick = 2 if provider in ("gemini", "groq") else 5
    num_agents = len(sim.agents)

    # Per-provider rate info (RPM is the real bottleneck, not probability)
    # Gemini: 4 RPM hard limit → max 240 calls/hour
    # Groq: 28 RPM hard limit → max 1680 calls/hour
    provider_rpm = {"gemini": 4, "groq": 28}
    for pid in providers_quota:
        rpm = provider_rpm.get(pid, 30)
        providers_quota[pid]["rpm"] = rpm
        providers_quota[pid]["max_calls_per_hour"] = rpm * 60

    # Expose rate-limit status (detects actual exhaustion from 429 errors)
    llm_status = getattr(llm, "llm_status", "active")
    if provider in providers_quota:
        providers_quota[provider]["status"] = llm_status
        if llm_status == "limited":
            # Override remaining to 0 — the API is actually returning 429s
            providers_quota[provider]["remaining"] = 0

    return {
        "provider": provider,
        "daily_limit": cur["daily_limit"],
        "daily_requests": cur["daily_requests"],
        "remaining": cur["remaining"],
        "providers": providers_quota,
        "ticks_per_hour": round(ticks_per_hour, 1),
        "max_calls_per_tick": max_calls_per_tick,
        "num_agents": num_agents,
        "sim_speed": _sim_speed,
    }


@router.get("/stats")
async def get_stats():
    """Get simulation statistics and LLM usage."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    return {
        "clock": sim.clock.to_dict(),
        "total_agents": len(sim.agents),
        "active_conversations": len(sim.active_conversations),
        "llm_usage": {
            "total_calls": sim.llm.usage.total_calls,
            "total_input_tokens": sim.llm.usage.total_input_tokens,
            "total_output_tokens": sim.llm.usage.total_output_tokens,
            "estimated_cost_usd": round(sim.llm.usage.estimated_cost_usd, 4),
            "calls_by_model": sim.llm.usage.calls_by_model,
        },
    }


# ── Auth endpoints ────────────────────────────────────────────────────────────

@router.post("/auth/register")
async def auth_register(request: AuthRequest):
    """Register a new user and auto-create their player agent."""
    from soci.api.server import get_simulation, get_database
    from soci.agents.agent import Agent
    from soci.agents.persona import Persona
    db = get_database()
    sim = get_simulation()

    if not request.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if len(request.password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    try:
        user = await db.create_user(request.username.strip(), request.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Auto-create default player agent with an apartment
    safe_name = request.username.strip()
    player_id = f"player_{safe_name.lower().replace(' ', '_')}"
    # Ensure unique ID
    suffix = 1
    base_id = player_id
    while player_id in sim.agents:
        player_id = f"{base_id}_{suffix}"
        suffix += 1

    home = _find_player_apartment(sim)
    persona = Persona(
        id=player_id,
        name=safe_name,
        age=30,
        occupation="Newcomer",
        gender="unknown",
        background="A newcomer to Soci City.",
        home_location=home,
        work_location="",
        extraversion=5,
        agreeableness=7,
        openness=6,
    )
    agent = Agent(persona)
    agent.is_player = True
    sim.add_agent(agent)
    await db.set_user_agent(safe_name, player_id)
    user["agent_id"] = player_id

    return user


@router.post("/auth/login")
async def auth_login(request: AuthRequest):
    """Login and return session token."""
    from soci.api.server import get_database
    db = get_database()
    user = await db.authenticate_user(request.username.strip(), request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return user


@router.get("/auth/me")
async def auth_me(authorization: str = Header(default="")):
    """Verify session token and return current user info."""
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="No token provided")
    from soci.api.server import get_database
    db = get_database()
    user = await db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


@router.post("/auth/logout")
async def auth_logout(authorization: str = Header(default="")):
    """Invalidate session token."""
    token = authorization.removeprefix("Bearer ").strip()
    from soci.api.server import get_database
    db = get_database()
    await db.logout_user(token)
    return {"ok": True}


# ── Player management endpoints ───────────────────────────────────────────────

@router.post("/player/create")
async def player_create(request: PlayerCreateRequest):
    """Create or replace the player's agent."""
    from soci.api.server import get_simulation, get_database
    from soci.agents.agent import Agent
    from soci.agents.persona import Persona
    db = get_database()
    sim = get_simulation()
    user = await db.get_user_by_token(request.token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    player_id = f"player_{request.name.lower().replace(' ', '_')}"
    suffix = 1
    base_id = player_id
    while player_id in sim.agents and sim.agents[player_id].persona.name != request.name:
        player_id = f"{base_id}_{suffix}"
        suffix += 1

    # Remove old agent if exists
    old_id = user.get("agent_id")
    if old_id and old_id in sim.agents:
        loc = sim.city.get_location(sim.agents[old_id].location)
        if loc:
            loc.remove_agent(old_id)
        del sim.agents[old_id]

    home = _find_player_apartment(sim)
    persona = Persona(
        id=player_id,
        name=request.name,
        age=request.age,
        occupation=request.occupation,
        gender=request.gender,
        background=request.background,
        home_location=home,
        work_location="",
        extraversion=request.extraversion,
        agreeableness=request.agreeableness,
        openness=request.openness,
    )
    agent = Agent(persona)
    agent.is_player = True
    sim.add_agent(agent)
    await db.set_user_agent(user["username"], player_id)
    return {"agent_id": player_id}


@router.put("/player/update")
async def player_update(request: PlayerUpdateRequest):
    """Update the player's persona fields in-place."""
    user, agent = await _get_player_from_token(request.token)
    if not agent:
        raise HTTPException(status_code=404, detail="No player agent found")
    p = agent.persona
    if request.name is not None:
        p.name = request.name
        agent.name = request.name
    if request.age is not None:
        p.age = request.age
    if request.occupation is not None:
        p.occupation = request.occupation
    if request.background is not None:
        p.background = request.background
    if request.gender is not None:
        p.gender = request.gender
    if request.extraversion is not None:
        p.extraversion = max(1, min(10, request.extraversion))
    if request.agreeableness is not None:
        p.agreeableness = max(1, min(10, request.agreeableness))
    if request.openness is not None:
        p.openness = max(1, min(10, request.openness))
    return {"ok": True}


@router.post("/player/move")
async def player_move(request: PlayerMoveRequest):
    """Move the player to a city location."""
    from soci.api.server import get_simulation
    from soci.agents.agent import AgentAction, AgentState
    sim = get_simulation()
    user, agent = await _get_player_from_token(request.token)
    if not agent:
        raise HTTPException(status_code=404, detail="No player agent found")
    loc = sim.city.get_location(request.location)
    if not loc:
        raise HTTPException(status_code=400, detail=f"Unknown location: {request.location}")
    action = AgentAction(type="move", target=request.location, detail=f"Walking to {loc.name}", duration_ticks=1)
    await sim._execute_action(agent, action)
    return {"ok": True, "location": agent.location}


@router.post("/player/talk")
async def player_talk(request: PlayerTalkRequest):
    """Send a message to an NPC and get their LLM-generated reply."""
    from soci.api.server import get_simulation
    from soci.actions.conversation import Conversation, ConversationTurn, continue_conversation
    import uuid
    sim = get_simulation()
    user, player = await _get_player_from_token(request.token)
    if not player:
        raise HTTPException(status_code=404, detail="No player agent found")
    target = sim.agents.get(request.target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target agent not found")

    # Build a conversation with the player's message as the opening turn
    conv_id = f"player_conv_{uuid.uuid4().hex[:8]}"
    conv = Conversation(
        id=conv_id,
        location=player.location,
        participants=[player.id, target.id],
        topic="player interaction",
        max_turns=20,
    )
    player_turn = ConversationTurn(
        speaker_id=player.id,
        speaker_name=player.name,
        message=request.message,
        tick=sim.clock.total_ticks,
    )
    conv.add_turn(player_turn)

    # Let the NPC respond via LLM
    reply_turn = await continue_conversation(conv, target, player, sim.llm, sim.clock)

    # Update relationship
    target.relationships.get_or_create(player.id, player.name)
    player.relationships.get_or_create(target.id, target.name)

    # Add memory to both
    player.add_observation(
        tick=sim.clock.total_ticks, day=sim.clock.day, time_str=sim.clock.time_str,
        content=f"I said to {target.name}: \"{request.message}\"",
        importance=4, involved_agents=[target.id],
    )
    if reply_turn and reply_turn.speaker_id == target.id:
        target.add_observation(
            tick=sim.clock.total_ticks, day=sim.clock.day, time_str=sim.clock.time_str,
            content=f"{player.name} talked to me: \"{request.message}\"",
            importance=4, involved_agents=[player.id],
        )
        return {"reply": reply_turn.message, "inner_thought": reply_turn.inner_thought}

    return {"reply": "(no response)", "inner_thought": ""}


@router.post("/player/plan")
async def player_add_plan(request: PlayerPlanRequest):
    """Append an item to the player's daily plan."""
    user, agent = await _get_player_from_token(request.token)
    if not agent:
        raise HTTPException(status_code=404, detail="No player agent found")
    agent.daily_plan.append(request.plan_item)
    return {"ok": True, "daily_plan": agent.daily_plan}


# ── Legacy join/action (kept for compatibility) ───────────────────────────────

@router.post("/player/join")
async def player_join(request: PlayerJoinRequest):
    """Register a human player as a new agent in the simulation."""
    from soci.agents.agent import Agent
    from soci.agents.persona import Persona
    from soci.api.server import get_simulation
    sim = get_simulation()

    player_id = f"player_{request.name.lower().replace(' ', '_')}"
    if player_id in sim.agents:
        raise HTTPException(status_code=400, detail="Player already exists")

    persona = Persona(
        id=player_id,
        name=request.name,
        age=25,
        occupation="newcomer",
        background=request.background,
        home_location="house_elena",
        work_location="",
    )
    agent = Agent(persona)
    agent.is_player = True
    sim.add_agent(agent)

    return {"id": player_id, "message": f"Welcome to Soci City, {request.name}!"}


@router.post("/player/{player_id}/action")
async def player_action(player_id: str, request: PlayerActionRequest):
    """Submit an action for a human player."""
    from soci.agents.agent import AgentAction
    from soci.actions.registry import resolve_action
    from soci.api.server import get_simulation
    sim = get_simulation()

    agent = sim.agents.get(player_id)
    if not agent or not agent.is_player:
        raise HTTPException(status_code=404, detail="Player not found")

    if agent.is_busy:
        return {"status": "busy", "message": f"You're currently {agent.current_action.detail}"}

    action = resolve_action(
        {"action": request.action, "target": request.target, "detail": request.detail},
        agent,
        sim.city,
    )
    await sim._execute_action(agent, action)

    return {
        "status": "ok",
        "action": action.to_dict(),
        "location": agent.location,
    }


@router.get("/relationships")
async def get_relationships():
    """Get the full relationship graph — all agent-to-agent connections."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    edges = []
    seen = set()
    for aid, agent in sim.agents.items():
        for rel in agent.relationships.get_closest(20):
            pair = tuple(sorted([aid, rel.agent_id]))
            if pair in seen:
                continue
            seen.add(pair)
            other_rel = None
            other = sim.agents.get(rel.agent_id)
            if other:
                other_rel = other.relationships.get(aid)
            edges.append({
                "source": aid,
                "target": rel.agent_id,
                "source_name": agent.name,
                "target_name": rel.agent_name,
                "familiarity": round(rel.familiarity, 2),
                "trust": round(rel.trust, 2),
                "sentiment": round(rel.sentiment, 2),
                "romantic_interest": round(rel.romantic_interest, 2),
                "relationship_status": rel.relationship_status,
                "mutual_romantic": round(other_rel.romantic_interest, 2) if other_rel else 0,
                "interaction_count": rel.interaction_count,
            })
    return {"edges": edges}


@router.get("/events")
async def get_events(limit: int = 50):
    """Get recent simulation events for the event log."""
    from soci.api.server import get_simulation
    sim = get_simulation()
    events = sim._event_history[-limit:]
    return {"events": events}


@router.get("/controls")
async def get_controls():
    """Get current simulation control state."""
    from soci.api.server import _sim_paused, _sim_speed, _llm_call_probability
    return {"paused": _sim_paused, "speed": _sim_speed, "llm_call_probability": _llm_call_probability}


@router.post("/controls/llm_probability")
async def set_llm_probability(value: float = 0.10):
    """Set LLM call probability (0.0–1.0). Controls how often agents use LLM vs. routine behaviour."""
    from soci.api.server import set_llm_call_probability
    set_llm_call_probability(value)
    from soci.api.server import _llm_call_probability
    return {"llm_call_probability": _llm_call_probability}


@router.post("/controls/pause")
async def pause_simulation():
    """Pause the simulation."""
    import soci.api.server as srv
    srv._sim_paused = True
    return {"paused": True}


@router.post("/controls/resume")
async def resume_simulation():
    """Resume the simulation."""
    import soci.api.server as srv
    srv._sim_paused = False
    return {"paused": False}


@router.post("/controls/speed")
async def set_speed(multiplier: float = 1.0):
    """Set simulation speed. 0.02=50x, 0.1=10x, 0.2=5x, 0.5=2x, 1.0=normal, 3.0=slow."""
    import soci.api.server as srv
    srv._sim_speed = max(0.01, min(5.0, multiplier))
    return {"speed": srv._sim_speed}


@router.post("/save")
async def save_state(name: str = "manual_save"):
    """Manually save the simulation state."""
    from soci.api.server import get_simulation, get_database
    sim = get_simulation()
    db = get_database()
    from soci.persistence.snapshots import save_simulation
    await save_simulation(sim, db, name)
    return {"status": "saved", "name": name, "tick": sim.clock.total_ticks}
