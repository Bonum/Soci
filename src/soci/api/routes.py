"""REST API routes — city state, agents, history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PlayerActionRequest(BaseModel):
    action: str  # move, talk, work, eat, etc.
    target: str = ""
    detail: str = ""


class PlayerJoinRequest(BaseModel):
    name: str
    background: str = "A newcomer to Soci City."


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
    return {
        "id": agent.id,
        "name": agent.name,
        "age": agent.persona.age,
        "gender": agent.persona.gender,
        "occupation": agent.persona.occupation,
        "traits": agent.persona.trait_summary,
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
    from soci.api.server import _sim_paused, _sim_speed
    return {"paused": _sim_paused, "speed": _sim_speed}


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
    """Set simulation speed. 0.25=fast, 1.0=normal, 3.0=slow."""
    import soci.api.server as srv
    srv._sim_speed = max(0.1, min(5.0, multiplier))
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
