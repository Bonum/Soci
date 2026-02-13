"""Offline integration test — runs the full simulation loop with a mock LLM.

This test validates the entire pipeline without requiring an API key.
Run: python test_simulation.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent / "src"))

from soci.world.city import City
from soci.world.clock import SimClock
from soci.world.events import EventSystem
from soci.agents.persona import load_personas, Persona
from soci.agents.agent import Agent, AgentAction, AgentState
from soci.agents.memory import MemoryStream, MemoryType
from soci.agents.needs import NeedsState
from soci.agents.relationships import RelationshipGraph, Relationship
from soci.actions.registry import resolve_action, ActionType
from soci.actions.movement import execute_move, get_best_location_for_need
from soci.actions.activities import execute_activity
from soci.actions.social import should_initiate_conversation, pick_conversation_partner
from soci.engine.entropy import EntropyManager
from soci.engine.scheduler import prioritize_agents, should_skip_llm
from soci.engine.simulation import Simulation
from soci.persistence.database import Database


class MockLLM:
    """Mock LLM that returns plausible JSON responses without calling the API."""

    def __init__(self):
        self.usage = MagicMock()
        self.usage.total_calls = 0
        self.usage.total_input_tokens = 0
        self.usage.total_output_tokens = 0
        self.usage.estimated_cost_usd = 0.0
        self.usage.calls_by_model = {}
        self.usage.summary.return_value = "Mock LLM: 0 calls, $0.00"

    async def complete(self, system, user_message, model=None, temperature=0.7, max_tokens=1024):
        self.usage.total_calls += 1
        return "I'm thinking about my day."

    async def complete_json(self, system, user_message, model=None, temperature=0.7, max_tokens=1024):
        self.usage.total_calls += 1

        # Detect what kind of prompt this is and return appropriate mock data
        msg = user_message.lower()

        if "plan your day" in msg:
            return {
                "plan": [
                    "Wake up and have breakfast at home",
                    "Go to work at the office",
                    "Have lunch at the cafe",
                    "Continue working",
                    "Go to the park for a walk",
                    "Have dinner",
                    "Relax at home",
                ],
                "reasoning": "A balanced day with work and leisure."
            }

        if "what do you do next" in msg:
            actions = ["work", "eat", "relax", "wander", "move", "exercise"]
            action = random.choice(actions)
            targets = {
                "move": random.choice(["cafe", "park", "home_north", "office", "grocery"]),
                "work": "",
                "eat": "",
                "relax": "",
                "wander": "",
                "exercise": "",
            }
            details = {
                "move": "heading somewhere new",
                "work": "focusing on a project",
                "eat": "having a quick meal",
                "relax": "taking it easy",
                "wander": "strolling around",
                "exercise": "doing some stretches",
            }
            return {
                "action": action,
                "target": targets.get(action, ""),
                "detail": details.get(action, "doing something"),
                "duration": random.randint(1, 3),
                "reasoning": "Felt like it."
            }

        if "how important" in msg:
            return {
                "importance": random.randint(3, 8),
                "reaction": "Interesting, I'll remember that."
            }

        if "reflect" in msg:
            return {
                "reflections": [
                    "I notice I've been spending a lot of time at work lately.",
                    "The neighborhood feels alive today."
                ],
                "mood_shift": random.uniform(-0.1, 0.2),
                "reasoning": "Just thinking about things."
            }

        if "start a conversation" in msg or "you decide to start" in msg:
            return {
                "message": "Hey, how's it going?",
                "inner_thought": "I should catch up with them.",
                "topic": "daily life"
            }

        if "says:" in msg:
            return {
                "message": "Yeah, things are good. How about you?",
                "inner_thought": "Nice to chat.",
                "sentiment_delta": 0.05,
                "trust_delta": 0.02
            }

        return {"status": "ok"}


async def run_tests():
    print("=" * 60)
    print("SOCI — OFFLINE INTEGRATION TEST")
    print("=" * 60)
    errors = 0

    # --- Test 1: Clock ---
    print("\n[1/12] Clock system...")
    clock = SimClock(tick_minutes=15, hour=6, minute=0)
    for _ in range(96):  # Full day
        clock.tick()
    assert clock.day == 2, f"Expected day 2, got {clock.day}"
    assert clock.hour == 6, f"Expected hour 6, got {clock.hour}"
    clock_dict = clock.to_dict()
    restored_clock = SimClock.from_dict(clock_dict)
    assert restored_clock.day == clock.day
    print("  PASS: Clock ticks correctly for a full day, serialization works")

    # --- Test 2: City ---
    print("\n[2/12] City system...")
    city = City.from_yaml("config/city.yaml")
    assert len(city.locations) == 12
    # Test connectivity
    cafe = city.get_location("cafe")
    assert cafe is not None
    assert "park" in cafe.connected_to
    connected = city.get_connected("cafe")
    assert len(connected) > 0
    # Test agent placement and movement
    city.place_agent("test_agent", "cafe")
    assert "test_agent" in city.get_agents_at("cafe")
    city.move_agent("test_agent", "cafe", "park")
    assert "test_agent" not in city.get_agents_at("cafe")
    assert "test_agent" in city.get_agents_at("park")
    assert city.find_agent("test_agent") == "park"
    city.locations["park"].remove_occupant("test_agent")
    print("  PASS: City loads, connections work, movement works")

    # --- Test 3: Personas ---
    print("\n[3/12] Persona system...")
    personas = load_personas("config/personas.yaml")
    assert len(personas) == 20
    # Check diversity
    ages = [p.age for p in personas]
    assert min(ages) <= 20, "Should have young people"
    assert max(ages) >= 60, "Should have older people"
    occupations = set(p.occupation for p in personas)
    assert len(occupations) >= 15, "Should have diverse occupations"
    # Test system prompt
    prompt = personas[0].system_prompt()
    assert personas[0].name in prompt
    assert "personality" in prompt.lower() or "PERSONALITY" in prompt
    print(f"  PASS: 20 personas loaded, ages {min(ages)}-{max(ages)}, {len(occupations)} occupations")

    # --- Test 4: Needs ---
    print("\n[4/12] Needs system...")
    needs = NeedsState()
    initial_hunger = needs.hunger
    for _ in range(20):
        needs.tick()
    assert needs.hunger < initial_hunger, "Hunger should decay"
    assert needs.energy < 1.0, "Energy should decay"
    needs.satisfy("hunger", 0.5)
    assert needs.hunger > 0.0, "Hunger should be partially satisfied"
    urgent = needs.urgent_needs
    desc = needs.describe()
    assert isinstance(desc, str)
    print(f"  PASS: Needs decay ({desc}), satisfaction works")

    # --- Test 5: Memory ---
    print("\n[5/12] Memory system...")
    mem = MemoryStream()
    for i in range(30):
        mem.add(i, 1, f"{6+i//4:02d}:{(i%4)*15:02d}",
                MemoryType.OBSERVATION, f"Event {i}", importance=random.randint(1, 10))
    assert len(mem.memories) == 30
    retrieved = mem.retrieve(30, top_k=5)
    assert len(retrieved) == 5
    recent = mem.get_recent(3)
    assert len(recent) == 3
    assert recent[-1].content == "Event 29"
    # Test reflection trigger
    mem._importance_accumulator = 100
    assert mem.should_reflect()
    mem.reset_reflection_accumulator()
    assert not mem.should_reflect()
    # Test serialization
    mem_dict = mem.to_dict()
    restored_mem = MemoryStream.from_dict(mem_dict)
    assert len(restored_mem.memories) == 30
    print("  PASS: Memory storage, retrieval, reflection trigger, serialization")

    # --- Test 6: Relationships ---
    print("\n[6/12] Relationship system...")
    graph = RelationshipGraph()
    rel = graph.get_or_create("elena", "Elena Vasquez")
    assert rel.familiarity == 0.0
    rel.update_after_interaction(tick=10, sentiment_delta=0.1, trust_delta=0.05, note="Had coffee together")
    assert rel.familiarity > 0.0
    assert rel.sentiment > 0.5
    assert len(rel.notes) == 1
    closest = graph.get_closest(5)
    assert len(closest) == 1
    desc = rel.describe()
    assert "Elena" in desc
    # Serialization
    g_dict = graph.to_dict()
    restored_g = RelationshipGraph.from_dict(g_dict)
    assert restored_g.get("elena") is not None
    print("  PASS: Relationships form, track sentiment/trust, serialize")

    # --- Test 7: Agent ---
    print("\n[7/12] Agent system...")
    persona = personas[0]  # Elena
    agent = Agent(persona)
    assert agent.name == "Elena Vasquez"
    assert agent.location == "home_north"
    assert agent.state == AgentState.IDLE
    # Test action
    action = AgentAction(type="work", detail="coding", duration_ticks=3, needs_satisfied={"purpose": 0.3})
    agent.start_action(action)
    assert agent.is_busy
    assert agent.state == AgentState.WORKING
    for _ in range(3):
        agent.tick_action()
    assert not agent.is_busy
    assert agent.state == AgentState.IDLE
    # Test mood + needs interaction
    for _ in range(10):
        agent.tick_needs()
    # Test observation
    agent.add_observation(0, 1, "06:00", "Saw a cat in the park", importance=4)
    assert len(agent.memory.memories) == 1
    # Serialization
    a_dict = agent.to_dict()
    restored_a = Agent.from_dict(a_dict)
    assert restored_a.name == agent.name
    assert len(restored_a.memory.memories) == 1
    print("  PASS: Agent actions, needs, mood, memory, serialization")

    # --- Test 8: Action resolution ---
    print("\n[8/12] Action resolution...")
    city2 = City.from_yaml("config/city.yaml")
    agent2 = Agent(personas[0])
    city2.place_agent(agent2.id, agent2.location)
    raw = {"action": "move", "target": "cafe", "detail": "heading to cafe", "duration": 1}
    resolved = resolve_action(raw, agent2, city2)
    assert resolved.type == "move"
    assert resolved.target == "cafe"
    # Invalid action falls back to wander
    raw_bad = {"action": "fly", "target": "moon"}
    resolved_bad = resolve_action(raw_bad, agent2, city2)
    assert resolved_bad.type == "wander"
    print("  PASS: Valid actions resolve, invalid actions fall back to wander")

    # --- Test 9: Movement ---
    print("\n[9/12] Movement system...")
    clock2 = SimClock()
    agent3 = Agent(personas[0])
    city3 = City.from_yaml("config/city.yaml")
    city3.place_agent(agent3.id, "home_north")
    move_action = AgentAction(type="move", target="cafe", detail="walking to cafe")
    desc = execute_move(agent3, move_action, city3, clock2)
    assert "cafe" in desc.lower() or "Daily Grind" in desc
    assert agent3.location == "cafe"
    # Test location suggestion
    suggested = get_best_location_for_need(agent3, "hunger", city3)
    assert suggested is not None
    print(f"  PASS: Movement works, need-based suggestion: {suggested}")

    # --- Test 10: Events & Entropy ---
    print("\n[10/12] Events and entropy...")
    events = EventSystem(event_chance_per_tick=1.0)  # Force events
    new = events.tick(["cafe", "park", "office"])
    assert len(events.active_events) > 0 or len(new) > 0
    world_desc = events.get_world_description()
    assert "Weather" in world_desc
    entropy = EntropyManager()
    agents_list = [Agent(p) for p in personas[:5]]
    # Simulate repetitive behavior
    entropy._action_history["elena"] = ["work"] * 15
    assert entropy._is_stuck_in_loop("elena")
    conflicts = entropy.get_conflict_catalysts(agents_list)
    print(f"  PASS: Events fire, entropy detects loops, {len(conflicts)} potential conflicts found")

    # --- Test 11: Full simulation loop (mock LLM) ---
    print("\n[11/12] Full simulation loop (mock LLM)...")
    mock_llm = MockLLM()
    city4 = City.from_yaml("config/city.yaml")
    clock4 = SimClock(tick_minutes=15, hour=6, minute=0)
    sim = Simulation(city=city4, clock=clock4, llm=mock_llm)
    sim.load_agents_from_yaml("config/personas.yaml")

    # Limit to 5 agents for speed
    agent_ids = list(sim.agents.keys())[:5]
    sim.agents = {aid: sim.agents[aid] for aid in agent_ids}

    events_collected = []
    sim.on_event = lambda msg: events_collected.append(msg)

    # Run 10 ticks
    for _ in range(10):
        await sim.tick()

    assert sim.clock.total_ticks == 10
    assert len(events_collected) > 0
    print(f"  PASS: 10 ticks completed, {len(events_collected)} events, "
          f"{mock_llm.usage.total_calls} LLM calls")

    # Check agents moved, have memories, etc.
    for aid, agent in sim.agents.items():
        assert len(agent.memory.memories) > 0, f"{agent.name} should have memories"

    # --- Test 12: State serialization roundtrip ---
    print("\n[12/12] Full state serialization...")
    state = sim.to_dict()
    state_json = json.dumps(state)
    assert len(state_json) > 1000, "State should be substantial"
    restored_state = json.loads(state_json)
    sim2 = Simulation.from_dict(restored_state, mock_llm)
    assert len(sim2.agents) == len(sim.agents)
    assert sim2.clock.total_ticks == sim.clock.total_ticks
    for aid in sim.agents:
        assert aid in sim2.agents
        assert sim2.agents[aid].name == sim.agents[aid].name
    print(f"  PASS: Full state serialized ({len(state_json):,} bytes) and restored")

    # --- Summary ---
    print("\n" + "=" * 60)
    if errors == 0:
        print("ALL 12 TESTS PASSED")
    else:
        print(f"{errors} TEST(S) FAILED")
    print("=" * 60)

    # Print some interesting stats
    print(f"\nSimulation state:")
    print(f"  Clock: {sim.clock.datetime_str}")
    print(f"  Weather: {sim.events.weather.value}")
    print(f"  Mock LLM calls: {mock_llm.usage.total_calls}")
    print(f"\nAgent status after 10 ticks:")
    for aid, agent in sim.agents.items():
        loc = sim.city.get_location(agent.location)
        loc_name = loc.name if loc else agent.location
        print(f"  {agent.name}: {agent.state.value} at {loc_name} "
              f"(mood={agent.mood:.2f}, memories={len(agent.memory.memories)})")

    return errors == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
