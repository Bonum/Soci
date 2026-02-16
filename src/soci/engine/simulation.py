"""Simulation — the main loop that orchestrates the entire city simulation."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, Optional

from soci.agents.agent import Agent, AgentAction
from soci.agents.memory import MemoryType
from soci.agents.persona import Persona, load_personas
from soci.actions.registry import resolve_action, ACTION_NEEDS, ACTION_DURATIONS
from soci.actions.movement import execute_move
from soci.actions.activities import execute_activity
from soci.actions.conversation import (
    Conversation, initiate_conversation, continue_conversation,
)
from soci.actions.social import should_initiate_conversation, pick_conversation_partner
from soci.engine.llm import (
    ClaudeClient, MODEL_SONNET, MODEL_HAIKU,
    PLAN_DAY_PROMPT, DECIDE_ACTION_PROMPT, OBSERVE_PROMPT, REFLECT_PROMPT,
)
from soci.engine.scheduler import prioritize_agents, batch_llm_calls, should_skip_llm
from soci.engine.entropy import EntropyManager
from soci.world.city import City
from soci.world.clock import SimClock
from soci.world.events import EventSystem

logger = logging.getLogger(__name__)


class Simulation:
    """The main simulation engine — manages the city, agents, and time."""

    def __init__(
        self,
        city: City,
        clock: SimClock,
        llm: ClaudeClient,
        max_concurrent_llm: int = 10,
    ) -> None:
        self.city = city
        self.clock = clock
        self.llm = llm
        self.agents: dict[str, Agent] = {}
        self.events = EventSystem()
        self.entropy = EntropyManager()
        self.active_conversations: dict[str, Conversation] = {}
        self.conversation_history: list[dict] = []  # Finished conversations for API
        self._max_conversation_history: int = 50
        self._conversation_counter: int = 0
        self._max_concurrent = max_concurrent_llm
        self._tick_log: list[str] = []  # Log of events this tick
        self._event_history: list[dict] = []  # Persistent event log for API
        self._max_event_history: int = 200
        # Callback for real-time output
        self.on_event: Optional[Callable[[str], None]] = None

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the simulation and place them in the city."""
        self.agents[agent.id] = agent
        self.city.place_agent(agent.id, agent.location)

    def load_agents_from_yaml(self, path: str) -> None:
        """Load all personas from YAML and create agents."""
        personas = load_personas(path)
        for persona in personas:
            agent = Agent(persona)
            self.add_agent(agent)
        logger.info(f"Loaded {len(personas)} agents from {path}")

    def _emit(self, message: str) -> None:
        """Emit an event message."""
        self._tick_log.append(message)
        self._event_history.append({
            "tick": self.clock.total_ticks,
            "time": self.clock.datetime_str,
            "message": message,
        })
        if len(self._event_history) > self._max_event_history:
            self._event_history = self._event_history[-self._max_event_history:]
        if self.on_event:
            self.on_event(message)

    async def tick(self) -> list[str]:
        """Advance the simulation by one tick. Returns list of event descriptions."""
        self._tick_log = []
        self._emit(f"\n--- {self.clock.datetime_str} ({self.clock.time_of_day.value}) ---")

        # 1. Entropy management and world events
        entropy_messages = self.entropy.tick(
            list(self.agents.values()),
            self.events,
            self.clock,
            list(self.city.locations.keys()),
        )
        for msg in entropy_messages:
            self._emit(msg)

        # 2. New day — reset plans
        if self.clock.hour == 6 and self.clock.minute == 0:
            for agent in self.agents.values():
                agent.reset_daily_plan()

        # 3. Prioritize and process agents
        ordered_agents = prioritize_agents(list(self.agents.values()), self.clock)

        # 4. Generate daily plans for agents that need them
        plan_coros = []
        plan_agents = []
        for agent in ordered_agents:
            if agent.needs_new_plan(self.clock) and not should_skip_llm(agent, self.clock):
                plan_coros.append(self._generate_daily_plan(agent))
                plan_agents.append(agent)

        if plan_coros:
            await batch_llm_calls(plan_coros, self._max_concurrent)
            for agent in plan_agents:
                self._emit(f"[PLAN] {agent.name} planned their day: {'; '.join(agent.daily_plan[:3])}...")

        # 5. Process each agent — tick their needs, handle actions
        action_coros = []
        action_agents = []
        for agent in ordered_agents:
            # Tick needs
            is_sleeping = agent.state.value == "sleeping"
            agent.tick_needs(is_sleeping=is_sleeping)

            # Tick current action
            if agent.is_busy:
                completed = agent.tick_action()
                if completed and agent.current_action:
                    self._emit(f"  {agent.name} finished: {agent.current_action.detail}")
                continue

            # Skip LLM for sleeping agents during sleep hours, etc.
            if should_skip_llm(agent, self.clock):
                continue

            # Agent is idle — needs a new action
            action_coros.append(self._decide_action(agent))
            action_agents.append(agent)

        # Run all action decisions concurrently
        if action_coros:
            action_results = await batch_llm_calls(action_coros, self._max_concurrent)
            for agent, result in zip(action_agents, action_results):
                if result and isinstance(result, AgentAction):
                    await self._execute_action(agent, result)

        # 6. Handle active conversations
        conv_coros = []
        for conv_id, conv in list(self.active_conversations.items()):
            if conv.is_finished:
                self._finish_conversation(conv)
                del self.active_conversations[conv_id]
                continue
            # Determine who speaks next
            last_speaker = conv.turns[-1].speaker_id if conv.turns else None
            next_speaker_id = [p for p in conv.participants if p != last_speaker]
            if next_speaker_id:
                responder = self.agents.get(next_speaker_id[0])
                other = self.agents.get(last_speaker) if last_speaker else None
                if responder and other:
                    conv_coros.append(
                        continue_conversation(conv, responder, other, self.llm, self.clock)
                    )

        if conv_coros:
            await batch_llm_calls(conv_coros, self._max_concurrent)

        # 7. Social: maybe start new conversations
        await self._handle_social_interactions(ordered_agents)

        # 8. Reflections for agents with enough accumulated importance
        reflect_coros = []
        reflect_agents = []
        for agent in ordered_agents:
            if agent.memory.should_reflect() and not agent.is_player:
                reflect_coros.append(self._generate_reflection(agent))
                reflect_agents.append(agent)

        if reflect_coros:
            await batch_llm_calls(reflect_coros, self._max_concurrent)

        # 9. Romance — develop attractions and relationships
        self._tick_romance()

        # 10. Advance clock
        self.clock.tick()

        return self._tick_log

    async def _generate_daily_plan(self, agent: Agent) -> None:
        """Generate a daily plan for an agent via LLM."""
        world_desc = self.events.get_world_description()
        loc_desc = self.city.describe_location(agent.location, exclude_agent=agent.id)

        prompt = PLAN_DAY_PROMPT.format(
            time_str=self.clock.time_str,
            day=self.clock.day,
            context=agent.build_context(self.clock.total_ticks, world_desc, loc_desc),
        )

        result = await self.llm.complete_json(
            system=agent.persona.system_prompt(),
            user_message=prompt,
            model=MODEL_HAIKU,  # Plans are routine, use cheap model
            temperature=agent.persona.llm_temperature,
            max_tokens=512,
        )

        plan = result.get("plan", ["Go about my day"])
        if isinstance(plan, list):
            # Normalize: local LLMs sometimes return dicts or nested structures
            clean_plan = []
            for item in plan:
                if isinstance(item, str):
                    clean_plan.append(item)
                elif isinstance(item, dict):
                    # Try common keys: description, activity, task, text, name
                    for key in ("description", "activity", "task", "text", "name", "item"):
                        if key in item:
                            clean_plan.append(str(item[key]))
                            break
                    else:
                        clean_plan.append(str(item))
                else:
                    clean_plan.append(str(item))
            agent.set_daily_plan(clean_plan, self.clock.day, self.clock.total_ticks, self.clock.time_str)

    async def _decide_action(self, agent: Agent) -> Optional[AgentAction]:
        """Ask the LLM what action an agent should take next."""
        world_desc = self.events.get_world_description()
        loc_desc = self.city.describe_location(agent.location, exclude_agent=agent.id)

        # Get connected locations
        current_loc = self.city.get_location(agent.location)
        connected = []
        if current_loc:
            for cid in current_loc.connected_to:
                cloc = self.city.get_location(cid)
                if cloc:
                    connected.append(f"{cid} ({cloc.name})")

        # Get people at current location
        people_here = [
            self.agents[aid].name
            for aid in self.city.get_agents_at(agent.location)
            if aid != agent.id and aid in self.agents
        ]

        last_activity = "nothing in particular"
        if agent.current_action:
            last_activity = agent.current_action.detail or agent.current_action.type

        prompt = DECIDE_ACTION_PROMPT.format(
            time_str=self.clock.time_str,
            day=self.clock.day,
            context=agent.build_context(self.clock.total_ticks, world_desc, loc_desc),
            location_name=loc_desc,
            last_activity=last_activity,
            connected_locations=", ".join(connected) if connected else "none visible",
            people_here=", ".join(people_here) if people_here else "no one",
        )

        # Use Sonnet for novel situations, Haiku for routine
        model = MODEL_HAIKU
        if agent.needs.is_critical or self.events.active_events:
            model = MODEL_SONNET

        result = await self.llm.complete_json(
            system=agent.persona.system_prompt(),
            user_message=prompt,
            model=model,
            temperature=agent.persona.llm_temperature,
            max_tokens=512,
        )

        if not result:
            # Fallback: wander
            return AgentAction(type="wander", detail=f"{agent.name} wanders aimlessly")

        action = resolve_action(result, agent, self.city)
        agent._last_llm_tick = self.clock.total_ticks
        return action

    async def _execute_action(self, agent: Agent, action: AgentAction) -> None:
        """Execute an agent's chosen action."""
        if action.type == "move":
            desc = execute_move(agent, action, self.city, self.clock)
        elif action.type == "talk":
            # Talk action is handled via conversation system
            target_id = action.target
            if target_id and target_id in self.agents:
                await self._start_conversation(agent, self.agents[target_id])
                desc = f"{agent.name} starts talking to {self.agents[target_id].name}."
            else:
                desc = f"{agent.name} looks around for someone to talk to."
        else:
            desc = execute_activity(agent, action, self.city, self.clock)

        agent.start_action(action)
        self._emit(f"  {desc}")

        # Record observation
        agent.add_observation(
            tick=self.clock.total_ticks,
            day=self.clock.day,
            time_str=self.clock.time_str,
            content=desc,
            importance=3,
        )

    async def _handle_social_interactions(self, agents: list[Agent]) -> None:
        """Check if any idle co-located agents should start conversations."""
        # Don't flood with conversations — scale with population
        max_convos = max(3, len(self.agents) // 5)
        if len(self.active_conversations) >= max_convos:
            return

        for agent in agents:
            if agent.is_busy or agent.is_player:
                continue
            # Check if already in a conversation
            in_conv = any(
                agent.id in c.participants
                for c in self.active_conversations.values()
            )
            if in_conv:
                continue

            others = [
                aid for aid in self.city.get_agents_at(agent.location)
                if aid != agent.id
                and aid in self.agents
                and not self.agents[aid].is_busy
                and not any(aid in c.participants for c in self.active_conversations.values())
            ]
            if not others:
                continue

            partner_id = pick_conversation_partner(agent, others, self.clock)
            if partner_id and should_initiate_conversation(agent, partner_id, self.clock):
                await self._start_conversation(agent, self.agents[partner_id])
                break  # One new conversation per tick max

    async def _start_conversation(self, initiator: Agent, target: Agent) -> None:
        """Start a conversation between two agents."""
        self._conversation_counter += 1
        conv_id = f"conv_{self._conversation_counter}"

        conv = await initiate_conversation(
            initiator, target, self.llm, self.clock, conv_id,
        )
        self.active_conversations[conv_id] = conv

        # Both agents are now in conversation
        from soci.agents.agent import AgentAction
        talk_action = AgentAction(
            type="talk",
            target=target.id,
            detail=f"talking to {target.name}",
            duration_ticks=conv.max_turns,
            needs_satisfied={"social": 0.3},
        )
        initiator.start_action(talk_action)

        talk_action_target = AgentAction(
            type="talk",
            target=initiator.id,
            detail=f"talking to {initiator.name}",
            duration_ticks=conv.max_turns,
            needs_satisfied={"social": 0.3},
        )
        target.start_action(talk_action_target)

        self._emit(f"  [CONV] {initiator.name} starts talking to {target.name}")

        # Both agents observe the conversation start
        for agent, other in [(initiator, target), (target, initiator)]:
            agent.add_observation(
                tick=self.clock.total_ticks,
                day=self.clock.day,
                time_str=self.clock.time_str,
                content=f"Started a conversation with {other.name} at {agent.location}",
                importance=5,
                involved_agents=[other.id],
            )
            # Ensure relationship exists
            agent.relationships.get_or_create(other.id, other.name)

    def _finish_conversation(self, conv: Conversation) -> None:
        """Record a finished conversation in both agents' memories."""
        if len(conv.turns) < 2:
            return

        summary = f"Had a conversation about '{conv.topic}' with "
        for agent_id in conv.participants:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            other_ids = [p for p in conv.participants if p != agent_id]
            other_names = [self.agents[oid].name for oid in other_ids if oid in self.agents]
            agent.memory.add(
                tick=self.clock.total_ticks,
                day=self.clock.day,
                time_str=self.clock.time_str,
                memory_type=MemoryType.CONVERSATION,
                content=f"Had a conversation about '{conv.topic}' with {', '.join(other_names)}.",
                importance=6,
                location=conv.location,
                involved_agents=other_ids,
            )

        # Store in conversation history for API
        self.conversation_history.append(conv.to_dict())
        if len(self.conversation_history) > self._max_conversation_history:
            self.conversation_history = self.conversation_history[-self._max_conversation_history:]

        self._emit(
            f"  [CONV END] Conversation about '{conv.topic}' between "
            f"{', '.join(self.agents[p].name for p in conv.participants if p in self.agents)} ended."
        )

        # Gossip: chance of mentioning a third person both know
        self._maybe_gossip(conv)

    def _maybe_gossip(self, conv: Conversation) -> None:
        """After a conversation, participants might share info about a third person."""
        if len(conv.participants) < 2 or random.random() > 0.35:
            return

        from soci.actions.social import propagate_gossip

        p1 = self.agents.get(conv.participants[0])
        p2 = self.agents.get(conv.participants[1])
        if not p1 or not p2:
            return

        # Find a third person both know
        p1_known = {r.agent_id for r in p1.relationships.get_closest(10) if r.familiarity > 0.2}
        p2_known = {r.agent_id for r in p2.relationships.get_closest(10) if r.familiarity > 0.2}
        mutual = (p1_known & p2_known) - {p1.id, p2.id}

        if not mutual:
            return

        about_id = random.choice(list(mutual))
        about = self.agents.get(about_id)
        if not about:
            return

        # Speaker shares their impression
        speaker, listener = (p1, p2) if random.random() < 0.5 else (p2, p1)
        rel = speaker.relationships.get(about_id)
        if not rel:
            return

        # Generate gossip note based on sentiment
        if rel.sentiment > 0.7:
            note = f"{about.name} is really great, always so helpful"
        elif rel.sentiment < 0.3:
            note = f"I've had some issues with {about.name} lately"
        else:
            note = f"I ran into {about.name} the other day"

        propagate_gossip(speaker, listener, about_id, about.name, note, self.clock.total_ticks)
        self._emit(f"  [GOSSIP] {speaker.name} told {listener.name} about {about.name}")

    async def _generate_reflection(self, agent: Agent) -> None:
        """Generate a reflection for an agent about recent experiences."""
        recent = agent.memory.get_recent(15)
        recent_text = "\n".join(
            f"- [{m.time_str}] {m.content}" for m in recent
        )

        world_desc = self.events.get_world_description()
        loc_desc = self.city.describe_location(agent.location, exclude_agent=agent.id)

        prompt = REFLECT_PROMPT.format(
            time_str=self.clock.time_str,
            day=self.clock.day,
            context=agent.build_context(self.clock.total_ticks, world_desc, loc_desc),
            recent_memories=recent_text,
        )

        result = await self.llm.complete_json(
            system=agent.persona.system_prompt(),
            user_message=prompt,
            model=MODEL_HAIKU,
            temperature=agent.persona.llm_temperature,
            max_tokens=512,
        )

        reflections = result.get("reflections", [])
        mood_shift = result.get("mood_shift", 0.0)
        # Clamp mood_shift to valid range
        try:
            mood_shift = max(-0.3, min(0.3, float(mood_shift)))
        except (TypeError, ValueError):
            mood_shift = 0.0

        for ref_text in reflections:
            if not isinstance(ref_text, str):
                ref_text = str(ref_text)
            agent.add_reflection(
                tick=self.clock.total_ticks,
                day=self.clock.day,
                time_str=self.clock.time_str,
                content=ref_text,
            )

        agent.mood = max(-1.0, min(1.0, agent.mood + mood_shift))
        agent.memory.reset_reflection_accumulator()

        if reflections:
            self._emit(f"  [REFLECT] {agent.name}: {reflections[0]}")

    def _tick_romance(self) -> None:
        """Develop romantic attractions between compatible agents at the same location."""
        agents_list = list(self.agents.values())

        for agent in agents_list:
            if agent.is_player:
                continue
            loc = self.city.get_location(agent.location)
            if not loc:
                continue

            for other_id in loc.occupants:
                if other_id == agent.id or other_id not in self.agents:
                    continue
                other = self.agents[other_id]

                rel = agent.relationships.get(other_id)
                if not rel or rel.familiarity < 0.15:
                    continue  # Need to know someone a bit first

                # Skip if already married to someone else
                if agent.partner_id and agent.partner_id != other_id:
                    continue

                # Attraction grows from positive interactions
                if rel.sentiment > 0.6 and rel.trust > 0.5:
                    # Base attraction growth per tick
                    growth = 0.008
                    # Boost from high agreeableness and extraversion
                    growth += (agent.persona.agreeableness / 100.0) * 0.005
                    # Boost from familiarity
                    growth += rel.familiarity * 0.005
                    # Boost when both at same location and interacting
                    if agent.state.value == "in_conversation":
                        growth += 0.01

                    rel.romantic_interest = min(1.0, rel.romantic_interest + growth)

                # Relationship status progression (no LLM calls — pure rules)
                ri = rel.romantic_interest
                status = rel.relationship_status

                if status == "none" and ri > 0.25:
                    rel.relationship_status = "crushing"
                    self._emit(f"  [ROMANCE] {agent.name} has developed a crush on {other.name}!")
                    agent.add_observation(
                        tick=self.clock.total_ticks, day=self.clock.day,
                        time_str=self.clock.time_str,
                        content=f"I think I'm developing feelings for {other.name}...",
                        importance=7, involved_agents=[other_id],
                    )

                elif status == "crushing" and ri > 0.5 and rel.familiarity > 0.4:
                    # Check if the other person also has feelings
                    other_rel = other.relationships.get(agent.id)
                    if other_rel and other_rel.romantic_interest > 0.3:
                        rel.relationship_status = "dating"
                        other_rel.relationship_status = "dating"
                        agent.partner_id = other_id
                        other.partner_id = agent.id
                        self._emit(f"  [ROMANCE] {agent.name} and {other.name} have started dating!")
                        for a, o in [(agent, other), (other, agent)]:
                            a.add_observation(
                                tick=self.clock.total_ticks, day=self.clock.day,
                                time_str=self.clock.time_str,
                                content=f"I'm now dating {o.name}! I feel excited and nervous.",
                                importance=9, involved_agents=[o.id],
                            )
                        agent.mood = min(1.0, agent.mood + 0.3)
                        other.mood = min(1.0, other.mood + 0.3)

                elif status == "dating" and ri > 0.75 and rel.familiarity > 0.7:
                    other_rel = other.relationships.get(agent.id)
                    if other_rel and other_rel.romantic_interest > 0.65:
                        rel.relationship_status = "engaged"
                        other_rel.relationship_status = "engaged"
                        self._emit(f"  [ROMANCE] {agent.name} and {other.name} got engaged!")
                        for a, o in [(agent, other), (other, agent)]:
                            a.add_observation(
                                tick=self.clock.total_ticks, day=self.clock.day,
                                time_str=self.clock.time_str,
                                content=f"{o.name} and I are engaged! This is the happiest day of my life.",
                                importance=10, involved_agents=[o.id],
                            )
                        agent.mood = min(1.0, agent.mood + 0.4)
                        other.mood = min(1.0, other.mood + 0.4)

                elif status == "engaged" and ri > 0.9 and rel.interaction_count > 15:
                    other_rel = other.relationships.get(agent.id)
                    if other_rel and other_rel.romantic_interest > 0.8:
                        rel.relationship_status = "married"
                        other_rel.relationship_status = "married"
                        self._emit(f"  [ROMANCE] {agent.name} and {other.name} got married!")
                        for a, o in [(agent, other), (other, agent)]:
                            a.add_observation(
                                tick=self.clock.total_ticks, day=self.clock.day,
                                time_str=self.clock.time_str,
                                content=f"I married {o.name} today. I couldn't be happier.",
                                importance=10, involved_agents=[o.id],
                            )

    def get_state_summary(self) -> dict:
        """Get a summary of the current simulation state."""
        return {
            "clock": self.clock.to_dict(),
            "weather": self.events.weather.value,
            "active_events": [e.to_dict() for e in self.events.active_events],
            "agents": {
                aid: {
                    "name": a.name,
                    "gender": a.persona.gender,
                    "location": a.location,
                    "state": a.state.value,
                    "mood": round(a.mood, 2),
                    "needs": a.needs.to_dict(),
                    "action": a.current_action.detail if a.current_action else "idle",
                    "partner_id": a.partner_id,
                }
                for aid, a in self.agents.items()
            },
            "active_conversations": len(self.active_conversations),
            "llm_usage": self.llm.usage.summary(),
        }

    def to_dict(self) -> dict:
        """Serialize full simulation state."""
        return {
            "city": self.city.to_dict(),
            "clock": self.clock.to_dict(),
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "events": self.events.to_dict(),
            "entropy": self.entropy.to_dict(),
            "conversation_counter": self._conversation_counter,
        }

    @classmethod
    def from_dict(cls, data: dict, llm: ClaudeClient) -> Simulation:
        """Restore a simulation from serialized state."""
        city = City.from_dict(data["city"])
        clock = SimClock.from_dict(data["clock"])
        sim = cls(city=city, clock=clock, llm=llm)
        sim.events = EventSystem.from_dict(data["events"])
        sim.entropy = EntropyManager.from_dict(data["entropy"])
        sim._conversation_counter = data.get("conversation_counter", 0)
        for aid, agent_data in data["agents"].items():
            agent = Agent.from_dict(agent_data)
            sim.agents[agent.id] = agent
        return sim
