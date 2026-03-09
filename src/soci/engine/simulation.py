"""Simulation — the main loop that orchestrates the entire city simulation."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, Optional

from soci.agents.agent import Agent, AgentAction, AgentState
from soci.agents.memory import MemoryType
from soci.agents.persona import Persona, load_personas
from soci.agents.generator import generate_personas
from soci.agents.routine import DailyRoutine, build_routine, check_motivation_override
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
from soci.world.city import City, generate_houses
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
        # Daily routines per agent (rebuilt from persona each day)
        self.routines: dict[str, DailyRoutine] = {}
        self._last_routine_day: int = -1
        # Speed-aware flags (set by server loop for fast-forward)
        self._skip_llm_this_tick: bool = False
        self._max_convos_this_tick: int = 0  # 0 = no limit
        self._max_llm_calls_this_tick: int = 0  # 0 = no limit; global budget across all categories
        self._llm_calls_this_tick: int = 0  # counter, reset each tick
        # LLM call probability: 0.0 = never use LLM (routine only), 1.0 = always (default).
        # Applied per potential LLM call site. Tuned at 0.45 for ~10h Gemini free-tier runtime.
        self.llm_call_probability: float = 1.0
        # Callback for real-time output
        self.on_event: Optional[Callable[[str], None]] = None

    def _llm_budget_remaining(self) -> int:
        """How many LLM calls can still be made this tick. 0 = unlimited."""
        if self._max_llm_calls_this_tick <= 0:
            return 999  # unlimited
        return max(0, self._max_llm_calls_this_tick - self._llm_calls_this_tick)

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

    def generate_agents(self, count: int) -> None:
        """Procedurally generate `count` agents with houses and routines."""
        # Generate houses for the new agents
        house_ids = generate_houses(self.city, count)
        logger.info(f"Generated {len(house_ids)} houses for new agents")

        # Generate personas assigned to those houses
        personas = generate_personas(count, self.city)
        for persona in personas:
            agent = Agent(persona)
            self.add_agent(agent)
        logger.info(f"Generated {len(personas)} procedural agents")

    def _rebuild_routines(self) -> None:
        """Rebuild daily routines for all agents (called at start of each day)."""
        day = self.clock.day
        if self._last_routine_day == day:
            return
        self._last_routine_day = day
        for agent in self.agents.values():
            if not agent.is_player:
                self.routines[agent.id] = build_routine(agent.persona, day)
        logger.info(f"Built daily routines for {len(self.routines)} agents (day {day})")

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
        self._llm_calls_this_tick = 0  # Reset budget counter
        self._emit(f"\n--- {self.clock.datetime_str} ({self.clock.time_of_day.value}) ---")

        # 0. Rebuild routines at the start of each day (or first tick)
        self._rebuild_routines()

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

        # 4. Daily plans — routine IS the plan, skip LLM plan generation for agents with routines
        plan_coros = []
        plan_agents = []
        for agent in ordered_agents:
            if agent.needs_new_plan(self.clock) and not should_skip_llm(agent, self.clock):
                # If agent has a routine, set a simple plan from routine slots
                if agent.id in self.routines:
                    routine = self.routines[agent.id]
                    plan_items = []
                    seen = set()
                    for slot in routine.slots:
                        label = slot.detail
                        if label not in seen:
                            plan_items.append(label)
                            seen.add(label)
                    agent.set_daily_plan(
                        plan_items[:8], self.clock.day,
                        self.clock.total_ticks, self.clock.time_str,
                    )
                elif random.random() < self.llm_call_probability:
                    plan_coros.append(self._generate_daily_plan(agent))
                    plan_agents.append(agent)

        # Cap plan coros to LLM budget
        budget = self._llm_budget_remaining()
        if budget < len(plan_coros):
            for c in plan_coros[budget:]:
                c.close()
            plan_coros = plan_coros[:budget]
            plan_agents = plan_agents[:budget]

        if plan_coros:
            await batch_llm_calls(plan_coros, self._max_concurrent)
            self._llm_calls_this_tick += len(plan_coros)
            for agent in plan_agents:
                self._emit(f"[PLAN] {agent.name} planned their day: {'; '.join(agent.daily_plan[:3])}...")

        # 5. Process each agent — tick their needs, handle actions
        action_coros = []
        action_agents = []
        routine_actions: list[tuple[Agent, AgentAction]] = []

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

            # Skip sleeping agents using per-agent awareness
            if should_skip_llm(agent, self.clock, self.routines.get(agent.id)):
                continue

            # Agent is idle — check routine first, then LLM fallback
            routine = self.routines.get(agent.id)
            if routine:
                slot = routine.get_action_for_time(self.clock.hour, self.clock.minute)
                if slot:
                    # Check if motivation overrides the routine
                    override = check_motivation_override(
                        slot, agent.needs, agent.mood,
                        agent.persona.extraversion,
                        agent.persona.conscientiousness,
                    )
                    if override:
                        slot = override
                        self._emit(f"  [MOTIVATION] {agent.name}: {override.detail}")
                    action = AgentAction(
                        type=slot.action_type,
                        target=slot.target_location,
                        detail=slot.detail,
                        duration_ticks=slot.duration_ticks,
                        needs_satisfied=slot.needs_satisfied,
                    )
                    routine_actions.append((agent, action))
                    continue

            # No routine slot — fallback to LLM (rare), skip in fast-forward
            if not self._skip_llm_this_tick and random.random() < self.llm_call_probability:
                action_coros.append(self._decide_action(agent))
                action_agents.append(agent)

        # Execute routine-driven actions (no LLM needed)
        for agent, action in routine_actions:
            await self._execute_action(agent, action)

        # Run LLM action decisions concurrently (only for agents without routine match)
        # Cap to avoid rate limits with many agents
        cap = min(
            self._max_convos_this_tick if self._max_convos_this_tick > 0 else len(action_coros),
            self._llm_budget_remaining(),
        )
        if len(action_coros) > cap:
            for c in action_coros[cap:]:
                c.close()
            action_coros = action_coros[:cap]
            action_agents = action_agents[:cap]
        if action_coros and not self._skip_llm_this_tick:
            action_results = await batch_llm_calls(action_coros, self._max_concurrent)
            self._llm_calls_this_tick += len(action_coros)
            for agent, result in zip(action_agents, action_results):
                if result and isinstance(result, AgentAction):
                    await self._execute_action(agent, result)

        # 5b. Player auto-sleep: put idle players to sleep at night
        self._handle_player_sleep()

        # 6. Handle active conversations (skip in 50x mode)
        if not self._skip_llm_this_tick:
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
                        # Always continue active conversations — they already passed
                        # the probability gate when initiated; don't double-gate them.
                        conv_coros.append(
                            continue_conversation(conv, responder, other, self.llm, self.clock)
                        )

            # Limit conversations by speed cap and global budget
            conv_cap = min(
                self._max_convos_this_tick if self._max_convos_this_tick > 0 else len(conv_coros),
                self._llm_budget_remaining(),
            )
            if len(conv_coros) > conv_cap:
                for c in conv_coros[conv_cap:]:
                    c.close()
                conv_coros = conv_coros[:conv_cap]

            if conv_coros:
                await batch_llm_calls(conv_coros, self._max_concurrent)
                self._llm_calls_this_tick += len(conv_coros)
        else:
            # 50x mode: force-finish all active conversations
            for conv_id, conv in list(self.active_conversations.items()):
                self._finish_conversation(conv)
            self.active_conversations.clear()

        # 7. Social: maybe start new conversations (respect speed limits + budget)
        if not self._skip_llm_this_tick and self._llm_budget_remaining() > 0:
            if self._max_convos_this_tick == 0 or len(self.active_conversations) < self._max_convos_this_tick:
                if random.random() < self.llm_call_probability:
                    await self._handle_social_interactions(ordered_agents)

        # 8. Reflections for agents with enough accumulated importance
        if not self._skip_llm_this_tick and self._llm_budget_remaining() > 0:
            reflect_coros = []
            reflect_agents = []
            for agent in ordered_agents:
                if agent.memory.should_reflect() and not agent.is_player:
                    if random.random() < self.llm_call_probability:
                        reflect_coros.append(self._generate_reflection(agent))
                        reflect_agents.append(agent)

            # Limit by speed cap and global budget
            reflect_cap = min(
                1 if self._max_convos_this_tick > 0 else len(reflect_coros),
                self._llm_budget_remaining(),
            )
            if len(reflect_coros) > reflect_cap:
                for c in reflect_coros[reflect_cap:]:
                    c.close()
                reflect_coros = reflect_coros[:reflect_cap]

            if reflect_coros:
                await batch_llm_calls(reflect_coros, self._max_concurrent)
                self._llm_calls_this_tick += len(reflect_coros)

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
        """Check if any co-located agents should start conversations."""
        # Hard cap: at most 3 simultaneous conversations
        max_convos = 3
        if len(self.active_conversations) >= max_convos:
            return

        # States where an agent can initiate or join a conversation
        _CONVERSABLE = {"idle", "eating", "relaxing", "working", "exercising", "shopping"}

        for agent in agents:
            # Skip sleeping, moving, or agents already in conversation
            if agent.state.value not in _CONVERSABLE:
                continue
            # Skip players (they initiate via the /player/talk API)
            if agent.is_player:
                continue
            # Skip if already in a conversation
            in_conv = any(
                agent.id in c.participants
                for c in self.active_conversations.values()
            )
            if in_conv:
                continue

            # Potential partners: anyone at same location who is also conversable
            others = [
                aid for aid in self.city.get_agents_at(agent.location)
                if aid != agent.id
                and aid in self.agents
                and self.agents[aid].state.value in _CONVERSABLE
                and not any(aid in c.participants for c in self.active_conversations.values())
            ]
            if not others:
                continue

            partner_id = pick_conversation_partner(agent, others, self.clock)
            if partner_id and should_initiate_conversation(agent, partner_id, self.clock):
                await self._start_conversation(agent, self.agents[partner_id])
                break  # One new conversation per tick max

    def _handle_player_sleep(self) -> None:
        """Auto-sleep idle players during sleeping hours, wake them in the morning."""
        for agent in self.agents.values():
            if not agent.is_player:
                continue
            if self.clock.is_sleeping_hours:
                if not agent.is_busy and agent.state.value != "sleeping":
                    sleep_action = AgentAction(
                        type="sleep",
                        target=agent.persona.home_location,
                        detail=f"{agent.name} goes to sleep for the night",
                        duration_ticks=32,  # ~8 hours
                        needs_satisfied={"energy": 0.9},
                    )
                    # Move home if needed (teleport — player is asleep)
                    home = agent.persona.home_location
                    if agent.location != home and self.city.get_location(home):
                        old_loc = self.city.get_location(agent.location)
                        new_loc = self.city.get_location(home)
                        if old_loc:
                            old_loc.remove_occupant(agent.id)
                        new_loc.add_occupant(agent.id)
                        agent.location = home
                    agent.start_action(sleep_action)
                    self._emit(f"  {agent.name} goes to sleep for the night.")
            else:
                # Wake player if still sleeping past sleeping hours
                if agent.state.value == "sleeping":
                    agent.state = AgentState.IDLE
                    agent.current_action = None
                    agent._action_ticks_remaining = 0

    async def _start_conversation(self, initiator: Agent, target: Agent) -> None:
        """Start a conversation between two agents."""
        self._conversation_counter += 1
        conv_id = f"conv_{self._conversation_counter}"

        conv = await initiate_conversation(
            initiator, target, self.llm, self.clock, conv_id,
        )
        if conv is None:
            return  # LLM unavailable — skip this conversation
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

                # Gender-based attraction: opposite genders attract (nonbinary attracted to all)
                a_gender = agent.persona.gender
                o_gender = other.persona.gender
                attracted = (
                    a_gender == "unknown"
                    or a_gender == "nonbinary"
                    or o_gender == "unknown"
                    or o_gender == "nonbinary"
                    or a_gender != o_gender
                )
                if not attracted:
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
            "llm_provider": getattr(self.llm, "provider", "unknown"),
            "llm_model": getattr(self.llm, "default_model", "unknown"),
            "llm_status": getattr(self.llm, "llm_status", "active"),
            "llm_calls_last_tick": self._llm_calls_this_tick,
            "llm_skipped": self._skip_llm_this_tick,
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
            "conversation_history": self.conversation_history,
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
        sim.conversation_history = data.get("conversation_history", [])
        for aid, agent_data in data["agents"].items():
            agent = Agent.from_dict(agent_data)
            sim.agents[agent.id] = agent
        # Rebuild routines from restored personas (deterministic)
        sim._rebuild_routines()
        return sim
