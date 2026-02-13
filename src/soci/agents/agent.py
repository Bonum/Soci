"""Agent — a simulated person with persona, memory, needs, and relationships."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from soci.agents.persona import Persona
from soci.agents.memory import MemoryStream, MemoryType
from soci.agents.needs import NeedsState
from soci.agents.relationships import RelationshipGraph

if TYPE_CHECKING:
    from soci.world.clock import SimClock


class AgentState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    WORKING = "working"
    EATING = "eating"
    SLEEPING = "sleeping"
    SOCIALIZING = "socializing"
    EXERCISING = "exercising"
    SHOPPING = "shopping"
    RELAXING = "relaxing"
    IN_CONVERSATION = "in_conversation"


@dataclass
class AgentAction:
    """An action an agent has decided to take."""

    type: str           # move, work, eat, sleep, talk, exercise, shop, relax, wander
    target: str = ""    # Location ID or agent ID depending on action
    detail: str = ""    # Free-text detail: what specifically they're doing
    duration_ticks: int = 1  # How many ticks this action takes
    needs_satisfied: dict[str, float] = field(default_factory=dict)  # e.g. {"hunger": 0.4}

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target": self.target,
            "detail": self.detail,
            "duration_ticks": self.duration_ticks,
            "needs_satisfied": self.needs_satisfied,
        }


class Agent:
    """A simulated person living in the city."""

    def __init__(self, persona: Persona) -> None:
        self.persona = persona
        self.id = persona.id
        self.name = persona.name
        self.memory = MemoryStream()
        self.needs = NeedsState()
        self.relationships = RelationshipGraph()

        # Current state
        self.state: AgentState = AgentState.IDLE
        self.location: str = persona.home_location
        self.current_action: Optional[AgentAction] = None
        self._action_ticks_remaining: int = 0

        # Mood: -1.0 (terrible) to 1.0 (great)
        self.mood: float = 0.3
        # Daily plan (list of planned actions for today)
        self.daily_plan: list[str] = []
        self._has_plan_today: bool = False
        # Last time we made an LLM call for this agent
        self._last_llm_tick: int = -1
        # Whether this agent is a human player
        self.is_player: bool = False

    @property
    def is_busy(self) -> bool:
        return self._action_ticks_remaining > 0

    def needs_new_plan(self, clock: SimClock) -> bool:
        """Does this agent need a new daily plan?"""
        if self.is_player:
            return False
        # Plan at the start of each day (6am)
        if clock.hour == 6 and clock.minute == 0 and not self._has_plan_today:
            return True
        return not self._has_plan_today

    def start_action(self, action: AgentAction) -> None:
        """Begin executing an action."""
        self.current_action = action
        self._action_ticks_remaining = action.duration_ticks
        # Set agent state based on action type
        state_map = {
            "move": AgentState.MOVING,
            "work": AgentState.WORKING,
            "eat": AgentState.EATING,
            "sleep": AgentState.SLEEPING,
            "talk": AgentState.IN_CONVERSATION,
            "exercise": AgentState.EXERCISING,
            "shop": AgentState.SHOPPING,
            "relax": AgentState.RELAXING,
            "wander": AgentState.IDLE,
        }
        self.state = state_map.get(action.type, AgentState.IDLE)

    def tick_action(self) -> bool:
        """Advance current action by one tick. Returns True if action completed."""
        if self._action_ticks_remaining > 0:
            self._action_ticks_remaining -= 1
            # Satisfy needs based on action
            if self.current_action:
                for need, amount in self.current_action.needs_satisfied.items():
                    per_tick = amount / max(1, self.current_action.duration_ticks)
                    self.needs.satisfy(need, per_tick)
            if self._action_ticks_remaining <= 0:
                self.state = AgentState.IDLE
                self.current_action = None
                return True
        return False

    def tick_needs(self, is_sleeping: bool = False) -> None:
        """Decay needs by one tick."""
        self.needs.tick(is_sleeping=is_sleeping)
        # Mood is influenced by need satisfaction
        avg_needs = (
            self.needs.hunger + self.needs.energy + self.needs.social +
            self.needs.purpose + self.needs.comfort + self.needs.fun
        ) / 6.0
        # Mood drifts toward need satisfaction level
        self.mood += (avg_needs - 0.5 - self.mood) * 0.1
        self.mood = max(-1.0, min(1.0, self.mood))

    def add_observation(
        self,
        tick: int,
        day: int,
        time_str: str,
        content: str,
        importance: int = 5,
        location: str = "",
        involved_agents: Optional[list[str]] = None,
    ) -> None:
        """Record an observation in memory."""
        self.memory.add(
            tick=tick,
            day=day,
            time_str=time_str,
            memory_type=MemoryType.OBSERVATION,
            content=content,
            importance=importance,
            location=location or self.location,
            involved_agents=involved_agents,
        )

    def add_reflection(
        self,
        tick: int,
        day: int,
        time_str: str,
        content: str,
        importance: int = 7,
    ) -> None:
        """Record a reflection in memory."""
        self.memory.add(
            tick=tick,
            day=day,
            time_str=time_str,
            memory_type=MemoryType.REFLECTION,
            content=content,
            importance=importance,
            location=self.location,
        )

    def set_daily_plan(self, plan: list[str], day: int, tick: int, time_str: str) -> None:
        """Set the agent's plan for today."""
        self.daily_plan = plan
        self._has_plan_today = True
        plan_text = "; ".join(plan)
        self.memory.add(
            tick=tick,
            day=day,
            time_str=time_str,
            memory_type=MemoryType.PLAN,
            content=f"My plan for today: {plan_text}",
            importance=6,
            location=self.location,
        )

    def reset_daily_plan(self) -> None:
        """Reset plan flag for a new day."""
        self._has_plan_today = False

    def build_context(self, tick: int, world_description: str, location_description: str) -> str:
        """Build the full context string for LLM prompts."""
        parts = [
            f"CURRENT STATE:",
            f"- Time: Day {self.memory.memories[-1].day if self.memory.memories else 1}",
            f"- Location: {location_description}",
            f"- Mood: {self._mood_description()}",
            f"- Needs: {self.needs.describe()}",
            f"- Currently: {self.state.value}",
            f"",
            f"WORLD: {world_description}",
            f"",
            f"PEOPLE I KNOW:",
            self.relationships.describe_known_people(),
            f"",
            f"RECENT MEMORIES:",
            self.memory.context_summary(tick),
        ]
        if self.daily_plan:
            parts.insert(5, f"- Today's plan: {'; '.join(self.daily_plan)}")
        return "\n".join(parts)

    def _mood_description(self) -> str:
        if self.mood > 0.6:
            return "feeling great"
        elif self.mood > 0.2:
            return "in a good mood"
        elif self.mood > -0.2:
            return "feeling okay"
        elif self.mood > -0.6:
            return "in a bad mood"
        else:
            return "feeling terrible"

    def to_dict(self) -> dict:
        return {
            "persona": self.persona.to_dict(),
            "memory": self.memory.to_dict(),
            "needs": self.needs.to_dict(),
            "relationships": self.relationships.to_dict(),
            "state": self.state.value,
            "location": self.location,
            "current_action": self.current_action.to_dict() if self.current_action else None,
            "action_ticks_remaining": self._action_ticks_remaining,
            "mood": round(self.mood, 3),
            "daily_plan": self.daily_plan,
            "has_plan_today": self._has_plan_today,
            "last_llm_tick": self._last_llm_tick,
            "is_player": self.is_player,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Agent:
        persona = Persona.from_dict(data["persona"])
        agent = cls(persona)
        agent.memory = MemoryStream.from_dict(data["memory"])
        agent.needs = NeedsState.from_dict(data["needs"])
        agent.relationships = RelationshipGraph.from_dict(data["relationships"])
        agent.state = AgentState(data["state"])
        agent.location = data["location"]
        if data["current_action"]:
            agent.current_action = AgentAction(**data["current_action"])
        agent._action_ticks_remaining = data["action_ticks_remaining"]
        agent.mood = data["mood"]
        agent.daily_plan = data["daily_plan"]
        agent._has_plan_today = data["has_plan_today"]
        agent._last_llm_tick = data["last_llm_tick"]
        agent.is_player = data["is_player"]
        return agent
