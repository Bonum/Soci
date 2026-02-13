"""Entropy management — keeps the simulation diverse and interesting."""

from __future__ import annotations

import random
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent
    from soci.world.clock import SimClock
    from soci.world.events import EventSystem

logger = logging.getLogger(__name__)


class EntropyManager:
    """Manages simulation entropy to prevent bland, repetitive behavior."""

    def __init__(self) -> None:
        # How often to inject events (every N ticks)
        self.event_injection_interval: int = 10
        # Ticks since last injection
        self._ticks_since_event: int = 0
        # Track agent behavior patterns for drift detection
        self._action_history: dict[str, list[str]] = {}  # agent_id -> last N actions
        self._history_window: int = 20

    def tick(
        self,
        agents: list[Agent],
        event_system: EventSystem,
        clock: SimClock,
        city_location_ids: list[str],
    ) -> list[str]:
        """Process one tick of entropy management. Returns list of notable events/messages."""
        messages: list[str] = []
        self._ticks_since_event += 1

        # Track action patterns
        for agent in agents:
            if agent.current_action:
                history = self._action_history.setdefault(agent.id, [])
                history.append(agent.current_action.type)
                if len(history) > self._history_window:
                    self._action_history[agent.id] = history[-self._history_window:]

        # Detect repetitive behavior
        for agent in agents:
            if self._is_stuck_in_loop(agent.id):
                messages.append(
                    f"[ENTROPY] {agent.name} seems stuck in a behavioral loop — "
                    f"injecting stimulus."
                )
                self._inject_personal_stimulus(agent, clock)

        # Periodic event injection
        if self._ticks_since_event >= self.event_injection_interval:
            new_events = event_system.tick(city_location_ids)
            self._ticks_since_event = 0
            for evt in new_events:
                messages.append(f"[EVENT] {evt.name}: {evt.description}")
        else:
            # Still tick the event system for weather/expiry
            event_system.tick(city_location_ids)

        # Time-based entropy: inject daily rhythm changes
        if clock.hour == 12 and clock.minute == 0:
            messages.append("[RHYTHM] Noon — the city bustles with lunch crowds.")
        elif clock.hour == 18 and clock.minute == 0:
            messages.append("[RHYTHM] Evening — people head home or to the bar.")
        elif clock.hour == 22 and clock.minute == 0:
            messages.append("[RHYTHM] Late night — the city quiets down.")

        return messages

    def _is_stuck_in_loop(self, agent_id: str) -> bool:
        """Detect if an agent is repeating the same actions."""
        history = self._action_history.get(agent_id, [])
        if len(history) < 10:
            return False
        # Check if last 10 actions are all the same
        recent = history[-10:]
        unique = set(recent)
        return len(unique) <= 2 and "sleep" not in unique

    def _inject_personal_stimulus(self, agent: Agent, clock: SimClock) -> None:
        """Inject a personal event to break an agent out of a loop."""
        stimuli = [
            f"{agent.name} suddenly remembers something important they forgot to do.",
            f"{agent.name} gets an unexpected phone call from an old friend.",
            f"{agent.name} notices something unusual in their surroundings.",
            f"{agent.name} overhears an interesting conversation nearby.",
            f"{agent.name} finds a forgotten note in their pocket.",
            f"{agent.name} suddenly craves something completely different.",
        ]
        stimulus = random.choice(stimuli)
        agent.add_observation(
            tick=clock.total_ticks,
            day=clock.day,
            time_str=clock.time_str,
            content=stimulus,
            importance=7,
        )

    def get_conflict_catalysts(self, agents: list[Agent]) -> list[tuple[str, str, str]]:
        """Identify potential conflicts between agents based on their personas.
        Returns list of (agent1_id, agent2_id, tension_description) tuples.
        """
        catalysts = []

        # Find agents with opposing values or competing interests
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                tension = self._find_tension(a, b)
                if tension:
                    catalysts.append((a.id, b.id, tension))

        return catalysts

    def _find_tension(self, a: Agent, b: Agent) -> str | None:
        """Find natural tension between two agents."""
        # Big personality differences can create friction
        extraversion_gap = abs(a.persona.extraversion - b.persona.extraversion)
        agreeableness_gap = abs(a.persona.agreeableness - b.persona.agreeableness)

        if extraversion_gap >= 6 and agreeableness_gap >= 4:
            return "personality clash — one is outgoing and blunt, the other is reserved and sensitive"

        # Competing values
        a_values = set(a.persona.values)
        b_values = set(b.persona.values)
        if a_values and b_values and not a_values.intersection(b_values):
            return f"different values — {a.name} values {', '.join(a.persona.values)}, while {b.name} values {', '.join(b.persona.values)}"

        return None

    def to_dict(self) -> dict:
        return {
            "event_injection_interval": self.event_injection_interval,
            "ticks_since_event": self._ticks_since_event,
            "action_history": dict(self._action_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntropyManager:
        mgr = cls()
        mgr.event_injection_interval = data.get("event_injection_interval", 10)
        mgr._ticks_since_event = data.get("ticks_since_event", 0)
        mgr._action_history = data.get("action_history", {})
        return mgr
