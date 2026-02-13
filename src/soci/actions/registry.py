"""Action registry — maps action types to handlers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent, AgentAction
    from soci.world.city import City


class ActionType(Enum):
    MOVE = "move"
    WORK = "work"
    EAT = "eat"
    SLEEP = "sleep"
    TALK = "talk"
    EXERCISE = "exercise"
    SHOP = "shop"
    RELAX = "relax"
    WANDER = "wander"


# Default needs satisfaction per action type
ACTION_NEEDS: dict[str, dict[str, float]] = {
    "work": {"purpose": 0.3},
    "eat": {"hunger": 0.5},
    "sleep": {"energy": 0.6},
    "talk": {"social": 0.3},
    "exercise": {"energy": -0.1, "fun": 0.2, "comfort": 0.1},
    "shop": {"hunger": 0.1, "comfort": 0.1},
    "relax": {"energy": 0.1, "fun": 0.2, "comfort": 0.2},
    "wander": {"fun": 0.1},
    "move": {},
}

# Default duration in ticks per action type
ACTION_DURATIONS: dict[str, int] = {
    "move": 1,
    "work": 4,
    "eat": 2,
    "sleep": 8,
    "talk": 2,
    "exercise": 3,
    "shop": 2,
    "relax": 2,
    "wander": 1,
}


def resolve_action(raw_action: dict, agent: Agent, city: City) -> AgentAction:
    """Convert a raw LLM action dict into a validated AgentAction."""
    from soci.agents.agent import AgentAction

    action_type = raw_action.get("action", "wander")
    if action_type not in {a.value for a in ActionType}:
        action_type = "wander"

    target = raw_action.get("target", "")
    detail = raw_action.get("detail", "")
    duration = raw_action.get("duration", ACTION_DURATIONS.get(action_type, 1))
    duration = max(1, min(8, int(duration)))  # Clamp to 1-8

    # Validate move targets
    if action_type == "move" and target:
        loc = city.get_location(target)
        if not loc:
            # Try to find a matching location by name
            for lid, l in city.locations.items():
                if target.lower() in l.name.lower():
                    target = lid
                    break
            else:
                # Invalid target, just wander instead
                action_type = "wander"
                target = ""

    # Get default needs satisfaction, allow LLM override via detail
    needs = dict(ACTION_NEEDS.get(action_type, {}))

    return AgentAction(
        type=action_type,
        target=target,
        detail=detail,
        duration_ticks=duration,
        needs_satisfied=needs,
    )
