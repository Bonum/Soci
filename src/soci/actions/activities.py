"""Activities — non-social actions like working, eating, sleeping, etc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent, AgentAction
    from soci.world.city import City
    from soci.world.clock import SimClock

logger = logging.getLogger(__name__)


def execute_activity(agent: Agent, action: AgentAction, city: City, clock: SimClock) -> str:
    """Execute a non-movement, non-social activity. Returns a description."""
    location = city.get_location(agent.location)
    loc_name = location.name if location else "somewhere"

    match action.type:
        case "work":
            return _do_work(agent, action, loc_name, clock)
        case "eat":
            return _do_eat(agent, action, loc_name, clock)
        case "sleep":
            return _do_sleep(agent, action, loc_name, clock)
        case "exercise":
            return _do_exercise(agent, action, loc_name, clock)
        case "shop":
            return _do_shop(agent, action, loc_name, clock)
        case "relax":
            return _do_relax(agent, action, loc_name, clock)
        case "wander":
            return _do_wander(agent, action, loc_name, clock)
        case _:
            return f"{agent.name} does something at {loc_name}."


def _do_work(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"working at {loc_name}"
    return f"{agent.name} is {detail}."


def _do_eat(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"having a meal at {loc_name}"
    return f"{agent.name} is {detail}."


def _do_sleep(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    return f"{agent.name} is sleeping at {loc_name}."


def _do_exercise(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"exercising at {loc_name}"
    return f"{agent.name} is {detail}."


def _do_shop(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"shopping at {loc_name}"
    return f"{agent.name} is {detail}."


def _do_relax(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"relaxing at {loc_name}"
    return f"{agent.name} is {detail}."


def _do_wander(agent: Agent, action: AgentAction, loc_name: str, clock: SimClock) -> str:
    detail = action.detail or f"wandering around {loc_name}"
    return f"{agent.name} is {detail}."
