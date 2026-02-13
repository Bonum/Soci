"""Movement — handles agent movement between locations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent, AgentAction
    from soci.world.city import City
    from soci.world.clock import SimClock

logger = logging.getLogger(__name__)


def execute_move(agent: Agent, action: AgentAction, city: City, clock: SimClock) -> str:
    """Move an agent to a new location. Returns a description of what happened."""
    target = action.target
    if not target:
        return f"{agent.name} tried to move but had no destination."

    current = agent.location
    target_loc = city.get_location(target)
    if not target_loc:
        return f"{agent.name} tried to go somewhere that doesn't exist."

    # Check if locations are connected
    current_loc = city.get_location(current)
    if current_loc and target not in current_loc.connected_to:
        # Not directly connected — check if reachable through one hop
        for mid_id in current_loc.connected_to:
            mid_loc = city.get_location(mid_id)
            if mid_loc and target in mid_loc.connected_to:
                # Route through intermediate location
                break
        else:
            return f"{agent.name} can't get to {target_loc.name} from here directly."

    if target_loc.is_full:
        return f"{agent.name} tried to go to {target_loc.name} but it's full."

    # Execute the move
    success = city.move_agent(agent.id, current, target)
    if success:
        agent.location = target
        return f"{agent.name} walked to {target_loc.name}."
    else:
        return f"{agent.name} couldn't move to {target_loc.name}."


def get_best_location_for_need(agent: Agent, need: str, city: City) -> str | None:
    """Suggest the best location for satisfying a need."""
    need_to_zones: dict[str, list[str]] = {
        "hunger": ["commercial"],  # cafe, grocery, restaurant
        "energy": ["residential"],  # home
        "social": ["commercial", "public"],  # cafe, bar, park
        "purpose": ["work"],  # office
        "comfort": ["residential"],
        "fun": ["public", "commercial"],  # park, bar, gym
    }

    preferred_zones = need_to_zones.get(need, ["public"])
    current_loc = city.get_location(agent.location)
    if not current_loc:
        return None

    # If current location already satisfies the need, stay
    if current_loc.zone in preferred_zones:
        return agent.location

    # Check connected locations
    candidates = []
    for loc_id in current_loc.connected_to:
        loc = city.get_location(loc_id)
        if loc and loc.zone in preferred_zones and not loc.is_full:
            candidates.append(loc_id)

    if candidates:
        return candidates[0]

    # Check two hops away
    for loc_id in current_loc.connected_to:
        mid_loc = city.get_location(loc_id)
        if not mid_loc:
            continue
        for far_id in mid_loc.connected_to:
            far_loc = city.get_location(far_id)
            if far_loc and far_loc.zone in preferred_zones and not far_loc.is_full:
                return far_id

    return None
