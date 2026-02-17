"""Scheduler — manages agent turn order and batching for LLM calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent
    from soci.agents.routine import DailyRoutine
    from soci.world.clock import SimClock

logger = logging.getLogger(__name__)


def prioritize_agents(agents: list[Agent], clock: SimClock) -> list[Agent]:
    """Sort agents by priority for this tick. Agents with urgent needs go first."""
    def priority_score(agent: Agent) -> float:
        score = 0.0
        # Urgent needs boost priority
        if agent.needs.is_critical:
            score += 10.0
        urgent = agent.needs.urgent_needs
        score += len(urgent) * 2.0
        # Idle agents need decisions
        if not agent.is_busy:
            score += 5.0
        # Sleeping agents are low priority
        if agent.state.value == "sleeping":
            score -= 5.0
        # Players always get processed
        if agent.is_player:
            score += 20.0
        return score

    return sorted(agents, key=priority_score, reverse=True)


async def batch_llm_calls(
    coros: list,
    max_concurrent: int = 10,
) -> list:
    """Run multiple LLM coroutines concurrently with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro):
        async with semaphore:
            return await coro

    results = await asyncio.gather(
        *[limited(c) for c in coros],
        return_exceptions=True,
    )

    # Log any errors
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"LLM call {i} failed: {r}")
            results[i] = None

    return results


def should_skip_llm(
    agent: Agent,
    clock: SimClock,
    routine: Optional[DailyRoutine] = None,
) -> bool:
    """Determine if we can skip the LLM call for this agent (habit caching)."""
    # Never skip players
    if agent.is_player:
        return False

    # If agent is busy with a multi-tick action, skip
    if agent.is_busy:
        return True

    # Per-agent sleep awareness via routine
    if routine:
        if not routine.is_awake_at(clock.hour) and agent.state.value == "sleeping":
            return True
    elif agent.state.value == "sleeping" and clock.is_sleeping_hours:
        # Fallback to global sleeping hours if no routine
        return True

    return False
