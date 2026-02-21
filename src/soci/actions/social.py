"""Social actions — relationship formation, gossip, and social dynamics."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent
    from soci.world.city import City
    from soci.world.clock import SimClock


def should_initiate_conversation(agent: Agent, other_id: str, clock: SimClock) -> bool:
    """Decide whether an agent should start a conversation with someone."""
    if agent.is_busy or agent.state.value == "sleeping":
        return False

    # Extraversion drives conversation initiation
    base_chance = agent.persona.extraversion / 20.0  # 0.05 to 0.5

    # Boost if social need is low (lonely agents seek conversation)
    if agent.needs.social < 0.3:
        base_chance += 0.2
    if agent.needs.social < 0.15:
        base_chance += 0.15  # Very lonely — even introverts will try

    # Boost if we know the person
    rel = agent.relationships.get(other_id)
    if rel and rel.familiarity > 0.3:
        base_chance += 0.15
    # Strong boost for romantic partners
    if rel and agent.partner_id == other_id:
        base_chance += 0.3

    # Reduce if we recently talked to them
    if rel and rel.last_interaction_tick > 0:
        ticks_since = clock.total_ticks - rel.last_interaction_tick
        if ticks_since < 8:  # Less than 2 hours
            base_chance -= 0.3

    # Sleeping hours — very unlikely
    if clock.is_sleeping_hours:
        base_chance *= 0.1

    return random.random() < max(0.0, base_chance)


def pick_conversation_partner(agent: Agent, others_at_location: list[str], clock: SimClock) -> str | None:
    """Pick who to talk to from the people at the current location."""
    if not others_at_location:
        return None

    candidates: list[tuple[float, str]] = []
    for other_id in others_at_location:
        score = 1.0
        rel = agent.relationships.get(other_id)
        if rel:
            # Prefer people we know and like
            score += rel.closeness * 2.0
            # Strong pull toward romantic partners
            if rel.romantic_interest > 0.3:
                score += rel.romantic_interest * 3.0
            if agent.partner_id == other_id:
                score += 4.0  # Partners strongly prefer each other
            # But also have some chance of talking to strangers (curiosity)
            ticks_since = clock.total_ticks - rel.last_interaction_tick
            if ticks_since < 8:
                score *= 0.3  # Cooldown
        else:
            # Strangers: moderate interest based on openness
            score += agent.persona.openness / 20.0
        candidates.append((score, other_id))

    # Weighted random selection
    total = sum(s for s, _ in candidates)
    if total <= 0:
        return None

    r = random.random() * total
    cumulative = 0.0
    for score, other_id in candidates:
        cumulative += score
        if r <= cumulative:
            return other_id

    return candidates[-1][1] if candidates else None


def propagate_gossip(
    speaker: Agent,
    listener: Agent,
    about_id: str,
    about_name: str,
    note: str,
    tick: int,
) -> None:
    """When agents talk, information about third parties can spread."""
    # The listener forms/updates an impression of the person being discussed
    listener_rel = listener.relationships.get_or_create(about_id, about_name)

    # Gossip influence is modulated by trust in the speaker
    speaker_rel = listener.relationships.get(speaker.id)
    trust_weight = speaker_rel.trust if speaker_rel else 0.3

    # The note from the speaker influences the listener's sentiment
    listener_rel.update_after_interaction(
        tick=tick,
        sentiment_delta=0.0,  # Gossip doesn't change sentiment directly
        trust_delta=0.0,
        note=f"Heard from {speaker.name}: {note}",
    )
    # Small familiarity bump — you now know about this person
    listener_rel.familiarity = min(
        1.0,
        listener_rel.familiarity + 0.02 * trust_weight,
    )
