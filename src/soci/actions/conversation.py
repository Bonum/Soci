"""Conversation — multi-agent dialogue generation via LLM."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from soci.agents.agent import Agent
    from soci.engine.llm import ClaudeClient
    from soci.world.clock import SimClock

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """One turn in a conversation."""

    speaker_id: str
    speaker_name: str
    message: str
    inner_thought: str = ""
    tick: int = 0


@dataclass
class Conversation:
    """A multi-turn conversation between agents."""

    id: str
    location: str
    participants: list[str]  # Agent IDs
    turns: list[ConversationTurn] = field(default_factory=list)
    topic: str = ""
    is_active: bool = True
    max_turns: int = 3

    @property
    def is_finished(self) -> bool:
        return not self.is_active or len(self.turns) >= self.max_turns

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) >= self.max_turns:
            self.is_active = False

    def get_history_text(self, max_turns: int = 10) -> str:
        """Format conversation history for LLM context."""
        if not self.turns:
            return "This is the start of the conversation."
        lines = [f"CONVERSATION SO FAR (topic: {self.topic}):"]
        for turn in self.turns[-max_turns:]:
            lines.append(f"  {turn.speaker_name}: \"{turn.message}\"")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "location": self.location,
            "participants": self.participants,
            "turns": [
                {
                    "speaker_id": t.speaker_id,
                    "speaker_name": t.speaker_name,
                    "message": t.message,
                    "inner_thought": t.inner_thought,
                    "tick": t.tick,
                }
                for t in self.turns
            ],
            "topic": self.topic,
            "is_active": self.is_active,
            "max_turns": self.max_turns,
        }


async def initiate_conversation(
    initiator: Agent,
    target: Agent,
    llm: ClaudeClient,
    clock: SimClock,
    conversation_id: str,
) -> Optional[Conversation]:
    """Start a conversation between two agents. Returns None if LLM unavailable."""
    from soci.engine.llm import CONVERSATION_INITIATE_PROMPT, MODEL_SONNET

    # Build relationship context
    rel = initiator.relationships.get(target.id)
    if rel:
        rel_context = rel.describe()
    else:
        rel_context = f"{target.name} — someone I don't know yet."

    prompt = CONVERSATION_INITIATE_PROMPT.format(
        time_str=clock.time_str,
        day=clock.day,
        context=initiator.build_context(
            clock.total_ticks,
            "",
            f"at {initiator.location}",
        ),
        location_name=initiator.location,
        other_name=target.name,
        relationship_context=rel_context,
    )

    result = await llm.complete_json(
        system=initiator.persona.system_prompt(),
        user_message=prompt,
        model=MODEL_SONNET,
        temperature=initiator.persona.llm_temperature,
        max_tokens=512,
    )

    # LLM unavailable — use scripted fallback so conversations still animate in the UI
    if not result:
        starters = [
            {"message": f"Hey {target.name}, how's it going?", "topic": "greeting", "inner_thought": "Making small talk."},
            {"message": f"Oh, {target.name}! Didn't expect to run into you here.", "topic": "chance meeting", "inner_thought": "Good to see a familiar face."},
            {"message": "What have you been up to lately?", "topic": "small talk", "inner_thought": "Curious about their day."},
            {"message": "Lovely weather today, isn't it?", "topic": "weather", "inner_thought": "Breaking the ice."},
            {"message": f"Hi {target.name}! Have you heard any news lately?", "topic": "news", "inner_thought": "Looking for something to talk about."},
            {"message": "I was just thinking about grabbing something to eat. You?", "topic": "food", "inner_thought": "Maybe we can go together."},
        ]
        result = random.choice(starters)

    message = result.get("message", f"Hey, {target.name}.")
    topic = result.get("topic", "small talk")

    conv = Conversation(
        id=conversation_id,
        location=initiator.location,
        participants=[initiator.id, target.id],
        topic=topic,
    )

    turn = ConversationTurn(
        speaker_id=initiator.id,
        speaker_name=initiator.name,
        message=message,
        inner_thought=result.get("inner_thought", ""),
        tick=clock.total_ticks,
    )
    conv.add_turn(turn)

    logger.info(f"Conversation started: {initiator.name} -> {target.name}: \"{message}\"")
    return conv


async def continue_conversation(
    conversation: Conversation,
    responder: Agent,
    other: Agent,
    llm: ClaudeClient,
    clock: SimClock,
) -> ConversationTurn:
    """Generate the next response in a conversation."""
    from soci.engine.llm import CONVERSATION_PROMPT, MODEL_SONNET

    # Get the last message from the other person
    last_turn = conversation.turns[-1]
    if last_turn.speaker_id == responder.id:
        # It's not our turn
        return last_turn

    # Build relationship context
    rel = responder.relationships.get(other.id)
    if rel:
        rel_context = rel.describe()
    else:
        rel_context = f"{other.name} — someone I just met."

    prompt = CONVERSATION_PROMPT.format(
        time_str=clock.time_str,
        day=clock.day,
        context=responder.build_context(
            clock.total_ticks,
            "",
            f"at {responder.location}",
        ),
        location_name=responder.location,
        other_name=other.name,
        relationship_context=rel_context,
        conversation_history=conversation.get_history_text(),
        other_message=last_turn.message,
    )

    result = await llm.complete_json(
        system=responder.persona.system_prompt(),
        user_message=prompt,
        model=MODEL_SONNET,
        temperature=responder.persona.llm_temperature,
        max_tokens=512,
    )

    # LLM unavailable — scripted response keeps conversation alive in the UI
    if not result:
        replies = [
            {"message": "Ha, yeah, I was just thinking the same thing!", "inner_thought": "Go with the flow.", "sentiment_delta": 0.05, "trust_delta": 0.02},
            {"message": "Not too bad, honestly. Just keeping busy.", "inner_thought": "Keep it light.", "sentiment_delta": 0.03, "trust_delta": 0.01},
            {"message": "Interesting! Tell me more.", "inner_thought": "Show some curiosity.", "sentiment_delta": 0.04, "trust_delta": 0.02},
            {"message": "Yeah, it's been that kind of day.", "inner_thought": "Relate to them.", "sentiment_delta": 0.02, "trust_delta": 0.01},
            {"message": "I hear you. Things have been a bit hectic on my end too.", "inner_thought": "Empathize.", "sentiment_delta": 0.04, "trust_delta": 0.03},
            {"message": "Good point. I hadn't thought of it that way.", "inner_thought": "Give them credit.", "sentiment_delta": 0.05, "trust_delta": 0.03},
        ]
        result = random.choice(replies)

    message = result.get("message", "Hmm, interesting.")

    turn = ConversationTurn(
        speaker_id=responder.id,
        speaker_name=responder.name,
        message=message,
        inner_thought=result.get("inner_thought", ""),
        tick=clock.total_ticks,
    )
    conversation.add_turn(turn)

    # Update relationship
    sentiment_delta = result.get("sentiment_delta", 0.0)
    trust_delta = result.get("trust_delta", 0.0)
    rel = responder.relationships.get_or_create(other.id, other.name)
    rel.update_after_interaction(
        tick=clock.total_ticks,
        sentiment_delta=sentiment_delta,
        trust_delta=trust_delta,
        note=f"Talked about {conversation.topic}",
    )

    logger.info(f"  {responder.name}: \"{message}\"")
    return turn
