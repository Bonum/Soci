"""LLM client — Claude API wrapper with model routing, cost tracking, and prompt templates."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# Model IDs
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Approximate cost per 1M tokens (USD)
COST_PER_1M = {
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
    MODEL_HAIKU: {"input": 0.80, "output": 4.0},
}


@dataclass
class LLMUsage:
    """Tracks API usage and costs."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_by_model: dict[str, int] = field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        if model not in self.tokens_by_model:
            self.tokens_by_model[model] = {"input": 0, "output": 0}
        self.tokens_by_model[model]["input"] += input_tokens
        self.tokens_by_model[model]["output"] += output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        total = 0.0
        for model, tokens in self.tokens_by_model.items():
            costs = COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
            total += tokens["input"] / 1_000_000 * costs["input"]
            total += tokens["output"] / 1_000_000 * costs["output"]
        return total

    def summary(self) -> str:
        lines = [
            f"Total API calls: {self.total_calls}",
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
            f"Estimated cost: ${self.estimated_cost_usd:.4f}",
        ]
        for model, count in self.calls_by_model.items():
            short = model.split("-")[1] if "-" in model else model
            lines.append(f"  {short}: {count} calls")
        return "\n".join(lines)


class ClaudeClient:
    """Wrapper around the Anthropic Claude API with model routing and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_HAIKU,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a message to Claude and return the text response."""
        model = model or self.default_model

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                # Track usage
                self.usage.record(
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                return response.content[0].text

            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

        return ""

    async def complete_json(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """Send a message and parse the response as JSON."""
        # Add JSON instruction to the prompt
        json_instruction = (
            "\n\nRespond ONLY with valid JSON. No markdown, no explanation, no extra text. "
            "Just the JSON object."
        )
        text = await self.complete(
            system=system,
            user_message=user_message + json_instruction,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Try to extract JSON from the response
        text = text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
            return {}


# --- Prompt Templates ---

PLAN_DAY_PROMPT = """\
It is {time_str} on Day {day}. You just woke up.

{context}

Based on your personality, needs, and memories, plan your day. What will you do today?
Think about your obligations (work, responsibilities) and your desires (socializing, fun, rest).

Respond with a JSON object:
{{
  "plan": ["item 1", "item 2", ...],
  "reasoning": "brief explanation of why this plan"
}}

Keep the plan to 5-8 items. Be specific about locations and times.
"""

DECIDE_ACTION_PROMPT = """\
It is {time_str} on Day {day}.

{context}

You are currently at {location_name}. You just finished: {last_activity}.

What do you do next? Consider your needs, your plan, who's around, and any events happening.

Respond with a JSON object:
{{
  "action": "move|work|eat|sleep|talk|exercise|shop|relax|wander",
  "target": "location_id or agent_id (if talking) or empty string",
  "detail": "what specifically you're doing, in first person",
  "duration": 1-4,
  "reasoning": "brief internal thought about why"
}}

Available locations you can move to: {connected_locations}
People at your current location: {people_here}
"""

OBSERVE_PROMPT = """\
It is {time_str} on Day {day}.

{context}

You just noticed: {observation}

How important is this to you (1-10)? What do you think about it?

Respond with a JSON object:
{{
  "importance": 1-10,
  "reaction": "your brief internal thought or feeling about this"
}}
"""

REFLECT_PROMPT = """\
It is {time_str} on Day {day}.

{context}

RECENT EXPERIENCES:
{recent_memories}

Take a moment to reflect on your recent experiences. What patterns do you notice?
What have you learned? How do you feel about things?

Respond with a JSON object:
{{
  "reflections": ["reflection 1", "reflection 2", ...],
  "mood_shift": -0.3 to 0.3,
  "reasoning": "why your mood shifted this way"
}}

Generate 1-3 reflections. Each should be a genuine insight, not just a summary.
"""

CONVERSATION_PROMPT = """\
It is {time_str} on Day {day}.

{context}

You are at {location_name}. {other_name} is here too.

WHAT YOU KNOW ABOUT {other_name}:
{relationship_context}

{conversation_history}

{other_name} says: "{other_message}"

How do you respond? Stay in character. Be natural — not every conversation is deep.
Sometimes people make small talk, sometimes they argue, sometimes they're awkward.

Respond with a JSON object:
{{
  "message": "your spoken response",
  "inner_thought": "what you're actually thinking",
  "sentiment_delta": -0.1 to 0.1,
  "trust_delta": -0.05 to 0.05
}}
"""

CONVERSATION_INITIATE_PROMPT = """\
It is {time_str} on Day {day}.

{context}

You are at {location_name}. {other_name} is here.

WHAT YOU KNOW ABOUT {other_name}:
{relationship_context}

You decide to start a conversation with {other_name}. What do you say?
Consider the time of day, location, your mood, and your history with them.

Respond with a JSON object:
{{
  "message": "what you say to start the conversation",
  "inner_thought": "why you're initiating this conversation",
  "topic": "brief topic label"
}}
"""
