"""LLM client — supports Claude API and Ollama (local LLMs) with model routing and cost tracking."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# --- Provider constants ---
PROVIDER_CLAUDE = "claude"
PROVIDER_OLLAMA = "ollama"

# Claude model IDs
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Ollama model IDs (popular open-source models)
MODEL_LLAMA = "llama3.1:8b"
MODEL_LLAMA_SMALL = "llama3.1:8b"
MODEL_MISTRAL = "mistral"
MODEL_QWEN = "qwen2.5"
MODEL_GEMMA = "gemma2"

# Approximate cost per 1M tokens (USD) — Ollama is free
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
            costs = COST_PER_1M.get(model, {"input": 0.0, "output": 0.0})
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


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, handling markdown blocks and extra text."""
    text = text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return {}


# ============================================================
# Claude (Anthropic API) Client
# ============================================================

class ClaudeClient:
    """Wrapper around the Anthropic Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_HAIKU,
        max_retries: int = 3,
    ) -> None:
        import anthropic
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()
        self.provider = PROVIDER_CLAUDE

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        import anthropic
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
        return _parse_json_response(text)


# ============================================================
# Ollama (Local LLM) Client
# ============================================================

class OllamaClient:
    """Wrapper around Ollama's local API for running open-source LLMs.

    Ollama serves models locally at http://localhost:11434.
    Install: https://ollama.com
    Pull a model: ollama pull llama3.1
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = MODEL_LLAMA,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()
        self.provider = PROVIDER_OLLAMA
        self._http = httpx.AsyncClient(timeout=180.0)

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a message to the local Ollama model (async)."""
        model = model or self.default_model
        model = self._map_model(model)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                response = await self._http.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                input_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)
                self.usage.record(model, input_tokens, output_tokens)

                return data.get("message", {}).get("content", "")

            except httpx.ConnectError:
                msg = (
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running: 'ollama serve'"
                )
                logger.error(msg)
                if attempt == self.max_retries - 1:
                    raise ConnectionError(msg)
                await asyncio.sleep(1)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    msg = (
                        f"Model '{model}' not found in Ollama. "
                        f"Pull it first: 'ollama pull {model}'"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                logger.error(f"Ollama API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        return ""

    async def complete_json(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """Send a JSON-mode request to Ollama (async, uses native format: json)."""
        model = model or self.default_model
        model = self._map_model(model)

        json_instruction = (
            "\n\nRespond ONLY with valid JSON. No markdown, no explanation, no extra text. "
            "Just the JSON object."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message + json_instruction},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                response = await self._http.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                input_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)
                self.usage.record(model, input_tokens, output_tokens)

                text = data.get("message", {}).get("content", "")
                return _parse_json_response(text)

            except httpx.ConnectError:
                logger.error(f"Cannot connect to Ollama at {self.base_url}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Ollama JSON error: {e}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(1)
        return {}

    def _map_model(self, model: str) -> str:
        """Map Claude model names to Ollama equivalents so existing code works."""
        mapping = {
            MODEL_SONNET: self.default_model,  # Use the main local model
            MODEL_HAIKU: self.default_model,    # Same model for both (local is free)
        }
        return mapping.get(model, model)


# ============================================================
# Factory — create the right client based on config
# ============================================================

def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
) -> ClaudeClient | OllamaClient:
    """Create an LLM client based on environment or explicit config.

    Provider detection order:
    1. Explicit provider argument
    2. LLM_PROVIDER env var
    3. If ANTHROPIC_API_KEY is set → Claude
    4. Default → Ollama (free, local)
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "").lower()

    if not provider:
        # Auto-detect: use Claude if key is set, otherwise Ollama
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = PROVIDER_CLAUDE
        else:
            provider = PROVIDER_OLLAMA

    if provider == PROVIDER_CLAUDE:
        default_model = model or MODEL_HAIKU
        return ClaudeClient(default_model=default_model)
    elif provider == PROVIDER_OLLAMA:
        default_model = model or MODEL_LLAMA
        return OllamaClient(base_url=ollama_url, default_model=default_model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'claude' or 'ollama'.")


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
