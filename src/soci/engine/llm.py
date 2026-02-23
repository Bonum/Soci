"""LLM client — supports Claude API, Groq, and Ollama (local LLMs) with model routing and cost tracking."""

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
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"
PROVIDER_HF = "hf"

# Claude model IDs
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Ollama model IDs (popular open-source models)
MODEL_LLAMA = "llama3.1:8b"
MODEL_LLAMA_SMALL = "llama3.1:8b"
MODEL_MISTRAL = "mistral"
MODEL_QWEN = "qwen2.5"
MODEL_GEMMA = "gemma2"

# Groq model IDs (fast cloud inference)
MODEL_GROQ_LLAMA_8B = "llama-3.1-8b-instant"
MODEL_GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
MODEL_GROQ_MIXTRAL = "mixtral-8x7b-32768"

# Google Gemini model IDs (free tier via AI Studio)
MODEL_GEMINI_FLASH = "gemini-2.0-flash"
MODEL_GEMINI_PRO = "gemini-1.5-pro"

# Hugging Face router model IDs (router.huggingface.co/v1 — auto-routes to best provider)
MODEL_HF_QWEN = "Qwen/Qwen2.5-7B-Instruct"          # default — auto-routed, great quality
MODEL_HF_LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_HF_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_HF_SMOL = "HuggingFaceTB/SmolLM3-3B:hf-inference"  # CPU inference, no credits needed

# Approximate cost per 1M tokens (USD) — Ollama is free, Groq is very cheap
COST_PER_1M = {
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
    MODEL_HAIKU: {"input": 0.80, "output": 4.0},
    MODEL_GROQ_LLAMA_8B: {"input": 0.05, "output": 0.08},
    MODEL_GROQ_LLAMA_70B: {"input": 0.59, "output": 0.79},
    MODEL_GROQ_MIXTRAL: {"input": 0.24, "output": 0.24},
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
    if not text:
        return {}
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
        self._rate_limited_until: float = 0.0  # monotonic timestamp

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
                self._rate_limited_until = time.monotonic() + wait
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
        self._rate_limited_until = time.monotonic() + 60  # mark as limited after all retries failed
        return ""

    @property
    def llm_status(self) -> str:
        if time.monotonic() < self._rate_limited_until:
            return "limited"
        return "active"

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
        self._last_error: float = 0.0  # monotonic timestamp of last connection failure

    @property
    def llm_status(self) -> str:
        if time.monotonic() - self._last_error < 30:
            return "limited"   # recent connection error
        return "active"

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
                self._last_error = time.monotonic()
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
# Groq (Fast Cloud Inference) Client
# ============================================================

class GroqClient:
    """Wrapper around the Groq API for fast cloud inference.

    Groq provides extremely fast inference (~500 tok/s) with parallel request support.
    Free tier: 30 requests/min on llama-3.1-8b-instant.
    Sign up: https://console.groq.com
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_GROQ_LLAMA_8B,
        max_retries: int = 3,
        max_rpm: int = 28,  # Stay just under 30 req/min free tier
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()
        self.provider = PROVIDER_GROQ
        self._http = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        # Rate limiter: enforce minimum delay between requests
        # 30 req/min = 1 req per 2s; use 2.2s to stay safely under
        self._min_request_interval = 60.0 / max_rpm
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()
        # Circuit breaker: if Groq returns a long retry-after (daily quota),
        # skip all calls until the quota window resets.
        self._rate_limited_until: float = 0.0  # monotonic timestamp

    def _is_quota_exhausted(self) -> bool:
        """Return True if we are inside a long-wait circuit-breaker window."""
        import time
        return time.monotonic() < self._rate_limited_until

    def _handle_429(self, retry_after_str: str, attempt: int) -> float:
        """Parse retry-after and update circuit breaker. Returns seconds to sleep.

        Short waits (≤15s, per-minute limit) → return the wait so caller retries.
        Long waits (>15s, daily quota) → arm the circuit breaker and return 0
        so the caller gives up immediately instead of blocking for minutes.
        """
        import time
        try:
            retry_after = float(retry_after_str)
        except (ValueError, TypeError):
            retry_after = max(3.0, 2 ** attempt + 1)

        if retry_after > 15:
            self._rate_limited_until = time.monotonic() + retry_after
            logger.warning(
                f"Groq quota exhausted — skipping LLM calls for {retry_after:.0f}s "
                f"(until quota resets). Simulation continues without LLM."
            )
            return 0.0  # caller should give up immediately
        return retry_after  # short per-minute limit — wait and retry

    async def _wait_for_rate_limit(self) -> None:
        """Wait if needed to stay under the RPM limit."""
        import time
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                wait_time = self._min_request_interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request to Groq (async, rate-limited)."""
        model = self._map_model(model or self.default_model)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self._is_quota_exhausted():
            logger.debug("Groq quota circuit breaker active — skipping complete()")
            return ""

        for attempt in range(self.max_retries):
            try:
                await self._wait_for_rate_limit()
                response = await self._http.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                usage = data.get("usage", {})
                self.usage.record(
                    model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )

                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    body = e.response.text[:200] if e.response.text else ""
                    sleep_for = self._handle_429(
                        e.response.headers.get("retry-after", ""), attempt
                    )
                    logger.warning(f"Groq 429: {body[:120]}")
                    if sleep_for == 0:
                        return ""  # quota exhausted — skip immediately
                    await asyncio.sleep(sleep_for)
                elif e.response.status_code == 401:
                    raise ValueError("Invalid GROQ_API_KEY")
                else:
                    logger.error(f"Groq API error: {e.response.status_code} {e.response.text[:200]}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Groq error: {e}")
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
        """Send a JSON-mode request to Groq."""
        model = self._map_model(model or self.default_model)

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        if self._is_quota_exhausted():
            logger.debug("Groq quota circuit breaker active — skipping complete_json()")
            return {}

        for attempt in range(self.max_retries):
            try:
                await self._wait_for_rate_limit()
                response = await self._http.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                usage = data.get("usage", {})
                self.usage.record(
                    model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )

                text = data["choices"][0]["message"]["content"]
                return _parse_json_response(text)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    body = e.response.text[:200] if e.response.text else ""
                    sleep_for = self._handle_429(
                        e.response.headers.get("retry-after", ""), attempt
                    )
                    logger.warning(f"Groq 429 (json): {body[:120]}")
                    if sleep_for == 0:
                        return {}  # quota exhausted — skip immediately
                    await asyncio.sleep(sleep_for)
                else:
                    logger.error(f"Groq JSON error: {e.response.status_code}")
                    if attempt == self.max_retries - 1:
                        return {}
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Groq JSON error: {e}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(1)
        return {}

    def _map_model(self, model: str) -> str:
        """Map Claude/Ollama model names to Groq equivalents."""
        mapping = {
            MODEL_SONNET: self.default_model,     # Use 8B for all — 70B has low daily token limit
            MODEL_HAIKU: self.default_model,       # Use default (8B) for routine
            MODEL_LLAMA: MODEL_GROQ_LLAMA_8B,
        }
        return mapping.get(model, model)

    @property
    def llm_status(self) -> str:
        return "limited" if self._is_quota_exhausted() else "active"


# ============================================================
# Google Gemini Client (free tier via OpenAI-compatible endpoint)
# ============================================================

class GeminiClient:
    """Google Gemini via the OpenAI-compatible AI Studio endpoint.

    Free tier (no credit card):
      - gemini-2.0-flash: 15 RPM, 1 M tokens/day — plenty for a simulation.
      - Get a free key at https://aistudio.google.com/apikey
    Uses the OpenAI-compatible endpoint so no extra SDK is needed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_GEMINI_FLASH,
        max_retries: int = 3,
        max_rpm: int = 14,  # stay under the 15 RPM free-tier limit
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()
        self.provider = PROVIDER_GEMINI
        self._http = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self._min_request_interval = 60.0 / max_rpm
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()
        self._rate_limited_until: float = 0.0

    def _is_quota_exhausted(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    async def _wait_for_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.monotonic()

    def _map_model(self, model: str) -> str:
        """Map Claude/Groq model names to Gemini equivalents."""
        mapping = {
            MODEL_SONNET: self.default_model,
            MODEL_HAIKU: self.default_model,
            MODEL_GROQ_LLAMA_8B: MODEL_GEMINI_FLASH,
        }
        return mapping.get(model, model)

    @property
    def llm_status(self) -> str:
        return "limited" if self._is_quota_exhausted() else "active"

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request to Gemini."""
        if self._is_quota_exhausted():
            logger.debug("Gemini quota circuit breaker active — skipping complete()")
            return ""

        model = self._map_model(model or self.default_model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                await self._wait_for_rate_limit()
                resp = await self._http.post("chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.usage.record(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get("retry-after", "5")
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 5.0
                    if wait > 30:
                        self._rate_limited_until = time.monotonic() + wait
                        logger.warning(f"Gemini quota exhausted for {wait:.0f}s")
                        return ""
                    logger.warning(f"Gemini rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gemini HTTP error: {e.response.status_code}")
                    if attempt == self.max_retries - 1:
                        return ""
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                if attempt == self.max_retries - 1:
                    return ""
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
        """Send a JSON-mode request to Gemini."""
        if self._is_quota_exhausted():
            logger.debug("Gemini quota circuit breaker active — skipping complete_json()")
            return {}

        model = self._map_model(model or self.default_model)
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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.max_retries):
            try:
                await self._wait_for_rate_limit()
                resp = await self._http.post("chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.usage.record(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                text = data["choices"][0]["message"]["content"]
                return _parse_json_response(text)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get("retry-after", "5")
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 5.0
                    if wait > 30:
                        self._rate_limited_until = time.monotonic() + wait
                        logger.warning(f"Gemini quota exhausted for {wait:.0f}s")
                        return {}
                    logger.warning(f"Gemini rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gemini JSON error: {e.response.status_code}")
                    if attempt == self.max_retries - 1:
                        return {}
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Gemini JSON error: {e}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(1)
        return {}


# ============================================================
# Hugging Face Serverless Inference Client (free tier)
# ============================================================

class HFInferenceClient:
    """Hugging Face Serverless Inference via OpenAI-compatible endpoint.

    Free tier (no credit card required):
      - Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B, and many others.
      - HF_TOKEN is auto-injected in HF Spaces — no manual setup needed.
      - Get a token at https://huggingface.co/settings/tokens
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_HF_SMOL,
        max_retries: int = 3,
    ) -> None:
        # Priority: explicit arg → named secrets (personal token) → Space auto-injected HF_TOKEN
        # HF_TOKEN is auto-injected in HF Spaces but only has basic inference (no credits for routed models).
        # A personal token stored as hf_soci_token / soci_token / HW_WR_TOKEN takes precedence.
        self.api_key = (
            api_key
            or os.environ.get("hf_soci_token", "")
            or os.environ.get("soci_token", "")
            or os.environ.get("HW_WR_TOKEN", "")
            or os.environ.get("HF_TOKEN", "")
        )
        if not self.api_key:
            logger.warning(
                "Neither HF_TOKEN nor soci_token is set — HF Inference will not make LLM calls. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.default_model = default_model
        self.max_retries = max_retries
        self.usage = LLMUsage()
        self.provider = PROVIDER_HF
        self._http = httpx.AsyncClient(
            base_url="https://router.huggingface.co/v1/",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # HF can be slow under load
        )
        self._rate_limited_until: float = 0.0
        self._auth_error: str = ""
        self._last_error: str = ""   # last non-auth error for diagnostics

    def _is_quota_exhausted(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    def _map_model(self, model: str) -> str:
        """Map Claude/Groq/Gemini model names to HF router equivalents."""
        mapping = {
            MODEL_SONNET: self.default_model,
            MODEL_HAIKU: self.default_model,
            MODEL_GROQ_LLAMA_8B: MODEL_HF_LLAMA,
            MODEL_GEMINI_FLASH: self.default_model,
        }
        return mapping.get(model, model)

    @property
    def llm_status(self) -> str:
        if not self.api_key:
            return "nokey"
        if self._auth_error:
            return "nokey"   # gated model / bad token
        return "limited" if self._is_quota_exhausted() else "active"

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            self._last_error = "HF_TOKEN / soci_token not set — add it to your HF Space secrets"
            return ""
        if self._is_quota_exhausted():
            logger.debug("HF quota circuit breaker active — skipping complete()")
            return ""

        model = self._map_model(model or self.default_model)
        # /no_think disables chain-of-thought on SmolLM3 and similar thinking models;
        # harmless for other models since it's prepended before the system prompt.
        system_with_flag = "/no_think\n" + system
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_with_flag},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                resp = await self._http.post("chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.usage.record(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                self._last_error = ""  # clear on success
                text = data["choices"][0]["message"]["content"] or ""
                # Strip any <think>...</think> blocks that thinking models may emit
                import re as _re
                text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
                return text
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:300]
                if status == 429:
                    retry_after = e.response.headers.get("retry-after", "10")
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 10.0
                    if wait > 60:
                        self._rate_limited_until = time.monotonic() + wait
                        logger.warning(f"HF quota exhausted for {wait:.0f}s")
                        return ""
                    logger.warning(f"HF rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                elif status in (401, 402, 403, 410):
                    # Auth/payment failure — circuit-break for 1h to stop spam retries.
                    # 402 means no credits (token lacks Inference Providers permission).
                    # Fix: add hf_soci_token secret in Space with a token that has inference perms.
                    self._rate_limited_until = time.monotonic() + 3600
                    self._auth_error = body
                    logger.error(
                        f"HF auth error ({status}): {body[:120]} — "
                        "Add hf_soci_token Space secret with a token that has Inference Providers permission"
                    )
                    return ""
                elif status in (503, 504):
                    # Model cold-start — circuit-break immediately so the sim
                    # tick is not blocked; retry once the estimated window passes.
                    try:
                        import json as _json
                        estimated = float(_json.loads(e.response.text).get("estimated_time", 30))
                    except Exception:
                        estimated = 30.0
                    wait = max(estimated, 20.0)
                    self._rate_limited_until = time.monotonic() + wait
                    self._last_error = f"503 model loading — retry in {wait:.0f}s"
                    logger.warning(f"HF model cold-start, circuit-breaking for {wait:.0f}s")
                    return ""
                else:
                    self._last_error = f"HTTP {status}: {body}"
                    logger.error(f"HF HTTP error: {status} {body}")
                    if attempt == self.max_retries - 1:
                        return ""
                    await asyncio.sleep(2)
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"HF error: {e}")
                if attempt == self.max_retries - 1:
                    return ""
                await asyncio.sleep(2)
        if not self._last_error:
            self._last_error = "all retries exhausted"
        return ""

    async def complete_json(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        if self._is_quota_exhausted():
            logger.debug("HF quota circuit breaker active — skipping complete_json()")
            return {}

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
# Factory — create the right client based on config
# ============================================================

def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
) -> ClaudeClient | OllamaClient | GroqClient | GeminiClient | HFInferenceClient:
    """Create an LLM client based on environment or explicit config.

    Provider detection order:
    1. Explicit provider argument
    2. LLM_PROVIDER env var
    3. If ANTHROPIC_API_KEY is set → Claude
    4. If GROQ_API_KEY is set → Groq (fast cloud)
    5. If GEMINI_API_KEY is set → Gemini (free tier)
    6. If HF_TOKEN is set → HF Inference (free, auto-available in HF Spaces)
    7. Default → Ollama (local)
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "").lower()

    if not provider:
        # Auto-detect: Claude → Groq → Gemini → HF → Ollama
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = PROVIDER_CLAUDE
        elif os.environ.get("GROQ_API_KEY"):
            provider = PROVIDER_GROQ
        elif os.environ.get("GEMINI_API_KEY"):
            provider = PROVIDER_GEMINI
        elif (os.environ.get("HF_TOKEN") or os.environ.get("hf_soci_token")
              or os.environ.get("soci_token") or os.environ.get("HW_WR_TOKEN")):
            provider = PROVIDER_HF
        else:
            provider = PROVIDER_OLLAMA

    if provider == PROVIDER_CLAUDE:
        default_model = model or MODEL_HAIKU
        return ClaudeClient(default_model=default_model)
    elif provider == PROVIDER_GROQ:
        default_model = model or os.environ.get("GROQ_MODEL", MODEL_GROQ_LLAMA_8B)
        return GroqClient(default_model=default_model)
    elif provider == PROVIDER_GEMINI:
        default_model = model or os.environ.get("GEMINI_MODEL", MODEL_GEMINI_FLASH)
        return GeminiClient(default_model=default_model)
    elif provider == PROVIDER_HF:
        default_model = model or os.environ.get("HF_MODEL", MODEL_HF_SMOL)
        return HFInferenceClient(default_model=default_model)
    elif provider == PROVIDER_OLLAMA:
        default_model = model or os.environ.get("OLLAMA_MODEL", MODEL_LLAMA)
        return OllamaClient(base_url=ollama_url, default_model=default_model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'claude', 'groq', 'gemini', 'hf', or 'ollama'.")


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
