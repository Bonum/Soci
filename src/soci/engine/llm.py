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
PROVIDER_NN = "nn"

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
# gemini-2.0-flash-lite is the reliable free-tier default on the OpenAI-compatible endpoint.
# gemini-1.5-flash* models return 404 on current API keys — do not use them.
MODEL_GEMINI_FLASH = "gemini-2.0-flash-lite"       # free tier, confirmed working
MODEL_GEMINI_FLASH_FALLBACK = "gemini-2.0-flash-001"  # versioned fallback
MODEL_GEMINI_PRO = "gemini-1.5-pro"

# Fallback chain: tried in order when a model returns a not-available error
_GEMINI_FALLBACK_CHAIN: dict[str, str] = {
    "gemini-2.0-flash":     "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp": "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001": "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite": MODEL_GEMINI_FLASH_FALLBACK,
    # 1.5-flash models return 404 on current API keys — skip the entire 1.5 family
}

# Keywords in any Gemini error body that indicate the model is unavailable on this endpoint
_GEMINI_MODEL_UNAVAILABLE_KWS = (
    "not found", "not supported", "invalid argument",
    "does not exist", "unavailable", "serverless",
)

# Soci NN model (ONNX, runs locally — no API key needed)
MODEL_NN_SOCI = "RayMelius/soci-agent-nn"

# Ollama model IDs for Soci fine-tuned models
MODEL_OLLAMA_SOCI = "soci-agent-7b"   # load via: ollama create soci-agent-7b -f Modelfile

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
        default_model: str = MODEL_OLLAMA_SOCI,
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
        self._last_error: str = ""
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

    def _handle_429(self, retry_after_str: str, attempt: int, body: str = "") -> float:
        """Parse retry-after and update circuit breaker. Returns seconds to sleep.

        Short waits (≤120s, per-minute limit) → return the wait so caller retries.
        Long waits (>120s, daily quota) → arm the circuit breaker and return 0
        so the caller gives up immediately instead of blocking for minutes.
        """
        import time
        try:
            retry_after = float(retry_after_str)
        except (ValueError, TypeError):
            retry_after = max(3.0, 2 ** attempt + 1)

        # Only circuit-break on genuinely long waits (daily quota) or explicit quota messages.
        # Groq can send retry-after: 30-60 for per-minute limits — those should just wait & retry.
        is_daily_quota = retry_after > 120 or "daily" in body.lower() or "limit" in body.lower()
        if is_daily_quota:
            self._rate_limited_until = time.monotonic() + retry_after
            logger.warning(
                f"Groq daily quota exhausted — skipping LLM calls for {retry_after:.0f}s "
                f"(until quota resets). Simulation continues without LLM."
            )
            return 0.0  # caller should give up immediately
        # Per-minute throttle — wait and retry (cap at 60s to avoid blocking too long)
        wait = min(retry_after, 60.0)
        logger.info(f"Groq per-minute rate limit — waiting {wait:.0f}s before retry")
        return wait

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
            self._last_error = f"quota exhausted (resets in {(self._rate_limited_until - time.monotonic())/3600:.1f}h)"
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

                self._last_error = ""
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    body = e.response.text[:200] if e.response.text else ""
                    sleep_for = self._handle_429(
                        e.response.headers.get("retry-after", ""), attempt, body
                    )
                    if sleep_for == 0:
                        self._last_error = f"429 daily quota exhausted: {body[:120]}"
                        return ""  # daily quota exhausted — skip immediately
                    await asyncio.sleep(sleep_for)
                elif e.response.status_code == 401:
                    raise ValueError("Invalid GROQ_API_KEY")
                else:
                    self._last_error = f"HTTP {e.response.status_code}: {e.response.text[:120]}"
                    logger.error(f"Groq API error: {e.response.status_code} {e.response.text[:200]}")
                    if attempt == self.max_retries - 1:
                        return ""
                    await asyncio.sleep(1)
            except Exception as e:
                self._last_error = str(e)[:120]
                logger.error(f"Groq error: {e}")
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
                        e.response.headers.get("retry-after", ""), attempt, body
                    )
                    if sleep_for == 0:
                        return {}  # daily quota exhausted — skip immediately
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

    Free tier (no credit card, as of 2026):
      - gemini-2.0-flash: 5 RPM, ~1,500 RPD, 250,000 TPM
      - Daily quota resets at midnight Pacific Time
      - Paid tier: $0.10/1M input tokens, $0.40/1M output tokens
      - Get a free key at https://aistudio.google.com/apikey
    Uses the OpenAI-compatible endpoint so no extra SDK is needed.

    Token/cost guide (typical Soci request ~1,000 input + 90 output tokens):
      - Cost per request (paid): ~$0.000133  ($0.10 input + $0.40 output per 1M)
      - 1,500 RPD free tier ≈ $0.20/day on paid tier
      -   500 RPD usage     ≈ $0.07/day
      - Override daily limit via GEMINI_DAILY_LIMIT env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = MODEL_GEMINI_FLASH,
        max_retries: int = 3,
        max_rpm: int = 4,   # stay under 5 RPM free-tier limit (was 14, caused constant 429s)
        daily_limit: int = 1500,  # free-tier RPD; override with GEMINI_DAILY_LIMIT
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
        self._last_error: str = ""
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
        # Automatic model fallback: if the configured model is unavailable on the endpoint,
        # we silently downgrade to the next in the chain (e.g. 2.0-flash → 1.5-flash).
        self._unavailable_models: set[str] = set()
        # Daily usage tracking — resets at midnight Pacific (UTC-8/-7)
        self._daily_limit: int = int(os.environ.get("GEMINI_DAILY_LIMIT", str(daily_limit)))
        self._daily_requests: int = 0
        self._daily_date: str = ""           # "YYYY-MM-DD" in Pacific time
        self._warned_thresholds: set = set()  # tracks which % levels were already logged

    def _is_quota_exhausted(self) -> bool:
        return time.monotonic() < self._rate_limited_until

    @staticmethod
    def _secs_until_pacific_midnight() -> float:
        """Seconds from now until the next midnight Pacific Time (UTC-8).

        Gemini free-tier quotas reset at midnight Pacific, so this is the
        correct circuit-breaker duration after daily quota exhaustion.
        """
        import datetime as _dt
        pacific = _dt.timezone(_dt.timedelta(hours=-8))
        now = _dt.datetime.now(pacific)
        midnight = (now + _dt.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        secs = (midnight - now).total_seconds()
        return max(secs, 60.0)  # at least 60s even if we're right at midnight

    def _track_daily_request(self) -> None:
        """Increment daily counter and log warnings at 50/70/90/99% of the daily limit."""
        import datetime as _dt
        # Pacific time offset: UTC-8 (PST) / UTC-7 (PDT). Use -8 as a safe conservative value.
        pacific_offset = _dt.timezone(_dt.timedelta(hours=-8))
        today = _dt.datetime.now(pacific_offset).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_requests = 0
            self._warned_thresholds = set()
        self._daily_requests += 1
        pct = self._daily_requests / self._daily_limit
        remaining = self._daily_limit - self._daily_requests
        for threshold in (0.50, 0.70, 0.90, 0.99):
            if pct >= threshold and threshold not in self._warned_thresholds:
                self._warned_thresholds.add(threshold)
                hrs = self._secs_until_pacific_midnight() / 3600
                logger.warning(
                    f"Gemini daily quota: {self._daily_requests}/{self._daily_limit} requests used "
                    f"({pct * 100:.0f}%) — {remaining} remaining, resets in {hrs:.1f}h (midnight Pacific)"
                )

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
        mapped = mapping.get(model, model)
        # If the mapped model is known unavailable, walk the fallback chain
        while mapped in self._unavailable_models:
            fallback = _GEMINI_FALLBACK_CHAIN.get(mapped)
            if fallback is None or fallback == mapped:
                break
            mapped = fallback
        return mapped

    def _handle_model_not_found(self, model: str) -> Optional[str]:
        """Mark model unavailable and return the fallback model ID, or None if no fallback."""
        self._unavailable_models.add(model)
        # Update default_model so future calls skip straight to the fallback
        if self.default_model == model:
            fallback = _GEMINI_FALLBACK_CHAIN.get(model)
            if fallback:
                self.default_model = fallback
                logger.warning(
                    f"Gemini model '{model}' not available on this endpoint — "
                    f"switching to '{fallback}' for all future calls"
                )
                return fallback
        return None

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
            self._last_error = f"quota exhausted (resets in {self._secs_until_pacific_midnight()/3600:.1f}h)"
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
                self._track_daily_request()
                self._last_error = ""
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body_raw = e.response.text or ""
                body = body_raw[:200].replace("{", "(").replace("}", ")")
                if status == 429:
                    retry_after = e.response.headers.get("retry-after", "5")
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 5.0
                    # Distinguish daily quota from per-minute rate limit.
                    # Gemini uses "quota" in ALL 429 bodies, so check for daily-specific keywords.
                    body_lower = body_raw.lower()
                    is_daily = "per-day" in body_lower or "per day" in body_lower or "daily" in body_lower or wait > 120
                    if is_daily:
                        circuit_wait = self._secs_until_pacific_midnight()
                        self._rate_limited_until = time.monotonic() + circuit_wait
                        self._last_error = f"daily quota exhausted — resets in {circuit_wait/3600:.1f}h"
                        logger.warning(f"Gemini daily quota exhausted — circuit-breaking for {circuit_wait/3600:.1f}h (until midnight Pacific): {body}")
                        return ""
                    # Per-minute rate limit — wait and retry
                    wait = min(wait, 30.0)
                    logger.info(f"Gemini per-minute rate limit — waiting {wait:.0f}s before retry")
                    await asyncio.sleep(wait)
                elif any(kw in body_raw.lower() for kw in _GEMINI_MODEL_UNAVAILABLE_KWS):
                    # Model not available on this endpoint (any status code) — try fallback
                    self._last_error = f"model unavailable ({status}): {body[:100]}"
                    fallback = self._handle_model_not_found(model)
                    if fallback:
                        model = fallback
                        payload["model"] = model
                        continue  # retry immediately with fallback model
                    logger.error(f"Gemini model '{model}' not found and no fallback: {body}")
                    return ""
                else:
                    self._last_error = f"HTTP {status}: {body[:120]}"
                    logger.error(f"Gemini HTTP error: {status} {body}")
                    if attempt == self.max_retries - 1:
                        return ""
                    await asyncio.sleep(1)
            except Exception as e:
                self._last_error = str(e)[:120]
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
            self._last_error = f"quota exhausted (resets in {self._secs_until_pacific_midnight()/3600:.1f}h)"
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
                self._track_daily_request()
                text = data["choices"][0]["message"]["content"]
                return _parse_json_response(text)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body_raw = e.response.text or ""
                body = body_raw[:200].replace("{", "(").replace("}", ")")
                if status == 429:
                    retry_after = e.response.headers.get("retry-after", "5")
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 5.0
                    body_lower = body_raw.lower()
                    is_daily = "per-day" in body_lower or "per day" in body_lower or "daily" in body_lower or wait > 120
                    if is_daily:
                        circuit_wait = self._secs_until_pacific_midnight()
                        self._rate_limited_until = time.monotonic() + circuit_wait
                        logger.warning(f"Gemini daily quota exhausted — circuit-breaking for {circuit_wait/3600:.1f}h: {body}")
                        return {}
                    wait = min(wait, 30.0)
                    logger.info(f"Gemini per-minute rate limit — waiting {wait:.0f}s before retry")
                    await asyncio.sleep(wait)
                elif any(kw in body_raw.lower() for kw in _GEMINI_MODEL_UNAVAILABLE_KWS):
                    # Model not available on this endpoint (any status code) — try fallback
                    fallback = self._handle_model_not_found(model)
                    if fallback:
                        model = fallback
                        payload["model"] = model
                        continue  # retry immediately with fallback model
                    logger.error(f"Gemini model '{model}' not found and no fallback: {body}")
                    return {}
                else:
                    logger.error(f"Gemini JSON error: {status} {body}")
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
# Factory — create the right client based on config
# ============================================================

def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
):
    """Create an LLM client based on environment or explicit config.

    Provider detection order:
    1. Explicit provider argument
    2. LLM_PROVIDER env var
    3. Default → NN (local ONNX model, zero cost)
    4. If ANTHROPIC_API_KEY is set → Claude
    5. If GROQ_API_KEY is set → Groq
    6. If GEMINI_API_KEY is set → Gemini
    7. Fallback → Ollama (local)
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "").lower()

    if not provider:
        # Auto-detect: NN first (always available), then cloud providers
        # NN is the default — free, fast, no API key needed.
        provider = PROVIDER_NN

    if provider == PROVIDER_NN:
        from soci.engine.nn_client import NNClient
        return NNClient()
    elif provider == PROVIDER_CLAUDE:
        default_model = model or MODEL_HAIKU
        return ClaudeClient(default_model=default_model)
    elif provider == PROVIDER_GROQ:
        default_model = model or os.environ.get("GROQ_MODEL", MODEL_GROQ_LLAMA_8B)
        return GroqClient(default_model=default_model)
    elif provider == PROVIDER_GEMINI:
        default_model = model or os.environ.get("GEMINI_MODEL", MODEL_GEMINI_FLASH)
        return GeminiClient(default_model=default_model)
    elif provider == PROVIDER_OLLAMA:
        default_model = model or os.environ.get("OLLAMA_MODEL", MODEL_OLLAMA_SOCI)
        return OllamaClient(base_url=ollama_url, default_model=default_model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'nn', 'claude', 'groq', 'gemini', or 'ollama'.")


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
