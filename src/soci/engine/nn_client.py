"""Neural Network LLM client — replaces cloud LLM with local ONNX model.

Downloads soci-agent-nn from HuggingFace Hub on first use, then runs
inference via ONNX Runtime. Zero API calls, zero cost, ~1ms per batch.

Drop-in replacement for GeminiClient/GroqClient — implements the same
complete() / complete_json() interface expected by Simulation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

# ── Domain constants (must match the training notebook exactly) ──────────

ACTION_TYPES = ["move", "work", "eat", "sleep", "talk", "exercise", "shop", "relax", "wander"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

LOCATIONS = [
    # Residential (17)
    "house_elena", "house_marcus", "house_helen", "house_diana", "house_kai",
    "house_priya", "house_james", "house_rosa", "house_yuki", "house_frank",
    "apartment_block_1", "apartment_block_2", "apartment_block_3",
    "apt_northeast", "apt_northwest", "apt_southeast", "apt_southwest",
    # Commercial (8)
    "cafe", "grocery", "bar", "restaurant", "bakery", "cinema", "diner", "pharmacy",
    # Work (5)
    "office", "office_tower", "factory", "school", "hospital",
    # Public (10)
    "park", "gym", "library", "church", "town_square", "sports_field",
    "street_north", "street_south", "street_east", "street_west",
]
LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCATIONS)}

NEED_NAMES = ["hunger", "energy", "social", "purpose", "comfort", "fun"]

ACTION_DURATIONS = {"move": 1, "work": 4, "eat": 2, "sleep": 8, "talk": 2, "exercise": 3, "shop": 2, "relax": 2, "wander": 1}

FEATURE_DIM = 47

# Conversation templates — small set of personality-driven lines so the NN
# provider can produce *something* for conversations without a text model.
_GREETINGS = [
    "Hey, how's it going?", "Morning!", "Hi there.", "What's up?",
    "Hey! Haven't seen you in a while.", "Oh hey, I was just thinking about you.",
]
_SMALLTALK = [
    "Nice weather today, isn't it?", "Been busy lately?",
    "Did you hear about the event in the square?", "I've been meaning to ask you something.",
    "This place is getting crowded.", "I could really use a coffee.",
]
_REPLIES = [
    "Yeah, totally.", "I know what you mean.", "Ha, right?",
    "That's interesting.", "Tell me more.", "I hadn't thought about it that way.",
    "Hmm, I'm not so sure about that.", "Oh really?", "Same here.",
]


# ── Feature encoding ────────────────────────────────────────────────────

def _time_period(hour: int) -> int:
    if hour < 6: return 0
    if hour < 9: return 1
    if hour < 12: return 2
    if hour < 14: return 3
    if hour < 18: return 4
    if hour < 22: return 5
    return 6


def encode_features(
    personality: dict[str, float],
    age: float,
    hour: int,
    minute: int,
    day: int,
    needs: dict[str, float],
    mood: float,
    current_loc: str,
    home_loc: str = "",
    work_loc: str = "",
    num_people_here: int = 0,
) -> np.ndarray:
    """Encode agent state into the 47-dim feature vector the ONNX model expects."""
    f: list[float] = []

    # [0-4] Personality (Big Five, 0-1)
    f.append(personality.get("openness", 5) / 10.0)
    f.append(personality.get("conscientiousness", 5) / 10.0)
    f.append(personality.get("extraversion", 5) / 10.0)
    f.append(personality.get("agreeableness", 5) / 10.0)
    f.append(personality.get("neuroticism", 5) / 10.0)

    # [5] Age
    f.append(age / 100.0)

    # [6-9] Time (cyclical)
    f.append(math.sin(2 * math.pi * hour / 24))
    f.append(math.cos(2 * math.pi * hour / 24))
    f.append(math.sin(2 * math.pi * minute / 60))
    f.append(math.cos(2 * math.pi * minute / 60))

    # [10-11] Day
    dow = (day - 1) % 7
    f.append(dow / 7.0)
    f.append(1.0 if dow >= 5 else 0.0)

    # [12-17] Needs
    for n in NEED_NAMES:
        f.append(needs.get(n, 0.5))

    # [18] Mood
    f.append(max(-1.0, min(1.0, mood)))

    # [19] Most urgent need index
    vals = [needs.get(n, 0.5) for n in NEED_NAMES]
    urgent_idx = int(np.argmin(vals))
    f.append(urgent_idx / 5.0)

    # [20] Has critical need
    f.append(1.0 if any(v < 0.15 for v in vals) else 0.0)

    # [21-24] Location context
    zone = 0 if current_loc.startswith(("house_", "apartment_", "apt_")) else (
        1 if current_loc in ("cafe", "grocery", "bar", "restaurant", "bakery", "cinema", "diner", "pharmacy") else (
        2 if current_loc in ("office", "office_tower", "factory", "school", "hospital") else 3))
    f.append(zone / 3.0)
    f.append(1.0 if current_loc == home_loc else 0.0)
    f.append(1.0 if current_loc == work_loc else 0.0)
    f.append(min(num_people_here / 10.0, 1.0))

    # [25-30] Location type one-hot (6)
    loc_oh = [0.0] * 6
    if zone == 0:
        loc_oh[0] = 1.0
    elif zone == 1:
        loc_oh[1] = 1.0
    elif zone == 2:
        loc_oh[2] = 1.0
    elif current_loc.startswith("street_"):
        loc_oh[4] = 1.0
    else:
        loc_oh[3] = 1.0
    if current_loc == home_loc:
        loc_oh[5] = 1.0
    f.extend(loc_oh)

    # [31-37] Time period one-hot (7)
    tp = [0.0] * 7
    tp[_time_period(hour)] = 1.0
    f.extend(tp)

    # [38-46] Last action one-hot (9) — zeros (no history in this call)
    f.extend([0.0] * 9)

    return np.array([f], dtype=np.float32)


# ── Prompt parsing helpers ──────────────────────────────────────────────

def _extract_persona_from_system(system: str) -> dict:
    """Pull personality traits, age, home/work from the system prompt."""
    info: dict = {}
    # Try to match persona fields
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        # The system prompt includes trait descriptions, not raw numbers.
        # We use defaults; the simulation passes persona objects.
        info[trait] = 5
    # Try age
    age_m = re.search(r"(\d+)-year-old", system)
    if age_m:
        info["age"] = int(age_m.group(1))
    else:
        info["age"] = 30
    return info


def _extract_state_from_user(user_message: str) -> dict:
    """Extract time, location, needs from the user prompt."""
    state: dict = {"hour": 12, "minute": 0, "day": 1, "location": "", "needs": {}, "mood": 0.0}

    # Time: "It is HH:MM on Day N" or "Day N, HH:MM"
    time_m = re.search(r"(\d{1,2}):(\d{2})", user_message)
    if time_m:
        state["hour"] = int(time_m.group(1))
        state["minute"] = int(time_m.group(2))
    day_m = re.search(r"Day\s+(\d+)", user_message)
    if day_m:
        state["day"] = int(day_m.group(1))

    # Location: "at <location_name>" — try to match location IDs
    loc_m = re.search(r"at (\w[\w\s&']+?)[\.\,]", user_message)
    if loc_m:
        loc_name = loc_m.group(1).strip().lower()
        for loc_id in LOCATIONS:
            if loc_id.replace("_", " ") in loc_name or loc_name in loc_id:
                state["location"] = loc_id
                break

    # Needs: "hunger: 0.X" or "Hunger=0.X"
    for need in NEED_NAMES:
        nm = re.search(rf"{need}\s*[=:]\s*([\d.]+)", user_message, re.IGNORECASE)
        if nm:
            state["needs"][need] = float(nm.group(1))

    return state


# ── ONNX Model download ────────────────────────────────────────────────

_DEFAULT_REPO = "RayMelius/soci-agent-nn"
_MODEL_FILENAME = "soci_agent.onnx"


def _download_model(repo_id: str = _DEFAULT_REPO, cache_dir: str = "models") -> str:
    """Download the ONNX model from HuggingFace Hub if not cached."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / _MODEL_FILENAME

    if local_path.exists():
        logger.info(f"NN model cached at {local_path}")
        return str(local_path)

    logger.info(f"Downloading NN model from {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=_MODEL_FILENAME,
            local_dir=str(cache),
        )
        logger.info(f"NN model downloaded to {downloaded}")
        return downloaded
    except ImportError:
        # Fallback: direct HTTP download
        import httpx as _httpx
        url = f"https://huggingface.co/{repo_id}/resolve/main/{_MODEL_FILENAME}"
        logger.info(f"Downloading from {url}")
        resp = _httpx.get(url, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        logger.info(f"NN model saved to {local_path} ({len(resp.content):,} bytes)")
        return str(local_path)


# ── Usage tracker (compatible with other LLM clients) ───────────────────

@dataclass
class NNUsage:
    calls: int = 0
    def summary(self) -> str:
        return f"calls: {self.calls}, $0.00"
    def record(self, *_args, **_kwargs) -> None:
        self.calls += 1


# ── NNClient ────────────────────────────────────────────────────────────

class NNClient:
    """ONNX-based neural network client — drop-in LLM replacement for Soci.

    Downloads soci-agent-nn from HuggingFace Hub on first use.
    Runs inference via ONNX Runtime on CPU (~1ms for 50 agents).
    Zero API calls, zero cost, works offline.
    """

    provider = "nn"
    default_model = "soci-agent-nn"
    llm_status = "active"

    def __init__(self, model_path: Optional[str] = None, repo_id: str = _DEFAULT_REPO):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for the NN provider. "
                "Install it with: pip install onnxruntime"
            )
        self._repo_id = repo_id
        if model_path is None:
            model_path = _download_model(repo_id)
        self._model_path = model_path
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.usage = NNUsage()
        self._last_error = ""
        logger.info(f"NN client loaded: {model_path}")

    def reload(self) -> str:
        """Re-download the ONNX model from HF Hub and reload the session.

        Returns a status message describing what happened.
        """
        local_path = Path(self._model_path)

        # Delete cached model to force re-download
        if local_path.exists():
            old_size = local_path.stat().st_size
            local_path.unlink()
            logger.info(f"Deleted cached model ({old_size:,} bytes)")

        # Re-download
        new_path = _download_model(self._repo_id)
        new_size = Path(new_path).stat().st_size

        # Reload ONNX session
        self.session = ort.InferenceSession(
            new_path,
            providers=["CPUExecutionProvider"],
        )
        self._model_path = new_path

        msg = f"NN model reloaded from {self._repo_id} ({new_size / 1024:.0f} KB)"
        logger.info(msg)
        return msg

    async def complete(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return a JSON string action decision from the NN model."""
        result = await self.complete_json(
            system=system,
            user_message=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return json.dumps(result) if result else ""

    async def complete_json(
        self,
        system: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """Parse prompt context and run the NN to produce an action/conversation/plan.

        Detects the prompt type (action decision, conversation, plan, reflection)
        from the user_message content and routes to the appropriate handler.
        """
        self.usage.record()

        # Detect prompt type and dispatch
        msg_lower = user_message.lower()
        if "plan your day" in msg_lower or "what will you do today" in msg_lower:
            return self._generate_plan(system, user_message)
        elif '"action"' in msg_lower and '"target"' in msg_lower:
            return self._decide_action(system, user_message, temperature)
        elif "how do you respond" in msg_lower or "you decide to start a conversation" in msg_lower:
            return self._generate_conversation(system, user_message)
        elif "reflect on your recent" in msg_lower:
            return self._generate_reflection(system, user_message)
        elif "how important is this" in msg_lower:
            return {"importance": random.randint(3, 7), "reaction": "Interesting."}
        else:
            # Default: treat as action decision
            return self._decide_action(system, user_message, temperature)

    def _decide_action(self, system: str, user_message: str, temperature: float = 0.7) -> dict:
        """Run the ONNX model to select an action."""
        persona = _extract_persona_from_system(system)
        state = _extract_state_from_user(user_message)

        # Default needs if not extracted from prompt
        needs = state["needs"]
        for n in NEED_NAMES:
            if n not in needs:
                needs[n] = 0.5

        features = encode_features(
            personality=persona,
            age=persona.get("age", 30),
            hour=state["hour"],
            minute=state["minute"],
            day=state["day"],
            needs=needs,
            mood=state.get("mood", 0.0),
            current_loc=state.get("location", "town_square"),
            home_loc="",  # Not available from prompt alone
            work_loc="",
            num_people_here=0,
        )

        # Run ONNX inference
        outputs = self.session.run(None, {"features": features})
        action_logits = outputs[0][0]  # (9,)
        location_logits = outputs[1][0]  # (NUM_LOCATIONS,)
        duration_pred = outputs[2][0] if len(outputs) > 2 else 2.0

        # Sample action with temperature
        logits = action_logits / max(temperature, 0.1)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        action_idx = int(np.random.choice(len(ACTION_TYPES), p=probs))
        action = ACTION_TYPES[action_idx]

        # Top location
        loc_idx = int(np.argmax(location_logits))
        target = LOCATIONS[loc_idx] if loc_idx < len(LOCATIONS) else ""

        # Duration
        duration = max(1, min(8, round(float(duration_pred))))
        if action in ACTION_DURATIONS and abs(duration - ACTION_DURATIONS[action]) > 3:
            duration = ACTION_DURATIONS[action]

        return {
            "action": action,
            "target": target,
            "detail": f"NN: {action} at {target}",
            "duration": duration,
            "reasoning": f"NN model (conf: {probs[action_idx]:.0%})",
        }

    def _generate_plan(self, system: str, user_message: str) -> dict:
        """Generate a simple daily plan based on persona and time."""
        persona = _extract_persona_from_system(system)
        state = _extract_state_from_user(user_message)

        # Build a sensible plan based on time/personality
        plan = []
        if state["hour"] <= 8:
            plan.append("Have breakfast")
        plan.append("Head to work")
        plan.append("Work through the morning")
        plan.append("Lunch break")
        plan.append("Afternoon work session")

        # Personality-driven evening
        E = persona.get("extraversion", 5)
        if E >= 7:
            plan.append("Meet friends for dinner")
            plan.append("Go to the bar")
        elif E >= 4:
            plan.append("Dinner at a restaurant")
            plan.append("Relaxing walk in the park")
        else:
            plan.append("Quiet dinner at home")
            plan.append("Read a book")

        plan.append("Get some sleep")

        return {"plan": plan, "reasoning": "NN-generated daily plan"}

    def _generate_conversation(self, system: str, user_message: str) -> dict:
        """Generate a conversation turn."""
        # Detect if this is initiation or continuation
        if "you decide to start a conversation" in user_message.lower():
            return {
                "message": random.choice(_GREETINGS),
                "inner_thought": "Let's see what they're up to.",
                "topic": random.choice(["catching up", "the weather", "what's new", "plans"]),
            }
        else:
            # Continuation — respond to what was said
            return {
                "message": random.choice(_REPLIES + _SMALLTALK),
                "inner_thought": "Interesting conversation.",
                "sentiment_delta": round(random.uniform(-0.02, 0.05), 3),
                "trust_delta": round(random.uniform(-0.01, 0.03), 3),
            }

    def _generate_reflection(self, system: str, user_message: str) -> dict:
        """Generate a reflection with mood shift."""
        reflections = [
            "Things have been going well lately.",
            "I should spend more time doing what I enjoy.",
            "The people around me make this place feel like home.",
        ]
        return {
            "reflections": random.sample(reflections, k=min(2, len(reflections))),
            "mood_shift": round(random.uniform(-0.1, 0.15), 2),
            "reasoning": "Reflecting on recent experiences.",
        }
