#!/usr/bin/env python3
"""Soci Agent NN — Local Training Script

Equivalent to notebooks/soci_agent_nn.ipynb but runs as a standalone script.
Trains the SociAgentTransformer, exports to ONNX, and optionally pushes to HF Hub.

Usage:
    python scripts/nn_train.py                          # Train from scratch (synthetic data)
    python scripts/nn_train.py --data data/nn_training  # Train on collected + synthetic data
    python scripts/nn_train.py --push                   # Train and push to HF Hub
    python scripts/nn_train.py --epochs 50 --lr 1e-4    # Custom hyperparameters
    python scripts/nn_train.py --resume                 # Resume from existing weights

Requires: pip install torch onnx onnxruntime numpy huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("nn_train")

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "nn_training"
SAMPLES_FILE = DATA_DIR / "collected_samples.jsonl"

# ══════════════════════════════════════════════════════════════════════════
# 1. Domain Constants — must match the Soci simulation
# ══════════════════════════════════════════════════════════════════════════

ACTION_TYPES = ["move", "work", "eat", "sleep", "talk", "exercise", "shop", "relax", "wander"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
NUM_ACTIONS = len(ACTION_TYPES)

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
    # Public (8)
    "park", "gym", "library", "church", "town_square", "sports_field",
    "street_north", "street_south", "street_east", "street_west",
]
LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCATIONS)}
NUM_LOCATIONS = len(LOCATIONS)

# Zone encoding
LOC_ZONE = {}
for _loc in LOCATIONS:
    if _loc.startswith(("house_", "apartment_", "apt_")):
        LOC_ZONE[_loc] = 0
    elif _loc in ("cafe", "grocery", "bar", "restaurant", "bakery", "cinema", "diner", "pharmacy"):
        LOC_ZONE[_loc] = 1
    elif _loc in ("office", "office_tower", "factory", "school", "hospital"):
        LOC_ZONE[_loc] = 2
    else:
        LOC_ZONE[_loc] = 3

ACTION_NEEDS = {
    "work":     {"purpose": 0.3},
    "eat":      {"hunger": 0.5},
    "sleep":    {"energy": 0.6},
    "talk":     {"social": 0.3},
    "exercise": {"energy": -0.1, "fun": 0.2, "comfort": 0.1},
    "shop":     {"hunger": 0.1, "comfort": 0.1},
    "relax":    {"energy": 0.1, "fun": 0.2, "comfort": 0.2},
    "wander":   {"fun": 0.1},
    "move":     {},
}

ACTION_DURATIONS = {"move": 1, "work": 4, "eat": 2, "sleep": 8, "talk": 2, "exercise": 3, "shop": 2, "relax": 2, "wander": 1}
NEED_NAMES = ["hunger", "energy", "social", "purpose", "comfort", "fun"]
PERSONALITY_NAMES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

NUM_TIME_PERIODS = 7
FEATURE_DIM = 47


# ══════════════════════════════════════════════════════════════════════════
# 2. Personas — 20 Soci characters (from personas.yaml)
# ══════════════════════════════════════════════════════════════════════════

PERSONAS = [
    # House 1 — Elena & Lila (roommates)
    {"id": "elena",  "name": "Elena Vasquez",    "age": 34, "gender": "female",    "occ": "software engineer",
     "O": 8, "C": 7, "E": 4, "A": 6, "N": 5, "home": "house_elena", "work": "office",
     "tags": ["freelance", "introvert", "tech"],
     "hangouts": ["cafe", "library"],           # where she goes to think/work remotely
     "routine_bias": {}},

    {"id": "lila",   "name": "Lila Santos",      "age": 33, "gender": "female",    "occ": "artist",
     "O": 10, "C": 3, "E": 6, "A": 7, "N": 7, "home": "house_elena", "work": "library",
     "tags": ["creative", "emotional", "crush_elena"],
     "hangouts": ["park", "cafe", "library"],   # paints outdoors, hangs near Elena
     "routine_bias": {"relax": 0.15, "wander": 0.10}},

    # House 2 — Marcus & Zoe (siblings)
    {"id": "marcus", "name": "Marcus Chen",      "age": 28, "gender": "male",      "occ": "fitness trainer",
     "O": 5, "C": 8, "E": 9, "A": 7, "N": 3, "home": "house_marcus", "work": "gym",
     "tags": ["athletic", "extrovert", "community"],
     "hangouts": ["park", "sports_field", "cafe"],
     "routine_bias": {"exercise": 0.20, "talk": 0.10}},

    {"id": "zoe",    "name": "Zoe Chen-Williams", "age": 19, "gender": "female",   "occ": "college student",
     "O": 8, "C": 4, "E": 8, "A": 6, "N": 7, "home": "house_marcus", "work": "library",
     "tags": ["student", "social_media", "young"],
     "hangouts": ["cafe", "cinema", "park", "town_square"],
     "routine_bias": {"talk": 0.15, "wander": 0.10}},

    # House 3 — Helen & Alice (close friends)
    {"id": "helen",  "name": "Helen Park",       "age": 67, "gender": "female",    "occ": "retired teacher",
     "O": 6, "C": 8, "E": 6, "A": 8, "N": 4, "home": "house_helen", "work": "library",
     "tags": ["retired", "bookworm", "widow"],
     "hangouts": ["library", "park", "bakery", "church"],
     "routine_bias": {"relax": 0.15}},

    {"id": "alice",  "name": "Alice Fontaine",   "age": 58, "gender": "female",    "occ": "retired accountant",
     "O": 5, "C": 8, "E": 6, "A": 8, "N": 3, "home": "house_helen", "work": "bakery",
     "tags": ["retired", "baker", "nurturing"],
     "hangouts": ["bakery", "grocery", "church"],
     "routine_bias": {"work": 0.10}},   # loves baking, spends extra time at bakery

    # House 4 — Diana & Marco (mother & son)
    {"id": "diana",  "name": "Diana Novak",      "age": 41, "gender": "female",    "occ": "grocery store owner",
     "O": 4, "C": 9, "E": 5, "A": 6, "N": 7, "home": "house_diana", "work": "grocery",
     "tags": ["business_owner", "single_mother", "protective"],
     "hangouts": ["grocery"],                   # rarely leaves the store
     "routine_bias": {"work": 0.20}},

    {"id": "marco",  "name": "Marco Delgado",    "age": 16, "gender": "male",      "occ": "high school student",
     "O": 7, "C": 4, "E": 6, "A": 5, "N": 6, "home": "house_diana", "work": "school",
     "tags": ["student", "teen", "gamer"],
     "hangouts": ["park", "cinema", "cafe", "sports_field"],
     "routine_bias": {"relax": 0.10, "wander": 0.10}},

    # House 5 — Kai (lives alone)
    {"id": "kai",    "name": "Kai Okonkwo",      "age": 22, "gender": "nonbinary", "occ": "barista",
     "O": 9, "C": 3, "E": 7, "A": 5, "N": 6, "home": "house_kai", "work": "cafe",
     "tags": ["musician", "creative", "dropout"],
     "hangouts": ["bar", "park", "town_square"],  # plays music, socializes
     "routine_bias": {"relax": 0.10, "talk": 0.10}},

    # House 6 — Priya & Nina (flatmates)
    {"id": "priya",  "name": "Priya Sharma",     "age": 38, "gender": "female",    "occ": "doctor",
     "O": 7, "C": 9, "E": 5, "A": 8, "N": 6, "home": "house_priya", "work": "hospital",
     "tags": ["overworked", "caring", "guilt"],
     "hangouts": ["hospital", "pharmacy"],      # rarely leaves work orbit
     "routine_bias": {"work": 0.25}},           # long hospital hours

    {"id": "nina",   "name": "Nina Volkov",      "age": 29, "gender": "female",    "occ": "real estate agent",
     "O": 5, "C": 8, "E": 9, "A": 4, "N": 5, "home": "house_priya", "work": "office",
     "tags": ["ambitious", "networker", "suspicious"],
     "hangouts": ["cafe", "restaurant", "office_tower"],
     "routine_bias": {"talk": 0.15, "work": 0.10}},

    # House 7 — James & Theo (housemates)
    {"id": "james",  "name": "James O'Brien",    "age": 55, "gender": "male",      "occ": "bar owner",
     "O": 5, "C": 6, "E": 8, "A": 7, "N": 4, "home": "house_james", "work": "bar",
     "tags": ["social_hub", "divorced", "storyteller"],
     "hangouts": ["bar"],                       # his whole life revolves around the bar
     "routine_bias": {"talk": 0.20}},

    {"id": "theo",   "name": "Theo Blackwood",   "age": 45, "gender": "male",      "occ": "construction worker",
     "O": 3, "C": 7, "E": 4, "A": 5, "N": 5, "home": "house_james", "work": "factory",
     "tags": ["blue_collar", "stoic", "handy"],
     "hangouts": ["bar", "diner"],              # bar after work
     "routine_bias": {"work": 0.15}},

    # House 8 — Rosa & Omar
    {"id": "rosa",   "name": "Rosa Martelli",    "age": 62, "gender": "female",    "occ": "restaurant owner",
     "O": 6, "C": 9, "E": 7, "A": 8, "N": 5, "home": "house_rosa", "work": "restaurant",
     "tags": ["nurturing", "italian", "community_mother"],
     "hangouts": ["restaurant", "grocery"],     # buys ingredients, feeds everyone
     "routine_bias": {"work": 0.20, "eat": 0.05}},

    {"id": "omar",   "name": "Omar Hassan",      "age": 50, "gender": "male",      "occ": "taxi driver",
     "O": 6, "C": 6, "E": 7, "A": 7, "N": 4, "home": "house_rosa", "work": "restaurant",
     "tags": ["immigrant", "philosophical", "hardworking"],
     "hangouts": ["restaurant", "cafe", "park"],
     "routine_bias": {"wander": 0.15}},         # drives around town = wander

    # House 9 — Yuki & Devon (flatmates)
    {"id": "yuki",   "name": "Yuki Tanaka",      "age": 26, "gender": "female",    "occ": "yoga instructor",
     "O": 8, "C": 6, "E": 5, "A": 9, "N": 3, "home": "house_yuki", "work": "gym",
     "tags": ["mindful", "calm", "empathetic"],
     "hangouts": ["park", "gym", "library"],    # meditates in park
     "routine_bias": {"exercise": 0.15, "relax": 0.10}},

    {"id": "devon",  "name": "Devon Reeves",     "age": 30, "gender": "male",      "occ": "freelance journalist",
     "O": 9, "C": 5, "E": 6, "A": 4, "N": 6, "home": "house_yuki", "work": "office",
     "tags": ["investigative", "paranoid", "curious"],
     "hangouts": ["cafe", "bar", "library", "town_square"],  # interviews, research
     "routine_bias": {"wander": 0.15, "talk": 0.10}},

    # House 10 — Frank, George & Sam
    {"id": "frank",  "name": "Frank Kowalski",   "age": 72, "gender": "male",      "occ": "retired mechanic",
     "O": 3, "C": 7, "E": 5, "A": 4, "N": 5, "home": "house_frank", "work": "bar",
     "tags": ["retired", "cantankerous", "creature_of_habit"],
     "hangouts": ["bar", "diner"],              # same bar stool every night
     "routine_bias": {"relax": 0.15}},

    {"id": "george", "name": "George Adeyemi",   "age": 47, "gender": "male",      "occ": "night shift security",
     "O": 4, "C": 7, "E": 3, "A": 6, "N": 4, "home": "house_frank", "work": "factory",
     "tags": ["night_shift", "widower", "observant"],
     "hangouts": ["park"],                      # naps in park during day
     "routine_bias": {}},                       # schedule handled by night_shift tag

    {"id": "sam",    "name": "Sam Nakamura",     "age": 40, "gender": "nonbinary", "occ": "librarian",
     "O": 7, "C": 8, "E": 3, "A": 7, "N": 4, "home": "house_frank", "work": "library",
     "tags": ["quiet", "bookish", "inclusive"],
     "hangouts": ["library", "park", "cafe"],
     "routine_bias": {"work": 0.10, "relax": 0.05}},
]


# ══════════════════════════════════════════════════════════════════════════
# 3. Feature Encoding
# ══════════════════════════════════════════════════════════════════════════

def _time_period(hour: int) -> int:
    if hour < 6: return 0
    if hour < 9: return 1
    if hour < 12: return 2
    if hour < 14: return 3
    if hour < 18: return 4
    if hour < 22: return 5
    return 6


def encode_features(
    persona: dict, hour: int, minute: int, day: int,
    needs: dict, mood: float, current_loc: str,
    num_people_here: int = 0,
) -> list[float]:
    """Encode agent state into 47-dim feature vector."""
    f: list[float] = []
    # Personality (5)
    f.append(persona.get("O", persona.get("openness", 5)) / 10.0)
    f.append(persona.get("C", persona.get("conscientiousness", 5)) / 10.0)
    f.append(persona.get("E", persona.get("extraversion", 5)) / 10.0)
    f.append(persona.get("A", persona.get("agreeableness", 5)) / 10.0)
    f.append(persona.get("N", persona.get("neuroticism", 5)) / 10.0)
    # Age (1)
    f.append(persona.get("age", 30) / 100.0)
    # Time cyclical (4)
    f.append(math.sin(2 * math.pi * hour / 24))
    f.append(math.cos(2 * math.pi * hour / 24))
    f.append(math.sin(2 * math.pi * minute / 60))
    f.append(math.cos(2 * math.pi * minute / 60))
    # Day (2)
    dow = ((day - 1) % 7)
    f.append(dow / 7.0)
    f.append(1.0 if dow >= 5 else 0.0)
    # Needs (6)
    for n in NEED_NAMES:
        f.append(needs.get(n, 0.5))
    # Mood (1)
    f.append(max(-1.0, min(1.0, mood)))
    # Urgency (2)
    vals = [needs.get(n, 0.5) for n in NEED_NAMES]
    urgent_idx = int(np.argmin(vals))
    f.append(urgent_idx / 5.0)
    f.append(1.0 if any(v < 0.15 for v in vals) else 0.0)
    # Location zone (1)
    zone = LOC_ZONE.get(current_loc, 3)
    f.append(zone / 3.0)
    # Home/work flags (2)
    home = persona.get("home", persona.get("home_location", ""))
    work = persona.get("work", persona.get("work_location", ""))
    f.append(1.0 if current_loc == home else 0.0)
    f.append(1.0 if current_loc == work else 0.0)
    # People density (1)
    f.append(min(num_people_here / 10.0, 1.0))
    # Location type one-hot (6)
    loc_oh = [0.0] * 6
    if current_loc.startswith(("house_", "apartment_", "apt_")):
        loc_oh[0] = 1.0
    elif zone == 1:
        loc_oh[1] = 1.0
    elif zone == 2:
        loc_oh[2] = 1.0
    elif current_loc.startswith("street_"):
        loc_oh[4] = 1.0
    else:
        loc_oh[3] = 1.0
    if current_loc == home:
        loc_oh[5] = 1.0
    f.extend(loc_oh)
    # Time period one-hot (7)
    tp = [0.0] * NUM_TIME_PERIODS
    tp[_time_period(hour)] = 1.0
    f.extend(tp)
    # Last action one-hot (9) — random for synthetic, zeros for real
    last_action_oh = [0.0] * NUM_ACTIONS
    if random.random() < 0.8:
        last_action_oh[random.randint(0, NUM_ACTIONS - 1)] = 1.0
    f.extend(last_action_oh)
    return f


# ══════════════════════════════════════════════════════════════════════════
# 4. Synthetic Data Generator
# ══════════════════════════════════════════════════════════════════════════

def _is_night_shift(persona: dict) -> bool:
    return "night_shift" in persona.get("tags", [])


def _is_retired(persona: dict) -> bool:
    return "retired" in persona.get("tags", [])


def _is_student(persona: dict) -> bool:
    return "student" in persona.get("tags", [])


def _persona_hangout(persona: dict, fallbacks: list[str]) -> str:
    """Pick a location the persona naturally gravitates toward."""
    hangouts = persona.get("hangouts", [])
    if hangouts and random.random() < 0.6:
        return random.choice(hangouts)
    return random.choice(fallbacks)


def _apply_routine_bias(persona: dict, action: str | None) -> str | None:
    """Probabilistically override action based on persona routine_bias."""
    bias = persona.get("routine_bias", {})
    for biased_action, prob in bias.items():
        if random.random() < prob:
            return biased_action
    return action


def _generate_needs_for_persona(persona: dict, hour: int) -> dict:
    """Generate needs influenced by persona lifestyle, not purely random."""
    needs = {}
    tags = persona.get("tags", [])
    is_night = _is_night_shift(persona)

    for n in NEED_NAMES:
        # Base: 15% chance critical, else moderate-to-full
        if random.random() < 0.15:
            needs[n] = round(random.uniform(0.0, 0.2), 2)
        else:
            needs[n] = round(random.uniform(0.2, 1.0), 2)

    # Persona-specific need tendencies
    if "overworked" in tags:
        # Priya: chronically low energy, low social
        needs["energy"] = round(min(needs["energy"], random.uniform(0.1, 0.5)), 2)
        needs["social"] = round(min(needs["social"], random.uniform(0.1, 0.5)), 2)
    if "athletic" in tags:
        # Marcus: high energy baseline, low fun without exercise
        needs["energy"] = round(max(needs["energy"], random.uniform(0.5, 0.9)), 2)
    if "emotional" in tags:
        # Lila: volatile needs
        swing = random.choice(NEED_NAMES)
        needs[swing] = round(random.uniform(0.0, 0.3), 2)
    if "creature_of_habit" in tags:
        # Frank: stable moderate needs
        for n in NEED_NAMES:
            needs[n] = round(needs[n] * 0.7 + 0.2, 2)
    if is_night:
        # George: energy inverted — tired during day, awake at night
        if 6 <= hour <= 18:
            needs["energy"] = round(min(needs["energy"], random.uniform(0.05, 0.35)), 2)
        else:
            needs["energy"] = round(max(needs["energy"], random.uniform(0.5, 0.9)), 2)
    if "student" in tags:
        # Students: higher social need, lower purpose
        needs["social"] = round(max(needs["social"], random.uniform(0.3, 0.7)), 2)
        needs["fun"] = round(max(needs["fun"], random.uniform(0.2, 0.5)), 2)
    if "nurturing" in tags or "community_mother" in tags:
        # Rosa, Alice: high comfort, purpose from feeding/helping others
        needs["purpose"] = round(max(needs["purpose"], random.uniform(0.4, 0.8)), 2)
    if "mindful" in tags:
        # Yuki: generally balanced, rarely critical
        for n in NEED_NAMES:
            needs[n] = round(max(needs[n], 0.2), 2)

    return needs


def _mood_for_persona(persona: dict, needs: dict) -> float:
    """Generate mood influenced by personality and current needs."""
    tags = persona.get("tags", [])
    # Base mood from needs average
    avg_need = sum(needs.values()) / len(needs)
    base_mood = (avg_need - 0.5) * 2  # maps 0-1 to -1..+1

    # Neuroticism makes mood more volatile
    n_factor = persona.get("N", 5) / 10.0
    volatility = random.uniform(-0.5, 0.5) * n_factor
    base_mood += volatility

    if "calm" in tags or "mindful" in tags:
        base_mood = base_mood * 0.6 + 0.2  # dampen toward positive
    if "emotional" in tags:
        base_mood += random.uniform(-0.4, 0.4)

    return round(max(-1.0, min(1.0, base_mood)), 2)


def _starting_location(persona: dict, hour: int, is_weekend: bool) -> str:
    """Pick a realistic starting location based on time and persona."""
    tags = persona.get("tags", [])
    is_night = _is_night_shift(persona)
    period = _time_period(hour)

    # Night shift workers: at work during night, home during day
    if is_night:
        if period in (0, 6):  # late night / night — at work
            return persona["work"]
        elif period in (1, 2):  # morning — heading home or sleeping
            return random.choice([persona["home"], persona["work"]])
        else:  # daytime — at home (sleeping) or park (napping)
            return random.choice([persona["home"], "park"] if random.random() < 0.7
                                 else [persona["home"]])

    # Normal schedule
    if period == 0:  # late night — home
        return persona["home"]
    elif period == 1:  # early morning — home or commuting
        return random.choice([persona["home"], persona["work"]])
    elif period in (2, 4) and not is_weekend:  # working hours
        if _is_retired(persona):
            return random.choice([persona["home"]] + persona.get("hangouts", ["park"]))
        if _is_student(persona):
            return random.choice([persona["work"], "library", persona["home"]])
        return random.choice([persona["work"], persona["work"], persona["work"],
                              _persona_hangout(persona, ["cafe"])])
    elif period == 3:  # lunch
        return random.choice([persona["work"], "cafe", "restaurant", "diner", "park"])
    elif period == 5:  # evening
        return random.choice([persona["home"], _persona_hangout(persona, ["bar", "cafe", "park"])])
    elif period == 6:  # night
        return random.choice([persona["home"], persona["home"], _persona_hangout(persona, ["bar"])])

    return persona["home"]


def generate_action_example(persona: dict) -> dict:
    """Generate one training example with persona-aware rule-based labels."""
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])
    day = random.randint(1, 30)
    is_weekend = ((day - 1) % 7) >= 5
    tags = persona.get("tags", [])
    is_night = _is_night_shift(persona)

    needs = _generate_needs_for_persona(persona, hour)
    mood = _mood_for_persona(persona, needs)
    current_loc = _starting_location(persona, hour, is_weekend)

    # --- Determine action using rule-based logic ---
    # Priority 1: Critical needs
    urgent = [(n, v) for n, v in needs.items() if v < 0.15]
    urgent.sort(key=lambda x: x[1])

    action = None
    target_loc = current_loc
    duration = 1

    if urgent:
        need_name = urgent[0][0]
        if need_name == "hunger":
            action = "eat"
            # Persona-aware eating locations
            eat_locs = ["cafe", "restaurant", "grocery", "bakery", "diner", persona["home"]]
            if "community_mother" in tags:  # Rosa eats at her restaurant
                eat_locs = ["restaurant", persona["home"]]
            elif "baker" in tags:  # Alice eats at bakery or home
                eat_locs = ["bakery", persona["home"]]
            target_loc = random.choice(eat_locs)
            duration = 2
        elif need_name == "energy":
            action = "sleep"
            target_loc = persona["home"]
            duration = random.choice([4, 6, 8])
        elif need_name == "social":
            action = "talk"
            social_locs = ["cafe", "bar", "park", "town_square", current_loc]
            if "social_hub" in tags:  # James talks at his bar
                social_locs = ["bar", "bar", "restaurant", "park"]
            elif "networker" in tags:  # Nina networks everywhere
                social_locs = ["cafe", "restaurant", "office", "office_tower"]
            target_loc = random.choice(social_locs)
            duration = 2
        elif need_name == "purpose":
            action = "work"
            target_loc = persona["work"]
            duration = 4
        elif need_name == "comfort":
            action = "relax"
            target_loc = random.choice([persona["home"], "park", "library"])
            duration = 2
        elif need_name == "fun":
            action = random.choice(["relax", "exercise", "wander"])
            fun_locs = ["park", "gym", "cinema", "bar", "sports_field"]
            if "teen" in tags or "student" in tags:
                fun_locs = ["cinema", "park", "cafe", "sports_field", "town_square"]
            target_loc = random.choice(fun_locs)
            duration = 2

    # Priority 2: Night shift inverted schedule (George)
    if action is None and is_night:
        period = _time_period(hour)
        if period in (0, 6):  # night — George is at work
            action = "work"
            target_loc = persona["work"]
            duration = 4
        elif period == 1:  # early morning — heading home
            action = "move"
            target_loc = persona["home"]
            duration = 1
        elif period in (2, 3):  # day — sleeping
            if needs["energy"] < 0.6:
                action = "sleep"
                target_loc = persona["home"]
                duration = random.choice([4, 6, 8])
            else:
                # Sometimes naps in park
                action = "relax"
                target_loc = random.choice([persona["home"], "park"])
                duration = 2
        elif period in (4, 5):  # afternoon/evening — wake up, eat, prep for work
            r = random.random()
            if needs["hunger"] < 0.5:
                action = "eat"
                target_loc = random.choice(["diner", "restaurant", persona["home"]])
                duration = 2
            elif r < 0.3:
                action = "talk"
                target_loc = random.choice(["park", "cafe"])
                duration = 2
            else:
                action = "move"
                target_loc = persona["work"]
                duration = 1

    # Priority 3: Persona-specific behavioral patterns
    if action is None:
        period = _time_period(hour)

        # Frank: same bar stool every evening/night
        if persona["id"] == "frank" and period in (5, 6):
            if random.random() < 0.7:
                action = "relax"
                target_loc = "bar"
                duration = 3

        # Lila: gravitates toward Elena (crush) — seeks her hangouts
        elif persona["id"] == "lila" and random.random() < 0.15:
            action = random.choice(["wander", "talk", "relax"])
            target_loc = random.choice(["house_elena", "cafe", "library", "office"])
            duration = 2

        # Rosa: spends mornings buying ingredients, cooks all day
        elif persona["id"] == "rosa" and period in (1, 2):
            if random.random() < 0.4:
                action = "shop"
                target_loc = "grocery"
                duration = 2

        # Devon: investigative journalist, wanders and interviews
        elif persona["id"] == "devon" and period in (2, 4):
            if random.random() < 0.3:
                action = random.choice(["wander", "talk"])
                target_loc = random.choice(["cafe", "bar", "town_square", "library", "park"])
                duration = 2

        # Omar: taxi driver — wanders the streets during work hours
        elif persona["id"] == "omar" and period in (2, 3, 4) and not is_weekend:
            if random.random() < 0.5:
                action = "wander"
                target_loc = random.choice(["street_north", "street_south", "street_east", "street_west",
                                            "town_square", "cafe", "restaurant"])
                duration = 2

        # Diana: barely leaves the grocery store on weekdays
        elif persona["id"] == "diana" and not is_weekend and period in (2, 3, 4):
            if random.random() < 0.7:
                action = "work"
                target_loc = "grocery"
                duration = 4

        # Marcus: morning exercise is sacred
        elif persona["id"] == "marcus" and period == 1:
            if random.random() < 0.6:
                action = "exercise"
                target_loc = random.choice(["gym", "park", "sports_field"])
                duration = 3

        # Yuki: morning meditation/yoga
        elif persona["id"] == "yuki" and period == 1:
            if random.random() < 0.5:
                action = "exercise"
                target_loc = random.choice(["park", "gym"])
                duration = 3

    # Priority 4: Apply routine_bias override
    if action is None:
        biased = _apply_routine_bias(persona, None)
        if biased:
            action = biased
            target_loc = _persona_hangout(persona, ["park", "cafe", persona["home"]])
            duration = 2

    # Priority 5: General time-of-day patterns (fallback)
    if action is None:
        period = _time_period(hour)

        if period == 0:  # Late night
            action = "sleep"
            target_loc = persona["home"]
            duration = 8

        elif period == 1:  # Early morning
            r = random.random()
            if needs["hunger"] < 0.5:
                action = "eat"
                target_loc = random.choice(["cafe", "bakery", persona["home"]])
                duration = 2
            elif r < 0.3 and persona["E"] >= 6:
                action = "exercise"
                target_loc = random.choice(["gym", "park", "sports_field"])
                duration = 3
            else:
                action = "move"
                target_loc = persona["work"]
                duration = 1

        elif period in (2, 4):  # Mid-morning / Afternoon
            if is_weekend:
                r = random.random()
                if _is_retired(persona):
                    # Retired: relaxed weekend routine
                    if r < 0.35:
                        action = "relax"
                        target_loc = _persona_hangout(persona, ["park", "library", persona["home"]])
                    elif r < 0.55:
                        action = "talk"
                        target_loc = _persona_hangout(persona, ["cafe", "park", "church"])
                    elif r < 0.7:
                        action = "shop"
                        target_loc = random.choice(["grocery", "pharmacy", "bakery"])
                    else:
                        action = "wander"
                        target_loc = random.choice(["park", "town_square", "street_north"])
                    duration = random.choice([2, 3])
                elif _is_student(persona):
                    # Students: social weekends
                    if r < 0.3:
                        action = "talk"
                        target_loc = random.choice(["cafe", "park", "cinema", "town_square"])
                    elif r < 0.5:
                        action = "relax"
                        target_loc = random.choice(["cinema", "park", persona["home"]])
                    elif r < 0.65:
                        action = "exercise"
                        target_loc = random.choice(["gym", "park", "sports_field"])
                    elif r < 0.8:
                        action = "wander"
                        target_loc = random.choice(["town_square", "street_north", "street_south"])
                    else:
                        action = "shop"
                        target_loc = random.choice(["grocery", "pharmacy"])
                    duration = random.choice([2, 3])
                else:
                    if r < 0.25:
                        action = "relax"
                        target_loc = _persona_hangout(persona, ["park", "cafe", "library", persona["home"]])
                    elif r < 0.45 and persona["E"] >= 6:
                        action = "talk"
                        target_loc = _persona_hangout(persona, ["cafe", "park", "town_square"])
                    elif r < 0.6:
                        action = "shop"
                        target_loc = random.choice(["grocery", "pharmacy"])
                    elif r < 0.8:
                        action = "exercise"
                        target_loc = random.choice(["gym", "park", "sports_field"])
                    else:
                        action = "wander"
                        target_loc = random.choice(["park", "town_square", "street_north", "street_south"])
                    duration = random.choice([2, 3])
            else:
                # Weekday work hours
                work_prob = 0.5 + persona["C"] * 0.05
                # Business owners and doctors work even harder
                if "business_owner" in tags or persona["occ"] == "doctor":
                    work_prob += 0.15
                if _is_retired(persona):
                    work_prob = 0.15  # retired people rarely "work"
                if random.random() < work_prob:
                    action = "work"
                    target_loc = persona["work"]
                    duration = 4
                else:
                    action = random.choice(["wander", "relax", "talk"])
                    target_loc = _persona_hangout(persona, ["cafe", "park", "town_square"])
                    duration = 2

        elif period == 3:  # Midday / lunch
            if needs["hunger"] < 0.6:
                action = "eat"
                lunch_locs = ["cafe", "restaurant", "bakery", "diner", "park"]
                # People eat near their workplace
                if current_loc == persona["work"]:
                    lunch_locs = ["cafe", "restaurant", "diner", "bakery"]
                target_loc = random.choice(lunch_locs)
                duration = 2
            else:
                action = "relax"
                target_loc = random.choice(["park", "cafe"])
                duration = 1

        elif period == 5:  # Evening
            r = random.random()
            social_bias = persona["E"] / 10.0
            if r < social_bias * 0.5:
                action = "talk"
                evening_social = ["bar", "restaurant", "park", "cafe"]
                if "social_hub" in tags:
                    evening_social = ["bar", "bar", "restaurant"]
                target_loc = random.choice(evening_social)
                duration = 2
            elif r < 0.4:
                action = "eat"
                target_loc = random.choice(["restaurant", "bar", "diner", persona["home"]])
                duration = 2
            elif r < 0.55:
                action = "exercise"
                target_loc = random.choice(["gym", "park", "sports_field"])
                duration = 3
            elif r < 0.7:
                action = "relax"
                target_loc = _persona_hangout(persona, ["cinema", "bar", persona["home"], "library"])
                duration = 2
            else:
                action = "relax"
                target_loc = persona["home"]
                duration = 2

        elif period == 6:  # Night
            if needs["energy"] < 0.4:
                action = "sleep"
                target_loc = persona["home"]
                duration = 8
            else:
                action = "relax"
                target_loc = persona["home"]
                duration = 2

    # 30% chance of picking "move" if target != current
    if target_loc != current_loc and action != "move":
        if random.random() < 0.3:
            action = "move"
            duration = 1

    # Retired and elderly people do shorter activities
    if _is_retired(persona) and duration > 3 and action not in ("sleep", "work"):
        duration = min(duration, 3)

    # Teens/students have shorter attention spans for non-social activities
    if _is_student(persona) and action in ("relax", "work") and random.random() < 0.3:
        duration = max(1, duration - 1)

    features = encode_features(
        persona=persona, hour=hour, minute=minute, day=day,
        needs=needs, mood=mood, current_loc=current_loc,
        num_people_here=random.randint(0, 8),
    )

    return {
        "features": features,
        "action_idx": ACTION_TO_IDX[action],
        "target_loc_idx": LOC_TO_IDX.get(target_loc, 0),
        "duration": min(max(duration, 1), 8),
    }


def generate_dataset(n: int) -> list[dict]:
    """Generate n synthetic training examples."""
    data = []
    for _ in range(n):
        persona = random.choice(PERSONAS)
        data.append(generate_action_example(persona))
    return data


# ══════════════════════════════════════════════════════════════════════════
# 5. Model Architecture — SociAgentTransformer
# ══════════════════════════════════════════════════════════════════════════

def build_model():
    """Build the SociAgentTransformer model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FeatureTokenizer(nn.Module):
        GROUPS = [
            ("personality", 0, 6),
            ("time", 6, 12),
            ("needs", 12, 21),
            ("location", 21, 31),
            ("time_period", 31, 38),
            ("last_action", 38, 47),
        ]

        def __init__(self, d_model: int):
            super().__init__()
            self.projections = nn.ModuleList()
            for name, start, end in self.GROUPS:
                self.projections.append(nn.Sequential(
                    nn.Linear(end - start, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                ))
            self.pos_embed = nn.Parameter(torch.randn(1, len(self.GROUPS), d_model) * 0.02)

        def forward(self, features):
            tokens = []
            for i, (_, start, end) in enumerate(self.GROUPS):
                tokens.append(self.projections[i](features[:, start:end]))
            tokens = torch.stack(tokens, dim=1)
            return tokens + self.pos_embed

    class MoEFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.gate = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
                for _ in range(num_experts)
            ])

        def forward(self, x):
            B, S, D = x.shape
            gate_probs = F.softmax(self.gate(x), dim=-1)
            top_k_probs, top_k_idx = gate_probs.topk(self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                eidx = top_k_idx[:, :, k]
                w = top_k_probs[:, :, k].unsqueeze(-1)
                for e in range(self.num_experts):
                    mask = (eidx == e).unsqueeze(-1)
                    if mask.any():
                        output = output + mask.float() * w * self.experts[e](x)
            return output

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, nhead, d_ff, num_experts=4, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.moe_ff = MoEFeedForward(d_model, d_ff, num_experts)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.moe_ff(x)
            return self.norm2(x + self.dropout(ff_out))

    class SociAgentTransformer(nn.Module):
        def __init__(self, d_model=128, nhead=8, num_layers=4, d_ff=256,
                     num_experts=4, dropout=0.1):
            super().__init__()
            self.tokenizer = FeatureTokenizer(d_model)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, nhead, d_ff, num_experts, dropout)
                for _ in range(num_layers)
            ])
            self.cls_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cls_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cls_norm = nn.LayerNorm(d_model)
            self.action_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, NUM_ACTIONS),
            )
            self.location_head = nn.Sequential(
                nn.Linear(d_model + NUM_ACTIONS, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, NUM_LOCATIONS),
            )
            self.duration_head = nn.Sequential(
                nn.Linear(d_model + NUM_ACTIONS, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, features):
            tokens = self.tokenizer(features)
            for layer in self.layers:
                tokens = layer(tokens)
            B = features.shape[0]
            cls = self.cls_query.expand(B, -1, -1)
            cls_out, _ = self.cls_attn(cls, tokens, tokens)
            h = self.cls_norm(cls_out.squeeze(1))
            action_logits = self.action_head(h)
            action_probs = F.softmax(action_logits.detach(), dim=-1)
            h_a = torch.cat([h, action_probs], dim=-1)
            location_logits = self.location_head(h_a)
            duration = torch.sigmoid(self.duration_head(h_a)) * 7.0 + 1.0
            return {
                "action_logits": action_logits,
                "location_logits": location_logits,
                "duration": duration.squeeze(-1),
            }

    return SociAgentTransformer()


# ══════════════════════════════════════════════════════════════════════════
# 6. Training
# ══════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 3e-4,
    num_train: int = 100_000,
    num_val: int = 10_000,
    data_dir: str | None = None,
    resume: bool = False,
):
    """Full training pipeline: generate/load data, train, export ONNX."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = MODEL_DIR / "soci_agent_best.pt"
    onnx_path = MODEL_DIR / "soci_agent.onnx"

    # ── Load / generate data ─────────────────────────────────────────
    collected = []
    source_counts: dict[str, int] = {}

    # Load collected samples from live sim (if available)
    samples_file = Path(data_dir) / "collected_samples.jsonl" if data_dir else SAMPLES_FILE
    if samples_file.exists():
        with open(samples_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    collected.append(sample)
                    src = sample.get("source", "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1
        logger.info(f"Loaded {len(collected):,} collected samples — sources: {source_counts}")

        # Oversample LLM-sourced data 3x (higher quality than NN/routine)
        llm_sources = {"gemini", "claude", "groq"}
        llm_samples = [s for s in collected if s.get("source", "") in llm_sources]
        if llm_samples:
            logger.info(f"Oversampling {len(llm_samples):,} LLM-sourced samples (3x weight)")
            collected.extend(llm_samples * 2)

    # Generate synthetic data to fill up to target size
    total_target = num_train + num_val
    synthetic_needed = max(0, total_target - len(collected))
    if synthetic_needed > 0:
        logger.info(f"Generating {synthetic_needed:,} synthetic samples...")
        random.seed(42)
        collected.extend(generate_dataset(synthetic_needed))

    random.shuffle(collected)
    split = int(len(collected) * 0.9)
    train_data = collected[:split]
    val_data = collected[split:]

    # ── Dataset ──────────────────────────────────────────────────────
    class ActionDataset(Dataset):
        def __init__(self, data):
            self.features = torch.tensor([d["features"] for d in data], dtype=torch.float32)
            self.actions = torch.tensor([d["action_idx"] for d in data], dtype=torch.long)
            self.locations = torch.tensor([d["target_loc_idx"] for d in data], dtype=torch.long)
            self.durations = torch.tensor([d["duration"] for d in data], dtype=torch.float32)

        def __len__(self):
            return len(self.actions)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "action": self.actions[idx],
                "location": self.locations[idx],
                "duration": self.durations[idx],
            }

    train_ds = ActionDataset(train_data)
    val_ds = ActionDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    logger.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB fp32)")

    if resume and best_pt.exists():
        model.load_state_dict(torch.load(str(best_pt), map_location=DEVICE, weights_only=True))
        logger.info(f"Resumed from {best_pt}")

    # ── Class weights ────────────────────────────────────────────────
    action_counts = torch.zeros(NUM_ACTIONS)
    for d in train_data:
        action_counts[d["action_idx"]] += 1
    action_weights = 1.0 / (action_counts + 1.0)
    action_weights = action_weights / action_weights.sum() * NUM_ACTIONS
    action_weights = action_weights.to(DEVICE)

    logger.info("Action distribution:")
    for idx in range(NUM_ACTIONS):
        count = int(action_counts[idx])
        pct = count / len(train_data) * 100
        logger.info(f"  {ACTION_TYPES[idx]:>10s}: {count:6d} ({pct:.1f}%)")

    # ── Loss & optimizer ─────────────────────────────────────────────
    action_loss_fn = nn.CrossEntropyLoss(weight=action_weights)
    location_loss_fn = nn.CrossEntropyLoss()
    duration_loss_fn = nn.MSELoss()

    W_ACTION = 1.0
    W_LOCATION = 0.5
    W_DURATION = 0.2

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    logger.info(f"Training for {epochs} epochs, LR={lr}, batch_size={batch_size}")

    # ── Training loop ────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_action_acc": [], "val_loc_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            feat = batch["features"].to(DEVICE)
            out = model(feat)
            loss = (
                W_ACTION * action_loss_fn(out["action_logits"], batch["action"].to(DEVICE))
                + W_LOCATION * location_loss_fn(out["location_logits"], batch["location"].to(DEVICE))
                + W_DURATION * duration_loss_fn(out["duration"], batch["duration"].to(DEVICE))
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train_loss = total_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        correct_action = 0
        correct_loc = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(DEVICE)
                out = model(feat)
                loss = (
                    W_ACTION * action_loss_fn(out["action_logits"], batch["action"].to(DEVICE))
                    + W_LOCATION * location_loss_fn(out["location_logits"], batch["location"].to(DEVICE))
                    + W_DURATION * duration_loss_fn(out["duration"], batch["duration"].to(DEVICE))
                )
                val_loss += loss.item()
                pred_action = out["action_logits"].argmax(dim=-1)
                pred_loc = out["location_logits"].argmax(dim=-1)
                correct_action += (pred_action == batch["action"].to(DEVICE)).sum().item()
                correct_loc += (pred_loc == batch["location"].to(DEVICE)).sum().item()
                total += feat.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        action_acc = correct_action / total if total > 0 else 0
        loc_acc = correct_loc / total if total > 0 else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_action_acc"].append(action_acc)
        history["val_loc_acc"].append(loc_acc)

        if action_acc > best_val_acc:
            best_val_acc = action_acc
            torch.save(model.state_dict(), str(best_pt))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Act Acc: {action_acc:.1%} | "
                f"Loc Acc: {loc_acc:.1%} | "
                f"LR: {lr_now:.2e}"
            )

    logger.info(f"Best validation action accuracy: {best_val_acc:.1%}")

    # ── Per-action accuracy ──────────────────────────────────────────
    model.load_state_dict(torch.load(str(best_pt), map_location=DEVICE, weights_only=True))
    model.eval()
    cm = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=int)
    with torch.no_grad():
        for batch in val_loader:
            feat = batch["features"].to(DEVICE)
            out = model(feat)
            preds = out["action_logits"].argmax(dim=-1).cpu().numpy()
            labels = batch["action"].numpy()
            for p, l in zip(preds, labels):
                cm[l][p] += 1

    logger.info("Per-action accuracy:")
    for i, action in enumerate(ACTION_TYPES):
        row_total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / row_total if row_total > 0 else 0
        logger.info(f"  {action:>10s}: {acc:.1%} ({correct}/{row_total})")

    # ── Test scenarios ───────────────────────────────────────────────
    import torch.nn.functional as F

    @torch.no_grad()
    def predict(persona, hour, minute, day, needs, mood, loc, num_people=0):
        features = encode_features(persona, hour, minute, day, needs, mood, loc, num_people)
        feat_t = torch.tensor([features], dtype=torch.float32, device=DEVICE)
        out = model(feat_t)
        action_probs = F.softmax(out["action_logits"][0] / 0.7, dim=-1)
        action_idx = action_probs.argmax().item()
        loc_idx = out["location_logits"][0].argmax().item()
        dur = max(1, min(8, round(out["duration"][0].item())))
        return ACTION_TYPES[action_idx], LOCATIONS[loc_idx], dur, action_probs[action_idx].item()

    logger.info("Test scenarios:")
    a, l, d, c = predict(PERSONAS[0], 0, 30, 5,
                         {"hunger": 0.5, "energy": 0.05, "social": 0.4, "purpose": 0.6, "comfort": 0.3, "fun": 0.3},
                         -0.3, "office")
    logger.info(f"  Elena midnight exhausted at office: {a} -> {l} ({d} ticks, {c:.0%})")

    a, l, d, c = predict(PERSONAS[2], 12, 30, 3,
                         {"hunger": 0.05, "energy": 0.7, "social": 0.5, "purpose": 0.6, "comfort": 0.5, "fun": 0.4},
                         0.2, "gym", 5)
    logger.info(f"  Marcus lunchtime starving at gym: {a} -> {l} ({d} ticks, {c:.0%})")

    a, l, d, c = predict(PERSONAS[8], 10, 0, 6,
                         {"hunger": 0.6, "energy": 0.7, "social": 0.5, "purpose": 0.5, "comfort": 0.7, "fun": 0.4},
                         0.5, "house_kai")
    logger.info(f"  Kai Saturday morning at home: {a} -> {l} ({d} ticks, {c:.0%})")

    # George (night shift) — should sleep during the day
    george = [p for p in PERSONAS if p["id"] == "george"][0]
    a, l, d, c = predict(george, 11, 0, 3,
                         {"hunger": 0.4, "energy": 0.15, "social": 0.5, "purpose": 0.7, "comfort": 0.5, "fun": 0.4},
                         -0.1, "house_frank")
    logger.info(f"  George midday after night shift: {a} -> {l} ({d} ticks, {c:.0%})")

    # Frank — evening at the bar
    frank = [p for p in PERSONAS if p["id"] == "frank"][0]
    a, l, d, c = predict(frank, 20, 0, 4,
                         {"hunger": 0.5, "energy": 0.4, "social": 0.3, "purpose": 0.6, "comfort": 0.5, "fun": 0.3},
                         0.1, "bar")
    logger.info(f"  Frank evening at the bar: {a} -> {l} ({d} ticks, {c:.0%})")

    # Priya — overworked at hospital
    priya = [p for p in PERSONAS if p["id"] == "priya"][0]
    a, l, d, c = predict(priya, 15, 0, 2,
                         {"hunger": 0.3, "energy": 0.2, "social": 0.3, "purpose": 0.8, "comfort": 0.4, "fun": 0.2},
                         -0.2, "hospital")
    logger.info(f"  Priya afternoon exhausted at hospital: {a} -> {l} ({d} ticks, {c:.0%})")

    # ── Export to ONNX ───────────────────────────────────────────────
    logger.info("Exporting to ONNX...")
    model.cpu().eval()
    dummy = torch.randn(1, FEATURE_DIM)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["features"],
        output_names=["action_logits", "location_logits", "duration"],
        dynamic_axes={"features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_size = onnx_path.stat().st_size / 1024
    logger.info(f"ONNX exported: {onnx_path} ({onnx_size:.0f} KB)")

    # Benchmark ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    batch_input = np.random.randn(50, FEATURE_DIM).astype(np.float32)
    start = time.perf_counter()
    for _ in range(100):
        session.run(None, {"features": batch_input})
    elapsed = (time.perf_counter() - start) / 100
    logger.info(f"ONNX inference (50 agents): {elapsed*1000:.1f} ms per batch")

    # ── Save training stats ──────────────────────────────────────────
    stats = {
        "best_val_action_acc": best_val_acc,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "collected_samples": sum(source_counts.values()),
        "source_counts": source_counts,
        "model_size_kb": onnx_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "history": history,
    }
    stats_path = MODEL_DIR / "training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats saved to {stats_path}")

    # ── Plot training graphs ──────────────────────────────────────────
    plot_training_graphs(stats_path)

    return best_val_acc


def plot_training_graphs(stats_path: Path | str | None = None):
    """Plot training loss and accuracy curves from saved training stats.

    Saves the plot to models/training_graphs.png and displays it.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend as fallback
    import matplotlib.pyplot as plt

    stats_path = Path(stats_path) if stats_path else MODEL_DIR / "training_stats.json"
    if not stats_path.exists():
        logger.error(f"No training stats found at {stats_path}")
        return

    stats = json.loads(stats_path.read_text())
    history = stats.get("history", {})

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_action_acc = history.get("val_action_acc", [])
    val_loc_acc = history.get("val_loc_acc", [])

    if not train_loss:
        logger.error("No training history found in stats file")
        return

    epochs_range = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Soci Agent NN Training — {stats.get('timestamp', '?')}  |  "
        f"Best Action Acc: {stats.get('best_val_action_acc', 0):.1%}",
        fontsize=13, fontweight="bold",
    )

    # Loss curves
    ax = axes[0]
    ax.plot(epochs_range, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(epochs_range, val_loss, label="Val Loss", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    # Action accuracy
    ax = axes[1]
    ax.plot(epochs_range, [a * 100 for a in val_action_acc], label="Action Accuracy",
            color="#4CAF50", linewidth=2)
    best_epoch = int(np.argmax(val_action_acc)) + 1
    best_acc = max(val_action_acc) * 100
    ax.axhline(y=best_acc, color="#4CAF50", linestyle="--", alpha=0.4)
    ax.annotate(f"Best: {best_acc:.1f}% (epoch {best_epoch})",
                xy=(best_epoch, best_acc), fontsize=9,
                xytext=(best_epoch + 1, best_acc - 3),
                arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                color="#4CAF50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    # Location accuracy
    ax = axes[2]
    if val_loc_acc:
        ax.plot(epochs_range, [a * 100 for a in val_loc_acc], label="Location Accuracy",
                color="#FF9800", linewidth=2)
        best_loc_epoch = int(np.argmax(val_loc_acc)) + 1
        best_loc = max(val_loc_acc) * 100
        ax.axhline(y=best_loc, color="#FF9800", linestyle="--", alpha=0.4)
        ax.annotate(f"Best: {best_loc:.1f}% (epoch {best_loc_epoch})",
                    xy=(best_loc_epoch, best_loc), fontsize=9,
                    xytext=(best_loc_epoch + 1, best_loc - 3),
                    arrowprops=dict(arrowstyle="->", color="#FF9800"),
                    color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Location Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    # Footer with training info
    footer = (
        f"Train: {stats.get('train_samples', '?'):,} samples  |  "
        f"Val: {stats.get('val_samples', '?'):,} samples  |  "
        f"Collected: {stats.get('collected_samples', 0):,}  |  "
        f"Model: {stats.get('model_size_kb', 0):.0f} KB"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    graph_path = MODEL_DIR / "training_graphs.png"
    fig.savefig(str(graph_path), dpi=150, bbox_inches="tight")
    logger.info(f"Training graphs saved to {graph_path}")

    # Try to display interactively
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("TkAgg")
            plt.show(block=False)
            plt.pause(0.5)
    except Exception:
        pass  # headless environment, PNG saved is enough

    plt.close(fig)


def _push_to_hub(best_pt, onnx_path, stats_path, repo_id, best_val_acc, epochs, num_train,
                 base_url: str = "https://raymelius-soci2.hf.space"):
    """Upload model files to HuggingFace Hub, then trigger live reload."""
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        logger.error("HF_TOKEN not set — cannot push. Export it: export HF_TOKEN=hf_...")
        return

    login(token=token)
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    # Config
    config = {
        "architecture": "SociAgentTransformer",
        "d_model": 128, "nhead": 8, "num_layers": 4, "d_ff": 256, "num_experts": 4,
        "feature_dim": FEATURE_DIM, "num_actions": NUM_ACTIONS, "num_locations": NUM_LOCATIONS,
        "action_types": ACTION_TYPES, "locations": LOCATIONS,
        "action_durations": ACTION_DURATIONS, "need_names": NEED_NAMES,
        "personality_names": PERSONALITY_NAMES,
        "best_val_action_acc": best_val_acc,
        "training_samples": num_train, "epochs": epochs,
    }
    config_path = MODEL_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    for local, remote in [
        (onnx_path, "soci_agent.onnx"),
        (best_pt, "soci_agent_best.pt"),
        (config_path, "config.json"),
        (stats_path, "training_stats.json"),
    ]:
        if local.exists():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
                commit_message=f"Train: acc={best_val_acc:.1%}, {epochs} epochs",
            )
            logger.info(f"Uploaded {remote}")

    logger.info(f"Model pushed to https://huggingface.co/{repo_id}")

    # Trigger hot-reload on the live simulation
    try:
        import httpx
        resp = httpx.post(f"{base_url}/api/nn/reload", timeout=30.0)
        if resp.status_code == 200:
            logger.info(f"Live sim NN reloaded: {resp.json().get('message', 'ok')}")
        else:
            logger.warning(f"Could not reload live sim NN: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not reach live sim for reload: {e}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Soci Agent NN — Local Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/nn_train.py                             # Train from scratch
  python scripts/nn_train.py --resume --epochs 50        # Continue training
  python scripts/nn_train.py --data data/nn_training     # Use collected samples
  python scripts/nn_train.py --push                      # Push existing model to HF Hub
  python scripts/nn_train.py --graph                     # Show graphs from last training
""",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--train-samples", type=int, default=100_000,
                        help="Number of synthetic training samples (default: 100000)")
    parser.add_argument("--val-samples", type=int, default=10_000,
                        help="Number of validation samples (default: 10000)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to directory with collected_samples.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing weights in models/")
    parser.add_argument("--push", action="store_true",
                        help="Push existing model to HuggingFace Hub (no training)")
    parser.add_argument("--graph", action="store_true",
                        help="Display training graphs from last training run")
    parser.add_argument("--repo", default="RayMelius/soci-agent-nn",
                        help="HF Hub repo ID (default: RayMelius/soci-agent-nn)")
    parser.add_argument("--url", default="https://raymelius-soci2.hf.space",
                        help="Live simulation URL for hot-reload after push (default: HF Space)")
    args = parser.parse_args()

    # --graph: just display graphs and exit
    if args.graph:
        plot_training_graphs()
        return

    # --push: just push existing model to HF Hub and exit
    if args.push:
        stats_path = MODEL_DIR / "training_stats.json"
        best_pt = MODEL_DIR / "soci_agent_best.pt"
        onnx_path = MODEL_DIR / "soci_agent.onnx"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            best_val_acc = stats.get("best_val_action_acc", 0)
            ep = stats.get("epochs", 0)
            n_train = stats.get("train_samples", 0)
        else:
            best_val_acc, ep, n_train = 0, 0, 0
        _push_to_hub(best_pt, onnx_path, stats_path, args.repo, best_val_acc, ep, n_train,
                     base_url=args.url)
        return

    # Default: train
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_train=args.train_samples,
        num_val=args.val_samples,
        data_dir=args.data,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
