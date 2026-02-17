"""Procedural persona generator — creates unique personas for scaling beyond YAML."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from soci.agents.persona import Persona

if TYPE_CHECKING:
    from soci.world.city import City

# --- Name pools (diverse) ---
FIRST_NAMES_MALE = [
    "James", "Marcus", "Omar", "Kai", "Devon", "Theo", "Frank", "George", "Sam",
    "Liam", "Noah", "Ethan", "Lucas", "Mason", "Logan", "Aiden", "Jackson", "Caleb",
    "Owen", "Carter", "Jayden", "Dylan", "Gabriel", "Anthony", "Isaac", "Adrian",
    "Mateo", "Ryan", "Leo", "Sebastian", "Jaxon", "Dominic", "Nathan", "Ezra",
    "Ravi", "Hiroshi", "Dmitri", "Kwame", "Alejandro", "Tariq", "Jian", "Nikolai",
    "Emeka", "Yousef", "Andrei", "Kofi", "Rafael", "Jin", "Arjun", "Tomás",
    "Bryce", "Malcolm", "Rohan", "Declan", "Felix", "Miles", "Hugo", "Jasper",
    "Elliot", "Wesley", "Damian", "Silas", "Tristan", "Vincent", "Abel", "Cyrus",
    "Kenneth", "Curtis", "Derek", "Troy", "Mitchell", "Grant", "Russell", "Brent",
    "Daryl", "Reginald", "Cecil", "Wallace", "Clifford", "Howard", "Vernon", "Earl",
    "Cedric", "Marvin", "Desmond", "Ruben", "Terrence", "Darius", "Lamar", "Winston",
    "Trevor", "Patrick", "Cody", "Brett", "Lance", "Reed", "Clark", "Blake",
]

FIRST_NAMES_FEMALE = [
    "Elena", "Lila", "Zoe", "Helen", "Alice", "Diana", "Priya", "Nina", "Rosa",
    "Yuki", "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Abigail", "Ella", "Scarlett", "Grace", "Lily",
    "Aria", "Riley", "Nora", "Zoey", "Penelope", "Layla", "Chloe", "Victoria",
    "Aisha", "Mei", "Fatima", "Anya", "Sakura", "Ingrid", "Carmen", "Leila",
    "Nalini", "Chioma", "Esmeralda", "Suki", "Tatiana", "Amara", "Ximena", "Hana",
    "Iris", "Jade", "Stella", "Violet", "Luna", "Ivy", "Hazel", "Aurora",
    "Savannah", "Audrey", "Brooklyn", "Bella", "Claire", "Lucy", "Skylar", "Paisley",
    "Clara", "Margot", "Fiona", "Wren", "Elise", "Daphne", "Celeste", "Lydia",
    "Bea", "Greta", "Tessa", "June", "Pearl", "Opal", "Vera", "Ruth",
    "Dorothy", "Mabel", "Agnes", "Edith", "Gladys", "Mildred", "Bernice", "Lucille",
    "Tamara", "Simone", "Rochelle", "Denise", "Monica", "Bianca", "Giselle", "Naomi",
]

FIRST_NAMES_NB = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Quinn", "Avery", "Riley",
    "Dakota", "Skyler", "Sage", "Rowan", "Finley", "Emery", "River", "Hayden",
]

LAST_NAMES = [
    "Chen", "Rodriguez", "Patel", "Kim", "Garcia", "Williams", "Johnson", "Brown",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Thompson", "Lee", "Walker", "Hall", "Allen", "Young",
    "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams",
    "Baker", "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner",
    "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez",
    "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey",
    "Okafor", "Nakamura", "Petrov", "Johansson", "Müller", "Dubois", "Rossi", "Silva",
    "Tanaka", "Singh", "Ali", "Sato", "Ivanov", "Larsson", "Kowalski", "Novak",
    "O'Brien", "Brennan", "Reeves", "Holt", "Vasquez", "Santiago", "Delgado", "Moreno",
    "Fischer", "Wagner", "Becker", "Meyer", "Weber", "Hoffman", "Schultz", "Lang",
    "Stone", "Fox", "Cross", "Lane", "Rush", "Day", "Snow", "Frost",
    "Wolfe", "Marsh", "Banks", "Hope", "Wise", "Chase", "Steele", "Drake",
    "Blair", "Hale", "Vega", "Luna", "Rios", "Campos", "Soto", "Reyes",
    "Mendez", "Ortiz", "Flores", "Ramos", "Cruz", "Gutierrez", "Vargas", "Medina",
    "Choi", "Park", "Yoon", "Han", "Lim", "Kwon", "Cho", "Jang",
    "Wang", "Liu", "Zhang", "Li", "Yang", "Huang", "Zhou", "Wu",
    "Gupta", "Sharma", "Kumar", "Verma", "Joshi", "Mehta", "Shah", "Reddy",
    "Osei", "Mensah", "Asante", "Boateng", "Amoah", "Opoku", "Owusu", "Adjei",
]

OCCUPATIONS = [
    # White collar
    ("software engineer", "work"), ("accountant", "work"), ("marketing manager", "work"),
    ("architect", "work"), ("data analyst", "work"), ("project manager", "work"),
    ("graphic designer", "work"), ("lawyer", "work"), ("consultant", "work"),
    ("financial advisor", "work"),
    # Blue collar / service
    ("mechanic", "work"), ("electrician", "work"), ("plumber", "work"),
    ("construction worker", "work"),
    # Service / hospitality (evening shifts)
    ("bartender", "commercial"), ("chef", "commercial"), ("waiter", "commercial"),
    ("barista", "commercial"),
    # Creative
    ("writer", "work"), ("musician", "work"), ("photographer", "work"),
    ("artist", "work"),
    # Education
    ("teacher", "work"), ("professor", "work"), ("tutor", "work"),
    # Health
    ("nurse", "work"), ("personal trainer", "public"), ("therapist", "work"),
    # Student / retired
    ("college student", "work"), ("retired", None),
]

VALUES_POOL = [
    "family", "career", "honesty", "creativity", "adventure", "community",
    "independence", "knowledge", "health", "tradition", "justice", "compassion",
    "wealth", "spirituality", "loyalty", "ambition", "simplicity", "humor",
    "freedom", "respect",
]

QUIRKS_POOL = [
    "always carries a book", "hums while walking", "talks to plants",
    "obsessed with coffee", "compulsive note-taker", "never remembers names",
    "always early", "chronic over-sharer", "apologizes too much",
    "uses old-fashioned slang", "collects random things", "doodles during conversations",
    "quotes movies constantly", "always has snacks", "fidgets with keys",
    "checks phone compulsively", "whistles off-key", "gives unsolicited advice",
    "afraid of pigeons", "tells the same stories", "makes up words",
    "eats loudly", "gestures wildly when talking", "has strong opinions about weather",
    "always wears a hat",
]

# Occupation categories for schedule variation
EVENING_SHIFT_JOBS = {"bartender", "chef", "waiter", "barista"}
STUDENT_OCCUPATIONS = {"college student"}
RETIRED_OCCUPATIONS = {"retired"}
PHYSICAL_JOBS = {"mechanic", "electrician", "plumber", "construction worker", "personal trainer"}


def _pick_gender() -> str:
    """Weighted random gender."""
    r = random.random()
    if r < 0.47:
        return "male"
    elif r < 0.94:
        return "female"
    else:
        return "nonbinary"


def _pick_name(gender: str, used_names: set[str]) -> str:
    """Pick a unique full name."""
    for _ in range(100):
        if gender == "male":
            first = random.choice(FIRST_NAMES_MALE)
        elif gender == "female":
            first = random.choice(FIRST_NAMES_FEMALE)
        else:
            first = random.choice(FIRST_NAMES_NB)
        last = random.choice(LAST_NAMES)
        full = f"{first} {last}"
        if full not in used_names:
            used_names.add(full)
            return full
    # Fallback: add a number
    full = f"{first} {last} Jr"
    used_names.add(full)
    return full


def _pick_occupation(age: int) -> tuple[str, str | None]:
    """Pick occupation based on age. Returns (title, work_zone)."""
    if age >= 65 and random.random() < 0.7:
        return "retired", None
    if 18 <= age <= 22 and random.random() < 0.6:
        return "college student", "work"
    return random.choice(OCCUPATIONS)


def _generate_traits() -> dict[str, int]:
    """Generate Big Five traits with slight correlations."""
    o = random.randint(2, 9)
    c = random.randint(2, 9)
    e = random.randint(2, 9)
    a = random.randint(2, 9)
    # High conscientiousness slightly correlates with lower neuroticism
    n_base = random.randint(2, 9)
    n = max(1, min(10, n_base - (c - 5) // 3))
    return {
        "openness": o,
        "conscientiousness": c,
        "extraversion": e,
        "agreeableness": a,
        "neuroticism": n,
    }


def _pick_values(traits: dict[str, int]) -> list[str]:
    """Pick 2-4 values weighted by personality."""
    count = random.randint(2, 4)
    weights = {}
    for v in VALUES_POOL:
        w = 1.0
        if v == "career" and traits["conscientiousness"] >= 7:
            w = 2.0
        elif v == "creativity" and traits["openness"] >= 7:
            w = 2.0
        elif v == "community" and traits["agreeableness"] >= 7:
            w = 2.0
        elif v == "adventure" and traits["openness"] >= 7:
            w = 2.0
        elif v == "independence" and traits["extraversion"] <= 4:
            w = 1.5
        elif v == "health" and traits["conscientiousness"] >= 6:
            w = 1.5
        weights[v] = w
    pool = list(weights.keys())
    w_list = [weights[v] for v in pool]
    chosen = []
    for _ in range(count):
        if not pool:
            break
        selected = random.choices(pool, weights=w_list, k=1)[0]
        chosen.append(selected)
        idx = pool.index(selected)
        pool.pop(idx)
        w_list.pop(idx)
    return chosen


def _pick_quirks() -> list[str]:
    """Pick 1-3 random quirks."""
    return random.sample(QUIRKS_POOL, k=random.randint(1, 3))


def _communication_style(extraversion: int, agreeableness: int) -> str:
    """Derive communication style from traits."""
    if extraversion >= 7 and agreeableness >= 7:
        return "warm and chatty"
    elif extraversion >= 7 and agreeableness <= 4:
        return "loud and blunt"
    elif extraversion <= 3 and agreeableness >= 7:
        return "quiet and polite"
    elif extraversion <= 3 and agreeableness <= 4:
        return "terse and reserved"
    elif extraversion >= 7:
        return "talkative and expressive"
    elif extraversion <= 3:
        return "quiet and thoughtful"
    elif agreeableness >= 7:
        return "friendly and considerate"
    elif agreeableness <= 4:
        return "direct and no-nonsense"
    return "neutral"


def _generate_background(name: str, age: int, occupation: str, traits: dict[str, int]) -> str:
    """Generate a 2-3 sentence background."""
    first = name.split()[0]

    # Age-based life stage
    if age <= 22:
        stage = f"{first} is a {age}-year-old finding their way in life"
    elif age <= 35:
        stage = f"{first} is a {age}-year-old building their career"
    elif age <= 55:
        stage = f"{first} is a {age}-year-old well-established in the community"
    elif age <= 65:
        stage = f"{first} is a {age}-year-old approaching the later chapters of life"
    else:
        stage = f"{first} is a {age}-year-old enjoying their golden years"

    # Occupation context
    if occupation == "retired":
        job_part = "After decades of work, they now fill their days with hobbies and neighborhood life."
    elif occupation == "college student":
        subjects = random.choice([
            "literature", "engineering", "biology", "business", "art history",
            "computer science", "psychology", "nursing", "philosophy", "music",
        ])
        job_part = f"They're studying {subjects} and juggling classes with a social life."
    else:
        job_part = f"They work as a {occupation} and take pride in what they do."

    # Personality flavor
    flavors = []
    if traits["openness"] >= 7:
        flavors.append("loves trying new things")
    if traits["conscientiousness"] >= 7:
        flavors.append("keeps a tight schedule")
    if traits["extraversion"] >= 7:
        flavors.append("lights up every room they enter")
    if traits["agreeableness"] >= 7:
        flavors.append("is always ready to lend a hand")
    if traits["neuroticism"] >= 7:
        flavors.append("tends to overthink things")
    if traits["extraversion"] <= 3:
        flavors.append("prefers a quiet evening at home")
    if traits["conscientiousness"] <= 3:
        flavors.append("goes with the flow")

    personality_part = ""
    if flavors:
        picked = random.sample(flavors, k=min(2, len(flavors)))
        personality_part = f" {first} {' and '.join(picked)}."

    return f"{stage}. {job_part}{personality_part}"


def _llm_temperature(openness: int) -> float:
    """Map openness to LLM temperature."""
    return 0.5 + (openness / 10.0) * 0.4  # 0.5 - 0.9


def _assign_locations(
    persona_data: dict,
    occupation: str,
    work_zone: str | None,
    residential_ids: list[str],
    work_ids: list[str],
    commercial_ids: list[str],
    res_index: int,
) -> tuple[str, str]:
    """Assign home and work locations. Returns (home_id, work_id)."""
    home_id = residential_ids[res_index % len(residential_ids)]

    if occupation in RETIRED_OCCUPATIONS or work_zone is None:
        work_id = home_id  # Retired folks "work" from home (leisure)
    elif occupation in EVENING_SHIFT_JOBS:
        work_id = random.choice(commercial_ids) if commercial_ids else home_id
    else:
        work_id = random.choice(work_ids) if work_ids else home_id

    return home_id, work_id


def generate_personas(count: int, city: City) -> list[Persona]:
    """Generate `count` unique personas with assigned home/work locations."""
    # Gather location pools
    residential_ids = [lid for lid, loc in city.locations.items() if loc.zone == "residential"]
    work_ids = [lid for lid, loc in city.locations.items() if loc.zone == "work"]
    commercial_ids = [lid for lid, loc in city.locations.items() if loc.zone == "commercial"]

    if not residential_ids:
        raise ValueError("City has no residential locations — cannot assign homes.")

    used_names: set[str] = set()
    personas: list[Persona] = []

    for i in range(count):
        gender = _pick_gender()
        name = _pick_name(gender, used_names)
        age = random.randint(18, 75)
        occupation, work_zone = _pick_occupation(age)
        traits = _generate_traits()
        values = _pick_values(traits)
        quirks = _pick_quirks()
        comm_style = _communication_style(traits["extraversion"], traits["agreeableness"])
        background = _generate_background(name, age, occupation, traits)
        temperature = _llm_temperature(traits["openness"])

        home_id, work_id = _assign_locations(
            {},
            occupation,
            work_zone,
            residential_ids,
            work_ids,
            commercial_ids,
            i,
        )

        persona = Persona(
            id=f"gen_{i+1:03d}",
            name=name,
            age=age,
            occupation=occupation,
            gender=gender,
            openness=traits["openness"],
            conscientiousness=traits["conscientiousness"],
            extraversion=traits["extraversion"],
            agreeableness=traits["agreeableness"],
            neuroticism=traits["neuroticism"],
            background=background,
            values=values,
            quirks=quirks,
            communication_style=comm_style,
            home_location=home_id,
            work_location=work_id,
            llm_temperature=round(temperature, 2),
        )
        personas.append(persona)

    return personas
