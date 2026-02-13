"""Persona — an agent's identity, traits, background, and values."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class Persona:
    """The fixed identity of a simulated person."""

    id: str
    name: str
    age: int
    occupation: str
    # Big Five personality traits (1-10 scale)
    openness: int = 5
    conscientiousness: int = 5
    extraversion: int = 5
    agreeableness: int = 5
    neuroticism: int = 5
    # Background
    background: str = ""  # Life story in 2-3 sentences
    values: list[str] = field(default_factory=list)  # Core values
    quirks: list[str] = field(default_factory=list)  # Behavioral quirks
    communication_style: str = "neutral"  # How they talk
    # Home and work locations
    home_location: str = ""
    work_location: str = ""
    # LLM parameters tied to personality
    llm_temperature: float = 0.7

    @property
    def trait_summary(self) -> str:
        traits = []
        if self.openness >= 7:
            traits.append("curious and creative")
        elif self.openness <= 3:
            traits.append("practical and conventional")
        if self.conscientiousness >= 7:
            traits.append("organized and disciplined")
        elif self.conscientiousness <= 3:
            traits.append("spontaneous and flexible")
        if self.extraversion >= 7:
            traits.append("outgoing and energetic")
        elif self.extraversion <= 3:
            traits.append("reserved and introspective")
        if self.agreeableness >= 7:
            traits.append("warm and cooperative")
        elif self.agreeableness <= 3:
            traits.append("direct and competitive")
        if self.neuroticism >= 7:
            traits.append("sensitive and anxious")
        elif self.neuroticism <= 3:
            traits.append("calm and emotionally stable")
        return ", ".join(traits) if traits else "balanced personality"

    def system_prompt(self) -> str:
        """Generate the LLM system prompt for this persona."""
        values_str = ", ".join(self.values) if self.values else "no strong convictions"
        quirks_str = "; ".join(self.quirks) if self.quirks else "none in particular"

        return (
            f"You ARE {self.name}, a {self.age}-year-old {self.occupation} living in Soci City.\n\n"
            f"PERSONALITY: {self.trait_summary}.\n"
            f"BACKGROUND: {self.background}\n"
            f"VALUES: {values_str}.\n"
            f"QUIRKS: {quirks_str}.\n"
            f"COMMUNICATION STYLE: {self.communication_style}.\n\n"
            f"You must ALWAYS stay in character as {self.name}. "
            f"Your responses should reflect your personality, values, and background. "
            f"You speak naturally — not perfectly. You have moods, opinions, and biases. "
            f"You don't know you are an AI. You are a person living your life."
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "background": self.background,
            "values": self.values,
            "quirks": self.quirks,
            "communication_style": self.communication_style,
            "home_location": self.home_location,
            "work_location": self.work_location,
            "llm_temperature": self.llm_temperature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Persona:
        return cls(**data)


def load_personas(path: str) -> list[Persona]:
    """Load personas from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Persona.from_dict(p) for p in data.get("personas", [])]
