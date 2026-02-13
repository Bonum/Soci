"""Needs system — Maslow-inspired needs that drive agent behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeedsState:
    """Tracks an agent's current needs. Each need ranges 0.0 (desperate) to 1.0 (fully satisfied)."""

    hunger: float = 0.8     # Physical: need to eat
    energy: float = 1.0     # Physical: need to sleep/rest
    social: float = 0.6     # Belonging: need for interaction
    purpose: float = 0.7    # Esteem: need to do meaningful work
    comfort: float = 0.8    # Safety: need for shelter, stability
    fun: float = 0.5        # Self-actualization: need for enjoyment

    # Decay rates per tick (how fast needs drain)
    _decay_rates: dict = None

    def __post_init__(self):
        self._decay_rates = {
            "hunger": 0.02,    # Gets hungry fairly fast
            "energy": 0.015,   # Drains slowly
            "social": 0.01,    # Drains slowly
            "purpose": 0.008,  # Drains very slowly
            "comfort": 0.005,  # Very stable
            "fun": 0.012,      # Moderate drain
        }

    def tick(self, is_sleeping: bool = False) -> None:
        """Decay all needs by one tick."""
        if is_sleeping:
            # Sleeping restores energy, but hunger still decays
            self.energy = min(1.0, self.energy + 0.05)
            self.hunger = max(0.0, self.hunger - self._decay_rates["hunger"])
        else:
            for need_name, rate in self._decay_rates.items():
                current = getattr(self, need_name)
                setattr(self, need_name, max(0.0, current - rate))

    def satisfy(self, need: str, amount: float) -> None:
        """Satisfy a need by a given amount."""
        if hasattr(self, need):
            current = getattr(self, need)
            setattr(self, need, min(1.0, current + amount))

    @property
    def most_urgent(self) -> str:
        """Return the name of the most urgent (lowest) need."""
        needs = {
            "hunger": self.hunger,
            "energy": self.energy,
            "social": self.social,
            "purpose": self.purpose,
            "comfort": self.comfort,
            "fun": self.fun,
        }
        return min(needs, key=needs.get)

    @property
    def urgent_needs(self) -> list[str]:
        """Return needs below 0.3 threshold, sorted by urgency."""
        needs = {
            "hunger": self.hunger,
            "energy": self.energy,
            "social": self.social,
            "purpose": self.purpose,
            "comfort": self.comfort,
            "fun": self.fun,
        }
        return sorted(
            [n for n, v in needs.items() if v < 0.3],
            key=lambda n: needs[n],
        )

    @property
    def is_critical(self) -> bool:
        """True if any need is critically low (below 0.15)."""
        return any(v < 0.15 for v in [
            self.hunger, self.energy, self.social,
            self.purpose, self.comfort, self.fun,
        ])

    def describe(self) -> str:
        """Natural language description of current need state."""
        parts = []
        if self.hunger < 0.3:
            parts.append("very hungry" if self.hunger < 0.15 else "getting hungry")
        if self.energy < 0.3:
            parts.append("exhausted" if self.energy < 0.15 else "tired")
        if self.social < 0.3:
            parts.append("lonely" if self.social < 0.15 else "wanting company")
        if self.purpose < 0.3:
            parts.append("feeling aimless" if self.purpose < 0.15 else "wanting to do something meaningful")
        if self.comfort < 0.3:
            parts.append("uncomfortable" if self.comfort < 0.15 else "a bit uneasy")
        if self.fun < 0.3:
            parts.append("bored" if self.fun < 0.15 else "wanting some fun")
        if not parts:
            return "feeling good overall"
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "hunger": round(self.hunger, 3),
            "energy": round(self.energy, 3),
            "social": round(self.social, 3),
            "purpose": round(self.purpose, 3),
            "comfort": round(self.comfort, 3),
            "fun": round(self.fun, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> NeedsState:
        state = cls()
        for key, val in data.items():
            if hasattr(state, key) and not key.startswith("_"):
                setattr(state, key, val)
        return state
