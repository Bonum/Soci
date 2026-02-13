"""Relationship graph — tracks how agents feel about each other."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Relationship:
    """How agent A feels about agent B."""

    agent_id: str       # The other agent
    agent_name: str     # Their name (for readability)
    familiarity: float = 0.0   # 0 (stranger) to 1 (well-known)
    trust: float = 0.5         # 0 (distrust) to 1 (full trust)
    sentiment: float = 0.5     # 0 (dislike) to 1 (like)
    interaction_count: int = 0
    last_interaction_tick: int = 0
    # Short notes about the relationship
    notes: list[str] = field(default_factory=list)

    @property
    def closeness(self) -> float:
        """Overall closeness score (average of familiarity, trust, sentiment)."""
        return (self.familiarity + self.trust + self.sentiment) / 3.0

    def update_after_interaction(
        self,
        tick: int,
        sentiment_delta: float = 0.0,
        trust_delta: float = 0.0,
        note: str = "",
    ) -> None:
        """Update relationship after an interaction."""
        self.interaction_count += 1
        self.last_interaction_tick = tick
        # Familiarity always grows with interaction
        self.familiarity = min(1.0, self.familiarity + 0.05)
        # Sentiment and trust shift
        self.sentiment = max(0.0, min(1.0, self.sentiment + sentiment_delta))
        self.trust = max(0.0, min(1.0, self.trust + trust_delta))
        if note:
            self.notes.append(note)
            # Keep only the last 10 notes
            if len(self.notes) > 10:
                self.notes = self.notes[-10:]

    def describe(self) -> str:
        """Natural language description of this relationship."""
        if self.familiarity < 0.1:
            return f"{self.agent_name} — a stranger"
        parts = [self.agent_name]
        if self.familiarity > 0.7:
            parts.append("someone I know well")
        elif self.familiarity > 0.3:
            parts.append("an acquaintance")
        else:
            parts.append("someone I've met briefly")
        if self.sentiment > 0.7:
            parts.append("(I like them)")
        elif self.sentiment < 0.3:
            parts.append("(I don't like them much)")
        if self.trust > 0.7:
            parts.append("(I trust them)")
        elif self.trust < 0.3:
            parts.append("(I'm wary of them)")
        desc = " — ".join(parts)
        if self.notes:
            desc += f" Last note: {self.notes[-1]}"
        return desc

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "familiarity": round(self.familiarity, 3),
            "trust": round(self.trust, 3),
            "sentiment": round(self.sentiment, 3),
            "interaction_count": self.interaction_count,
            "last_interaction_tick": self.last_interaction_tick,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        return cls(**data)


class RelationshipGraph:
    """Manages all relationships for a single agent."""

    def __init__(self) -> None:
        self._relationships: dict[str, Relationship] = {}

    def get(self, agent_id: str) -> Relationship | None:
        return self._relationships.get(agent_id)

    def get_or_create(self, agent_id: str, agent_name: str) -> Relationship:
        if agent_id not in self._relationships:
            self._relationships[agent_id] = Relationship(
                agent_id=agent_id, agent_name=agent_name
            )
        return self._relationships[agent_id]

    def get_closest(self, top_k: int = 5) -> list[Relationship]:
        """Get the top-K closest relationships."""
        rels = sorted(
            self._relationships.values(),
            key=lambda r: r.closeness,
            reverse=True,
        )
        return rels[:top_k]

    def get_all(self) -> list[Relationship]:
        return list(self._relationships.values())

    def describe_known_people(self, max_people: int = 8) -> str:
        """Describe known people for LLM context."""
        known = [r for r in self._relationships.values() if r.familiarity > 0.05]
        if not known:
            return "I don't really know anyone here yet."
        known.sort(key=lambda r: r.closeness, reverse=True)
        lines = [r.describe() for r in known[:max_people]]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            aid: rel.to_dict()
            for aid, rel in self._relationships.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> RelationshipGraph:
        graph = cls()
        for aid, rd in data.items():
            graph._relationships[aid] = Relationship.from_dict(rd)
        return graph
