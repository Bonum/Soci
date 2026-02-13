"""Memory stream — episodic memory with importance scoring and retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryType(Enum):
    OBSERVATION = "observation"  # "I saw Maria at the cafe"
    REFLECTION = "reflection"   # "Maria seems to visit the cafe every morning"
    PLAN = "plan"              # "I will go to the office at 9am"
    CONVERSATION = "conversation"  # "I talked to John about the weather"
    EVENT = "event"            # "A storm hit the city"


@dataclass
class Memory:
    """A single memory entry in an agent's memory stream."""

    id: int
    tick: int                 # When this memory was created (simulation tick)
    day: int                  # Day number
    time_str: str             # Human-readable time "09:15"
    type: MemoryType
    content: str              # Natural language description
    importance: int = 5       # 1-10 scale, assigned by LLM
    location: str = ""        # Where it happened
    involved_agents: list[str] = field(default_factory=list)  # Other agents involved
    # For retrieval scoring
    access_count: int = 0
    last_accessed_tick: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tick": self.tick,
            "day": self.day,
            "time_str": self.time_str,
            "type": self.type.value,
            "content": self.content,
            "importance": self.importance,
            "location": self.location,
            "involved_agents": self.involved_agents,
            "access_count": self.access_count,
            "last_accessed_tick": self.last_accessed_tick,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        data = dict(data)
        data["type"] = MemoryType(data["type"])
        return cls(**data)


class MemoryStream:
    """An agent's full memory — stores, scores, and retrieves memories."""

    def __init__(self, max_memories: int = 500) -> None:
        self.memories: list[Memory] = []
        self.max_memories = max_memories
        self._next_id: int = 0
        # Running total of importance since last reflection
        self._importance_accumulator: float = 0.0
        self.reflection_threshold: float = 50.0

    def add(
        self,
        tick: int,
        day: int,
        time_str: str,
        memory_type: MemoryType,
        content: str,
        importance: int = 5,
        location: str = "",
        involved_agents: Optional[list[str]] = None,
    ) -> Memory:
        """Add a new memory to the stream."""
        memory = Memory(
            id=self._next_id,
            tick=tick,
            day=day,
            time_str=time_str,
            type=memory_type,
            content=content,
            importance=importance,
            location=location,
            involved_agents=involved_agents or [],
        )
        self._next_id += 1
        self.memories.append(memory)
        self._importance_accumulator += importance

        # Prune if over capacity — drop lowest-importance, oldest memories
        if len(self.memories) > self.max_memories:
            self._prune()

        return memory

    def should_reflect(self) -> bool:
        """True if enough important things have happened to warrant a reflection."""
        return self._importance_accumulator >= self.reflection_threshold

    def reset_reflection_accumulator(self) -> None:
        self._importance_accumulator = 0.0

    def retrieve(
        self,
        current_tick: int,
        query: str = "",
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        involved_agent: Optional[str] = None,
    ) -> list[Memory]:
        """Retrieve top-K most relevant memories using recency + importance scoring.

        Score = recency_weight * recency + importance_weight * normalized_importance

        For a full implementation, relevance (embedding similarity to query) would be
        added as a third factor. For now, we use recency + importance only.
        """
        candidates = self.memories
        if memory_type:
            candidates = [m for m in candidates if m.type == memory_type]
        if involved_agent:
            candidates = [m for m in candidates if involved_agent in m.involved_agents]

        if not candidates:
            return []

        scored: list[tuple[float, Memory]] = []
        for mem in candidates:
            recency = self._recency_score(mem.tick, current_tick)
            importance = mem.importance / 10.0
            # Recency and importance weighted equally
            score = 0.5 * recency + 0.5 * importance
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in scored[:top_k]]

        # Update access tracking
        for mem in results:
            mem.access_count += 1
            mem.last_accessed_tick = current_tick

        return results

    def get_recent(self, n: int = 5) -> list[Memory]:
        """Get the N most recent memories."""
        return self.memories[-n:]

    def get_memories_about(self, agent_id: str, top_k: int = 5) -> list[Memory]:
        """Get memories involving a specific agent, most recent first."""
        relevant = [m for m in self.memories if agent_id in m.involved_agents]
        return relevant[-top_k:]

    def get_todays_plan(self, current_day: int) -> list[Memory]:
        """Get today's plan memories."""
        return [
            m for m in self.memories
            if m.type == MemoryType.PLAN and m.day == current_day
        ]

    def _recency_score(self, memory_tick: int, current_tick: int) -> float:
        """Exponential decay based on how many ticks ago the memory was formed."""
        age = current_tick - memory_tick
        # Decay factor: half-life of ~50 ticks (~12 hours at 15-min ticks)
        return math.exp(-0.014 * age)

    def _prune(self) -> None:
        """Remove least important, oldest memories when over capacity."""
        # Keep reflections and high-importance memories longer
        self.memories.sort(
            key=lambda m: (
                m.type == MemoryType.REFLECTION,  # Reflections last
                m.importance,
                m.tick,
            )
        )
        # Remove the bottom 10%
        cut = max(1, len(self.memories) - self.max_memories)
        self.memories = self.memories[cut:]
        # Re-sort by tick (chronological)
        self.memories.sort(key=lambda m: m.tick)

    def context_summary(self, current_tick: int, max_memories: int = 15) -> str:
        """Generate a context string of relevant memories for LLM prompts."""
        recent = self.retrieve(current_tick, top_k=max_memories)
        if not recent:
            return "No significant memories yet."

        lines = []
        for mem in recent:
            prefix = f"[Day {mem.day} {mem.time_str}]"
            lines.append(f"{prefix} ({mem.type.value}) {mem.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "next_id": self._next_id,
            "importance_accumulator": self._importance_accumulator,
            "reflection_threshold": self.reflection_threshold,
            "max_memories": self.max_memories,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryStream:
        stream = cls(max_memories=data.get("max_memories", 500))
        stream._next_id = data["next_id"]
        stream._importance_accumulator = data["importance_accumulator"]
        stream.reflection_threshold = data.get("reflection_threshold", 50.0)
        for md in data["memories"]:
            stream.memories.append(Memory.from_dict(md))
        return stream
