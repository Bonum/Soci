"""World events — random occurrences that inject entropy into the simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventSeverity(Enum):
    MINOR = "minor"       # Weather change, small talk topic
    MODERATE = "moderate"  # New shop opens, street performer, local news
    MAJOR = "major"       # Festival, power outage, celebrity visit
    CRITICAL = "critical"  # Emergency, natural disaster, evacuation


class WeatherState(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"


@dataclass
class WorldEvent:
    """A single event that occurs in the simulation world."""

    id: str
    name: str
    description: str
    severity: EventSeverity
    affected_locations: list[str] = field(default_factory=list)  # empty = city-wide
    duration_ticks: int = 1  # how long it persists
    remaining_ticks: int = 0

    def is_active(self) -> bool:
        return self.remaining_ticks > 0

    def tick(self) -> None:
        if self.remaining_ticks > 0:
            self.remaining_ticks -= 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "affected_locations": self.affected_locations,
            "duration_ticks": self.duration_ticks,
            "remaining_ticks": self.remaining_ticks,
        }


# Pool of possible random events
EVENT_TEMPLATES: list[dict] = [
    {
        "name": "Sudden Rain",
        "description": "Dark clouds roll in and rain begins to pour. People rush for cover.",
        "severity": EventSeverity.MINOR,
        "duration_ticks": 8,
    },
    {
        "name": "Street Musician",
        "description": "A talented street musician sets up and starts playing beautiful music.",
        "severity": EventSeverity.MINOR,
        "duration_ticks": 4,
        "location_zone": "public",
    },
    {
        "name": "Food Truck Arrives",
        "description": "A popular food truck parks nearby, filling the air with delicious aromas.",
        "severity": EventSeverity.MINOR,
        "duration_ticks": 6,
        "location_zone": "commercial",
    },
    {
        "name": "Lost Dog",
        "description": "A friendly but lost dog is wandering around, looking for its owner.",
        "severity": EventSeverity.MINOR,
        "duration_ticks": 12,
    },
    {
        "name": "Local Art Exhibition",
        "description": "A pop-up art exhibition opens, showcasing works by local artists.",
        "severity": EventSeverity.MODERATE,
        "duration_ticks": 16,
        "location_zone": "public",
    },
    {
        "name": "Neighborhood Meeting",
        "description": "A community meeting is called to discuss changes in the neighborhood.",
        "severity": EventSeverity.MODERATE,
        "duration_ticks": 4,
    },
    {
        "name": "Power Flicker",
        "description": "The power flickers briefly, causing momentary darkness and disruption.",
        "severity": EventSeverity.MODERATE,
        "duration_ticks": 2,
    },
    {
        "name": "New Shop Grand Opening",
        "description": "A new shop opens with fanfare, offering discounts and free samples.",
        "severity": EventSeverity.MODERATE,
        "duration_ticks": 8,
        "location_zone": "commercial",
    },
    {
        "name": "Summer Festival",
        "description": "The annual summer festival begins! Music, food stalls, and games fill the park.",
        "severity": EventSeverity.MAJOR,
        "duration_ticks": 24,
        "location_zone": "public",
    },
    {
        "name": "Power Outage",
        "description": "A major power outage hits the city. Businesses close early, streets go dark.",
        "severity": EventSeverity.MAJOR,
        "duration_ticks": 8,
    },
    {
        "name": "Celebrity Sighting",
        "description": "A famous celebrity is spotted in town, causing excitement and crowds.",
        "severity": EventSeverity.MAJOR,
        "duration_ticks": 4,
    },
    {
        "name": "Water Main Break",
        "description": "A water main breaks, flooding a street and disrupting traffic.",
        "severity": EventSeverity.CRITICAL,
        "duration_ticks": 12,
    },
    {
        "name": "Severe Storm Warning",
        "description": "Emergency alert: a severe storm is approaching. Seek shelter immediately.",
        "severity": EventSeverity.CRITICAL,
        "duration_ticks": 6,
    },
]


class EventSystem:
    """Manages world events, weather, and entropy injection."""

    def __init__(self, event_chance_per_tick: float = 0.08) -> None:
        self.event_chance = event_chance_per_tick
        self.weather: WeatherState = WeatherState.SUNNY
        self.active_events: list[WorldEvent] = []
        self._event_counter: int = 0

    def tick(self, city_location_ids: list[str]) -> list[WorldEvent]:
        """Process one tick: expire old events, maybe spawn new ones."""
        # Tick existing events
        for event in self.active_events:
            event.tick()
        self.active_events = [e for e in self.active_events if e.is_active()]

        new_events: list[WorldEvent] = []

        # Maybe change weather
        if random.random() < 0.03:
            old = self.weather
            self.weather = random.choice(list(WeatherState))
            if self.weather != old:
                evt = WorldEvent(
                    id=f"weather_{self._event_counter}",
                    name=f"Weather Change",
                    description=f"The weather shifts from {old.value} to {self.weather.value}.",
                    severity=EventSeverity.MINOR,
                    duration_ticks=1,
                    remaining_ticks=1,
                )
                self._event_counter += 1
                new_events.append(evt)

        # Maybe spawn a random event
        if random.random() < self.event_chance:
            template = random.choice(EVENT_TEMPLATES)
            # Pick affected location(s)
            affected: list[str] = []
            if "location_zone" in template:
                # Just pick a random location for now; the simulation can filter by zone
                if city_location_ids:
                    affected = [random.choice(city_location_ids)]
            elif random.random() < 0.5 and city_location_ids:
                affected = [random.choice(city_location_ids)]
            # else: city-wide

            evt = WorldEvent(
                id=f"event_{self._event_counter}",
                name=template["name"],
                description=template["description"],
                severity=template["severity"],
                affected_locations=affected,
                duration_ticks=template["duration_ticks"],
                remaining_ticks=template["duration_ticks"],
            )
            self._event_counter += 1
            self.active_events.append(evt)
            new_events.append(evt)

        return new_events

    def get_events_at(self, location_id: str) -> list[WorldEvent]:
        """Get active events affecting a specific location."""
        return [
            e for e in self.active_events
            if not e.affected_locations or location_id in e.affected_locations
        ]

    def get_world_description(self) -> str:
        """Summary of current world state for agent context."""
        parts = [f"Weather: {self.weather.value}."]
        for event in self.active_events:
            parts.append(f"[{event.severity.value.upper()}] {event.name}: {event.description}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "weather": self.weather.value,
            "active_events": [e.to_dict() for e in self.active_events],
            "event_counter": self._event_counter,
            "event_chance": self.event_chance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EventSystem:
        system = cls(event_chance_per_tick=data["event_chance"])
        system.weather = WeatherState(data["weather"])
        system._event_counter = data["event_counter"]
        for ed in data["active_events"]:
            evt = WorldEvent(
                id=ed["id"],
                name=ed["name"],
                description=ed["description"],
                severity=EventSeverity(ed["severity"]),
                affected_locations=ed["affected_locations"],
                duration_ticks=ed["duration_ticks"],
                remaining_ticks=ed["remaining_ticks"],
            )
            system.active_events.append(evt)
        return system
