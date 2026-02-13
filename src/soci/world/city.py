"""City map — locations, zones, and connections between them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Location:
    """A place in the city where agents can be."""

    id: str
    name: str
    zone: str  # residential, commercial, public, work
    description: str
    capacity: int = 20
    connected_to: list[str] = field(default_factory=list)
    # Current occupants (agent IDs)
    occupants: list[str] = field(default_factory=list, repr=False)

    @property
    def is_full(self) -> bool:
        return len(self.occupants) >= self.capacity

    @property
    def occupant_count(self) -> int:
        return len(self.occupants)

    def add_occupant(self, agent_id: str) -> None:
        if agent_id not in self.occupants:
            self.occupants.append(agent_id)

    def remove_occupant(self, agent_id: str) -> None:
        if agent_id in self.occupants:
            self.occupants.remove(agent_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "zone": self.zone,
            "description": self.description,
            "capacity": self.capacity,
            "connected_to": self.connected_to,
            "occupants": list(self.occupants),
        }


class City:
    """The city map — a graph of connected locations."""

    def __init__(self, name: str = "Soci City") -> None:
        self.name = name
        self.locations: dict[str, Location] = {}

    def add_location(self, location: Location) -> None:
        self.locations[location.id] = location

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_connected(self, location_id: str) -> list[Location]:
        loc = self.locations.get(location_id)
        if not loc:
            return []
        return [self.locations[cid] for cid in loc.connected_to if cid in self.locations]

    def get_locations_in_zone(self, zone: str) -> list[Location]:
        return [loc for loc in self.locations.values() if loc.zone == zone]

    def get_agents_at(self, location_id: str) -> list[str]:
        loc = self.locations.get(location_id)
        return list(loc.occupants) if loc else []

    def move_agent(self, agent_id: str, from_id: str, to_id: str) -> bool:
        """Move an agent between locations. Returns True if successful."""
        from_loc = self.locations.get(from_id)
        to_loc = self.locations.get(to_id)
        if not from_loc or not to_loc:
            return False
        if to_loc.is_full:
            return False
        from_loc.remove_occupant(agent_id)
        to_loc.add_occupant(agent_id)
        return True

    def place_agent(self, agent_id: str, location_id: str) -> bool:
        """Place an agent at a location (initial placement)."""
        loc = self.locations.get(location_id)
        if not loc or loc.is_full:
            return False
        loc.add_occupant(agent_id)
        return True

    def find_agent(self, agent_id: str) -> Optional[str]:
        """Find which location an agent is at. Returns location_id or None."""
        for loc in self.locations.values():
            if agent_id in loc.occupants:
                return loc.id
        return None

    def describe_location(self, location_id: str, exclude_agent: str = "") -> str:
        """Get a natural language description of a location and who's there."""
        loc = self.locations.get(location_id)
        if not loc:
            return "Unknown location."
        others = [a for a in loc.occupants if a != exclude_agent]
        desc = f"{loc.name} — {loc.description}"
        if others:
            desc += f" Present: {', '.join(others)}."
        else:
            desc += " No one else is here."
        return desc

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "locations": {lid: loc.to_dict() for lid, loc in self.locations.items()},
        }

    @classmethod
    def from_yaml(cls, path: str) -> City:
        """Load city from a YAML config file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        city = cls(name=data.get("name", "Soci City"))
        for loc_data in data.get("locations", []):
            location = Location(
                id=loc_data["id"],
                name=loc_data["name"],
                zone=loc_data.get("zone", "public"),
                description=loc_data.get("description", ""),
                capacity=loc_data.get("capacity", 20),
                connected_to=loc_data.get("connected_to", []),
            )
            city.add_location(location)
        return city

    @classmethod
    def from_dict(cls, data: dict) -> City:
        city = cls(name=data["name"])
        for loc_data in data["locations"].values():
            location = Location(
                id=loc_data["id"],
                name=loc_data["name"],
                zone=loc_data["zone"],
                description=loc_data["description"],
                capacity=loc_data["capacity"],
                connected_to=loc_data["connected_to"],
            )
            location.occupants = loc_data.get("occupants", [])
            city.add_location(location)
        return city
