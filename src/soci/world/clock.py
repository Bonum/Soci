"""Simulation clock — tracks in-game time with configurable tick duration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TimeOfDay(Enum):
    DAWN = "dawn"           # 5:00 - 7:59
    MORNING = "morning"     # 8:00 - 11:59
    AFTERNOON = "afternoon" # 12:00 - 16:59
    EVENING = "evening"     # 17:00 - 20:59
    NIGHT = "night"         # 21:00 - 4:59


@dataclass
class SimClock:
    """Tracks simulation time. One tick = tick_minutes of in-game time."""

    tick_minutes: int = 15
    day: int = 1
    hour: int = 6
    minute: int = 0
    _total_ticks: int = field(default=0, repr=False)

    def tick(self) -> None:
        """Advance time by one tick."""
        self._total_ticks += 1
        self.minute += self.tick_minutes
        while self.minute >= 60:
            self.minute -= 60
            self.hour += 1
        while self.hour >= 24:
            self.hour -= 24
            self.day += 1

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    @property
    def time_of_day(self) -> TimeOfDay:
        if 5 <= self.hour < 8:
            return TimeOfDay.DAWN
        elif 8 <= self.hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= self.hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= self.hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    @property
    def is_sleeping_hours(self) -> bool:
        return self.hour >= 23 or self.hour < 6

    @property
    def time_str(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"

    @property
    def datetime_str(self) -> str:
        return f"Day {self.day}, {self.time_str}"

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "tick_minutes": self.tick_minutes,
            "total_ticks": self._total_ticks,
            "time_of_day": self.time_of_day.value,
            "time_str": self.time_str,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SimClock:
        clock = cls(
            tick_minutes=data["tick_minutes"],
            day=data["day"],
            hour=data["hour"],
            minute=data["minute"],
        )
        clock._total_ticks = data["total_ticks"]
        return clock
