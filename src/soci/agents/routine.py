"""Daily routine system — deterministic schedules based on persona traits."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from soci.agents.generator import (
    EVENING_SHIFT_JOBS, STUDENT_OCCUPATIONS, RETIRED_OCCUPATIONS,
)

if TYPE_CHECKING:
    from soci.agents.persona import Persona


@dataclass
class RoutineSlot:
    """A single block in an agent's daily schedule."""

    hour: int           # Start hour (0-23)
    minute: int         # Start minute (0 or 15 or 30 or 45)
    action_type: str    # sleep, eat, work, move, relax, exercise, etc.
    target_location: str  # Where to do it
    duration_ticks: int   # How many 15-min ticks
    detail: str         # Human-readable description
    needs_satisfied: dict[str, float] = field(default_factory=dict)

    @property
    def end_minutes(self) -> int:
        """Total minutes from midnight when this slot ends."""
        return self.hour * 60 + self.minute + self.duration_ticks * 15


class DailyRoutine:
    """A deterministic daily schedule built from persona traits.

    The routine is fully derived from the persona, so it doesn't need
    serialization — it's rebuilt on load.
    """

    def __init__(self, persona: Persona, is_weekend: bool = False) -> None:
        self.persona_id = persona.id
        self.slots: list[RoutineSlot] = []
        self.wake_hour: int = 7
        self.sleep_hour: int = 22
        self._rng = random.Random(hash(persona.id))  # Deterministic per persona
        self._build_routine(persona, is_weekend)

    def get_action_for_time(self, hour: int, minute: int) -> Optional[RoutineSlot]:
        """Return the routine slot active at the given time, or None."""
        time_mins = hour * 60 + minute
        for slot in self.slots:
            slot_start = slot.hour * 60 + slot.minute
            slot_end = slot_start + slot.duration_ticks * 15
            if slot_start <= time_mins < slot_end:
                return slot
        return None

    def is_awake_at(self, hour: int) -> bool:
        """Whether this agent should be awake at the given hour."""
        if self.wake_hour <= self.sleep_hour:
            return self.wake_hour <= hour < self.sleep_hour
        else:
            # Wraps past midnight (e.g. wake=14, sleep=2)
            return hour >= self.wake_hour or hour < self.sleep_hour

    def _jitter(self, base_minutes: int, spread: int = 30) -> tuple[int, int]:
        """Add random jitter to a time. Returns (hour, minute) snapped to 15-min."""
        offset = self._rng.randint(-spread, spread)
        total = max(0, min(23 * 60 + 45, base_minutes + offset))
        # Snap to 15-minute intervals
        total = (total // 15) * 15
        return total // 60, total % 60

    def _build_routine(self, persona: Persona, is_weekend: bool) -> None:
        """Build the full daily schedule from persona traits."""
        c = persona.conscientiousness
        e = persona.extraversion
        home = persona.home_location
        work = persona.work_location

        is_evening_shift = persona.occupation.lower() in EVENING_SHIFT_JOBS
        is_student = persona.occupation.lower() in STUDENT_OCCUPATIONS
        is_retired = persona.occupation.lower() in RETIRED_OCCUPATIONS

        # --- Determine wake/sleep times ---
        if is_evening_shift:
            base_wake = 10 * 60  # 10:00
            base_sleep = 2 * 60  # 02:00 (next day, treated as 26:00)
        elif is_student:
            base_wake = 8 * 60 + 30  # 08:30
            base_sleep = 23 * 60 + 30
        elif is_retired:
            base_wake = 7 * 60  # 07:00
            base_sleep = 21 * 60 + 30
        else:
            # High conscientiousness → early riser
            base_wake = 6 * 60 + (10 - c) * 15  # 6:00 (c=10) to 7:15 (c=5) to 8:15+ (c=1)
            base_sleep = 22 * 60

        wake_h, wake_m = self._jitter(base_wake, 20)
        sleep_h, sleep_m = self._jitter(base_sleep, 30)
        self.wake_hour = wake_h
        self.sleep_hour = sleep_h

        if is_weekend or is_retired:
            self._build_leisure_day(persona, home, work, wake_h, wake_m, sleep_h, sleep_m,
                                    is_retired)
        elif is_evening_shift:
            self._build_evening_shift(persona, home, work, wake_h, wake_m, sleep_h, sleep_m)
        else:
            self._build_work_day(persona, home, work, wake_h, wake_m, sleep_h, sleep_m,
                                 is_student)

    def _add(self, hour: int, minute: int, action: str, location: str,
             ticks: int, detail: str, needs: dict[str, float] | None = None) -> int:
        """Add a slot and return the end time in minutes."""
        self.slots.append(RoutineSlot(
            hour=hour, minute=minute, action_type=action,
            target_location=location, duration_ticks=ticks, detail=detail,
            needs_satisfied=needs or {},
        ))
        return hour * 60 + minute + ticks * 15

    def _build_work_day(self, persona: Persona, home: str, work: str,
                        wake_h: int, wake_m: int, sleep_h: int, sleep_m: int,
                        is_student: bool) -> None:
        """Standard 9-to-5 (ish) work day."""
        t = wake_h * 60 + wake_m

        # Wake up + morning routine
        h, m = t // 60, t % 60
        t = self._add(h, m, "relax", home, 2, "Morning routine — getting ready",
                       {"comfort": 0.1, "energy": 0.05})

        # Breakfast
        h, m = t // 60, t % 60
        t = self._add(h, m, "eat", home, 2, "Having breakfast at home",
                       {"hunger": 0.4})

        # Commute to work
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", work, 1, f"Commuting to {work}",
                       {})

        # Morning work block
        work_label = "Studying" if is_student else "Working"
        morning_ticks = self._rng.randint(7, 10)
        h, m = t // 60, t % 60
        t = self._add(h, m, "work", work, morning_ticks,
                       f"{work_label} — morning block",
                       {"purpose": 0.3})

        # Lunch — pick a food place or stay at work
        food_places = ["cafe", "restaurant", "grocery", "bakery"]
        lunch_spot = self._rng.choice(food_places)
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", lunch_spot, 1, f"Walking to lunch at {lunch_spot}",
                       {})
        h, m = t // 60, t % 60
        t = self._add(h, m, "eat", lunch_spot, 2, "Lunch break",
                       {"hunger": 0.5, "social": 0.1})

        # Walk back to work
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", work, 1, f"Walking back to {work}",
                       {})

        # Afternoon work block
        afternoon_ticks = self._rng.randint(6, 9)
        h, m = t // 60, t % 60
        t = self._add(h, m, "work", work, afternoon_ticks,
                       f"{work_label} — afternoon block",
                       {"purpose": 0.3})

        # Commute home
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", home, 1, "Heading home",
                       {})

        # Dinner
        h, m = t // 60, t % 60
        t = self._add(h, m, "eat", home, 2, "Having dinner at home",
                       {"hunger": 0.5})

        # Evening entertainment
        t = self._add_evening_block(persona, home, t, sleep_h, sleep_m)

        # Sleep
        h, m = t // 60, t % 60
        # Sleep until wake time next day — approximate with 20 ticks
        sleep_ticks = max(4, ((24 * 60 - t) + self.wake_hour * 60) // 15)
        sleep_ticks = min(sleep_ticks, 32)  # Cap at 8 hours
        self._add(h, m, "sleep", home, sleep_ticks, "Sleeping",
                  {"energy": 0.8})

    def _build_evening_shift(self, persona: Persona, home: str, work: str,
                             wake_h: int, wake_m: int,
                             sleep_h: int, sleep_m: int) -> None:
        """Evening shift workers: bartenders, chefs, etc."""
        t = wake_h * 60 + wake_m

        # Late morning routine
        h, m = t // 60, t % 60
        t = self._add(h, m, "relax", home, 2, "Late morning routine",
                       {"comfort": 0.1, "energy": 0.05})

        # Brunch
        h, m = t // 60, t % 60
        t = self._add(h, m, "eat", home, 2, "Having brunch",
                       {"hunger": 0.4})

        # Free time before shift
        t = self._add_leisure_block(persona, home, t, min(t + 4 * 15, 15 * 60))

        # Commute to work
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", work, 1, f"Heading to {work} for shift",
                       {})

        # Evening shift (long)
        shift_ticks = self._rng.randint(12, 16)
        h, m = t // 60, t % 60
        t = self._add(h, m, "work", work, shift_ticks,
                       f"Working the evening shift at {work}",
                       {"purpose": 0.4})

        # Head home
        h, m = (t // 60) % 24, t % 60
        t = self._add(h, m, "move", home, 1, "Heading home after shift",
                       {})

        # Late snack + sleep
        h, m = (t // 60) % 24, t % 60
        t = self._add(h, m, "eat", home, 1, "Late night snack",
                       {"hunger": 0.3})
        h, m = (t // 60) % 24, t % 60
        self._add(h, m, "sleep", home, 20, "Sleeping",
                  {"energy": 0.8})

    def _build_leisure_day(self, persona: Persona, home: str, work: str,
                           wake_h: int, wake_m: int, sleep_h: int, sleep_m: int,
                           is_retired: bool) -> None:
        """Weekend or retired day — no work, loose schedule, more entertainment."""
        t = wake_h * 60 + wake_m
        e = persona.extraversion
        o = persona.openness

        # Sleep in extra on weekends (not retired — they keep regular hours)
        if not is_retired:
            t += self._rng.randint(30, 60)
            # Snap to 15-min
            t = (t // 15) * 15

        # Slow morning
        h, m = t // 60, t % 60
        t = self._add(h, m, "relax", home, 3, "Lazy morning — sleeping in and lounging",
                       {"comfort": 0.25, "energy": 0.15})

        # Late breakfast / brunch
        brunch_out = e >= 6 and self._rng.random() < 0.5
        if brunch_out:
            brunch_spot = self._rng.choice(["cafe", "restaurant"])
            h, m = t // 60, t % 60
            t = self._add(h, m, "move", brunch_spot, 1, f"Going to {brunch_spot} for brunch",
                           {})
            h, m = t // 60, t % 60
            t = self._add(h, m, "eat", brunch_spot, 3, f"Enjoying brunch at {brunch_spot}",
                           {"hunger": 0.5, "social": 0.2})
            h, m = t // 60, t % 60
            t = self._add(h, m, "move", home, 1, "Heading home",
                           {})
        else:
            h, m = t // 60, t % 60
            t = self._add(h, m, "eat", home, 2, "Leisurely breakfast at home",
                           {"hunger": 0.4})

        # Morning/early afternoon activity — longer and more varied than weekdays
        morning_end = t + self._rng.randint(8, 14) * 15
        t = self._add_leisure_block(persona, home, t, morning_end)

        # Lunch — target ~12:30-13:30
        lunch_target = 12 * 60 + 30
        if t < lunch_target:
            gap_ticks = (lunch_target - t) // 15
            if gap_ticks > 0:
                h, m = t // 60, t % 60
                t = self._add(h, m, "relax", home, gap_ticks, "Chilling at home",
                               {"comfort": 0.1, "fun": 0.05})

        lunch_spot = self._rng.choice(["cafe", "restaurant", "park", "bakery", "town_square"])
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", lunch_spot, 1, f"Heading to {lunch_spot}",
                       {})
        h, m = t // 60, t % 60
        t = self._add(h, m, "eat", lunch_spot, 2, "Lunch out",
                       {"hunger": 0.5, "social": 0.15})

        # Afternoon — extroverts do social things, introverts do solo things
        if e >= 6:
            # Social afternoon: visit multiple places
            afternoon_places = self._rng.sample(
                ["park", "cafe", "bar", "gym", "library", "cinema",
                 "town_square", "sports_field"],
                k=min(2, self._rng.randint(1, 2)),
            )
            for place in afternoon_places:
                h, m = t // 60, t % 60
                t = self._add(h, m, "move", place, 1, f"Going to {place}",
                               {})
                act_ticks = self._rng.randint(3, 6)
                act_type = "exercise" if place == "gym" else "relax"
                h, m = t // 60, t % 60
                t = self._add(h, m, act_type, place, act_ticks,
                               f"Hanging out at {place}",
                               {"social": 0.2, "fun": 0.25})
        else:
            # Quiet afternoon
            afternoon_end = t + self._rng.randint(8, 16) * 15
            t = self._add_leisure_block(persona, home, t, afternoon_end)

        # Dinner — target ~18:00-19:30 (later on weekends)
        dinner_target = 18 * 60 + self._rng.randint(0, 6) * 15
        if t < dinner_target:
            gap_ticks = (dinner_target - t) // 15
            if gap_ticks > 0:
                h, m = t // 60, t % 60
                t = self._add(h, m, "relax", home, gap_ticks, "Relaxing at home",
                               {"comfort": 0.15, "fun": 0.1})

        # Weekend dinner — more likely to eat out
        dinner_out = e >= 5 or self._rng.random() < 0.3
        if dinner_out:
            dinner_spot = self._rng.choice(["restaurant", "bar"])
            h, m = t // 60, t % 60
            t = self._add(h, m, "move", dinner_spot, 1, f"Going to {dinner_spot} for dinner",
                           {})
            h, m = t // 60, t % 60
            t = self._add(h, m, "eat", dinner_spot, 3, f"Dinner at {dinner_spot}",
                           {"hunger": 0.5, "social": 0.2})
        else:
            h, m = t // 60, t % 60
            t = self._add(h, m, "eat", home, 2, "Dinner at home",
                           {"hunger": 0.5})

        # Weekend evening — extended entertainment, later bedtime
        weekend_sleep_h = min(23, sleep_h + 1)  # Stay up later on weekends
        weekend_sleep_m = sleep_m
        t = self._add_evening_block(persona, home, t, weekend_sleep_h, weekend_sleep_m)

        # Sleep
        h, m = t // 60, t % 60
        self._add(h, m, "sleep", home, 28, "Sleeping",
                  {"energy": 0.8})

    def _add_evening_block(self, persona: Persona, home: str,
                           t: int, sleep_h: int, sleep_m: int) -> int:
        """Add evening entertainment filling the full period until sleep.

        Extroverts go out then wind down; introverts stay home the whole time.
        Returns updated time in minutes.
        """
        e = persona.extraversion
        sleep_t = sleep_h * 60 + sleep_m
        if sleep_t <= t:
            return t

        if e >= 6:
            # Extroverts: go out, stay until ~30-45 min before sleep, then come home
            venue = self._rng.choice(["bar", "restaurant", "park", "cinema", "town_square"])
            h, m = t // 60, t % 60
            t = self._add(h, m, "move", venue, 1, f"Heading to {venue}", {})
            wind_down_start = sleep_t - self._rng.randint(2, 3) * 15
            ent_ticks = max(0, min((wind_down_start - t) // 15, self._rng.randint(4, 8)))
            if ent_ticks > 0:
                h, m = t // 60, t % 60
                t = self._add(h, m, "relax", venue, ent_ticks,
                               f"Socializing at {venue}",
                               {"fun": 0.3, "social": 0.3})
            # Head home
            if t < sleep_t:
                h, m = t // 60, t % 60
                t = self._add(h, m, "move", home, 1, "Heading home", {})

        # Wind down at home until sleep (covers all remaining time)
        wind_down_ticks = max(0, (sleep_t - t) // 15)
        if wind_down_ticks > 0:
            h, m = t // 60, t % 60
            detail = self._rng.choice([
                "Reading before bed", "Watching TV", "Browsing the internet",
                "Journaling", "Listening to music", "Winding down at home",
            ])
            t = self._add(h, m, "relax", home, wind_down_ticks, detail,
                           {"comfort": 0.25, "fun": 0.15, "energy": 0.05})

        return t

    def _add_leisure_block(self, persona: Persona, home: str,
                           t: int, end_t: int) -> int:
        """Fill a leisure period with activities based on personality."""
        activities = []
        if persona.extraversion >= 6:
            activities.extend(["park", "cafe", "gym", "town_square", "sports_field"])
        else:
            activities.extend(["library", "park", "church"])
        if persona.conscientiousness >= 6:
            activities.append("gym")
        if persona.openness >= 6:
            activities.extend(["library", "park", "cinema"])

        dest = self._rng.choice(activities)
        available_ticks = max(0, (end_t - t) // 15)

        if available_ticks <= 1:
            return t

        # Move to destination
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", dest, 1, f"Going to {dest}",
                       {})
        available_ticks -= 1

        # Activity there
        act_ticks = min(available_ticks - 1, self._rng.randint(2, max(3, available_ticks - 1)))
        if act_ticks > 0:
            act_type = "exercise" if dest == "gym" else "relax"
            act_detail = {
                "park": "Taking a walk in the park",
                "cafe": "Hanging out at the cafe",
                "gym": "Working out at the gym",
                "library": "Reading at the library",
                "cinema": "Watching a movie",
                "town_square": "People-watching at the square",
                "sports_field": "Playing sports at the field",
                "church": "Quiet time at the church",
            }.get(dest, f"Spending time at {dest}")
            needs = {
                "park": {"fun": 0.2, "comfort": 0.1},
                "cafe": {"social": 0.2, "fun": 0.1},
                "gym": {"energy": -0.1, "fun": 0.2},
                "library": {"fun": 0.15, "comfort": 0.1},
                "cinema": {"fun": 0.3, "social": 0.1},
                "town_square": {"social": 0.2, "fun": 0.15},
                "sports_field": {"fun": 0.25, "energy": -0.1},
                "church": {"comfort": 0.2, "purpose": 0.1},
            }.get(dest, {"fun": 0.1})
            h, m = t // 60, t % 60
            t = self._add(h, m, act_type, dest, act_ticks, act_detail, needs)

        # Head back home
        h, m = t // 60, t % 60
        t = self._add(h, m, "move", home, 1, "Heading home",
                       {})

        return t


def check_motivation_override(
    slot: RoutineSlot,
    needs: "NeedsState",
    mood: float,
    extraversion: int,
    conscientiousness: int,
) -> Optional[RoutineSlot]:
    """Check if an agent's needs or mood are strong enough to override their routine.

    Returns an alternative RoutineSlot if motivated to deviate, or None to follow routine.
    High conscientiousness agents are harder to derail. Low mood or critical needs
    can force deviation.
    """
    from soci.agents.needs import NeedsState  # avoid circular at module level

    # Conscientiousness determines resistance to deviation (0.3 to 0.9)
    resistance = 0.3 + conscientiousness * 0.06

    # Check critical needs — these override regardless of personality
    if needs.hunger < 0.15 and slot.action_type not in ("eat", "move", "sleep"):
        return RoutineSlot(
            hour=slot.hour, minute=slot.minute, action_type="eat",
            target_location=slot.target_location, duration_ticks=2,
            detail="Desperately need to eat — skipping routine",
            needs_satisfied={"hunger": 0.5},
        )

    if needs.energy < 0.1 and slot.action_type not in ("sleep", "relax"):
        return RoutineSlot(
            hour=slot.hour, minute=slot.minute, action_type="sleep",
            target_location=slot.target_location, duration_ticks=4,
            detail="Exhausted — need to rest",
            needs_satisfied={"energy": 0.5},
        )

    # Social need override — extroverts are more driven by social needs
    social_threshold = 0.25 if extraversion >= 7 else 0.15
    if (needs.social < social_threshold and slot.action_type in ("work", "relax")
            and random.random() > resistance):
        return RoutineSlot(
            hour=slot.hour, minute=slot.minute, action_type="move",
            target_location="cafe", duration_ticks=1,
            detail="Feeling lonely — heading somewhere social",
            needs_satisfied={},
        )

    # Very bad mood can derail work (skip to relax)
    if (mood < -0.5 and slot.action_type == "work"
            and random.random() > resistance):
        return RoutineSlot(
            hour=slot.hour, minute=slot.minute, action_type="relax",
            target_location=slot.target_location, duration_ticks=slot.duration_ticks,
            detail="Not feeling up to work — taking it easy",
            needs_satisfied={"comfort": 0.2, "fun": 0.1},
        )

    # Fun deprivation — spontaneous entertainment
    if (needs.fun < 0.15 and slot.action_type == "work"
            and random.random() > resistance + 0.1):
        return RoutineSlot(
            hour=slot.hour, minute=slot.minute, action_type="relax",
            target_location="park", duration_ticks=2,
            detail="Need a break — going for a walk",
            needs_satisfied={"fun": 0.2, "comfort": 0.1},
        )

    return None


def build_routine(persona: Persona, day: int) -> DailyRoutine:
    """Build a daily routine for a persona. Weekend every 6th-7th day."""
    is_weekend = (day % 7) in (6, 0)  # Days 6 and 7 are weekends
    return DailyRoutine(persona, is_weekend=is_weekend)
