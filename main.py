"""Soci — LLM-powered city population simulator.

Usage:
    python main.py [--ticks N] [--agents N] [--speed SPEED]

Controls:
    Press Ctrl+C to pause and save the simulation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from soci.engine.llm import create_llm_client
from soci.engine.simulation import Simulation
from soci.persistence.database import Database
from soci.persistence.snapshots import save_simulation, load_simulation
from soci.world.city import City
from soci.world.clock import SimClock

load_dotenv()

console = Console()
logger = logging.getLogger("soci")


def build_dashboard(sim: Simulation, recent_events: list[str]) -> Layout:
    """Build the Rich layout for the live dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=5),
    )
    layout["body"].split_row(
        Layout(name="city", ratio=1),
        Layout(name="events", ratio=2),
    )

    # Header
    clock = sim.clock
    weather = sim.events.weather.value
    cost = f"${sim.llm.usage.estimated_cost_usd:.4f}"
    calls = sim.llm.usage.total_calls
    header_text = (
        f"  SOCI CITY  |  {clock.datetime_str} ({clock.time_of_day.value})  |  "
        f"Weather: {weather}  |  Agents: {len(sim.agents)}  |  "
        f"API calls: {calls}  |  Cost: {cost}"
    )
    layout["header"].update(Panel(header_text, style="bold white on blue"))

    # City locations table
    loc_table = Table(title="City Locations", expand=True, show_lines=True)
    loc_table.add_column("Location", style="cyan", width=20)
    loc_table.add_column("People", style="green")
    loc_table.add_column("#", style="yellow", width=3)

    for loc in sim.city.locations.values():
        occupants = []
        for aid in loc.occupants:
            agent = sim.agents.get(aid)
            if agent:
                state_icon = {
                    "idle": ".",
                    "working": "W",
                    "eating": "E",
                    "sleeping": "Z",
                    "socializing": "S",
                    "exercising": "X",
                    "in_conversation": "C",
                    "moving": ">",
                    "shopping": "$",
                    "relaxing": "~",
                }.get(agent.state.value, "?")
                occupants.append(f"{agent.name}[{state_icon}]")
        loc_table.add_row(
            loc.name,
            ", ".join(occupants) if occupants else "-",
            str(len(loc.occupants)),
        )

    layout["city"].update(Panel(loc_table))

    # Recent events
    event_text = "\n".join(recent_events[-25:]) if recent_events else "Simulation starting..."
    layout["events"].update(Panel(event_text, title="Recent Activity", border_style="green"))

    # Footer — agent mood/needs summary
    footer_parts = []
    for agent in list(sim.agents.values())[:10]:
        mood_bar = "+" * max(0, int((agent.mood + 1) * 3)) + "-" * max(0, int((1 - agent.mood) * 3))
        urgent = agent.needs.most_urgent
        footer_parts.append(f"{agent.name[:8]}: [{mood_bar}] need:{urgent[:4]}")
    footer_text = "  |  ".join(footer_parts)
    layout["footer"].update(Panel(footer_text, title="Agent Status", border_style="dim"))

    return layout


async def run_simulation(
    ticks: int = 96,
    max_agents: int = 100,
    tick_delay: float = 0.5,
    resume: bool = False,
    generate: bool = False,
    provider: str = "",
    model: str = "",
) -> None:
    """Run the simulation with a live Rich dashboard."""
    # Initialize
    console.print("[bold blue]Initializing Soci City Simulation...[/]")

    try:
        llm = create_llm_client(
            provider=provider or None,
            model=model or None,
        )
        console.print(f"[green]LLM provider: {llm.provider} (model: {llm.default_model})[/]")
    except (ValueError, ConnectionError) as e:
        console.print(f"[bold red]Error: {e}[/]")
        return

    db = Database()
    await db.connect()

    sim = None
    if resume:
        sim = await load_simulation(db, llm)
        if sim:
            console.print(f"[green]Resumed simulation from Day {sim.clock.day}, {sim.clock.time_str}[/]")

    if sim is None:
        # Create new simulation
        config_dir = Path(__file__).parent / "config"
        city = City.from_yaml(str(config_dir / "city.yaml"))
        clock = SimClock(tick_minutes=15, hour=6, minute=0)
        sim = Simulation(city=city, clock=clock, llm=llm)

        # Load YAML personas as the first 20 agents (backward compatible)
        sim.load_agents_from_yaml(str(config_dir / "personas.yaml"))
        yaml_count = len(sim.agents)
        console.print(f"[green]Loaded {yaml_count} YAML agents.[/]")

        # Generate additional agents if requested or if max_agents > yaml count
        gen_count = max_agents - yaml_count
        if (generate or gen_count > 0) and gen_count > 0:
            sim.generate_agents(gen_count)
            console.print(f"[green]Generated {gen_count} procedural agents ({len(sim.agents)} total).[/]")
        else:
            console.print(f"[green]Created new simulation with {len(sim.agents)} agents.[/]")

    # Limit agents if requested
    if max_agents < len(sim.agents):
        agent_ids = list(sim.agents.keys())[:max_agents]
        sim.agents = {aid: sim.agents[aid] for aid in agent_ids}
        console.print(f"[yellow]Limited to {max_agents} agents.[/]")

    # Collect all events for display
    all_events: list[str] = []

    def on_event(msg: str):
        all_events.append(msg)

    sim.on_event = on_event

    console.print(f"[bold green]Starting simulation: {ticks} ticks ({ticks * 15 // 60} hours)[/]")
    console.print("[dim]Press Ctrl+C to pause and save.[/]")

    try:
        with Live(build_dashboard(sim, all_events), refresh_per_second=2, console=console) as live:
            for tick_num in range(ticks):
                tick_events = await sim.tick()

                # Update display
                live.update(build_dashboard(sim, all_events))

                # Auto-save every 24 ticks (6 hours in-game)
                if tick_num > 0 and tick_num % 24 == 0:
                    await save_simulation(sim, db, "autosave")

                # Small delay so the dashboard is readable
                await asyncio.sleep(tick_delay)

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation paused.[/]")

    # Save on exit
    await save_simulation(sim, db, "autosave")

    # Print summary
    console.print("\n[bold blue]Simulation Summary[/]")
    console.print(f"  Time: {sim.clock.datetime_str}")
    console.print(f"  Total ticks: {sim.clock.total_ticks}")
    console.print(f"  {sim.llm.usage.summary()}")

    # Print agent summaries
    console.print("\n[bold]Agent Status:[/]")
    for agent in sim.agents.values():
        mood_emoji = "+" if agent.mood > 0.2 else ("-" if agent.mood < -0.2 else "~")
        loc = sim.city.get_location(agent.location)
        loc_name = loc.name if loc else agent.location
        console.print(
            f"  [{mood_emoji}] {agent.name} ({agent.persona.occupation}) "
            f"at {loc_name} — {agent.needs.describe()}"
        )

    await db.close()


def main():
    parser = argparse.ArgumentParser(description="Soci — City Population Simulator")
    parser.add_argument("--ticks", type=int, default=96, help="Number of ticks to simulate (default: 96 = 1 day)")
    parser.add_argument("--agents", type=int, default=100, help="Max number of agents (default: 100)")
    parser.add_argument("--speed", type=float, default=0.5, help="Delay between ticks in seconds (default: 0.5)")
    parser.add_argument("--resume", action="store_true", help="Resume from last save")
    parser.add_argument("--generate", action="store_true",
                        help="Generate procedural agents to fill up to --agents count")
    parser.add_argument("--provider", type=str, default="", choices=["", "claude", "groq", "ollama"],
                        help="LLM provider: claude, groq, or ollama (default: auto-detect)")
    parser.add_argument("--model", type=str, default="",
                        help="Model name (e.g. llama3.1:8b, mistral, qwen2.5)")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("data/soci.log", mode="a", encoding="utf-8")],
    )

    asyncio.run(run_simulation(
        ticks=args.ticks,
        max_agents=args.agents,
        tick_delay=args.speed,
        resume=args.resume,
        generate=args.generate,
        provider=args.provider,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
