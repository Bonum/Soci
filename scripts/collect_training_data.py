"""
collect_training_data.py — Poll the running Soci simulation and save raw
conversation + event data for later training.

Polls every POLL_INTERVAL seconds, deduplicates by conversation ID,
and writes JSONL to data/training/raw/.

Usage:
    # Poll Render deployment (default):
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/collect_training_data.py

    # Poll a different base URL:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/collect_training_data.py --url http://localhost:8000

    # Run once (no loop):
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/collect_training_data.py --once
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

BASE_URL = "https://soci-tl3c.onrender.com"
POLL_INTERVAL = 30  # seconds
RAW_DIR = Path("data/training/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Files to accumulate into
today = datetime.now().strftime("%Y%m%d")
CONV_FILE = RAW_DIR / f"conversations_{today}.jsonl"
EVENT_FILE = RAW_DIR / f"events_{today}.jsonl"
AGENT_CACHE_FILE = RAW_DIR / "agents_cache.json"

# In-memory dedup sets (also rehydrated from disk on startup)
_seen_conv_ids: set[str] = set()
_seen_event_ticks_msgs: set[str] = set()


def fetch_json(url: str, timeout: int = 15) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  [WARN] fetch failed: {url} — {e}")
        return None
    except Exception as e:
        print(f"  [ERR] {url}: {e}")
        return None


def load_seen_ids() -> None:
    """Rehydrate dedup sets from existing JSONL files."""
    if CONV_FILE.exists():
        with open(CONV_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cid = d.get("id", "")
                    if cid:
                        _seen_conv_ids.add(cid)
                except json.JSONDecodeError:
                    pass
    if EVENT_FILE.exists():
        with open(EVENT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    key = f"{d.get('tick','')}|{d.get('message','')}"
                    _seen_event_ticks_msgs.add(key)
                except json.JSONDecodeError:
                    pass
    print(f"  Loaded dedup: {len(_seen_conv_ids)} convs, {len(_seen_event_ticks_msgs)} events")


def poll_conversations(base_url: str) -> int:
    """Fetch conversation history and save new ones. Returns count of new convs."""
    data = fetch_json(f"{base_url}/api/conversations?limit=200&include_history=true")
    if data is None:
        return 0

    new_count = 0
    with open(CONV_FILE, "a", encoding="utf-8") as f:
        for conv in data.get("active", []) + data.get("recent", []):
            cid = conv.get("id", "")
            if not cid or cid in _seen_conv_ids:
                continue
            if len(conv.get("turns", [])) < 2:
                # Skip single-turn (incomplete) conversations
                continue
            conv["_collected_at"] = datetime.now().isoformat()
            conv["_source"] = "api"
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            _seen_conv_ids.add(cid)
            new_count += 1

    return new_count


def poll_events(base_url: str) -> int:
    """Fetch recent events and save new ones. Returns count of new events."""
    data = fetch_json(f"{base_url}/api/events?limit=500")
    if data is None:
        return 0

    new_count = 0
    with open(EVENT_FILE, "a", encoding="utf-8") as f:
        for event in data.get("events", []):
            key = f"{event.get('tick','')}|{event.get('message','')}"
            if key in _seen_event_ticks_msgs:
                continue
            event["_collected_at"] = datetime.now().isoformat()
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            _seen_event_ticks_msgs.add(key)
            new_count += 1

    return new_count


def refresh_agent_cache(base_url: str) -> None:
    """Refresh the local agent persona cache (done once per session)."""
    agents_data = fetch_json(f"{base_url}/api/agents")
    if not agents_data:
        return
    # Fetch full detail for named (non-generated) agents
    full_agents = {}
    for aid in agents_data:
        detail = fetch_json(f"{base_url}/api/agents/{aid}")
        if detail:
            full_agents[aid] = detail
        time.sleep(0.2)  # Be gentle to the API

    AGENT_CACHE_FILE.write_text(
        json.dumps(full_agents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Agent cache refreshed: {len(full_agents)} agents -> {AGENT_CACHE_FILE}")


def print_stats() -> None:
    conv_count = 0
    if CONV_FILE.exists():
        with open(CONV_FILE, encoding="utf-8") as f:
            conv_count = sum(1 for line in f if line.strip())
    ev_count = 0
    if EVENT_FILE.exists():
        with open(EVENT_FILE, encoding="utf-8") as f:
            ev_count = sum(1 for line in f if line.strip())
    print(f"  Stats: {conv_count} convs, {ev_count} events saved")


def run(base_url: str, once: bool = False, skip_agent_cache: bool = False) -> None:
    print(f"Soci Training Data Collector")
    print(f"  Target: {base_url}")
    print(f"  Output: {RAW_DIR.resolve()}")
    print(f"  Poll interval: {POLL_INTERVAL}s")

    load_seen_ids()

    if not skip_agent_cache:
        print("  Refreshing agent cache...")
        refresh_agent_cache(base_url)

    iteration = 0
    try:
        while True:
            iteration += 1
            ts = datetime.now().strftime("%H:%M:%S")
            new_convs = poll_conversations(base_url)
            new_events = poll_events(base_url)
            print(f"[{ts}] iter={iteration}  +{new_convs} convs  +{new_events} events  "
                  f"(total: {len(_seen_conv_ids)} convs, {len(_seen_event_ticks_msgs)} events)")

            if once:
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print_stats()
    print(f"\nConversations: {CONV_FILE}")
    print(f"Events:        {EVENT_FILE}")
    print(f"Agent cache:   {AGENT_CACHE_FILE}")
    print(f"\nNext step: run  python scripts/convert_to_training_jsonl.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soci training data collector")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the Soci API")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit")
    parser.add_argument("--no-agent-cache", action="store_true", help="Skip agent cache refresh")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help="Poll interval in seconds (default 30)")
    args = parser.parse_args()

    POLL_INTERVAL = args.interval
    run(args.url, once=args.once, skip_agent_cache=args.no_agent_cache)
