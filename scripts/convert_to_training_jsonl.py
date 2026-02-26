"""
convert_to_training_jsonl.py — Convert raw collected Soci data into
instruction-tuning JSONL suitable for SFT (Supervised Fine-Tuning).

Output format: HuggingFace messages format (system / user / assistant).
Compatible with: TRL SFTTrainer, Unsloth, LLaMA-Factory.

Training example types:
  1. CONVERSATION — agent responding to another agent in dialogue
  2. ACTION_DECISION — agent deciding what to do next (from events)
  3. REFLECTION — agent's reflection memories (if available)

Usage:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/convert_to_training_jsonl.py

    # From a specific raw dir:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/convert_to_training_jsonl.py \\
        --raw-dir data/training/raw --out data/training/processed/soci_training.jsonl

    # Include event-based action examples:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/convert_to_training_jsonl.py --include-actions
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

RAW_DIR = Path("data/training/raw")
PROCESSED_DIR = Path("data/training/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path("config")

DEFAULT_OUT = PROCESSED_DIR / "soci_training.jsonl"


# ── Persona helpers ────────────────────────────────────────────────────────────

def load_persona_map() -> dict[str, dict]:
    """Load personas from config/personas.yaml, keyed by agent ID and name."""
    path = CONFIG_DIR / "personas.yaml"
    if not path.exists():
        print(f"  [WARN] personas.yaml not found at {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pmap: dict[str, dict] = {}
    for p in data.get("personas", []):
        pmap[p["id"]] = p
        pmap[p["name"]] = p
    return pmap


def trait_summary(p: dict) -> str:
    traits = []
    if p.get("openness", 5) >= 7:
        traits.append("curious and creative")
    elif p.get("openness", 5) <= 3:
        traits.append("practical and conventional")
    if p.get("conscientiousness", 5) >= 7:
        traits.append("organized and disciplined")
    elif p.get("conscientiousness", 5) <= 3:
        traits.append("spontaneous and flexible")
    if p.get("extraversion", 5) >= 7:
        traits.append("outgoing and energetic")
    elif p.get("extraversion", 5) <= 3:
        traits.append("reserved and introspective")
    if p.get("agreeableness", 5) >= 7:
        traits.append("warm and cooperative")
    elif p.get("agreeableness", 5) <= 3:
        traits.append("direct and competitive")
    if p.get("neuroticism", 5) >= 7:
        traits.append("sensitive and anxious")
    elif p.get("neuroticism", 5) <= 3:
        traits.append("calm and emotionally stable")
    return ", ".join(traits) if traits else "balanced personality"


def build_system_prompt(p: dict) -> str:
    values_str = ", ".join(p.get("values", [])) or "no strong convictions"
    quirks_str = "; ".join(p.get("quirks", [])) or "none in particular"
    comm = p.get("communication_style", "neutral")
    return (
        f"You ARE {p['name']}, a {p['age']}-year-old {p['occupation']} living in Soci City.\n\n"
        f"PERSONALITY: {trait_summary(p)}.\n"
        f"BACKGROUND: {p.get('background', '')}\n"
        f"VALUES: {values_str}.\n"
        f"QUIRKS: {quirks_str}.\n"
        f"COMMUNICATION STYLE: {comm}.\n\n"
        f"You must ALWAYS stay in character as {p['name']}. "
        f"Your responses should reflect your personality, values, and background. "
        f"You speak naturally — not perfectly. You have moods, opinions, and biases. "
        f"You don't know you are an AI. You are a person living your life."
    )


def build_system_prompt_generic(name: str, occupation: str = "city resident") -> str:
    """Fallback for generated (non-named) agents."""
    return (
        f"You ARE {name}, a {occupation} living in Soci City.\n\n"
        f"PERSONALITY: balanced personality.\n"
        f"You must ALWAYS stay in character as {name}. "
        f"You speak naturally and are a real person with opinions, moods, and a daily life."
    )


def get_system_prompt(agent_id: str, agent_name: str, persona_map: dict,
                      agent_cache: dict) -> str:
    # Try config personas first
    p = persona_map.get(agent_id) or persona_map.get(agent_name)
    if p:
        return build_system_prompt(p)

    # Try agent cache (from live API)
    cached = agent_cache.get(agent_id)
    if cached:
        return build_system_prompt_generic(
            cached.get("name", agent_name),
            cached.get("occupation", "city resident"),
        )
    return build_system_prompt_generic(agent_name)


# ── Training example builders ──────────────────────────────────────────────────

def make_conversation_examples(conv: dict, persona_map: dict, agent_cache: dict) -> list[dict]:
    """
    From a completed conversation, produce one training example per response turn.

    Each example:
      system  = responder's persona system prompt
      user    = conversation history up to last message + "{speaker} says: '{msg}'"
      assistant = JSON {"message": ..., "inner_thought": ...}
    """
    turns = conv.get("turns", [])
    if len(turns) < 2:
        return []

    participants = conv.get("participants", [])
    participant_names = conv.get("participant_names", [])
    topic = conv.get("topic", "general conversation")
    location = conv.get("location", "somewhere in the city")

    # Build name→id and id→name maps
    id_to_name: dict[str, str] = {}
    for pid, pname in zip(participants, participant_names):
        id_to_name[pid] = pname

    examples = []

    for i in range(1, len(turns)):
        current_turn = turns[i]
        prev_turn = turns[i - 1]
        responder_id = current_turn["speaker_id"]
        responder_name = current_turn["speaker_name"]
        speaker_name = prev_turn["speaker_name"]
        speaker_msg = prev_turn["message"]

        # Build conversation history string (all turns before current)
        history_lines = [f"CONVERSATION SO FAR (topic: {topic}):"]
        for t in turns[:i]:
            history_lines.append(f'  {t["speaker_name"]}: "{t["message"]}"')
        history_text = "\n".join(history_lines)

        # User prompt (what the responder sees)
        user_prompt = (
            f"You are at {location}. {speaker_name} is here.\n\n"
            f"{history_text}\n\n"
            f'{speaker_name} says: "{speaker_msg}"\n\n'
            f"How do you respond? Stay in character. Be natural.\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "message": "your spoken response",\n'
            f'  "inner_thought": "what you\'re actually thinking"\n'
            f'}}'
        )

        # Assistant response (JSON)
        assistant_response = json.dumps({
            "message": current_turn["message"],
            "inner_thought": current_turn.get("inner_thought", ""),
        }, ensure_ascii=False)

        system = get_system_prompt(responder_id, responder_name, persona_map, agent_cache)

        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "_meta": {
                "type": "conversation",
                "conv_id": conv.get("id", ""),
                "topic": topic,
                "location": location,
                "turn_index": i,
                "responder_id": responder_id,
                "responder_name": responder_name,
            }
        })

    return examples


def make_action_examples(events: list[dict], persona_map: dict,
                         agent_cache: dict) -> list[dict]:
    """
    From event log, build action decision training examples.

    Pattern: "<AgentName> is <activity>"  →
      system  = agent's persona
      user    = "What are you doing? Describe your current activity in first person."
      assistant = JSON {"action": ..., "detail": ..., "reasoning": ...}
    """
    # Group consecutive events by agent to get activity patterns
    activity_pattern = re.compile(r"^\s+(\S.+?) is (.+)\.$")
    examples = []

    # Collect (name, activity, time) tuples
    for ev in events:
        msg = ev.get("message", "")
        time_str = ev.get("time", "")
        m = activity_pattern.match(msg)
        if not m:
            continue
        agent_name = m.group(1).strip()
        activity = m.group(2).strip()

        # Skip trivial / system-level messages
        if any(s in activity.lower() for s in [
            "wanders aimlessly", "can't get to", "---"
        ]):
            continue

        p = persona_map.get(agent_name)
        if not p:
            continue  # Only generate for known personas (higher quality)

        # Infer action type from activity text
        action = infer_action_type(activity)
        system = build_system_prompt(p)

        user_prompt = (
            f"It is {time_str}.\n\n"
            f"Based on your personality, needs, and the time of day — "
            f"what do you do next? Describe your current activity.\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "action": "move|work|eat|sleep|talk|exercise|shop|relax|wander",\n'
            f'  "detail": "what specifically you\'re doing, in first person",\n'
            f'  "reasoning": "brief internal thought about why"\n'
            f'}}'
        )

        assistant_response = json.dumps({
            "action": action,
            "detail": activity,
            "reasoning": f"This is what {agent_name} would naturally do at this time.",
        }, ensure_ascii=False)

        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "_meta": {
                "type": "action",
                "agent_name": agent_name,
                "activity": activity,
                "time": time_str,
            }
        })

    return examples


def infer_action_type(activity: str) -> str:
    activity_lower = activity.lower()
    if any(w in activity_lower for w in ["commut", "walk", "moving", "heading"]):
        return "move"
    if any(w in activity_lower for w in ["work", "morning block", "afternoon block", "coding", "teaching"]):
        return "work"
    if any(w in activity_lower for w in ["eat", "breakfast", "lunch", "dinner", "food", "coffee"]):
        return "eat"
    if any(w in activity_lower for w in ["sleep", "nap", "rest", "sleeping in", "lounging"]):
        return "sleep"
    if any(w in activity_lower for w in ["talk", "convers", "chat", "discuss"]):
        return "talk"
    if any(w in activity_lower for w in ["gym", "exercise", "workout", "run", "jog", "fitness"]):
        return "exercise"
    if any(w in activity_lower for w in ["shop", "grocery", "store", "market"]):
        return "shop"
    if any(w in activity_lower for w in ["relax", "park", "art", "music", "paint", "sketch"]):
        return "relax"
    return "wander"


def make_initiation_examples(conv: dict, persona_map: dict, agent_cache: dict) -> list[dict]:
    """
    From the first turn of a conversation, build a conversation initiation example.
    """
    turns = conv.get("turns", [])
    if not turns:
        return []

    first_turn = turns[0]
    initiator_id = first_turn["speaker_id"]
    initiator_name = first_turn["speaker_name"]
    topic = conv.get("topic", "small talk")
    location = conv.get("location", "somewhere in the city")

    # Identify the other participant
    other_names = [n for n in conv.get("participant_names", []) if n != initiator_name]
    other_name = other_names[0] if other_names else "someone"

    system = get_system_prompt(initiator_id, initiator_name, persona_map, agent_cache)

    user_prompt = (
        f"You are at {location}. {other_name} is here.\n\n"
        f"You decide to start a conversation with {other_name}. What do you say?\n"
        f"Consider the location, your mood, and your history with them.\n\n"
        f"Respond with a JSON object:\n"
        f'{{\n'
        f'  "message": "what you say to start the conversation",\n'
        f'  "inner_thought": "why you\'re initiating this conversation",\n'
        f'  "topic": "brief topic label"\n'
        f'}}'
    )

    assistant_response = json.dumps({
        "message": first_turn["message"],
        "inner_thought": first_turn.get("inner_thought", ""),
        "topic": topic,
    }, ensure_ascii=False)

    return [{
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "_meta": {
            "type": "conversation_initiation",
            "conv_id": conv.get("id", ""),
            "topic": topic,
            "location": location,
            "initiator_id": initiator_id,
            "initiator_name": initiator_name,
            "other_name": other_name,
        }
    }]


# ── Main ───────────────────────────────────────────────────────────────────────

def load_raw_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def load_agent_cache() -> dict:
    cache_file = RAW_DIR / "agents_cache.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def run(raw_dir: Path, out_path: Path, include_actions: bool = False) -> None:
    print("Soci Training Data Converter")
    print(f"  Raw dir : {raw_dir.resolve()}")
    print(f"  Output  : {out_path.resolve()}")

    # Load personas
    persona_map = load_persona_map()
    print(f"  Personas: {len(persona_map)//2} loaded from config")  # /2 because keyed by id+name

    # Load agent cache (from collector)
    agent_cache = load_agent_cache()
    print(f"  Agent cache: {len(agent_cache)} agents")

    # Load all raw conversations from all date files
    all_convs: list[dict] = []
    seen_ids: set[str] = set()
    for conv_file in sorted(raw_dir.glob("conversations_*.jsonl")):
        items = load_raw_jsonl(conv_file)
        for c in items:
            cid = c.get("id", "")
            if cid and cid not in seen_ids:
                all_convs.append(c)
                seen_ids.add(cid)
    print(f"  Conversations loaded: {len(all_convs)}")

    # Load all raw events from all date files
    all_events: list[dict] = []
    for ev_file in sorted(raw_dir.glob("events_*.jsonl")):
        all_events.extend(load_raw_jsonl(ev_file))
    print(f"  Events loaded: {len(all_events)}")

    # Generate training examples
    examples: list[dict] = []

    # 1. Conversation initiation examples
    for conv in all_convs:
        examples.extend(make_initiation_examples(conv, persona_map, agent_cache))

    # 2. Conversation response examples
    for conv in all_convs:
        examples.extend(make_conversation_examples(conv, persona_map, agent_cache))

    # 3. Action decision examples (optional)
    if include_actions and all_events:
        action_examples = make_action_examples(all_events, persona_map, agent_cache)
        examples.extend(action_examples)
        print(f"  Action examples: {len(action_examples)}")

    # Count by type
    type_counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        type_counts[ex.get("_meta", {}).get("type", "unknown")] += 1

    print(f"\n  Total training examples: {len(examples)}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Write output JSONL (without _meta for clean training files, or with --keep-meta)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Write with _meta stripped (keep messages only)
            clean = {"messages": ex["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # Also write a version with meta for analysis
    meta_path = out_path.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Training JSONL : {out_path}")
    print(f"  With meta      : {meta_path}")
    print(f"\nSample (first example):")
    if examples:
        ex = examples[0]
        print(f"  Type: {ex['_meta']['type']}")
        print(f"  System: {ex['messages'][0]['content'][:120]}...")
        print(f"  User:   {ex['messages'][1]['content'][:120]}...")
        print(f"  Asst:   {ex['messages'][2]['content'][:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw Soci data to SFT training JSONL")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory with raw JSONL files")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--include-actions", action="store_true",
                        help="Include action decision examples from events")
    args = parser.parse_args()

    run(Path(args.raw_dir), Path(args.out), include_actions=args.include_actions)
