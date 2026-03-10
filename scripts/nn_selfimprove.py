#!/usr/bin/env python3
"""Soci Agent NN — Self-Improvement Pipeline

Collects training data from the live simulation, retrains the ONNX model,
and pushes the improved version back to HuggingFace Hub.

Three modes:
    python nn_selfimprove.py collect   — Watch live sim, collect training samples
    python nn_selfimprove.py train     — Retrain NN on collected data
    python nn_selfimprove.py push      — Push improved model to HF Hub
    python nn_selfimprove.py all       — Do all three in sequence

Requires: pip install torch onnx onnxruntime httpx huggingface_hub numpy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nn_selfimprove")

# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "nn_training"
SAMPLES_FILE = DATA_DIR / "collected_samples.jsonl"
MODEL_DIR = PROJECT_DIR / "models"
BEST_PT = MODEL_DIR / "soci_agent_best.pt"
ONNX_PATH = MODEL_DIR / "soci_agent.onnx"

# ── Domain constants (must match nn_client.py and notebook) ──────────────

ACTION_TYPES = ["move", "work", "eat", "sleep", "talk", "exercise", "shop", "relax", "wander"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

LOCATIONS = [
    "house_elena", "house_marcus", "house_helen", "house_diana", "house_kai",
    "house_priya", "house_james", "house_rosa", "house_yuki", "house_frank",
    "apartment_block_1", "apartment_block_2", "apartment_block_3",
    "apt_northeast", "apt_northwest", "apt_southeast", "apt_southwest",
    "cafe", "grocery", "bar", "restaurant", "bakery", "cinema", "diner", "pharmacy",
    "office", "office_tower", "factory", "school", "hospital",
    "park", "gym", "library", "church", "town_square", "sports_field",
    "street_north", "street_south", "street_east", "street_west",
]
LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCATIONS)}
NUM_LOCATIONS = len(LOCATIONS)

NEED_NAMES = ["hunger", "energy", "social", "purpose", "comfort", "fun"]
ACTION_DURATIONS = {"move": 1, "work": 4, "eat": 2, "sleep": 8, "talk": 2, "exercise": 3, "shop": 2, "relax": 2, "wander": 1}

FEATURE_DIM = 47
NUM_ACTIONS = len(ACTION_TYPES)

# ── Feature encoding (same as nn_client.py) ──────────────────────────────

def _time_period(hour: int) -> int:
    if hour < 6: return 0
    if hour < 9: return 1
    if hour < 12: return 2
    if hour < 14: return 3
    if hour < 18: return 4
    if hour < 22: return 5
    return 6


def encode_features(
    personality: dict, age: float, hour: int, minute: int, day: int,
    needs: dict, mood: float, current_loc: str,
    home_loc: str = "", work_loc: str = "", num_people: int = 0,
) -> list[float]:
    """Encode agent state into 47-dim feature vector."""
    f: list[float] = []
    f.append(personality.get("openness", 5) / 10.0)
    f.append(personality.get("conscientiousness", 5) / 10.0)
    f.append(personality.get("extraversion", 5) / 10.0)
    f.append(personality.get("agreeableness", 5) / 10.0)
    f.append(personality.get("neuroticism", 5) / 10.0)
    f.append(age / 100.0)
    f.append(math.sin(2 * math.pi * hour / 24))
    f.append(math.cos(2 * math.pi * hour / 24))
    f.append(math.sin(2 * math.pi * minute / 60))
    f.append(math.cos(2 * math.pi * minute / 60))
    dow = (day - 1) % 7
    f.append(dow / 7.0)
    f.append(1.0 if dow >= 5 else 0.0)
    for n in NEED_NAMES:
        f.append(needs.get(n, 0.5))
    f.append(max(-1.0, min(1.0, mood)))
    vals = [needs.get(n, 0.5) for n in NEED_NAMES]
    urgent_idx = int(np.argmin(vals))
    f.append(urgent_idx / 5.0)
    f.append(1.0 if any(v < 0.15 for v in vals) else 0.0)
    zone = 0 if current_loc.startswith(("house_", "apartment_", "apt_")) else (
        1 if current_loc in ("cafe", "grocery", "bar", "restaurant", "bakery", "cinema", "diner", "pharmacy") else (
        2 if current_loc in ("office", "office_tower", "factory", "school", "hospital") else 3))
    f.append(zone / 3.0)
    f.append(1.0 if current_loc == home_loc else 0.0)
    f.append(1.0 if current_loc == work_loc else 0.0)
    f.append(min(num_people / 10.0, 1.0))
    loc_oh = [0.0] * 6
    if zone == 0: loc_oh[0] = 1.0
    elif zone == 1: loc_oh[1] = 1.0
    elif zone == 2: loc_oh[2] = 1.0
    elif current_loc.startswith("street_"): loc_oh[4] = 1.0
    else: loc_oh[3] = 1.0
    if current_loc == home_loc: loc_oh[5] = 1.0
    f.extend(loc_oh)
    tp = [0.0] * 7
    tp[_time_period(hour)] = 1.0
    f.extend(tp)
    f.extend([0.0] * 9)  # last action
    return f


# ════════════════════════════════════════════════════════════════════════
# STEP 1: COLLECT — Watch live sim and record training samples
# ════════════════════════════════════════════════════════════════════════

async def collect(
    base_url: str = "https://raymelius-soci2.hf.space",
    duration_minutes: int = 60,
    poll_interval: float = 3.0,
):
    """Poll the live simulation and collect (state, action) training pairs.

    Each tick, for each agent we observe:
    - Input: agent persona + needs + mood + location + time
    - Label: the action they actually chose (whether from NN, Gemini, or routine)

    This is teacher-free learning — whatever the simulation does IS the label.
    When Gemini makes a decision (10% of the time), it's a high-quality sample.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting from {base_url} for {duration_minutes} min...")
    logger.info(f"Saving to {SAMPLES_FILE}")

    # Fetch agent personas (static data)
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # /api/agents returns a dict keyed by agent ID
        agents_resp = await client.get("/api/agents")
        agents_resp.raise_for_status()
        agents_dict = agents_resp.json()  # {aid: {name, age, location, ...}}

        # Build persona cache — detail endpoint has needs/relationships but
        # not raw personality scores, so we use summary age + defaults
        persona_cache: dict[str, dict] = {}
        for aid, agent_summary in agents_dict.items():
            try:
                detail_resp = await client.get(f"/api/agents/{aid}")
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    pers = detail.get("personality", {})
                    persona_cache[aid] = {
                        "openness": pers.get("openness", 5),
                        "conscientiousness": pers.get("conscientiousness", 5),
                        "extraversion": pers.get("extraversion", 5),
                        "agreeableness": pers.get("agreeableness", 5),
                        "neuroticism": pers.get("neuroticism", 5),
                        "age": detail.get("age", 30),
                        "home": detail.get("home_location", ""),
                        "work": detail.get("work_location", ""),
                    }
            except Exception:
                pass

        logger.info(f"Cached {len(persona_cache)} agent personas")

        # Poll loop
        samples_collected = 0
        last_tick = -1
        start_time = time.monotonic()
        end_time = start_time + duration_minutes * 60

        with open(SAMPLES_FILE, "a") as f:
            while time.monotonic() < end_time:
                try:
                    # Get current city state
                    city_resp = await client.get("/api/city")
                    if city_resp.status_code != 200:
                        await asyncio.sleep(poll_interval)
                        continue
                    city = city_resp.json()

                    clock = city.get("clock", {})
                    tick = clock.get("total_ticks", 0)

                    # Skip if same tick
                    if tick == last_tick:
                        await asyncio.sleep(poll_interval)
                        continue
                    last_tick = tick

                    hour = clock.get("hour", 12)
                    minute = clock.get("minute", 0)
                    day = clock.get("day", 1)

                    # Count agents per location
                    loc_counts: dict[str, int] = {}
                    for aid, adata in city.get("agents", {}).items():
                        loc = adata.get("location", "")
                        loc_counts[loc] = loc_counts.get(loc, 0) + 1

                    # Collect a sample for each agent
                    for aid, adata in city.get("agents", {}).items():
                        action_str = adata.get("action", "idle")
                        state = adata.get("state", "idle")
                        location = adata.get("location", "")
                        mood = adata.get("mood", 0.0)
                        needs = adata.get("needs", {})

                        # Map state to action type
                        state_to_action = {
                            "idle": "wander", "moving": "move", "working": "work",
                            "eating": "eat", "sleeping": "sleep",
                            "socializing": "talk", "in_conversation": "talk",
                            "exercising": "exercise", "shopping": "shop",
                            "relaxing": "relax",
                        }
                        action_type = state_to_action.get(state, "wander")

                        if action_type not in ACTION_TO_IDX:
                            continue

                        persona = persona_cache.get(aid, {
                            "openness": 5, "conscientiousness": 5, "extraversion": 5,
                            "agreeableness": 5, "neuroticism": 5, "age": 30,
                            "home": "", "work": "",
                        })

                        features = encode_features(
                            personality=persona,
                            age=persona.get("age", 30),
                            hour=hour, minute=minute, day=day,
                            needs=needs, mood=mood,
                            current_loc=location,
                            home_loc=persona.get("home", ""),
                            work_loc=persona.get("work", ""),
                            num_people=loc_counts.get(location, 0),
                        )

                        sample = {
                            "features": features,
                            "action_idx": ACTION_TO_IDX[action_type],
                            "target_loc_idx": LOC_TO_IDX.get(location, 0),
                            "duration": ACTION_DURATIONS.get(action_type, 2),
                            "tick": tick,
                            "agent_id": aid,
                            "source": city.get("llm_provider", "unknown"),
                        }

                        f.write(json.dumps(sample) + "\n")
                        samples_collected += 1

                    elapsed = (time.monotonic() - start_time) / 60
                    logger.info(
                        f"Tick {tick} | Day {day} {hour:02d}:{minute:02d} | "
                        f"{samples_collected:,} samples | {elapsed:.1f} min"
                    )

                except httpx.HTTPError as e:
                    logger.warning(f"HTTP error: {e}")
                except Exception as e:
                    logger.error(f"Collection error: {e}", exc_info=True)

                await asyncio.sleep(poll_interval)

    logger.info(f"Collection done: {samples_collected:,} samples saved to {SAMPLES_FILE}")
    return samples_collected


# ════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN — Retrain the NN on collected + synthetic data
# ════════════════════════════════════════════════════════════════════════

def train(epochs: int = 20, batch_size: int = 512, lr: float = 3e-4):
    """Retrain the SociAgentTransformer on collected data.

    Loads collected samples from the live sim, mixes with synthetic data
    for robustness, and fine-tunes the existing model weights.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {DEVICE}")

    # ── Load collected data ──────────────────────────────────────────
    collected = []
    source_counts: dict[str, int] = {}
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    collected.append(sample)
                    src = sample.get("source", "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1
        logger.info(f"Loaded {len(collected):,} collected samples — sources: {source_counts}")
    else:
        logger.warning(f"No collected samples at {SAMPLES_FILE}")

    # Oversample LLM-sourced data (Gemini/Claude/Groq) — these are higher quality
    # than NN or routine-generated samples, so we duplicate them 3x
    llm_sources = {"gemini", "claude", "groq"}
    llm_samples = [s for s in collected if s.get("source", "") in llm_sources]
    if llm_samples:
        logger.info(f"Oversampling {len(llm_samples):,} LLM-sourced samples (3x weight)")
        collected.extend(llm_samples * 2)  # 2 extra copies = 3x total weight

    if len(collected) < 100:
        logger.warning("Too few collected samples — generating synthetic data to supplement")
        collected.extend(_generate_synthetic(50_000 - len(collected)))

    # ── Dataset ──────────────────────────────────────────────────────
    random.shuffle(collected)
    split = int(len(collected) * 0.9)
    train_data = collected[:split]
    val_data = collected[split:]

    class ActionDataset(Dataset):
        def __init__(self, data):
            self.features = torch.tensor([d["features"] for d in data], dtype=torch.float32)
            self.actions = torch.tensor([d["action_idx"] for d in data], dtype=torch.long)
            self.locations = torch.tensor([d["target_loc_idx"] for d in data], dtype=torch.long)
            self.durations = torch.tensor([d["duration"] for d in data], dtype=torch.float32)

        def __len__(self):
            return len(self.actions)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "action": self.actions[idx],
                "location": self.locations[idx],
                "duration": self.durations[idx],
            }

    train_ds = ActionDataset(train_data)
    val_ds = ActionDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    logger.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # ── Model (same architecture as notebook) ────────────────────────
    # Import model class inline to avoid dependency on notebook
    model = _build_model().to(DEVICE)

    # Load existing weights if available
    if BEST_PT.exists():
        model.load_state_dict(torch.load(BEST_PT, map_location=DEVICE, weights_only=True))
        logger.info(f"Loaded existing weights from {BEST_PT}")
    else:
        logger.info("Training from scratch (no existing weights)")

    # ── Training loop ────────────────────────────────────────────────
    # Class weights
    action_counts = torch.zeros(NUM_ACTIONS)
    for d in train_data:
        action_counts[d["action_idx"]] += 1
    action_weights = (1.0 / (action_counts + 1.0))
    action_weights = action_weights / action_weights.sum() * NUM_ACTIONS
    action_weights = action_weights.to(DEVICE)

    action_loss_fn = nn.CrossEntropyLoss(weight=action_weights)
    location_loss_fn = nn.CrossEntropyLoss()
    duration_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_action_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            feat = batch["features"].to(DEVICE)
            out = model(feat)
            loss = (
                1.0 * action_loss_fn(out["action_logits"], batch["action"].to(DEVICE))
                + 0.5 * location_loss_fn(out["location_logits"], batch["location"].to(DEVICE))
                + 0.2 * duration_loss_fn(out["duration"], batch["duration"].to(DEVICE))
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        avg_train_loss = total_loss / n

        # Validate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(DEVICE)
                out = model(feat)
                loss = (
                    1.0 * action_loss_fn(out["action_logits"], batch["action"].to(DEVICE))
                    + 0.5 * location_loss_fn(out["location_logits"], batch["location"].to(DEVICE))
                    + 0.2 * duration_loss_fn(out["duration"], batch["duration"].to(DEVICE))
                )
                val_loss += loss.item()
                pred = out["action_logits"].argmax(dim=-1)
                correct += (pred == batch["action"].to(DEVICE)).sum().item()
                total += feat.shape[0]
        acc = correct / total if total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_action_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(BEST_PT))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Acc: {acc:.1%} | "
                f"Best: {best_acc:.1%}"
            )

    logger.info(f"Training done. Best accuracy: {best_acc:.1%}")

    # ── Export to ONNX ───────────────────────────────────────────────
    model.load_state_dict(torch.load(str(BEST_PT), map_location="cpu", weights_only=True))
    model.cpu().eval()

    dummy = torch.randn(1, FEATURE_DIM)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["features"],
        output_names=["action_logits", "location_logits", "duration"],
        dynamic_axes={"features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_size = ONNX_PATH.stat().st_size / 1024
    logger.info(f"ONNX exported: {ONNX_PATH} ({onnx_size:.0f} KB)")

    # ── Save training stats ───────────────────────────────────────────
    stats = {
        "best_val_action_acc": best_acc,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "collected_samples": sum(source_counts.values()),
        "source_counts": source_counts,
        "model_size_kb": onnx_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "history": history,
    }
    stats_path = MODEL_DIR / "training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats saved to {stats_path}")

    # ── Plot training graphs ──────────────────────────────────────────
    plot_training_graphs(stats_path)

    return best_acc


# ════════════════════════════════════════════════════════════════════════
# STEP 3: PUSH — Upload improved model to HuggingFace Hub
# ════════════════════════════════════════════════════════════════════════

def push(repo_id: str = "RayMelius/soci-agent-nn", accuracy: float = None,
         base_url: str = "https://raymelius-soci2.hf.space"):
    """Push the retrained ONNX model to HuggingFace Hub, then trigger live reload."""
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        logger.error("HF_TOKEN not set. Export it: export HF_TOKEN=hf_...")
        sys.exit(1)

    if not ONNX_PATH.exists():
        logger.error(f"No ONNX model at {ONNX_PATH}. Run 'train' first.")
        sys.exit(1)

    login(token=token)
    api = HfApi()

    # Compare against previous accuracy if available
    try:
        from huggingface_hub import hf_hub_download
        prev_stats_path = hf_hub_download(repo_id=repo_id, filename="training_stats.json", token=token)
        prev_stats = json.loads(open(prev_stats_path).read())
        prev_acc = prev_stats.get("best_accuracy")
        if prev_acc is not None and accuracy is not None:
            delta = accuracy - prev_acc
            symbol = "+" if delta >= 0 else ""
            logger.info(f"Previous accuracy: {prev_acc:.1%} → New: {accuracy:.1%} ({symbol}{delta:.1%})")
        elif prev_acc is not None:
            logger.info(f"Previous accuracy: {prev_acc:.1%} (no new accuracy to compare)")
    except Exception:
        logger.info("No previous training_stats.json found — first push")
    api.create_repo(repo_id, exist_ok=True)

    # Upload ONNX
    api.upload_file(
        path_or_fileobj=str(ONNX_PATH),
        path_in_repo="soci_agent.onnx",
        repo_id=repo_id,
        commit_message="Self-improve: retrained on live sim data",
    )
    logger.info(f"ONNX model pushed to https://huggingface.co/{repo_id}")

    # Upload PyTorch weights too
    if BEST_PT.exists():
        api.upload_file(
            path_or_fileobj=str(BEST_PT),
            path_in_repo="soci_agent_best.pt",
            repo_id=repo_id,
            commit_message="Self-improve: retrained weights",
        )
        logger.info("PyTorch weights pushed")

    # Upload training stats
    stats = {
        "samples_file": str(SAMPLES_FILE),
        "num_samples": sum(1 for _ in open(SAMPLES_FILE)) if SAMPLES_FILE.exists() else 0,
        "model_size_kb": ONNX_PATH.stat().st_size / 1024,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if accuracy is not None:
        stats["best_accuracy"] = round(accuracy, 4)
    stats_path = MODEL_DIR / "training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    api.upload_file(
        path_or_fileobj=str(stats_path),
        path_in_repo="training_stats.json",
        repo_id=repo_id,
    )

    logger.info("Push complete!")

    # Trigger hot-reload on the live simulation if reachable
    try:
        resp = httpx.post(f"{base_url}/api/nn/reload", timeout=30.0)
        if resp.status_code == 200:
            logger.info(f"Live sim NN reloaded: {resp.json().get('message', 'ok')}")
        else:
            logger.warning(f"Could not reload live sim NN: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not reach live sim for reload: {e}")


# ════════════════════════════════════════════════════════════════════════
# Training Graphs
# ════════════════════════════════════════════════════════════════════════

def plot_training_graphs(stats_path: Path | str | None = None):
    """Plot training loss and accuracy curves from saved training stats.

    Saves the plot to models/training_graphs.png and displays it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats_path = Path(stats_path) if stats_path else MODEL_DIR / "training_stats.json"
    if not stats_path.exists():
        logger.error(f"No training stats found at {stats_path}")
        return

    stats = json.loads(stats_path.read_text())
    history = stats.get("history", {})

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_action_acc = history.get("val_action_acc", [])

    if not train_loss:
        logger.error("No training history found in stats file")
        return

    epochs_range = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Soci Self-Improve Training — {stats.get('timestamp', '?')}  |  "
        f"Best Acc: {stats.get('best_val_action_acc', stats.get('best_accuracy', 0)):.1%}",
        fontsize=13, fontweight="bold",
    )

    # Loss curves
    ax = axes[0]
    ax.plot(epochs_range, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
    if val_loss:
        ax.plot(epochs_range, val_loss, label="Val Loss", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    # Action accuracy
    ax = axes[1]
    if val_action_acc:
        ax.plot(epochs_range, [a * 100 for a in val_action_acc], label="Action Accuracy",
                color="#4CAF50", linewidth=2)
        best_epoch = int(np.argmax(val_action_acc)) + 1
        best_acc = max(val_action_acc) * 100
        ax.axhline(y=best_acc, color="#4CAF50", linestyle="--", alpha=0.4)
        ax.annotate(f"Best: {best_acc:.1f}% (epoch {best_epoch})",
                    xy=(best_epoch, best_acc), fontsize=9,
                    xytext=(best_epoch + 1, best_acc - 3),
                    arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                    color="#4CAF50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    # Footer
    footer = (
        f"Train: {stats.get('train_samples', '?'):,} samples  |  "
        f"Val: {stats.get('val_samples', '?'):,} samples  |  "
        f"Collected: {stats.get('collected_samples', 0):,}  |  "
        f"Model: {stats.get('model_size_kb', 0):.0f} KB"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    graph_path = MODEL_DIR / "training_graphs.png"
    fig.savefig(str(graph_path), dpi=150, bbox_inches="tight")
    logger.info(f"Training graphs saved to {graph_path}")

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.use("TkAgg")
            plt.show(block=False)
            plt.pause(0.5)
    except Exception:
        pass

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# Model architecture (inline to avoid import dependency)
# ════════════════════════════════════════════════════════════════════════

def _build_model():
    """Build SociAgentTransformer — same architecture as the training notebook."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FeatureTokenizer(nn.Module):
        GROUPS = [
            ("personality", 0, 6), ("time", 6, 12), ("needs", 12, 21),
            ("location", 21, 31), ("time_period", 31, 38), ("last_action", 38, 47),
        ]

        def __init__(self, d_model):
            super().__init__()
            self.projections = nn.ModuleList()
            for name, start, end in self.GROUPS:
                self.projections.append(nn.Sequential(
                    nn.Linear(end - start, d_model), nn.LayerNorm(d_model), nn.GELU(),
                ))
            self.pos_embed = nn.Parameter(torch.randn(1, len(self.GROUPS), d_model) * 0.02)

        def forward(self, features):
            tokens = []
            for i, (_, start, end) in enumerate(self.GROUPS):
                tokens.append(self.projections[i](features[:, start:end]))
            tokens = torch.stack(tokens, dim=1)
            return tokens + self.pos_embed

    class MoEFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.gate = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
                for _ in range(num_experts)
            ])

        def forward(self, x):
            B, S, D = x.shape
            gate_probs = F.softmax(self.gate(x), dim=-1)
            top_k_probs, top_k_idx = gate_probs.topk(self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                eidx = top_k_idx[:, :, k]
                w = top_k_probs[:, :, k].unsqueeze(-1)
                for e in range(self.num_experts):
                    mask = (eidx == e).unsqueeze(-1)
                    if mask.any():
                        output = output + mask.float() * w * self.experts[e](x)
            return output

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, nhead, d_ff, num_experts=4, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.moe_ff = MoEFeedForward(d_model, d_ff, num_experts)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.moe_ff(x)
            return self.norm2(x + self.dropout(ff_out))

    class SociAgentTransformer(nn.Module):
        def __init__(self, d_model=128, nhead=8, num_layers=4, d_ff=256, num_experts=4, dropout=0.1):
            super().__init__()
            self.tokenizer = FeatureTokenizer(d_model)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, nhead, d_ff, num_experts, dropout)
                for _ in range(num_layers)
            ])
            self.cls_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cls_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cls_norm = nn.LayerNorm(d_model)
            self.action_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, NUM_ACTIONS),
            )
            self.location_head = nn.Sequential(
                nn.Linear(d_model + NUM_ACTIONS, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, NUM_LOCATIONS),
            )
            self.duration_head = nn.Sequential(
                nn.Linear(d_model + NUM_ACTIONS, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, features):
            tokens = self.tokenizer(features)
            for layer in self.layers:
                tokens = layer(tokens)
            B = features.shape[0]
            cls = self.cls_query.expand(B, -1, -1)
            cls_out, _ = self.cls_attn(cls, tokens, tokens)
            h = self.cls_norm(cls_out.squeeze(1))
            action_logits = self.action_head(h)
            action_probs = F.softmax(action_logits.detach(), dim=-1)
            h_a = torch.cat([h, action_probs], dim=-1)
            location_logits = self.location_head(h_a)
            duration = torch.sigmoid(self.duration_head(h_a)) * 7.0 + 1.0
            return {
                "action_logits": action_logits,
                "location_logits": location_logits,
                "duration": duration.squeeze(-1),
            }

    return SociAgentTransformer()


# ════════════════════════════════════════════════════════════════════════
# Synthetic data fallback (when not enough collected samples)
# ════════════════════════════════════════════════════════════════════════

# Inline personas for synthetic generation — must match personas.yaml
_PERSONAS = [
    # House 1 — Elena & Lila (roommates)
    {"id": "elena",  "O": 8, "C": 7, "E": 4, "A": 6, "N": 5, "age": 34, "home": "house_elena", "work": "office",
     "tags": ["freelance", "introvert", "tech"], "hangouts": ["cafe", "library"]},
    {"id": "lila",   "O":10, "C": 3, "E": 6, "A": 7, "N": 7, "age": 33, "home": "house_elena", "work": "library",
     "tags": ["creative", "emotional", "crush_elena"], "hangouts": ["park", "cafe", "library"]},
    # House 2 — Marcus & Zoe (siblings)
    {"id": "marcus", "O": 5, "C": 8, "E": 9, "A": 7, "N": 3, "age": 28, "home": "house_marcus", "work": "gym",
     "tags": ["athletic", "extrovert", "community"], "hangouts": ["park", "sports_field", "cafe"]},
    {"id": "zoe",    "O": 8, "C": 4, "E": 8, "A": 6, "N": 7, "age": 19, "home": "house_marcus", "work": "library",
     "tags": ["student", "social_media", "young"], "hangouts": ["cafe", "cinema", "park", "town_square"]},
    # House 3 — Helen & Alice (close friends)
    {"id": "helen",  "O": 6, "C": 8, "E": 6, "A": 8, "N": 4, "age": 67, "home": "house_helen", "work": "library",
     "tags": ["retired", "bookworm", "widow"], "hangouts": ["library", "park", "bakery", "church"]},
    {"id": "alice",  "O": 5, "C": 8, "E": 6, "A": 8, "N": 3, "age": 58, "home": "house_helen", "work": "bakery",
     "tags": ["retired", "baker", "nurturing"], "hangouts": ["bakery", "grocery", "church"]},
    # House 4 — Diana & Marco (mother & son)
    {"id": "diana",  "O": 4, "C": 9, "E": 5, "A": 6, "N": 7, "age": 41, "home": "house_diana", "work": "grocery",
     "tags": ["business_owner", "single_mother", "protective"], "hangouts": ["grocery"]},
    {"id": "marco",  "O": 7, "C": 4, "E": 6, "A": 5, "N": 6, "age": 16, "home": "house_diana", "work": "school",
     "tags": ["student", "teen", "gamer"], "hangouts": ["park", "cinema", "cafe", "sports_field"]},
    # House 5 — Kai (lives alone)
    {"id": "kai",    "O": 9, "C": 3, "E": 7, "A": 5, "N": 6, "age": 22, "home": "house_kai", "work": "cafe",
     "tags": ["musician", "creative", "dropout"], "hangouts": ["bar", "park", "town_square"]},
    # House 6 — Priya & Nina (flatmates)
    {"id": "priya",  "O": 7, "C": 9, "E": 5, "A": 8, "N": 6, "age": 38, "home": "house_priya", "work": "hospital",
     "tags": ["overworked", "caring", "guilt"], "hangouts": ["hospital", "pharmacy"]},
    {"id": "nina",   "O": 5, "C": 8, "E": 9, "A": 4, "N": 5, "age": 29, "home": "house_priya", "work": "office",
     "tags": ["ambitious", "networker", "suspicious"], "hangouts": ["cafe", "restaurant", "office_tower"]},
    # House 7 — James & Theo (housemates)
    {"id": "james",  "O": 5, "C": 6, "E": 8, "A": 7, "N": 4, "age": 55, "home": "house_james", "work": "bar",
     "tags": ["social_hub", "divorced", "storyteller"], "hangouts": ["bar"]},
    {"id": "theo",   "O": 3, "C": 7, "E": 4, "A": 5, "N": 5, "age": 45, "home": "house_james", "work": "factory",
     "tags": ["blue_collar", "stoic", "handy"], "hangouts": ["bar", "diner"]},
    # House 8 — Rosa & Omar
    {"id": "rosa",   "O": 6, "C": 9, "E": 7, "A": 8, "N": 5, "age": 62, "home": "house_rosa", "work": "restaurant",
     "tags": ["nurturing", "italian", "community_mother"], "hangouts": ["restaurant", "grocery"]},
    {"id": "omar",   "O": 6, "C": 6, "E": 7, "A": 7, "N": 4, "age": 50, "home": "house_rosa", "work": "restaurant",
     "tags": ["immigrant", "philosophical", "hardworking"], "hangouts": ["restaurant", "cafe", "park"]},
    # House 9 — Yuki & Devon (flatmates)
    {"id": "yuki",   "O": 8, "C": 6, "E": 5, "A": 9, "N": 3, "age": 26, "home": "house_yuki", "work": "gym",
     "tags": ["mindful", "calm", "empathetic"], "hangouts": ["park", "gym", "library"]},
    {"id": "devon",  "O": 9, "C": 5, "E": 6, "A": 4, "N": 6, "age": 30, "home": "house_yuki", "work": "office",
     "tags": ["investigative", "paranoid", "curious"], "hangouts": ["cafe", "bar", "library", "town_square"]},
    # House 10 — Frank, George & Sam
    {"id": "frank",  "O": 3, "C": 7, "E": 5, "A": 4, "N": 5, "age": 72, "home": "house_frank", "work": "bar",
     "tags": ["retired", "cantankerous", "creature_of_habit"], "hangouts": ["bar", "diner"]},
    {"id": "george", "O": 4, "C": 7, "E": 3, "A": 6, "N": 4, "age": 47, "home": "house_frank", "work": "factory",
     "tags": ["night_shift", "widower", "observant"], "hangouts": ["park"]},
    {"id": "sam",    "O": 7, "C": 8, "E": 3, "A": 7, "N": 4, "age": 40, "home": "house_frank", "work": "library",
     "tags": ["quiet", "bookish", "inclusive"], "hangouts": ["library", "park", "cafe"]},
]


def _persona_hangout(p: dict, fallbacks: list[str]) -> str:
    """Pick a location the persona naturally gravitates toward."""
    hangouts = p.get("hangouts", [])
    if hangouts and random.random() < 0.6:
        return random.choice(hangouts)
    return random.choice(fallbacks)


def _generate_synthetic(n: int) -> list[dict]:
    """Generate persona-aware synthetic training samples."""
    data = []
    for _ in range(n):
        p = random.choice(_PERSONAS)
        persona = {
            "openness": p["O"], "conscientiousness": p["C"], "extraversion": p["E"],
            "agreeableness": p["A"], "neuroticism": p["N"],
        }
        tags = p.get("tags", [])
        is_night_shift = "night_shift" in tags
        is_retired = "retired" in tags
        is_student = "student" in tags

        hour = random.randint(0, 23)
        minute = random.choice([0, 15, 30, 45])
        day = random.randint(1, 30)
        is_weekend = ((day - 1) % 7) >= 5
        period = _time_period(hour)

        # Persona-aware needs generation
        needs = {}
        for nm in NEED_NAMES:
            if random.random() < 0.15:
                needs[nm] = round(random.uniform(0.0, 0.2), 2)
            else:
                needs[nm] = round(random.uniform(0.2, 1.0), 2)

        if "overworked" in tags:
            needs["energy"] = round(min(needs["energy"], random.uniform(0.1, 0.5)), 2)
            needs["social"] = round(min(needs["social"], random.uniform(0.1, 0.5)), 2)
        if "athletic" in tags:
            needs["energy"] = round(max(needs["energy"], random.uniform(0.5, 0.9)), 2)
        if "emotional" in tags:
            swing = random.choice(NEED_NAMES)
            needs[swing] = round(random.uniform(0.0, 0.3), 2)
        if "creature_of_habit" in tags:
            for nm in NEED_NAMES:
                needs[nm] = round(needs[nm] * 0.7 + 0.2, 2)
        if is_night_shift and 6 <= hour <= 18:
            needs["energy"] = round(min(needs["energy"], random.uniform(0.05, 0.35)), 2)
        if "mindful" in tags:
            for nm in NEED_NAMES:
                needs[nm] = round(max(needs[nm], 0.2), 2)
        if is_student:
            needs["social"] = round(max(needs["social"], random.uniform(0.3, 0.7)), 2)

        # Persona-aware mood
        avg_need = sum(needs.values()) / len(needs)
        mood = round(max(-1.0, min(1.0,
            (avg_need - 0.5) * 2 + random.uniform(-0.5, 0.5) * (p["N"] / 10.0)
        )), 2)

        # Persona-aware starting location
        if is_night_shift:
            if period in (0, 6):
                loc = p["work"]
            elif period in (2, 3):
                loc = p["home"]
            else:
                loc = random.choice([p["home"], "park"] if random.random() < 0.7 else [p["home"]])
        elif period == 0:
            loc = p["home"]
        elif period in (2, 4) and not is_weekend:
            if is_retired:
                loc = random.choice([p["home"]] + p.get("hangouts", ["park"]))
            else:
                loc = random.choice([p["work"], p["work"], _persona_hangout(p, ["cafe"])])
        elif period == 5:
            loc = random.choice([p["home"], _persona_hangout(p, ["bar", "cafe"])])
        else:
            loc = random.choice([p["home"], p["work"]])

        # --- Determine action ---
        urgent = [(nm, needs[nm]) for nm in NEED_NAMES if needs[nm] < 0.15]
        urgent.sort(key=lambda x: x[1])
        action = None
        target = loc

        # Priority 1: Critical needs
        if urgent:
            need_name = urgent[0][0]
            if need_name == "hunger":
                eat_locs = ["cafe", "restaurant", "bakery", "diner", p["home"]]
                if "community_mother" in tags:
                    eat_locs = ["restaurant", p["home"]]
                elif "baker" in tags:
                    eat_locs = ["bakery", p["home"]]
                action, target = "eat", random.choice(eat_locs)
            elif need_name == "energy":
                action, target = "sleep", p["home"]
            elif need_name == "social":
                social_locs = ["cafe", "bar", "park", "town_square"]
                if "social_hub" in tags:
                    social_locs = ["bar", "bar", "restaurant"]
                elif "networker" in tags:
                    social_locs = ["cafe", "restaurant", "office"]
                action, target = "talk", random.choice(social_locs)
            elif need_name == "purpose":
                action, target = "work", p["work"]
            elif need_name == "comfort":
                action, target = "relax", random.choice([p["home"], "park", "library"])
            elif need_name == "fun":
                fun_locs = ["park", "cinema", "bar", "sports_field"]
                if is_student:
                    fun_locs = ["cinema", "park", "cafe", "town_square"]
                action, target = random.choice(["relax", "exercise", "wander"]), random.choice(fun_locs)

        # Priority 2: Night shift inverted schedule (George)
        if action is None and is_night_shift:
            if period in (0, 6):
                action, target = "work", p["work"]
            elif period == 1:
                action, target = "move", p["home"]
            elif period in (2, 3):
                if needs["energy"] < 0.6:
                    action, target = "sleep", p["home"]
                else:
                    action, target = "relax", random.choice([p["home"], "park"])
            elif period in (4, 5):
                if needs["hunger"] < 0.5:
                    action, target = "eat", random.choice(["diner", "restaurant", p["home"]])
                else:
                    action, target = "move", p["work"]

        # Priority 3: Persona-specific patterns
        if action is None:
            pid = p.get("id", "")
            if pid == "frank" and period in (5, 6) and random.random() < 0.7:
                action, target = "relax", "bar"
            elif pid == "lila" and random.random() < 0.15:
                action = random.choice(["wander", "talk", "relax"])
                target = random.choice(["house_elena", "cafe", "library"])
            elif pid == "rosa" and period in (1, 2) and random.random() < 0.4:
                action, target = "shop", "grocery"
            elif pid == "omar" and period in (2, 3, 4) and not is_weekend and random.random() < 0.5:
                action, target = "wander", random.choice(["street_north", "street_south", "street_east", "street_west"])
            elif pid == "diana" and not is_weekend and period in (2, 3, 4) and random.random() < 0.7:
                action, target = "work", "grocery"
            elif pid == "marcus" and period == 1 and random.random() < 0.6:
                action, target = "exercise", random.choice(["gym", "park", "sports_field"])
            elif pid == "yuki" and period == 1 and random.random() < 0.5:
                action, target = "exercise", random.choice(["park", "gym"])
            elif pid == "devon" and period in (2, 4) and random.random() < 0.3:
                action = random.choice(["wander", "talk"])
                target = random.choice(["cafe", "bar", "town_square", "library"])

        # Priority 4: General time-of-day patterns
        if action is None:
            if period == 0:
                action, target = "sleep", p["home"]
            elif period == 1:
                if needs["hunger"] < 0.5:
                    action, target = "eat", random.choice(["cafe", "bakery", p["home"]])
                elif p["E"] >= 6 and random.random() < 0.3:
                    action, target = "exercise", random.choice(["gym", "park", "sports_field"])
                else:
                    action, target = "move", p["work"]
            elif period in (2, 4):
                if is_weekend:
                    r = random.random()
                    if is_retired:
                        if r < 0.35:
                            action, target = "relax", _persona_hangout(p, ["park", "library", p["home"]])
                        elif r < 0.55:
                            action, target = "talk", _persona_hangout(p, ["cafe", "park", "church"])
                        elif r < 0.7:
                            action, target = "shop", random.choice(["grocery", "pharmacy", "bakery"])
                        else:
                            action, target = "wander", random.choice(["park", "town_square"])
                    elif is_student:
                        if r < 0.3:
                            action, target = "talk", random.choice(["cafe", "park", "cinema", "town_square"])
                        elif r < 0.5:
                            action, target = "relax", random.choice(["cinema", "park", p["home"]])
                        elif r < 0.7:
                            action, target = "exercise", random.choice(["gym", "park", "sports_field"])
                        else:
                            action, target = "wander", random.choice(["town_square", "street_north"])
                    else:
                        if r < 0.25:
                            action, target = "relax", _persona_hangout(p, ["park", "cafe", p["home"]])
                        elif r < 0.45 and p["E"] >= 6:
                            action, target = "talk", _persona_hangout(p, ["cafe", "park", "town_square"])
                        elif r < 0.6:
                            action, target = "shop", random.choice(["grocery", "pharmacy"])
                        elif r < 0.8:
                            action, target = "exercise", random.choice(["gym", "park"])
                        else:
                            action, target = "wander", random.choice(["park", "town_square"])
                else:
                    work_prob = 0.5 + p["C"] * 0.05
                    if "business_owner" in tags or "overworked" in tags:
                        work_prob += 0.15
                    if is_retired:
                        work_prob = 0.15
                    if random.random() < work_prob:
                        action, target = "work", p["work"]
                    else:
                        action = random.choice(["wander", "relax", "talk"])
                        target = _persona_hangout(p, ["cafe", "park", "town_square"])
            elif period == 3:
                if needs["hunger"] < 0.6:
                    action, target = "eat", random.choice(["cafe", "restaurant", "bakery", "diner"])
                else:
                    action, target = "relax", random.choice(["park", "cafe"])
            elif period == 5:
                social_bias = p["E"] / 10.0
                r = random.random()
                if r < social_bias * 0.5:
                    action, target = "talk", random.choice(["bar", "restaurant", "park", "cafe"])
                elif r < 0.4:
                    action, target = "eat", random.choice(["restaurant", "bar", "diner", p["home"]])
                elif r < 0.55:
                    action, target = "exercise", random.choice(["gym", "park"])
                elif r < 0.7:
                    action, target = "relax", _persona_hangout(p, ["cinema", "bar", p["home"]])
                else:
                    action, target = "relax", p["home"]
            elif period == 6:
                if needs["energy"] < 0.4:
                    action, target = "sleep", p["home"]
                else:
                    action, target = "relax", p["home"]

        # Move override
        if target != loc and action != "move" and random.random() < 0.3:
            action = "move"

        # Duration adjustments
        dur = ACTION_DURATIONS.get(action, 2)
        if is_retired and dur > 3 and action not in ("sleep", "work"):
            dur = min(dur, 3)

        features = encode_features(
            personality=persona, age=p["age"],
            hour=hour, minute=minute, day=day,
            needs=needs, mood=mood, current_loc=loc,
            home_loc=p["home"], work_loc=p["work"],
        )

        data.append({
            "features": features,
            "action_idx": ACTION_TO_IDX.get(action, 0),
            "target_loc_idx": LOC_TO_IDX.get(target, 0),
            "duration": min(max(dur, 1), 8),
        })

    return data


# ════════════════════════════════════════════════════════════════════════
# STEP 4: SCHEDULED — Nightly Gemini collection + retrain cycle
# ════════════════════════════════════════════════════════════════════════

async def scheduled(
    base_url: str = "https://raymelius-soci2.hf.space",
    collect_minutes: int = 120,
    epochs: int = 25,
    repo_id: str = "RayMelius/soci-agent-nn",
    gemini_prob: float = 0.50,
):
    """Daily training cycle: switch to Gemini at quota reset, collect, retrain, push.

    Flow:
      1. Wait until Gemini quota resets (10:00 AM Athens / Europe/Athens)
      2. Switch live sim to Gemini provider, raise probability
      3. Collect high-quality (state, action) samples from Gemini decisions
      4. Switch back to NN when done (or when quota exhausted)
      5. Train on collected Gemini samples (weighted 3x vs NN/routine samples)
      6. Push improved model to HF Hub
      7. Repeat next night

    Usage:
        python nn_selfimprove.py scheduled --collect-minutes 120 --gemini-prob 0.50
    """
    import datetime

    async def _api_call(client: httpx.AsyncClient, method: str, path: str, **kwargs):
        """Make API call with retries."""
        for attempt in range(3):
            try:
                resp = await getattr(client, method)(path, timeout=30.0, **kwargs)
                return resp
            except httpx.HTTPError as e:
                logger.warning(f"API {method.upper()} {path} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(5)
        return None

    async def switch_provider(client: httpx.AsyncClient, provider: str, prob: float):
        """Switch the live sim's LLM provider and probability."""
        resp = await _api_call(client, "post", "/api/llm/provider",
                               json={"provider": provider})
        if resp and resp.status_code == 200:
            logger.info(f"Switched provider to: {provider}")
        else:
            logger.error(f"Failed to switch to {provider}: {resp.status_code if resp else 'no response'}")
            return False

        resp = await _api_call(client, "post", f"/api/controls/llm_probability?value={prob}")
        if resp and resp.status_code == 200:
            logger.info(f"Set probability to: {prob:.0%}")
        else:
            logger.warning(f"Failed to set probability: {resp.status_code if resp else 'no response'}")

        return True

    async def calculate_probability(client: httpx.AsyncClient, target_minutes: int) -> float:
        """Query remaining Gemini quota and return a reasonable probability.

        The real bottleneck is RPM (requests per minute), not probability.
        With 50 agents, even low probability saturates the RPM rate limiter.
        Gemini: 4 RPM → max 240 calls/hour → 1500 RPD lasts ~6.25h.
        Probability mainly controls LLM-vs-routine quality, not quota duration.
        """
        resp = await _api_call(client, "get", "/api/llm/quota")
        if not resp or resp.status_code != 200:
            logger.warning("Could not fetch quota — using default probability")
            return gemini_prob

        quota = resp.json()
        remaining = quota.get("remaining", 1500)

        if remaining <= 0:
            logger.warning("No Gemini quota remaining!")
            return 0.0

        # Get per-provider RPM info
        providers = quota.get("providers", {})
        gemini_info = providers.get("gemini", {})
        rpm = gemini_info.get("rpm", 4)
        max_calls_per_hour = rpm * 60
        hours_available = remaining / max_calls_per_hour
        target_hours = target_minutes / 60.0

        logger.info(
            f"Quota: {remaining} remaining, RPM={rpm} → "
            f"max {max_calls_per_hour} calls/h → ~{hours_available:.1f}h available"
        )

        if hours_available >= target_hours:
            prob = gemini_prob
            logger.info(f"Quota sufficient for {target_minutes}min target → using {prob:.0%}")
        else:
            # Quota won't last — reduce probability (marginal help with many agents)
            prob = max(0.02, 0.10 * (hours_available / target_hours))
            logger.warning(
                f"Quota only lasts ~{hours_available:.1f}h but target is {target_hours:.1f}h "
                f"→ reducing probability to {prob:.1%}"
            )

        return round(prob, 4)

    async def wait_until_reset():
        """Wait until next Gemini quota reset (10:00 AM Athens / Europe/Athens)."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        athens = ZoneInfo("Europe/Athens")
        now = datetime.datetime.now(athens)
        reset_today = now.replace(hour=10, minute=0, second=5, microsecond=0)

        # If we've already passed 10:00 AM today, target tomorrow
        if now >= reset_today:
            reset_target = reset_today + datetime.timedelta(days=1)
        else:
            reset_target = reset_today

        wait_secs = (reset_target - now).total_seconds()
        logger.info(f"Waiting {wait_secs/3600:.1f}h until Gemini reset ({reset_target.strftime('%Y-%m-%d %H:%M %Z')})")
        await asyncio.sleep(wait_secs)

    # ── Main loop ─────────────────────────────────────────────────────
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"{'='*60}")
        logger.info(f"TRAINING CYCLE {cycle}")
        logger.info(f"{'='*60}")

        # 1. Wait for Gemini quota reset (10:00 AM Athens)
        await wait_until_reset()

        async with httpx.AsyncClient(base_url=base_url) as client:
            # 2. Switch to Gemini first
            logger.info("Switching live sim to Gemini...")
            ok = await switch_provider(client, "gemini", 0.01)  # start low
            if not ok:
                logger.error("Could not switch to Gemini — skipping this cycle")
                continue

            # 3. Calculate probability to spread quota over collection period
            calc_prob = await calculate_probability(client, collect_minutes)
            await switch_provider(client, "gemini", calc_prob)
            logger.info(f"Collecting for {collect_minutes} min with Gemini at {calc_prob:.1%} probability...")

        # collect() creates its own client
        n_samples = await collect(
            base_url=base_url,
            duration_minutes=collect_minutes,
            poll_interval=3.0,
        )
        logger.info(f"Collected {n_samples:,} samples this cycle")

        # 4. Switch back to NN + restore default probability
        async with httpx.AsyncClient(base_url=base_url) as client:
            await switch_provider(client, "nn", 1.0)

        # 5. Count Gemini-sourced samples
        gemini_samples = 0
        if SAMPLES_FILE.exists():
            with open(SAMPLES_FILE) as f:
                for line in f:
                    if '"source": "gemini"' in line or '"source":"gemini"' in line:
                        gemini_samples += 1
        logger.info(f"Total Gemini-sourced samples in file: {gemini_samples:,}")

        if gemini_samples < 50:
            logger.warning("Too few Gemini samples — skipping training this cycle")
            continue

        # 6. Train (Gemini samples get 3x weight in the training loop)
        logger.info("Starting retraining...")
        best_acc = train(epochs=epochs)
        logger.info(f"Training done — best accuracy: {best_acc:.1%}")

        # 7. Push improved model
        if os.environ.get("HF_TOKEN"):
            logger.info("Pushing improved model to HF Hub...")
            push(repo_id=repo_id, accuracy=best_acc, base_url=base_url)
        else:
            logger.warning("HF_TOKEN not set — skipping push")

        logger.info(f"Cycle {cycle} complete! Next cycle at 10:00 AM Athens.")


# ════════════════════════════════════════════════════════════════════════
# STEP 5: BUDGET — Check quota and auto-set probability
# ════════════════════════════════════════════════════════════════════════

async def budget(
    base_url: str = "https://raymelius-soci2.hf.space",
    target_minutes: int = 60,
    apply: bool = True,
):
    """Check Gemini quota, calculate and optionally apply the right probability.

    Usage:
        python nn_selfimprove.py budget --minutes 60   # spread quota over 1 hour
        python nn_selfimprove.py budget --minutes 120  # spread over 2 hours
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        resp = await client.get("/api/llm/quota")
        if resp.status_code != 200:
            logger.error(f"Could not fetch quota: {resp.status_code}")
            return

        quota = resp.json()
        provider = quota.get("provider", "?")
        num_agents = quota.get("num_agents", 0)

        # Get Gemini-specific quota from providers dict
        providers = quota.get("providers", {})
        gemini_info = providers.get("gemini", {})
        remaining = gemini_info.get("remaining", quota.get("remaining", 0))
        daily_limit = gemini_info.get("daily_limit", quota.get("daily_limit", 1500))
        daily_requests = gemini_info.get("daily_requests", quota.get("daily_requests", 0))
        rpm = gemini_info.get("rpm", 4)
        max_calls_per_hour = rpm * 60
        hours_available = remaining / max_calls_per_hour if max_calls_per_hour > 0 else 0

        logger.info(f"Provider: {provider}")
        logger.info(f"Daily quota: {daily_requests}/{daily_limit} used, {remaining} remaining")
        logger.info(f"Rate limit: {rpm} RPM → max {max_calls_per_hour} calls/hour")
        logger.info(f"Estimated runtime at max RPM: ~{hours_available:.1f}h")
        logger.info(f"Sim: {num_agents} agents")

        if remaining <= 0:
            logger.warning("No quota remaining! Wait for reset (10:00 AM Athens).")
            return

        target_hours = target_minutes / 60.0
        # Probability controls LLM-vs-routine quality, RPM is the real bottleneck
        if hours_available >= target_hours:
            prob = 0.20  # moderate: good mix of LLM and routine
        else:
            prob = max(0.02, 0.10 * (hours_available / target_hours))

        logger.info(
            f"Target: {target_minutes} min → probability {prob:.2%} "
            f"(RPM-limited to ~{max_calls_per_hour} calls/h, {remaining} remaining)"
        )

        if apply:
            # Switch to Gemini if not already
            if provider != "gemini":
                resp = await client.post("/api/llm/provider", json={"provider": "gemini"})
                if resp.status_code == 200:
                    logger.info("Switched to Gemini")
                else:
                    logger.warning(f"Could not switch to Gemini: {resp.status_code}")

            resp = await client.post(f"/api/controls/llm_probability?value={prob}")
            if resp.status_code == 200:
                logger.info(f"Applied probability: {prob:.2%}")
            else:
                logger.warning(f"Could not set probability: {resp.status_code}")

            logger.info(f"Done! Gemini will run at {prob:.2%} for ~{target_minutes} min. "
                        f"Start collecting: python nn_selfimprove.py collect --minutes {target_minutes}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Soci Agent NN — Self-Improvement Pipeline")
    parser.add_argument("mode", choices=["collect", "train", "push", "all", "scheduled", "budget", "graph"],
                        help="collect=watch live sim, train=retrain NN, push=upload to HF, "
                             "all=full pipeline, scheduled=daily Gemini cycle, "
                             "budget=check quota & set probability, "
                             "graph=display training graphs from last run")
    parser.add_argument("--url", default="https://raymelius-soci2.hf.space",
                        help="Live simulation URL (default: HF Space)")
    parser.add_argument("--minutes", type=int, default=60,
                        help="Collection duration in minutes (default: 60)")
    parser.add_argument("--collect-minutes", type=int, default=120,
                        help="Scheduled mode: collection duration in minutes (default: 120)")
    parser.add_argument("--gemini-prob", type=float, default=0.50,
                        help="Scheduled mode: LLM probability during Gemini collection (default: 0.50)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--repo", default="RayMelius/soci-agent-nn",
                        help="HF Hub repo ID")
    args = parser.parse_args()

    if args.mode == "graph":
        plot_training_graphs()
        return

    if args.mode in ("collect", "all"):
        asyncio.run(collect(base_url=args.url, duration_minutes=args.minutes))

    if args.mode in ("train", "all"):
        best_acc = train(epochs=args.epochs)

    if args.mode in ("push", "all"):
        acc = best_acc if args.mode == "all" else None
        push(repo_id=args.repo, accuracy=acc, base_url=args.url)

    if args.mode == "scheduled":
        asyncio.run(scheduled(
            base_url=args.url,
            collect_minutes=args.collect_minutes,
            epochs=args.epochs,
            repo_id=args.repo,
            gemini_prob=args.gemini_prob,
        ))

    if args.mode == "budget":
        asyncio.run(budget(base_url=args.url, target_minutes=args.minutes, apply=True))


if __name__ == "__main__":
    main()
