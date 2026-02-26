"""
finetune_local.py — Local adaptation of Soci_FineTune_3_Incremental
Fine-tunes Qwen2.5-0.5B-Instruct on Soci city-simulation tasks using Unsloth.

Differences from the Colab version:
  - No Google Drive / google.colab dependencies
  - Local checkpoint and adapter storage in data/training/
  - Loads live conversation data from data/training/processed/
  - HF token from HF_TOKEN env var (or .env file)
  - --debug flag for quick 1-epoch smoke test (no HF push)
  - --resume flag to continue from saved LoRA adapters

Usage (from project root):
    # Debug / smoke test (fast, no push):
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/finetune_local.py --debug

    # Full round-1 training on default 0.5b model + push to HF:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/finetune_local.py

    # Fine-tune specific model sizes:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/finetune_local.py --base-model 7b
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/finetune_local.py --base-model 8b

    # Resume round 2 for a specific model:
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/finetune_local.py --base-model 7b --resume

Model profiles (base model -> HF repo):
    0.5b -> RayMelius/soci-agent-q4   (Qwen2.5-0.5B, batch=2, seq=2048)
    1.5b -> RayMelius/soci-agent-1b5  (Qwen2.5-1.5B, batch=2, seq=2048)
    3b   -> RayMelius/soci-agent-3b   (Qwen2.5-3B,   batch=2, seq=2048)
    7b   -> RayMelius/soci-agent-7b   (Qwen2.5-7B,   batch=1, seq=1024)
    8b   -> RayMelius/soci-agent-8b   (Llama-3.1-8B, batch=1, seq=1024)
"""

from __future__ import annotations

import sys
import io
import os

# Force UTF-8 stdout/stderr on Windows (unsloth prints emoji characters)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Disable torch.compile/inductor — triton 3.x on Windows doesn't export 'triton_key'
# which inductor needs at compile time.  Training still uses CUDA kernels, just not
# the AOT-compiled fusion path.  Has no meaningful effect on a single-GPU setup.
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Import unsloth FIRST so it can patch transformers before anything else loads.
# Then patch list_repo_templates to skip the 'additional_chat_templates' HF Hub
# check that fails on unsloth's quantized repos (transformers 4.56+ behavior).
import unsloth  # noqa: F401 — must be first
import transformers.utils.hub
import transformers.tokenization_utils_base
_noop = lambda *a, **kw: []
transformers.tokenization_utils_base.list_repo_templates = _noop
transformers.utils.hub.list_repo_templates = _noop

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# ── Parse args first (before heavy imports) ───────────────────────────────────
parser = argparse.ArgumentParser(description="Soci local fine-tune")
parser.add_argument("--resume",     action="store_true", help="Resume from saved LoRA adapters")
parser.add_argument("--debug",      action="store_true", help="Debug/smoke-test: 1 epoch, 20 examples, no push")
parser.add_argument("--no-push",    action="store_true", help="Skip HF Hub push")
parser.add_argument("--no-gguf",    action="store_true", help="Skip GGUF export")
parser.add_argument("--epochs",     type=int, default=None, help="Override epoch count")
parser.add_argument("--hf-repo",    default=None, help="HF repo ID (overrides default)")
parser.add_argument("--base-model", default="0.5b",
                    choices=["0.5b", "1.5b", "3b", "7b", "8b"],
                    help="Base model size to fine-tune (default: 0.5b)")
args = parser.parse_args()

# ── Model profiles (base model → unsloth ID, HF repo, VRAM settings) ──────────
_MODEL_PROFILES = {
    "0.5b": dict(
        model_id      = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        repo_name     = "soci-agent-q4",
        seq_len       = 2048,
        batch         = 2,
        grad_accum    = 4,
        lora_r        = 16,
        lora_targets  = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    ),
    "1.5b": dict(
        model_id      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        repo_name     = "soci-agent-1b5",
        seq_len       = 2048,
        batch         = 2,
        grad_accum    = 4,
        lora_r        = 16,
        lora_targets  = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    ),
    "3b": dict(
        model_id      = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        repo_name     = "soci-agent-3b",
        seq_len       = 2048,
        batch         = 2,
        grad_accum    = 4,
        lora_r        = 16,
        lora_targets  = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    ),
    # 7B and 8B: minimal LoRA to stay within 6.4 GB VRAM on RTX 4050 Laptop.
    # 7B in 4-bit uses ~3.8GB; only ~2.6GB left for activations + optimizer.
    # r=8, q+v only → ~5M trainable params, small optimizer footprint.
    "7b": dict(
        model_id      = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        repo_name     = "soci-agent-7b",
        seq_len       = 512,
        batch         = 1,
        grad_accum    = 8,
        lora_r        = 8,
        lora_targets  = ["q_proj", "v_proj"],
    ),
    "8b": dict(
        model_id      = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        repo_name     = "soci-agent-8b",
        seq_len       = 512,
        batch         = 1,
        grad_accum    = 8,
        lora_r        = 8,
        lora_targets  = ["q_proj", "v_proj"],
    ),
}
_PROFILE = _MODEL_PROFILES[args.base_model]

# ── Paths (per-model subdirs so runs don't clobber each other) ─────────────────
TRAIN_DIR        = Path("data/training")
MODEL_DIR        = TRAIN_DIR / args.base_model          # e.g. data/training/7b/
LORA_SAVE_DIR    = MODEL_DIR / "lora_adapters"
DATA_ARCHIVE_DIR = MODEL_DIR / "data_archive"
GGUF_DIR         = MODEL_DIR / "gguf"
CHECKPOINTS_DIR  = MODEL_DIR / "checkpoints"
ROUND_FILE       = MODEL_DIR / "training_round.json"
CORE_DATA_FILE   = TRAIN_DIR / "core_examples.json"
LIVE_DATA_FILE   = TRAIN_DIR / "processed" / "soci_training.jsonl"

for d in [LORA_SAVE_DIR, DATA_ARCHIVE_DIR, GGUF_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = _PROFILE["seq_len"]
HF_USERNAME    = "RayMelius"
HF_REPO_ID     = args.hf_repo or f"{HF_USERNAME}/{_PROFILE['repo_name']}"

# Load HF token
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    # Try to read from the project .env
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"')

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("[WARN] No CUDA GPU detected — training will be very slow on CPU.")
    print("       Consider running on Colab or a machine with a GPU.")
else:
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Patch unsloth fused CE loss for low-VRAM GPUs ─────────────────────────────
# unsloth_zoo._get_chunk_multiplier checks free VRAM *after* model load.
# On 6.4 GB GPUs the 7B model consumes almost all VRAM, leaving ~0 free,
# which causes it to raise "No or negligible GPU memory available".
# Replace with a version that falls back to 100 MB budget instead of raising.
import functools
import unsloth_zoo.fused_losses.cross_entropy_loss as _unsloth_ce

@functools.cache
def _safe_chunk_multiplier(vocab_size, target_gb=None):
    if target_gb is None:
        try:
            free, _ = torch.cuda.mem_get_info(0)
            free_gb = free / (1024 ** 3) * 0.5
        except Exception:
            free_gb = 0.0
        target_gb = max(free_gb, 0.1)   # always at least 100 MB budget
    if target_gb <= 1e-9:
        target_gb = 0.1
    multiplier = (vocab_size * 4 / (1024 ** 3)) / target_gb
    multiplier = multiplier / 4
    return multiplier

_unsloth_ce._get_chunk_multiplier = _safe_chunk_multiplier
print("Patched unsloth fused CE loss for low-VRAM GPU")

# ── Determine training round ──────────────────────────────────────────────────
RESUME = args.resume
if RESUME and ROUND_FILE.exists():
    round_info = json.loads(ROUND_FILE.read_text())
    CURRENT_ROUND = round_info["round"] + 1
    print(f"Resuming from round {round_info['round']} -> round {CURRENT_ROUND}")
    print(f"Previous loss: {round_info.get('final_loss', 'N/A')}")
elif RESUME:
    CURRENT_ROUND = 2
    print("No round file found, assuming round 2")
else:
    CURRENT_ROUND = 1
    print("Starting fresh (round 1)")

# ── Load model ────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel  # noqa: already imported via 'import unsloth'

if RESUME and LORA_SAVE_DIR.exists() and any(LORA_SAVE_DIR.iterdir()):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(LORA_SAVE_DIR),
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f"Resumed LoRA adapters from {LORA_SAVE_DIR}")
else:
    if RESUME:
        print(f"[WARN] No LoRA adapters at {LORA_SAVE_DIR}, starting fresh.")
        CURRENT_ROUND = 1
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = _PROFILE["model_id"],
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f"Fresh base model loaded (round 1): {_PROFILE['model_id']}")

# ── Attach LoRA ───────────────────────────────────────────────────────────────
if CURRENT_ROUND == 1:
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = _PROFILE["lora_r"],
        target_modules             = _PROFILE["lora_targets"],
        lora_alpha                 = _PROFILE["lora_r"],   # lora_alpha == r is standard
        lora_dropout               = 0,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )
    print("Fresh LoRA adapters attached")
else:
    model.gradient_checkpointing_enable()
    print(f"Resumed LoRA adapters from round {CURRENT_ROUND - 1}")

model.print_trainable_parameters()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are the reasoning engine for Soci, an LLM-powered city population simulator. "
    "You control AI agents (NPCs) living in a city. Each agent has a persona, needs "
    "(hunger, energy, social, purpose, comfort, fun), memories, and relationships. "
    "You receive structured context and must respond ONLY with valid JSON. "
    "Never add explanation outside the JSON."
)

# ── Load training data ────────────────────────────────────────────────────────
print("\nLoading training data...")

# 1. Core examples (from data/training/core_examples.json, extracted from v3 script)
core_examples: list[dict] = []
if CORE_DATA_FILE.exists():
    core_examples = json.loads(CORE_DATA_FILE.read_text(encoding="utf-8"))
    print(f"  Core examples: {len(core_examples)}")
else:
    print(f"  [WARN] {CORE_DATA_FILE} not found — run extract step or collect_training_data.py first")

# 2. Live collected data from the running simulation
live_examples: list[dict] = []
if LIVE_DATA_FILE.exists():
    with open(LIVE_DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                # Convert messages format -> instruction/response format
                msgs = ex.get("messages", [])
                if len(msgs) >= 3:
                    # Find system-ish context in user message; use Soci system prompt
                    user_content = msgs[1]["content"]
                    asst_content = msgs[2]["content"]
                    # Prepend persona context from system message as part of instruction
                    persona_ctx = msgs[0]["content"]
                    # Keep persona as part of instruction since we use unified system prompt
                    instruction = f"{persona_ctx}\n\n{user_content}"
                    live_examples.append({
                        "instruction": instruction,
                        "response": asst_content,
                    })
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"  Live examples: {len(live_examples)} (from Render simulation)")

# 3. Replay archived examples from previous rounds
replay_examples: list[dict] = []
if CURRENT_ROUND > 1:
    for archive_f in sorted(DATA_ARCHIVE_DIR.glob("round_*.json")):
        try:
            batch = json.loads(archive_f.read_text(encoding="utf-8"))
            replay_examples.extend(batch)
        except Exception:
            pass
    print(f"  Replay examples: {len(replay_examples)}")

# 4. New examples for this round (add yours here for incremental training)
new_examples_this_round: list[dict] = [
    # Add new instruction/response pairs here for incremental training rounds.
    # Example:
    # {"instruction": "You are playing Diana Novak, 41, grocery store owner. ...",
    #  "response": '{"action": "work", "location": "grocery_store", "reason": "..."}'},
]
if new_examples_this_round:
    print(f"  New examples this round: {len(new_examples_this_round)}")

# Merge and deduplicate by instruction
seen: set[str] = set()
all_examples: list[dict] = []
for ex in core_examples + live_examples + new_examples_this_round + replay_examples:
    key = ex.get("instruction", "")[:100]
    if key not in seen:
        seen.add(key)
        all_examples.append(ex)

if args.debug:
    all_examples = all_examples[:20]
    print(f"  DEBUG mode: using {len(all_examples)} examples")

print(f"  Total (deduped): {len(all_examples)}")

# ── Format into chat template ─────────────────────────────────────────────────
from datasets import Dataset

def format_example(ex: dict) -> dict:
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": ex["instruction"]},
        {"role": "assistant", "content": ex["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )}

dataset = Dataset.from_list(all_examples).map(format_example)
print(f"Formatted {len(dataset)} examples. Sample:")
print(dataset[0]["text"][:400])

# ── Training config ───────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

if args.debug:
    LR, EPOCHS, WARMUP, SCHEDULER = 2e-4, 1, 2, "linear"
    print(f"\nDEBUG: 1 epoch smoke test")
elif CURRENT_ROUND == 1:
    LR, EPOCHS, WARMUP, SCHEDULER = 2e-4, 3, 5, "linear"
    print(f"\nRound 1: Full training — LR={LR}, epochs={EPOCHS}")
else:
    LR, EPOCHS, WARMUP, SCHEDULER = 5e-5, 2, 10, "cosine"
    print(f"\nRound {CURRENT_ROUND}: Incremental — LR={LR}, epochs={EPOCHS}")

if args.epochs is not None:
    EPOCHS = args.epochs
    print(f"Epoch override: {EPOCHS}")

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 2,
    args = SFTConfig(
        per_device_train_batch_size = _PROFILE["batch"],
        gradient_accumulation_steps = _PROFILE["grad_accum"],
        warmup_steps                = WARMUP,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 5,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = SCHEDULER,
        seed                        = 42,
        output_dir                  = str(CHECKPOINTS_DIR),
        report_to                   = "none",
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
    ),
)

print(f"\nTraining round {CURRENT_ROUND} on {len(dataset)} examples...")
torch.cuda.empty_cache()   # free any cached fragments before training starts
stats = trainer.train()
print(f"\nRound {CURRENT_ROUND} complete!")
print(f"   Steps: {stats.global_step}  |  Final loss: {stats.training_loss:.4f}")

# ── Save LoRA adapters ────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapters to {LORA_SAVE_DIR}...")
model.save_pretrained(str(LORA_SAVE_DIR))
tokenizer.save_pretrained(str(LORA_SAVE_DIR))
print("  Saved.")

# ── Save round metadata ───────────────────────────────────────────────────────
round_info = {
    "round":          CURRENT_ROUND,
    "final_loss":     stats.training_loss,
    "global_steps":   stats.global_step,
    "total_examples": len(all_examples),
    "new_examples":   len(new_examples_this_round) + len(live_examples),
    "learning_rate":  LR,
    "epochs":         EPOCHS,
    "timestamp":      datetime.now().isoformat(),
}
ROUND_FILE.write_text(json.dumps(round_info, indent=2))
print(f"  Round info: {ROUND_FILE}")

# Archive new examples
all_new = new_examples_this_round + live_examples
if all_new:
    archive_file = DATA_ARCHIVE_DIR / f"round_{CURRENT_ROUND:03d}.json"
    archive_file.write_text(json.dumps(all_new, indent=2, ensure_ascii=False))
    print(f"  Archived {len(all_new)} new examples")

# Training history
history_file = TRAIN_DIR / "training_history.jsonl"
with open(history_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(round_info) + "\n")

# ── Quick inference test ──────────────────────────────────────────────────────
print(f"\n=== Testing after Round {CURRENT_ROUND} ===\n")
FastLanguageModel.for_inference(model)

def ask(question: str, label: str = "") -> None:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    encoded = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(encoded, "input_ids"):
        inp = encoded.input_ids.to("cuda")
    else:
        inp = encoded.to("cuda")
    out = model.generate(
        input_ids=inp, max_new_tokens=200,
        temperature=0.7, top_p=0.9, do_sample=True,
    )
    resp = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    print(f"[{label}]")
    print(f"Q: {question[:100]}...")
    try:
        parsed = json.loads(resp)
        print(f"A (valid JSON):\n{json.dumps(parsed, indent=2)}")
    except Exception:
        print(f"A (raw): {resp}")
    print("-" * 60)

ask(
    "You are playing Elena Vasquez, 34, software engineer. "
    "Needs: energy=0.3, hunger=0.7. Location: office. Time: 12:30. "
    "Decide next action. JSON: {\"action\": str, \"location\": str, \"reason\": str}",
    "decide_action",
)
ask(
    "You are playing Marcus Chen talking to Zoe. "
    "Zoe says: 'Marcus, I bombed my exam.' Continue as Marcus. "
    "JSON: {\"speech\": str, \"emotion\": str}",
    "conversation_turn",
)

# ── GGUF export ───────────────────────────────────────────────────────────────
# Windows: unsloth GGUF export requires building llama.cpp via apt-get (Linux only).
# Auto-skip on Windows; use --no-gguf on Linux too if llama.cpp isn't set up.
import platform
_on_windows = platform.system() == "Windows"
skip_gguf = args.no_gguf or args.debug or _on_windows
if _on_windows and not args.no_gguf and not args.debug:
    print("\nSkipping GGUF export (Windows — llama.cpp build not supported via unsloth on Win)")
    print("  To export GGUF manually, use llama.cpp's convert_hf_to_gguf.py")
    print(f"  LoRA merged weights saved to: {GGUF_DIR}/  (after push)")

if not skip_gguf:
    print(f"\nExporting GGUF Q4_K_M (takes a few minutes)...")
    model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q4_k_m")
    gguf_files = list(GGUF_DIR.glob("*.gguf"))
    for gf in gguf_files:
        print(f"  GGUF: {gf.name}  ({gf.stat().st_size / 1e6:.0f} MB)")
else:
    if args.debug:
        print("\nSkipping GGUF export (debug mode)")
    gguf_files = []

# ── Push to HuggingFace Hub ───────────────────────────────────────────────────
skip_push = args.no_push or args.debug
if skip_push:
    print("\nSkipping HF push (debug mode or --no-push)")
else:
    if not HF_TOKEN:
        print("\n[WARN] No HF_TOKEN found — skipping push.")
        print("  Set HF_TOKEN env var or add to .env file.")
    else:
        from huggingface_hub import login, HfApi
        print(f"\nPushing to HuggingFace: {HF_REPO_ID}")
        login(token=HF_TOKEN)
        api = HfApi()
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)

        # Push LoRA adapters
        print("  Uploading LoRA adapters...")
        api.upload_folder(
            folder_path = str(LORA_SAVE_DIR),
            repo_id     = HF_REPO_ID,
            repo_type   = "model",
            path_in_repo= "lora_adapters",
        )
        print(f"  LoRA -> https://huggingface.co/{HF_REPO_ID}/tree/main/lora_adapters")

        # Push GGUF file(s)
        for gf in gguf_files:
            mb = gf.stat().st_size / 1e6
            print(f"  Uploading {gf.name} ({mb:.0f} MB)...")
            api.upload_file(
                path_or_fileobj = str(gf),
                path_in_repo    = gf.name,
                repo_id         = HF_REPO_ID,
                repo_type       = "model",
            )
            print(f"  Done: https://huggingface.co/{HF_REPO_ID}/blob/main/{gf.name}")

        # Push round metadata
        api.upload_file(
            path_or_fileobj = str(ROUND_FILE),
            path_in_repo    = "training_round.json",
            repo_id         = HF_REPO_ID,
            repo_type       = "model",
        )

        print(f"\nUpload complete! Model at: https://huggingface.co/{HF_REPO_ID}")

# ── Training history display ──────────────────────────────────────────────────
print("\n=== Training History ===\n")
if history_file.exists():
    print(f"{'Round':>6} {'Loss':>8} {'Steps':>7} {'Examples':>9} {'New':>5} {'LR':>10} {'Date':>12}")
    print("-" * 65)
    with open(history_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            date = r.get("timestamp", "")[:10]
            print(f"{r['round']:>6} {r['final_loss']:>8.4f} {r['global_steps']:>7} "
                  f"{r['total_examples']:>9} {r['new_examples']:>5} "
                  f"{r['learning_rate']:>10.1e} {date:>12}")

print(f"\nTo resume: python scripts/finetune_local.py --resume")
print(f"LoRA adapters: {LORA_SAVE_DIR}")
if gguf_files:
    print(f"GGUF: {gguf_files[0]}")
print(f"\nOllama integration:")
print(f"  ollama create soci-agent -f Modelfile")
print(f"  set SOCI_PROVIDER=ollama && set OLLAMA_MODEL=soci-agent")
