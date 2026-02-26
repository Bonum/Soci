"""
export_gguf_windows.py — Merge LoRA adapters and export to GGUF on Windows.

Pipeline:
  1. Load base model + LoRA adapters via Unsloth
  2. Merge LoRA into weights, save 16-bit safetensors (HF format)
  3. Download convert_hf_to_gguf.py from llama.cpp (if not cached)
  4. Convert merged model → F16 GGUF
  5. Quantize F16 GGUF → Q4_K_M via llama_cpp.llama_model_quantize
  6. Update Modelfile to point at the Q4_K_M GGUF

Usage (from project root):
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/export_gguf_windows.py
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/export_gguf_windows.py --model 7b
    "C:/Users/xabon/.conda/envs/ml-env/python.exe" scripts/export_gguf_windows.py --model 0.5b --push
"""

from __future__ import annotations

import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Unsloth must be first
import unsloth  # noqa: F401
import transformers.utils.hub
import transformers.tokenization_utils_base
_noop = lambda *a, **kw: []
transformers.tokenization_utils_base.list_repo_templates = _noop
transformers.utils.hub.list_repo_templates = _noop

import argparse
import subprocess
import urllib.request
from pathlib import Path

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Merge LoRA + export GGUF on Windows")
parser.add_argument("--model",      default="7b", choices=["0.5b","1.5b","3b","7b","8b"],
                    help="Which fine-tuned model to export (default: 7b)")
parser.add_argument("--quant",      default="q4_k_m",
                    choices=["f16","q4_k_m","q5_k_m","q8_0"],
                    help="Output quantisation (default: q4_k_m)")
parser.add_argument("--push",       action="store_true", help="Push GGUF to HF Hub after export")
parser.add_argument("--skip-merge", action="store_true", help="Skip merge if merged/ dir already exists")
parser.add_argument("--skip-quant", action="store_true", help="Skip quantisation, keep F16 GGUF only")
args = parser.parse_args()

# ── Model profile lookup ──────────────────────────────────────────────────────
_PROFILES = {
    "0.5b": dict(base_id="unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
                 hf_repo="RayMelius/soci-agent-q4", seq_len=2048),
    "1.5b": dict(base_id="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                 hf_repo="RayMelius/soci-agent-1b5", seq_len=2048),
    "3b":   dict(base_id="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                 hf_repo="RayMelius/soci-agent-3b", seq_len=2048),
    "7b":   dict(base_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                 hf_repo="RayMelius/soci-agent-7b", seq_len=512),
    "8b":   dict(base_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                 hf_repo="RayMelius/soci-agent-8b", seq_len=512),
}
PROFILE  = _PROFILES[args.model]
HF_REPO  = PROFILE["hf_repo"]
SEQ_LEN  = PROFILE["seq_len"]

TRAIN_DIR    = Path("data/training")
MODEL_DIR    = TRAIN_DIR / args.model           # e.g. data/training/7b/
LORA_DIR     = MODEL_DIR / "lora_adapters"
MERGED_DIR   = MODEL_DIR / "merged"
GGUF_DIR     = MODEL_DIR / "gguf"
CONVERT_CACHE = TRAIN_DIR / "_llama_convert"   # shared cache for the convert script

GGUF_DIR.mkdir(parents=True, exist_ok=True)
CONVERT_CACHE.mkdir(parents=True, exist_ok=True)

if not LORA_DIR.exists() or not any(LORA_DIR.iterdir()):
    print(f"[ERROR] No LoRA adapters found at {LORA_DIR}")
    print(f"  Run: python scripts/finetune_local.py --base-model {args.model}")
    sys.exit(1)

# ── Step 1: Merge LoRA → 16-bit safetensors ──────────────────────────────────
print(f"\n=== Step 1: Merge LoRA adapters ({args.model}) ===")

if args.skip_merge and MERGED_DIR.exists() and any(MERGED_DIR.glob("*.safetensors")):
    print(f"  Skipping merge — {MERGED_DIR} already exists.")
else:
    from unsloth import FastLanguageModel

    print(f"  Loading {LORA_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(LORA_DIR),
        max_seq_length = SEQ_LEN,
        dtype          = None,
        load_in_4bit   = True,
    )

    print(f"  Merging LoRA and saving 16-bit weights to {MERGED_DIR} ...")
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"  Merged model saved.")

# ── Step 2: Get convert_hf_to_gguf.py ────────────────────────────────────────
print(f"\n=== Step 2: Prepare llama.cpp convert script ===")

CONVERT_SCRIPT = CONVERT_CACHE / "convert_hf_to_gguf.py"
CONVERT_REQS   = CONVERT_CACHE / "requirements_convert.txt"

if not CONVERT_SCRIPT.exists():
    BASE_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master"
    print(f"  Downloading convert_hf_to_gguf.py ...")
    urllib.request.urlretrieve(f"{BASE_URL}/convert_hf_to_gguf.py", CONVERT_SCRIPT)

    # Also download the requirements file (needed for sentencepiece / tiktoken)
    try:
        urllib.request.urlretrieve(
            f"{BASE_URL}/requirements/requirements-convert_hf_to_gguf.txt",
            CONVERT_REQS,
        )
        print(f"  Installing convert dependencies ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "-r", str(CONVERT_REQS)])
    except Exception as e:
        print(f"  [WARN] Could not fetch/install convert requirements: {e}")
else:
    print(f"  Using cached {CONVERT_SCRIPT}")

# ── Step 3: Convert merged model → F16 GGUF ──────────────────────────────────
print(f"\n=== Step 3: Convert to F16 GGUF ===")

GGUF_F16 = GGUF_DIR / f"{args.model}-f16.gguf"

if GGUF_F16.exists():
    print(f"  Already exists: {GGUF_F16} ({GGUF_F16.stat().st_size / 1e9:.2f} GB)")
else:
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outfile", str(GGUF_F16),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed (exit {result.returncode})")
        sys.exit(1)
    print(f"  F16 GGUF: {GGUF_F16} ({GGUF_F16.stat().st_size / 1e9:.2f} GB)")

# ── Step 4: Quantise F16 → Q4_K_M (or other) ─────────────────────────────────
QUANT_TYPE_MAP = {
    "f16":    4,   # LLAMA_FTYPE_MOSTLY_F16
    "q8_0":  7,   # LLAMA_FTYPE_MOSTLY_Q8_0
    "q4_k_m": 15, # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "q5_k_m": 17, # LLAMA_FTYPE_MOSTLY_Q5_K_M
}

if args.skip_quant or args.quant == "f16":
    GGUF_FINAL = GGUF_F16
    print(f"\n=== Step 4: Skipping quantisation (using F16) ===")
else:
    print(f"\n=== Step 4: Quantise → {args.quant.upper()} ===")
    GGUF_FINAL = GGUF_DIR / f"{args.model}-{args.quant}.gguf"

    if GGUF_FINAL.exists():
        print(f"  Already exists: {GGUF_FINAL} ({GGUF_FINAL.stat().st_size / 1e6:.0f} MB)")
    else:
        import ctypes
        import llama_cpp

        ftype = QUANT_TYPE_MAP[args.quant]
        params = llama_cpp.llama_model_quantize_default_params()
        params.ftype = ftype
        params.nthread = 4
        params.allow_requantize = False

        print(f"  Quantising {GGUF_F16.name} → {GGUF_FINAL.name} ...")
        ret = llama_cpp.llama_model_quantize(
            str(GGUF_F16).encode(),
            str(GGUF_FINAL).encode(),
            ctypes.byref(params),
        )
        if ret != 0:
            print(f"[ERROR] Quantisation failed (return code {ret})")
            sys.exit(1)
        mb = GGUF_FINAL.stat().st_size / 1e6
        print(f"  {args.quant.upper()} GGUF: {GGUF_FINAL} ({mb:.0f} MB)")

# ── Step 5: Update Modelfile ──────────────────────────────────────────────────
print(f"\n=== Step 5: Update Modelfile ===")

modelfile_path = Path("Modelfile")
if modelfile_path.exists():
    content = modelfile_path.read_text(encoding="utf-8")
    # Comment out any existing FROM lines, then insert real one at top of FROM block
    gguf_rel = GGUF_FINAL.as_posix()   # forward slashes work in Modelfile on Windows
    new_from  = f"FROM ./{gguf_rel}"

    lines = content.splitlines()
    updated = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FROM ") and not stripped.startswith("#"):
            # Comment out old FROM
            updated.append(f"#{line}")
            if not inserted:
                updated.append(new_from)
                inserted = True
        else:
            updated.append(line)
    if not inserted:
        updated.insert(0, new_from)

    modelfile_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"  Modelfile updated: FROM → ./{gguf_rel}")
else:
    print(f"  [WARN] Modelfile not found — skipping update")

# ── Step 6: Push GGUF to HF Hub ──────────────────────────────────────────────
if args.push:
    print(f"\n=== Step 6: Push GGUF to {HF_REPO} ===")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    HF_TOKEN = line.split("=", 1)[1].strip().strip('"')

    if not HF_TOKEN:
        print("  [WARN] No HF_TOKEN — skipping push. Set HF_TOKEN in .env or env var.")
    else:
        from huggingface_hub import login, HfApi
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
        mb = GGUF_FINAL.stat().st_size / 1e6
        print(f"  Uploading {GGUF_FINAL.name} ({mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj = str(GGUF_FINAL),
            path_in_repo    = GGUF_FINAL.name,
            repo_id         = HF_REPO,
            repo_type       = "model",
        )
        print(f"  Done: https://huggingface.co/{HF_REPO}/blob/main/{GGUF_FINAL.name}")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"""
=== Export complete ===
GGUF : {GGUF_FINAL}
Size : {GGUF_FINAL.stat().st_size / 1e6:.0f} MB

To use with Ollama:
  ollama create soci-agent -f Modelfile
  ollama run soci-agent

Or for {args.model}:
  ollama create soci-agent-{args.model} -f Modelfile
  set OLLAMA_MODEL=soci-agent-{args.model}
  set SOCI_PROVIDER=ollama
""")
