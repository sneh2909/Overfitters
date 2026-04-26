#!/usr/bin/env python
"""
Submit the optimized GRPO training job to HuggingFace compute.

What this does
──────────────
1. Uploads train_grpo_optimized.py (and its dependencies) to your HF Hub.
2. Submits a training job on an A100 40 GB GPU.
3. Polls logs every 30 s until the job finishes.

The trained adapter lands at:
    https://huggingface.co/{HF_USERNAME}/house-md-grpo-optimized-gemma3-4b-v3

Cost estimate
─────────────
A10G-large with the full action menu is slower, so this file defaults to a
10-step validity experiment before committing to a longer GRPO run.

Usage
─────
    export HF_TOKEN=hf_...
    export HF_USERNAME=your_username
    export WANDB_API_KEY=wandb_v1_...
    python scripts/submit_grpo_optimized_job.py

    # To run 50 steps instead of the default 10:
    TOTAL_STEPS=50 python scripts/submit_grpo_optimized_job.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config — read from environment (same as SFT submit script)
# ──────────────────────────────────────────────────────────────────────────────

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USERNAME   = os.environ.get("HF_USERNAME", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
TOTAL_STEPS   = os.environ.get("TOTAL_STEPS", "50")   # override → TOTAL_STEPS=50
INSTANT_TESTS = os.environ.get("INSTANT_TESTS", "false")
ADAPTIVE_DISEASE_SAMPLING = os.environ.get("ADAPTIVE_DISEASE_SAMPLING", "false")
ADAPTIVE_SAMPLING_BETA = os.environ.get("ADAPTIVE_SAMPLING_BETA", "0.0")
ADAPTIVE_SAMPLING_MIN_STEPS = os.environ.get("ADAPTIVE_SAMPLING_MIN_STEPS", "10")

# The SFT adapter to start GRPO from.
SFT_MODEL_ID  = os.environ.get("SFT_MODEL_ID", f"{HF_USERNAME}/house-md-sft-gemma3-4b")

# HF Hub repos used by the job.
SCRIPTS_REPO  = f"{HF_USERNAME}/house-md-scripts-optimized"  # holds train_grpo_optimized.py
ENV_REPO      = f"{HF_USERNAME}/house-md-grpo-env"            # clinical_rl/ package + data/ (uploaded by submit_eval_job.py)

OUTPUT_REPO   = f"{HF_USERNAME}/house-md-grpo-optimized-gemma3-4b-v3-menu_compact"

# L4 24 GB — handles 4-bit Gemma 3 4B + group_size=8 comfortably.
# L4 is ~2x slower than A100 but less congested and cheaper.
# 100 steps ≈ 45-75 min on A10G/L4-class GPUs.
FLAVOR        = "a100-large"
JOB_TIMEOUT   = "8h"

DOCKER_IMAGE  = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def _require(val: str, name: str) -> str:
    if not val:
        print(f"ERROR: {name} is not set.\n  export {name}=<value>")
        sys.exit(1)
    return val


def _repo_exists(api, repo_id: str, repo_type: str) -> bool:
    """Return True if a HF Hub repo already exists (accessible with current token)."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except Exception:
        return False


def main() -> None:
    _require(HF_TOKEN,    "HF_TOKEN")
    _require(HF_USERNAME, "HF_USERNAME")

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    # ── Step 1: Upload train_grpo_optimized.py ───────────────────────────────
    print(f"\n[1/4] Uploading train_grpo_optimized.py to {SCRIPTS_REPO} ...")
    api.create_repo(repo_id=SCRIPTS_REPO, repo_type="model",
                    private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj="scripts/train_grpo_optimized.py",
        path_in_repo="train_grpo_optimized.py",
        repo_id=SCRIPTS_REPO, repo_type="model",
    )
    print("   Done.")

    # ── Step 2: Upload clinical_rl/ + data/ to env repo (skip if already there) ─
    if _repo_exists(api, ENV_REPO, "dataset"):
        print(f"\n[2/4] Env repo {ENV_REPO} already exists — skipping upload.")
    else:
        print(f"\n[2/4] Uploading clinical_rl/ + data/ to {ENV_REPO} ...")
        api.create_repo(repo_id=ENV_REPO, repo_type="dataset",
                        private=True, exist_ok=True)
        api.upload_folder(
            folder_path="clinical_rl",
            path_in_repo="clinical_rl",
            repo_id=ENV_REPO, repo_type="dataset",
            ignore_patterns=["__pycache__", "*.pyc"],
        )
        api.upload_folder(
            folder_path="data",
            path_in_repo="data",
            repo_id=ENV_REPO, repo_type="dataset",
        )
        print("   Done.")

    # ── Step 3: Build and submit the job ─────────────────────────────────────
    print(f"\n[3/4] Submitting GRPO job on {FLAVOR} ({TOTAL_STEPS} steps) ...")

    # The job container:
    #   - Installs git (needed by Unsloth's install from source)
    #   - Upgrades torch to ≥2.7 (fixes torchao compatibility — same issue as SFT)
    #   - Installs Unsloth, TRL, PEFT, etc.
    #   - Downloads train_grpo_optimized.py and all clinical_rl source from HF
    #   - Runs training
    job_command = [
        "bash", "-c",
        " && ".join([

            # Install system deps.
            "apt-get update -qq && apt-get install -y -q git",

            # Upgrade torch + torchvision only (torchaudio pins torch==2.6).
            "pip install -q --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124",

            # Install Unsloth + training deps.
            "pip install -q unsloth peft trl>=0.9.0 datasets accelerate "
            "bitsandbytes wandb huggingface_hub",

            # Download train_grpo_optimized.py from scripts repo.
            f'python -c "'
            f'from huggingface_hub import hf_hub_download; import shutil; '
            f'shutil.copy(hf_hub_download(repo_id=\\"{SCRIPTS_REPO}\\", '
            f'filename=\\"train_grpo_optimized.py\\", repo_type=\\"model\\"), '
            f'\\"train_grpo_optimized.py\\")"',

            # Download house-md-env — contains BOTH clinical_rl/ package
            # AND data/ (catalogs, cards, eval_set). This is the one-stop
            # download that gives the job everything it needs.
            f'python -c "'
            f'from huggingface_hub import snapshot_download; '
            f'snapshot_download(repo_id=\\"{ENV_REPO}\\", '
            f'repo_type=\\"dataset\\", local_dir=\\".\\")"',

            # Run training. clinical_rl/ and data/ are now in the working dir.
            "python train_grpo_optimized.py",
        ])
    ]

    job = api.run_job(
        image=DOCKER_IMAGE,
        command=job_command,
        flavor=FLAVOR,
        timeout=JOB_TIMEOUT,
        secrets={
            "HF_TOKEN":      HF_TOKEN,
            "HF_USERNAME":   HF_USERNAME,
            "WANDB_API_KEY": WANDB_API_KEY,
            "WANDB_PROJECT": "house-md",
            "SFT_MODEL_ID":  SFT_MODEL_ID,
            "TOTAL_STEPS":   TOTAL_STEPS,
            "GROUP_SIZE":    "8",
            "TEMPERATURE":   "0.9",
            "LR":            "1e-5",
            "CLIP_EPS":      "0.2",
            "MAX_NEW_TOKENS": "150",
            "FAST_GRPO_LOGPROBS": "true",
            "ADAPTIVE_DISEASE_SAMPLING": ADAPTIVE_DISEASE_SAMPLING,
            "ADAPTIVE_SAMPLING_BETA": ADAPTIVE_SAMPLING_BETA,
            "ADAPTIVE_SAMPLING_MIN_STEPS": ADAPTIVE_SAMPLING_MIN_STEPS,
            "EVAL_EVERY":    "999",
            "EVAL_PATIENTS": "5",
            "EVAL_AT_STEP_ZERO": "false",
            "CHECKPOINT_EVERY": "999",
            "MONITOR_EVERY": "1",
            "CURRICULUM_LEVEL_2_FRACTION": "0.33",
            "CURRICULUM_LEVEL_3_FRACTION": "0.67",
            "INSTANT_TESTS": INSTANT_TESTS,
            "INCLUDE_MENU":  "false",
            "COMPACT_MENU":  "true",
            "OUTPUT_HUB_ID": "house-md-grpo-optimized-gemma3-4b-v3",
        },
    )

    print(f"\n   Job submitted!")
    print(f"   Job ID : {job.id}")
    print(f"   Status : {job.status}")
    print(f"   Logs   : https://huggingface.co/jobs/{job.id}")
    print(f"   Output : https://huggingface.co/{OUTPUT_REPO}")

    # ── Step 4: Poll and stream logs ─────────────────────────────────────────
    print("\n[4/4] Polling for completion (Ctrl+C to detach without cancelling) ...")
    lines_seen = 0
    try:
        while True:
            info = api.inspect_job(job_id=job.id)

            # Print any new log lines.
            try:
                all_lines = list(api.fetch_job_logs(job_id=job.id))
                for line in all_lines[lines_seen:]:
                    print(f"[LOG] {line}", end="")
                lines_seen = len(all_lines)
            except Exception:
                pass

            if info.status.stage in ("completed", "failed", "cancelled", "timeout"):
                print(f"\nJob finished: {info.status.stage}")
                if info.status.stage == "completed":
                    print(f"\nGRPO adapter at: https://huggingface.co/{OUTPUT_REPO}")
                    print("Next step: run evaluation (E6) to compare SFT vs GRPO.")
                else:
                    print("Check logs above. Common fixes:")
                    print("  OOM → reduce GROUP_SIZE or MAX_NEW_TOKENS")
                    print("  KL > 0.5 repeatedly → lower LR from 1e-5 to 5e-6")
                break

            print(f"   status={info.status.stage}", flush=True)
            time.sleep(30)

    except KeyboardInterrupt:
        print(f"\nDetached from poller. Job {job.id} still running on HF.")
        print(f"Re-attach: python -c \"from huggingface_hub import HfApi; "
              f"[print(l,end='') for l in HfApi(token='{HF_TOKEN}').fetch_job_logs(job_id='{job.id}')]\"")


if __name__ == "__main__":
    main()
