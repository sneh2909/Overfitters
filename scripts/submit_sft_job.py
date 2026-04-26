#!/usr/bin/env python
"""
Submit the SFT training job to HuggingFace L4 compute.

What this script does
---------------------
1. Logs into HF Hub with your token.
2. Uploads data/sft_dataset.jsonl to your HF Hub as a dataset repo.
3. Uploads scripts/train_sft.py to your HF Hub (so the job can download it).
4. Calls api.run_job() with flavor="l4x1" — a single L4 GPU (24 GB VRAM).
5. Polls the job every 30 s and prints live logs until it finishes.

Cost estimate
-------------
L4 x1 ~ $0.60-0.80 / hr.  SFT on 2,151 samples for 1 epoch takes ~45-60 min.
Total cost: roughly $0.50 — well within your $30 budget.

Usage
-----
    HF_TOKEN=hf_... HF_USERNAME=your_username python scripts/submit_sft_job.py

Or set the env vars in your shell once:
    export HF_TOKEN=hf_...
    export HF_USERNAME=your_username
    python scripts/submit_sft_job.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USERNAME   = os.environ.get("HF_USERNAME", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# These are the repos we'll create (or reuse) on HF Hub.
DATASET_REPO = f"{HF_USERNAME}/house-md-sft-data"    # holds sft_dataset.jsonl
SCRIPTS_REPO = f"{HF_USERNAME}/house-md-scripts"     # holds train_sft.py
OUTPUT_REPO  = f"{HF_USERNAME}/house-md-sft-gemma3-4b"  # where the adapter lands

# Hardware: l4x1 = single L4 24 GB GPU.
FLAVOR = "l4x1"

# Job timeout: 2 hours should be very comfortable for a 1-epoch SFT run.
JOB_TIMEOUT = "2h"

# Docker image: PyTorch + CUDA 12.4 (matches L4 driver).
# Unsloth and TRL will be pip-installed inside the container.
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def _require(var: str, name: str) -> str:
    if not var:
        print(f"ERROR: {name} environment variable is not set.")
        print(f"  export {name}=<your value>")
        sys.exit(1)
    return var


def main() -> None:
    _require(HF_TOKEN,    "HF_TOKEN")
    _require(HF_USERNAME, "HF_USERNAME")

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # ------------------------------------------------------------------
    # Step 1: Upload the SFT dataset to HF Hub
    # ------------------------------------------------------------------
    print(f"\n[1/4] Uploading sft_dataset.jsonl to {DATASET_REPO} ...")
    api.create_repo(
        repo_id=DATASET_REPO, repo_type="dataset",
        private=True, exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj="data/sft_dataset.jsonl",
        path_in_repo="sft_dataset.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"    Dataset uploaded.")

    # ------------------------------------------------------------------
    # Step 2: Upload the training script to HF Hub
    # ------------------------------------------------------------------
    print(f"\n[2/4] Uploading train_sft.py to {SCRIPTS_REPO} ...")
    api.create_repo(
        repo_id=SCRIPTS_REPO, repo_type="model",
        private=True, exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj="scripts/train_sft.py",
        path_in_repo="train_sft.py",
        repo_id=SCRIPTS_REPO,
        repo_type="model",
    )
    print(f"    Script uploaded.")

    # ------------------------------------------------------------------
    # Step 3: Submit the job
    # The job container:
    #   - Starts from pytorch/pytorch (has CUDA + torch pre-installed)
    #   - pip-installs Unsloth + TRL (takes ~3 min)
    #   - Downloads train_sft.py from the Hub repo
    #   - Runs it, which downloads the dataset, trains, and pushes the adapter
    # ------------------------------------------------------------------
    print(f"\n[3/4] Submitting L4 training job ...")

    # Build the shell command that runs inside the container.
    # We use bash -c so we can chain multiple commands with &&.
    job_command = [
        "bash", "-c",
        # Install deps, download the training script, run it.
        " && ".join([
            # Step 1: install git (not in the pytorch base image).
            "apt-get update -qq && apt-get install -y -q git",

            # Step 2: upgrade torch to >=2.7 BEFORE installing unsloth.
            # Root cause of previous failure: unsloth pulls the latest torchao
            # which calls torch.utils._pytree.register_constant — a function
            # added in torch 2.7. The pytorch/pytorch:2.6.0 image only has 2.6.
            # Upgrading torch first lets unsloth/torchao auto-select a
            # compatible version pair.
            "pip install -q --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu124",

            # Step 3: install unsloth + training deps.
            # No [cu124-torch260] extra — we're on upgraded torch now.
            "pip install -q unsloth trl>=0.9.0 datasets accelerate bitsandbytes wandb",

            # Step 4: download the training script from HF Hub.
            f'python -c "from huggingface_hub import hf_hub_download; '
            f'import shutil; '
            f'shutil.copy(hf_hub_download(repo_id=\\"{SCRIPTS_REPO}\\", filename=\\"train_sft.py\\"), \\"train_sft.py\\")"',

            # Step 5: run training.
            "python train_sft.py",
        ])
    ]

    job = api.run_job(
        image=DOCKER_IMAGE,
        command=job_command,
        flavor=FLAVOR,
        timeout=JOB_TIMEOUT,
        # Secrets are injected as env vars — never stored in logs.
        secrets={
            "HF_TOKEN":      HF_TOKEN,
            "HF_USERNAME":   HF_USERNAME,
            "WANDB_API_KEY": WANDB_API_KEY,
            "WANDB_PROJECT": "house-md",
            # Training hyperparams — override via env if needed.
            "DATA_PATH":     DATASET_REPO,
            "OUTPUT_HUB_ID": "house-md-sft-gemma3-4b",
            "NUM_EPOCHS":    "1",
            "LR":            "2e-4",
            "BATCH_SIZE":    "4",
            "GRAD_ACCUM":    "4",
        },
    )

    print(f"    Job submitted!")
    print(f"    Job ID : {job.id}")
    print(f"    Status : {job.status}")
    print(f"    Logs   : https://huggingface.co/jobs/{job.id}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Poll the job and stream logs
    # ------------------------------------------------------------------
    print("[4/4] Waiting for job to complete (polling every 30s) ...")
    print("      Press Ctrl+C to stop polling — the job keeps running on HF.\n")

    poll_interval = 30
    lines_seen = 0   # track how many log lines we've already printed
    try:
        while True:
            info = api.inspect_job(job_id=job.id)
            status = info.status

            # Fetch ALL logs, skip the ones we've already printed.
            # fetch_job_logs returns all lines from the start each time.
            try:
                all_lines = list(api.fetch_job_logs(job_id=job.id))
                new_lines = all_lines[lines_seen:]
                for line in new_lines:
                    print(f"    [LOG] {line}", end="")
                lines_seen += len(new_lines)
            except Exception:
                pass  # Logs not always available mid-run; that's fine.

            if status in ("completed", "failed", "cancelled", "timeout"):
                print(f"\nJob finished with status: {status}")
                if status == "completed":
                    print(f"\nAdapter saved at: https://huggingface.co/{OUTPUT_REPO}")
                    print("Next step: run scripts/run_grpo.py (E5)")
                else:
                    print("Check the logs above for the error. Common fixes:")
                    print("  - OOM: lower BATCH_SIZE to 2 and increase GRAD_ACCUM to 8")
                    print("  - Unsloth install error: check PyTorch / CUDA version match")
                break

            print(f"    Status: {status}  (next check in {poll_interval}s)")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\nStopped polling. Job {job.id} is still running on HF.")
        print(f"Check status at: https://huggingface.co/jobs/{job.id}")


if __name__ == "__main__":
    main()
