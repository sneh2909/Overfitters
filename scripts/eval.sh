#!/usr/bin/env bash
# eval.sh — Submit a House M.D. evaluation job to Hugging Face Jobs (L4 GPU)
#
# Uses a single Python session for upload + job submission so the HF /whoami-v2
# endpoint is only hit once (rate limit fix).
#
# Usage:
#   # Evaluate base model
#   HF_TOKEN=hf_... HF_USERNAME=yourname ./scripts/eval.sh base
#
#   # Evaluate SFT checkpoint (adapter on HF Hub)
#   HF_TOKEN=hf_... HF_USERNAME=yourname ./scripts/eval.sh sft yourname/house-md-sft
#
#   # Evaluate SFT+GRPO checkpoint
#   HF_TOKEN=hf_... HF_USERNAME=yourname ./scripts/eval.sh grpo yourname/house-md-grpo
#
# Required env vars:
#   HF_TOKEN      — HuggingFace token with write access
#   HF_USERNAME   — Your HuggingFace username
#
# Optional env vars:
#   MODEL_ID      — Base model (default: google/gemma-3-4b-it)
#   ENV_REPO      — HF dataset repo for env package  (default: $HF_USERNAME/house-md-env)
#   RESULTS_REPO  — HF dataset repo for results      (default: $HF_USERNAME/house-md-results)
#   SKIP_UPLOAD   — Set to 1 to skip the upload step on repeat runs

set -euo pipefail
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Args + validation
# ---------------------------------------------------------------------------
MODEL_TAG="${1:-base}"
ADAPTER_PATH="${2:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  exit 1
fi
if [[ -z "${HF_USERNAME:-}" ]]; then
  echo "ERROR: HF_USERNAME is not set."
  exit 1
fi

MODEL_ID="${MODEL_ID:-google/gemma-3-4b-it}"
ENV_REPO="${ENV_REPO:-${HF_USERNAME}/house-md-env}"
RESULTS_REPO="${RESULTS_REPO:-${HF_USERNAME}/house-md-results}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

echo "============================================================"
echo "  House M.D. — HF Jobs Eval"
echo "============================================================"
echo "  Model tag    : ${MODEL_TAG}"
echo "  Base model   : ${MODEL_ID}"
echo "  Adapter      : ${ADAPTER_PATH:-<none>}"
echo "  Env repo     : ${ENV_REPO}  (private)"
echo "  Results repo : ${RESULTS_REPO}  (private)"
echo "  Skip upload  : ${SKIP_UPLOAD}"
echo "============================================================"
echo ""
echo "  Prerequisites:"
echo "  [!] HF Pro/Team/Enterprise account required for Jobs"
echo "  [!] Accept Gemma 3 license at huggingface.co/google/gemma-3-4b-it"
echo ""

# ---------------------------------------------------------------------------
# Single Python session: upload (optional) + job submission
# Keeps all HF API calls in one process — one whoami, no rate-limit issues.
# ---------------------------------------------------------------------------
python3 - <<PYEOF
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

token        = os.environ["HF_TOKEN"]
hf_username  = "${HF_USERNAME}"
model_tag    = "${MODEL_TAG}"
model_id     = "${MODEL_ID}"
adapter_path = "${ADAPTER_PATH}" or None
env_repo     = "${ENV_REPO}"
results_repo = "${RESULTS_REPO}"
skip_upload  = "${SKIP_UPLOAD}" == "1"

# Single API instance. No whoami call — namespace is passed explicitly
# to run_uv_job so the internal whoami fallback is never triggered.
api = HfApi(token=token)
print(f"Submitting as namespace: {hf_username}")

# --- upload ---
if not skip_upload:
    print("\n[1/3] Uploading env package + creating repos ...")

    api.create_repo(env_repo,     repo_type="dataset", exist_ok=True, private=True)
    api.create_repo(results_repo, repo_type="dataset", exist_ok=True, private=True)
    print(f"  Repos ready: {env_repo}, {results_repo}")

    api.upload_folder(
        folder_path="clinical_rl",
        path_in_repo="clinical_rl",
        repo_id=env_repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/**", "*.pyc", "*.pyo"],
    )
    print("  Uploaded: clinical_rl/")

    api.upload_folder(
        folder_path="data",
        path_in_repo="data",
        repo_id=env_repo,
        repo_type="dataset",
        ignore_patterns=["sft_dataset.jsonl", "oracle_trajectories.jsonl"],
    )
    print("  Uploaded: data/  (eval_set.jsonl + cards + catalogs)")
else:
    print("[1/3] Skipping upload (SKIP_UPLOAD=1)")
    # Still ensure results repo exists without re-uploading.
    api.create_repo(results_repo, repo_type="dataset", exist_ok=True, private=True)
    print(f"  Results repo ready: {results_repo}")

# --- submit job ---
print("\n[2/3] Submitting eval job to HF Jobs (l4x1, 90m timeout) ...")

env_vars = {
    "HF_USERNAME":   hf_username,
    "MODEL_TAG":     model_tag,
    "MODEL_ID":      model_id,
    "ENV_REPO":      env_repo,
    "RESULTS_REPO":  results_repo,
}
if adapter_path:
    env_vars["ADAPTER_PATH"] = adapter_path

job = api.run_uv_job(
    script="scripts/eval_hf.py",        # local file — hf packages it automatically
    flavor="l4x1",
    timeout="90m",
    env=env_vars,
    secrets={"HF_TOKEN": token},        # injected into container as HF_TOKEN env var
    namespace=hf_username,              # bypass internal whoami call
    token=token,
)

print(f"\n[3/3] Job submitted!")
print(f"  Job ID   : {job.id}")
print(f"  Monitor  : https://huggingface.co/jobs/{job.id}")
print(f"  Results  : https://huggingface.co/datasets/{results_repo}")
print()
print("Download results when complete:")
print(f"  hf download {results_repo} eval_{model_tag}.json --type dataset --local-dir results/")
PYEOF
