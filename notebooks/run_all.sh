#!/usr/bin/env bash
# run_all.sh — execute every notebook end-to-end without the Jupyter UI.
#
# Usage:
#   HF_TOKEN=hf_... HF_USERNAME=SnehShah ./notebooks/run_all.sh [stage]
#
# stages: explore | sft | grpo | eval | all (default)
#
# Requirements:
#   pip install jupyter nbconvert papermill ipykernel
#
# Notes:
#   - explore           : ~3 min, CPU only
#   - sft               : ~12 min, GPU recommended
#   - grpo              : ~25 min, GPU recommended
#   - eval              : ~1 min, CPU only (loads pre-computed JSONs)

set -euo pipefail

NB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-all}"

run_nb() {
  local nb="$1"
  echo ""
  echo "=========================================================="
  echo " running $nb"
  echo "=========================================================="
  jupyter nbconvert --to notebook --execute "${NB_DIR}/$nb" \
    --output "executed_$nb" \
    --ExecutePreprocessor.timeout=3600
}

case "$STAGE" in
  explore) run_nb 01_explore_env.ipynb ;;
  sft)     run_nb 02_sft.ipynb ;;
  grpo)    run_nb 03_grpo.ipynb ;;
  eval)    run_nb 04_eval_compare.ipynb ;;
  all)
    run_nb 01_explore_env.ipynb
    run_nb 02_sft.ipynb
    run_nb 03_grpo.ipynb
    run_nb 04_eval_compare.ipynb
    ;;
  *)
    echo "unknown stage '$STAGE' — use one of: explore, sft, grpo, eval, all"
    exit 1
    ;;
esac

echo ""
echo "all done. executed notebooks saved as ${NB_DIR}/executed_*.ipynb"
