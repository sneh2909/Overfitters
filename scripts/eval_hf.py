# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2.0",
#   "transformers>=4.50.0",
#   "bitsandbytes>=0.43.0",
#   "peft>=0.11.0",
#   "accelerate>=0.30.0",
#   "huggingface-hub>=0.23.0",
#   "pyyaml>=6.0",
#   "openenv-core>=0.2.2",
#   "fastapi>=0.115.0",
#   "uvicorn[standard]>=0.24.0",
#   "pydantic>=2.0.0",
# ]
# ///
"""
Evaluation script for the House M.D. clinical RL environment.

Drives the model (base / SFT / GRPO) against the **OpenEnv server**
running inside the same container — the eval client talks WebSocket to
``http://127.0.0.1:8000`` rather than instantiating ``ClinicalEnv``
directly. This validates the deployed contract end-to-end while keeping
the job self-contained (no remote HF Space dependency).

The script:
  1. Downloads the env package (the ``house-md-env`` repo contents) from a HF
     dataset repo and stages it as an importable ``house_md_env`` package
     under ``/tmp``.
  2. Boots ``uvicorn house_md_env.server.app:app`` in the background and
     waits for ``/health``.
  3. Loads the model + adapter (4-bit Gemma 3 4B-IT by default).
  4. Runs all 45 held-out eval patients through the OpenEnv WebSocket
     client and pushes the results JSON to HF Hub.

Environment variables (set via secrets in the hf jobs CLI):
  HF_TOKEN        — write-capable HF token
  HF_USERNAME     — your HF username (used to locate the env repo + results repo)
  MODEL_TAG       — "base" | "sft" | "grpo"  (default: "base")
  ADAPTER_PATH    — HF Hub adapter id, e.g. "sneh/house-md-sft"  (optional)
  MODEL_ID        — base model id  (default: "google/gemma-3-4b-it")
  ENV_REPO        — HF dataset repo that holds the house-md-env package
                    (default: "{HF_USERNAME}/house-md-env")
  RESULTS_REPO    — HF dataset repo to push results to
                    (default: "{HF_USERNAME}/house-md-results")
  EVAL_BASE_URL   — override the env URL the eval client talks to
                    (default: "http://127.0.0.1:8000"). Set this if you'd
                    rather point at a remote HF Space than spawn a local
                    uvicorn — useful when iterating on the env.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download


# ---------------------------------------------------------------------------
# 1. Download env package + stage as importable house_md_env package
# ---------------------------------------------------------------------------

def _setup_env_package() -> Path:
    """Pull the env from HF Hub and lay it out so both the *server*
    (``uvicorn house_md_env.server.app:app``) and the *client*
    (``from house_md_env import HouseMDEnv``) can import the same code.

    The HF dataset repo is expected to contain the contents of the
    repo's root (i.e. the repo
    has ``__init__.py``, ``models.py``, ``client.py``, ``server/``,
    ``clinical_rl/``, ``data/`` directly at the top).

    snapshot_download returns a hash-named directory, so we copy it under
    ``/tmp/house_md_pkg/house_md_env/`` to give the package a stable,
    importable name.
    """
    hf_user = os.environ["HF_USERNAME"]
    env_repo = os.environ.get("ENV_REPO", f"{hf_user}/house-md-env")
    print(f"Downloading env package from {env_repo} ...")
    src = Path(
        snapshot_download(env_repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    )

    # Stage under a stable importable name. Any prior copy is wiped so
    # cached job retries can't see stale code.
    pkg_root = Path("/tmp/house_md_pkg")
    dst = pkg_root / "house_md_env"
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".cache"),
    )

    # Make `house_md_env` importable. Also expose `clinical_rl` at the
    # top level (the eval helpers import `from clinical_rl.env...` and
    # `from clinical_rl.prompt import ...`); we just symlink/copy the
    # vendored copy so both import paths resolve.
    sys.path.insert(0, str(pkg_root))
    sys.path.insert(0, str(dst))
    print(f"Env package staged at: {dst}")
    return dst


# ---------------------------------------------------------------------------
# 2. Boot uvicorn for the OpenEnv server inside the same container
# ---------------------------------------------------------------------------

def _start_env_server(pkg_root: Path) -> tuple[object, str]:
    """Spawn ``uvicorn house_md_env.server.app:app`` in the background and
    return ``(popen, base_url)`` once /health responds 200."""
    from _eval_openenv import find_free_port, spawn_uvicorn, wait_for_health

    port = find_free_port(default=8000)
    log_path = Path("/tmp/house_md_env_server.log")
    print(f"Starting OpenEnv server (uvicorn) on port {port} ...")
    print(f"Server logs: {log_path}")

    popen = spawn_uvicorn(
        app="house_md_env.server.app:app",
        cwd=pkg_root.parent,            # /tmp/house_md_pkg
        port=port,
        pythonpath=pkg_root.parent,     # so `import house_md_env` resolves
        log_path=log_path,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(base_url, timeout_s=120.0)
    except Exception:
        # Surface server logs into the job log so the failure is debuggable.
        if log_path.exists():
            print("\n=== uvicorn log (tail) ===")
            print(log_path.read_text()[-4000:])
            print("=== end uvicorn log ===\n")
        popen.terminate()
        raise
    print(f"Env server is healthy: {base_url}")
    return popen, base_url


# ---------------------------------------------------------------------------
# 3. Model loading (4-bit quantised, optional LoRA adapter)
# ---------------------------------------------------------------------------

def _load_model(model_id: str, adapter_path: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (4-bit) ... adapter={adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # Gemma 3 + bitsandbytes 4-bit needs eager attention on Ampere/Lovelace;
        # SDPA path can raise RuntimeError with quantised weights.
        attn_implementation="eager",
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA adapter loaded.")

    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# 4. Single-turn generation (closure-friendly)
# ---------------------------------------------------------------------------

def _make_generate_fn(tokenizer, model, max_new_tokens: int = 300):
    def _generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        # apply_chat_template(tokenize=False) -> formatted string, then
        # tokenize separately. This avoids the BatchEncoding vs Tensor
        # ambiguity introduced in transformers >=4.50 when return_tensors="pt"
        # is passed to apply_chat_template directly.
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


# ---------------------------------------------------------------------------
# 5. Aggregate summary
# ---------------------------------------------------------------------------

def _summarise(results: list[dict], n_patients: int) -> dict:
    if not results:
        return {}

    rubrics = ["r1_accuracy", "r2_cost", "r6_anchoring", "r7_safety", "r8_format", "total"]
    avg_r = {
        k: round(sum(r["rewards"].get(k, 0.0) for r in results) / len(results), 4)
        for k in rubrics
    }

    difficulties = sorted({r["difficulty"] for r in results})
    by_diff: dict[str, dict] = {}
    for diff in difficulties:
        sub = [r for r in results if r["difficulty"] == diff]
        by_diff[diff] = {
            "n":               len(sub),
            "correct_pct":     round(100 * sum(1 for r in sub if r["correct"]) / len(sub), 1),
            "avg_total":       round(sum(r["rewards"].get("total", 0.0) for r in sub) / len(sub), 4),
            "avg_oracle_pct":  round(sum(r["oracle_ceiling_pct"] for r in sub) / len(sub), 1),
        }

    total_actions   = sum(r["steps_taken"] for r in results)
    total_malformed = sum(r["malformed_actions"] for r in results)

    return {
        "n_evaluated":         len(results),
        "n_patients_total":    n_patients,
        "correct":             sum(1 for r in results if r["correct"]),
        "correct_pct":         round(100 * sum(1 for r in results if r["correct"]) / len(results), 1),
        "avg_rewards":         avg_r,
        "avg_oracle_pct":      round(sum(r["oracle_ceiling_pct"] for r in results) / len(results), 1),
        "avg_steps":           round(sum(r["steps_taken"] for r in results) / len(results), 1),
        "avg_cost":            round(sum(r["cost"] for r in results) / len(results), 1),
        "total_malformed":     total_malformed,
        "total_actions":       total_actions,
        "malformed_rate":      round(total_malformed / max(total_actions, 1), 3),
        "by_difficulty":       by_diff,
    }


def _print_summary(summary: dict, tag: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  EVAL RESULTS — {tag.upper()}")
    print(sep)
    print(f"  Accuracy       : {summary['correct']}/{summary['n_evaluated']}  ({summary['correct_pct']}%)")
    print(f"  Avg total      : {summary['avg_rewards']['total']}")
    print(f"  vs Oracle ceil : {summary['avg_oracle_pct']}%")
    print(f"  Avg steps used : {summary['avg_steps']}")
    print(f"  Avg cost ($)   : {summary['avg_cost']}")
    print(f"  Malformed JSON : {summary['total_malformed']}/{summary['total_actions']} actions  "
          f"({summary['malformed_rate']*100:.1f}%)"
          f"  ← SFT target: <5%")
    print()
    print("  Per-rubric averages:")
    for k, v in summary["avg_rewards"].items():
        if k != "total":
            print(f"    {k:<22} {v:>7.4f}")
    print()
    print("  By difficulty  (correct% / avg_total / % of oracle):")
    for diff, d in summary["by_difficulty"].items():
        print(
            f"    {diff:<30}  {d['correct_pct']:>5.1f}%  "
            f"{d['avg_total']:>5.2f}  {d['avg_oracle_pct']:>5.1f}%"
        )
    print(sep)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    hf_user      = os.environ["HF_USERNAME"]
    model_tag    = os.environ.get("MODEL_TAG", "base")
    model_id     = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")
    adapter_path = os.environ.get("ADAPTER_PATH") or None
    results_repo = os.environ.get("RESULTS_REPO", f"{hf_user}/house-md-results")
    eval_url_override = os.environ.get("EVAL_BASE_URL")  # optional remote target

    print(f"Job config: tag={model_tag}  model={model_id}  adapter={adapter_path}")
    print(f"Results will be pushed to: {results_repo}")

    # --- setup env package + (optionally) spin up local server ---
    pkg_root = _setup_env_package()

    # eval_hf.py lives next to _eval_openenv.py in the repo, but on HF Jobs
    # we pulled this script down standalone — Path(__file__).parent is the
    # cwd where it was placed. Make sure the helper module is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    server_proc = None
    if eval_url_override:
        base_url = eval_url_override
        print(f"Using EVAL_BASE_URL override: {base_url}")
    else:
        server_proc, base_url = _start_env_server(pkg_root)

    try:
        from clinical_rl.env import load_catalogs
        from house_md_env import HouseMDEnv
        from _eval_openenv import EpisodeResult, run_episode_openenv

        # Catalogs come from the staged env package's data/ dir — same files
        # the server is reading, so the prompt menu is in lock-step with the
        # action vocabulary the server will accept.
        data_dir = pkg_root / "data"
        catalogs = load_catalogs(data_dir)

        eval_patients: list[dict] = []
        with open(data_dir / "eval_set.jsonl") as fh:
            for line in fh:
                eval_patients.append(json.loads(line))
        print(f"Loaded {len(eval_patients)} eval patients")

        # --- load model ---
        tokenizer, model = _load_model(model_id, adapter_path)
        generate_fn = _make_generate_fn(tokenizer, model)

        # --- eval loop over the OpenEnv WebSocket client ---
        sync_env = HouseMDEnv(base_url=base_url).sync()
        results: list[dict] = []
        with sync_env as env:
            for i, patient in enumerate(eval_patients):
                pid = patient["patient_id"]
                print(f"[{i+1:>2}/{len(eval_patients)}] {pid} ...", end=" ", flush=True)
                try:
                    ep_res: EpisodeResult = run_episode_openenv(
                        env, catalogs, patient, generate_fn
                    )
                    row = ep_res.to_dict()
                    row["model_tag"]    = model_tag
                    row["model_id"]     = model_id
                    row["adapter_path"] = adapter_path
                    results.append(row)
                    print(
                        f"correct={row['correct']}  "
                        f"total={row['rewards'].get('total', 0.0):.2f}  "
                        f"steps={row['steps_taken']}  "
                        f"malformed={row['malformed_actions']}"
                    )
                except Exception as exc:
                    print(f"ERROR — {exc}")
                    traceback.print_exc()

        # --- summarise ---
        summary = _summarise(results, len(eval_patients))
        _print_summary(summary, model_tag)

        output = {
            "model_tag":    model_tag,
            "model_id":     model_id,
            "adapter_path": adapter_path,
            "base_url":     base_url,
            "summary":      summary,
            "patients":     results,
        }

        # --- save + push ---
        out_file = Path(f"/tmp/eval_{model_tag}.json")
        out_file.write_text(json.dumps(output, indent=2))
        print(f"\nSaved locally: {out_file}")

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        try:
            api.create_repo(results_repo, repo_type="dataset", exist_ok=True, private=True)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=str(out_file),
            path_in_repo=f"eval_{model_tag}.json",
            repo_id=results_repo,
            repo_type="dataset",
        )
        print(f"Pushed → {results_repo}/eval_{model_tag}.json")
    finally:
        if server_proc is not None:
            print("Shutting down env server ...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except Exception:
                server_proc.kill()


if __name__ == "__main__":
    main()
