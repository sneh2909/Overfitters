#!/usr/bin/env python
"""
E4 — SFT warm-start: fine-tune Gemma 3 4B on oracle diagnostic trajectories.

What this script does
---------------------
Loads the (prompt, completion) pairs from Phase 3, wraps them in Gemma's
chat template, then fine-tunes unsloth/gemma-3-4b-it with 4-bit LoRA.

Loss is applied ONLY to the completion tokens (the JSON action the model
emits). Prompt tokens — the system instructions, patient history, status
block — are masked out. This teaches the model to output valid actions
without wasting capacity memorising env-generated text.

After SFT the model should:
  - Emit valid JSON action objects ≥ 95% of the time (R8)
  - Follow the rough sequencing the oracle showed (ask → test → diagnose)
  - Know the closed vocabulary (25 question ids, 35 test ids, etc.)

GRPO (E5) then drives the model toward actually diagnosing correctly and
cost-efficiently; SFT just makes the format solid first.

Running locally
---------------
    conda activate metallm
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets accelerate
    HF_TOKEN=hf_... python scripts/train_sft.py

Running as HF Job
-----------------
    Use scripts/submit_sft_job.py — it uploads the dataset and launches this
    script inside an L4 container automatically.

Environment variables (set them or pass as args)
-------------------------------------------------
    HF_TOKEN         Your HuggingFace access token (required for Hub push)
    HF_USERNAME      Your HF username (required for Hub push)
    DATA_PATH        Local path OR HF dataset repo id  [data/sft_dataset.jsonl]
    OUTPUT_HUB_ID    HF model repo to push the adapter  [house-md-sft-gemma3-4b]
    NUM_EPOCHS       Training epochs                     [1]
    LR               Learning rate                       [2e-4]
    BATCH_SIZE       Per-device batch size               [4]
    GRAD_ACCUM       Gradient accumulation steps         [4]  → effective 16
    MAX_SEQ_LEN      Token budget per sample             [4096]
    LORA_RANK        LoRA rank                           [16]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — pulled from env vars with sensible defaults
# ---------------------------------------------------------------------------

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_USERNAME   = os.environ.get("HF_USERNAME", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "house-md")
DATA_PATH     = os.environ.get("DATA_PATH", "data/sft_dataset.jsonl")
OUTPUT_HUB_ID = os.environ.get("OUTPUT_HUB_ID", "house-md-sft-gemma3-4b")
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS", "1"))
LR            = float(os.environ.get("LR", "2e-4"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", "4"))   # effective batch = 16
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN", "4096"))
LORA_RANK     = int(os.environ.get("LORA_RANK", "16"))

MODEL_NAME = "unsloth/gemma-3-4b-it"

# Full hub repo id for pushing the trained adapter.
PUSH_HUB_ID = f"{HF_USERNAME}/{OUTPUT_HUB_ID}" if HF_USERNAME else None


# ---------------------------------------------------------------------------
# 1. HuggingFace login
# ---------------------------------------------------------------------------

def login_to_hub() -> None:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set — model won't be pushed to Hub.")
        return
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Logged in to HuggingFace Hub.")

    # W&B login — happens here so it's ready before the trainer starts.
    if WANDB_API_KEY:
        import wandb
        wandb.login(key=WANDB_API_KEY, relogin=True)
        print(f"Logged in to W&B (project: {WANDB_PROJECT}).")
    else:
        print("WANDB_API_KEY not set — training metrics won't be logged to W&B.")


# ---------------------------------------------------------------------------
# 2. Load SFT dataset
#    Supports both local JSONL (dev) and HF Hub dataset repo (job mode).
# ---------------------------------------------------------------------------

def load_sft_data() -> "datasets.Dataset":
    from datasets import Dataset, load_dataset

    if Path(DATA_PATH).exists():
        # Local file — used during development and local runs.
        print(f"Loading dataset from local file: {DATA_PATH}")
        rows = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
        # Keep only the two fields the trainer needs.
        return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    else:
        # Treat DATA_PATH as a HF Hub dataset repo id.
        # This is what the HF Job uses after submit_sft_job.py uploads the data.
        print(f"Loading dataset from HF Hub: {DATA_PATH}")
        ds = load_dataset(DATA_PATH, split="train")
        return ds


# ---------------------------------------------------------------------------
# 3. Load model + tokenizer with Unsloth
# ---------------------------------------------------------------------------

def load_model():
    from unsloth import FastLanguageModel
    import torch

    print(f"Loading {MODEL_NAME} in 4-bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto-detect (bfloat16 on A100/L4, float16 on T4)
        load_in_4bit=True,   # QLoRA — keeps the 4B model within L4's 24 GB
        token=HF_TOKEN or None,
    )

    # Gemma 3 uses the standard causal LM target modules.
    # LoRA only on attention + MLP projections — embeddings stay frozen.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,       # alpha == rank is a safe default
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # reduces VRAM ~30%
        random_state=42,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Format dataset with Gemma 3 chat template
#    Gemma 3's template wraps each turn:
#       <bos><start_of_turn>user
#       {prompt}<end_of_turn>
#       <start_of_turn>model
#       {completion}<end_of_turn><eos>
#
#    The tokenizer.apply_chat_template handles this automatically.
#    We store the final string in a "text" field for SFTTrainer.
# ---------------------------------------------------------------------------

def format_dataset(dataset, tokenizer):
    def _format(example):
        messages = [
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # don't add the trailing model turn start
            )
        }

    dataset = dataset.map(_format, desc="Formatting with Gemma chat template")
    # Drop the original fields — SFTTrainer only needs "text".
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    # Quick sanity: print one sample so we can verify the template is right.
    print("\n--- Sample formatted example (first 500 chars) ---")
    print(dataset[0]["text"][:500])
    print("...")
    return dataset


# ---------------------------------------------------------------------------
# 5. Build the SFT trainer
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, dataset):
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported, train_on_responses_only

    # Count training steps so we can set a sensible warmup and logging cadence.
    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(10, total_steps // 10)
    logging_steps   = max(1, total_steps // 50)   # ~50 log entries total

    print(f"\nTraining config:")
    print(f"  samples={len(dataset)}, epochs={NUM_EPOCHS}")
    print(f"  effective_batch={BATCH_SIZE * GRAD_ACCUM}")
    print(f"  steps/epoch={steps_per_epoch}, total_steps={total_steps}")
    print(f"  warmup={warmup_steps}, lr={LR}, lora_rank={LORA_RANK}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            # --- core training ---
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",

            # --- precision ---
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            # --- optimizer (8-bit saves ~1 GB vs standard adamw) ---
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,

            # --- dataset ---
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            dataset_num_proc=2,
            packing=False,   # False = one sample per sequence; safer for var-len

            # --- output ---
            output_dir="outputs/gemma3-4b-sft",
            save_strategy="epoch",
            save_total_limit=1,

            # --- logging ---
            logging_steps=logging_steps,
            # Use W&B if the API key was provided; otherwise just stdout.
            report_to="wandb" if WANDB_API_KEY else "none",
            run_name="sft-gemma3-4b-house-md",

            seed=42,
        ),
    )

    # KEY: mask loss on the prompt tokens — only train on the action JSON.
    # train_on_responses_only tells the trainer to zero-out all token
    # positions before <start_of_turn>model\n.  That's where the action
    # JSON lives, so only those ~30-80 tokens per step contribute gradient.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    return trainer


# ---------------------------------------------------------------------------
# 6. Save the adapter (and optionally push to Hub)
# ---------------------------------------------------------------------------

def save_and_push(model, tokenizer) -> None:
    local_out = "outputs/gemma3-4b-sft/final"
    print(f"\nSaving adapter to {local_out} ...")
    model.save_pretrained(local_out)
    tokenizer.save_pretrained(local_out)

    if PUSH_HUB_ID:
        print(f"Pushing adapter to HuggingFace Hub: {PUSH_HUB_ID} ...")
        model.push_to_hub(PUSH_HUB_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(PUSH_HUB_ID, token=HF_TOKEN)
        print(f"Done — model is at https://huggingface.co/{PUSH_HUB_ID}")
    else:
        print("HF_USERNAME not set — skipping Hub push. Adapter saved locally only.")


# ---------------------------------------------------------------------------
# 7. Post-training: quick format-validity check
#    Run the model on 5 eval patients and report how often it emits valid JSON.
#    This is a fast smoke test — full eval happens in E6.
# ---------------------------------------------------------------------------

def quick_format_check(model, tokenizer) -> None:
    """Inference on 5 eval patients; count valid-JSON output rate."""
    try:
        _here = Path(__file__).resolve()
        for _p in [_here.parents[1] / "house_md_env", _here.parents[1], _here.parent]:
            if (_p / "clinical_rl").exists():
                sys.path.insert(0, str(_p))
                break
        from clinical_rl.env import ClinicalEnv, load_catalogs, load_cards
        from clinical_rl.prompt import render_prompt, parse_action_json
    except ImportError:
        print("Skipping format check — clinical_rl package not on PYTHONPATH.")
        return

    eval_path = Path("data/eval_set.jsonl")
    if not eval_path.exists():
        print("Skipping format check — eval_set.jsonl not found.")
        return

    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    catalogs = load_catalogs("data")
    cards    = load_cards("data/cards")
    env      = ClinicalEnv(catalogs, cards)

    records = [json.loads(l) for l in eval_path.open()][:5]
    valid = 0
    for r in records:
        obs = env.reset(r["disease"], variant_id=r["variant_id"], seed=r["seed"])
        prompt = render_prompt(obs, catalogs)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(inputs, max_new_tokens=128, temperature=0.1,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        try:
            parse_action_json(generated)
            valid += 1
        except ValueError:
            pass

    print(f"\nFormat validity (5 patients): {valid}/5 = {valid*20}%")
    print("Target: ≥ 95% before GRPO. If below, consider running another epoch.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("E4 — SFT warm-start: Gemma 3 4B on House M.D. trajectories")
    print("=" * 60)

    login_to_hub()
    dataset   = load_sft_data()
    model, tokenizer = load_model()
    dataset   = format_dataset(dataset, tokenizer)
    trainer   = build_trainer(model, tokenizer, dataset)

    print("\nStarting training ...")
    trainer.train()
    print("Training complete.")

    save_and_push(model, tokenizer)
    quick_format_check(model, tokenizer)


if __name__ == "__main__":
    main()
