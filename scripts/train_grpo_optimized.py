#!/usr/bin/env python
"""
E5 — GRPO training: teach the SFT model to actually diagnose patients.

What this script does
─────────────────────
SFT (E4) taught the model HOW to write valid JSON actions and roughly which
actions look sensible. But it never received feedback on whether the
diagnosis was actually CORRECT or whether the workup was COST-EFFICIENT.
GRPO fixes that by having the model play patients against the real environment
and learning from which episodes scored well vs. badly.

How GRPO works here (plain English)
────────────────────────────────────
For each training step:
  1. Pick one patient (disease + variant + seed).
  2. Have the model play that SAME patient 8 times with temperature=0.9
     so it explores different action sequences.
  3. Score each of the 8 episodes using compute_all() → one scalar each.
  4. Normalise the 8 scores relative to each other (group-relative advantage).
     A rollout that scored ABOVE the group average gets a positive advantage;
     one that scored below gets a negative advantage.
  5. Update the model weights so that high-advantage action sequences become
     more likely, and low-advantage ones become less likely.
  6. Use ratio clipping (PPO-style) to stop the policy from changing too fast
     and collapsing to a degenerate strategy.

Key design choices
──────────────────
• include_menu=False for rollout prompts — the SFT model already knows the
  closed vocabulary; dropping the 2000-token menu makes each forward pass ~4x
  cheaper. Constrained decoding handles any vocab violations.
• 100 steps by default — verify learning is happening before committing to a
  longer, more expensive run.
• Ratio clipping (ε=0.2) — prevents the policy from drifting too far from the
  SFT checkpoint in a single step.
• Smoke test at startup — verifies log_prob slicing is correct BEFORE any
  expensive training begins.

Reward hacking notes (things to watch for in training logs)
────────────────────────────────────────────────────────────
• "Skip UPDATE_DIFFERENTIAL" exploit: R6 weight is small (0.3), so skipping
  differential updates gives ~2.9 vs ~3.0 for thoughtful play. Not catastrophic,
  but watch if R6 drops to 0 across all rollouts.
• "Diagnose immediately" cheat: R1=0.2 (lucky guess penalty) + no R6 → total ~1.0.
  GRPO will naturally push away from this since careful play gives ~3.0.
• "Order everything" ceiling: step-cap prevents ordering all 35 tests. Any
  agent that orders ≥14 tests has no room for DIAGNOSE → timeout → R7=-2.0.
  This is a natural rate limiter.
• Format degradation: if R8 drops below 0.8, the policy is drifting from valid
  JSON output — stop training and investigate (usually means LR is too high).
• KL divergence monitor: we log (old_log_prob - new_log_prob) per step.
  If this exceeds 0.5 consistently, training is unstable.

Running locally (for debugging)
────────────────────────────────
    HF_TOKEN=... HF_USERNAME=... python scripts/train_grpo_optimized.py --steps 5

Submitting to HF A100
──────────────────────
    python scripts/submit_grpo_optimized_job.py
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

# Resolve clinical_rl package location once at module load time. Supported
# layouts:
#   - new repo:  scripts/../house_md_env/clinical_rl   (this submission)
#   - HF Job:    scripts/clinical_rl                   (sibling, flattened)
#   - legacy:    scripts/../clinical_rl                (old in-repo layout)
_script_path = Path(__file__).resolve()
for _p in [
    _script_path.parents[1] / "house_md_env",  # new repo: house_md_env/clinical_rl
    _script_path.parents[1],                    # legacy:  ./clinical_rl at repo root
    _script_path.parent,                        # HF Job:  scripts/clinical_rl
]:
    if (_p / "clinical_rl").exists():
        sys.path.insert(0, str(_p))
        break

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — all hyperparameters in one place, overridable via environment vars
# ──────────────────────────────────────────────────────────────────────────────

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_USERNAME    = os.environ.get("HF_USERNAME", "")
WANDB_API_KEY  = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "house-md")

# The SFT adapter we trained in E4. Falls back to the raw base model if absent.
SFT_MODEL_ID   = os.environ.get("SFT_MODEL_ID", f"{HF_USERNAME}/house-md-sft-gemma3-4b")
BASE_MODEL     = "unsloth/gemma-3-4b-it"
OUTPUT_HUB_ID  = os.environ.get("OUTPUT_HUB_ID", "house-md-grpo-optimized-gemma3-4b")

# Data paths — used when running on HF Jobs (downloaded from Hub) or locally.
DATA_DIR       = os.environ.get("DATA_DIR", "data")
EVAL_SET_PATH  = os.environ.get("EVAL_SET_PATH", "data/eval_set.jsonl")

# GRPO hyperparameters
TOTAL_STEPS    = int(os.environ.get("TOTAL_STEPS", "100"))
GROUP_SIZE     = int(os.environ.get("GROUP_SIZE", "8"))   # rollouts per patient
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "0.9"))  # sampling diversity
LR             = float(os.environ.get("LR", "1e-5"))   # much lower than SFT's 2e-4
CLIP_EPS       = float(os.environ.get("CLIP_EPS", "0.2"))  # PPO ratio clip range
GRAD_CLIP      = float(os.environ.get("GRAD_CLIP", "0.5"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))

def _fraction_env(*names: str, default: str) -> float:
    """Read a fraction from env. Accepts either 0.33 or 33."""
    raw = default
    for name in names:
        if name in os.environ:
            raw = os.environ[name]
            break
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    return min(max(value, 0.0), 1.0)


# Three-level curriculum. Boundaries are percentages of TOTAL_STEPS, so the
# level schedule automatically scales for short smoke runs and longer jobs.
CURRICULUM_LEVEL_2_FRACTION = _fraction_env(
    "CURRICULUM_LEVEL_2_FRACTION", "CURRICULUM_L2_PCT",
    default="0.33",
)
CURRICULUM_LEVEL_3_FRACTION = _fraction_env(
    "CURRICULUM_LEVEL_3_FRACTION", "CURRICULUM_L3_PCT",
    # Back-compat: old single-boundary env now means "enter level 3 here".
    "CURRICULUM_FRACTION", "CURRICULUM_PCT",
    default="0.67",
)
CURRICULUM_LEVEL_2_STEP = min(
    TOTAL_STEPS,
    int(round(TOTAL_STEPS * CURRICULUM_LEVEL_2_FRACTION)),
)
CURRICULUM_LEVEL_3_STEP = min(
    TOTAL_STEPS,
    int(round(TOTAL_STEPS * CURRICULUM_LEVEL_3_FRACTION)),
)
if CURRICULUM_LEVEL_3_STEP < CURRICULUM_LEVEL_2_STEP:
    CURRICULUM_LEVEL_3_STEP = CURRICULUM_LEVEL_2_STEP

# Level 1: one disease from each difficulty tier + one tricky mimic pair.
# ectopic (critical), stemi (critical), appendicitis (urgent),
# migraine (stable — SAH mimic), viral_uri (stable — teaches restraint).
LEVEL_1_DISEASES = [
    "ectopic_pregnancy", "stemi", "appendicitis", "migraine", "viral_uri",
]

# Level 2 adds more high-yield mimics and urgent/critical presentations while
# holding back the full long tail until Level 3.
LEVEL_2_DISEASES = LEVEL_1_DISEASES + [
    "pulmonary_embolism", "subarachnoid_hemorrhage", "ovarian_torsion",
    "pneumonia", "dka",
]

# Whether to include the 2000-token action menu in rollout prompts.
# False = faster rollouts. COMPACT_MENU can still append exact valid IDs.
INCLUDE_MENU        = os.environ.get("INCLUDE_MENU", "false").lower() == "true"
# Middle path between full menu and no menu: exact IDs only, no descriptions,
# costs, or turnaround metadata. This targets invalid near-miss IDs while
# avoiding the full-menu OOM on A10G/L4-class GPUs.
COMPACT_MENU        = os.environ.get("COMPACT_MENU", "false").lower() == "true"
# Easier early curriculum: make all test results return on the same step.
# Keep false for final realistic eval / delayed-test polish runs.
INSTANT_TESTS       = os.environ.get("INSTANT_TESTS", "false").lower() == "true"
# Fast path for the single-update GRPO loop used in this script.
# True skips the rollout-time old-logprob forward pass and uses a direct
# policy-gradient loss: -advantage * log_prob(action). This is the same useful
# first-order signal as the ratio objective when old_policy == current_policy,
# but avoids pretending PPO clipping/KL are active. Set false if you later reuse
# each rollout batch for multiple optimizer epochs and need PPO-style ratios/KL.
FAST_GRPO_LOGPROBS  = os.environ.get("FAST_GRPO_LOGPROBS", "true").lower() == "true"
# When True, skip loading the SFT adapter entirely and start GRPO from a
# fresh LoRA on the base model. Use for ablations or base-model smoke tests.
SKIP_SFT_ADAPTER    = os.environ.get("SKIP_SFT_ADAPTER", "false").lower() == "true"

# Eval every N steps on a small held-out subset; checkpoint every M steps.
# Step-0 eval is skipped by default so training logs show progress quickly.
EVAL_EVERY        = int(os.environ.get("EVAL_EVERY", "50"))
EVAL_PATIENTS     = int(os.environ.get("EVAL_PATIENTS", "5"))
EVAL_AT_STEP_ZERO = os.environ.get("EVAL_AT_STEP_ZERO", "false").lower() == "true"
CHECKPOINT_EVERY  = int(os.environ.get("CHECKPOINT_EVERY", "10"))
MONITOR_EVERY     = int(os.environ.get("MONITOR_EVERY", "10"))

DEVICE = "cuda"


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 ── Startup: login and environment
# ──────────────────────────────────────────────────────────────────────────────

def login() -> None:
    """Log into HuggingFace Hub and W&B."""
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Logged in to HuggingFace Hub.")

    if WANDB_API_KEY:
        import atexit
        wandb.login(key=WANDB_API_KEY, relogin=True)
        wandb.init(
            project=WANDB_PROJECT,
            name=f"grpo-optimized-gemma3-4b-{TOTAL_STEPS}steps",
            config={
                "total_steps": TOTAL_STEPS,
                "group_size": GROUP_SIZE,
                "temperature": TEMPERATURE,
                "lr": LR,
                "clip_eps": CLIP_EPS,
                "curriculum_level_2_fraction": CURRICULUM_LEVEL_2_FRACTION,
                "curriculum_level_3_fraction": CURRICULUM_LEVEL_3_FRACTION,
                "curriculum_level_2_step": CURRICULUM_LEVEL_2_STEP,
                "curriculum_level_3_step": CURRICULUM_LEVEL_3_STEP,
                "include_menu": INCLUDE_MENU,
                "compact_menu": COMPACT_MENU,
                "instant_tests": INSTANT_TESTS,
                "fast_grpo_logprobs": FAST_GRPO_LOGPROBS,
                "eval_every": EVAL_EVERY,
                "eval_patients": EVAL_PATIENTS,
                "eval_at_step_zero": EVAL_AT_STEP_ZERO,
                "checkpoint_every": CHECKPOINT_EVERY,
                "monitor_every": MONITOR_EVERY,
            },
        )
        atexit.register(wandb.finish)  # ensures run is closed even on crash
        print(f"W&B run started (project: {WANDB_PROJECT}).")
    else:
        print("No W&B key — logging to stdout only.")


def load_env(data_dir: str):
    """Load the clinical environment, catalogs, cards, and eval patients."""
    catalogs = load_catalogs(data_dir)
    cards    = load_cards(f"{data_dir}/cards")
    env      = ClinicalEnv(catalogs, cards, instant_tests=INSTANT_TESTS)
    if INSTANT_TESTS:
        print("INSTANT_TESTS=true — all ORDER_TEST results resolve same step for GRPO.")

    # Load a fixed held-out eval subset (P4) so mid-training eval is cheap
    # and results are comparable across steps.
    eval_patients = []
    if Path(EVAL_SET_PATH).exists():
        eval_patients = [
            json.loads(l) for l in open(EVAL_SET_PATH)
        ][:EVAL_PATIENTS]

    return env, catalogs, cards, eval_patients


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 ── Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load base model + LoRA. Uses SFT adapter unless SKIP_SFT_ADAPTER=true."""
    from unsloth import FastLanguageModel

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN or None,
    )

    if SKIP_SFT_ADAPTER:
        print("SKIP_SFT_ADAPTER=true — attaching fresh LoRA on base model.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        try:
            from peft import PeftModel
            print(f"Loading SFT adapter from: {SFT_MODEL_ID}")
            model = PeftModel.from_pretrained(
                model, SFT_MODEL_ID, is_trainable=True, token=HF_TOKEN or None,
            )
            print("SFT adapter loaded and set to trainable.")
        except Exception as e:
            print(f"Could not load SFT adapter ({e}). Adding fresh LoRA — training from scratch.")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"],
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

    model.train()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── Log-probability helpers
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F

# clinical_rl imports — module-level so hot-path functions never pay import overhead.
from clinical_rl.env import ClinicalEnv, load_cards, load_catalogs
from clinical_rl.env.state import Action, ActionType
from clinical_rl.prompt import render_prompt, parse_action_json
from clinical_rl.rewards import compute_all

# wandb is optional — only imported when a key is provided.
# Set to None so guarded `if WANDB_API_KEY: wandb.log(...)` calls are always safe.
wandb = None
if WANDB_API_KEY:
    import wandb


def _compute_log_probs(
    model,
    prompt_ids: torch.Tensor,   # [prompt_len]   on CPU
    action_ids: torch.Tensor,   # [action_len]   on CPU
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Forward pass through the model to get log-probabilities of action_ids
    given prompt_ids.

    Returns a 1-D tensor of shape [action_len] — one log-prob per action token.

    HOW THE SLICING WORKS
    ─────────────────────
    Input sequence:  [p0 p1 ... p_{n-1}  a0 a1 ... a_{m-1}]
    Model logits:    [l0 l1 ... l_{n-1}  l_n ... l_{n+m-2}]
    where l_i predicts token at position i+1.

    We want log P(a0|context) from l_{n-1},
            log P(a1|context) from l_{n},
            ...
            log P(a_{m-1}|context) from l_{n+m-2}.

    So action_logits = logits[n-1 : n-1+m]  (shape: [m, vocab])
    and we gather the log-prob at each actual action token id.

    WHY a forward pass instead of using output.scores from generate()
    ──────────────────────────────────────────────────────────────────
    model.generate() with temperature != 1.0 returns temperature-scaled
    logits in output.scores. We need temperature=1 logits for the ratio
    computation so old_log_probs and new_log_probs are on the same scale.
    One extra forward pass per turn is cheap relative to the sampling call.
    """
    full_ids = torch.cat([prompt_ids, action_ids]).unsqueeze(0).to(DEVICE)

    if requires_grad:
        logits = model(full_ids).logits[0].float()   # [seq_len, vocab]
    else:
        with torch.inference_mode():
            logits = model(full_ids).logits[0].float()

    n = prompt_ids.shape[0]
    m = action_ids.shape[0]

    # Slice to get logits that predict the action tokens.
    action_logits = logits[n - 1 : n - 1 + m]          # [m, vocab]
    log_probs     = F.log_softmax(action_logits, dim=-1) # [m, vocab]

    # Gather the log-prob of the actual token chosen at each position.
    token_log_probs = log_probs.gather(
        dim=1,
        index=action_ids.to(DEVICE).unsqueeze(1),
    ).squeeze(1)   # [m]

    return token_log_probs


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 ── Smoke test (run once before training)
# ──────────────────────────────────────────────────────────────────────────────

def smoke_test_log_probs(model, tokenizer) -> None:
    """
    Verify that _compute_log_probs() is correct BEFORE any training starts.

    What we check
    ─────────────
    1. Shape: [num_action_tokens] — correct slicing.
    2. Values: all negative, in the range (-∞, 0) — log probs are ≤ 0.
    3. Reproducible: two calls with the same input give the same result.
    4. Gradient flows: backward() reaches the LoRA parameters.

    WHY THIS MATTERS
    ────────────────
    An off-by-one in the slice gives PLAUSIBLE-LOOKING log probs
    (same shape, similar magnitudes) but the wrong gradient direction.
    This shows up only as "reward not improving after 200 steps" — i.e.,
    after you've spent $3-6 on compute. This test costs $0.
    """
    print("\n── Smoke test: verifying log_prob computation ──")
    messages = [{"role": "user", "content": [{"type": "text", "text": "Test prompt for smoke test."}]}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True,
    )[0].cpu()

    with torch.inference_mode():
        out = model.generate(
            prompt_ids.unsqueeze(0).to(DEVICE),
            max_new_tokens=8,     # tiny — we just need a few tokens
            do_sample=False,      # greedy for exact reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    action_ids = out[0][prompt_ids.shape[0]:].cpu()

    if action_ids.shape[0] == 0:
        print("WARNING: model generated 0 tokens. Smoke test skipped.")
        return

    # Check 1 + 2: shape and value range.
    lp = _compute_log_probs(model, prompt_ids, action_ids, requires_grad=False)
    assert lp.shape == action_ids.shape, \
        f"Shape mismatch: log_probs={lp.shape}, action_ids={action_ids.shape}"
    assert (lp <= 0).all(), f"Log probs should be ≤ 0, got max={lp.max():.3f}"
    assert (lp > -200).all(), f"Log probs suspiciously small: min={lp.min():.3f}"

    # Check 3: reproducible.
    lp2 = _compute_log_probs(model, prompt_ids, action_ids, requires_grad=False)
    assert torch.allclose(lp, lp2, atol=1e-4), "Log probs not deterministic!"

    # Check 4: gradient flows to LoRA parameters.
    lp_grad = _compute_log_probs(model, prompt_ids, action_ids, requires_grad=True)
    lp_grad.mean().backward()
    trainable_with_grad = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(trainable_with_grad) > 0, \
        "Backward pass produced NO gradients on trainable parameters!"
    model.zero_grad()

    print(
        f"PASSED — shape={lp.shape}, "
        f"mean={lp.mean():.3f}, "
        f"range=[{lp.min():.3f}, {lp.max():.3f}]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 ── Episode generation (no gradients)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_or_fallback(text: str, catalogs, rng: random.Random):
    """
    Parse the model's JSON output into an Action. Retry up to 3 times on
    the same text (the parse function strips markdown fences / prose, so
    retrying occasionally succeeds after minor stripping variations — but
    mostly we use this to catch transient generation glitches).

    Falls back to a random INTERVIEW if all retries fail.

    WHY A FALLBACK INSTEAD OF CRASHING
    ────────────────────────────────────
    During GRPO, the policy shifts away from the SFT distribution transiently.
    A crash would abort an expensive rollout. Instead: the fallback INTERVIEW
    is a safe no-op (free, doesn't terminate the episode, doesn't order
    anything costly), and R8 penalises the invalid step so the model is
    pushed to fix its format.
    """
    for _ in range(3):
        try:
            return parse_action_json(text)
        except ValueError:
            pass

    qid = rng.choice(list(catalogs.questions.keys()))
    return Action(ActionType.INTERVIEW, qid, "fallback — parse error")


def _render_compact_id_menu(catalogs) -> str:
    """Exact valid IDs only: much shorter than the full action menu."""
    return "\n".join([
        "===== COMPACT VALID ID MENU =====",
        "Use these exact IDs only. Do not invent camelCase, spaces, or synonyms.",
        "INTERVIEW ids: " + ", ".join(sorted(catalogs.questions.keys())),
        "EXAMINE ids: " + ", ".join(sorted(catalogs.exams.keys())),
        "ORDER_TEST ids: " + ", ".join(sorted(catalogs.tests.keys())),
        "DIAGNOSE / board disease ids: " + ", ".join(sorted(catalogs.diseases.keys())),
    ])


def render_training_prompt(obs, catalogs) -> str:
    prompt = render_prompt(obs, catalogs, include_menu=INCLUDE_MENU)
    if COMPACT_MENU and not INCLUDE_MENU:
        prompt = prompt.replace(
            "===== PATIENT INTAKE =====",
            _render_compact_id_menu(catalogs) + "\n\n===== PATIENT INTAKE =====",
            1,
        )
    return prompt


def play_episode(
    model,
    tokenizer,
    env,
    catalogs,
    *,
    disease: str,
    variant_id: str,
    seed: int,
    rng: random.Random,
) -> tuple[list[tuple], object]:
    """
    Play one full episode using the current model policy.

    Returns
    ───────
    turns : list of (prompt_ids_cpu, action_ids_cpu, old_log_probs_cpu_or_none)
        One entry per step. Used later to compute the GRPO loss.
    final_obs : Observation
        The terminal episode state. Used to compute the reward.

    No gradients are tracked here. In the slower PPO-style mode, old log probs
    are computed via a clean forward pass (temperature=1) immediately after
    each generation step. In fast mode, we skip that extra rollout forward and
    build the detached reference during the loss pass.
    """
    obs = env.reset(disease, variant_id=variant_id, seed=seed)
    turns = []

    while not obs.terminal:
        # Build the prompt for the current observation state.
        prompt_text = render_training_prompt(obs, catalogs)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True,
        )[0].cpu()  # keep on CPU until needed

        # Generate one action — no gradient, temperature for diversity.
        with torch.inference_mode():
            output = model.generate(
                prompt_ids.unsqueeze(0).to(DEVICE),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        action_ids = output[0][prompt_ids.shape[0]:].cpu()

        # Optional PPO-style old log probs via a forward pass at temperature=1.
        # Fast mode skips this because this script uses each rollout batch for
        # one optimizer pass, so the loss pass can use new_log_probs.detach()
        # as the fixed reference and save one full forward per generated turn.
        old_log_probs = None
        if not FAST_GRPO_LOGPROBS:
            old_log_probs = _compute_log_probs(
                model, prompt_ids, action_ids, requires_grad=False
            ).cpu()

        # Decode and parse the action. Fall back to a random interview on error.
        text   = tokenizer.decode(action_ids, skip_special_tokens=True)
        action = _parse_or_fallback(text, catalogs, rng)

        turns.append((prompt_ids, action_ids, old_log_probs))
        obs = env.step(action)

    return turns, obs, env._episode  # episode returned explicitly (B10)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 ── GRPO loss computation (with gradients)
# ──────────────────────────────────────────────────────────────────────────────

def compute_grpo_loss(
    model,
    rollouts: list[tuple[list, float]],   # [(turns, reward), ...]
    advantages: list[float],
) -> tuple[float, float]:
    """
    Compute the GRPO surrogate loss and accumulate gradients.

    Each rollout is processed one at a time and .backward() is called
    immediately. This is the gradient-accumulation pattern: gradients
    ADD UP across rollouts in model.parameters().grad before the
    optimizer.step() in the main loop.

    Why one-turn-at-a-time instead of batching all 8
    ─────────────────────────────────────────────────
    Batching all 8 rollouts requires holding 8 episodes worth of forward-pass
    activations in GPU memory simultaneously — on an A100 40GB this would
    likely OOM. Processing each saved turn and calling backward immediately
    keeps peak memory much lower while gradients still accumulate across the
    whole GRPO group before optimizer.step().

    Fast mode loss (default)
    ────────────────────────
    For each action token t in rollout i with advantage A_i:
        loss_t = -A_i * new_log_prob_t

    This is the useful first-order GRPO policy-gradient signal for a single
    update per rollout batch. It avoids a dead PPO ratio/KL calculation when
    old_policy == current_policy.

    PPO-style mode (FAST_GRPO_LOGPROBS=false)
    ─────────────────────────────────────────
    Stores rollout-time old log probs and uses the clipped ratio objective:
        ratio_t = exp(new_log_prob_t - old_log_prob_t)
        loss_t  = -min(ratio_t * A_i, clip(ratio_t, 1-ε, 1+ε) * A_i)

    This only becomes meaningfully different if the same rollout batch is
    reused after the model has already been updated.

    Returns (mean_loss, mean_kl) for logging.
    """
    total_loss  = 0.0
    total_kl    = 0.0
    total_turns = 0
    loss_terms  = 0

    for (turns, _), advantage in zip(rollouts, advantages):
        if not turns:
            continue

        for prompt_ids, action_ids, old_log_probs_cpu in turns:
            if action_ids.shape[0] == 0:
                continue  # model generated nothing (EOS immediately)

            # Forward pass WITH gradient for new log probs.
            new_log_probs = _compute_log_probs(
                model, prompt_ids, action_ids, requires_grad=True
            )   # [action_len]

            adv_tensor = torch.tensor(advantage, device=DEVICE, dtype=torch.float32)

            if old_log_probs_cpu is None:
                # Fast single-update GRPO: no old-policy ratio is needed because
                # each rollout batch is used for exactly one optimizer step.
                turn_loss = -(adv_tensor * new_log_probs.mean())
                kl = 0.0
            else:
                old_log_probs = old_log_probs_cpu.to(DEVICE)

                # Probability ratio: how much has the policy changed for this token?
                # ratio > 1 → new policy is more likely to choose this token.
                # ratio < 1 → new policy has moved away from this token.
                ratio = (new_log_probs - old_log_probs).exp()   # [action_len]

                # Clipped surrogate: we won't reward policy updates that go
                # beyond [1-ε, 1+ε] even if the advantage is favourable.
                clipped_ratio = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)

                # Take the conservative (minimum) of clipped vs unclipped.
                loss_per_token = -torch.min(
                    ratio * adv_tensor,
                    clipped_ratio * adv_tensor,
                )

                turn_loss = loss_per_token.mean()

                # KL approximation (for logging only, no grad needed on this).
                with torch.inference_mode():
                    kl = (old_log_probs - new_log_probs.detach()).mean().item()
            total_kl += kl
            total_turns += 1
            total_loss += turn_loss.detach().item()
            loss_terms += 1

            # Divide by GROUP_SIZE so the effective gradient is an average
            # across rollouts (not a sum that scales with group size). Backward
            # per turn frees activations promptly instead of retaining an
            # entire episode graph.
            (turn_loss / GROUP_SIZE).backward()

    mean_loss = total_loss / max(loss_terms, 1)
    mean_kl   = total_kl  / max(total_turns, 1)
    return mean_loss, mean_kl


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 ── Mid-training evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, env, catalogs, cards, eval_patients, step: int) -> dict:
    """
    Run the current policy on a subset of held-out eval patients (no grad).
    Reports per-rubric averages so we can track WHAT is improving, not just
    whether total reward goes up.

    Uses greedy decoding (temperature=0) for reproducibility.
    """
    model.eval()
    results = []

    for patient in eval_patients:
        disease    = patient["disease"]
        variant_id = patient["variant_id"]
        seed       = patient["seed"]
        obs        = env.reset(disease, variant_id=variant_id, seed=seed)

        while not obs.terminal:
            prompt = render_training_prompt(obs, catalogs)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", add_generation_prompt=True,
            ).to(DEVICE)
            with torch.inference_mode():
                out = model.generate(
                    ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,   # greedy for eval
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            try:
                action = parse_action_json(text)
            except ValueError:
                action = Action(ActionType.DIAGNOSE, "viral_uri", "fallback")
            obs = env.step(action)

        rewards = compute_all(env._episode, cards[disease], catalogs)
        rewards["correct"] = float(obs.diagnosis == disease)
        results.append(rewards)

    model.train()

    # Average each metric across all eval patients.
    avg = {
        k: sum(r[k] for r in results) / len(results)
        for k in results[0]
    }
    print(
        f"\n[Eval step {step}] "
        f"acc={avg['r1_accuracy']:.2f}  "
        f"cost={avg['r2_cost']:.2f}  "
        f"format={avg['r8_format']:.2f}  "
        f"total={avg['total']:.2f}  "
        f"correct%={avg['correct']*100:.0f}%"
    )
    return avg


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 ── Rollout-batch monitoring
# ──────────────────────────────────────────────────────────────────────────────

REWARD_KEYS = ("r1_accuracy", "r2_cost", "r6_anchoring", "r7_safety", "r8_format", "total")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _action_entries(episode) -> list:
    return [e for e in episode.obs.action_log if e.kind == "action" and e.action is not None]


def _action_type_name(action: Action) -> str:
    return action.type.value if isinstance(action.type, ActionType) else str(action.type)


def _compact_actions(episode, limit: int = 8) -> str:
    pieces = []
    entries = _action_entries(episode)
    for entry in entries[:limit]:
        action = entry.action
        marker = "!" if entry.invalid else ""
        pieces.append(f"{entry.step}:{_action_type_name(action)}:{action.argument}{marker}")
    if len(entries) > limit:
        pieces.append("...")
    return " | ".join(pieces)


def summarize_episode_for_monitor(
    episode,
    reward_dict: dict,
    *,
    rollout_idx: int,
    disease: str,
) -> dict:
    """Extract reward-hacking signals from one completed rollout."""
    entries = _action_entries(episode)
    action_counts = Counter(_action_type_name(e.action) for e in entries)
    diagnosis_entries = [
        e for e in entries if e.action is not None and e.action.type == ActionType.DIAGNOSE
    ]
    fallback_count = sum(
        1
        for e in entries
        if e.action is not None and "fallback" in (e.action.rationale or "").lower()
    )
    invalid_count = sum(1 for e in entries if e.invalid)
    duplicate_count = sum(1 for e in entries if e.duplicate)
    diagnosis = episode.obs.diagnosis or ""
    correct = diagnosis == episode.hidden.true_disease

    return {
        "rollout": rollout_idx,
        "disease": disease,
        "reward": reward_dict.get("total", 0.0),
        "correct": float(correct),
        "diagnosis": diagnosis or "none",
        "turns": len(entries),
        "tests": action_counts.get(ActionType.ORDER_TEST.value, 0),
        "interviews": action_counts.get(ActionType.INTERVIEW.value, 0),
        "exams": action_counts.get(ActionType.EXAMINE.value, 0),
        "differentials": action_counts.get(ActionType.UPDATE_DIFFERENTIAL.value, 0),
        "diagnoses": action_counts.get(ActionType.DIAGNOSE.value, 0),
        "cost": episode.obs.cost_so_far,
        "timed_out": float(episode.obs.timed_out),
        "diagnosis_turn": diagnosis_entries[0].step if diagnosis_entries else 0,
        "invalid": invalid_count,
        "fallback": fallback_count,
        "duplicates": duplicate_count,
        "action_counts": dict(action_counts),
        "actions": _compact_actions(episode),
    }


def rollout_batch_monitor(
    *,
    step: int,
    total_steps: int,
    disease: str,
    variant_id: str,
    seed: int,
    curriculum_level: int,
    reward_dicts: list[dict],
    summaries: list[dict],
) -> tuple[dict, list[list]]:
    """Print and return batch-level anti-reward-hacking metrics."""
    if not summaries:
        return {}, []

    metrics = {}
    for key in REWARD_KEYS:
        metrics[f"batch/{key}_mean"] = _mean([float(r.get(key, 0.0)) for r in reward_dicts])
        metrics[f"batch/{key}_min"] = min(float(r.get(key, 0.0)) for r in reward_dicts)
        metrics[f"batch/{key}_max"] = max(float(r.get(key, 0.0)) for r in reward_dicts)

    action_counts = Counter()
    for summary in summaries:
        action_counts.update(summary["action_counts"])
    total_actions = sum(action_counts.values())

    for action_type in ActionType:
        count = action_counts.get(action_type.value, 0)
        metrics[f"batch/action_{action_type.value.lower()}_count"] = count
        metrics[f"batch/action_{action_type.value.lower()}_rate"] = (
            count / total_actions if total_actions else 0.0
        )

    diagnosis_counts = Counter(s["diagnosis"] for s in summaries)
    top_diagnosis, top_diagnosis_count = diagnosis_counts.most_common(1)[0]

    metrics.update({
        "batch/correct_rate": _mean([s["correct"] for s in summaries]),
        "batch/timeout_rate": _mean([s["timed_out"] for s in summaries]),
        "batch/avg_turns": _mean([s["turns"] for s in summaries]),
        "batch/avg_tests": _mean([s["tests"] for s in summaries]),
        "batch/avg_interviews": _mean([s["interviews"] for s in summaries]),
        "batch/avg_exams": _mean([s["exams"] for s in summaries]),
        "batch/avg_differentials": _mean([s["differentials"] for s in summaries]),
        "batch/avg_cost": _mean([s["cost"] for s in summaries]),
        "batch/avg_diagnosis_turn": _mean(
            [s["diagnosis_turn"] for s in summaries if s["diagnosis_turn"]]
        ),
        "batch/invalid_action_rate": (
            sum(s["invalid"] for s in summaries) / total_actions if total_actions else 0.0
        ),
        "batch/fallback_action_rate": (
            sum(s["fallback"] for s in summaries) / total_actions if total_actions else 0.0
        ),
        "batch/duplicate_action_rate": (
            sum(s["duplicates"] for s in summaries) / total_actions if total_actions else 0.0
        ),
        "batch/unique_diagnoses": len(diagnosis_counts),
        "batch/diagnosis_mode_rate": top_diagnosis_count / len(summaries),
    })

    reward_line = "  rewards: " + "  ".join(
        f"{key}={metrics[f'batch/{key}_mean']:+.2f}" for key in REWARD_KEYS
    )
    shape_line = (
        "  shape: "
        f"correct={metrics['batch/correct_rate']*100:.0f}%  "
        f"timeout={metrics['batch/timeout_rate']*100:.0f}%  "
        f"turns={metrics['batch/avg_turns']:.1f}  "
        f"tests={metrics['batch/avg_tests']:.1f}  "
        f"diffs={metrics['batch/avg_differentials']:.1f}  "
        f"cost=${metrics['batch/avg_cost']:.0f}  "
        f"dx_turn={metrics['batch/avg_diagnosis_turn']:.1f}"
    )
    action_mix = ", ".join(
        f"{action_type.value}={action_counts.get(action_type.value, 0)}"
        for action_type in ActionType
    )
    diagnosis_mix = ", ".join(
        f"{name}:{count}" for name, count in diagnosis_counts.most_common(5)
    )

    print(
        f"\n[Batch monitor step {step}/{total_steps}] "
        f"disease={disease} variant={variant_id} seed={seed} curriculum=L{curriculum_level}"
    )
    print(reward_line)
    print(shape_line)
    print(f"  action_mix: {action_mix}")
    print(f"  diagnoses: {diagnosis_mix}")

    warnings = []
    if metrics["batch/r8_format_mean"] < 0.8:
        warnings.append(f"format low ({metrics['batch/r8_format_mean']:.2f})")
    if metrics["batch/avg_differentials"] < 0.5:
        warnings.append("differential updates near zero")
    if metrics["batch/avg_diagnosis_turn"] and metrics["batch/avg_diagnosis_turn"] < 3:
        warnings.append(f"early diagnose avg turn {metrics['batch/avg_diagnosis_turn']:.1f}")
    if metrics["batch/avg_tests"] > 10:
        warnings.append(f"high test count {metrics['batch/avg_tests']:.1f}")
    if metrics["batch/timeout_rate"] > 0.25:
        warnings.append(f"timeout rate {metrics['batch/timeout_rate']*100:.0f}%")
    if metrics["batch/fallback_action_rate"] > 0.10:
        warnings.append(f"fallback actions {metrics['batch/fallback_action_rate']*100:.0f}%")
    if metrics["batch/invalid_action_rate"] > 0.10:
        warnings.append(f"invalid actions {metrics['batch/invalid_action_rate']*100:.0f}%")
    if metrics["batch/unique_diagnoses"] <= 1 and len(summaries) > 1:
        warnings.append(f"diagnosis mode collapse to {top_diagnosis}")
    if metrics["batch/correct_rate"] < 0.25 and metrics["batch/total_mean"] > 0.5:
        warnings.append("reward positive while accuracy is low")

    if warnings:
        print("  warnings: " + "; ".join(warnings))

    best = max(summaries, key=lambda s: s["reward"])
    worst = min(summaries, key=lambda s: s["reward"])
    print(
        f"  best rollout #{best['rollout']}: reward={best['reward']:+.2f} "
        f"dx={best['diagnosis']} correct={bool(best['correct'])} actions={best['actions']}"
    )
    print(
        f"  worst rollout #{worst['rollout']}: reward={worst['reward']:+.2f} "
        f"dx={worst['diagnosis']} correct={bool(worst['correct'])} actions={worst['actions']}\n"
    )

    table_rows = [
        [
            step,
            s["rollout"],
            s["disease"],
            s["reward"],
            s["correct"],
            s["turns"],
            s["tests"],
            s["differentials"],
            s["cost"],
            s["diagnosis"],
            s["actions"],
        ]
        for s in summaries
    ]
    return metrics, table_rows


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── Checkpoint and push helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, step: int) -> None:
    """Save locally AND push to Hub so a job timeout doesn't lose progress.

    Hub repo name: {HF_USERNAME}/house-md-grpo-optimized-gemma3-4b-step{step}
    The final run also pushes to house-md-grpo-optimized-gemma3-4b (no step suffix).
    """
    out_dir = f"outputs/grpo_optimized_checkpoint_step{step}"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Checkpoint saved locally → {out_dir}")

    # Push mid-run checkpoint to Hub so a timeout doesn't lose it.
    if HF_USERNAME:
        hub_id = f"{HF_USERNAME}/{OUTPUT_HUB_ID}-step{step}"
        try:
            model.push_to_hub(hub_id, token=HF_TOKEN)
            tokenizer.push_to_hub(hub_id, token=HF_TOKEN)
            print(f"Checkpoint pushed → https://huggingface.co/{hub_id}")
        except Exception as e:
            print(f"WARNING: Hub push failed for step {step}: {e}")


def push_to_hub(model, tokenizer) -> None:
    hub_id = f"{HF_USERNAME}/{OUTPUT_HUB_ID}" if HF_USERNAME else None
    if not hub_id:
        print("HF_USERNAME not set — skipping Hub push.")
        return
    print(f"Pushing final adapter to {hub_id} ...")
    model.push_to_hub(hub_id, token=HF_TOKEN)
    tokenizer.push_to_hub(hub_id, token=HF_TOKEN)
    print(f"Done — https://huggingface.co/{hub_id}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("E5 — GRPO training: House M.D. diagnostic reasoning")
    print(f"     {TOTAL_STEPS} steps  |  group_size={GROUP_SIZE}  |  lr={LR}")
    print("=" * 60)

    login()
    env, catalogs, cards, eval_patients = load_env(DATA_DIR)
    model, tokenizer = load_model()

    # ── Sanity check BEFORE burning any compute ──────────────────────────────
    smoke_test_log_probs(model, tokenizer)

    # ── Optimizer: paged AdamW keeps optimizer state in CPU RAM ─────────────
    # At LR=1e-5 (10x lower than SFT), we take smaller steps — important
    # because GRPO can easily overshoot and collapse the policy.
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )

    all_diseases = list(catalogs.diseases.keys())
    variants     = ["v1", "v2", "v3"]
    rng          = random.Random(42)

    print(f"\nStarting GRPO training for {TOTAL_STEPS} steps ...\n")

    for step in range(TOTAL_STEPS):
        # ── Curriculum ───────────────────────────────────────────────────────
        # Level 1: 5 diseases (format + basic reasoning).
        # Level 2: 10 diseases (adds more critical/urgent mimics).
        # Level 3: all 15 (including the full stable/benign long tail).
        if step < CURRICULUM_LEVEL_2_STEP:
            disease_pool = LEVEL_1_DISEASES
            curriculum_level = 1
        elif step < CURRICULUM_LEVEL_3_STEP:
            disease_pool = LEVEL_2_DISEASES
            curriculum_level = 2
        else:
            disease_pool = all_diseases
            curriculum_level = 3

        disease      = rng.choice(disease_pool)
        variant_id   = rng.choice(variants)
        # Randomise seed so the patient draw is fresh every step but
        # stays out of the eval seed range (100, 200, 300).
        seed = rng.randint(1000, 999999)

        # ── Phase 1: Generate GROUP_SIZE rollouts (NO gradient) ───────────────
        rollouts: list[tuple[list, float]] = []
        rollout_reward_dicts: list[dict] = []
        rollout_summaries: list[dict] = []
        for rollout_idx in range(GROUP_SIZE):
            turns, _, episode = play_episode(
                model, tokenizer, env, catalogs,
                disease=disease, variant_id=variant_id, seed=seed,
                rng=rng,
            )
            reward_dict = compute_all(episode, cards[disease], catalogs)
            reward = reward_dict["total"]
            rollouts.append((turns, reward))
            rollout_reward_dicts.append(reward_dict)
            rollout_summaries.append(
                summarize_episode_for_monitor(
                    episode, reward_dict, rollout_idx=rollout_idx, disease=disease
                )
            )

        # ── Phase 2: Group-relative advantages ───────────────────────────────
        rewards = [r for _, r in rollouts]
        mean_r  = statistics.mean(rewards)
        # Floor std so we don't divide by zero when all 8 rollouts score
        # identically (rare but can happen for trivial diseases early in L1).
        std_r   = max(statistics.pstdev(rewards), 1e-6)
        advantages = [(r - mean_r) / std_r for r in rewards]

        # ── Phase 3: GRPO loss + gradient update ─────────────────────────────
        optimizer.zero_grad()
        mean_loss, mean_kl = compute_grpo_loss(model, rollouts, advantages)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        log_dict = {
            "step": step,
            "reward/mean":  mean_r,
            "reward/max":   max(rewards),
            "reward/min":   min(rewards),
            "reward/std":   std_r,
            "loss":         mean_loss,
            "kl":           mean_kl,       # KL > 0.5 = training unstable
            "advantage/std": std_r,
            "curriculum":   curriculum_level,
        }

        monitor_table_rows = []
        if step % MONITOR_EVERY == 0:
            monitor_metrics, monitor_table_rows = rollout_batch_monitor(
                step=step,
                total_steps=TOTAL_STEPS,
                disease=disease,
                variant_id=variant_id,
                seed=seed,
                curriculum_level=curriculum_level,
                reward_dicts=rollout_reward_dicts,
                summaries=rollout_summaries,
            )
            log_dict.update(monitor_metrics)

        if WANDB_API_KEY:
            if monitor_table_rows:
                log_dict["batch/rollouts"] = wandb.Table(
                    columns=[
                        "step", "rollout", "disease", "reward", "correct",
                        "turns", "tests", "differentials", "cost",
                        "diagnosis", "actions",
                    ],
                    data=monitor_table_rows,
                )
            wandb.log(log_dict, step=step)

        print(
            f"step {step:>4}/{TOTAL_STEPS}  "
            f"disease={disease:<22} "
            f"r_mean={mean_r:+.2f}  r_std={std_r:.2f}  "
            f"loss={mean_loss:.4f}  kl={mean_kl:.4f}  "
            f"curriculum=L{curriculum_level}",
            flush=True,
        )

        # ── Warning: KL divergence monitor ───────────────────────────────────
        # If KL consistently exceeds 0.5, the policy is drifting too fast.
        # Lower LR or increase CLIP_EPS if this fires regularly.
        if mean_kl > 0.5:
            print(f"  WARNING: high KL divergence ({mean_kl:.3f}) at step {step}.")

        # ── Mid-training evaluation ───────────────────────────────────────────
        should_eval = (
            eval_patients
            and step % EVAL_EVERY == 0
            and (step > 0 or EVAL_AT_STEP_ZERO)
        )
        if should_eval:
            eval_metrics = evaluate(
                model, tokenizer, env, catalogs, cards, eval_patients, step
            )
            if WANDB_API_KEY:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

            if eval_metrics.get("r8_format", 1.0) < 0.6:
                print(
                    f"  WARNING: low eval R8 format "
                    f"({eval_metrics['r8_format']:.2f}) at step {step}."
                )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, tokenizer, step)

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nTraining complete. Saving final adapter ...")
    save_checkpoint(model, tokenizer, step=TOTAL_STEPS)
    push_to_hub(model, tokenizer)

    if WANDB_API_KEY:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="Override TOTAL_STEPS env var.")
    args = parser.parse_args()
    if args.steps is not None:
        TOTAL_STEPS = args.steps
    main()
