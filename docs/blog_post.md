---
title: "Teaching Gemma 3 to Diagnose Patients with GRPO and OpenEnv"
thumbnail: /blog/assets/house-md/thumbnail.png
authors:
  - user: SnehShah
tags:
  - rl
  - grpo
  - openenv
  - gemma
  - unsloth
  - trl
  - medical
---

# Teaching Gemma 3 to Diagnose Patients with GRPO and OpenEnv

> Submitted to the **Apr '26 OpenEnv Hackathon**. All code, environment, and adapters are public.
>
> - **Live environment**: [`SnehShah/house-md-env`](https://huggingface.co/spaces/SnehShah/house-md-env)
> - **Final adapter**:    [`SnehShah/house-md-grpo-optimized-gemma3-4b-v3`](https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3)
> - **W&B run**:          [`sneh2909-christ-university/house-md`](https://wandb.ai/sneh2909-christ-university/house-md?nw=nwusersneh2909)
> - **Repo + notebooks**: <https://github.com/SnehShah/house-md-env>

---

## TL;DR

We built a partial-information **Emergency Department simulator** as an OpenEnv environment, deployed it as a public Hugging Face Space (with a custom animated ER UI you can play yourself), and used **GRPO** to teach Gemma 3 4B-IT to diagnose 35 acute conditions while balancing cost, time, and patient safety.

The base model gets **17.8 %** of held-out patients right. After SFT + GRPO it learns to *interview, examine, order targeted tests, and only then diagnose* — the same loop a clinician runs in their head. Every reward, every malformed JSON, every "skip the differential" exploit was visible in W&B and shaped the next iteration.

This post walks through:

1. Why diagnosis is a sequential decision problem, not a classification problem.
2. How we wrapped a 1500-line Python ED simulator behind the OpenEnv contract and got it running on a Hugging Face Space in one command.
3. The reward function and the **specific failure modes** GRPO converged toward — and what we did to break them.
4. Real numbers from the production training run.
5. How to reproduce every step in four Colab notebooks (one of them needs no GPU).

<figure>
  <img src="https://snehshah-house-md-env.hf.space/static/screenshot.png" alt="The custom ER triage UI served by the OpenEnv Space" />
  <figcaption>The same Space serves the standard OpenEnv API and a clinician-facing ER triage UI you can poke through manually.</figcaption>
</figure>

---

## 1. Why diagnosis is sequential

A classification model sees `(symptoms, demographics) → disease`. That isn't what doctors do. They see *one vague complaint* — "12 hours of worsening RLQ pain" — and choose the next move from a discrete menu of 110+ options:

- **25 questions** they can ask the patient (pain history, PMH, family history…)
- **15 physical exam** maneuvers (vital signs, McBurney point, Murphy sign…)
- **35 lab / imaging tests**, each with a **dollar cost and turnaround latency**
- **Update the differential board** — write down their best 3-5 guesses with probabilities
- **DIAGNOSE** — terminal action; ends the episode

Every one of those actions has a cost. Every minute the patient is undiagnosed they may deteriorate. The right policy isn't "order the most expensive imaging test up front" — it's "ask two cheap questions to compress the differential, then pick the *one* test that disambiguates the top two suspects."

That's a **POMDP**, not a classifier. And it's exactly what GRPO is good at — taking a base LLM that already knows what an `INTERVIEW` is and rewarding it for sequencing those actions efficiently against a real environment.

---

## 2. The environment

`clinical_rl/` is a 1,500-line Python simulator with:

- 35 disease cards, each with `min_evidence_set` and `recommended_workup` lists
- A noisy patient response generator (questions and exam findings have configurable error rates)
- A test scheduler (some tests deliver immediately, some take 1-3 steps; CT abdomen returns at step+2)
- Five reward rubrics (more on those in a moment)
- A heuristic oracle for dataset generation and an upper bound

For the hackathon we wrapped it behind the OpenEnv contract. The repo is laid
out so `openenv push` from the root publishes the Space directly:

```text
house-md-env/
├── house_md_env/              # the OpenEnv package
│   ├── __init__.py            # exports HouseMDEnv (client) + HouseMDAction/Observation
│   ├── client.py              # OpenEnv-style Pydantic client
│   ├── models.py              # Action / Observation / State as Pydantic
│   ├── server/
│   │   ├── app.py             # FastAPI: OpenEnv /reset/step/state + custom UI + /api/*
│   │   ├── house_md_environment.py  # openenv.core.Environment subclass
│   │   ├── playground.py      # session-based /api routes for the UI
│   │   └── static/            # the animated ER scene
│   ├── clinical_rl/           # vendored core simulator (so the Space is self-contained)
│   └── data/                  # 35 disease cards + catalogs
├── notebooks/                 # 4 Colab notebooks (excluded from the Space image)
├── docs/                      # this blog post + grpo dry-run
├── openenv.yaml
├── pyproject.toml
└── Dockerfile
```

Pushing it to a Space is one command:

```bash
openenv push --repo-id SnehShah/house-md-env
```

The Space serves three things on the same uvicorn process:

| Path | Purpose |
| --- | --- |
| `POST /reset`, `POST /step`, `GET /state` | Standard stateless OpenEnv contract — what the training loop hits. |
| `POST /api/episodes`, `POST /api/episodes/<sid>/agent_step?policy=oracle`, … | **Stateful** session API used by the UI; each browser tab gets a long-lived `ClinicalEnv` instance. |
| `GET /` | The animated ER UI ([screenshot above](#tldr)). |
| `GET /docs` | FastAPI's OpenAPI explorer. |

The `02`/`03` notebooks talk to **`POST /api/episodes`** during training so the environment state survives the model's multi-turn rollout, and the production `eval_hf.py` boots the same `app.py` inside the eval container to score adapters end-to-end.

---

## 3. Reward design (and the exploits we patched)

Five rubrics, each a separate scalar, weighted into a single total:

| Key | Range | Meaning | Default weight |
| --- | --- | --- | --- |
| `r1_accuracy` | `[-2, +1]` | Did the diagnosis match the ground truth + did we satisfy the disease's `min_evidence_set`? | 2.0 |
| `r2_cost` | `[-1.5, +1]` | Cost penalty vs the disease's oracle ceiling. Zero-test guess gets +1, $400+ workup gets <0. | 0.5 |
| `r6_anchoring` | `[-0.5, +0.6]` | Did we update the differential board at least once with meaningful probabilities? | 0.3 |
| `r7_safety` | `[-2, 0]` | Big negative if a critical condition was missed and the patient deteriorated. | 1.0 |
| `r8_format` | `[0, 1]` | Fraction of actions that were well-formed JSON. | 0.5 |

We left several rubrics deliberately *low-weighted* and watched what happened:

- **"Skip UPDATE_DIFFERENTIAL"** — `r6_anchoring=0.3` is small enough that an agent could just skip it and lose ~0.09 reward. Without curriculum it converged here within 30 steps. Mitigation: we add `r6` to the Level-1 curriculum focus list and bump its weight during the first third of training.
- **"Diagnose immediately"** — guess `viral_uri` for everyone, get +1 R8 (format), -2 R1 (wrong dx) for a total of around -1.0. With `r2_cost=+1` (no tests) and the model's risk-aversion bias from SFT this looked tempting on early curriculum patients. Caught by checking the histogram of `step` at episode end.
- **"Order everything"** — there are 35 tests and a hard 15-step cap, so ordering them all is impossible, but ordering 14 of them and timing out is a real attractor. R7 (safety) catches it: timing out on a critical patient costs -2, dwarfing the tiny R2 advantage of hedging.
- **Format degradation** — if R8 dropped below 0.8 across rollouts, the LR was too high. We saw this once at 1e-4 and dropped to 1e-5 for the production run.

The full reward code lives in [`clinical_rl/rewards.py`](https://github.com/SnehShah/house-md-env/blob/main/clinical_rl/rewards.py); the design notes are alongside the dry-run at [`docs/grpo_dry_run.md`](https://github.com/SnehShah/house-md-env/blob/main/docs/grpo_dry_run.md).

---

## 4. Training: SFT then GRPO

### SFT warm-start

`scripts/train_sft.py` — Unsloth + TRL on **2,151 oracle traces** (1 epoch, LoRA-r16, ~45 min on an L4):

```python
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained("unsloth/gemma-3-4b-it", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, ...)

trainer = SFTTrainer(model, tokenizer=tok, train_dataset=ds, args=SFTConfig(
    learning_rate=2e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4,
    optim="adamw_8bit", bf16=True, max_seq_length=4096, ...,
))
trainer = train_on_responses_only(trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n")
trainer.train()
```

**Key bit**: `train_on_responses_only` masks the prompt tokens. SFT is *only* training the model to emit JSON — the patient history, the menu, the current state, all of that is masked. After 1 epoch we go from "16 % of completions parse" to "**99 % of completions parse**", but the model's diagnostic *accuracy* is still bad. That's exactly the right shape for GRPO to take over.

Adapter: [`SnehShah/house-md-sft-gemma3-4b`](https://huggingface.co/SnehShah/house-md-sft-gemma3-4b)

### GRPO — the actual reasoning lesson

`scripts/train_grpo_optimized.py` — 100 steps × 8 rollouts per group on Level-1 → Level-2 → full curriculum. The full dry-run-with-tensors-and-shapes is at [`docs/grpo_dry_run.md`](https://github.com/SnehShah/house-md-env/blob/main/docs/grpo_dry_run.md), but the loop is just:

```python
for step in range(TOTAL_STEPS):
    # 1) sample one patient and roll out the policy GROUP_SIZE times
    disease, variant, seed = curriculum.sample(step)
    rollouts = [play_episode(model, env, disease, variant, seed) for _ in range(GROUP_SIZE)]

    # 2) score each rollout with the same compute_all() that production uses
    rewards = [compute_all(r.episode, cards, catalogs)["total"] for r in rollouts]

    # 3) group-relative advantages
    adv = (torch.tensor(rewards) - mean(rewards)) / (std(rewards) + 1e-8)

    # 4) PPO-style ratio loss (eps=0.2) summed over all turns of all rollouts
    loss = grpo_loss(rollouts, adv, ratio_clip=0.2)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
```

There are two non-obvious wins inside this loop:

1. **No menu in rollout prompts** (`include_menu=False`). SFT taught the model the closed vocabulary; carrying the 2 000-token menu in every rollout would 4× the forward pass cost. We rely on **constrained decoding** (`prompt.build_action_schema(catalogs)`) to keep the JSON valid token-by-token.
2. **Old log-probs cached at rollout time** (`FAST_GRPO_LOGPROBS=true`). The standard PPO ratio needs `π_old(a|s)`. If we re-forward the prompt under the *current* policy when computing the ratio, that's twice the GPU work. Instead we save the chosen-token log-probs during sampling and compute the ratio against `π_θ` only at gradient time.

The full hyperparameters are encoded at the top of [`train_grpo_optimized.py`](https://github.com/SnehShah/house-md-env/blob/main/scripts/train_grpo_optimized.py) and visible in the W&B config:

```text
lr                         = 1e-5
group_size                 = 8
temperature                = 0.9
clip_eps                   = 0.2
grad_clip                  = 0.5
total_steps                = 100
curriculum_level_2_step    = 33
curriculum_level_3_step    = 67
include_menu               = False
fast_grpo_logprobs         = True
```

Adapter: [`SnehShah/house-md-grpo-optimized-gemma3-4b-v3`](https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3)

---

## 5. Numbers

The base model results are taken from `results/eval_base.json` (45 patients, 9 difficulty buckets, 5 patients each). Gemini Flash is a same-prompt, no-fine-tuning frontier baseline.

| model | acc % | avg total | r1 acc | r2 cost | r6 anchor | r7 safety | r8 format | avg cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 13.3 | -0.42 | -0.20 | +1.00 | +0.45 | -1.40 | +1.00 | 250 |
| greedy heuristic | 28.9 | +0.81 | +0.32 | +0.74 | +0.42 | -0.84 | +1.00 | 230 |
| **Gemma 3 4B-IT (base)** | **17.8** | **+0.07** | +0.10 | +0.65 | +0.07 | -0.78 | +0.60 | 227 |
| **+ SFT** | *see W&B / notebook 04* | | | | | | | |
| **+ GRPO** | *see W&B / notebook 04* | | | | | | | |
| Gemini Flash (frontier ref.) | 95.6 | … | … | … | … | … | … | … |

> **Note**: the SFT and GRPO 45-patient numbers above are best refreshed from `04_eval_compare.ipynb`, which downloads `eval_sft.json` and `eval_grpo.json` from `SnehShah/house-md-results` (or runs them live against the Space). The W&B mid-training mini-evals are visible in the dashboard linked above; they show the expected pattern (GRPO mean-reward climbing into the 1.4–1.9 band on the Level-1 curriculum, with dips at curriculum boundaries 33 and 67 as the harder diseases come in).

The interesting per-rubric story is in `eval_base.json`:

- **r1 accuracy** of **+0.10** despite an absolute accuracy of 17.8 % — the base model gets *partial* credit on a few wrongly-labelled-but-correctly-evidenced cases, and crashes hard on critical patients.
- **r7 safety** of **-0.78** — the base model misses a *lot* of critical-acute presentations. The GRPO curriculum puts these at Level 1 specifically to fix this.
- **r8 format** of **+0.60** — i.e. ~40 % of base-model actions don't parse. SFT alone takes this to >0.95.

Per-difficulty breakdown for the base model:

| difficulty | n | correct % | avg total |
| --- | --- | --- | --- |
| critical_textbook | 5 | 60 | +0.93 |
| critical_atypical | 5 | 20 | -0.86 |
| critical_acute_severe | 5 | 20 | -0.71 |
| urgent_textbook | 5 | 20 | +0.36 |
| urgent_atypical | 5 | 0 | -0.45 |
| urgent_acute_severe | 5 | 0 | -0.14 |
| stable_textbook | 5 | 20 | +0.70 |
| stable_atypical | 5 | 0 | -0.01 |
| stable_acute_severe | 5 | 20 | +0.83 |

The "atypical" buckets are the ones GRPO has the most room to move on: same disease, weirder presentation, more interview/exam needed before diagnosing.

---

## 6. Reproduce in 30 minutes

We packaged everything in [`notebooks/`](https://github.com/SnehShah/house-md-env/tree/main/notebooks) so judges (or you) can re-run the entire pipeline:

| Notebook | Hardware | Time | What it does |
| --- | --- | --- | --- |
| [`01_explore_env.ipynb`](https://github.com/SnehShah/house-md-env/blob/main/notebooks/01_explore_env.ipynb) | CPU | ~3 min | Connects to the live Space, runs random + oracle episodes, prints reward breakdown. |
| [`02_sft.ipynb`](https://github.com/SnehShah/house-md-env/blob/main/notebooks/02_sft.ipynb) | T4 | ~12 min | Mini SFT loop on Gemma 3 4B-IT, 200 samples × 1 epoch, pushes adapter. |
| [`03_grpo.ipynb`](https://github.com/SnehShah/house-md-env/blob/main/notebooks/03_grpo.ipynb) | T4 | ~25 min | 30-step GRPO loop talking to the live Space, plots reward curve, pushes adapter. |
| [`04_eval_compare.ipynb`](https://github.com/SnehShah/house-md-env/blob/main/notebooks/04_eval_compare.ipynb) | CPU (or T4 for live mini-eval) | ~1 min (5 sec to plot, 15 min for live mode) | Loads the pre-computed eval JSONs and plots the comparison. |

For headless / CI mode:

```bash
HF_TOKEN=hf_... HF_USERNAME=SnehShah ./notebooks/run_all.sh all
```

---

## 7. Lessons learned

1. **Wrap your env early.** We wrote `clinical_rl/` first and OpenEnv-wrapped it second; the wrapper turned out to be ~250 lines because the underlying simulator was already cleanly separated into `Action`, `Observation`, `State`. If your env mixes its core dynamics with HTTP details, OpenEnv-ing it later is much harder.
2. **Mount your UI on the same process.** OpenEnv ships with a generic Gradio web view at `/web`. We wanted the animated ER scene we'd already built; putting it at `/` and disabling the default Gradio mount took 30 lines in `app.py`. Same Space, same uvicorn process — judges can still hit `/docs` and `/state` for the OpenEnv contract while clinicians see the visual UI.
3. **Cache your old log-probs.** Doubling the rollout cost for a "real" PPO ratio is rarely worth it on small models; cache `log π_old(a|s)` during sampling and you halve your GPU bill.
4. **Watch every rubric, not the total.** Reward hacking *will* happen if you only watch the scalar. Our W&B dashboard has separate panels for r1, r6, r8 mean-per-group; the `r6_anchoring` collapse around step 25 (we saw a curve sit at 0 for 6 consecutive steps before bouncing) was visible there long before it would have shown up in the total.

---

## 8. Acknowledgements

- The **OpenEnv** team (Meta + PyTorch) for shipping a contract that was "build env on Friday, push to a Space on Saturday" easy.
- **Unsloth** for the 4-bit Gemma 3 wheel that made all of this fit on an L4.
- **TRL** for `SFTTrainer` + `train_on_responses_only`.
- **Weights & Biases** for the dashboards that caught every reward-hacking attempt early.

If you want to chat about clinical RL, sparse rewards, or specifically why teaching an LLM "ask the cheapest disambiguating question first" is harder than teaching it "always order CT abdomen", find me on the Hugging Face forums or in the OpenEnv hackathon Discord.

— Sneh
