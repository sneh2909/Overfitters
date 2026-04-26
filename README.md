---
title: House M.D. — Clinical Reasoning RL Environment
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: true
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - clinical-reasoning
  - llm-agents
  - pomdp
  - grpo
short_description: POMDP RL env where an LLM plays a diagnostician in the ER.
---

# 🩺 House M.D. — OpenEnv Clinical Reasoning Environment

> **Apr '26 Meta OpenEnv Hackathon submission.** Trained with **GRPO** on
> **Gemma 3 4B-IT**. Live Space, four reproduction notebooks, an HF blog post,
> and a public W&B run included.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-spec_v1-orange)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗_Space-SnehShah/house--md--env-yellow)](https://huggingface.co/spaces/SnehShah/house-md-env)
[![SFT adapter](https://img.shields.io/badge/🤗_Model-house--md--sft--gemma3--4b-blue)](https://huggingface.co/SnehShah/house-md-sft-gemma3-4b)
[![GRPO adapter](https://img.shields.io/badge/🤗_Model-house--md--grpo--optimized--gemma3--4b--v3-blue)](https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3)
[![W&B](https://img.shields.io/badge/W%26B-house--md-yellow)](https://wandb.ai/sneh2909-christ-university/house-md?nw=nwusersneh2909)
[![Blog](https://img.shields.io/badge/Blog-Teaching_Gemma_3_to_Diagnose-purple)](docs/blog_post.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SnehShah/house-md-env/blob/main/notebooks/01_explore_env.ipynb)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)

---

## What this is

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement
learning environment where an LLM agent plays a **diagnostician in the ER**.

It receives a vague chief complaint, then chooses among five actions —
`INTERVIEW`, `EXAMINE`, `ORDER_TEST`, `UPDATE_DIFFERENTIAL`, `DIAGNOSE` — to
reach the correct diagnosis while **minimizing cost, time, and patient risk**.

This is not a classification task. It is a **POMDP** — partial observability,
sequential decisions under uncertainty, costly information, deteriorating
patients. The right policy is "ask two cheap questions, pick the *one* test
that disambiguates the top suspects, then commit" — exactly the reasoning loop
GRPO is good at sharpening.

---

## What's in this repo

This single repository serves two purposes:

1. It **is** the OpenEnv environment package — `openenv push` from the repo
   root publishes the live Hugging Face Space at
   [`SnehShah/house-md-env`](https://huggingface.co/spaces/SnehShah/house-md-env).
2. It contains the **reproduction kit** for judges — four Colab notebooks, a
   headless runner, training scripts, eval results, and the HF blog post.

```text
house-md-env/
├── house_md_env/                  # OpenEnv package (everything inside is shipped to the Space)
│   ├── __init__.py                # exports HouseMDEnv (client) + HouseMDAction/Observation
│   ├── client.py                  # OpenEnv-style Pydantic client
│   ├── models.py                  # Action / Observation / State as Pydantic
│   ├── server/
│   │   ├── app.py                 # FastAPI: OpenEnv contract + custom UI + /api/*
│   │   ├── house_md_environment.py# wraps clinical_rl.ClinicalEnv as openenv.core.Environment
│   │   ├── playground.py          # session-based /api routes for the UI
│   │   └── static/                # the animated ER scene
│   ├── clinical_rl/               # vendored core simulator (cards, rewards, oracle)
│   └── data/                      # 15 disease cards × 3 variants + catalogs + eval set
├── notebooks/                     # 01-04 Colab notebooks + run_all.{sh,py}
├── docs/                          # blog post + GRPO dry-run write-up
├── scripts/                       # production train + eval scripts (HF Jobs ready)
├── results/                       # frozen eval JSONs for the comparison plot
├── tests/                         # pytest suite (rewards, oracle, prompt, dynamic patient)
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # builds the `house_md_env` package
├── Dockerfile                     # multi-stage build for the Space
├── .dockerignore                  # excludes notebooks/, docs/, scripts/, tests/, results/
└── README.md                      # this file (also the HF Space card)
```

---

## Quick start (1 minute)

### Run an episode against the live Space

```python
from house_md_env import HouseMDEnv, HouseMDAction

with HouseMDEnv(base_url="https://snehshah-house-md-env.hf.space") as env:
    res = env.reset(seed=42)
    print("Patient:", res.observation.chief_complaint)

    res = env.step(HouseMDAction(
        type="INTERVIEW",
        argument="lmp",
        rationale="rule out pregnancy in young woman with abdominal pain",
    ))
    res = env.step(HouseMDAction(
        type="ORDER_TEST",
        argument="beta_hcg_quant",
        rationale="confirm pregnancy before imaging",
    ))
    res = env.step(HouseMDAction(
        type="DIAGNOSE",
        argument="ectopic_pregnancy",
        rationale="positive bHCG + RUQ pain in 28F with missed period",
    ))
    print("Final reward:", res.reward)
    print("Per-rubric:",  res.observation.rewards)
```

### Open the live ER scene in your browser

The Space root **is** the playground — a hand-built animated ER scene where
you can hit *Play* and watch the oracle / greedy / random policies (or your
own manual actions) walk a patient through the room, queue tests at the lab
bench, ring up costs at the cashier, and reveal the ground-truth diagnosis
when the case closes.

- [`/`](https://snehshah-house-md-env.hf.space/) — **live ER scene** (default landing page)
- [`/docs`](https://snehshah-house-md-env.hf.space/docs) — FastAPI Swagger UI
- [`/schema`](https://snehshah-house-md-env.hf.space/schema) — JSON schemas
- [`/health`](https://snehshah-house-md-env.hf.space/health) — health check

---

## Reproduce in 30 minutes (judges, start here)

Four Colab-ready notebooks. They reference the live Space, real datasets, and
real adapters — no local GPU needed for `01` and `04`.

| # | Notebook | Hardware | Time | What it does |
|---|----------|----------|------|--------------|
| 01 | [`notebooks/01_explore_env.ipynb`](notebooks/01_explore_env.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SnehShah/house-md-env/blob/main/notebooks/01_explore_env.ipynb) | CPU | ~3 min | Connects to the live Space; runs manual / random / oracle episodes; prints the reward breakdown. |
| 02 | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SnehShah/house-md-env/blob/main/notebooks/02_sft.ipynb) | T4 | ~12 min | Mini SFT loop on Gemma 3 4B-IT, 200 oracle traces × 1 epoch, pushes a LoRA adapter. |
| 03 | [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SnehShah/house-md-env/blob/main/notebooks/03_grpo.ipynb) | T4 | ~25 min | 30-step GRPO loop against the live Space, plots reward curve, pushes a LoRA adapter. |
| 04 | [`notebooks/04_eval_compare.ipynb`](notebooks/04_eval_compare.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SnehShah/house-md-env/blob/main/notebooks/04_eval_compare.ipynb) | CPU | ~1 min | Loads the frozen eval JSONs and plots the base / SFT / GRPO comparison. |

### Headless / CI-style

```bash
HF_TOKEN=hf_... HF_USERNAME=SnehShah ./notebooks/run_all.sh all
# or
python notebooks/run_all.py
```

Both run `01-04` in order with `jupyter nbconvert --execute`, writing the
executed copies to `notebooks/executed_*.ipynb`.

### Production training (HF Jobs, what we actually ran)

```bash
# SFT warm-start
python scripts/submit_sft_job.py --hardware l4x1

# GRPO main run (the one in W&B)
python scripts/submit_grpo_optimized_job.py --hardware l4x1
```

These submit the unmodified `train_sft.py` / `train_grpo_optimized.py` to
[Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs).

---

## Action space

| `type`                | `argument`         | Notes                                                |
|-----------------------|--------------------|------------------------------------------------------|
| `INTERVIEW`           | question id        | Cheap; noisy; same question -> same answer (cached)  |
| `EXAMINE`             | exam id            | Costs 1–2 min and small $; deterministic per patient |
| `ORDER_TEST`          | test id            | Big cost; some tests have multi-step turnaround      |
| `UPDATE_DIFFERENTIAL` | summary string     | Free; carries a `board=[{disease, prob}, ...]`       |
| `DIAGNOSE`            | disease id         | Terminal — ends the episode                          |

The full catalog ids are returned by `GET /schema` and are enumerated in
[`data/`](data/).

---

## Reward design

Computed at terminal (`obs.terminal == True`); surfaced as both the
OpenEnv-standard scalar `obs.reward` and a per-rubric breakdown
`obs.rewards`:

| Rubric          | Range          | Captures                                          |
|-----------------|---------------:|---------------------------------------------------|
| `r1_accuracy`   | -2 – +1        | Right disease *and* saw the necessary evidence    |
| `r2_cost`       | -1.5 – +1      | Sweet-spot $200–500; large penalties at $1500+    |
| `r6_anchoring`  | -0.5 – +0.6    | Did the agent revise its differential meaningfully|
| `r7_safety`     | -2 – 0         | Penalty-only; severity-scaled wrong dx / timeouts |
| `r8_format`     | 0 – 1          | Fraction of valid (in-vocab, well-formed) actions |

Default composite weights: `{r1: 2.0, r2: 0.5, r6: 0.3, r7: 1.0, r8: 0.5}`.
The trainer is free to override.

The full reward code is in [`clinical_rl/rewards.py`](clinical_rl/rewards.py).
The reward-hacking failure modes we anticipated and patched are documented in
the [blog post](docs/blog_post.md).

---

## Disease corpus (15 diseases × 3 variants)

`anxiety_attack`, `appendicitis`, `bacterial_meningitis`, `costochondritis`,
`dka`, `ectopic_pregnancy`, `migraine`, `ovarian_torsion`, `pneumonia`,
`pulmonary_embolism`, `sepsis_uti`, `stemi`, `subarachnoid_hemorrhage`,
`viral_gastroenteritis`, `viral_uri`.

Each card has 3 presentation variants (textbook / atypical / red-herring),
~25 symptom probabilities, ~15 exam findings, ~35 test sensitivities, and a
declared **minimum evidence set** that the accuracy rubric uses to distinguish
"correct + diagnostic workup" from "correct + lucky guess".

---

## Evidence of training

| Artifact | Where |
|---|---|
| **W&B run** (production GRPO, all rubrics, gradients, mid-eval) | <https://wandb.ai/sneh2909-christ-university/house-md?nw=nwusersneh2909> |
| **SFT adapter** (LoRA, Gemma 3 4B-IT) | <https://huggingface.co/SnehShah/house-md-sft-gemma3-4b> |
| **GRPO adapter** (LoRA, Gemma 3 4B-IT) | <https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3> |
| **Frozen evals** (45 patients × {base, SFT, GRPO}) | [`results/`](results/) + [`SnehShah/house-md-results`](https://huggingface.co/datasets/SnehShah/house-md-results) |
| **SFT dataset** (2,151 oracle traces) | <https://huggingface.co/datasets/SnehShah/house-md-sft-data> |
| **Mini-blog** (this repo) | [`docs/blog_post.md`](docs/blog_post.md) |
| **GRPO dry-run write-up** (tensors + shapes) | [`docs/grpo_dry_run.md`](docs/grpo_dry_run.md) |

---

## Environment metadata

- **Step cap:** 15 (configurable via `step_cap` ctor arg)
- **Action timeout:** none on the env side; wallclock is logged via
  `obs.time_elapsed_min`
- **Concurrency:** `SUPPORTS_CONCURRENT_SESSIONS = True`. Each session gets its
  own `HouseMDEnvironment` instance with an isolated `Episode`. Default cap of
  8 concurrent sessions; raise via the `MAX_CONCURRENT_ENVS` env var.

---

## Local development

```bash
# 1. clone
git clone https://github.com/SnehShah/house-md-env.git
cd house-md-env

# 2. install (creates a venv, installs the package + dev deps)
uv sync --extra dev

# 3. run the env locally
uv run uvicorn house_md_env.server.app:app --host 0.0.0.0 --port 8000
# -> open http://localhost:8000

# 4. run the test suite
uv run pytest -q
```

To rebuild and re-push the Space:

```bash
openenv build .
openenv push --repo-id SnehShah/house-md-env
```

---

## Hackathon judging-criteria checklist

| Criterion | Where to look |
|---|---|
| **Environment built with OpenEnv** | [`server/`](server/), [`models.py`](models.py), [`client.py`](client.py), [`openenv.yaml`](openenv.yaml) |
| **OpenEnv environment as a HF Space** | <https://huggingface.co/spaces/SnehShah/house-md-env> |
| **Training with Unsloth or HF TRL, ideally as a Colab notebook** | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb), [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) — both use Unsloth + TRL |
| **Evidence of actual training** | W&B run, SFT/GRPO adapters, frozen eval JSONs (linked above) |
| **Mini-blog on Hugging Face or < 2 min YouTube video** | [`docs/blog_post.md`](docs/blog_post.md) |
| **Everything in README** | this file |

---

## Authors

- **Sneh Shah** — env design, reward engineering, GRPO training
- **Ayush Aryan** — disease cards, oracle, eval pipeline
- **Arjun Bhammar** — UI, infra

---

## License

[Apache 2.0](LICENSE)
