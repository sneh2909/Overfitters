# 🩺 House M.D. — OpenEnv Clinical Reasoning RL Environment

> **Apr '26 Meta OpenEnv Hackathon submission.** Trained with **GRPO** on
> **Gemma 3 4B-IT**. Live Space, four reproduction notebooks, an HF blog post,
> and a public W&B run included.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-spec_v1-orange)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗_Space-SnehShah/house--md--env-yellow)](https://huggingface.co/spaces/SnehShah/house-md-env)
[![OpenEnv UI Space](https://img.shields.io/badge/🤗_OpenEnv_UI-SnehShah/house--md--env--openenv-yellow)](https://huggingface.co/spaces/SnehShah/house-md-env-openenv)
[![SFT adapter](https://img.shields.io/badge/🤗_Model-house--md--sft--gemma3--4b-blue)](https://huggingface.co/SnehShah/house-md-sft-gemma3-4b)
[![GRPO adapter](https://img.shields.io/badge/🤗_Model-house--md--grpo--optimized--gemma3--4b--v3-blue)](https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3)
[![W&B](https://img.shields.io/badge/W%26B-house--md-yellow)](https://wandb.ai/sneh2909-christ-university/house-md?nw=nwusersneh2909)
[![Blog](https://img.shields.io/badge/Blog-Teaching_Gemma_3_to_Diagnose-purple)](docs/blog_post.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/01_explore_env.ipynb)
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

## Key paths

| What | Path |
|---|---|
| **blog post** | [`docs/blog.md`](docs/blog.md) |
| **HF Space — custom ER scene** | [`SnehShah/house-md-env`](https://huggingface.co/spaces/SnehShah/house-md-env) → <https://snehshah-house-md-env.hf.space/> |
| **HF Space — OpenEnv UI** | [`SnehShah/house-md-env-openenv`](https://huggingface.co/spaces/SnehShah/house-md-env-openenv) → <https://snehshah-house-md-env-openenv.hf.space/web/> |
| **All-in-one notebook** (env tour + mini SFT + mini GRPO + eval) | [`notebooks/00_run_all.ipynb`](notebooks/00_run_all.ipynb) |
| **SFT training notebook** | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb) |
| **GRPO training notebook** | [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) |
| **Production SFT script** (HF Jobs) | [`scripts/submit_sft_job.py`](scripts/submit_sft_job.py) → [`scripts/train_sft.py`](scripts/train_sft.py) |
| **Production GRPO script** (HF Jobs) | [`scripts/submit_grpo_optimized_job.py`](scripts/submit_grpo_optimized_job.py) → [`scripts/train_grpo_optimized.py`](scripts/train_grpo_optimized.py) |
| **Eval script** (HF Jobs) | [`scripts/eval.sh`](scripts/eval.sh) → [`scripts/eval_hf.py`](scripts/eval_hf.py) |

---

## Repository layout

```text
house-md-env/
├── README.md                      # this file (GitHub README)
├── LICENSE
├── .gitignore
├── pytest.ini                     # repo-level test config
│
├── house_md_env/                  # the OpenEnv environment package
│   │                              # → cd here and run `openenv push`
│   ├── __init__.py                # exports HouseMDEnv (client) + HouseMDAction/Observation
│   ├── client.py                  # OpenEnv-style Pydantic client
│   ├── models.py                  # Action / Observation / State as Pydantic
│   ├── server/                    # FastAPI app + custom UI + /api routes
│   ├── clinical_rl/               # vendored core simulator
│   ├── data/                      # 15 disease cards × 3 variants + catalogs + eval set
│   ├── pyproject.toml             # builds the `house_md_env` package
│   ├── openenv.yaml               # OpenEnv manifest (used by `openenv push`)
│   ├── Dockerfile                 # multi-stage build for the Space
│   ├── .dockerignore
│   └── README.md                  # HF Space card (with YAML front-matter)
│
├── notebooks/                     # 01-04 Colab notebooks + run_all.{sh,py}
├── docs/                          # blog post + GRPO dry-run write-up
├── scripts/                       # production train + eval scripts (HF Jobs ready)
├── results/                       # frozen eval JSONs for the comparison plot
└── tests/                         # 94-test pytest suite
```

The repo serves two purposes:

1. **`house_md_env/` IS the OpenEnv environment package.** From inside that
   directory, `openenv push` publishes the live Hugging Face Space at
   [`SnehShah/house-md-env`](https://huggingface.co/spaces/SnehShah/house-md-env).
2. **The repo root is the reproduction kit.** Notebooks, docs, scripts, eval
   results, and tests live alongside it but are excluded from the Space image.

---

## Quick start (1 minute)

### Run an episode against the live Space

```python
from house_md_env import HouseMDEnv, HouseMDAction

with HouseMDEnv(base_url="https://snehshah-house-md-env.hf.space") as env:
    res = env.reset(seed=42)
    print("Patient:", res.observation.chief_complaint)

    res = env.step(HouseMDAction(
        type="ORDER_TEST",
        argument="beta_hcg_quant",
        rationale="confirm pregnancy before imaging",
    ))
    res = env.step(HouseMDAction(
        type="DIAGNOSE",
        argument="ectopic_pregnancy",
        rationale="positive bHCG + RUQ pain in 28F",
    ))
    print("Final reward:", res.reward)
    print("Per-rubric:",  res.observation.rewards)
```

(`pip install git+https://github.com/sneh2909/Overfitters.git#subdirectory=house_md_env`
to install the package locally; the four notebooks already do this for you.)

### Open the live ER scene in your browser

The Space root **is** the playground — a hand-built animated ER scene where
you can hit *Play* and watch the oracle / greedy / random policies (or your
own manual actions) walk a patient through the room.

- <https://snehshah-house-md-env.hf.space/> — live ER scene
- <https://snehshah-house-md-env-openenv.hf.space/web/> — standard OpenEnv playground
- <https://snehshah-house-md-env.hf.space/docs> — FastAPI Swagger UI
- <https://snehshah-house-md-env.hf.space/schema> — JSON schemas

---

## Reproduce in 30 minutes (judges, start here)

Four Colab-ready notebooks. They reference the live Space, real datasets, and
real adapters — no local GPU needed for `01` and `04`.

| # | Notebook | Hardware | Time | What it does |
|---|----------|----------|------|--------------|
| 00 | [`notebooks/00_run_all.ipynb`](notebooks/00_run_all.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/00_run_all.ipynb) | CPU (GPU optional) | ~3 min CPU / ~40 min GPU | **All-in-one**: connect to the Space, schema tour, manual/random/oracle episodes, OpenEnv UI test inputs, eval comparison plot, plus optional mini SFT + mini GRPO + live mini-eval. |
| 01 | [`notebooks/01_explore_env.ipynb`](notebooks/01_explore_env.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/01_explore_env.ipynb) | CPU | ~3 min | Connects to the live Space; runs manual / random / oracle episodes; prints the reward breakdown. |
| 02 | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/02_sft.ipynb) | T4 | ~12 min | Mini SFT loop on Gemma 3 4B-IT, 200 oracle traces × 1 epoch, pushes a LoRA adapter. |
| 03 | [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/03_grpo.ipynb) | T4 | ~25 min | 30-step GRPO loop against the live Space, plots reward curve, pushes a LoRA adapter. |
| 04 | [`notebooks/04_eval_compare.ipynb`](notebooks/04_eval_compare.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sneh2909/Overfitters/blob/main/notebooks/04_eval_compare.ipynb) | CPU | ~1 min | Loads the frozen eval JSONs and plots the base / SFT / GRPO comparison. |

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
python scripts/submit_sft_job.py --hardware l4x1            # SFT warm-start
python scripts/submit_grpo_optimized_job.py --hardware l4x1 # GRPO main run (the one in W&B)
```

---

## Reward design (TL;DR)

Five rubrics, each a separate scalar, weighted into a single total. See
[`house_md_env/clinical_rl/rewards.py`](house_md_env/clinical_rl/rewards.py)
for the source and [`docs/blog_post.md`](docs/blog_post.md) for the
reward-hacking failure modes we anticipated and patched.

| Rubric          | Range          | Captures                                          |
|-----------------|---------------:|---------------------------------------------------|
| `r1_accuracy`   | -2 – +1        | Right disease *and* saw the necessary evidence    |
| `r2_cost`       | -1.5 – +1      | Sweet-spot $200–500; large penalties at $1500+    |
| `r6_anchoring`  | -0.5 – +0.6    | Did the agent revise its differential meaningfully|
| `r7_safety`     | -2 – 0         | Penalty-only; severity-scaled wrong dx / timeouts |
| `r8_format`     | 0 – 1          | Fraction of valid (in-vocab, well-formed) actions |

Default composite weights: `{r1: 2.0, r2: 0.5, r6: 0.3, r7: 1.0, r8: 0.5}`.

---

## Evidence of training

| Artifact | Where |
|---|---|
| **W&B run** (production GRPO, all rubrics, gradients, mid-eval) | <https://wandb.ai/sneh2909-christ-university/house-md?nw=nwusersneh2909> |
| **SFT adapter** (LoRA, Gemma 3 4B-IT) | <https://huggingface.co/SnehShah/house-md-sft-gemma3-4b> |
| **GRPO adapter** (LoRA, Gemma 3 4B-IT) | <https://huggingface.co/SnehShah/house-md-grpo-optimized-gemma3-4b-v3> |
| **Frozen evals** (45 patients × {base, SFT, GRPO}) | [`results/`](results/) + [`SnehShah/house-md-results`](https://huggingface.co/datasets/SnehShah/house-md-results) |
| **SFT dataset** (2,151 oracle traces) | <https://huggingface.co/datasets/SnehShah/house-md-sft-data> |
| **Mini-blog** | [`docs/blog_post.md`](docs/blog_post.md) |
| **GRPO dry-run write-up** (tensors + shapes) | [`docs/grpo_dry_run.md`](docs/grpo_dry_run.md) |

---

## Local development

```bash
git clone https://github.com/sneh2909/Overfitters.git
cd house-md-env

# Run the env locally
cd house_md_env
pip install -e .[dev]
uvicorn server.app:app --host 0.0.0.0 --port 8000
# -> open http://localhost:8000

# Run the test suite (from the repo root)
cd ..
pip install pytest
pytest tests/
```

To rebuild and re-push the Space:

```bash
cd house_md_env
openenv build .
openenv push --repo-id SnehShah/house-md-env
```

> Note: `openenv push` requires `__init__.py`, `pyproject.toml`,
> `openenv.yaml`, and `Dockerfile` at the directory you push from — that's
> why those live inside `house_md_env/` rather than at the repo root.

---

## Hackathon judging-criteria checklist

| Criterion | Where to look |
|---|---|
| **Environment built with OpenEnv** | [`house_md_env/`](house_md_env/) (`server/`, `models.py`, `client.py`, `openenv.yaml`) |
| **OpenEnv environment as a HF Space** | <https://huggingface.co/spaces/SnehShah/house-md-env> |
| **Training with Unsloth or HF TRL, ideally as a Colab notebook** | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb), [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) — both use Unsloth + TRL |
| **Evidence of actual training** | W&B run, SFT/GRPO adapters, frozen eval JSONs (linked above) |
| **Mini-blog on Hugging Face or < 2 min YouTube video** | [`docs/blog_post.md`](docs/blog_post.md) |
| **Everything in README** | this file |

---

## Authors

- **Sneh Shah**
- **Ayush Aryan** 
- **Arjun Bhammar** 

---

## License

[Apache 2.0](LICENSE)
