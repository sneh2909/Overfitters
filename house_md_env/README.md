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

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement
learning environment where an LLM agent plays a **diagnostician in the ER**.

It receives a vague chief complaint, then chooses among five actions —
`INTERVIEW`, `EXAMINE`, `ORDER_TEST`, `UPDATE_DIFFERENTIAL`, `DIAGNOSE` — to
reach a correct diagnosis while **minimizing cost, time, and patient risk**.

> Built for the **Apr '26 Meta OpenEnv Hackathon**. Trained with **GRPO** on
> **Gemma 3 4B-IT**. See the
> [GitHub repo](https://github.com/sneh2909/Overfitters) for the full
> training pipeline, four reproduction notebooks, and the HF blog post.

---

## Quick start

### From Python

```python
from house_md_env import HouseMDEnv, HouseMDAction

with HouseMDEnv(base_url="https://snehshah-house-md-env.hf.space") as env:
    res = env.reset(seed=42)
    obs = res.observation
    print("Patient:", obs.chief_complaint)
    print("Vitals:", obs.intake_vitals)

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
    print("Per-rubric:", res.observation.rewards)
```

### From the browser

The Space root **is** the playground — a hand-built animated ER scene where
you can hit *Play* and watch the oracle / greedy / random policies (or your
own manual actions) walk a patient through the room, queue tests at the lab
bench, ring up costs at the cashier, and reveal the ground-truth diagnosis
when the case closes.

- [`/`](./) — **live ER scene** (default landing page)
- [`/docs`](./docs) — FastAPI Swagger UI
- [`/schema`](./schema) — JSON schemas for action / observation / state
- [`/health`](./health) — health check

---

## Action space

| `type`                | `argument`         | Notes                                                |
|-----------------------|--------------------|------------------------------------------------------|
| `INTERVIEW`           | question id        | Cheap; noisy; same question -> same answer (cached)  |
| `EXAMINE`             | exam id            | Costs 1–2 min and small $; deterministic per patient |
| `ORDER_TEST`          | test id            | Big cost; some tests have multi-step turnaround      |
| `UPDATE_DIFFERENTIAL` | summary string     | Free; carries a `board=[{disease, prob}, ...]`       |
| `DIAGNOSE`            | disease id         | Terminal — ends the episode                          |

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

---

## Disease corpus (15 diseases × 3 variants)

`anxiety_attack`, `appendicitis`, `bacterial_meningitis`, `costochondritis`,
`dka`, `ectopic_pregnancy`, `migraine`, `ovarian_torsion`, `pneumonia`,
`pulmonary_embolism`, `sepsis_uti`, `stemi`, `subarachnoid_hemorrhage`,
`viral_gastroenteritis`, `viral_uri`.

---

## Environment metadata

- **Step cap:** 15 (configurable via `step_cap` ctor arg)
- **Concurrency:** `SUPPORTS_CONCURRENT_SESSIONS = True`. Each session gets its
  own `HouseMDEnvironment` instance with an isolated `Episode`. Default cap of
  8 concurrent sessions; raise via the `MAX_CONCURRENT_ENVS` env var.

---

## License

Apache 2.0. See [the LICENSE in the parent repo](https://github.com/sneh2909/Overfitters/blob/main/LICENSE).
