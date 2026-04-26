# A Doctor That Learns to Investigate

## Teaching a 4B Open Model to Ask, Test, Think, and Diagnose

**Authors:** Ayush Aryan, Arjun Bhammar, Sneh Shah

**Live demo:** [huggingface.co/spaces/SnehShah/house-md-env](https://huggingface.co/spaces/SnehShah/house-md-env)

---

> Before you read the rest, open the demo for thirty seconds. Pick a random patient. Press play. Watch the agent ask questions, order tests, wait for results, spend money, and finally commit to a diagnosis.
>
> That moment is the whole project — the model is not answering a prepared case summary. It is surviving a live investigation where every action changes the next observation.

---

## Table of Contents

1. [The Pitch](#the-pitch)
2. [Why 0 and 1 Are Not Enough](#why-0-and-1-are-not-enough)
3. [The Environment](#the-environment)
4. [The Reward System — The Real Design Problem](#the-reward-system--the-real-design-problem)
5. [Building the World: Disease Cards](#building-the-world-disease-cards)
6. [The Oracle: Our First Doctor](#the-oracle-our-first-doctor)
7. [SFT: Teaching the Language of Action](#sft-teaching-the-language-of-action)
8. [GRPO: Learning From Consequences](#grpo-learning-from-consequences)
9. [Engineering: Making It Trainable](#engineering-making-it-trainable)
10. [Experiments and What We Learned](#experiments-and-what-we-learned)
11. [Reward Hacking: The Enemy We Designed Against](#reward-hacking-the-enemy-we-designed-against)
12. [What the Early Runs Showed](#what-the-early-runs-showed)
13. [Results](#results)
14. [The Playable Demo](#the-playable-demo)
15. [What We'd Do With More Time](#what-wed-do-with-more-time)
16. [The Final Story](#the-final-story)

---

## The Pitch

Most medical AI demos answer questions.

We wanted to build something harder: an AI that has to **earn** the answer.

No neat multiple-choice prompt. No paragraph with all the clues already packed inside. No shortcut where the model says a disease name and looks smart.

Our agent wakes up inside an emergency room. A patient is in front of it. The disease is hidden. The symptoms are incomplete. Tests cost money. Results take time. Critical patients can deteriorate. Every action matters.

The model must **interview, examine, order tests, update its hypothesis, and diagnose** — before the case collapses under cost, delay, or uncertainty.

This is not a medical quiz bot. It is a sequential decision-making environment where a small open model has to learn the rhythm of real investigation: ask the sharp question, choose the useful test, avoid overconfidence, stop when there is enough evidence, and never waste a critical patient's time.

Our hackathon bet:

> **Can a 4B open model learn not just what a diagnosis is, but how a diagnosis is reached?**

We built the environment, the reward system, the oracle, the SFT data pipeline, the GRPO trainer, the patient simulator, the rollout optimizations, the reward-hacking monitors, and the live demo to prove it.

---

## Why 0 and 1 Are Not Enough

This is the most important section in the blog. Everything else flows from it.

The obvious reward for medical diagnosis seems simple:

```
correct diagnosis  →  1
wrong diagnosis    →  0
```

That reward is useless. Not just suboptimal — **actively misleading**.

Here is why.

Suppose the model times out on a STEMI patient (critical, life-threatening). Score: `0`.

Now suppose the model wrong-diagnoses a viral URI (benign, stable). Score: `0`.

Same score. Completely different failure modes. One is a catastrophic safety failure; the other is a mild classification error. A binary label treats them as identical events and gives the model no direction to improve.

But it gets worse.

A model can get `1` by guessing the most common disease on step 1 and being right by chance. Under a binary reward, a lucky step-1 guess looks identical to a carefully reasoned eight-step investigation with full evidence. The model has no reason to prefer the careful path. So it won't.

In practice, the binary reward immediately teaches the model to **diagnose on step 1**, collect the lucky correct hits, and treat all other outcomes as noise. It is not learning medicine. It is learning gambling.

A real doctor does not just end up at the right answer. They earn it through a specific process: gather evidence, update hypotheses, rule out dangerous alternatives, decide when they know enough. **The path is the behavior we want to train — not just the endpoint.**

This is why we needed a structured reward with five components. Each component encodes a dimension of good diagnostic behavior that the binary label collapses away:

| What binary `0/1` hides | What our reward makes visible |
|:---|:---|
| Did the model gather any evidence? | R1 evidence requirement |
| Was the workup efficient or wasteful? | R2 cost shaping |
| Did the model update its thinking as evidence arrived? | R6 anchoring resistance |
| Was a critical patient left to deteriorate? | R7 safety penalty |
| Did the model produce valid actions? | R8 format score |

Every design decision in this project traces back to this table. The five rubrics exist because `0` and `1` are not a signal — they are a lottery.

---

## The Environment

The simulator is an emergency department. The model plays the physician.

A patient arrives with age, sex, chief complaint, vitals, and a hidden disease state. The model sees only what a doctor would see so far: the transcript, revealed symptoms, pending tests, costs, and its own previous differential.

### Five action types

| Action | Meaning | Cost |
|:---|:---|:---:|
| `INTERVIEW` | Ask one patient-history question | Free |
| `EXAMINE` | Perform one physical exam maneuver | $10–30 |
| `ORDER_TEST` | Order one lab, imaging, or bedside test | $10–$1,500 |
| `UPDATE_DIFFERENTIAL` | Update the ranked diagnosis probability board | Free |
| `DIAGNOSE` | Commit to the final diagnosis and end the episode | — |

### Episode mechanics

- **15-step hard cap** — fail to diagnose and the case times out (penalized)
- **Hidden ground truth** — the model never sees the true disease or the full test panel
- **Stochastic results** — test outcomes are pre-sampled from `prob_abnormal` distributions; every episode of the same disease plays out slightly differently
- **Cost pressure** — over-testing accumulates cost, reducing the cost-efficiency reward
- **Deterioration** — critical patients can deteriorate mid-episode, triggering a safety penalty even on a correct diagnosis

### The disease universe

We used 15 diseases across three tiers. The tiers were not cosmetic — they shaped the entire reward and curriculum design.

| Tier | Diseases |
|:---|:---|
| 🔴 **Critical** | Ectopic pregnancy, STEMI, Pulmonary embolism, Subarachnoid hemorrhage, Bacterial meningitis |
| 🟠 **Urgent** | Appendicitis, Ovarian torsion, Pneumonia, DKA, Sepsis |
| 🟢 **Stable / benign** | Migraine, Costochondritis, Viral URI, Viral gastroenteritis, Anxiety attack |

The benign diseases were not filler. They made the task substantially harder. A model that treats every patient like a critical case will over-test, over-spend, and lose reward. A strong policy must know not just when to act aggressively — but when stopping is the right move.

This made the task much more interesting than classification. It became a small but rich RL problem with:

- hidden state,
- delayed observations,
- stochastic test results,
- cost constraints,
- safety constraints,
- reward-hacking risks,
- and a natural-language interaction channel.

---

## The Reward System — The Real Design Problem

With the binary label failure in mind, we designed the reward around the behavior we actually wanted: **correct, evidence-backed, efficient, safe diagnosis**.

We compressed this into five rubrics — simple enough to debug, rich enough to shape real behavior.

```python
DEFAULT_WEIGHTS = {
    "r1_accuracy": 2.0,    # correct diagnosis with evidence
    "r2_cost":     0.5,    # efficient workup
    "r6_anchoring": 0.3,   # updating hypotheses
    "r7_safety":   1.0,    # critical-case safety
    "r8_format":   0.5,    # valid JSON actions
}
```

### R1 — Accuracy with evidence: the 0.2 band that defeats lazy guessing

```
1.0   correct diagnosis + all minimum-evidence groups satisfied
0.5   correct diagnosis + partial evidence
0.2   correct diagnosis + zero supporting evidence  ← the "lucky guess" band
0.0   wrong diagnosis or no diagnosis
```

The `0.2` band is the most important number in the project.

Without it, a lucky step-1 correct guess would score `0.0` — identical to a completely wrong diagnosis. GRPO would conclude that guessing is no better than failing with evidence, which inverts the incentive. The model has no reason to gather evidence at all.

With `0.2`, a lucky guess gets *some* credit so the model doesn't treat it as total failure. But it is still **5× worse** than actually working the case to `1.0`.

> That 5× gap is the gradient GRPO climbs. Every other reward component is tuned around keeping that gap stable.

### R2 — Cost efficiency: why the peak is at $200, not $0

A naïve cost reward would make `$0` the best outcome. That is dangerous — it means diagnosing on step 1 and spending nothing would maximize R2.

So R2 peaks in the realistic workup zone:

```
$0           →  0.2    under-tested, suspicious
$200–500     →  1.0    the sweet spot
$1,500       →  0.0    probably over-tested
$3,000+      → -0.5    clearly wasteful
```

This teaches the model that the right answer is not "no tests." The right answer is "the right tests."

Even a benign anxiety attack needs an ECG and troponin to rule out cardiac causes. Even a viral URI needs a chest X-ray to differentiate from pneumonia. Under-testing is as penalized as over-testing.

### R6 — Anchoring resistance: did you actually update your thinking?

A good clinician updates their hypothesis as new evidence arrives. R6 gives credit when an `UPDATE_DIFFERENTIAL` action changes disease probabilities meaningfully.

The threshold: probability shift must exceed `0.05`. Without this, the model could spam nearly identical differential boards and farm R6 for free. With it, only genuine updates count.

This rubric exists because one of the most common failure modes in early runs was a model that wrote its first differential and then never touched it again — anchoring on the initial hypothesis regardless of what the tests showed.

### R7 — Safety: the penalty that makes critical cases genuinely scary

R7 is penalty-only. It does not give positive reward. It only says when a trajectory was dangerous.

| Outcome | Penalty |
|:---|---:|
| Timeout on a **critical** patient | **−2.0** |
| Wrong dx on a **critical** patient | −1.0 |
| Timeout on an **urgent** patient | −1.0 |
| Wrong dx on an **urgent** patient | −0.5 |
| Correct dx but patient deteriorated | −0.2 |

The `−2.0` critical timeout is the strongest gradient signal in the entire reward. GRPO learns very quickly that dithering on STEMI or ectopic pregnancy is the worst possible outcome — not just bad, but catastrophically bad compared to everything else.

### R8 — Format: a continuous signal, not a binary gate

The policy emits JSON actions. If the JSON fails, the environment cannot execute the action. We made R8 **continuous** — if 8 out of 10 actions are valid, R8 is `0.8`, not `0`.

A hard gate would make format collapses catastrophically unrecoverable during GRPO. The continuous version gives the optimizer a smooth gradient to climb back out.

> When R8 drops below `0.8` during training, something is wrong. Almost always it means the learning rate is too high and the policy is drifting from the SFT format distribution.

### What different strategies actually score

| Strategy | Typical total | What happens |
|:---|---:|:---|
| 🏆 Thoughtful play | **+2.98** | Correct dx, full evidence, reasonable cost |
| 📦 Order everything | +2.14 | Correct but wasteful — R2 hurts |
| 🎲 Lucky guess (step 1) | +1.00 | Correct, no evidence — R1 = 0.2 × weight 2.0 |
| ❌ Wrong diagnosis | ~−0.06 | Bad, but not catastrophic on non-critical |
| ⏰ Timeout (critical) | ~**−1.40** | The floor — R7 dominates |

That gap between thoughtful play (+2.98) and lucky guess (+1.00) is the entire training signal. The model cannot plateau at the lazy strategy.

---

## Building the World: Disease Cards

Before training a model, we needed the world to be real enough.

Each disease card contains:

- Three presenting variants (textbook, acute-severe, atypical)
- Symptom responses to 25 interview questions
- Physical exam findings
- Lab and imaging result distributions (3 abnormal values + 1 normal per test)
- Minimum evidence groups — what must be satisfied for a diagnosis to count as evidence-backed
- Deterioration behavior
- Disease severity tier

### The generation pipeline

We generated the initial 15 cards with **Gemini Pro** via Vertex AI — specifically chosen for its deterministic outputs within a session for a fixed seed. That reproducibility mattered: all 222 oracle trajectories need to re-play against the same underlying cards to produce consistent training data.

The schema was strict. Gemini Pro sometimes missed fields, produced inconsistent distributions, or violated evidence-group constraints. We added a **Gemini Flash repair loop** that re-ran schema validation and prompted for corrections.

The result: all 15 cards passed validation. Every card could generate three distinct presentations that felt clinically plausible, not just randomly shuffled symptom lists.

This mattered for RL. If the simulator is inconsistent, the model learns noise. The strictness of the card schema was not overhead — it was the foundation that made everything else meaningful.

---

## The Oracle: Our First Doctor

Before RL, we needed a teacher. Before SFT, we needed training data.

We wrote a **heuristic oracle** that plays each case near-optimally. It knows the disease card and can choose a good workup. We swept it across:

```
15 diseases × 3 variants × 5 seeds = 225 trajectories
```

After filtering low-reward trajectories, we kept:

```
222 / 225 oracle trajectories
2.86 average reward
100% correct final diagnosis on kept trajectories
```

### The oracle's playbook

1. Ask 2–3 targeted interview questions (always ask LMP/pregnancy for reproductive-age women)
2. Write an initial differential board
3. Perform one relevant physical exam
4. Order the cheapest tests that satisfy minimum evidence groups
5. Fill dead time with more interviews while tests are pending
6. Update the differential with final test results
7. Diagnose

### The subtle detail: polarity-aware test selection

This is the kind of thing that does not show up in a headline but broke early oracle runs.

Suppose pneumonia needs imaging evidence. A chest X-ray might come back normal — test results are sampled from distributions, so a `prob_abnormal = 0.7` test comes back clear roughly 30% of the time.

If the oracle blindly picks the cheapest test in each evidence group, it might order a CXR, get a normal result, and fail to satisfy the evidence group. The trajectory is then useless for training.

The fix: the oracle peeks at the hidden sampled test results (`env._episode.hidden.test_results`) and picks a test whose actual sampled branch satisfies the required evidence polarity. This is cheating — and intentionally so. The oracle is a teacher, not the final policy. The trained model never sees hidden state.

This gave us clean demonstrations of good diagnostic behavior, not lucky ones.

---

## SFT: Teaching the Language of Action

GRPO from a raw model is painful. If the model cannot reliably emit valid JSON, RL spends its entire budget rediscovering syntax instead of learning diagnosis.

So we first did supervised fine-tuning from oracle trajectories.

We replayed all 222 oracle trajectories through the live environment and recorded every observation-action pair:

```
222 trajectories
2,151 prompt/action pairs
Gemma 3 4B-IT
Unsloth 4-bit LoRA
assistant-only loss
```

### Action distribution

The training data looked like a real diagnostic loop — not random, not trivially skewed:

| Action | Share |
|:---|---:|
| `INTERVIEW` | 42.1% |
| `UPDATE_DIFFERENTIAL` | 20.6% |
| `ORDER_TEST` | 16.6% |
| `EXAMINE` | 10.3% |
| `DIAGNOSE` | 10.3% |

### What SFT was and was not meant to do

SFT was not meant to solve the task. It was meant to teach:

- valid JSON syntax and structure
- the five action types and their valid ID formats
- that episodes must end with `DIAGNOSE`
- the rough rhythm of a diagnostic workup: interview → update → test → update → diagnose

It teaches *none of this from reward* — it just imitates the oracle.

> Why SFT first? Because GRPO from a model that can't produce valid JSON wastes all its early steps on parse failures. R8 → 0, the reward signal is dominated by format noise, and the optimizer is fighting syntax instead of learning medicine. SFT gives GRPO a clean starting distribution to search from.

### Model choice: Gemma vs. Qwen

We originally considered Qwen 2.5-3B, but switched to **Gemma 3 4B-IT** after early experiments. Gemma's instruction-following quality made format learning converge faster and stay more stable through GRPO. The format stability during GRPO was the deciding factor — we needed R8 to start high so the optimizer could spend its budget on clinical behavior.

---

## GRPO: Learning From Consequences

### We had to write this from scratch — and that was non-negotiable

The first thing we reached for was TRL's `GRPOTrainer`. It did not fit. Unsloth's RL library did not fit either.

Both assume **single-turn completions**: generate one response, score it, update. That is not what we have. Our environment is fundamentally episodic:

- Each step produces a new observation that conditions the next action.
- Reward is a function of the **entire episode trajectory** — not a single completion.
- A single rollout is 8–15 sequential actions, each with its own prompt, generated tokens, and log-probabilities.
- The reward for action 3 depends on what happens at action 12.

There is no way to shoehorn multi-turn episodic RL into a standard single-turn trainer. You cannot simply concatenate all turns into one long sequence — the context shifts, the environment state changes, and the gradient must be attributed per action token, not per completion.

So we wrote the entire GRPO training loop ourselves:

```
Custom multi-turn GRPO trainer (scripts/train_grpo_optimized.py)

┌─────────────────────────────────────────────────────┐
│  Per-turn log-probability computation               │
│  (correct token slice, smoke-tested before every    │
│   run — an off-by-one here is invisible but fatal)  │
├─────────────────────────────────────────────────────┤
│  Multi-environment rollout batching                 │
│  (K env states alive simultaneously, batched        │
│   generate() per turn instead of K sequential runs) │
├─────────────────────────────────────────────────────┤
│  Sequential per-turn backward                       │
│  (gradients accumulate across turns + rollouts,     │
│   peak memory = one turn's graph, not K×15 graphs)  │
├─────────────────────────────────────────────────────┤
│  Optimized GRPO loss (Fast-GRPO)                    │
│  (single-update regime → old-logprob pass removed,  │
│   3 model calls → 2 per turn, ~33% cost reduction)  │
├─────────────────────────────────────────────────────┤
│  Per-rubric reward logging + hacking monitors       │
│  (5 separate curves, text-based alarm system)       │
└─────────────────────────────────────────────────────┘
```

This was the right call. An off-the-shelf trainer would have required more effort to work around than to replace. Building it ourselves gave us full control over gradient attribution, memory footprint, and the diagnostic tooling that made failures legible.

### The loop

1. Pick one patient (disease + variant + fresh random seed)
2. Have the model play that same patient **K times** with temperature — generating K independent rollouts
3. Score every rollout using the reward engine: `compute_all(episode, card, catalogs)["total"]`
4. Compute group-relative advantage:

```
Aᵢ = (rewardᵢ − group_mean) / group_std
```

5. Push the policy toward above-average trajectories, away from below-average ones

The group-relative normalization is the key insight. The signal is **relative, not absolute**. If all K rollouts score 3.0, advantages are all 0 — no gradient, no learning, but that is fine because the policy is already doing well on this patient. If one rollout scores 2.0 and another scores 3.0, the second gets positive advantage even though 2.0 is already pretty good. This keeps the learning signal alive throughout training.

### The log-prob detail that could have broken everything

For policy gradient, we needed the log-probability of the sampled action tokens, not the whole prompt.

The model sees:

```
[prompt tokens][assistant action tokens]
```

The logits that predict action token `a₀` are produced at position `n-1` (the last prompt token). The correct slice:

```python
action_logits = logits[prompt_len - 1 : prompt_len - 1 + action_len]
```

An off-by-one bug still produces tensors of the right shape and similar magnitude. It looks fine. But the gradient points at the wrong tokens. The model would train for hundreds of expensive steps with no improvement, and the only symptom would be "RL is not converging."

Before every run, we added smoke tests: shape, value range (all log-probs ≤ 0), determinism across identical inputs, and that `.backward()` actually writes gradients into the LoRA parameters.

This is the kind of engineering detail that does not show up in a demo or a reward curve. It is the difference between "RL is not improving" and "the math is actually connected."

---

## Engineering: Making It Trainable

GRPO cost is multiplicative:

```
training_steps
× group_size
× turns_per_episode
× prompt_tokens
× forward_passes_per_turn
```

A 30% cut on two factors beats a 50% cut on one factor. We attacked every factor.

| Factor | What we did | Impact |
|:---|:---|:---|
| Prompt tokens | Compact menu / no-menu experiments | ~4× on prompt length |
| Forward passes | Fast-GRPO old-logprob removal | ~33% fewer forward calls |
| Rollout generation | Batched K rollouts per environment turn | ~2.3× rollout speedup |
| Memory | Sequential per-turn backward | Fits larger groups on same GPU |
| Early episode length | Instant-test curriculum | Faster credit assignment |
| Sample efficiency | Disease curriculum + adaptive sampling | Better use of training steps |

### Fast-GRPO: our optimized loss for the single-update regime

Standard GRPO implementations — including the PPO-style variant in TRL — carry an old-logprob computation designed for multi-epoch PPO reuse. In a single-update training loop like ours, that computation is pure overhead. We derived and implemented an optimized loss that removes it entirely.

The textbook PPO-style GRPO uses an importance-weighted clipped ratio:

```
ratio_t = exp(log π_new(aₜ) − log π_old(aₜ))
loss_t  = −min(ratio_t · A, clip(ratio_t, 1−ε, 1+ε) · A)
```

This requires storing `log π_old` at rollout time — an extra forward pass per generated turn. The pipeline was:

```
generate → old-logprob forward pass → (later) new-logprob forward pass
= 3 model calls per turn
```

Here is the observation that unlocked the speedup:

> We use each rollout batch for exactly **one** optimizer step. At the moment we compute the loss, the policy has not been updated yet — so `π_old == π_current`. That means `ratio = exp(0) = 1` for every token. The clip never activates. The KL is exactly 0.

The full PPO objective collapses to a plain group-relative policy gradient:

```python
# Fast mode (default)
turn_loss = -(advantage * new_log_probs.mean())
kl = 0.0   # literally zero, not approximated
```

**3 model calls → 2 model calls per turn.** Across 8 rollouts × ~10 turns, that is ~80 avoided forward passes per training step — roughly a 33% reduction in the dominant cost, plus matching memory relief.

The PPO-style path is preserved behind `FAST_GRPO_LOGPROBS=false` for the case where rollout batches are reused for multiple optimizer epochs. That is the only regime where the ratio actually does meaningful work.

### Batching K rollouts across environment turns

This was one of the biggest architectural wins.

Previously, if `K = 4`, the trainer ran four complete multi-turn investigations **sequentially**:

```
4 rollouts × 10 turns = up to 40 model.generate() calls per prompt
```

We rewrote the rollout engine. `rollout_multi_turn_batch()` now keeps K independent environment states alive simultaneously. At each turn, it batches all live conversations into a single `model.generate()` call, then steps each environment separately with its generated action.

Finished rollouts drop out as they complete. The remaining alive rollouts continue in the next batched call.

Result:

```
10 batched generate() calls per prompt
instead of
40 sequential generate() calls
```

The crucial part: this did not change the learning data. We preserved exact generated token IDs, assistant masks, action tokens, old log-probs when needed, rollout identity, and per-environment reward traces. We only changed how efficiently we collected them.

**Observed speedup: ~2.3×**

In RL, better algorithms help, but better rollout collection often decides whether you can run the algorithm at all.

### Sequential per-turn backward

A naïve GRPO implementation would batch every turn from every rollout and keep all computation graphs in memory simultaneously.

That explodes quickly:

```
K rollouts × 10–15 turns × full prompt/action forward graphs
```

Instead, we streamed the loss:

```python
for rollout in group:
    for turn in rollout:
        logp = compute_action_logprob(turn)
        loss = -(advantage * logp) / GROUP_SIZE
        loss.backward()       # gradients accumulate in .grad
optimizer.step()              # single update per group
```

Gradients accumulate across turns and rollouts, but peak memory is only one turn's computation graph at a time. This let us fit larger groups and longer trajectories without needing bigger hardware.

---

## Experiments and What We Learned

### Experiment: full menu → compact menu → no menu

The action space includes many valid IDs — interview question IDs, exam IDs, test IDs, diagnosis IDs.

At first, every prompt included the full action menu:

| Prompt style | Tokens | Behavior |
|:---|---:|:---|
| Full menu | ~2,500 | Stable but very expensive |
| No menu | ~575 | Fast but invalid IDs appeared sometimes |
| Compact ID menu | ~900 | Stable enough and much faster |

The full menu explained every action, cost, and turnaround time. During SFT that context was useful. During GRPO it became dead weight — the model already knew what `cbc_diff` and `urine_hcg` meant.

The no-menu version was fast, but the model sometimes produced near-miss IDs like `troponin_i` instead of the exact valid ID `troponin`.

The compact menu was the best middle path:

```
INTERVIEW ids:    q_associated_symptoms, q_family_history, q_lmp, ...
EXAMINE ids:      ex_abdomen, ex_cardiac, ex_neuro, ...
ORDER_TEST ids:   beta_hcg_quant, cbc_diff, ct_abd_pelvis, ...
DIAGNOSE ids:     appendicitis, ectopic_pregnancy, stemi, ...
```

Valid IDs in context, no descriptions wasting tokens. In manual 50-step runs, `r8_format` stayed around `0.96–1.00` with the compact menu.

### Experiment: instant tests as curriculum

In the realistic environment, test results arrive after a 0–3 step delay. Clinically realistic, but hard to learn from early in training. The model orders the right test, does other things while waiting, sees the result several steps later, and connecting the result back to the ordering decision requires credit assignment over a long horizon.

`INSTANT_TESTS=true` collapses that gap — all results resolve immediately. It is not the final evaluation setting. It is an early curriculum lever. Once the model has learned *what tests to order*, we can restore realistic delays and teach planning with pending results.

### Experiment: three-level disease curriculum

We first tried binary: 5 easy diseases for the first half, then all 15. The transition was rough. The model had never seen PE or SAH, and getting thrown into the full catalog caused a noticeable reward dip before recovery.

PE and SAH are not just "more diseases." They are the hardest confusion pairs. **PE mimics STEMI. SAH mimics migraine.** Without exposure, the model learns the wrong inductive biases for the critical tier.

So we moved to three levels:

| Level | Range | Diseases | Goal |
|:---:|:---:|:---|:---|
| **L1** | 0–33% | 5 diseases, one from each tier + STEMI/anxiety pair | Format stability, basic urgency reasoning |
| **L2** | 33–67% | + PE, SAH, ovarian torsion, pneumonia, DKA | High-yield urgent/critical confusion space |
| **L3** | 67–100% | + all stable/benign | Long-tail restraint — learning when *not* to test |

**Why benign cases go last?** We wanted the model to first internalize that critical diseases require aggressive action. Adding viral gastroenteritis too early sends a confusing signal: "sometimes doing almost nothing is correct" — true, but only after the model has learned *when*. The curriculum controlled what the model was allowed to discover and in what order.

Boundaries are expressed as fractions of total training steps, so a 5-step smoke test and a 500-step production run both get the right proportions automatically.

### Experiment: adaptive disease sampling (BoundedBetaSampler)

Uniform sampling is simple but wasteful. If the model keeps failing STEMI but has mastered viral URI, equal sampling spends compute on easy wins.

But aggressive hard-case replay causes forgetting. The model overfits to the hard diseases and forgets the ones it already knew.

Our compromise: a near-uniform sampler with **bounded, performance-aware perturbations**:

```
P(disease_i) ∈ [ (1 − β) / N ,  (1 + β) / N ]
```

With `β = 0.2`, no disease can be sampled more than 20% above or below uniform. The sampler tracks an EMA of R1 accuracy per disease and slightly upweights weaker diseases within that range.

Important guardrails:

- Off by default, opt-in via `ADAPTIVE_BETA=0.2`
- Activates only after 50% of training (early steps are format learning)
- Requires at least 2 observations per disease before departing from uniform
- Uses R1 accuracy only, not total reward — prevents gaming through cost or format improvements
- Never lets a disease disappear from sampling

With `β = 0.0` it degenerates to uniform random choice — identical to the original behavior.

### Experiment: R9 calibration reward — built, tested, and removed

This is one of the best examples of engineering discipline in the project.

We wanted a calibration reward:

> Does the final differential board assign probability mass honestly?

We designed R9 using a Brier-style score over the final differential. A model should not say it is 95% sure of the wrong disease. It should be rewarded for calibrated belief, not just correct belief.

We built it fully:

- **~250 lines** of reward logic
- Brier-style scoring over the final differential board
- Multiple gates to prevent hacking: no R9 if R1 is too low, no R9 if no valid differential exists, staleness discount, argmax-consistency discount, malformed-board gate, renormalization logic
- **17 unit tests** covering edge cases
- A **7-archetype dry-run** to verify behavior before training:

| Archetype | Expected R9 | What it tested |
|:---|:---:|:---|
| Oracle | High | Good calibration baseline |
| Stale oracle | Discounted | Staleness discount activates |
| Uniform hedger | Medium | Spreading probability hedges |
| Honest uncertain | High | Uncertainty expressed correctly |
| No board | Zero-gated | Missing differential handled |
| Lucky guess | Low | Correct but poorly calibrated |
| Confidently wrong | Negative | Punish overconfident errors |

All seven archetypes behaved as expected. The reward was correct.

Then we removed it.

Why? Because under a deadline, stability mattered more than cleverness. R9 depended on R1 and differential semantics simultaneously. It added coupling, complexity, and another axis of possible reward hacking.

The five-rubric story was cleaner, more stable, and easier to explain. The capability sits in git, ready when the deadline is not the constraint.

> **A feature can be correct and still not belong in the final system.** Knowing when to cut is as important as knowing how to build.

### Experiment: group-size sweep

| Group size | Benefit | Cost |
|:---:|:---|:---|
| 4 | Fast, GPU-friendly | Noisier advantage estimates |
| 6 | Good middle ground | More expensive than 4 |
| 8 | Cleaner within-patient comparison | Higher generation cost |

GRPO needs variance inside the group. If all rollouts behave the same, advantages collapse and there is no learning signal. The sweet spot depends on GPU, prompt length, and average episode turns. Our rollout batching and fast-GRPO changes made larger groups much more realistic within a hackathon compute budget.

### Experiment: skip-SFT smoke test

We added a way to bypass SFT and train GRPO from the base model with a fresh LoRA. This was not expected to perform well. It was an engineering ablation.

It answered:

- Is the RL loop itself correct?
- Are gradients flowing to LoRA parameters?
- Is the log-prob slicing correct independently of SFT initialization?
- Can the base model parse the full menu when shown all valid IDs?
- How much does SFT help format stability specifically?

Without SFT, the model must learn JSON, valid IDs, episode termination, and diagnostic behavior all from reward alone. That is much harder. This ablation helped separate "the RL trainer is broken" from "the policy is just weak." Both diagnoses call for different fixes.

---

## Reward Hacking: The Enemy We Designed Against

GRPO is very good at finding loopholes. We planned for each one ahead of time.

### Hack 1: diagnose immediately

The model could diagnose on step 1 to avoid the timeout penalty.

**Defense:** R1 gives only 0.2 for correct-without-evidence — 5× worse than a worked case. The monitor flags when average diagnosis turn drops below 3.

### Hack 2: order everything

The model could order every test and brute-force evidence satisfaction.

**Defense:** R2 penalizes high cost. The 15-step cap rate-limits tests naturally. The monitor flags when average test count exceeds a threshold.

### Hack 3: spam differential updates

The model could repeatedly emit nearly identical differential boards to farm R6.

**Defense:** R6 counts only updates where probability shift exceeds 0.05. A model cannot farm R6 by submitting the same board twice.

### Hack 4: format drift

The model could drift away from JSON during RL and stop producing parseable actions.

**Defense:** SFT warm start, compact ID menu, R8 continuous reward, fallback parsing (`_parse_or_fallback()`), and monitor warnings.

### Hack 5: reward rises while accuracy falls

A shaped reward can be gamed even while the main task degrades. The model finds a way to score well on R2, R6, and R8 while getting diagnoses wrong.

**Defense:** Log every rubric separately. Inspect best and worst traces. Warn explicitly when total reward is positive but R1 accuracy is low.

### The monitor

We logged everything:

- R1 accuracy, R2 cost, R6 anchoring, R7 safety, R8 format
- Total reward
- Timeout rate, average diagnosis turn
- Average cost, average number of tests
- Action distribution, diagnosis distribution
- Fallback and invalid action rate

The monitor printed **text warnings** during training for suspicious patterns:

| Suspicious pattern | Possible hack |
|:---|:---|
| Diagnosis turn < 3 | Immediate-diagnosis shortcut |
| Too many tests | Order-everything strategy |
| R8 drops significantly | Format collapse |
| Zero differential updates | Skipping reasoning board |
| High timeout rate | Not committing |
| One diagnosis dominates all patients | Mode collapse |
| Reward up but accuracy low | Reward loophole |

This was one of the most valuable pieces of the entire project. It let us see not only whether the model improved, but **how** it improved — and whether the "how" was the behavior we actually wanted.

---

## What the Early Runs Showed

In a manual 50-step run, we saw encouraging signs:

- Mean total reward around **+2.4** in visible steps
- Final diagnosis correctness around **77%**
- `r8_format` around **0.96–1.00** — the format was stable
- No timeout collapse
- No immediate-diagnosis exploit
- Action rhythm looked clinically reasonable: interview → update → test → update → diagnose

But we also saw specific weak spots:

- **STEMI** confused with PE, costochondritis, and anxiety attack
- **Migraine** confused with SAH
- **Viral URI** drifted toward pneumonia, meningitis, or sepsis
- Some correct critical cases lacked full evidence
- Ectopic pregnancy sometimes became over-tested

These failures were useful. They told us the environment was doing its job. The hard cases were the clinically meaningful mimics. A system where every disease separated cleanly would be too easy to be useful.

They also motivated the next steps: adaptive sampling to focus on weak diseases, and recovery-trajectory SFT where the oracle is perturbed mid-episode to teach the model to recover from its own mistakes (deferred post-hackathon).

---

## The Playable Demo

**Live:** [huggingface.co/spaces/SnehShah/house-md-env](https://huggingface.co/spaces/SnehShah/house-md-env)

We did not want readers to only trust charts and training logs. We wanted them to *feel* the task. So we built a full interactive Hugging Face Space.

The Space is a **FastAPI + static-frontend Docker container** wrapping the same `ClinicalEnv` used for training. Every UI action is one API call. Nothing about the simulation is "demo-only."

### What you see

- **Top HUD** — Step `X/15`, running cost `$`, simulated time, severity badge, policy selector, reset button
- **Main stage** — An illustrated ER room in pure CSS. Dr. House (lab coat, stethoscope, cane) and the patient emit speech bubbles as actions fire. Three test zones (🧪 Lab, ☢ Imaging, 📈 Bedside) light up when the doctor orders tests
- **Deterioration aura** — If the patient begins deteriorating, they glow red — a visible ticking clock for the R7 penalty
- **Right rail** — Pending tests with ETA, full action log, toggleable ground-truth panel
- **Bottom controls** — `Auto` (Oracle / Greedy / Random) or `Manual`; patient selector; play / step / 1× 2× 4× speed; Kokoro TTS voice narration
- **Terminal overlay** — `CASE CLOSED` banner with final diagnosis vs. truth, cost, steps used, and the **full per-rubric reward breakdown** — the exact same `compute_all` dict the training loop sees

### Why it matters

Every reward component in the blog has a corresponding visual moment on screen:

| Reward signal | UI moment |
|:---|:---|
| R2 cost pressure | Cashier counter incrementing on every test order |
| R7 safety penalty | Patient outline turning red |
| R6 anchoring | Differential board updating mid-case |
| R8 format | Valid vs. invalid action speech bubbles |

The blog is readable. The demo is playable. The environment is trainable. All three wrap the same simulation.

If someone has only two minutes with the project, the Space is the fastest way to understand why it is different. You do not just read that the agent investigates. You watch it investigate.

---

## The Training Pipeline at a Glance

> **Note on tooling:** TRL's `GRPOTrainer` and Unsloth's RL library both assume single-turn completions. Our environment is episodic — 8–15 sequential actions per case, with per-turn observations and trajectory-level reward. No existing library supported this. We wrote the entire multi-turn GRPO training loop from scratch.

```
┌──────────────────────────────┐
│   Gemini Pro + Flash repair   │   15 disease cards × 3 variants
│                               │   Schema validation + repair loop
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Heuristic Oracle            │   225 sweep (15 × 3 × 5 seeds)
│   (polarity-aware)            │   222 / 225 kept → avg reward 2.86
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   SFT warm-start              │   2,151 (prompt, action_json) pairs
│   Gemma 3 4B-IT               │   Unsloth 4-bit LoRA, assistant-only loss
│   Unsloth LoRA                │   Teaches format + episode rhythm
└──────────────┬───────────────┘
               │
               ▼
┌══════════════════════════════╗
║  Custom Multi-Turn GRPO       ║   ← written from scratch
║  (TRL / Unsloth RL do not     ║     no library supported this
║   support episodic RL)        ║
╠══════════════════════════════╣
║  • Group-relative advantage   ║   Aᵢ = (rᵢ − mean) / std
║  • Per-turn log-prob slicing  ║   correct token attribution
║  • Batched K-rollout engine   ║   ~2.3× faster rollout collection
║  • Sequential per-turn ∇      ║   peak mem = 1 graph, not K×15
║  • Fast-GRPO loss             ║   3 model calls → 2 per turn
║  • 3-level disease curriculum ║   stable exposure schedule
║  • 5-rubric reward + monitor  ║   per-component hack detection
╚══════════════════════════════╝
```

---

## Results

### Held-out evaluation — 45 patients (15 diseases × 3 variants)

| Policy | Accuracy | Avg total reward | Avg patient cost | Notes |
|:---|---:|---:|---:|:---|
| 🎯 Heuristic Oracle | **100%** | **2.84** | — | Omniscient teacher — the ceiling |
| ☁️ Gemini Flash | **96%** | 2.22 | **$408** | Strong, but nearly equal cost to GRPO |
| 🚀 **GRPO (this work)** | **62%** | **1.97** | **$399** | **28× reward gain over base; cost-competitive with Gemini** |
| 🧱 Base Gemma 3 4B-IT | 18% | 0.07 | $227 | Rarely calls `DIAGNOSE` — mostly times out |

Three numbers stand out immediately.

**Accuracy: 18% → 62%** (+344% over base). That is not a fine-tuning bump. The base model knew the medicine. What it did not know was the game — it rarely committed to `DIAGNOSE` and timed out instead. GRPO taught it when to stop gathering and act.

**Reward: 0.07 → 1.97** (+28×). Because total reward measures accuracy, efficiency, safety, anchoring, and format together, this number reflects behavioral improvement across all five axes — not just getting the answer right.

**Cost: GRPO $399 vs. Gemini Flash $408**. A 4B model trained with R2 as an explicit efficiency signal spent essentially the same money as a frontier model with no cost pressure. But Gemini is accurate on 96% of cases vs. our 62%. That gap is real — and exactly what more training steps, larger group sizes, and process reward would close.

---

### Training curves: what the runs actually showed

The production run ran 50 steps with group size 6. The final-window summary (last 10% of each run) across the three ablation configurations:

| Run | Final group-mean reward | Final batch correct % | Final r8 format % |
|:---|---:|---:|---:|
| 10-step baseline (g=4) | 1.25 | 25% | **100%** |
| 50-step main (g=6) | **1.86** | **62%** | **100%** |
| 50-step compact menu (g=8) | 1.58 | 69% | 98% |

The compact menu run (g=8) achieved the highest batch-level correct % (69%) while using the smallest prompt. The main run (g=6) had the highest final reward, suggesting slightly better behavior beyond just getting the diagnosis right.

**Format stability (r8) stayed pinned at ≈ 1.0 throughout all runs.** This confirms the SFT warm-start did its job — the optimizer spent its entire budget on clinical behavior, never on syntax recovery.

**KL ≈ 0 throughout the production run.** This is not a coincidence — it is the Fast-GRPO proof working correctly. In single-update mode, the old policy and the current policy are identical at loss computation time, so `ratio = exp(0) = 1` and KL collapses to zero. The flat KL line is empirical confirmation that the math was right.

**Policy-gradient loss bounded in ±0.01 with no runaway.** The loss oscillates around zero as expected — positive on steps where the group mean falls and negative on steps where the policy is pushed toward high-reward trajectories. No divergence, no gradient explosion.

---

### Per-rubric curves across the 50-step production run

| Rubric | Observed behavior | What it means |
|:---|:---|:---|
| **r1 accuracy** | Starts ~0.8, fluctuates, ends ~0.65 | Drops visibly at curriculum transitions (L1→L2 at step 16, L2→L3 at step 34) — harder diseases are harder. Expected. |
| **r2 cost** | Starts ~1.0, gradually drifts to ~0.7 | Model learns to order more tests as it gains confidence. Still in the reward-positive zone. |
| **r6 anchoring** | Stable ~0.58–0.60 throughout | Consistent differential updating. No spamming, no skipping. |
| **r7 safety** | Starts ~−0.4, rises to ~−0.1, then back to ~−0.4 | Improves initially as the model learns urgency; dips again at L2 introduction where PE and SAH appear. New hard diseases = new safety errors. |
| **r8 format** | Pinned at 0.97–1.00 | SFT warm-start working. Never a concern during training. |

The curriculum transition at step 16 (L1→L2) is clearly visible as a step-down in r1 and r7 before recovery. This is the expected behavior — it signals the environment is doing its job, not that the model is failing.

---

### Action distribution: no mode collapse, no shortcuts

Throughout all 50 steps of the production run, the action mix was stable and clinically sensible:

| Action | Share (rolling avg) | Interpretation |
|:---|---:|:---|
| `INTERVIEW` | ~40% | Dominant — model asks questions |
| `UPDATE_DIFFERENTIAL` | ~20% | Consistent hypothesis updating |
| `ORDER_TEST` | ~18–20% | Testing at the right rate |
| `EXAMINE` | ~10% | Physical exam as part of workup |
| `DIAGNOSE` | ~10% | Committing when ready |

No single action dominated. The model did not collapse into interview-only, test-everything, or immediate-diagnose mode. Diagnostic diversity (unique diagnoses per batch) rose from ~2 at the start to ~3–5 mid-run, with mode-collapse fraction tracking no systematic collapse toward a single disease.

---

### Per-disease breakdown

GRPO improved over the base model on **11 of 15 diseases**. When including the SFT warm-start, **13 of 15 diseases** showed improvement over base.

| Disease | Base | SFT | GRPO | Gemini | Notable |
|:---|:---:|:---:|:---:|:---:|:---|
| Pulmonary embolism | 0% | ~33% | **100%** | 100% | GRPO matches Gemini ceiling |
| Viral URI | 0% | ~66% | **100%** | 100% | GRPO matches Gemini ceiling |
| Pneumonia | 0% | ~66% | **100%** | 100% | GRPO matches Gemini ceiling |
| Migraine | ~66% | ~33% | ~66% | ~66% | SFT hurt; GRPO recovered |
| Ectopic pregnancy | 0% | ~66% | ~66% | ~66% | SFT unlocked, GRPO held |
| Appendicitis | ~33% | ~33% | ~66% | ~66% | GRPO doubled base |
| Subarachnoid hemorrhage | ~33% | ~66% | ~66% | 100% | Hard mimic; gap to Gemini remains |
| Viral gastroenteritis | ~33% | ~66% | ~66% | 100% | Stable but gap remains |
| DKA | ~33% | ~33% | ~66% | 100% | GRPO doubled base |
| Sepsis / UTI | 0% | ~66% | ~66% | 100% | SFT critical for unlocking |
| Costochondritis | 0% | ~33% | ~33% | 100% | Hard benign mimic — known weak spot |

Three diseases where GRPO hit **100%** — matching Gemini Flash exactly — were pulmonary embolism, viral URI, and pneumonia. These are cases where the evidence path is clear once the model learns the workup rhythm.

The two consistent weak spots: **costochondritis** (hard to distinguish from cardiac causes without expensive testing) and **subarachnoid hemorrhage** (easily confused with migraine in early workup). Both are on the adaptive sampling target list for the next training run.

The base model's 18% overall accuracy was not uniform failure — it scored 66% on migraine from the very start (textbook presentation, common) while scoring 0% on pulmonary embolism (requires specific test ordering and pattern recognition across multiple results). That pattern confirms the task design: easy diseases are easy from pretraining alone; hard diseases require the RL training loop.

---

> **A note on training scale.** We ran multiple experiments sweeping hyperparameters — group size (4, 6, 8), prompt style (full menu, compact menu, no menu), curriculum speed, instant-test toggles, and fast-GRPO vs. PPO-style loss. Across all of them, the maximum we could reach within the hackathon compute window was **50 training steps**. Every number above — the 62% accuracy, the 1.97 reward, the per-disease breakdown — comes from a 50-step run. That is a very early-stage policy. The reward curves had not plateaued. The per-rubric curves were still moving. These results are a proof-of-concept, not a ceiling.

> The bet the project is built on: a 4B model with the right reward signal should out-discipline a frontier model at being *appropriately cautious*. The cost numbers ($399 vs $408) suggest that discipline is already there at 50 steps. Closing the accuracy gap from 62% to 90%+ is not a research question — it is a compute question. The infrastructure to get there is already built.

---

## What We'd Do With More Time

**Process reward from the oracle plan** — The highest-ROI idea we did not ship. Give small per-step credit when the model's next action matches a remaining oracle step. Would make reward curves start climbing from step ~5 instead of ~40. Deferred because it introduces a new axis of reward instability — the oracle is a good teacher, not the only valid doctor.

**Multi-epoch PPO reuse** — Reuse each rollout batch for 2–4 optimizer epochs. Turns the ratio clip and KL into meaningful stabilizers. Requires `FAST_GRPO_LOGPROBS=false` and a KL penalty term.

**KL anchor to SFT checkpoint** — `KL_COEF × KL(π_current ‖ π_sft)` prevents late-training drift when a reward-hacking exploit starts to dominate.

**Recovery-trajectory SFT** — The oracle always plays perfectly, so the model never learns to recover from its own mistakes. Perturbing the oracle mid-episode and re-optimizing would close the learned-from-perfect-demonstrations gap.

**Parse-gated dynamic patient (PGDP)** — We designed but did not fully train with a two-stage generative channel: a Patient LLM generates natural-language utterances from structured ground truth; a Parser LLM converts those utterances back into structured polarity (present / absent / unclear). The reward engine reads the parser output, not raw text. This makes the environment more realistic without letting the model exploit raw-text quirks. The architecture, anti-hacking guards, and SFT data plan for the parser are fully designed and in the codebase. Full training deferred.

**Per-cell evaluation** — The 45-patient eval covers 9 cells (3 variant types × 3 severity tiers). The interesting story is whether GRPO improves uniformly or concentrates gains on easy cases.

---

## The Engineering Equation

The cost of multi-turn RL is multiplicative. We attacked nearly every factor:

```
training_steps
× group_size          ← sequential backward kept this tractable
× turns_per_episode   ← instant-test curriculum shortened early episodes
× prompt_tokens       ← compact menu: 2,500 → 900 tokens
× forward_passes      ← fast-GRPO: 3 → 2 calls per turn
  per_turn
```

And we improved rollout collection orthogonally:

```
sequential 40 generate() calls per prompt
→ batched 10 generate() calls per prompt
= ~2.3× faster rollout collection
```

This is why the project was not just "we trained a model." We built the environment, the reward, the data, the trainer, the monitor, the UI, and the speed path that made training possible within a hackathon window.

To be specific about what "the trainer" means: TRL and Unsloth RL do not support multi-turn episodic GRPO. We wrote every part of the training loop ourselves — per-turn log-prob slicing, batched multi-environment rollout collection, sequential per-turn backward, and the optimized single-update GRPO loss. That is not a library call. That is a custom RL training stack.

---

## The Final Story

We started with a simple question: can a 4B model think like a doctor?

By the end, the better question was: can we build an environment where thinking carefully is the **only way to win**?

A lazy model guesses on step 1 and loses evidence reward.
A panicked model orders everything and loses cost reward.
A careless model times out on STEMI and gets hit by the safety penalty.
A messy model emits broken JSON and loses format reward.
A rigid model anchors on its first hypothesis and misses differential-update reward.

None of those shortcuts work. The reward landscape was designed so that the target behavior — ask, examine, test, reconsider, diagnose, stop — is the behavior that scores highest.

That is why this project feels different. It does not just ask whether a model knows the answer. It asks whether the model can **earn** the answer.

And it turns out — with the right environment, the right reward, and the right training pipeline — a 4B open model can.

---

*Built for the Meta OpenEnv Hackathon · April 2026*

[Live Demo](https://huggingface.co/spaces/SnehShah/house-md-env) · [Codebase](../)
