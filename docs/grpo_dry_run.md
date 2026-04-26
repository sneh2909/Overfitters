# GRPO Dry Run — One Rollout, 5 Turns

**Disease:** `appendicitis` (severity: urgent)  
**Variant:** `v2`  
**Seed:** `471832`  
**Group position:** rollout 3 of 8 (same patient played 8 times per step)

---

## Pre-rollout: patient setup (main loop, `train_grpo.py:712`)

```python
disease    = "appendicitis"   # rng.choice(LEVEL_1_DISEASES)
variant_id = "v2"             # rng.choice(["v1","v2","v3"])
seed       = 471832           # rng.randint(1000, 999999)
```

---

## `env.reset("appendicitis", variant_id="v2", seed=471832)`

**What it does:** initialises a hidden patient state and returns the first `Observation`.

**Output — `obs` (Observation object):**
```
obs.age              = 24
obs.sex              = "F"
obs.chief_complaint  = "12 hours of worsening right lower quadrant pain, nausea"
obs.intake_vitals    = "HR 101, BP 118/72, Temp 38.1°C, RR 17, SpO2 99%"
obs.severity_signal  = "urgent"
obs.step             = 0
obs.step_cap         = 15
obs.cost_so_far      = 0
obs.time_elapsed_min = 0
obs.action_log       = []
obs.pending_tests    = []
obs.differential_board = []
obs.terminal         = False

# hidden (not visible to agent):
hidden.true_disease  = "appendicitis"
hidden.deteriorating = False
hidden.test_results  = {"cbc_diff": ("WBC 14.2 K/uL", "H"),
                         "ct_abd_pelvis": ("Enlarged appendix 9mm, periappendiceal fat stranding", "H"),
                         "urine_hcg": ("Negative", "N"), ...}
```

---

## Turn 1

### 1a. `render_prompt(obs, catalogs, include_menu=False)`

**Input:** `obs` above (step=0, empty history)

**Output — `prompt_text` (string, ~300 tokens):**
```
You are a clinical reasoning agent in an emergency department simulation.
Your goal is to diagnose the patient correctly within 15 steps, balancing
speed, cost, and safety. ...

===== PATIENT INTAKE =====
Demographics: 24yo F
Chief complaint: 12 hours of worsening right lower quadrant pain, nausea
Initial vitals: HR 101, BP 118/72, Temp 38.1°C, RR 17, SpO2 99%

===== HISTORY =====
(none — first action of the episode)

===== STATUS =====
Step: 0/15    Cost so far: $0    Time elapsed: 0min    Severity signal: urgent
Pending tests: none
Current differential: (not yet set — emit UPDATE_DIFFERENTIAL when ready)

===== YOUR TURN =====
Output ONE JSON action object now.
```

### 1b. Tokenise

```python
messages   = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                           return_tensors="pt",
                                           add_generation_prompt=True)[0].cpu()
```

**Output:**
```
prompt_ids.shape = torch.Size([287])   # 287 tokens, on CPU
```

### 1c. `model.generate(...)` — no gradient, temperature=0.9

```python
output = model.generate(
    prompt_ids.unsqueeze(0).to("cuda"),
    max_new_tokens=150,
    temperature=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
action_ids = output[0][287:].cpu()
```

**Output:**
```
action_ids.shape = torch.Size([22])    # 22 new tokens

decoded text:
'{"type": "INTERVIEW", "argument": "q_pain_character",
  "rationale": "Characterise the pain onset and quality to differentiate appendicitis from other RLQ causes."}'
```

### 1d. `_compute_log_probs(model, prompt_ids, action_ids, requires_grad=False)`

**What it does:** one forward pass at temperature=1 (raw logits), slices
`logits[286 : 286+22]`, runs log_softmax, gathers the log-prob of each
actual action token.

```python
full_ids = cat([prompt_ids, action_ids]).unsqueeze(0).to("cuda")
# shape [1, 309]

logits = model(full_ids).logits[0]          # [309, vocab_size=262144]
action_logits = logits[286 : 286+22]        # [22, 262144]  ← slice
log_probs_all = log_softmax(action_logits)  # [22, 262144]
old_log_probs = log_probs_all.gather(1, action_ids.to("cuda").unsqueeze(1)).squeeze(1)
```

**Output:**
```
old_log_probs (shape [22], on CPU):
tensor([-0.31, -1.12, -0.08, -2.44, -0.19, -0.97, -1.63, -0.22,
        -0.41, -0.88, -0.14, -0.37, -0.55, -1.20, -0.09, -0.44,
        -0.61, -0.79, -0.22, -0.33, -0.18, -0.52])
mean = -0.54   # comfortable confidence on a well-formed JSON token sequence
```

### 1e. Parse & `env.step(action)`

```python
text   = '{"type": "INTERVIEW", "argument": "q_pain_character", ...}'
action = parse_action_json(text)
# → Action(type=INTERVIEW, argument="q_pain_character", rationale="...")

obs = env.step(action)
```

**`turns` after turn 1:**
```python
turns = [
    (prompt_ids_t1,   # shape [287]
     action_ids_t1,   # shape [22]
     old_log_probs_t1)  # shape [22]
]
```

**New `obs`:**
```
obs.step         = 1
obs.action_log   = [LogEntry(step=1, kind="action", text="q_pain_character ..."),
                    LogEntry(step=1, kind="result", text="Sharp, constant pain, started periumbilically then migrated to RLQ over 8h")]
obs.terminal     = False
obs.cost_so_far  = 0      # INTERVIEW is free
obs.time_elapsed = 1      # 1 min
```

---

## Turn 2

### 2a. `render_prompt(obs, catalogs, include_menu=False)`

**Input:** obs with 1 completed step

**Output — `prompt_text` (now includes history, ~360 tokens):**
```
...
===== HISTORY =====
[ step  1] INTERVIEW q_pain_character
  Sharp, constant pain, started periumbilically then migrated to RLQ over 8h

===== STATUS =====
Step: 1/15    Cost so far: $0    Time elapsed: 1min    Severity signal: urgent
Pending tests: none
Current differential: (not yet set — emit UPDATE_DIFFERENTIAL when ready)
...
```

### 2b–c. Tokenise → generate

```
prompt_ids.shape  = torch.Size([341])

decoded action text:
'{"type": "EXAMINE", "argument": "ex_mcburney",
  "rationale": "Classic McBurney point tenderness is highly specific for appendicitis."}'

action_ids.shape  = torch.Size([19])
```

### 2d. `_compute_log_probs`

```
old_log_probs_t2 (shape [19]):
tensor([-0.28, -0.67, -0.11, -1.89, -0.22, -0.54, -0.44, -0.31,
        -0.17, -0.83, -0.19, -0.38, -0.62, -0.27, -0.41, -0.55,
        -0.22, -0.34, -0.44])
mean = -0.44
```

### 2e. `env.step(action)`

```
Action: EXAMINE ex_mcburney   (cost $15)

New obs:
  step=2, cost_so_far=$15, time_elapsed=3min
  action_log gains two entries:
    [step 2] EXAMINE ex_mcburney ($15)
      Rebound tenderness at McBurney's point. Rovsing sign positive.
    [step 2] → Rebound tenderness at McBurney's point. Rovsing sign positive.
```

---

## Turn 3

### 3a. `render_prompt` (~430 tokens)

```
===== HISTORY =====
[ step  1] INTERVIEW q_pain_character
  Sharp, constant pain, started periumbilically then migrated to RLQ over 8h
[ step  2] EXAMINE ex_mcburney ($15)
  Rebound tenderness at McBurney's point. Rovsing sign positive.

===== STATUS =====
Step: 2/15    Cost so far: $15    Time elapsed: 3min    Severity signal: urgent
Pending tests: none
Current differential: (not yet set — emit UPDATE_DIFFERENTIAL when ready)
```

### 3b–c. generate

```
decoded action text:
'{"type": "UPDATE_DIFFERENTIAL",
  "argument": "appendicitis most likely given migrated RLQ pain + McBurney tenderness",
  "rationale": "Strong clinical picture for appendicitis; must also consider ovarian pathology.",
  "board": [{"disease": "appendicitis", "prob": 0.70},
            {"disease": "ovarian_torsion", "prob": 0.15},
            {"disease": "ectopic_pregnancy", "prob": 0.10},
            {"disease": "viral_uri", "prob": 0.05}]}'

action_ids.shape = torch.Size([61])   # longer — board field adds tokens
```

### 3d. `_compute_log_probs`

```
old_log_probs_t3 (shape [61]):
tensor([-0.29, -0.44, -0.13, ... -0.68, -0.31])
mean = -0.51
```

### 3e. `env.step`

```
Action: UPDATE_DIFFERENTIAL (free, instant)

New obs:
  step=3, cost_so_far=$15, time_elapsed=3min
  differential_board = [{"disease":"appendicitis","prob":0.70},
                        {"disease":"ovarian_torsion","prob":0.15}, ...]
  action_log gains one entry:
    [step 3] UPDATE_DIFFERENTIAL — appendicitis most likely given migrated RLQ pain + McBurney tenderness
```

---

## Turn 4

### 4a. `render_prompt` (~500 tokens)

```
===== HISTORY =====
[ step  1] INTERVIEW q_pain_character
  Sharp, constant pain, started periumbilically then migrated to RLQ over 8h
[ step  2] EXAMINE ex_mcburney ($15)
  Rebound tenderness at McBurney's point. Rovsing sign positive.
[ step  3] UPDATE_DIFFERENTIAL — appendicitis most likely given migrated RLQ pain + McBurney tenderness

===== STATUS =====
Step: 3/15    Cost so far: $15    Time elapsed: 3min    Severity signal: urgent
Pending tests: none
Current differential (top 5): appendicitis(0.70), ovarian_torsion(0.15), ectopic_pregnancy(0.10), viral_uri(0.05)
```

### 4b–c. generate

```
decoded action text:
'{"type": "ORDER_TEST", "argument": "cbc_diff",
  "rationale": "WBC count will support or refute inflammatory process; low cost, fast turnaround."}'

action_ids.shape = torch.Size([24])
```

### 4d. `_compute_log_probs`

```
old_log_probs_t4 (shape [24]):
tensor([-0.33, -0.55, -0.12, -1.67, -0.22, -0.41, -0.49, -0.18,
        -0.37, -0.71, -0.29, -0.44, -0.63, -0.27, -0.38, -0.55,
        -0.22, -0.34, -0.19, -0.61, -0.28, -0.43, -0.31, -0.55])
mean = -0.42
```

### 4e. `env.step`

```
Action: ORDER_TEST cbc_diff   (cost $45, turnaround 1 step)

New obs:
  step=4, cost_so_far=$60, time_elapsed=5min
  pending_tests = [PendingTest(test_id="cbc_diff", deliver_at_step=5)]
  action_log gains:
    [step 4] ORDER_TEST cbc_diff ($45)
```

---

## Turn 5

### 5a. `render_prompt` (~570 tokens)

```
===== HISTORY =====
[ step  1] INTERVIEW q_pain_character
  Sharp, constant pain, started periumbilically then migrated to RLQ over 8h
[ step  2] EXAMINE ex_mcburney ($15)
  Rebound tenderness at McBurney's point. Rovsing sign positive.
[ step  3] UPDATE_DIFFERENTIAL — appendicitis most likely given migrated RLQ pain + McBurney tenderness
[ step  4] ORDER_TEST cbc_diff ($45)
[ step  4] → Result pending (delivers at step 5)

===== STATUS =====
Step: 4/15    Cost so far: $60    Time elapsed: 5min    Severity signal: urgent
Pending tests: cbc_diff (1-step wait)
Current differential (top 5): appendicitis(0.70), ovarian_torsion(0.15), ectopic_pregnancy(0.10), viral_uri(0.05)
```

### 5b–c. generate

The model waits for the lab result, then after env auto-delivers it at the
start of step 5 the prompt will show the CBC result. The model chooses to
diagnose now (the SFT policy learned to close when evidence is sufficient):

```
decoded action text:
'{"type": "DIAGNOSE", "argument": "appendicitis",
  "rationale": "Classic migrated RLQ pain, McBurney tenderness, leukocytosis WBC 14.2 — appendicitis confirmed."}'

action_ids.shape = torch.Size([28])
```

### 5d. `_compute_log_probs`

```
old_log_probs_t5 (shape [28]):
tensor([-0.19, -0.41, -0.09, -1.33, -0.28, -0.54, -0.37, -0.23,
        -0.44, -0.67, -0.18, -0.33, -0.51, -0.29, -0.42, -0.38,
        -0.25, -0.56, -0.34, -0.47, -0.21, -0.39, -0.28, -0.44,
        -0.33, -0.19, -0.52, -0.41])
mean = -0.37   # model is fairly confident — DIAGNOSE token is high-prob
```

### 5e. `env.step`

```
Action: DIAGNOSE appendicitis

New obs:
  terminal         = True
  diagnosis        = "appendicitis"
  step             = 5
  cost_so_far      = $60
  timed_out        = False
```

`play_episode` exits the while loop and returns:
```python
turns = [
    (prompt_ids_t1[287], action_ids_t1[22], old_lp_t1[22]),  # INTERVIEW
    (prompt_ids_t2[341], action_ids_t2[19], old_lp_t2[19]),  # EXAMINE
    (prompt_ids_t3[387], action_ids_t3[61], old_lp_t3[61]),  # UPDATE_DIFF
    (prompt_ids_t4[464], action_ids_t4[24], old_lp_t4[24]),  # ORDER_TEST
    (prompt_ids_t5[512], action_ids_t5[28], old_lp_t5[28]),  # DIAGNOSE
]
final_obs.terminal = True
episode            = env._episode
```

---

## Reward computation — `compute_all(episode, cards["appendicitis"], catalogs)`

```python
# R1 — accuracy
# correct dx=True; min_evidence_set requires cbc_diff (abnormal)
# cbc_diff was ordered AND its flag="H" (WBC 14.2) → "abnormal" satisfied
r1 = 1.0    # full credit: correct dx + all evidence groups satisfied

# R2 — cost
# cost_so_far = $60 → below the $200 sweet-spot lower edge
# piecewise: 0.2 + 0.8 * (60/200) = 0.2 + 0.24 = 0.44
r2 = 0.44

# R6 — anchoring
# one UPDATE_DIFFERENTIAL with a well-formed board → meaningful=1
# score = min(1.0, 0.3 * 1) = 0.3
r6 = 0.3

# R7 — safety
# not timed_out, correct dx, not deteriorating → 0.0
r7 = 0.0

# R8 — format
# 5 actions, 0 invalid → 5/5 = 1.0
r8 = 1.0

# Weighted total (DEFAULT_WEIGHTS)
# total = 2.0*1.0 + 0.5*0.44 + 0.3*0.3 + 1.0*0.0 + 0.5*1.0
#       = 2.00 + 0.22 + 0.09 + 0.00 + 0.50
total = 2.81
```

**This rollout's reward:** `2.81`

---

## After all 8 rollouts complete — advantage normalisation

```python
rewards = [2.81, 1.20, 3.10, 2.44, 0.50, 2.93, 1.80, 3.05]

mean_r = 2.23
std_r  = 0.86    # max(pstdev, 1e-6) — safe even if all identical

advantages = [(r - 2.23) / 0.86 for r in rewards]
#           = [+0.67, -1.20, +1.01, +0.24, -2.01, +0.81, -0.50, +0.95]

# This rollout (index 0, reward=2.81) → advantage = +0.67
```

---

## GRPO loss for THIS rollout (inside `compute_grpo_loss`, advantage=+0.67)

Repeated for each of the 5 turns. Shown for **turn 1** (INTERVIEW):

```python
# NEW forward pass WITH gradient (policy may have shifted from 8 earlier rollouts)
new_log_probs = _compute_log_probs(model, prompt_ids_t1, action_ids_t1,
                                   requires_grad=True)
# tensor([-0.29, -1.10, -0.09, -2.41, ...])  shape [22]

old_log_probs = old_lp_t1.to("cuda")
# tensor([-0.31, -1.12, -0.08, -2.44, ...])  shape [22]

# ratio: how much has policy changed for each token?
ratio = exp(new_log_probs - old_log_probs)
# ≈ tensor([1.02, 1.02, 0.99, 1.03, ...])  — small change; first rollout in the batch

# clipped ratio
clipped_ratio = ratio.clamp(0.8, 1.2)   # ε=0.2; same as ratio here (within bounds)

adv = tensor(0.67)

loss_per_token = -min(ratio * 0.67, clipped_ratio * 0.67)
# ≈ tensor([-0.684, -0.683, -0.663, -0.691, ...])
# all negative → increasing these token probs reduces loss → correct direction

rollout_loss (sum over 5 turns) = ~3.82

# divide by GROUP_SIZE=8 before backward so gradient is an average not a sum
(rollout_loss / 8).backward()   # 3.82/8 = 0.478
```

For **rollout 5** (reward=0.50, advantage=-2.01):
```python
# same math, but advantage is negative
loss_per_token = -min(ratio * (-2.01), clipped_ratio * (-2.01))
# ≈ tensor([+2.05, +2.04, ...])   positive → these tokens' probs will decrease
```

After all 8 rollouts have called `.backward()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
optimizer.step()   # PagedAdamW8bit: one Adam update using accumulated gradients
optimizer.zero_grad()
```

**Net effect of this step:** the model's LoRA weights shift so that the
5-turn trajectory (INTERVIEW → EXAMINE → UPDATE_DIFF → ORDER_LAB → DIAGNOSE)
with correct reasoning becomes more probable, while the low-scoring rollouts
(e.g. immediate wrong DIAGNOSE) become less probable.
