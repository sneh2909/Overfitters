"""
Reward rubrics for the diagnostic-reasoning RL environment.

Five rubrics that score one episode's trajectory:
  R1 accuracy   — correct disease + minimum evidence set satisfied
  R2 cost       — efficient workup; rewards $200–500 zone, penalizes $1500+
  R6 anchoring  — agent revised its differential as evidence accumulated
  R7 safety     — penalty for timeouts and wrong dx (severity-scaled)
  R8 format     — fraction of actions that were validly formatted

Each rubric is a PURE function of (Episode [+ card / catalogs]) → float.
`compute_all(...)` returns a dict with all five plus a weighted total.

Why pure functions, no shared state:
  - GRPO calls these once per rollout at episode end. Pure means trivially
    unit-testable on hand-crafted Episodes.
  - Swappable: when we ablate (e.g. drop R7 to study its effect on training),
    we can null out one weight without touching the others.
  - Reproducibility: identical (Episode, card, catalogs) inputs always yield
    identical scores, so reward bugs stay local.

Numbers were tuned to give the "play correctly" path ~+2.5 vs the "always
diagnose immediately" path ~+1.0 — see test_rewards.py for worked examples.
The trainer (E5) can re-tune weights via `compute_all(weights=...)`.
"""

from __future__ import annotations

from typing import Any, Optional

from .env.catalogs import Catalogs
from .env.state import ActionType, Episode


# ---------------------------------------------------------------------------
# Default weights for the composite total.
# Correctness dominates (R1 weight 2.0). Safety and cost are roughly equal
# secondary signals. Format is a steady-state nudge, not a primary objective.
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "r1_accuracy": 2.0,
    "r2_cost": 0.5,
    "r6_anchoring": 0.3,
    "r7_safety": 1.0,
    "r8_format": 0.5,
}


# ===========================================================================
# R1 — Accuracy
# ===========================================================================

def r1_accuracy(episode: Episode, card: dict[str, Any]) -> float:
    """Correct disease AND saw the evidence required to know it.

    Score:
      - 1.0  correct dx, ALL min_evidence groups satisfied
      - 0.5  correct dx, SOME groups satisfied (partial credit)
      - 0.2  correct dx, NO groups satisfied (lucky guess — discourages
             premature DIAGNOSE before any tests)
      - 0.0  wrong dx, or no dx at all (timeout)

    A min_evidence group looks like:
        {"any_of": ["beta_hcg_quant","urine_pregnancy_qualitative"],
         "require": "abnormal"}

    A group is satisfied iff at least one test in `any_of` was ordered AND
    its sampled flag matches `require`:
      - require="abnormal"  → flag in {"H","L","CRIT"}  (anything but "N")
      - require="normal"    → flag == "N"
    """
    obs = episode.obs
    hidden = episode.hidden

    if obs.diagnosis != hidden.true_disease:
        return 0.0

    min_evidence = card.get("minimum_evidence_set") or []
    if not min_evidence:
        # Card requires no evidence (very rare). Correct dx is fully sufficient.
        return 1.0

    # Collect the test_ids the agent actually ordered (skip invalid actions).
    ordered = {
        e.action.argument
        for e in obs.action_log
        if e.kind == "action"
        and e.action is not None
        and e.action.type == ActionType.ORDER_TEST
        and not e.invalid
    }

    satisfied = 0
    for group in min_evidence:
        any_of = group.get("any_of", [])
        require = group.get("require", "abnormal")
        # Group satisfied iff at least one listed test was ordered AND its
        # flag matches the required polarity.
        for tid in any_of:
            if tid not in ordered:
                continue
            _, flag = hidden.test_results.get(tid, ("", "N"))
            ok = (require == "abnormal" and flag != "N") or (
                require == "normal" and flag == "N"
            )
            if ok:
                satisfied += 1
                break  # don't double-count within the same group

    if satisfied == len(min_evidence):
        return 1.0
    if satisfied > 0:
        return 0.5
    return 0.2  # correct guess but un-justified


# ===========================================================================
# R2 — Cost efficiency
# ===========================================================================

def r2_cost(episode: Episode) -> float:
    """Reward for hitting the cost-efficient sweet spot, penalty for over-
    or under-testing.

    Piecewise (cost in USD):
        $0     → 0.2   (under-test penalty: didn't even rule out red flags)
        $200   → 1.0   (sweet spot lower edge)
        $500   → 1.0   (sweet spot upper edge)
        $1000  → 0.5   (overtest, modest penalty)
        $1500  → 0.0
        $3000+ → -0.5  (capped: stops "order every imaging study" cheats)

    Why $200 not $0 is the peak: even benign cases require ~$150–250 of
    rule-out tests (see viral_uri's CXR, anxiety_attack's ECG+troponin in
    the disease cards). Rewarding $0 would push the agent toward DIAGNOSE-
    immediately behavior, which R1 already partly catches but R2 should
    reinforce.
    """
    cost = episode.obs.cost_so_far

    if cost < 200:
        # Linear ramp 0.2 → 1.0
        return 0.2 + 0.8 * (cost / 200.0)
    if cost <= 500:
        return 1.0
    if cost <= 1500:
        # Linear ramp 1.0 → 0.0
        return 1.0 - (cost - 500) / 1000.0
    if cost <= 3000:
        # Linear ramp 0.0 → -0.5
        return -0.5 * (cost - 1500) / 1500.0
    return -0.5


# ===========================================================================
# R6 — Anchoring resistance
# ===========================================================================

def r6_anchoring(episode: Episode) -> float:
    """Did the agent revise its differential as new evidence came in?

    Counts UPDATE_DIFFERENTIAL events with MEANINGFUL probability movement
    from the previous board (>0.05 max-disease delta).

    Score = min(1.0, 0.3 * meaningful_count):
        0 boards            → 0.0   (no metacognition shown)
        1 board             → 0.3   (one stake in the ground)
        2 boards (real shift) → 0.6
        3 boards (shifts)   → 0.9
        4+ boards (shifts)  → 1.0  (capped)

    Why a shift threshold: without it, the agent can spam UPDATE_DIFFERENTIAL
    with the same board to farm the rubric. The 0.05 floor means probabilities
    must actually move on at least one disease to count.
    """
    boards: list[list[dict[str, Any]]] = []
    for entry in episode.obs.action_log:
        if (
            entry.kind == "action"
            and entry.action is not None
            and entry.action.type == ActionType.UPDATE_DIFFERENTIAL
            and not entry.invalid
            and entry.action.board
        ):
            boards.append(entry.action.board)

    if not boards:
        return 0.0

    # First board always counts as 1 (it's the agent's first commit to a hypothesis).
    meaningful = 1
    for prev, curr in zip(boards, boards[1:]):
        prev_probs = {
            e["disease"]: float(e.get("prob", 0.0))
            for e in prev
            if isinstance(e, dict) and "disease" in e
        }
        curr_probs = {
            e["disease"]: float(e.get("prob", 0.0))
            for e in curr
            if isinstance(e, dict) and "disease" in e
        }
        diseases = set(prev_probs) | set(curr_probs)
        if not diseases:
            continue
        max_diff = max(
            abs(prev_probs.get(d, 0.0) - curr_probs.get(d, 0.0)) for d in diseases
        )
        if max_diff > 0.05:
            meaningful += 1

    return min(1.0, 0.3 * meaningful)


# ===========================================================================
# R7 — Safety
# ===========================================================================

def r7_safety(episode: Episode, catalogs: Catalogs) -> float:
    """Penalty-only rubric. Returns ≤ 0 always. Catches dangerous outcomes:
    timeouts (especially on critical patients) and confidently-wrong diagnoses.

    Scaling matrix:

                           critical  urgent   stable
        timed_out (no dx):  -2.0     -1.0    -0.5
        wrong dx:           -1.0     -0.5    -0.2
        correct + deterio:                          -0.2 (got there late)
        correct + stable:                            0.0 (no harm)

    Why penalty-only: R1 already rewards correctness. R7's job is to signal
    "this trajectory was DANGEROUS". A reward-shaped safety term would muddy
    the signal — we want safety violations to feel like a hard floor, not a
    bonus to chase.
    """
    obs = episode.obs
    hidden = episode.hidden
    correct = obs.diagnosis == hidden.true_disease
    severity = catalogs.diseases[hidden.true_disease].severity

    if obs.timed_out:
        if severity == "critical":
            return -2.0
        if severity == "urgent":
            return -1.0
        return -0.5

    if not correct:
        if severity == "critical":
            return -1.0
        if severity == "urgent":
            return -0.5
        return -0.2

    # Correct dx — did the patient deteriorate while we worked?
    if hidden.deteriorating:
        return -0.2

    return 0.0  # no harm done


# ===========================================================================
# R8 — Format compliance
# ===========================================================================

def r8_format(episode: Episode) -> float:
    """Fraction of agent action entries that parsed validly.

    "Validly" = in-vocab argument (known qid/eid/tid/disease) AND, for
    UPDATE_DIFFERENTIAL, a well-formed board (dicts with disease+prob,
    probs ∈ [0,1] summing near 1, only known disease ids).

    Result entries (kind="result") are excluded — they're env-emitted, not
    agent-emitted, so always valid by construction.

    An episode with zero actions returns 0.0. This shouldn't happen in
    practice (the agent must emit something), but it's the natural floor.
    """
    actions = [e for e in episode.obs.action_log if e.kind == "action"]
    if not actions:
        return 0.0
    valid = sum(1 for e in actions if not e.invalid)
    return valid / len(actions)


# ===========================================================================
# Composite — runs all five and returns dict + weighted total
# ===========================================================================

def compute_all(
    episode: Episode,
    card: dict[str, Any],
    catalogs: Catalogs,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Run all five rubrics. Return a dict with each score plus a weighted
    `total`.

    Returning a dict (rather than a tuple or single number) lets the trainer:
      - Log per-rubric values to wandb without re-running.
      - Spot-check ablations: "trained model dropped R6 — is that good?"
      - Compute episode-level diagnostics for the demo plots in E7.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS

    rewards: dict[str, float] = {
        "r1_accuracy": r1_accuracy(episode, card),
        "r2_cost": r2_cost(episode),
        "r6_anchoring": r6_anchoring(episode),
        "r7_safety": r7_safety(episode, catalogs),
        "r8_format": r8_format(episode),
    }
    rewards["total"] = sum(w[k] * rewards[k] for k in rewards if k in w)
    return rewards
