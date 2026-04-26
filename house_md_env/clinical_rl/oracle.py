"""
Heuristic oracle policy — produces high-quality demonstration trajectories
for SFT (E4) without using an LLM.

The oracle "cheats": it has access to the disease card (ground truth) so it
knows exactly which tests satisfy `minimum_evidence_set`. This is fine — the
oracle is a TEACHER, not a competitor. The model learning from these
trajectories does NOT see disease identity in the prompt; it only sees the
chief complaint, age/sex, vitals, and accumulated history.

What the oracle produces (per episode):
  Phase 1: HPI interviews (2-3 questions, family-aware)
  Phase 2: Initial UPDATE_DIFFERENTIAL with a plausible spread
  Phase 3: Targeted EXAMINE (1 maneuver, family-aware)
  Phase 4: ORDER_TESTs covering minimum_evidence_set (cheapest in each group)
  Phase 5: Filler INTERVIEWs while delayed results arrive
  Phase 6: Final UPDATE_DIFFERENTIAL focused on the true disease
  Phase 7: DIAGNOSE

Why this shape:
  - 2+ UPDATE_DIFFERENTIAL events maxes R6 (anchoring score) at 0.6+.
  - HPI interviews + targeted exam teach the model the diagnostic order
    that's missing from R1's reward (R1 only checks ordered tests; SFT
    bridges the gap).
  - Cheapest-test-in-group keeps R2 in the sweet spot.
  - 8-12 steps total, well under the 15 cap.

Determinism: per-(disease, variant, seed) the oracle is fully deterministic
(seeded RNG drives all internal choices). Same inputs -> same trajectory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .env.catalogs import Catalogs
from .env.env import ClinicalEnv
from .env.state import Action, ActionType, Observation


# ===========================================================================
# Family-aware menus — what to ask / examine / test for each disease family
# ===========================================================================

# Top-priority interview questions per disease family. Picked so the FIRST
# question is the one a real clinician would ask first ("Where does it hurt?",
# "Got chest pain?", etc.).
FAMILY_INTERVIEWS: dict[str, list[str]] = {
    "abdominal":   ["pain_location", "pain_onset", "nausea_vomiting", "bowel_changes"],
    "chest":       ["chest_pain_probe", "pain_radiation", "shortness_of_breath", "palpitations"],
    "respiratory": ["cough", "fever_chills", "shortness_of_breath", "chest_pain_probe"],
    "neuro":       ["headache_probe", "visual_neuro_changes", "pain_onset", "fever_chills"],
    "endocrine":   ["polyuria_polydipsia", "nausea_vomiting", "shortness_of_breath", "fever_chills"],
    "infectious":  ["fever_chills", "urinary_symptoms", "headache_probe", "cough"],
    "benign":      ["pain_severity", "fever_chills", "medications", "social_travel_exposure"],
}

# Top exam maneuver per family. One is enough — exams aren't required for R1.
FAMILY_EXAMS: dict[str, str] = {
    "abdominal":   "abdominal_palpation",
    "chest":       "cardiac_auscultation",
    "respiratory": "pulmonary_auscultation",
    "neuro":       "neck_exam",
    "endocrine":   "mental_status",
    "infectious":  "neck_exam",
    "benign":      "general_appearance",
}

# Filler interviews used to pad while delayed test results come back.
# Drawn from a generic pool that's plausible for almost any patient.
FILLER_INTERVIEWS = [
    "past_medical_history", "medications", "allergies",
    "family_history", "pain_severity", "social_travel_exposure",
]

# ===========================================================================
# Sex/age-conditional add-ons
# ===========================================================================

# Reproductive-age females always get LMP + pregnancy questions before any
# abdominal/pelvic workup — this is THE key pivot for ectopic.
def _is_reproductive_age_female(age: int, sex: str) -> bool:
    return sex == "female" and 12 <= age <= 55


# ===========================================================================
# Helper: pick the cheapest test from each min_evidence group
# ===========================================================================

def _pick_required_tests(
    card: dict[str, Any],
    catalogs: Catalogs,
    test_results: dict[str, tuple[str, str]],
) -> list[str]:
    """Walk minimum_evidence_set and pick the cheapest test FROM EACH GROUP
    whose pre-sampled flag matches the required polarity.

    Why we peek at pre-sampled flags: a real RL agent wouldn't, but the
    oracle is a TEACHER. If the cheapest test happened to land on its
    "normal" branch (e.g. pneumonia CXR comes back clear ~30% of the time
    due to `prob_abnormal=0.7`), picking it would fail R1's polarity check
    and we'd produce a half-credit trajectory — bad SFT data.

    Returns test_ids ordered by (turnaround, cost) so POC labs go first
    (results inform the next step) and expensive imaging goes last.
    """
    chosen: list[str] = []
    for group in card.get("minimum_evidence_set", []):
        any_of = group.get("any_of", [])
        require = group.get("require", "abnormal")
        priced = sorted(
            (catalogs.tests[tid].cost, tid)
            for tid in any_of
            if tid in catalogs.tests
        )
        if not priced:
            continue
        # Walk cheapest -> most expensive, keep the first whose flag matches.
        picked: str | None = None
        for _cost, tid in priced:
            _, flag = test_results.get(tid, ("", "N"))
            polarity_ok = (
                (require == "abnormal" and flag != "N")
                or (require == "normal" and flag == "N")
            )
            if polarity_ok:
                picked = tid
                break
        # Fallback: every test in the group landed on the wrong polarity.
        # Vanishingly rare (would require all `prob_abnormal` draws to miss
        # AND `require` to be "abnormal"); if it ever happens we still emit
        # the cheapest test so the trajectory is valid, even if R1 takes a
        # partial-credit hit.
        if picked is None:
            picked = priced[0][1]
        chosen.append(picked)

    chosen.sort(key=lambda t: (catalogs.tests[t].turnaround_steps, catalogs.tests[t].cost))
    return chosen


# ===========================================================================
# Differential board generators
# ===========================================================================

def _build_initial_board(disease: str, catalogs: Catalogs, rng: random.Random) -> list[dict[str, Any]]:
    """Plausible initial differential: target as top guess, red herring nearby,
    one or two same-family neighbours.

    Probabilities are slightly randomized per seed so different episodes
    don't all submit identical boards (which would zero out R6's shift bonus
    on subsequent boards).
    """
    target = catalogs.diseases[disease]
    family = target.family

    candidates: list[tuple[str, float]] = []
    # Target with weight ~0.35-0.45.
    candidates.append((disease, 0.35 + rng.random() * 0.10))

    # Red-herring neighbour at ~0.25-0.30.
    rh = target.suggested_red_herring
    if rh and rh in catalogs.diseases:
        candidates.append((rh, 0.25 + rng.random() * 0.05))

    # Up to 2 same-family extras at 0.10-0.15 each.
    family_pool = [
        d.id for d in catalogs.diseases.values()
        if d.family == family and d.id != disease and d.id != rh
    ]
    rng.shuffle(family_pool)
    for extra in family_pool[:2]:
        candidates.append((extra, 0.10 + rng.random() * 0.05))

    # Normalize to sum exactly 1.0.
    total = sum(p for _, p in candidates)
    return [{"disease": d, "prob": round(p / total, 3)} for d, p in candidates]


def _build_final_board(disease: str, catalogs: Catalogs) -> list[dict[str, Any]]:
    """High-confidence focused board — true disease at ~0.92, residual mass
    spread across 2 alternates."""
    target = catalogs.diseases[disease]
    rh = target.suggested_red_herring or next(
        (d.id for d in catalogs.diseases.values() if d.id != disease), None
    )
    # Find one more same-family neighbour for residual mass.
    extra = next(
        (d.id for d in catalogs.diseases.values()
         if d.family == target.family and d.id != disease and d.id != rh),
        None,
    )
    board = [{"disease": disease, "prob": 0.92}]
    if rh:
        board.append({"disease": rh, "prob": 0.05})
    if extra:
        board.append({"disease": extra, "prob": 0.03})
    return board


# ===========================================================================
# Rationale templates — short, clinically plausible strings
# ===========================================================================

# Kept generic so we don't have to maintain per-disease text. The model
# learns the action; the rationale is a soft training signal.
_RATIONALE_TEMPLATES = {
    ActionType.INTERVIEW: "Gather history relevant to differential.",
    ActionType.EXAMINE: "Physical exam to refine the differential.",
    ActionType.ORDER_TEST: "Targeted workup for the leading hypotheses.",
    ActionType.UPDATE_DIFFERENTIAL: "Update beliefs given evidence so far.",
    ActionType.DIAGNOSE: "Evidence supports this diagnosis.",
}

# A handful of more specific rationales we override when we know the context.
_SPECIFIC_RATIONALES = {
    "lmp": "LMP critical in any reproductive-age female with abdominal pain.",
    "pregnancy_possibility": "Rule pregnancy in/out before pelvic imaging.",
    "headache_probe": "Characterize headache — thunderclap vs gradual is decisive.",
    "chest_pain_probe": "Characterize chest pain — pleuritic vs pressure narrows the differential.",
    "polyuria_polydipsia": "Screen for hyperglycemia / DKA presentation.",
    "fever_chills": "Systemic infection screen.",
    "abdominal_palpation": "Localize tenderness, check for peritoneal signs.",
    "neck_exam": "Meningismus screening — Kernig/Brudzinski.",
    "ecg": "Rapid bedside cardiac evaluation.",
    "urine_pregnancy_qualitative": "POC pregnancy — gates pelvic imaging.",
    "beta_hcg_quant": "Quantitative hCG — confirm pregnancy and trend.",
    "troponin": "Cardiac biomarker — rule out STEMI/NSTEMI.",
    "ct_head_noncontrast": "Rule out hemorrhage in acute neuro presentation.",
    "ct_chest_pe_protocol": "Definitive imaging for suspected PE.",
}


def _rationale_for(action: ActionType, argument: str) -> str:
    return _SPECIFIC_RATIONALES.get(argument, _RATIONALE_TEMPLATES[action])


# ===========================================================================
# The oracle itself
# ===========================================================================

@dataclass
class OracleResult:
    """One played-through episode, ready to save to disk for SFT data gen."""

    disease: str
    variant_id: str
    seed: int
    actions: list[Action]
    final_obs: Observation


class HeuristicOracle:
    """Plays the env using card-aware rules. One instance can play many
    episodes — `play()` calls `env.reset()` internally each time."""

    def __init__(self, env: ClinicalEnv, catalogs: Catalogs, cards: dict[str, dict[str, Any]]) -> None:
        self.env = env
        self.catalogs = catalogs
        self.cards = cards

    # -----------------------------------------------------------------------
    # Plan-building (pure: no env interaction)
    # -----------------------------------------------------------------------

    def _build_plan(
        self,
        disease: str,
        age: int,
        sex: str,
        rng: random.Random,
        test_results: dict[str, tuple[str, str]],
    ) -> list[Action]:
        """Construct the full action plan up-front. The plan is then executed
        in `play()` which calls `env.step()` for each action.

        Building the plan in one shot (instead of step-by-step reactive
        decisions) is OK here because the oracle has full info — it doesn't
        need to "react" to results. The trained model WILL react to results
        because GRPO drives that behavior; SFT just teaches the action
        VOCABULARY and rough sequencing.
        """
        family = self.catalogs.diseases[disease].family
        card = self.cards[disease]

        # ---- Phase 1: HPI interviews (2-3 family-aware questions) ----
        interview_pool = list(FAMILY_INTERVIEWS.get(family, []))
        # For reproductive-age females with abdominal/infectious/neuro/chest
        # presentations, prepend pregnancy-related questions — that's the
        # canonical "always-ask" set we want the model to internalise.
        if _is_reproductive_age_female(age, sex):
            for q in ("lmp", "pregnancy_possibility"):
                if q in self.catalogs.questions and q not in interview_pool:
                    interview_pool.insert(0, q)

        # Take 2-3 questions; jitter the count and order per seed for variety.
        n_initial = rng.choice([2, 3])
        rng.shuffle(interview_pool)
        # Keep the first family-priority question pinned at index 0 a bit so
        # the model still learns "ask the obvious one first" — small bias only.
        initial_interviews = interview_pool[:n_initial]

        # ---- Phase 2: Initial differential ----
        initial_board = _build_initial_board(disease, self.catalogs, rng)

        # ---- Phase 3: One targeted exam ----
        exam_id = FAMILY_EXAMS.get(family)

        # ---- Phase 4: Required tests (cheapest from each min_evidence group) ----
        required_tests = _pick_required_tests(card, self.catalogs, test_results)

        # ---- Phase 5: Filler interviews (one per pending step we'll wait) ----
        max_pending = max(
            (self.catalogs.tests[t].turnaround_steps for t in required_tests),
            default=0,
        )
        # Use a couple of fillers regardless — ensures non-zero R6 if we ever
        # add a third UPDATE_DIFFERENTIAL slot, and keeps trajectories looking
        # like real history-taking.
        n_filler = max(1, min(2, max_pending))
        used = set(initial_interviews)
        # If the patient is reproductive-age female, lmp/pregnancy_possibility
        # are likely already asked; skip them in fillers.
        if _is_reproductive_age_female(age, sex):
            used.update({"lmp", "pregnancy_possibility"})
        filler_pool = [q for q in FILLER_INTERVIEWS if q not in used]
        rng.shuffle(filler_pool)
        fillers = filler_pool[:n_filler]

        # ---- Phase 6: Final differential ----
        final_board = _build_final_board(disease, self.catalogs)

        # ---- Compose the plan ----
        plan: list[Action] = []
        for qid in initial_interviews:
            plan.append(Action(ActionType.INTERVIEW, qid, _rationale_for(ActionType.INTERVIEW, qid)))
        plan.append(Action(
            ActionType.UPDATE_DIFFERENTIAL,
            f"Initial differential for {family} presentation.",
            _rationale_for(ActionType.UPDATE_DIFFERENTIAL, ""),
            board=initial_board,
        ))
        if exam_id:
            plan.append(Action(ActionType.EXAMINE, exam_id, _rationale_for(ActionType.EXAMINE, exam_id)))
        for tid in required_tests:
            plan.append(Action(ActionType.ORDER_TEST, tid, _rationale_for(ActionType.ORDER_TEST, tid)))
        for qid in fillers:
            plan.append(Action(ActionType.INTERVIEW, qid, _rationale_for(ActionType.INTERVIEW, qid)))
        plan.append(Action(
            ActionType.UPDATE_DIFFERENTIAL,
            f"Evidence supports {disease}.",
            _rationale_for(ActionType.UPDATE_DIFFERENTIAL, ""),
            board=final_board,
        ))
        plan.append(Action(
            ActionType.DIAGNOSE, disease,
            _rationale_for(ActionType.DIAGNOSE, disease),
        ))

        return plan

    # -----------------------------------------------------------------------
    # Public: play one episode end-to-end
    # -----------------------------------------------------------------------

    def play(
        self,
        disease: str,
        variant_id: str | None = None,
        seed: int = 0,
    ) -> OracleResult:
        """Reset the env to (disease, variant_id, seed) and execute a full
        oracle plan. Returns the actions taken plus final observation so the
        caller can compute rewards."""
        # 1. Reset the env. Need age/sex from the chosen variant for the
        #    plan-building step, so we reset first to learn them.
        obs = self.env.reset(disease=disease, variant_id=variant_id, seed=seed)

        # 2. Build the action plan using a SEPARATE RNG seeded from the
        #    same seed (offset). Don't reuse env's RNG — that drives
        #    response sampling and we don't want to perturb it.
        rng = random.Random(seed + 1_000_000)
        # Peek at the pre-sampled test outcomes so we can pick tests whose
        # flags match each min_evidence group's polarity requirement.
        test_results = self.env._episode.hidden.test_results  # noqa: SLF001
        plan = self._build_plan(disease, obs.age, obs.sex, rng, test_results)

        # 3. Execute. Stop early if the env terminates (shouldn't happen
        #    under a well-formed plan, but guard in case the plan exceeds
        #    the step cap).
        actions_taken: list[Action] = []
        for action in plan:
            if obs.terminal:
                break
            obs = self.env.step(action)
            actions_taken.append(action)

        # Read the actually-chosen variant_id off the env's hidden state so
        # the result records what was played, not what was requested (the
        # caller may have passed variant_id=None for random selection).
        actual_variant = self.env._episode.hidden.variant_id  # noqa: SLF001
        return OracleResult(
            disease=disease,
            variant_id=actual_variant,
            seed=seed,
            actions=actions_taken,
            final_obs=obs,
        )
