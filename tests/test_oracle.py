"""
Tests for clinical_rl.oracle.

Covers:
  - Single-episode correctness across all 15 diseases
  - Determinism (same seed → identical trajectory)
  - Plan shape (UPDATE_DIFFERENTIAL bookends, DIAGNOSE last)
  - Female-specific add-ons (lmp asked for reproductive-age female)
  - Polarity-aware test selection across normal/abnormal min_evidence

Runnable two ways:
    pytest tests/test_oracle.py
    python tests/test_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_rl.env import (
    Action,
    ActionType,
    ClinicalEnv,
    load_cards,
    load_catalogs,
)
from clinical_rl.oracle import HeuristicOracle
from clinical_rl.rewards import compute_all


from pathlib import Path as _Path
_DATA_DIR = _Path(__file__).resolve().parents[1] / "house_md_env" / "data"
CATALOGS = load_catalogs(_DATA_DIR)
CARDS = load_cards(_DATA_DIR / "cards")


def _fresh():
    env = ClinicalEnv(CATALOGS, CARDS)
    return env, HeuristicOracle(env, CATALOGS, CARDS)


# ---------------------------------------------------------------------------
# Single-episode shape tests
# ---------------------------------------------------------------------------

def test_oracle_diagnoses_correctly_on_ectopic():
    env, oracle = _fresh()
    result = oracle.play("ectopic_pregnancy", "v1", seed=42)
    assert result.final_obs.terminal
    assert result.final_obs.diagnosis == "ectopic_pregnancy"


def test_oracle_plan_starts_with_interview():
    env, oracle = _fresh()
    result = oracle.play("ectopic_pregnancy", "v1", seed=42)
    assert result.actions[0].type == ActionType.INTERVIEW


def test_oracle_plan_ends_with_diagnose():
    env, oracle = _fresh()
    result = oracle.play("ectopic_pregnancy", "v1", seed=42)
    assert result.actions[-1].type == ActionType.DIAGNOSE
    assert result.actions[-1].argument == "ectopic_pregnancy"


def test_oracle_plan_has_two_differential_updates():
    """The reward design wants R6 ≥ 0.6 — that needs ≥2 UPDATE_DIFFERENTIAL
    events with a meaningful prob shift. Make sure the oracle emits them."""
    env, oracle = _fresh()
    result = oracle.play("stemi", "v1", seed=42)
    updates = [a for a in result.actions if a.type == ActionType.UPDATE_DIFFERENTIAL]
    assert len(updates) >= 2, f"expected >=2 UPDATEs, got {len(updates)}"


def test_oracle_diagnose_is_terminal_for_episode():
    env, oracle = _fresh()
    result = oracle.play("appendicitis", "v1", seed=42)
    assert result.final_obs.terminal
    # No actions should appear AFTER DIAGNOSE.
    diag_index = next(
        i for i, a in enumerate(result.actions)
        if a.type == ActionType.DIAGNOSE
    )
    assert diag_index == len(result.actions) - 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_oracle_deterministic_per_seed():
    env1, oracle1 = _fresh()
    env2, oracle2 = _fresh()
    r1 = oracle1.play("pneumonia", "v2", seed=99)
    r2 = oracle2.play("pneumonia", "v2", seed=99)

    def to_tuple(actions):
        return [(a.type.value, a.argument) for a in actions]

    assert to_tuple(r1.actions) == to_tuple(r2.actions)


def test_oracle_different_seeds_diverge():
    """Different seeds should produce at least slightly different action
    orderings. This ensures the SFT dataset has trajectory diversity."""
    env, oracle = _fresh()

    # Reset the env between calls — each play() does its own reset, but we
    # need separate envs because env state would otherwise carry over.
    env1, oracle1 = _fresh()
    env2, oracle2 = _fresh()
    r1 = oracle1.play("dka", "v1", seed=1)
    r2 = oracle2.play("dka", "v1", seed=2)

    def to_tuple(actions):
        return [(a.type.value, a.argument) for a in actions]

    # At minimum, interview ORDER should vary across seeds even if the test
    # set ends up identical.
    assert to_tuple(r1.actions) != to_tuple(r2.actions), (
        "different seeds produced identical trajectories — RNG dead?"
    )


# ---------------------------------------------------------------------------
# Female-specific behaviour
# ---------------------------------------------------------------------------

def test_oracle_asks_lmp_for_repro_age_female_with_abdominal_pain():
    env, oracle = _fresh()
    result = oracle.play("ectopic_pregnancy", "v1", seed=42)  # 34yo female
    interviewed = [a.argument for a in result.actions if a.type == ActionType.INTERVIEW]
    assert "lmp" in interviewed, f"lmp must be in interviews: {interviewed}"


def test_oracle_does_not_force_lmp_for_male_chest_pain():
    """STEMI patient with male variant — lmp shouldn't appear in the plan."""
    env, oracle = _fresh()
    # Find a male variant (stemi has both sexes in catalog; check the first
    # one selected at seed 42 to be deterministic).
    result = oracle.play("stemi", "v1", seed=42)
    interviewed = [a.argument for a in result.actions if a.type == ActionType.INTERVIEW]
    if result.final_obs.sex == "male":
        assert "lmp" not in interviewed
        assert "pregnancy_possibility" not in interviewed


# ---------------------------------------------------------------------------
# Polarity-aware test selection (the bug we fixed mid-build)
# ---------------------------------------------------------------------------

def test_oracle_handles_normal_polarity_groups_for_benign_cases():
    """Benign disease cards (viral_uri, anxiety_attack, costochondritis) use
    `require: normal` for rule-out tests. The oracle should still satisfy
    R1 fully on these."""
    for disease in ["viral_uri", "anxiety_attack", "costochondritis", "migraine"]:
        env, oracle = _fresh()
        result = oracle.play(disease, "v1", seed=42)
        rewards = compute_all(env._episode, CARDS[disease], CATALOGS)  # noqa: SLF001
        assert result.final_obs.diagnosis == disease, f"{disease}: wrong dx"
        assert rewards["r1_accuracy"] >= 0.5, (
            f"{disease}: R1={rewards['r1_accuracy']} (oracle didn't satisfy min_evidence)"
        )


# ---------------------------------------------------------------------------
# Bulk: high reward across the full corpus
# ---------------------------------------------------------------------------

def test_oracle_avg_reward_above_threshold():
    """Across all 15 diseases × 3 variants × 3 seeds (135 episodes), the
    average reward should be well above the 1.5 cutoff used by the
    trajectory-generation script."""
    env, oracle = _fresh()
    totals: list[float] = []
    correct = 0
    for did in CATALOGS.diseases:
        for vid in ("v1", "v2", "v3"):
            for seed in (11, 22, 33):
                result = oracle.play(did, vid, seed=seed)
                rewards = compute_all(env._episode, CARDS[did], CATALOGS)  # noqa: SLF001
                totals.append(rewards["total"])
                if result.final_obs.diagnosis == did:
                    correct += 1
    assert correct == len(totals), f"oracle missed {len(totals) - correct} cases"
    avg = sum(totals) / len(totals)
    assert avg >= 2.5, f"avg total reward {avg:.2f} below safety floor 2.5"


# ---------------------------------------------------------------------------
# Validity: every action the oracle emits is in-vocab
# ---------------------------------------------------------------------------

def test_oracle_emits_only_valid_actions():
    env, oracle = _fresh()
    result = oracle.play("ectopic_pregnancy", "v1", seed=42)
    for a in result.actions:
        if a.type == ActionType.INTERVIEW:
            assert a.argument in CATALOGS.questions
        elif a.type == ActionType.EXAMINE:
            assert a.argument in CATALOGS.exams
        elif a.type == ActionType.ORDER_TEST:
            assert a.argument in CATALOGS.tests
        elif a.type == ActionType.DIAGNOSE:
            assert a.argument in CATALOGS.diseases
        elif a.type == ActionType.UPDATE_DIFFERENTIAL:
            assert a.board is not None
            for entry in a.board:
                assert entry["disease"] in CATALOGS.diseases
                assert 0.0 <= entry["prob"] <= 1.0


# ---------------------------------------------------------------------------
# Test runner (works without pytest)
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except BaseException as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    print(f"Ran {len(tests)} tests, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_all())
