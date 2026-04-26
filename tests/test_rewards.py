"""
Unit tests for clinical_rl.rewards.

Most tests build minimal Episode objects directly — no env needed — so we
can isolate each rubric's scoring logic and pin specific values to specific
input shapes. The final `test_end_to_end` runs the real env on a real card
to make sure the wiring works against actual data.

Runnable two ways:
    pytest tests/test_rewards.py        (preferred)
    python tests/test_rewards.py        (also works; prints PASS/FAIL summary)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path as _Path
from clinical_rl.env.catalogs import Catalogs, Disease, load_catalogs

_DATA_DIR = _Path(__file__).resolve().parents[1] / "house_md_env" / "data"
from clinical_rl.env.state import (
    Action,
    ActionType,
    Episode,
    HiddenState,
    LogEntry,
    Observation,
)
from clinical_rl.rewards import (
    compute_all,
    r1_accuracy,
    r2_cost,
    r6_anchoring,
    r7_safety,
    r8_format,
)

# ---------------------------------------------------------------------------
# Helpers — construct minimal Episodes for unit tests
# ---------------------------------------------------------------------------

def _make_episode(
    *,
    diagnosis: str | None = "ectopic_pregnancy",
    true_disease: str = "ectopic_pregnancy",
    log: list[LogEntry] | None = None,
    cost: int = 0,
    timed_out: bool = False,
    deteriorating: bool = False,
    test_results: dict[str, tuple[str, str]] | None = None,
) -> Episode:
    """Build a minimal Episode for unit tests. Only fields read by the rubrics
    are populated — everything else gets a sensible default."""
    obs = Observation(
        chief_complaint="test",
        age=30,
        sex="female",
        intake_vitals="normal",
        step=1,
        step_cap=15,
        cost_so_far=cost,
        time_elapsed_min=0,
        action_log=log or [],
        terminal=True,
        diagnosis=diagnosis,
        timed_out=timed_out,
    )
    hidden = HiddenState(
        true_disease=true_disease,
        variant_id="v1",
        deterioration_rate=0.2,
        seed=42,
        test_results=test_results or {},
        deteriorating=deteriorating,
    )
    return Episode(obs=obs, hidden=hidden)


def _action_log(*entries: LogEntry) -> list[LogEntry]:
    """Convenience wrapper. Use entries built by the helpers below."""
    return list(entries)


def _order_test_entry(step: int, tid: str, invalid: bool = False) -> LogEntry:
    return LogEntry(
        step=step, kind="action",
        action=Action(ActionType.ORDER_TEST, tid),
        text=f"Ordered: {tid}",
        cost=50, invalid=invalid,
    )


def _update_diff_entry(step: int, board: list[dict], invalid: bool = False) -> LogEntry:
    return LogEntry(
        step=step, kind="action",
        action=Action(ActionType.UPDATE_DIFFERENTIAL, "summary", board=board),
        text=f"Differential updated", invalid=invalid,
    )


def _interview_entry(step: int, qid: str, invalid: bool = False) -> LogEntry:
    return LogEntry(
        step=step, kind="action",
        action=Action(ActionType.INTERVIEW, qid),
        text=f"Q ({qid}): A: ...", invalid=invalid,
    )


def _diagnose_entry(step: int, did: str, invalid: bool = False) -> LogEntry:
    return LogEntry(
        step=step, kind="action",
        action=Action(ActionType.DIAGNOSE, did),
        text=f"Final diagnosis: {did}", invalid=invalid,
    )


def _result_entry(step: int, text: str = "Result — ...") -> LogEntry:
    """Result entries are env-emitted; their `kind` is 'result' so R8 ignores them."""
    return LogEntry(step=step, kind="result", action=None, text=text)


# A minimal card stub for R1 tests — has just the field R1 reads.
ECTOPIC_CARD = {
    "id": "ectopic_pregnancy",
    "minimum_evidence_set": [
        {"any_of": ["beta_hcg_quant", "urine_pregnancy_qualitative"], "require": "abnormal"},
        {"any_of": ["pelvic_ultrasound_tv", "ct_abdomen_pelvis"], "require": "abnormal"},
    ],
}

VIRAL_URI_CARD = {
    "id": "viral_uri",
    "minimum_evidence_set": [
        {"any_of": ["chest_xray"], "require": "normal"},
    ],
}

# Tiny fake catalog with just a few diseases of varying severity, for R7 tests.
def _fake_catalogs() -> Catalogs:
    diseases = {
        "ectopic_pregnancy": Disease(
            id="ectopic_pregnancy", name="x", family="abdominal", severity="critical",
            deterioration_rate=0.2, suggested_red_herring=None,
            sex_allowed=("female",), age_min=15, age_max=50,
        ),
        "appendicitis": Disease(
            id="appendicitis", name="x", family="abdominal", severity="urgent",
            deterioration_rate=0.1, suggested_red_herring=None,
            sex_allowed=("male", "female"), age_min=8, age_max=75,
        ),
        "viral_uri": Disease(
            id="viral_uri", name="x", family="benign", severity="stable",
            deterioration_rate=0.01, suggested_red_herring=None,
            sex_allowed=("male", "female"), age_min=2, age_max=95,
        ),
    }
    return Catalogs(diseases=diseases, questions={}, exams={}, tests={})


# ---------------------------------------------------------------------------
# R1 — Accuracy
# ---------------------------------------------------------------------------

def test_r1_correct_dx_full_evidence():
    ep = _make_episode(
        diagnosis="ectopic_pregnancy",
        true_disease="ectopic_pregnancy",
        log=_action_log(
            _order_test_entry(1, "urine_pregnancy_qualitative"),
            _order_test_entry(2, "pelvic_ultrasound_tv"),
        ),
        test_results={
            "urine_pregnancy_qualitative": ("Positive", "H"),
            "pelvic_ultrasound_tv": ("No IUP, adnexal mass", "CRIT"),
        },
    )
    assert r1_accuracy(ep, ECTOPIC_CARD) == 1.0


def test_r1_correct_dx_partial_evidence():
    """Only the hCG group satisfied; ultrasound never ordered."""
    ep = _make_episode(
        log=_action_log(_order_test_entry(1, "beta_hcg_quant")),
        test_results={"beta_hcg_quant": ("2840 mIU/mL", "H")},
    )
    assert r1_accuracy(ep, ECTOPIC_CARD) == 0.5


def test_r1_correct_dx_no_evidence():
    """Lucky-guess scenario — agent diagnoses without ordering anything."""
    ep = _make_episode(log=[], test_results={})
    assert r1_accuracy(ep, ECTOPIC_CARD) == 0.2


def test_r1_wrong_dx_returns_zero():
    ep = _make_episode(diagnosis="appendicitis", true_disease="ectopic_pregnancy")
    assert r1_accuracy(ep, ECTOPIC_CARD) == 0.0


def test_r1_timeout_returns_zero():
    """No diagnosis at all (timed out)."""
    ep = _make_episode(diagnosis=None, timed_out=True)
    assert r1_accuracy(ep, ECTOPIC_CARD) == 0.0


def test_r1_normal_polarity_satisfied():
    """Benign case where the rule-out test must come back NORMAL."""
    ep = _make_episode(
        diagnosis="viral_uri",
        true_disease="viral_uri",
        log=_action_log(_order_test_entry(1, "chest_xray")),
        test_results={"chest_xray": ("Clear lungs.", "N")},
    )
    assert r1_accuracy(ep, VIRAL_URI_CARD) == 1.0


def test_r1_normal_polarity_violated():
    """Ordered the rule-out but it came back ABNORMAL — group not satisfied."""
    ep = _make_episode(
        diagnosis="viral_uri",
        true_disease="viral_uri",
        log=_action_log(_order_test_entry(1, "chest_xray")),
        test_results={"chest_xray": ("RLL infiltrate.", "H")},
    )
    # Correct dx but the only group failed (CXR was abnormal, not normal as required).
    assert r1_accuracy(ep, VIRAL_URI_CARD) == 0.2


def test_r1_invalid_orders_dont_count():
    """An ORDER_TEST flagged invalid (out-of-vocab) shouldn't satisfy evidence."""
    ep = _make_episode(
        log=_action_log(
            _order_test_entry(1, "urine_pregnancy_qualitative", invalid=True),
            _order_test_entry(2, "pelvic_ultrasound_tv"),
        ),
        test_results={
            "urine_pregnancy_qualitative": ("Positive", "H"),
            "pelvic_ultrasound_tv": ("Adnexal mass", "CRIT"),
        },
    )
    # Only group 2 satisfied (group 1's only valid ordering was marked invalid).
    assert r1_accuracy(ep, ECTOPIC_CARD) == 0.5


# ---------------------------------------------------------------------------
# R2 — Cost
# ---------------------------------------------------------------------------

def test_r2_zero_cost_under_test_penalty():
    ep = _make_episode(cost=0)
    assert r2_cost(ep) == 0.2


def test_r2_sweet_spot_lower():
    ep = _make_episode(cost=200)
    assert abs(r2_cost(ep) - 1.0) < 1e-9


def test_r2_sweet_spot_upper():
    ep = _make_episode(cost=500)
    assert r2_cost(ep) == 1.0


def test_r2_overtest_zero_at_1500():
    ep = _make_episode(cost=1500)
    assert abs(r2_cost(ep) - 0.0) < 1e-9


def test_r2_capped_at_minus_half():
    ep = _make_episode(cost=10_000)
    assert r2_cost(ep) == -0.5


def test_r2_monotone_decreasing_above_500():
    """Sanity: cost increases past sweet spot → reward strictly decreases."""
    samples = [600, 800, 1000, 1500, 2000, 3000]
    scores = [r2_cost(_make_episode(cost=c)) for c in samples]
    for a, b in zip(scores, scores[1:]):
        assert a > b, f"R2 should be monotone decreasing past $500: {scores}"


# ---------------------------------------------------------------------------
# R6 — Anchoring
# ---------------------------------------------------------------------------

def test_r6_no_updates():
    ep = _make_episode(log=_action_log(_interview_entry(1, "pain_location")))
    assert r6_anchoring(ep) == 0.0


def test_r6_one_update():
    ep = _make_episode(log=_action_log(
        _update_diff_entry(1, [{"disease": "ectopic_pregnancy", "prob": 1.0}])
    ))
    assert abs(r6_anchoring(ep) - 0.3) < 1e-9


def test_r6_two_updates_with_shift():
    ep = _make_episode(log=_action_log(
        _update_diff_entry(1, [{"disease": "ectopic_pregnancy", "prob": 0.4}]),
        _update_diff_entry(2, [{"disease": "ectopic_pregnancy", "prob": 0.9}]),
    ))
    assert abs(r6_anchoring(ep) - 0.6) < 1e-9


def test_r6_two_updates_no_shift_only_counts_first():
    """Identical boards: second one fails the >0.05 shift check → meaningful=1."""
    same_board = [{"disease": "ectopic_pregnancy", "prob": 0.5}]
    ep = _make_episode(log=_action_log(
        _update_diff_entry(1, same_board),
        _update_diff_entry(2, same_board),
    ))
    assert abs(r6_anchoring(ep) - 0.3) < 1e-9


def test_r6_caps_at_one():
    """Many updates with shifts should saturate at 1.0."""
    log = []
    for i in range(8):
        log.append(_update_diff_entry(i + 1, [{"disease": "ectopic_pregnancy", "prob": 0.1 * (i + 1)}]))
    ep = _make_episode(log=_action_log(*log))
    assert r6_anchoring(ep) == 1.0


def test_r6_invalid_updates_excluded():
    """An UPDATE_DIFFERENTIAL flagged invalid should not contribute."""
    ep = _make_episode(log=_action_log(
        _update_diff_entry(1, [{"disease": "ectopic_pregnancy", "prob": 0.4}], invalid=True),
        _update_diff_entry(2, [{"disease": "ectopic_pregnancy", "prob": 0.9}]),
    ))
    # Only one valid board → 0.3
    assert abs(r6_anchoring(ep) - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# R7 — Safety
# ---------------------------------------------------------------------------

def test_r7_timeout_critical_worst_case():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis=None, true_disease="ectopic_pregnancy", timed_out=True)
    assert r7_safety(ep, cats) == -2.0


def test_r7_timeout_urgent():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis=None, true_disease="appendicitis", timed_out=True)
    assert r7_safety(ep, cats) == -1.0


def test_r7_timeout_stable():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis=None, true_disease="viral_uri", timed_out=True)
    assert r7_safety(ep, cats) == -0.5


def test_r7_wrong_dx_critical():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis="appendicitis", true_disease="ectopic_pregnancy")
    assert r7_safety(ep, cats) == -1.0


def test_r7_wrong_dx_stable():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis="appendicitis", true_disease="viral_uri")
    assert r7_safety(ep, cats) == -0.2


def test_r7_correct_stable_no_harm():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis="ectopic_pregnancy", true_disease="ectopic_pregnancy")
    assert r7_safety(ep, cats) == 0.0


def test_r7_correct_but_deteriorated():
    cats = _fake_catalogs()
    ep = _make_episode(
        diagnosis="ectopic_pregnancy", true_disease="ectopic_pregnancy",
        deteriorating=True,
    )
    assert r7_safety(ep, cats) == -0.2


# ---------------------------------------------------------------------------
# R8 — Format
# ---------------------------------------------------------------------------

def test_r8_all_valid():
    ep = _make_episode(log=_action_log(
        _interview_entry(1, "pain_location"),
        _interview_entry(2, "pain_onset"),
        _diagnose_entry(3, "ectopic_pregnancy"),
    ))
    assert r8_format(ep) == 1.0


def test_r8_half_invalid():
    ep = _make_episode(log=_action_log(
        _interview_entry(1, "pain_location"),
        _interview_entry(2, "bogus_qid", invalid=True),
        _interview_entry(3, "pain_onset"),
        _interview_entry(4, "another_bogus", invalid=True),
    ))
    assert r8_format(ep) == 0.5


def test_r8_result_entries_excluded():
    """Result LogEntries are env-emitted and shouldn't change the format score."""
    ep = _make_episode(log=_action_log(
        _interview_entry(1, "pain_location"),
        _result_entry(2, "Result — urinalysis: ..."),
        _interview_entry(3, "pain_onset"),
    ))
    assert r8_format(ep) == 1.0


def test_r8_empty_log_returns_zero():
    ep = _make_episode(log=[])
    assert r8_format(ep) == 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def test_compute_all_returns_six_keys():
    cats = _fake_catalogs()
    ep = _make_episode()
    result = compute_all(ep, ECTOPIC_CARD, cats)
    assert set(result.keys()) == {
        "r1_accuracy", "r2_cost", "r6_anchoring", "r7_safety", "r8_format", "total"
    }


def test_compute_all_total_uses_default_weights():
    """Manually compute the weighted sum and check `total` matches."""
    cats = _fake_catalogs()
    ep = _make_episode(
        diagnosis="ectopic_pregnancy",
        log=_action_log(
            _order_test_entry(1, "urine_pregnancy_qualitative"),
            _order_test_entry(2, "pelvic_ultrasound_tv"),
            _update_diff_entry(3, [{"disease": "ectopic_pregnancy", "prob": 0.4}]),
            _update_diff_entry(4, [{"disease": "ectopic_pregnancy", "prob": 0.92}]),
            _diagnose_entry(5, "ectopic_pregnancy"),
        ),
        cost=350,
        test_results={
            "urine_pregnancy_qualitative": ("Positive", "H"),
            "pelvic_ultrasound_tv": ("No IUP, adnexal mass", "CRIT"),
        },
    )
    r = compute_all(ep, ECTOPIC_CARD, cats)
    # Sanity: the per-rubric scores we expect for this trajectory.
    assert r["r1_accuracy"] == 1.0
    assert r["r2_cost"] == 1.0     # $350 sits in the sweet spot
    assert abs(r["r6_anchoring"] - 0.6) < 1e-9
    assert r["r7_safety"] == 0.0   # critical disease, correct, not deteriorated
    assert r["r8_format"] == 1.0
    expected = 2.0 * 1.0 + 0.5 * 1.0 + 0.3 * 0.6 + 1.0 * 0.0 + 0.5 * 1.0
    assert abs(r["total"] - expected) < 1e-9


def test_compute_all_custom_weights():
    cats = _fake_catalogs()
    ep = _make_episode(diagnosis="ectopic_pregnancy")
    weights = {"r1_accuracy": 5.0, "r2_cost": 0.0, "r6_anchoring": 0.0,
               "r7_safety": 0.0, "r8_format": 0.0}
    r = compute_all(ep, ECTOPIC_CARD, cats, weights=weights)
    # Only R1 contributes. R1 = 0.2 (correct, no evidence).
    assert abs(r["total"] - 5.0 * 0.2) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end against the real env + real card
# ---------------------------------------------------------------------------

def test_end_to_end_ectopic_walkthrough():
    """Replay the canonical ectopic episode through the real env and check
    that compute_all gives sensible numbers across all five rubrics."""
    from clinical_rl.env import ClinicalEnv, load_cards

    catalogs = load_catalogs(_DATA_DIR)
    cards = load_cards(_DATA_DIR / "cards")
    env = ClinicalEnv(catalogs, cards)
    obs = env.reset(disease="ectopic_pregnancy", variant_id="v1", seed=42)

    plan = [
        Action(ActionType.INTERVIEW, "pain_location"),
        Action(ActionType.INTERVIEW, "lmp"),
        Action(ActionType.UPDATE_DIFFERENTIAL, "early", board=[
            {"disease": "ectopic_pregnancy", "prob": 0.4},
            {"disease": "appendicitis", "prob": 0.3},
            {"disease": "ovarian_torsion", "prob": 0.3},
        ]),
        Action(ActionType.ORDER_TEST, "urine_pregnancy_qualitative"),
        Action(ActionType.ORDER_TEST, "pelvic_ultrasound_tv"),
        Action(ActionType.UPDATE_DIFFERENTIAL, "after evidence", board=[
            {"disease": "ectopic_pregnancy", "prob": 0.92},
            {"disease": "ovarian_torsion", "prob": 0.05},
            {"disease": "appendicitis", "prob": 0.03},
        ]),
        Action(ActionType.DIAGNOSE, "ectopic_pregnancy"),
    ]
    for a in plan:
        if obs.terminal:
            break
        obs = env.step(a)

    assert obs.terminal and obs.diagnosis == "ectopic_pregnancy"

    card = cards["ectopic_pregnancy"]
    rewards = compute_all(env._episode, card, catalogs)

    # Strong play should score positively on the major rubrics.
    assert rewards["r1_accuracy"] == 1.0, f"R1 should be full: {rewards}"
    assert rewards["r2_cost"] >= 0.8, f"R2 should be near sweet spot: {rewards}"
    assert rewards["r6_anchoring"] >= 0.6, f"R6 should reflect 2 meaningful updates: {rewards}"
    assert rewards["r8_format"] == 1.0
    # R7 may be 0 or -0.2 depending on whether deterioration RNG fired.
    assert rewards["r7_safety"] in (0.0, -0.2)
    assert rewards["total"] > 2.0, f"strong play should yield total > 2.0: {rewards}"


def test_end_to_end_diagnose_immediately_cheat():
    """The "DIAGNOSE on step 1 with no work" cheat should NOT outscore a
    real play. Specifically: total reward should be lower than the
    walkthrough's total."""
    from clinical_rl.env import ClinicalEnv, load_cards

    catalogs = load_catalogs(_DATA_DIR)
    cards = load_cards(_DATA_DIR / "cards")
    env = ClinicalEnv(catalogs, cards)
    obs = env.reset(disease="ectopic_pregnancy", variant_id="v1", seed=42)
    obs = env.step(Action(ActionType.DIAGNOSE, "ectopic_pregnancy"))

    rewards = compute_all(env._episode, cards["ectopic_pregnancy"], catalogs)
    # R1 is only 0.2 (no evidence), R6 is 0 (no updates).
    assert rewards["r1_accuracy"] == 0.2
    assert rewards["r6_anchoring"] == 0.0
    # Total should be modest — well below a thoughtful play.
    assert rewards["total"] < 1.5, f"cheat path scored too high: {rewards}"


# ---------------------------------------------------------------------------
# Test runner (works without pytest)
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures: list[tuple[str, BaseException]] = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except BaseException as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append((fn.__name__, e))
    print()
    print(f"Ran {len(tests)} tests, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_all())
