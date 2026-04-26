"""
Tests for clinical_rl.prompt — renderer, schema, and parser.

Runnable two ways:
    pytest tests/test_prompt.py
    python tests/test_prompt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_rl.env import (
    Action,
    ActionType,
    ClinicalEnv,
    load_cards,
    load_catalogs,
)
from clinical_rl.prompt import (
    SYSTEM_PROMPT,
    build_action_schema,
    parse_action_json,
    render_prompt,
    render_turn_inputs,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

from pathlib import Path as _Path
_DATA_DIR = _Path(__file__).resolve().parents[1] / "house_md_env" / "data"
CATALOGS = load_catalogs(_DATA_DIR)
CARDS = load_cards(_DATA_DIR / "cards")


def _fresh_env_with_obs():
    """Build a fresh env, reset to a known state. Returns (env, obs)."""
    env = ClinicalEnv(CATALOGS, CARDS)
    obs = env.reset(disease="ectopic_pregnancy", variant_id="v1", seed=42)
    return env, obs


# ---------------------------------------------------------------------------
# render_prompt — section presence
# ---------------------------------------------------------------------------

def test_render_prompt_contains_all_sections_at_reset():
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS)
    for marker in [
        "ACTION MENU",
        "PATIENT INTAKE",
        "HISTORY",
        "STATUS",
        "YOUR TURN",
        "Output ONE JSON action",
    ]:
        assert marker in text, f"missing section marker: {marker}"


def test_render_prompt_includes_system_instructions():
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS)
    # Must include the action type names so the model knows what to emit.
    for action_name in ["INTERVIEW", "EXAMINE", "ORDER_TEST", "UPDATE_DIFFERENTIAL", "DIAGNOSE"]:
        assert action_name in text


def test_render_prompt_intake_demographics_present():
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS)
    # Variant v1 is 34yo female with the canonical chief complaint.
    assert "34yo female" in text
    assert "lower abdominal pain" in text
    assert obs.intake_vitals in text


def test_render_prompt_first_turn_history_is_empty_marker():
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS)
    assert "first action of the episode" in text


def test_render_prompt_grows_with_actions():
    env, obs = _fresh_env_with_obs()
    initial = render_prompt(obs, CATALOGS)
    obs = env.step(Action(ActionType.INTERVIEW, "pain_location"))
    obs = env.step(Action(ActionType.ORDER_TEST, "urine_pregnancy_qualitative"))
    later = render_prompt(obs, CATALOGS)
    # New entries should make the text longer and reference what we did.
    assert len(later) > len(initial)
    assert "INTERVIEW pain_location" in later
    assert "ORDER_TEST urine_pregnancy_qualitative" in later


def test_render_prompt_status_reflects_step_and_cost():
    env, obs = _fresh_env_with_obs()
    obs = env.step(Action(ActionType.ORDER_TEST, "urine_pregnancy_qualitative"))
    text = render_prompt(obs, CATALOGS)
    assert "Step: 2/15" in text
    # urine_pregnancy_qualitative costs $15 in the catalog.
    assert "Cost so far: $15" in text


def test_render_prompt_pending_test_visible():
    env, obs = _fresh_env_with_obs()
    # CT abdomen has 2-step turnaround → still pending after 1 more action.
    obs = env.step(Action(ActionType.ORDER_TEST, "ct_abdomen_pelvis"))
    text = render_prompt(obs, CATALOGS)
    assert "ct_abdomen_pelvis" in text
    assert "step wait" in text


def test_instant_tests_resolve_same_step():
    env = ClinicalEnv(CATALOGS, CARDS, instant_tests=True)
    obs = env.reset(disease="ectopic_pregnancy", variant_id="v1", seed=42)
    obs = env.step(Action(ActionType.ORDER_TEST, "ct_abdomen_pelvis"))
    text = render_prompt(obs, CATALOGS)
    assert "→ CT abdomen/pelvis" in text
    assert "ct_abdomen_pelvis" not in [pt.test_id for pt in obs.pending_tests]
    assert "Pending tests: none" in text


def test_render_prompt_differential_visible_after_update():
    env, obs = _fresh_env_with_obs()
    obs = env.step(Action(
        ActionType.UPDATE_DIFFERENTIAL, "early",
        board=[{"disease": "ectopic_pregnancy", "prob": 0.6},
               {"disease": "appendicitis", "prob": 0.4}],
    ))
    text = render_prompt(obs, CATALOGS)
    assert "ectopic_pregnancy(0.60)" in text
    assert "appendicitis(0.40)" in text


def test_render_prompt_includes_all_catalog_ids_in_menu():
    """Sanity: every id from every catalog appears at least once in the menu."""
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS)
    for qid in CATALOGS.questions:
        assert qid in text, f"question id missing: {qid}"
    for eid in CATALOGS.exams:
        assert eid in text, f"exam id missing: {eid}"
    for tid in CATALOGS.tests:
        assert tid in text, f"test id missing: {tid}"
    for did in CATALOGS.diseases:
        assert did in text, f"disease id missing: {did}"


def test_render_prompt_omits_menu_when_disabled():
    _, obs = _fresh_env_with_obs()
    text = render_prompt(obs, CATALOGS, include_menu=False)
    assert "ACTION MENU" not in text
    # But system prompt and patient intake should still be there.
    assert "PATIENT INTAKE" in text


def test_render_prompt_invalid_action_flagged_in_history():
    env, obs = _fresh_env_with_obs()
    obs = env.step(Action(ActionType.INTERVIEW, "not_a_real_qid"))
    text = render_prompt(obs, CATALOGS)
    assert "INVALID" in text


def test_render_prompt_duplicate_flagged_in_history():
    env, obs = _fresh_env_with_obs()
    obs = env.step(Action(ActionType.INTERVIEW, "pain_location"))
    obs = env.step(Action(ActionType.INTERVIEW, "pain_location"))
    text = render_prompt(obs, CATALOGS)
    assert "DUP" in text


# ---------------------------------------------------------------------------
# build_action_schema — structural correctness
# ---------------------------------------------------------------------------

def test_schema_top_level_oneof():
    schema = build_action_schema(CATALOGS)
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 5  # 5 action types


def test_schema_interview_branch_lists_all_qids():
    schema = build_action_schema(CATALOGS)
    branches = {b["properties"]["type"]["const"]: b for b in schema["oneOf"]}
    interview = branches["INTERVIEW"]
    enum = set(interview["properties"]["argument"]["enum"])
    assert enum == set(CATALOGS.questions.keys())


def test_schema_order_test_branch_lists_all_tids():
    schema = build_action_schema(CATALOGS)
    branches = {b["properties"]["type"]["const"]: b for b in schema["oneOf"]}
    enum = set(branches["ORDER_TEST"]["properties"]["argument"]["enum"])
    assert enum == set(CATALOGS.tests.keys())


def test_schema_diagnose_branch_lists_all_diseases():
    schema = build_action_schema(CATALOGS)
    branches = {b["properties"]["type"]["const"]: b for b in schema["oneOf"]}
    enum = set(branches["DIAGNOSE"]["properties"]["argument"]["enum"])
    assert enum == set(CATALOGS.diseases.keys())


def test_schema_update_differential_has_board():
    schema = build_action_schema(CATALOGS)
    branches = {b["properties"]["type"]["const"]: b for b in schema["oneOf"]}
    upd = branches["UPDATE_DIFFERENTIAL"]
    assert "board" in upd["properties"]
    item = upd["properties"]["board"]["items"]
    assert set(item["properties"]["disease"]["enum"]) == set(CATALOGS.diseases.keys())
    # Probabilities bounded.
    assert item["properties"]["prob"]["minimum"] == 0.0
    assert item["properties"]["prob"]["maximum"] == 1.0


def test_schema_validates_with_jsonschema_if_available():
    """If the jsonschema package is available, validate concrete actions
    against the schema. Skip if not installed (no hard dep)."""
    try:
        import jsonschema
    except ImportError:
        return  # silent skip — plain `python` runner

    schema = build_action_schema(CATALOGS)

    # Valid actions should validate.
    valid_actions = [
        {"type": "INTERVIEW", "argument": "pain_location", "rationale": "localize"},
        {"type": "EXAMINE", "argument": "vital_signs", "rationale": "triage"},
        {"type": "ORDER_TEST", "argument": "cbc", "rationale": "screen"},
        {"type": "UPDATE_DIFFERENTIAL", "argument": "early", "rationale": "hypothesis",
         "board": [{"disease": "ectopic_pregnancy", "prob": 0.5},
                   {"disease": "appendicitis", "prob": 0.5}]},
        {"type": "DIAGNOSE", "argument": "ectopic_pregnancy", "rationale": "evidence sufficient"},
    ]
    for action in valid_actions:
        jsonschema.validate(action, schema)

    # Invalid actions should fail.
    invalid_actions = [
        {"type": "INTERVIEW", "argument": "not_a_qid", "rationale": "x"},
        {"type": "ORDER_TEST", "argument": "cbcc", "rationale": "typo"},
        {"type": "DIAGNOSE", "argument": "made_up", "rationale": "x"},
        {"type": "INTERVIEW", "argument": "pain_location"},  # missing rationale
        {"type": "UPDATE_DIFFERENTIAL", "argument": "x", "rationale": "x", "board": []},
        {"type": "BOGUS_ACTION", "argument": "x", "rationale": "x"},
    ]
    for action in invalid_actions:
        try:
            jsonschema.validate(action, schema)
            raise AssertionError(f"should have rejected: {action}")
        except jsonschema.ValidationError:
            pass


# ---------------------------------------------------------------------------
# parse_action_json — robustness against LLM output cruft
# ---------------------------------------------------------------------------

def test_parse_plain_json():
    raw = '{"type": "INTERVIEW", "argument": "pain_location", "rationale": "x"}'
    a = parse_action_json(raw)
    assert a.type == ActionType.INTERVIEW
    assert a.argument == "pain_location"


def test_parse_strips_markdown_fences():
    raw = '```json\n{"type": "EXAMINE", "argument": "vital_signs", "rationale": "x"}\n```'
    a = parse_action_json(raw)
    assert a.type == ActionType.EXAMINE
    assert a.argument == "vital_signs"


def test_parse_strips_unmarked_fences():
    raw = '```\n{"type": "EXAMINE", "argument": "vital_signs", "rationale": "x"}\n```'
    a = parse_action_json(raw)
    assert a.type == ActionType.EXAMINE


def test_parse_strips_leading_prose():
    raw = 'Here is my action:\n{"type": "DIAGNOSE", "argument": "stemi", "rationale": "ECG"}'
    a = parse_action_json(raw)
    assert a.type == ActionType.DIAGNOSE
    assert a.argument == "stemi"


def test_parse_handles_update_differential_with_board():
    raw = json.dumps({
        "type": "UPDATE_DIFFERENTIAL",
        "argument": "summary",
        "rationale": "x",
        "board": [{"disease": "ectopic_pregnancy", "prob": 0.7},
                  {"disease": "appendicitis", "prob": 0.3}],
    })
    a = parse_action_json(raw)
    assert a.type == ActionType.UPDATE_DIFFERENTIAL
    assert a.board is not None
    assert len(a.board) == 2


def test_parse_empty_raises():
    try:
        parse_action_json("")
        raise AssertionError("should have raised on empty input")
    except ValueError:
        pass


def test_parse_malformed_raises():
    try:
        parse_action_json("this is not json at all")
        raise AssertionError("should have raised on garbage")
    except ValueError:
        pass


def test_parse_unknown_type_raises():
    raw = '{"type": "MEDITATE", "argument": "deeply", "rationale": "calm"}'
    try:
        parse_action_json(raw)
        raise AssertionError("should have rejected unknown action type")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# End-to-end: render → fake model output → parse → step env
# ---------------------------------------------------------------------------

def test_round_trip_render_parse_step():
    """Simulate one full turn: render the prompt, pretend the model emits a
    valid JSON action, parse it, feed it to env.step."""
    env, obs = _fresh_env_with_obs()

    prompt = render_prompt(obs, CATALOGS)
    assert "PATIENT INTAKE" in prompt  # sanity

    fake_model_output = (
        '```json\n'
        '{"type": "INTERVIEW", "argument": "lmp", "rationale": "rule out pregnancy"}\n'
        '```'
    )
    action = parse_action_json(fake_model_output)
    obs = env.step(action)

    assert not obs.terminal
    assert any(
        e.kind == "action" and e.action and e.action.argument == "lmp"
        for e in obs.action_log
    )


def test_render_turn_inputs_returns_both():
    _, obs = _fresh_env_with_obs()
    bundle = render_turn_inputs(obs, CATALOGS)
    assert set(bundle.keys()) == {"prompt", "schema"}
    assert bundle["prompt"].startswith("You are a clinical reasoning agent")
    assert bundle["schema"]["oneOf"]


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
