"""
Prompt rendering and constrained-decoding grammar for the clinical env.

Two responsibilities:

1. **Renderer** (env -> text)  -- `render_prompt(obs, catalogs)` builds the
   string the LLM sees each turn: system instructions + closed-vocab action
   menu + patient intake + episode history + current status + "your turn".

2. **Schema** (text -> action) -- `build_action_schema(catalogs)` returns a
   JSON Schema constraining the model's output to ONE of the 5 action types
   with a closed-vocab `argument` (and a structured `board` for UPDATE_-
   DIFFERENTIAL). Plug into outlines / xgrammar / jsonformer / OpenAI
   structured-output / etc. — they all consume JSON Schema.

3. **Parser** (text -> Action) -- `parse_action_json(raw)` is a robust
   wrapper around `Action.from_json` that strips markdown fences and other
   common LLM-output cruft before parsing.

Why a single file: these three pieces always change together. If we add a
new action type, we touch the system prompt template, the action menu
formatter, AND the schema. Co-locating them keeps the diff trivial.

Token budget (Qwen2.5-3B has 32k context):
  - system + action menu  ~1100 tokens (cacheable)
  - patient intake         ~80 tokens
  - history (15 steps)    ~600 tokens
  - status + turn marker  ~120 tokens
  --------------------------------
  TOTAL                  ~2000 tokens / turn
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .env.catalogs import Catalogs
from .env.state import Action, ActionType, LogEntry, Observation


# ===========================================================================
# System prompt — fixed instructions the model sees every turn
# ===========================================================================

SYSTEM_PROMPT = """\
You are a clinical reasoning agent in an emergency department simulation.
Your goal is to diagnose the patient correctly within 15 steps, balancing
speed, cost, and safety. Tests cost money and take time; missing a critical
diagnosis is dangerous; over-ordering wastes resources.

Each turn you emit EXACTLY ONE JSON action with this shape:

  {"type": <ACTION_TYPE>, "argument": <ID>, "rationale": "<short reason>"}

For UPDATE_DIFFERENTIAL, additionally include a "board" field:
  {"type": "UPDATE_DIFFERENTIAL",
   "argument": "<one-line summary>",
   "rationale": "<short reason>",
   "board": [{"disease": "<id>", "prob": 0.40}, ... ]}

The five action types are:
  INTERVIEW           ask one of 25 fixed questions (free, ~1 min)
  EXAMINE             perform one of 15 physical exam maneuvers (small cost)
  ORDER_TEST          order one of 35 labs/imaging studies (cost varies; some
                      take 1-3 steps to come back)
  UPDATE_DIFFERENTIAL revise your differential board (free, instant)
  DIAGNOSE            commit to a final diagnosis (ENDS the episode)

Rules:
  - `argument` MUST be a valid id from the menu below (no free text).
  - `rationale` is a short string (1 sentence) explaining your choice.
  - For UPDATE_DIFFERENTIAL, `board` probabilities should sum to ~1.0 and
    cover only diseases from the menu.
  - Output ONLY the JSON object. No prose before or after.
"""


# ===========================================================================
# Action menu — closed-vocabulary reference shown every turn
# ===========================================================================

def _render_action_menu(catalogs: Catalogs) -> str:
    """Compact reference card listing every valid id the model can use.

    Why every turn (not once at the start): the renderer is stateless.
    Trainers with prompt-caching will dedupe the static prefix anyway, so we
    keep the function pure and let the cache do its job.
    """
    lines: list[str] = ["===== ACTION MENU ====="]

    # ---- INTERVIEW questions (25) ----
    lines.append("\nINTERVIEW question ids — fixed menu of 25 questions:")
    for q in catalogs.questions.values():
        lines.append(f"  {q.id:<25}  {q.text}")

    # ---- EXAMINE maneuvers (15) ----
    lines.append("\nEXAMINE ids — 15 physical exam maneuvers (cost):")
    for e in catalogs.exams.values():
        cost = f"${e.cost}" if e.cost else "$0"
        lines.append(f"  {e.id:<28}  ({cost})  {e.text}")

    # ---- ORDER_TEST ids (35) ----
    lines.append("\nORDER_TEST ids — 35 labs/imaging (cost / turnaround steps):")
    for t in catalogs.tests.values():
        lines.append(
            f"  {t.id:<30}  (${t.cost} / {t.turnaround_steps}-step)  {t.name}"
        )

    # ---- DIAGNOSE / disease ids (15) ----
    lines.append("\nDIAGNOSE / UPDATE_DIFFERENTIAL disease ids — 15 diseases:")
    for d in catalogs.diseases.values():
        lines.append(f"  {d.id:<28}  ({d.severity})  {d.name}")

    return "\n".join(lines)


# ===========================================================================
# Patient intake — the immutable header for this episode
# ===========================================================================

def _render_intake(obs: Observation) -> str:
    return (
        "===== PATIENT INTAKE =====\n"
        f"Demographics: {obs.age}yo {obs.sex}\n"
        f"Chief complaint: {obs.chief_complaint}\n"
        f"Initial vitals: {obs.intake_vitals}"
    )


# ===========================================================================
# History — the action_log, reformatted as a chat-style transcript
# ===========================================================================

def _render_log_entry(entry: LogEntry) -> str:
    """Format one LogEntry. Action entries get a header line + body indent;
    result entries (env-emitted) are shown as `→` continuation lines."""
    prefix = f"[step {entry.step:>2}]"

    if entry.kind == "result":
        # Strip the "Result — " prefix that the env added (cleaner here).
        body = entry.text
        if body.startswith("Result — "):
            body = body[len("Result — "):]
        return f"{prefix} → {body}"

    # ---- action entries ----
    a = entry.action
    if a is None:
        return f"{prefix} {entry.text}"

    flags: list[str] = []
    if entry.duplicate:
        flags.append("DUP")
    if entry.invalid:
        flags.append("INVALID")
    flag_str = f"  [{','.join(flags)}]" if flags else ""

    meta = f"{a.type.value} {a.argument}"
    cost_str = f" (${entry.cost})" if entry.cost else ""

    # Most LogEntry.text already contains the visible body (Q/A, finding,
    # etc.). Indent it under the header. UPDATE_DIFFERENTIAL is a special
    # case where the body is a one-liner summary — keep it on the same line.
    if a.type == ActionType.UPDATE_DIFFERENTIAL:
        return f"{prefix} {meta}{cost_str}{flag_str} — {entry.text}"
    if a.type == ActionType.DIAGNOSE:
        return f"{prefix} {meta}{cost_str}{flag_str}"

    body = entry.text.strip()
    if body:
        return f"{prefix} {meta}{cost_str}{flag_str}\n  {body}"
    return f"{prefix} {meta}{cost_str}{flag_str}"


def _render_history(action_log: list[LogEntry]) -> str:
    if not action_log:
        return "===== HISTORY =====\n(none — first action of the episode)"
    lines = ["===== HISTORY ====="]
    for entry in action_log:
        lines.append(_render_log_entry(entry))
    return "\n".join(lines)


# ===========================================================================
# Status — what the agent needs to know to plan the next move
# ===========================================================================

def _render_status(obs: Observation) -> str:
    lines = ["===== STATUS ====="]
    lines.append(
        f"Step: {obs.step}/{obs.step_cap}    "
        f"Cost so far: ${obs.cost_so_far}    "
        f"Time elapsed: {obs.time_elapsed_min}min    "
        f"Severity signal: {obs.severity_signal}"
    )

    if obs.pending_tests:
        pend = []
        for pt in obs.pending_tests:
            steps_left = max(0, pt.deliver_at_step - obs.step)
            pend.append(f"{pt.test_id} ({steps_left}-step wait)")
        lines.append(f"Pending tests: {', '.join(pend)}")
    else:
        lines.append("Pending tests: none")

    if obs.differential_board:
        # Sort by prob desc, top 5 only — enough for the agent to anchor.
        sorted_board = sorted(
            obs.differential_board,
            key=lambda e: float(e.get("prob", 0.0)) if isinstance(e, dict) else 0.0,
            reverse=True,
        )[:5]
        items = [
            f"{e['disease']}({float(e.get('prob', 0.0)):.2f})"
            for e in sorted_board
            if isinstance(e, dict) and "disease" in e
        ]
        lines.append(f"Current differential (top 5): {', '.join(items)}")
    else:
        lines.append("Current differential: (not yet set — emit UPDATE_DIFFERENTIAL when ready)")

    return "\n".join(lines)


# ===========================================================================
# Public renderer
# ===========================================================================

def render_prompt(
    obs: Observation,
    catalogs: Catalogs,
    *,
    include_menu: bool = True,
) -> str:
    """Build the LLM prompt for this turn.

    Parameters
    ----------
    obs
        Observation returned by env.reset() or env.step().
    catalogs
        Catalogs bundle (used to render the action menu).
    include_menu
        If False, omit the action menu — useful for compact eval prompts when
        the model has been trained on the menu and doesn't need it inline.

    Returns
    -------
    A single string. Trainers using a chat template should split this into
    system / user messages themselves; we keep it as one string for maximum
    portability across SFT formats.
    """
    parts: list[str] = [SYSTEM_PROMPT.rstrip()]
    if include_menu:
        parts.append(_render_action_menu(catalogs))
    parts.append(_render_intake(obs))
    parts.append(_render_history(obs.action_log))
    parts.append(_render_status(obs))
    parts.append("===== YOUR TURN =====\nOutput ONE JSON action object now.")
    return "\n\n".join(parts)


# ===========================================================================
# JSON Schema for constrained decoding
# ===========================================================================

def build_action_schema(catalogs: Catalogs) -> dict[str, Any]:
    """Produce a JSON Schema that constrains the model's output to a valid
    Action.

    The schema uses `oneOf` over the 5 action types so the decoder branches
    on the chosen `type` and then enforces the matching `argument` enum.
    This is what tools like outlines, xgrammar, jsonformer, and OpenAI
    structured-output consume.

    Why we put the catalogs in here at build-time (rather than declaring
    `argument: string` and validating later): constrained decoders enforce
    enums TOKEN-BY-TOKEN, so embedding the closed vocabulary in the schema
    makes the model literally unable to emit `cbcc` or `bagic_test` — the
    sampling step rejects those tokens before they're committed.
    """
    question_ids = sorted(catalogs.questions.keys())
    exam_ids = sorted(catalogs.exams.keys())
    test_ids = sorted(catalogs.tests.keys())
    disease_ids = sorted(catalogs.diseases.keys())

    def _branch(action_type: str, argument_enum: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {"const": action_type},
                "argument": {"enum": argument_enum},
                "rationale": {"type": "string"},
            },
            "required": ["type", "argument", "rationale"],
            "additionalProperties": False,
        }

    update_branch = {
        "type": "object",
        "properties": {
            "type": {"const": "UPDATE_DIFFERENTIAL"},
            "argument": {"type": "string"},  # free-text summary
            "rationale": {"type": "string"},
            "board": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "disease": {"enum": disease_ids},
                        "prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["disease", "prob"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "argument", "rationale", "board"],
        "additionalProperties": False,
    }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ClinicalAction",
        "oneOf": [
            _branch("INTERVIEW", question_ids),
            _branch("EXAMINE", exam_ids),
            _branch("ORDER_TEST", test_ids),
            update_branch,
            _branch("DIAGNOSE", disease_ids),
        ],
    }


# ===========================================================================
# Robust parser for model output (tolerates common LLM cruft)
# ===========================================================================

# Matches a fenced code block, with or without language tag: ```json ... ```
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```")
# Matches the first JSON object in a string (greedy on the outer braces).
_OBJECT_RE = re.compile(r"\{[\s\S]*?\}")


def parse_action_json(raw: str) -> Action:
    """Parse model output into an Action, tolerating fenced code blocks and
    leading/trailing prose.

    Rationale: even with constrained decoding, models occasionally wrap
    output in ``` fences (because they were trained on chat data) or add
    "Here is your action:" prefixes. This function strips those before
    parsing so the env doesn't see spurious "invalid" actions.

    Falls through to `Action.from_json` for the actual parse — so any
    `ValueError` from there propagates here unchanged.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty model output")

    # 1. If the output is wrapped in ```json ...``` fences, extract the
    #    contents. Use the FIRST match — multi-fence outputs are rare and
    #    almost always mean the model is being chatty; we take the first
    #    JSON block as the action.
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. If there's still surrounding prose, find the first {...} substring.
    if not text.startswith("{"):
        obj_match = _OBJECT_RE.search(text)
        if obj_match:
            text = obj_match.group(0)

    # 3. Hand off to the canonical parser on Action.
    try:
        return Action.from_json(text)
    except ValueError:
        # Re-raise with extra context so the trainer's logs show what failed.
        raise


# ===========================================================================
# Convenience: emit prompt + schema together (one call, one round-trip)
# ===========================================================================

def render_turn_inputs(
    obs: Observation,
    catalogs: Catalogs,
    *,
    include_menu: bool = True,
) -> dict[str, Any]:
    """Bundle prompt + schema for a single turn. Useful when the trainer
    wants to pass both into `model.generate(...)` in one shot."""
    return {
        "prompt": render_prompt(obs, catalogs, include_menu=include_menu),
        "schema": build_action_schema(catalogs),
    }
