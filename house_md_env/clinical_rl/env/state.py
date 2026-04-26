"""
Dataclasses for every value that flows through the env.

Why a separate file: env.py becomes much shorter when state shapes are
pinned down here. Anything that reads or writes env state imports from here.

Three layers of state, separated on purpose:
  - Action          : what the agent emits each turn (decoded JSON)
  - Observation     : what the agent sees back (this becomes the LLM prompt)
  - HiddenState     : ground truth + RNG state — must NEVER reach the agent

Dynamic-patient additions (Phase 3 of PLAN_PATIENT_LLM.md):
  - LogEntry gains `parsed` / `parser_failed` / `patient_source`. These are
    SERVER-ONLY fields — `clinical_rl.prompt._render_log_entry` reads only
    `text` / `action` / `cost` / `duplicate` / `invalid`, so adding them
    here cannot leak to the doctor by accident. A unit test pins this.
  - HiddenState tracks `interview_polarities` so the IdentityParserClient
    (used in static mode) can synthesize a faithful ParsedReply without
    asking an LLM. Lets us run R3 / R8' with the same code in both modes,
    which is what makes Phase 5's static-vs-dynamic ablation meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from clinical_rl.parser_schema import ParsedReply, Polarity


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """The 5 actions the policy can emit. SEARCH_KNOWLEDGE was dropped in P0.

    Subclassing `str` makes JSON-encoding the enum trivial — `json.dumps` will
    serialize an `ActionType.INTERVIEW` as `"INTERVIEW"` automatically.
    """

    INTERVIEW = "INTERVIEW"
    EXAMINE = "EXAMINE"
    ORDER_TEST = "ORDER_TEST"
    UPDATE_DIFFERENTIAL = "UPDATE_DIFFERENTIAL"
    DIAGNOSE = "DIAGNOSE"


@dataclass
class Action:
    """One decoded action.

    The constrained decoder (E3) produces JSON like:
        {"type":"ORDER_TEST","argument":"urine_hcg","rationale":"..."}
    For UPDATE_DIFFERENTIAL the JSON additionally carries `board`:
        {"type":"UPDATE_DIFFERENTIAL", "argument":"summary text",
         "rationale":"...", "board":[{"disease":"ectopic_pregnancy","prob":0.4}, ...]}
    """

    type: ActionType
    argument: str
    rationale: str = ""
    # Only populated for UPDATE_DIFFERENTIAL. Each entry: {"disease": str, "prob": float}.
    board: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_json(cls, raw: str | dict) -> "Action":
        """Parse a JSON string (or dict) into an Action.

        Tolerant of model output: missing `rationale` defaults to "", missing
        `board` stays None. Raises ValueError on truly malformed input so the
        env can mark the step invalid rather than crash.
        """
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Action JSON did not parse: {exc}") from exc
        else:
            obj = raw

        if not isinstance(obj, dict):
            raise ValueError(f"Action must be a JSON object, got {type(obj).__name__}")

        try:
            atype = ActionType(obj["type"])
        except KeyError:
            raise ValueError("Action missing 'type' field")
        except ValueError:
            raise ValueError(f"Unknown action type: {obj.get('type')!r}")

        return cls(
            type=atype,
            argument=str(obj.get("argument", "")),
            rationale=str(obj.get("rationale", "")),
            board=obj.get("board"),
        )


# ---------------------------------------------------------------------------
# Episode log
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """One line in the visible episode timeline.

    Two kinds of entries appear in the same log so the prompt renderer (E3)
    can replay the episode top-to-bottom as a chat-like transcript:
      - kind="action": the agent's action and its observable response
      - kind="result": a delayed test result that just arrived this turn
    """

    step: int                       # 1-indexed; matches Observation.step at time of entry
    kind: str                       # "action" | "result"
    action: Optional[Action] = None  # filled when kind == "action"
    text: str = ""                   # the response/finding/result string
    cost: int = 0                    # USD charged for this entry
    time_min: int = 0                # minutes added to the clock
    duplicate: bool = False          # action: was this exact (type, argument) used before?
    invalid: bool = False            # action: out-of-vocab argument or malformed board
    error: str = ""                  # populated when invalid=True

    # ---- Dynamic-patient fields (server-only; not rendered to the doctor) ----
    # ParsedReply from the parser client. Present iff this is an INTERVIEW
    # action (action.type == ActionType.INTERVIEW). Reward shapers (R3, R8')
    # read this; the doctor's prompt MUST NOT.
    parsed: Optional["ParsedReply"] = None
    # True iff the parser LLM failed for this step and we fell back. In static
    # mode this is always False. Reward code may discount or skip steps where
    # this is True to avoid amplifying parser noise.
    parser_failed: bool = False
    # "static"   - canned response from the disease card (default)
    # "llm"      - patient LLM produced the utterance
    # "fallback" - patient LLM failed; canned string was used
    patient_source: str = "static"


# ---------------------------------------------------------------------------
# Pending test queue (lab/imaging with non-zero turnaround)
# ---------------------------------------------------------------------------

@dataclass
class PendingTest:
    """A test that's been ordered but hasn't returned yet.

    `deliver_at_step` uses absolute step numbering — simpler than tracking a
    countdown. A test ordered at step S with turnaround T resolves at step S+T.
    Tests with turnaround=0 (POC like urinalysis, fingerstick) resolve in the
    same step they were ordered.
    """

    test_id: str
    deliver_at_step: int
    result_text: str   # pre-sampled at order time so the value is fixed once chosen
    flag: str          # "N" | "H" | "L" | "CRIT"
    cost_paid: int


# ---------------------------------------------------------------------------
# Observation: the only thing the agent ever sees
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Everything visible to the agent. Becomes the LLM prompt body in E3.

    Hidden state (true_disease, deterioration_rate, RNG seeds, the disease
    card itself) is intentionally NOT here — it lives on HiddenState and
    never crosses the env boundary.
    """

    # ---- patient header (set at reset, immutable thereafter) ----
    chief_complaint: str
    age: int
    sex: str
    intake_vitals: str    # one finding pre-sampled from card.exam_distribution.vital_signs

    # ---- episode bookkeeping ----
    step: int             # 1-indexed; first action observed by agent at step=1
    step_cap: int         # 15 by default
    cost_so_far: int      # USD
    time_elapsed_min: int

    # ---- visible history ----
    action_log: list[LogEntry] = field(default_factory=list)
    pending_tests: list[PendingTest] = field(default_factory=list)
    differential_board: list[dict[str, Any]] = field(default_factory=list)

    # ---- severity hint (no leak of the true severity tier) ----
    # "stable" until the deterioration RNG flips, then "deteriorating" forever.
    severity_signal: str = "stable"

    # ---- termination state ----
    terminal: bool = False
    diagnosis: Optional[str] = None     # disease_id the agent committed to
    timed_out: bool = False             # True if step_cap reached without DIAGNOSE


# ---------------------------------------------------------------------------
# Hidden state: ground truth + RNG, never exposed to the agent
# ---------------------------------------------------------------------------

@dataclass
class HiddenState:
    """Ground truth + pre-sampled responses + mutable bookkeeping.

    Pre-sampling rationale: at reset() we deterministically draw every
    possible question/exam/test outcome using the patient's seeded RNG, then
    cache the result strings. This guarantees:
      - same patient, same question, asked twice -> identical answer
      - different patients on the SAME disease -> different draws
      - reproducibility: a (disease, variant, seed) triple fully pins the
        episode trajectory given a deterministic policy.
    """

    true_disease: str
    variant_id: str
    deterioration_rate: float
    seed: int

    # --- pre-sampled per-patient outcomes (filled at reset) ---
    # qid -> response string (sampled from responses[] or denial_responses[])
    interview_responses: dict[str, str] = field(default_factory=dict)
    # qid -> Polarity (which pool was sampled: YES iff drawn from
    # responses[], NO iff drawn from denial_responses[]). Stored as the str
    # value (e.g. "yes") so this dataclass stays import-free; env.py converts
    # back to Polarity at use time. Used by IdentityParserClient in static
    # mode so reward shaping can read polarity without an LLM in the loop.
    interview_polarities: dict[str, str] = field(default_factory=dict)
    # eid -> finding string
    exam_findings: dict[str, str] = field(default_factory=dict)
    # tid -> (value_text, flag) tuple
    test_results: dict[str, tuple[str, str]] = field(default_factory=dict)

    # --- episode-time mutable state ---
    deteriorating: bool = False
    # tracks how many times each (action_type, argument) pair has been used,
    # so step() can flag duplicates without scanning the full log each turn.
    used_arguments: dict[tuple[str, str], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode wraps obs + hidden together
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """The full env state. ClinicalEnv keeps one of these between reset/step calls."""

    obs: Observation
    hidden: HiddenState
