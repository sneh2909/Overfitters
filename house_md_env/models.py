"""
Pydantic data models for the House M.D. clinical-reasoning environment.

These mirror the internal dataclasses in `clinical_rl.env.state` but are expressed
as Pydantic models so OpenEnv can:

  * generate a JSON schema the client uses for type-safe `.step(...)` payloads
  * validate every action received over HTTP/WebSocket before it touches the env
  * serialize the (potentially large) episode log to the client cheaply

Three top-level types:

  * :class:`HouseMDAction`       what the agent emits each turn
  * :class:`HouseMDObservation`  what the env returns each turn
  * :class:`HouseMDState`        introspection-only snapshot of episode bookkeeping

Everything that's *hidden* from the agent (true disease, RNG state, presampled
test results) lives on `clinical_rl.env.state.HiddenState` and never appears here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


_SUBMODEL_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    arbitrary_types_allowed=True,
)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class HouseMDActionType(str, Enum):
    INTERVIEW = "INTERVIEW"
    EXAMINE = "EXAMINE"
    ORDER_TEST = "ORDER_TEST"
    UPDATE_DIFFERENTIAL = "UPDATE_DIFFERENTIAL"
    DIAGNOSE = "DIAGNOSE"


class DifferentialEntry(BaseModel):
    """One row in an UPDATE_DIFFERENTIAL board."""

    model_config = _SUBMODEL_CONFIG

    disease: str = Field(..., description="Disease id from the corpus")
    prob: float = Field(..., ge=0.0, le=1.0, description="Subjective probability")


class HouseMDAction(Action):
    """One action emitted by the diagnostician agent.

    The five legal action types are detailed in `clinical_rl.env.state.ActionType`.
    `argument` is interpreted differently depending on `type`:

      * INTERVIEW            -> question id (e.g. "lmp")
      * EXAMINE              -> exam id (e.g. "abdominal_exam")
      * ORDER_TEST           -> test id (e.g. "beta_hcg_quant")
      * UPDATE_DIFFERENTIAL  -> free-form summary string; real payload is `board`
      * DIAGNOSE             -> disease id the agent commits to (terminal)
    """

    type: HouseMDActionType = Field(
        ..., description="Which of the 5 action verbs this is"
    )
    argument: str = Field(
        default="",
        description="Vocabulary item the action targets (qid / eid / tid / disease id)",
    )
    rationale: str = Field(
        default="",
        description="Free-form chain-of-thought; never affects env transitions",
    )
    board: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Only set for UPDATE_DIFFERENTIAL. List of "
            "{disease: <id>, prob: <0..1>} entries summing ~1.0."
        ),
    )


# ---------------------------------------------------------------------------
# Observation sub-types
# ---------------------------------------------------------------------------

class LogEntryModel(BaseModel):
    """One row of the visible episode timeline (kind == 'action' or 'result')."""

    model_config = _SUBMODEL_CONFIG

    step: int = Field(..., ge=1, description="1-indexed step at which entry was recorded")
    kind: str = Field(..., description="'action' (agent emitted) or 'result' (env emitted)")
    type: Optional[HouseMDActionType] = Field(
        default=None,
        description="Action verb if kind=='action', null for delivered test results",
    )
    argument: str = Field(default="", description="Argument of the original action")
    rationale: str = Field(default="", description="Rationale recorded at action time")
    text: str = Field(default="", description="Human-readable response/finding/result")
    cost: int = Field(default=0, ge=0, description="USD charged for this entry")
    time_min: int = Field(default=0, ge=0, description="Minutes added to the clock")
    duplicate: bool = Field(
        default=False,
        description="True if this exact (type, argument) pair was used previously",
    )
    invalid: bool = Field(
        default=False,
        description="True if action argument was out-of-vocab or board was malformed",
    )
    error: str = Field(default="", description="Populated when invalid is True")


class PendingTestModel(BaseModel):
    """A lab/imaging test that's been ordered but hasn't returned yet."""

    model_config = _SUBMODEL_CONFIG

    test_id: str = Field(..., description="Test id from the catalog")
    deliver_at_step: int = Field(
        ..., ge=1, description="Step on which the result becomes visible"
    )
    cost_paid: int = Field(default=0, ge=0, description="USD already billed at order time")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class HouseMDObservation(Observation):
    """Everything the agent is allowed to see about the current patient.

    Hidden state (true disease, RNG seed, deterioration flag) is intentionally
    excluded — it lives on the server in `HiddenState`.
    """

    # ---- patient header ----
    chief_complaint: str = Field(default="", description="One-line presenting complaint")
    age: int = Field(default=0, ge=0, le=120, description="Patient age in years")
    sex: str = Field(default="unknown", description="Patient sex ('M' / 'F' / 'unknown')")
    intake_vitals: str = Field(
        default="", description="Pre-sampled vital-signs reading at triage"
    )

    # ---- episode bookkeeping ----
    step: int = Field(
        default=1, ge=1, description="1-indexed; agent's next action lives at this step"
    )
    step_cap: int = Field(
        default=15, ge=1, description="Hard cap on steps (timeout if reached)"
    )
    cost_so_far: int = Field(default=0, ge=0, description="USD spent so far")
    time_elapsed_min: int = Field(default=0, ge=0, description="Minutes since intake")

    # ---- visible history ----
    action_log: List[LogEntryModel] = Field(
        default_factory=list, description="Full visible transcript of actions + results"
    )
    pending_tests: List[PendingTestModel] = Field(
        default_factory=list, description="Tests ordered but not yet returned"
    )
    differential_board: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Latest board the agent posted via UPDATE_DIFFERENTIAL",
    )

    # ---- severity hint (no leak of true severity tier) ----
    severity_signal: str = Field(
        default="stable",
        description="'stable' until the deterioration RNG flips, then 'deteriorating'",
    )

    # ---- termination ----
    terminal: bool = Field(default=False, description="True once episode has ended")
    diagnosis: Optional[str] = Field(
        default=None, description="Disease id the agent committed to (DIAGNOSE)"
    )
    timed_out: bool = Field(
        default=False, description="True iff step_cap was reached without a DIAGNOSE"
    )

    # ---- terminal-only reward breakdown (filled when terminal=True) ----
    rewards: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-rubric scores + 'total' weighted sum. None until episode terminates."
        ),
    )


# ---------------------------------------------------------------------------
# State (read-only introspection endpoint)
# ---------------------------------------------------------------------------

class HouseMDState(State):
    """Server-side episode bookkeeping. Returned by the `/state` endpoint."""

    chief_complaint: str = Field(default="", description="Current patient complaint")
    step: int = Field(default=1, ge=1)
    step_cap: int = Field(default=15, ge=1)
    cost_so_far: int = Field(default=0, ge=0)
    time_elapsed_min: int = Field(default=0, ge=0)
    terminal: bool = Field(default=False)
    diagnosis: Optional[str] = Field(default=None)
    timed_out: bool = Field(default=False)
