"""
Parser output schema (the contract between the parser LLM and the env/rewards).

This is the most safety-critical schema in the dynamic-patient build:
  * Patient LLM emits free-text utterance.
  * Parser LLM emits a `ParsedReply` instance conforming to this schema.
  * Reward functions read `ParsedReply` to compute R1 (accuracy), R3
    (interview value), R8' (channel hygiene), etc.

Therefore:
  * The schema MUST stay backward-compatible. Bumping `schema_version`
    forces a parser retrain AND a rewards.py audit.
  * Decoder grammar (xgrammar / outlines / openai-structured-output)
    should be derived from this schema so the parser physically cannot
    emit malformed JSON.
  * Reward code reading `findings` MUST tolerate missing keys — the
    parser is allowed to say "patient did not mention this".

The parser target VOCABULARY (which keys are valid for which question_id,
which enum values are legal for each key) lives in `data/parser_targets.yaml`.
This module is the wire format; that YAML is the per-question contract.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION: int = 1


class Polarity(str, Enum):
    """Did the patient confirm the symptom/topic the question asked about?

    YES        — patient affirmed (with any qualifiers)
    NO         — patient denied
    UNCLEAR    — hedging, contradictory, or off-topic; treat as missing
    REFUSED    — patient explicitly refused to answer
    OFF_TOPIC  — patient ignored the question entirely
    """

    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"
    REFUSED = "refused"
    OFF_TOPIC = "off_topic"


class Finding(BaseModel):
    """One extracted clinical fact the patient conveyed.

    `key` MUST be a member of the question's `expected_keys` set in
    parser_targets.yaml — anything else is a parser hallucination and
    the validator will reject it.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="Vocabulary key from parser_targets.yaml")
    value: Union[str, float, int, bool, List[str]] = Field(
        ..., description="Value as decoded; type per parser_targets.yaml"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Parser's confidence in this specific extracted finding.",
    )


class SafetyFlags(BaseModel):
    """Channel-hygiene flags. Reward shaping (R8') consumes these.

    The patient LLM is system-prompted NOT to leak diagnoses, test names,
    or medication names. These flags catch when it does anyway, which is
    a strong signal the rollout should be filtered or penalized.
    """

    model_config = ConfigDict(extra="forbid")

    leaked_disease_name: Optional[str] = Field(
        default=None,
        description=(
            "Disease id (or near-string) the patient named verbatim, e.g. "
            "'appendicitis'. Null if no leak detected."
        ),
    )
    leaked_test_name: Optional[str] = Field(
        default=None,
        description="Lab/imaging test name volunteered by the patient.",
    )
    leaked_medication_name: Optional[str] = Field(
        default=None,
        description="Medication name volunteered by the patient.",
    )
    doctor_parroting: bool = Field(
        default=False,
        description=(
            "True if the patient utterance simply echoes a leading framing "
            "from the doctor ('yes you're right I have X'). Anti-injection signal."
        ),
    )
    refusal: bool = Field(default=False, description="Patient refused to answer.")
    off_topic: bool = Field(
        default=False,
        description="Patient response did not address the question at all.",
    )
    self_contradictory: bool = Field(
        default=False,
        description=(
            "Utterance contains contradictory information (e.g. 'yes... no... "
            "well kind of'). Suggests UNCLEAR polarity is appropriate."
        ),
    )


class ParsedReply(BaseModel):
    """Parser LLM's structured interpretation of one patient utterance.

    All reward computations downstream of an INTERVIEW step read THIS
    object, not the raw utterance. Validation here catches the cheap
    failures (malformed enums, missing required fields); semantic
    correctness is enforced via the SFT dataset + held-out eval, not by
    this validator.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="Bump invalidates trained parser checkpoints.",
    )
    question_id: str = Field(
        ..., description="qid the patient was responding to (echoed for audit)."
    )
    polarity: Polarity = Field(
        ..., description="Was the symptom/topic confirmed, denied, or unclear?"
    )
    polarity_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Parser confidence in `polarity`. Reward code ignores findings "
            "below the configured threshold (default 0.5)."
        ),
    )
    findings: List[Finding] = Field(
        default_factory=list,
        description="Extracted clinical facts. Empty list is valid.",
    )
    safety: SafetyFlags = Field(
        default_factory=SafetyFlags,
        description="Channel-hygiene flags consumed by R8'.",
    )
    raw_quote: str = Field(
        default="",
        max_length=2000,
        description=(
            "The patient utterance the parser saw. Echoed back for audit and "
            "for downstream code that wants the original text alongside the "
            "structure (e.g. transcript renderers)."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("findings")
    @classmethod
    def _no_duplicate_keys(cls, v: List[Finding]) -> List[Finding]:
        """Each `key` should appear at most once. Multi-value lives in the
        `value` field as a list, not as repeated entries."""
        seen: set[str] = set()
        for f in v:
            if f.key in seen:
                raise ValueError(f"duplicate finding key: {f.key!r}")
            seen.add(f.key)
        return v


# ---------------------------------------------------------------------------
# JSON Schema export — for constrained decoding (xgrammar / outlines / etc.)
# ---------------------------------------------------------------------------

def parser_json_schema() -> dict:
    """Return the JSON Schema the parser LLM is constrained to emit.

    Hand this to your decoder library:
        outlines.generate.json(model, ParsedReply)
        xgrammar.from_json_schema(parser_json_schema())
    """
    return ParsedReply.model_json_schema()


__all__ = [
    "SCHEMA_VERSION",
    "Polarity",
    "Finding",
    "SafetyFlags",
    "ParsedReply",
    "parser_json_schema",
]
