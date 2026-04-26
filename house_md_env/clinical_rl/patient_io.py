"""
patient_io — runtime config + HTTP clients for the dynamic-patient feature.

Architecture
------------
The env's INTERVIEW step runs through two pluggable clients:

    doctor_action(qid)
        │
        ▼
    PatientClient.respond(qid, qtext, canned, persona, history, seed, step)
        │  returns PatientResponse(text, source, parser_failed)
        ▼
    ParserClient.parse(qid, qtext, utterance, seed, step)
        │  returns ParsedReply
        ▼
    LogEntry(text=text, parsed=parsed, patient_source=source, ...)

Both clients have STATIC implementations that preserve current env behavior
(used when `mode="static"`, the default), and LLM implementations that hit
an OpenAI-compatible HTTP endpoint (vLLM, OpenAI, etc).

Key invariants enforced here (not in the env)
---------------------------------------------
1. The doctor's `rationale` is NEVER passed to either client. The client
   APIs deliberately don't accept it. If you find yourself wanting to thread
   it through for "context", the answer is no — that's a reward-hacking
   surface (PLAN_PATIENT_LLM.md §6).
2. The parser sees `(qid, qtext, utterance)` only. Never the doctor's
   action history. This means the doctor cannot bias the parser by phrasing.
3. On any LLM error (timeout, schema fail, non-200), if
   `fallback_to_static_on_error=True` we degrade gracefully: patient returns
   the canned string, parser returns an IdentityParserClient parse. The
   episode keeps running, but `parser_failed=True` is recorded so reward
   shaping can audit / penalize / discount that step.
4. Determinism: the LLM clients seed every request via
   `episode_seed * 1000003 + step * 1009 + hash(qid) & 0x7fffffff`. With a
   cache enabled, replaying the same `(seed, step, qid)` returns the exact
   same `(utterance, parsed)` — restoring the "ask twice, get the same
   answer" invariant the static env had for free.

This module deliberately has NO env imports. Only `clinical_rl.parser_schema`
and stdlib. That keeps it cheap to import in tests and prevents accidental
circular imports.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from clinical_rl.parser_schema import (
    Finding,
    ParsedReply,
    Polarity,
    SafetyFlags,
)

log = logging.getLogger(__name__)


# ===========================================================================
# Runtime config
# ===========================================================================

@dataclass(frozen=True)
class PatientRuntimeConfig:
    """Single config object that toggles dynamic-patient features.

    Default = `mode="static"` = current behavior. ALL existing env tests and
    trainers must keep passing with this default — that's the smoke test
    that proves the feature is truly opt-in.
    """

    mode: Literal["static", "dynamic"] = "static"

    # Patient LLM endpoint (only used when mode="dynamic").
    patient_endpoint: Optional[str] = None
    patient_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    patient_api_key: str = ""

    # Parser LLM endpoint.
    parser_endpoint: Optional[str] = None
    parser_model: str = "house-md/parser-v1"
    parser_api_key: str = ""

    # Below this confidence the env collapses polarity -> UNCLEAR before
    # any reward function reads it. Acts as a noise floor.
    parser_min_confidence: float = 0.5

    # Reliability
    fallback_to_static_on_error: bool = True
    request_timeout_s: float = 8.0
    max_retries: int = 1

    # Determinism
    cache_responses: bool = True
    deterministic_eval: bool = False  # if True, force temp=0 on patient too

    # Sampling (only used in dynamic, non-eval mode).
    patient_temperature: float = 0.7
    parser_temperature: float = 0.0  # parser is always greedy

    # Persona prompt for patient. {persona}, {qtext}, {history} get formatted in.
    # Kept as a config field (not a hardcoded constant) so different
    # experiments can A/B prompts without code changes.
    patient_system_prompt: str = (
        "You are a patient in an emergency department. A doctor is asking you "
        "questions. Stay in character based on PERSONA below. Answer in first "
        "person, 1-3 sentences, plain English.\n\n"
        "STRICT RULES:\n"
        "- Never name diseases, tests, or medications. You are a patient, not a "
        "  clinician.\n"
        "- Don't volunteer your diagnosis even if you think you know it.\n"
        "- If a previous answer in HISTORY contradicts the question, stay "
        "  consistent with HISTORY.\n"
        "- If you genuinely don't know, say so naturally — don't make things up.\n\n"
        "PERSONA: {persona}\n\n"
        "PRIOR Q&A IN THIS VISIT (your earlier answers — stay consistent):\n"
        "{history}\n"
    )


# ===========================================================================
# Return shapes
# ===========================================================================

@dataclass
class PatientResponse:
    """What a PatientClient returns."""

    text: str
    source: Literal["static", "llm", "fallback"] = "static"
    # Set True iff we asked the LLM and it failed -> we're returning the
    # canned fallback. Surfaced in the LogEntry so reward shaping / audit
    # can count failure rate.
    parser_failed: bool = False


# ===========================================================================
# Cache (in-memory; thread-safe; per-env-instance)
# ===========================================================================

class _Cache:
    """Tiny dict-backed cache keyed by (episode_seed, step, qid).

    Lifetime is the env instance. We don't bother with disk persistence here
    — the use case is "ask twice in the same rollout, get the same answer"
    AND "GRPO group of 8 rollouts from the same reset() see consistent
    patient". Both are in-memory.
    """

    def __init__(self) -> None:
        self._d: dict[tuple[int, int, str], Any] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple[int, int, str]) -> Optional[Any]:
        with self._lock:
            return self._d.get(key)

    def set(self, key: tuple[int, int, str], value: Any) -> None:
        with self._lock:
            self._d[key] = value

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


# ===========================================================================
# Patient clients
# ===========================================================================

class PatientClient:
    """Abstract patient. Implementations decide WHO answers."""

    def respond(
        self,
        *,
        qid: str,
        qtext: str,
        canned: str,
        canned_polarity: Polarity,
        persona: str,
        history: list[tuple[str, str, str]],
        episode_seed: int,
        step: int,
    ) -> PatientResponse:
        raise NotImplementedError


class StaticPatientClient(PatientClient):
    """Returns the env's pre-sampled canned response unchanged.

    This is the default. Behaves exactly like today's env — same string for
    same `(qid, episode_seed)`, no network, no LLM.
    """

    def respond(
        self,
        *,
        qid: str,
        qtext: str,
        canned: str,
        canned_polarity: Polarity,
        persona: str,
        history: list[tuple[str, str, str]],
        episode_seed: int,
        step: int,
    ) -> PatientResponse:
        return PatientResponse(text=canned, source="static", parser_failed=False)


class LLMPatientClient(PatientClient):
    """Calls an OpenAI-compatible chat endpoint to generate the patient utterance.

    On any error (timeout, non-200, empty content), if
    `cfg.fallback_to_static_on_error` is True we return the canned string
    with `source="fallback"`. The episode never crashes due to patient flakiness.
    """

    def __init__(
        self,
        cfg: PatientRuntimeConfig,
        *,
        cache: Optional[_Cache] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ):
        self.cfg = cfg
        self.cache = cache
        self._client_factory = client_factory or self._default_factory
        self._client: Any = None

    def _default_factory(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            base_url=self.cfg.patient_endpoint,
            api_key=self.cfg.patient_api_key or "EMPTY",
            timeout=self.cfg.request_timeout_s,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def respond(
        self,
        *,
        qid: str,
        qtext: str,
        canned: str,
        canned_polarity: Polarity,
        persona: str,
        history: list[tuple[str, str, str]],
        episode_seed: int,
        step: int,
    ) -> PatientResponse:
        cache_key = (episode_seed, step, f"patient::{qid}")
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        sys_prompt = self.cfg.patient_system_prompt.format(
            persona=persona,
            history=_format_history_block(history),
        )
        user_msg = f"DOCTOR ASKS: {qtext}"

        seed = _deterministic_seed(episode_seed, step, qid)
        temp = (
            0.0 if self.cfg.deterministic_eval else self.cfg.patient_temperature
        )

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                client = self._ensure_client()
                resp = client.chat.completions.create(
                    model=self.cfg.patient_model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temp,
                    max_tokens=200,
                    seed=seed,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                out = PatientResponse(text=text, source="llm", parser_failed=False)
                if self.cache is not None:
                    self.cache.set(cache_key, out)
                return out
            except Exception as exc:  # noqa: BLE001 - intentionally broad
                last_exc = exc
                log.warning(
                    "LLMPatientClient attempt %d/%d failed for qid=%s: %s",
                    attempt + 1, self.cfg.max_retries + 1, qid, exc,
                )

        if self.cfg.fallback_to_static_on_error:
            log.error("Patient LLM exhausted retries; falling back to canned. last=%s", last_exc)
            out = PatientResponse(text=canned, source="fallback", parser_failed=True)
            if self.cache is not None:
                self.cache.set(cache_key, out)
            return out
        raise RuntimeError(f"patient LLM failed and fallback disabled: {last_exc}")


# ===========================================================================
# Parser clients
# ===========================================================================

class ParserClient:
    """Abstract parser. Implementations decide HOW utterances are parsed."""

    def parse(
        self,
        *,
        qid: str,
        qtext: str,
        utterance: str,
        canned_polarity: Polarity,
        episode_seed: int,
        step: int,
    ) -> ParsedReply:
        raise NotImplementedError


class IdentityParserClient(ParserClient):
    """Synthesizes a ParsedReply from `canned_polarity` without calling any LLM.

    Used when the patient is in static mode — we already know the polarity
    (the env sampled either from `responses[]` or `denial_responses[]`), so
    "parsing" is a lookup. Findings are left empty (the static env doesn't
    track structured fields) and safety flags are all False (canned strings
    are vetted not to leak diseases / tests).

    Why this matters for ablation: in static mode R3 (interview value) and
    R8' (channel hygiene) still see a valid `parsed` object, so we can
    compare static vs dynamic with the SAME reward function — the difference
    in reward then attributable purely to parser noise, not to "R3 is
    suddenly defined now".
    """

    def parse(
        self,
        *,
        qid: str,
        qtext: str,
        utterance: str,
        canned_polarity: Polarity,
        episode_seed: int,
        step: int,
    ) -> ParsedReply:
        return ParsedReply(
            question_id=qid,
            polarity=canned_polarity,
            polarity_confidence=1.0,
            findings=[],
            safety=SafetyFlags(),
            raw_quote=utterance,
        )


class LLMParserClient(ParserClient):
    """OpenAI-compatible JSON-mode parser.

    Always greedy (parser_temperature defaults to 0.0). On any failure to
    produce a valid `ParsedReply`, falls back to an IdentityParserClient
    parse using `canned_polarity` AND records `parser_failed=True` semantics
    by setting `polarity_confidence=0.0` (which the env's confidence gate
    will then collapse to UNCLEAR).
    """

    # The system prompt MUST stay in sync with scripts/train_parser_sft.py's
    # SYSTEM_TEMPLATE. We import it lazily because the trainer module pulls
    # in heavy deps (transformers, trl, peft) we don't want at env import time.
    # Instead we inline the template here and add a unit test that asserts
    # they match.
    SYSTEM_TEMPLATE = (
        "You are a clinical-NLP parser. The patient just answered the doctor's "
        "question. Output ONLY a single JSON object matching the ParsedReply "
        "schema (schema_version=1). "
        "Set findings keys ONLY from the question's expected vocabulary. "
        "Set safety flags only when actually present in the utterance. "
        "raw_quote MUST be the verbatim patient utterance.\n\n"
        "QUESTION_ID: {qid}"
    )

    def __init__(
        self,
        cfg: PatientRuntimeConfig,
        *,
        cache: Optional[_Cache] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ):
        self.cfg = cfg
        self.cache = cache
        self._client_factory = client_factory or self._default_factory
        self._client: Any = None

    def _default_factory(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            base_url=self.cfg.parser_endpoint,
            api_key=self.cfg.parser_api_key or "EMPTY",
            timeout=self.cfg.request_timeout_s,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def parse(
        self,
        *,
        qid: str,
        qtext: str,
        utterance: str,
        canned_polarity: Polarity,
        episode_seed: int,
        step: int,
    ) -> ParsedReply:
        cache_key = (episode_seed, step, f"parser::{qid}::{hash(utterance) & 0xFFFFFFFF}")
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        sys_prompt = self.SYSTEM_TEMPLATE.format(qid=qid)
        user_msg = f'QUESTION: {qtext}\nPATIENT: {utterance}'
        seed = _deterministic_seed(episode_seed, step, qid)

        raw_text = ""
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                client = self._ensure_client()
                resp = client.chat.completions.create(
                    model=self.cfg.parser_model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.cfg.parser_temperature,
                    max_tokens=512,
                    seed=seed,
                    response_format={"type": "json_object"},
                )
                raw_text = (resp.choices[0].message.content or "").strip()
                parsed = _validate_parsed(raw_text, qid=qid, utterance=utterance)
                if self.cache is not None:
                    self.cache.set(cache_key, parsed)
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning(
                    "LLMParserClient attempt %d/%d failed for qid=%s: %s (raw=%r)",
                    attempt + 1, self.cfg.max_retries + 1, qid, exc, raw_text[:120],
                )

        if self.cfg.fallback_to_static_on_error:
            log.error("Parser LLM exhausted retries; using identity parse with confidence=0. last=%s",
                      last_exc)
            fallback = ParsedReply(
                question_id=qid,
                polarity=canned_polarity,
                polarity_confidence=0.0,
                findings=[],
                safety=SafetyFlags(),
                raw_quote=utterance,
            )
            if self.cache is not None:
                self.cache.set(cache_key, fallback)
            return fallback
        raise RuntimeError(f"parser LLM failed and fallback disabled: {last_exc}")


# ===========================================================================
# Factories
# ===========================================================================

def build_patient_client(
    cfg: PatientRuntimeConfig,
    *,
    cache: Optional[_Cache] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> PatientClient:
    if cfg.mode == "static":
        return StaticPatientClient()
    if not cfg.patient_endpoint:
        raise ValueError(
            "patient_runtime.mode='dynamic' but patient_endpoint is unset"
        )
    return LLMPatientClient(cfg, cache=cache, client_factory=client_factory)


def build_parser_client(
    cfg: PatientRuntimeConfig,
    *,
    cache: Optional[_Cache] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> ParserClient:
    if cfg.mode == "static":
        return IdentityParserClient()
    if not cfg.parser_endpoint:
        raise ValueError(
            "patient_runtime.mode='dynamic' but parser_endpoint is unset"
        )
    return LLMParserClient(cfg, cache=cache, client_factory=client_factory)


def build_cache(cfg: PatientRuntimeConfig) -> Optional[_Cache]:
    return _Cache() if cfg.cache_responses else None


# ===========================================================================
# Helpers
# ===========================================================================

def _deterministic_seed(episode_seed: int, step: int, qid: str) -> int:
    """Stable per-(episode, step, qid) integer seed in [0, 2**31)."""
    h = (episode_seed * 1_000_003 + step * 1009 + hash(qid)) & 0x7FFF_FFFF
    return int(h)


def _format_history_block(history: list[tuple[str, str, str]]) -> str:
    """history is list of (qid, qtext, utterance). Rendered for the patient
    LLM so it can stay consistent with what it already said.

    Capped at the most recent 6 Q&A pairs — enough for short-term consistency
    without blowing the patient's context window.
    """
    if not history:
        return "(none yet)"
    lines = []
    for qid, qtext, utt in history[-6:]:
        lines.append(f"Q [{qid}]: {qtext}")
        lines.append(f"A: {utt}")
    return "\n".join(lines)


def _validate_parsed(raw_text: str, *, qid: str, utterance: str) -> ParsedReply:
    """Parse the parser-LLM JSON output into a ParsedReply. Raises on bad input.

    Defends against the parser model echoing a different question_id, omitting
    raw_quote, or wrapping the JSON in code fences.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fence if present.
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    blob = json.loads(text)
    if not isinstance(blob, dict):
        raise ValueError(f"parser output is not a JSON object: {type(blob).__name__}")
    blob.setdefault("question_id", qid)
    blob.setdefault("raw_quote", utterance)
    parsed = ParsedReply.model_validate(blob)
    if parsed.question_id != qid:
        # Force-correct rather than reject — we know the right qid.
        parsed = parsed.model_copy(update={"question_id": qid})
    return parsed


__all__ = [
    "PatientRuntimeConfig",
    "PatientResponse",
    "PatientClient",
    "StaticPatientClient",
    "LLMPatientClient",
    "ParserClient",
    "IdentityParserClient",
    "LLMParserClient",
    "build_patient_client",
    "build_parser_client",
    "build_cache",
]
