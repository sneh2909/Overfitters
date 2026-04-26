"""
Pretty playground UI backend for the House M.D. OpenEnv space.

The OpenEnv server already exposes the canonical `/reset`, `/step`, `/state`
endpoints (used by RL training jobs and the Python ``HouseMDEnv`` client).
This module adds a *parallel* REST surface under ``/api/...`` that powers
the rich animated ER scene at ``/`` (see ``static/index.html``).

Why a separate API surface?
---------------------------
The OpenEnv ``/step`` contract is intentionally *stateless*: each request
spins up a fresh environment so multiple agents can be trained in parallel
without cross-talk. The visual playground, however, needs *session-scoped*
state — a single browser tab steps through one patient turn-by-turn and
expects the same patient on every poll. So we maintain our own in-process
``SESSIONS`` dict and reuse a single ``ClinicalEnv`` instance per browser
session, swapping its ``_episode`` pointer the same way the legacy
``clinical_rl/server/app.py`` did.

This is fine because the playground is a *demo* surface, not the training
API — it's single-tenant per HF Space and resets are cheap.

Endpoints registered
--------------------
::

    GET    /api/catalogs                 vocab + display metadata
    POST   /api/episodes                 reset() with optional disease/seed
    GET    /api/episodes/{sid}           current observation snapshot
    POST   /api/episodes/{sid}/actions   step() with a manual action
    POST   /api/episodes/{sid}/agent_step step() with a built-in policy
    GET    /api/episodes/{sid}/rewards   compute_all() (terminal episodes only)
    GET    /api/episodes/{sid}/truth     spectator reveal of hidden state
    DELETE /api/episodes/{sid}           drop a session
    POST   /api/tts/speak               proxy to Kokoro TTS (optional; can disable)

All routes use vendored ``clinical_rl.*`` modules — no PYTHONPATH games.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Vendored package — sits next to this file inside the env image.
from clinical_rl.tts import speak_via_tts_service, tts_disabled, voice_for_speaker
from clinical_rl.env import (
    Action,
    ActionType,
    ClinicalEnv,
    load_cards,
    load_catalogs,
)
from clinical_rl.oracle import HeuristicOracle
from clinical_rl.playground_shared import Session, serialize_obs
from clinical_rl.rewards import compute_all


# ---------------------------------------------------------------------------
# Resolve the vendored data/ directory regardless of how the package was
# installed (editable, wheel, or Docker image). The HouseMDEnvironment
# wrapper does the same dance — kept intentionally consistent.
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data",                # repo_root/data (primary)
        here.parent.parent.parent / "data",         # one level up (fallback)
        Path.cwd() / "data",
    ]
    for c in candidates:
        if c.is_dir() and (c / "diseases.yaml").is_file():
            return c
    raise RuntimeError(
        f"Could not locate the data/ directory. Tried: {[str(c) for c in candidates]}"
    )


DATA_DIR = _resolve_data_dir()
CARDS_DIR = DATA_DIR / "cards"

# Loaded once per process. ClinicalEnv has no globals; we just borrow the
# catalogs/cards so every session shares the same vocab.
CATALOGS = load_catalogs(DATA_DIR)
CARDS = load_cards(CARDS_DIR)
SHARED_ENV = ClinicalEnv(CATALOGS, CARDS)
SHARED_ORACLE = HeuristicOracle(SHARED_ENV, CATALOGS, CARDS)


# ---------------------------------------------------------------------------
# Per-browser-tab session. Each session pins its own Episode + RNG so two
# tabs can play independent patients without trampling each other.
# ---------------------------------------------------------------------------

SESSIONS: dict[str, Session] = {}


def _attach(session: Session) -> None:
    """Bind a session's episode to the shared env so step() mutates it in-place."""
    SHARED_ENV._episode = session.episode
    SHARED_ENV._card = session.card
    SHARED_ENV._rng = session.rng


# ---------------------------------------------------------------------------
# Request models — kept loose because the playground UI sends a small
# vocabulary and we want clear 422 responses for typos.
# ---------------------------------------------------------------------------

class _ResetRequest(BaseModel):
    disease: Optional[str] = None
    variant_id: Optional[str] = None
    seed: Optional[int] = None
    policy: str = Field(default="oracle", description="oracle|greedy|random|manual")


class _ActionRequest(BaseModel):
    type: str
    argument: str
    rationale: str = ""
    board: Optional[list[dict[str, Any]]] = None


class _SpeakRequest(BaseModel):
    text: str
    speaker: str = Field(description="doctor|patient")
    patient_sex: Optional[str] = None


# ---------------------------------------------------------------------------
# Serializer — convert dataclasses to UI-friendly dicts. Field shapes are
# matched 1:1 with what `static/app.js` expects. Don't rename without
# updating the JS.
# ---------------------------------------------------------------------------

def _serialize_obs(session: Session) -> dict[str, Any]:
    return serialize_obs(session, CATALOGS)


# ---------------------------------------------------------------------------
# Built-in policies the UI can run under "Auto" mode.
#  - random   : pick uniformly across action types + arguments (worst case)
#  - greedy   : a tiny scripted plan to show "reasonable but not informed"
#  - oracle   : reuse the heuristic oracle baseline (knows the truth)
# These exist so judges can hit Play and watch the scene without writing code.
# ---------------------------------------------------------------------------

def _random_policy(session: Session) -> Action:
    rng = session.rng
    atype = rng.choice(list(ActionType))
    if atype == ActionType.INTERVIEW:
        return Action(atype, rng.choice(list(CATALOGS.questions)), "random pick")
    if atype == ActionType.EXAMINE:
        return Action(atype, rng.choice(list(CATALOGS.exams)), "random pick")
    if atype == ActionType.ORDER_TEST:
        return Action(atype, rng.choice(list(CATALOGS.tests)), "random pick")
    if atype == ActionType.UPDATE_DIFFERENTIAL:
        diseases = rng.sample(list(CATALOGS.diseases), 3)
        board = [{"disease": d, "prob": round(1 / 3, 3)} for d in diseases]
        return Action(atype, "random differential", "random pick", board=board)
    return Action(atype, rng.choice(list(CATALOGS.diseases)), "random pick")


def _greedy_policy(session: Session) -> Action:
    """1 interview -> 1 exam -> 2 cheap labs -> diagnose first disease alphabetically."""
    obs = session.episode.obs
    n_actions = sum(1 for e in obs.action_log if e.kind == "action")
    diseases_sorted = sorted(CATALOGS.diseases.keys())
    if n_actions == 0:
        return Action(ActionType.INTERVIEW, "pain_location", "screen for chief symptom")
    if n_actions == 1:
        return Action(ActionType.EXAMINE, "general_appearance", "quick global look")
    if n_actions == 2:
        return Action(ActionType.ORDER_TEST, "cbc", "broad screening lab")
    if n_actions == 3:
        return Action(ActionType.ORDER_TEST, "bmp", "metabolic screen")
    return Action(ActionType.DIAGNOSE, diseases_sorted[0], "greedy commit")


def _oracle_policy(session: Session) -> Action:
    """Lazy-build the heuristic oracle plan, then walk through it step-by-step."""
    if session.oracle_plan is None:
        ep = session.episode
        rng = random.Random(ep.hidden.seed + 1_000_000)
        session.oracle_plan = SHARED_ORACLE._build_plan(
            ep.hidden.true_disease, ep.obs.age, ep.obs.sex, rng, ep.hidden.test_results
        )
    if session.oracle_idx >= len(session.oracle_plan):
        return Action(ActionType.DIAGNOSE, session.episode.hidden.true_disease, "fallback")
    action = session.oracle_plan[session.oracle_idx]
    session.oracle_idx += 1
    return action


POLICIES = {
    "random": _random_policy,
    "greedy": _greedy_policy,
    "oracle": _oracle_policy,
}


# ---------------------------------------------------------------------------
# Public registration helper — keeps the import surface tiny and gives the
# caller a chance to nest under a different prefix in the future.
# ---------------------------------------------------------------------------

def register_playground(app: FastAPI, prefix: str = "/api") -> None:
    """Mount the legacy /api/* surface that powers the static ER scene UI."""

    # ---- catalogs ----------------------------------------------------------

    @app.get(f"{prefix}/catalogs", include_in_schema=False)
    def get_catalogs() -> dict[str, Any]:
        return {
            "questions": [
                {"id": q.id, "category": q.category, "text": q.text}
                for q in CATALOGS.questions.values()
            ],
            "exams": [
                {
                    "id": e.id,
                    "category": e.category,
                    "text": e.text,
                    "cost": e.cost,
                    "time_min": e.time_min,
                }
                for e in CATALOGS.exams.values()
            ],
            "tests": [
                {
                    "id": t.id,
                    "name": t.name,
                    "category": t.category,
                    "cost": t.cost,
                    "turnaround_steps": t.turnaround_steps,
                }
                for t in CATALOGS.tests.values()
            ],
            "diseases": [
                {
                    "id": d.id,
                    "name": d.name,
                    "family": d.family,
                    "severity": d.severity,
                    "sex_allowed": list(d.sex_allowed),
                    "age_min": d.age_min,
                    "age_max": d.age_max,
                }
                for d in CATALOGS.diseases.values()
            ],
        }

    # ---- TTS (browser calls this so the Space can keep the TTS API key off-client) --

    @app.post(f"{prefix}/tts/speak", include_in_schema=False)
    def speak(req: _SpeakRequest) -> Response:
        if tts_disabled():
            return Response(status_code=204)
        text = req.text.strip()
        if not text:
            return Response(status_code=204)
        try:
            voice = voice_for_speaker(req.speaker, req.patient_sex)
            result = speak_via_tts_service(text, voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        headers: dict[str, str] = {}
        if result.duration_seconds:
            headers["X-Duration-Seconds"] = result.duration_seconds
        return Response(
            content=result.audio, media_type=result.content_type, headers=headers
        )

    # ---- episode lifecycle -------------------------------------------------

    @app.post(f"{prefix}/episodes", include_in_schema=False)
    def create_episode(req: _ResetRequest) -> dict[str, Any]:
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        SHARED_ENV.reset(disease=req.disease, variant_id=req.variant_id, seed=seed)

        sid = uuid.uuid4().hex[:12]
        session = Session(
            id=sid,
            episode=SHARED_ENV._episode,
            card=SHARED_ENV._card,
            policy=req.policy,
            rng=SHARED_ENV._rng,
        )
        SESSIONS[sid] = session
        payload = _serialize_obs(session)
        payload["seed"] = seed
        return payload

    @app.get(f"{prefix}/episodes/{{sid}}", include_in_schema=False)
    def get_episode(sid: str) -> dict[str, Any]:
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return _serialize_obs(session)

    @app.post(f"{prefix}/episodes/{{sid}}/actions", include_in_schema=False)
    def post_action(sid: str, req: _ActionRequest) -> dict[str, Any]:
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if session.episode.obs.terminal:
            return _serialize_obs(session)

        try:
            atype = ActionType(req.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown action type: {req.type}")

        action = Action(
            type=atype,
            argument=req.argument,
            rationale=req.rationale,
            board=req.board,
        )

        _attach(session)
        SHARED_ENV.step(action)
        session.episode = SHARED_ENV._episode
        return _serialize_obs(session)

    @app.post(f"{prefix}/episodes/{{sid}}/agent_step", include_in_schema=False)
    def agent_step(sid: str) -> dict[str, Any]:
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if session.episode.obs.terminal:
            return _serialize_obs(session)

        policy_fn = POLICIES.get(session.policy)
        if policy_fn is None:
            raise HTTPException(
                status_code=400,
                detail=f"policy '{session.policy}' has no autoplay implementation",
            )

        _attach(session)
        action = policy_fn(session)
        SHARED_ENV.step(action)
        session.episode = SHARED_ENV._episode

        payload = _serialize_obs(session)
        payload["last_action"] = {
            "type": action.type.value,
            "argument": action.argument,
            "rationale": action.rationale,
            "board": action.board,
        }
        return payload

    @app.get(f"{prefix}/episodes/{{sid}}/rewards", include_in_schema=False)
    def get_rewards(sid: str) -> dict[str, Any]:
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if not session.episode.obs.terminal:
            raise HTTPException(status_code=409, detail="episode not terminal yet")
        return compute_all(session.episode, session.card, CATALOGS)

    @app.get(f"{prefix}/episodes/{{sid}}/truth", include_in_schema=False)
    def get_truth(sid: str) -> dict[str, Any]:
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        h = session.episode.hidden
        disease = CATALOGS.diseases[h.true_disease]
        return {
            "true_disease_id": h.true_disease,
            "true_disease_name": disease.name,
            "severity": disease.severity,
            "family": disease.family,
            "variant_id": h.variant_id,
            "deterioration_rate": h.deterioration_rate,
            "deteriorating": h.deteriorating,
            "seed": h.seed,
        }

    @app.delete(f"{prefix}/episodes/{{sid}}", include_in_schema=False)
    def delete_episode(sid: str) -> dict[str, str]:
        SESSIONS.pop(sid, None)
        return {"status": "ok"}
