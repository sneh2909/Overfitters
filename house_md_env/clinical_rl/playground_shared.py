"""Session bookkeeping + observation serialization for the playground UI.

The OpenEnv ``/reset`` and ``/step`` endpoints are stateless — each request
spins up a fresh ``HouseMDEnvironment``. The animated ER UI under ``/`` needs
*session-scoped* state instead (a single browser tab walks one patient
turn-by-turn), so ``server/playground.py`` keeps its own in-process
``SESSIONS`` dict keyed off the ``Session`` class defined here.

The serialization helper turns a ``ClinicalEnv`` ``Episode`` snapshot into the
JSON shape the front-end expects (``static/app.js``).

Both pieces are intentionally factored out so the playground module stays
small and unit-testable.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from clinical_rl.env import Action, Episode


class Session:
    """Per-browser-tab playground state.

    Each session holds its own ``Episode`` snapshot (and the ``card`` / ``rng``
    needed to restore mutation context on the *shared* ``ClinicalEnv``
    instance). The playground reattaches these fields to the shared env right
    before every step.
    """

    __slots__ = ("id", "episode", "card", "policy", "oracle_plan", "oracle_idx", "rng")

    def __init__(
        self,
        id: str,
        episode: Episode,
        card: dict[str, Any],
        policy: str,
        rng: random.Random,
    ) -> None:
        self.id = id
        self.episode = episode
        self.card = card
        self.policy = policy
        self.oracle_plan: Optional[list[Action]] = None
        self.oracle_idx = 0
        self.rng = rng


def _serialize_log_entry(entry: Any) -> dict[str, Any]:
    out = {
        "step": entry.step,
        "kind": entry.kind,
        "text": entry.text,
        "cost": entry.cost,
        "time_min": entry.time_min,
        "duplicate": entry.duplicate,
        "invalid": entry.invalid,
        "error": entry.error,
        "action": None,
    }
    if entry.action is not None:
        out["action"] = {
            "type": entry.action.type.value,
            "argument": entry.action.argument,
            "rationale": entry.action.rationale,
            "board": entry.action.board,
        }
    return out


def serialize_obs(session: Session, catalogs: Any) -> dict[str, Any]:
    """Render the session's current observation as a UI-friendly dict.

    ``catalogs`` is the loaded ``Catalogs`` object — used to resolve
    ``pending_tests`` test ids into human-readable test names.
    """
    obs = session.episode.obs
    return {
        "session_id": session.id,
        "policy": session.policy,
        "chief_complaint": obs.chief_complaint,
        "age": obs.age,
        "sex": obs.sex,
        "intake_vitals": obs.intake_vitals,
        "step": obs.step,
        "step_cap": obs.step_cap,
        "cost_so_far": obs.cost_so_far,
        "time_elapsed_min": obs.time_elapsed_min,
        "severity_signal": obs.severity_signal,
        "terminal": obs.terminal,
        "diagnosis": obs.diagnosis,
        "timed_out": obs.timed_out,
        "differential_board": obs.differential_board,
        "pending_tests": [
            {
                "test_id": pt.test_id,
                "deliver_at_step": pt.deliver_at_step,
                "steps_left": max(0, pt.deliver_at_step - obs.step),
                "test_name": (
                    catalogs.tests[pt.test_id].name
                    if pt.test_id in catalogs.tests
                    else pt.test_id
                ),
            }
            for pt in obs.pending_tests
        ],
        "action_log": [_serialize_log_entry(e) for e in obs.action_log],
    }
