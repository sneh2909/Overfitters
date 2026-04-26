"""
House M.D. environment client.

This module exposes :class:`HouseMDEnv`, a thin :class:`EnvClient` subclass
that lets RL training code interact with the House M.D. server with the
same surface as any other OpenEnv:

    >>> from house_md_env import HouseMDEnv, HouseMDAction
    >>> with HouseMDEnv(base_url="http://localhost:8000") as env:
    ...     result = env.reset(seed=42)
    ...     obs = result.observation
    ...     print(obs.chief_complaint, obs.intake_vitals)
    ...
    ...     # Walk through one full episode...
    ...     for _ in range(5):
    ...         result = env.step(HouseMDAction(
    ...             type="INTERVIEW", argument="pain_location",
    ...             rationale="locate the pain",
    ...         ))
    ...         if result.done:
    ...             print("Final reward:", result.reward)
    ...             break

Or pull a Hugging Face Space:

    >>> with HouseMDEnv.from_hub("SnehShah/house-md-env") as env: ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    HouseMDAction,
    HouseMDActionType,
    HouseMDObservation,
    HouseMDState,
    LogEntryModel,
    PendingTestModel,
)


class HouseMDEnv(EnvClient[HouseMDAction, HouseMDObservation, HouseMDState]):
    """Client for the House M.D. clinical-reasoning environment server.

    Inherits the standard OpenEnv lifecycle helpers:
      * ``reset(seed=..., episode_id=..., **extra)`` for ``POST /reset``
      * ``step(action)``                             for ``POST /step``
      * ``state()``                                  for ``GET  /state``
      * ``from_docker_image(...)`` / ``from_hub(...)`` for one-line spinup

    The three :meth:`_step_payload`, :meth:`_parse_result`, :meth:`_parse_state`
    overrides below are what teach the generic :class:`EnvClient` how to talk
    to *our* server's specific JSON shapes.
    """

    # =======================================================================
    # Outbound: HouseMDAction -> JSON
    # =======================================================================

    def _step_payload(self, action: HouseMDAction) -> Dict[str, Any]:
        """Convert a :class:`HouseMDAction` to the JSON body for ``POST /step``.

        The server-side validator is strict (``extra='forbid'``), so we send
        exactly the fields the server expects and nothing else.
        """
        # Allow callers to pass either the enum or the raw string; normalize once.
        atype = action.type if isinstance(action.type, HouseMDActionType) else HouseMDActionType(action.type)
        payload: Dict[str, Any] = {
            "type": atype.value,
            "argument": action.argument or "",
            "rationale": action.rationale or "",
        }
        if action.board is not None:
            payload["board"] = action.board
        return payload

    # =======================================================================
    # Inbound: server JSON -> StepResult[HouseMDObservation]
    # =======================================================================

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[HouseMDObservation]:
        """Parse a server response (from /reset or /step) into a typed StepResult."""
        obs_data: Dict[str, Any] = payload.get("observation", {}) or {}

        action_log: List[LogEntryModel] = [
            LogEntryModel(**entry)
            for entry in obs_data.get("action_log", [])
        ]
        pending_tests: List[PendingTestModel] = [
            PendingTestModel(**entry)
            for entry in obs_data.get("pending_tests", [])
        ]

        observation = HouseMDObservation(
            chief_complaint=obs_data.get("chief_complaint", ""),
            age=obs_data.get("age", 0),
            sex=obs_data.get("sex", "unknown"),
            intake_vitals=obs_data.get("intake_vitals", ""),
            step=obs_data.get("step", 1),
            step_cap=obs_data.get("step_cap", 15),
            cost_so_far=obs_data.get("cost_so_far", 0),
            time_elapsed_min=obs_data.get("time_elapsed_min", 0),
            action_log=action_log,
            pending_tests=pending_tests,
            differential_board=obs_data.get("differential_board", []) or [],
            severity_signal=obs_data.get("severity_signal", "stable"),
            terminal=obs_data.get("terminal", False),
            diagnosis=obs_data.get("diagnosis"),
            timed_out=obs_data.get("timed_out", False),
            rewards=obs_data.get("rewards"),
            done=payload.get("done", obs_data.get("terminal", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    # =======================================================================
    # Inbound: server JSON -> HouseMDState
    # =======================================================================

    def _parse_state(self, payload: Dict[str, Any]) -> HouseMDState:
        """Parse the response body of ``GET /state`` into :class:`HouseMDState`."""
        return HouseMDState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            chief_complaint=payload.get("chief_complaint", ""),
            step=payload.get("step", 1),
            step_cap=payload.get("step_cap", 15),
            cost_so_far=payload.get("cost_so_far", 0),
            time_elapsed_min=payload.get("time_elapsed_min", 0),
            terminal=payload.get("terminal", False),
            diagnosis=payload.get("diagnosis"),
            timed_out=payload.get("timed_out", False),
        )

    # =======================================================================
    # Convenience helpers
    # =======================================================================

    def reset_patient(
        self,
        disease: Optional[str] = None,
        variant_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> StepResult[HouseMDObservation]:
        """Sugar for :meth:`reset` that exposes the env-specific knobs by name.

        Same effect as ``self.reset(seed=seed, episode_id=episode_id,
        disease=..., variant_id=...)`` but reads more clearly at call sites
        in evaluation scripts.
        """
        kwargs: Dict[str, Any] = {}
        if disease is not None:
            kwargs["disease"] = disease
        if variant_id is not None:
            kwargs["variant_id"] = variant_id
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)
