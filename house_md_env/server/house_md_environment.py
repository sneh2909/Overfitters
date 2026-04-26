"""
HouseMDEnvironment - OpenEnv wrapper around the existing :class:`ClinicalEnv`.

This module is a thin adapter:

  * the *real* RL logic (presampling, deterioration, reward rubrics) lives in
    `clinical_rl.env.env.ClinicalEnv` and `clinical_rl.rewards`
  * this wrapper translates Pydantic ⟷ dataclass at the OpenEnv interface
  * it loads the catalog YAMLs + disease cards once at startup

Why delegate rather than re-implement?
  - Every prior unit test (`tests/test_oracle.py`, `tests/test_rewards.py`,
    `tests/test_env.py`) keeps passing untouched.
  - The two layers can diverge cleanly: OpenEnv concerns (auth, schemas, JSON
    round-trips) live here; clinical concerns (disease cards, deterioration
    physics) live in `clinical_rl/`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..clinical_rl.env import (
        Action as DataclassAction,
        ActionType as DataclassActionType,
        ClinicalEnv,
        Episode,
        load_cards,
        load_catalogs,
    )
    from ..clinical_rl.rewards import compute_all
    from ..models import (
        HouseMDAction,
        HouseMDActionType,
        HouseMDObservation,
        HouseMDState,
        LogEntryModel,
        PendingTestModel,
    )
except ImportError:
    # Fall back to absolute imports when the package is mounted as the top-level
    # `house_md_env` (e.g. when launched directly via `python -m server.app`).
    from clinical_rl.env import (
        Action as DataclassAction,
        ActionType as DataclassActionType,
        ClinicalEnv,
        Episode,
        load_cards,
        load_catalogs,
    )
    from clinical_rl.rewards import compute_all
    from models import (
        HouseMDAction,
        HouseMDActionType,
        HouseMDObservation,
        HouseMDState,
        LogEntryModel,
        PendingTestModel,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Locate vendored data/ directory regardless of how the package is launched.
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    """Find the bundled `data/` (catalogs + cards) directory.

    Search order is deliberate so the env works in three contexts:
      1. installed as `openenv-house-md-env` -> data/ next to the package
      2. running uvicorn from the repo root  -> data/ next to server/
      3. dev: cwd is repo root               -> repo_root/data/
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data",                   # repo_root/data (primary)
        here.parent.parent.parent / "data",            # one level up (fallback)
        Path.cwd() / "data",
    ]
    for c in candidates:
        if (c / "diseases.yaml").exists() and (c / "cards").is_dir():
            logger.info("HouseMDEnvironment: using data dir %s", c)
            return c
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate clinical data directory. Tried:\n  - {tried}"
    )


# ---------------------------------------------------------------------------
# HouseMDEnvironment
# ---------------------------------------------------------------------------

class HouseMDEnvironment(Environment):
    """OpenEnv-compliant wrapper around :class:`ClinicalEnv`.

    Per-instance state is small (one running `Episode`), so we mark this
    concurrent-safe — `create_app(..., max_concurrent_envs=N)` will spin up N
    independent `HouseMDEnvironment` instances, each with its own RNG-seeded
    patient.

    Args:
        step_cap: Hard cap on agent actions per episode (timeout if reached).
        data_dir: Override for the catalogs+cards directory. Auto-detected if None.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        step_cap: int = 15,
        data_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._step_cap = int(step_cap)

        # Lazy / cached load: catalogs + cards parse from YAML on first
        # instantiation, then every subsequent HouseMDEnvironment instance
        # in the same process can reuse them via class-level cache.
        self._data_dir = data_dir or _resolve_data_dir()
        self._catalogs, self._cards = self._get_or_load_data(self._data_dir)

        # Per-instance ClinicalEnv keeps one episode at a time. Reusing a
        # single instance across reset()s is intentional — it's how the
        # original env was designed and how GRPO uses it.
        self._env = ClinicalEnv(self._catalogs, self._cards, step_cap=self._step_cap)
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._latest_obs: Optional[HouseMDObservation] = None

        # Eager reset on construction. OpenEnv's HTTP routes spin up a fresh
        # environment per request via the factory, so /step right after /reset
        # without a WebSocket would otherwise hit an uninitialized env and
        # raise. Calling reset() here gives every env a valid initial episode.
        # WebSocket sessions overwrite this immediately on their first /reset,
        # so there's no behavioural cost — only safety.
        self.reset()

    # --- shared catalog/card cache (one parse per process) ---
    _shared_cache: dict[Path, tuple[Any, Any]] = {}

    @classmethod
    def _get_or_load_data(cls, data_dir: Path):
        if data_dir not in cls._shared_cache:
            cls._shared_cache[data_dir] = (
                load_catalogs(data_dir),
                load_cards(data_dir / "cards"),
            )
        return cls._shared_cache[data_dir]

    # =======================================================================
    # OpenEnv API: reset
    # =======================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        disease: Optional[str] = None,
        variant_id: Optional[str] = None,
        **_: Any,
    ) -> HouseMDObservation:
        """Start a new episode. Optionally pin disease/variant/seed for evaluation.

        Args:
            seed: Optional RNG seed. Same (disease, variant, seed) reproduces
                the trajectory under a deterministic policy.
            episode_id: Optional client-supplied id; auto-generated when None.
            disease: Optional disease id; randomly drawn from the corpus when None.
            variant_id: Optional variant id ("v1"/"v2"/"v3"); random when None.
        """
        dc_obs = self._env.reset(disease=disease, variant_id=variant_id, seed=seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._latest_obs = self._to_pydantic_obs(dc_obs, terminal_just_happened=False)
        return self._latest_obs

    # =======================================================================
    # OpenEnv API: step
    # =======================================================================

    def step(  # type: ignore[override]
        self,
        action: HouseMDAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> HouseMDObservation:
        """Execute one action. Returns the updated observation; reward populated only at terminal."""
        if self._env._episode is None:  # noqa: SLF001 (intentional probe)
            raise RuntimeError("Environment not initialized — call reset() first.")

        # Pydantic action -> internal dataclass action
        dc_action = DataclassAction(
            type=DataclassActionType(action.type.value),
            argument=action.argument or "",
            rationale=action.rationale or "",
            board=action.board,
        )

        was_terminal = self._env._episode.obs.terminal  # noqa: SLF001
        dc_obs = self._env.step(dc_action)
        self._step_count += 1
        terminal_just_happened = (not was_terminal) and dc_obs.terminal

        self._latest_obs = self._to_pydantic_obs(
            dc_obs, terminal_just_happened=terminal_just_happened
        )
        return self._latest_obs

    # =======================================================================
    # OpenEnv API: state (read-only introspection)
    # =======================================================================

    @property
    def state(self) -> HouseMDState:
        ep = self._env._episode  # noqa: SLF001
        if ep is None:
            return HouseMDState(episode_id=self._episode_id, step_count=self._step_count)
        obs = ep.obs
        return HouseMDState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            chief_complaint=obs.chief_complaint,
            step=obs.step,
            step_cap=obs.step_cap,
            cost_so_far=obs.cost_so_far,
            time_elapsed_min=obs.time_elapsed_min,
            terminal=obs.terminal,
            diagnosis=obs.diagnosis,
            timed_out=obs.timed_out,
        )

    # =======================================================================
    # Internal: dataclass -> pydantic conversion
    # =======================================================================

    def _to_pydantic_obs(
        self, dc_obs, *, terminal_just_happened: bool
    ) -> HouseMDObservation:
        """Translate the internal Observation dataclass into the wire-format Pydantic model.

        When the episode just ended this turn we additionally compute the reward
        breakdown via :func:`compute_all` and surface it on `obs.rewards` plus
        `obs.reward` (the OpenEnv-standard scalar slot, populated with the weighted
        total so RL trainers can read a single number).
        """
        log_entries = [self._convert_log_entry(e) for e in dc_obs.action_log]
        pendings = [
            PendingTestModel(
                test_id=p.test_id,
                deliver_at_step=p.deliver_at_step,
                cost_paid=p.cost_paid,
            )
            for p in dc_obs.pending_tests
        ]

        rewards: Optional[dict[str, float]] = None
        scalar_reward: Optional[float] = None
        if dc_obs.terminal:
            ep: Episode = self._env._episode  # noqa: SLF001
            card = self._cards[ep.hidden.true_disease]
            try:
                rewards = compute_all(ep, card, self._catalogs)
                scalar_reward = float(rewards["total"])
            except Exception as exc:  # pragma: no cover (defensive)
                logger.exception("Reward computation failed: %s", exc)
                rewards = {"total": 0.0, "error": str(exc)}  # type: ignore[dict-item]
                scalar_reward = 0.0

        return HouseMDObservation(
            chief_complaint=dc_obs.chief_complaint,
            age=dc_obs.age,
            sex=dc_obs.sex,
            intake_vitals=dc_obs.intake_vitals,
            step=dc_obs.step,
            step_cap=dc_obs.step_cap,
            cost_so_far=dc_obs.cost_so_far,
            time_elapsed_min=dc_obs.time_elapsed_min,
            action_log=log_entries,
            pending_tests=pendings,
            differential_board=list(dc_obs.differential_board),
            severity_signal=dc_obs.severity_signal,
            terminal=dc_obs.terminal,
            diagnosis=dc_obs.diagnosis,
            timed_out=dc_obs.timed_out,
            rewards=rewards,
            done=dc_obs.terminal,
            reward=scalar_reward,
            metadata={
                "episode_id": self._episode_id,
                "terminal_just_happened": terminal_just_happened,
            },
        )

    @staticmethod
    def _convert_log_entry(e) -> LogEntryModel:
        atype: Optional[HouseMDActionType] = None
        argument = ""
        rationale = ""
        if e.action is not None:
            atype = HouseMDActionType(e.action.type.value)
            argument = e.action.argument
            rationale = e.action.rationale
        return LogEntryModel(
            step=e.step,
            kind=e.kind,
            type=atype,
            argument=argument,
            rationale=rationale,
            text=e.text,
            cost=e.cost,
            time_min=e.time_min,
            duplicate=e.duplicate,
            invalid=e.invalid,
            error=e.error,
        )
