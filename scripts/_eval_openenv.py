"""
Shared helpers for evaluation scripts that drive the House M.D. environment
through its OpenEnv server (rather than instantiating ``ClinicalEnv``
directly in-process).

Both :mod:`scripts.eval_gemini` and :mod:`scripts.eval_hf` use this module
to:

  * convert the Pydantic :class:`HouseMDObservation` returned by the OpenEnv
    client back into the dataclass :class:`Observation` shape that
    :func:`clinical_rl.prompt.render_prompt` expects, so the prompt the model
    sees is bit-identical to the in-process baseline;
  * convert a parsed dataclass :class:`Action` into a
    :class:`HouseMDAction` so it can be sent over the WebSocket;
  * read rewards and the diagnosis straight off the terminal observation
    (no second call into ``compute_all`` — the server already evaluated the
    rubrics and surfaced ``obs.rewards`` + ``obs.reward``).

The module deliberately avoids importing the heavy server (``ClinicalEnv``,
catalog YAMLs etc.) for any work other than prompt rendering — a clean
client/server split that mirrors how a real RL trainer would talk to the
env over the network.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# clinical_rl is imported only for prompt rendering + JSON parsing.
# Both are pure helpers; neither touches the env state machine.
from clinical_rl.env.state import (
    Action as DCAction,
    ActionType as DCActionType,
    LogEntry as DCLogEntry,
    Observation as DCObservation,
    PendingTest as DCPendingTest,
)
from clinical_rl.prompt import parse_action_json, render_prompt

from house_md_env import (
    HouseMDAction,
    HouseMDActionType,
    HouseMDEnv,
    HouseMDObservation,
)
from house_md_env.models import LogEntryModel, PendingTestModel


# ---------------------------------------------------------------------------
# Pydantic -> dataclass adapters (purely for prompt rendering)
# ---------------------------------------------------------------------------

def _log_entry_pyd_to_dc(e: LogEntryModel) -> DCLogEntry:
    """Pydantic LogEntryModel -> dataclass LogEntry.

    The dataclass action.type uses :class:`DCActionType`; the Pydantic
    enum is a different class with the same string values, so we rebuild
    the enum to keep ``entry.action.type == ActionType.X`` comparisons
    in :func:`render_prompt` exact.
    """
    action: Optional[DCAction] = None
    if e.type is not None:
        action = DCAction(
            type=DCActionType(e.type.value),
            argument=e.argument,
            rationale=e.rationale,
        )
    return DCLogEntry(
        step=e.step,
        kind=e.kind,
        action=action,
        text=e.text,
        cost=e.cost,
        time_min=e.time_min,
        duplicate=e.duplicate,
        invalid=e.invalid,
        error=e.error,
    )


def _pending_pyd_to_dc(p: PendingTestModel) -> DCPendingTest:
    """Pydantic PendingTestModel -> dataclass PendingTest.

    Only ``test_id`` and ``deliver_at_step`` are surfaced over the wire —
    fields like ``result_text`` / ``flag`` live in HiddenState and stay
    server-side. The renderer never reads them, so empty defaults are safe.
    """
    return DCPendingTest(
        test_id=p.test_id,
        deliver_at_step=p.deliver_at_step,
        result_text="",
        flag="",
        cost_paid=p.cost_paid,
    )


def pyd_obs_to_dataclass(pyd: HouseMDObservation) -> DCObservation:
    """Convert a wire :class:`HouseMDObservation` into the dataclass shape
    that :func:`clinical_rl.prompt.render_prompt` was written against.

    We only need this for prompt rendering — the env step/reset path
    stays on the Pydantic side end-to-end.
    """
    return DCObservation(
        chief_complaint=pyd.chief_complaint,
        age=pyd.age,
        sex=pyd.sex,
        intake_vitals=pyd.intake_vitals,
        step=pyd.step,
        step_cap=pyd.step_cap,
        cost_so_far=pyd.cost_so_far,
        time_elapsed_min=pyd.time_elapsed_min,
        action_log=[_log_entry_pyd_to_dc(e) for e in pyd.action_log],
        pending_tests=[_pending_pyd_to_dc(p) for p in pyd.pending_tests],
        differential_board=list(pyd.differential_board),
        severity_signal=pyd.severity_signal,
        terminal=pyd.terminal,
        diagnosis=pyd.diagnosis,
        timed_out=pyd.timed_out,
    )


def dc_action_to_pydantic(a: DCAction) -> HouseMDAction:
    """dataclass Action -> Pydantic HouseMDAction (wire format)."""
    return HouseMDAction(
        type=HouseMDActionType(a.type.value),
        argument=a.argument or "",
        rationale=a.rationale or "",
        board=a.board,
    )


# ---------------------------------------------------------------------------
# Eval-loop result shape
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """One eval episode's outcome — easy for the caller to .to_dict()."""

    patient_id: str
    disease: str
    difficulty: str
    rewards: dict[str, float]
    steps_taken: int
    cost: float
    diagnosis: Optional[str]
    correct: bool
    timed_out: bool
    oracle_ceiling: float
    oracle_ceiling_pct: float
    malformed_actions: int
    malformed_rate: float
    step_log: list[dict]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# OpenEnv-driven episode loop (model-agnostic)
# ---------------------------------------------------------------------------

def run_episode_openenv(
    env: HouseMDEnv,
    catalogs: Any,
    patient: dict,
    generate_fn: Callable[[str], str],
) -> EpisodeResult:
    """Walk one patient through the OpenEnv server using ``generate_fn``.

    Args:
        env: An *async* :class:`HouseMDEnv`. We never touch async here —
            the caller is expected to wrap us with the sync wrapper
            (``HouseMDEnv(...).sync()``), which forwards
            ``reset``/``step`` synchronously over its dedicated event loop.
        catalogs: Loaded :class:`Catalogs` bundle (used only for prompt
            rendering, not for stepping).
        patient: One row of ``data/eval_set.jsonl``. Required keys:
            ``patient_id``, ``disease``, ``variant_id``, ``seed``,
            ``difficulty``, ``oracle.rewards.total``.
        generate_fn: Callable that takes the rendered prompt and returns
            the model's raw output text.

    Returns:
        :class:`EpisodeResult` with rewards taken straight from the terminal
        observation (the OpenEnv server already ran the rubrics).
    """
    disease = patient["disease"]
    variant_id = patient["variant_id"]
    seed = patient["seed"]

    # Pin (disease, variant, seed) so the trajectory is identical to what
    # eval_hf used to do via ClinicalEnv.reset(disease, variant_id, seed).
    result = env.reset(seed=seed, disease=disease, variant_id=variant_id)
    obs: HouseMDObservation = result.observation
    step_log: list[dict] = []
    malformed_count = 0

    while not obs.terminal:
        prompt = render_prompt(pyd_obs_to_dataclass(obs), catalogs)
        raw_out = generate_fn(prompt)

        try:
            dc_action = parse_action_json(raw_out)
            malformed = False
        except Exception:  # noqa: BLE001 (parse_action_json may raise anything)
            # Parser failed — fall back to a safe no-op so the episode
            # progresses (the env will mark the step invalid). Reward
            # counting is unaffected; we just track the rate.
            dc_action = DCAction(type=DCActionType.INTERVIEW, argument="__parse_error__")
            malformed = True
            malformed_count += 1

        step_log.append({
            "step":        obs.step,
            "action_type": dc_action.type.value,
            "argument":    dc_action.argument,
            "raw_output":  raw_out[:300],
            "malformed":   malformed,
        })

        result = env.step(dc_action_to_pydantic(dc_action))
        obs = result.observation

    # The server computes rewards on terminal and surfaces them on the
    # observation — no separate compute_all() pass needed.
    rewards = obs.rewards or {"total": 0.0}
    rewards = {k: round(float(v), 4) for k, v in rewards.items() if isinstance(v, (int, float))}

    steps_taken = sum(1 for e in obs.action_log if e.kind == "action")
    ceiling = patient["oracle"]["rewards"]["total"]
    ceiling_pct = round(100 * rewards.get("total", 0.0) / ceiling, 1) if ceiling else 0.0

    return EpisodeResult(
        patient_id=patient["patient_id"],
        disease=disease,
        difficulty=patient["difficulty"],
        rewards=rewards,
        steps_taken=steps_taken,
        cost=round(float(obs.cost_so_far), 2),
        diagnosis=obs.diagnosis,
        correct=obs.diagnosis == disease,
        timed_out=obs.timed_out,
        oracle_ceiling=round(ceiling, 4),
        oracle_ceiling_pct=ceiling_pct,
        malformed_actions=malformed_count,
        malformed_rate=round(malformed_count / max(steps_taken, 1), 3),
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# Server bootstrap helpers (used by eval_hf.py inside the HF Jobs container)
# ---------------------------------------------------------------------------

def wait_for_health(base_url: str, timeout_s: float = 60.0) -> None:
    """Poll ``/health`` until it returns 200 or ``timeout_s`` elapses.

    Used right after we ``Popen(uvicorn ...)`` so the eval doesn't try to
    open a WebSocket against a not-yet-bound port.
    """
    deadline = time.monotonic() + timeout_s
    health_url = base_url.rstrip("/") + "/health"
    last_err: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
                last_err = RuntimeError(f"/health returned HTTP {resp.status}")
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(
        f"Env server at {base_url} did not become healthy in {timeout_s:.0f}s"
        f" (last error: {last_err!r})"
    )


def find_free_port(default: int = 8000) -> int:
    """Return ``default`` if it's free, else ask the OS for a fresh port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_uvicorn(
    *,
    app: str,
    cwd: Path,
    port: int,
    pythonpath: Optional[Path] = None,
    log_path: Optional[Path] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> subprocess.Popen:
    """Start ``uvicorn <app> --host 127.0.0.1 --port <port>`` in the background.

    Returns the live :class:`subprocess.Popen` so the caller can ``terminate()``
    it during teardown. stderr is merged into stdout and redirected to
    ``log_path`` if given (recommended on HF Jobs so server crashes show up
    in the job log file rather than vanishing).
    """
    import os as _os

    env = _os.environ.copy()
    if pythonpath is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{pythonpath}:{existing}" if existing else str(pythonpath)
        )
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable, "-m", "uvicorn", app,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "warning",
    ]

    out_handle = (
        open(log_path, "w", buffering=1) if log_path is not None else subprocess.DEVNULL
    )
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=out_handle,
        stderr=subprocess.STDOUT,
    )
