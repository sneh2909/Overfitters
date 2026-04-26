"""
ClinicalEnv: the diagnostic-reasoning RL environment.

Public surface (Gym-style):
    env = ClinicalEnv(catalogs, cards, step_cap=15)
    obs = env.reset(disease="ectopic_pregnancy", seed=42)
    obs = env.step(Action(...))
    state = env.state()

Lifecycle of one episode
------------------------
reset(disease, variant, seed):
  1. Pick a disease (or honour the one passed in).
  2. Pick one of its 3 presentation variants.
  3. Seed an RNG and PRE-SAMPLE every possible outcome:
       - 25 INTERVIEW responses (Bernoulli on prob, then sample)
       - 15 EXAMINE findings
       - 35 test results (Bernoulli on prob_abnormal, then sample)
     Caching here is what makes "same patient, same question = same answer".
  4. Build the initial Observation (chief complaint, age/sex, intake vitals).

step(action):
  1. Reject further actions if the episode is already terminal.
  2. Validate + execute the action — charge cost, advance the clock, append
     a LogEntry. Track duplicates.
  3. Resolve any pending test results whose `deliver_at_step` <= current step.
  4. Roll the deterioration Bernoulli (once it flips, it stays flipped).
  5. Either terminate (DIAGNOSE called or step-cap hit) or advance step+1.

Why pre-sample at reset (not lazy on each step):
  - Determinism per (disease, variant, seed) — a unit test can replay an
    episode exactly and assert the trajectory.
  - Same question asked twice in one episode -> same answer (the duplicate
    rule). Pre-sampling makes this fall out automatically; we just look up.
  - Evaluation reproducibility: the held-out eval set (P4) needs frozen seeds
    so before/after model comparisons aren't muddied by sampling noise.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from clinical_rl.parser_schema import Polarity
from clinical_rl.patient_io import (
    PatientClient,
    PatientRuntimeConfig,
    ParserClient,
    build_cache,
    build_parser_client,
    build_patient_client,
)

from .cards import Variant, select_variant
from .catalogs import Catalogs
from .state import (
    Action,
    ActionType,
    Episode,
    HiddenState,
    LogEntry,
    Observation,
    PendingTest,
)


# ---------------------------------------------------------------------------
# Time charged per action type (in addition to per-exam time_min from catalog)
# ---------------------------------------------------------------------------
# Rationale:
#   INTERVIEW: a quick question — 1 min nudge so spamming questions still
#     advances the clock visibly.
#   EXAMINE:   per-exam time_min comes from the catalog (vitals=2, pelvic=5).
#   ORDER_TEST: 0 — placing the order is instant; the wait is captured by
#     turnaround_steps. Adding minutes here would double-count.
#   UPDATE_DIFFERENTIAL / DIAGNOSE: 0 — pure cognitive actions.
_INTERVIEW_TIME_MIN = 1


class ClinicalEnv:
    """The diagnostic-reasoning environment.

    Stateless across episodes — `reset()` rebuilds Observation and HiddenState
    from scratch each time. The single instance can be re-used across many
    rollouts, which is what GRPO does (8 rollouts per patient, all starting
    from the same `reset(disease=X, seed=Y)`).
    """

    def __init__(
        self,
        catalogs: Catalogs,
        cards: dict[str, dict[str, Any]],
        step_cap: int = 15,
        instant_tests: bool = False,
        patient_runtime: Optional[PatientRuntimeConfig] = None,
        *,
        patient_client: Optional[PatientClient] = None,
        parser_client: Optional[ParserClient] = None,
    ) -> None:
        """ClinicalEnv constructor.

        Parameters
        ----------
        patient_runtime
            Toggles dynamic-patient mode and configures the LLM endpoints.
            Default = `PatientRuntimeConfig()` = static, no LLMs, no network.
            Existing trainers and tests don't pass this and continue to behave
            exactly as before.
        patient_client / parser_client
            Optional injection points for tests. When provided, override the
            clients built from `patient_runtime`. Lets test_dynamic_patient.py
            stub the LLM with a Mock and assert invariants without HTTP.
        """
        self.catalogs = catalogs
        self.cards = cards
        self.step_cap = step_cap
        self.instant_tests = instant_tests

        self.patient_runtime = patient_runtime or PatientRuntimeConfig()
        # Per-env-instance cache (cleared on reset()).
        self._patient_cache = build_cache(self.patient_runtime)
        self._patient_client = patient_client or build_patient_client(
            self.patient_runtime, cache=self._patient_cache,
        )
        self._parser_client = parser_client or build_parser_client(
            self.patient_runtime, cache=self._patient_cache,
        )

        # Filled by reset()
        self._episode: Optional[Episode] = None
        self._rng: Optional[random.Random] = None
        self._card: Optional[dict[str, Any]] = None
        self._variant: Optional[Variant] = None

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        disease: Optional[str] = None,
        variant_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """Start a new episode. Returns the initial Observation.

        Parameters
        ----------
        disease
            Disease id to load. If None, picks uniformly at random from cards.
        variant_id
            Specific variant_id ("v1"/"v2"/"v3"). If None, picks at random.
        seed
            RNG seed. Same seed + same (disease, variant) = identical episode
            trajectory under a deterministic policy.
        """
        # --- 1. Seed the RNG that drives EVERY draw in this episode ---
        if seed is None:
            # Auto-seed but record what was used so we can reproduce.
            seed = random.randrange(2**31)
        rng = random.Random(seed)

        # --- 2. Choose disease ---
        if disease is None:
            disease = rng.choice(list(self.cards.keys()))
        if disease not in self.cards:
            raise ValueError(
                f"unknown disease {disease!r}; "
                f"available: {sorted(self.cards.keys())}"
            )
        card = self.cards[disease]

        # --- 3. Choose variant ---
        if variant_id is not None:
            matching = [v for v in card["presentation_variants"] if v["variant_id"] == variant_id]
            if not matching:
                raise ValueError(f"unknown variant {variant_id!r} for disease {disease!r}")
            chosen_dict = matching[0]
            variant = Variant(
                variant_id=chosen_dict["variant_id"],
                variant_type=chosen_dict["variant_type"],
                age=int(chosen_dict["age"]),
                sex=chosen_dict["sex"],
                chief_complaint=chosen_dict["chief_complaint"],
                patient_card=chosen_dict["patient_card"],
            )
        else:
            variant = select_variant(card, rng)

        # --- 4. Pre-sample every possible response/finding/test result ---
        # See module docstring for why this is up-front instead of lazy.
        interview_responses, interview_polarities = self._presample_interviews(card, rng)
        exam_findings = self._presample_exams(card, rng)
        test_results = self._presample_tests(card, rng)

        # --- 5. Build hidden state ---
        disease_meta = self.catalogs.diseases[disease]
        hidden = HiddenState(
            true_disease=disease,
            variant_id=variant.variant_id,
            deterioration_rate=disease_meta.deterioration_rate,
            seed=seed,
            interview_responses=interview_responses,
            interview_polarities=interview_polarities,
            exam_findings=exam_findings,
            test_results=test_results,
        )

        # Clear per-episode cache so a new patient doesn't see the previous
        # patient's cached LLM utterances. The cache is keyed by
        # (episode_seed, step, qid) so collisions across episodes are unlikely
        # but possible if the user passes the same seed twice — clearing
        # makes the contract obvious.
        if self._patient_cache is not None:
            self._patient_cache.clear()

        # --- 6. Build initial observation ---
        # Intake vitals: same finding the agent would see if they later
        # EXAMINE(vital_signs). We prepend it for free as a triage signal.
        intake_vitals = exam_findings.get("vital_signs", "Vitals deferred at intake.")

        obs = Observation(
            chief_complaint=variant.chief_complaint,
            age=variant.age,
            sex=variant.sex,
            intake_vitals=intake_vitals,
            step=1,                        # next action will be step 1
            step_cap=self.step_cap,
            cost_so_far=0,
            time_elapsed_min=0,
        )

        # Cache for step()
        self._card = card
        self._variant = variant
        self._rng = rng
        self._episode = Episode(obs=obs, hidden=hidden)

        return obs

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(self, action: Action) -> Observation:
        """Process one action and return the updated Observation.

        Calling step() on a terminal episode is a no-op that returns the
        existing observation unchanged — this matches the Gym convention
        where well-behaved callers check `obs.terminal` and stop.
        """
        if self._episode is None:
            raise RuntimeError("must call reset() before step()")
        ep = self._episode

        if ep.obs.terminal:
            return ep.obs

        current_step = ep.obs.step

        # --- 1. Dispatch on action type ---
        # Each handler appends LogEntry(s) and updates obs.cost_so_far / time.
        # Validation lives inside each handler so error LogEntries carry the
        # right "kind" / `invalid` flags for R8 (format reward) to read later.
        if action.type == ActionType.INTERVIEW:
            self._handle_interview(action, current_step)
        elif action.type == ActionType.EXAMINE:
            self._handle_examine(action, current_step)
        elif action.type == ActionType.ORDER_TEST:
            self._handle_order_test(action, current_step)
        elif action.type == ActionType.UPDATE_DIFFERENTIAL:
            self._handle_update_differential(action, current_step)
        elif action.type == ActionType.DIAGNOSE:
            self._handle_diagnose(action, current_step)
        else:
            # ActionType is an Enum so this is unreachable, but defensive
            # coding here means future additions don't silently no-op.
            raise ValueError(f"unhandled action type: {action.type!r}")

        # --- 2. Resolve any pending test results that have arrived ---
        # MUST run after action handling so that turnaround=0 tests ordered
        # this step show their result in the same step's log.
        self._resolve_pending(current_step)

        # --- 3. Deterioration roll ---
        # Each step is one Bernoulli draw; once flipped, stays flipped forever.
        # Done even on free actions (UPDATE/DIAGNOSE) since the patient's
        # clinical state advances with wall-clock, not the agent's choice.
        if not ep.hidden.deteriorating and self._rng.random() < ep.hidden.deterioration_rate:
            ep.hidden.deteriorating = True
            ep.obs.severity_signal = "deteriorating"

        # --- 4. Termination checks ---
        if ep.obs.terminal:
            # DIAGNOSE handler already set this; nothing more to do.
            return ep.obs

        if current_step >= self.step_cap:
            # Used the last step without committing. R7 will penalize.
            ep.obs.terminal = True
            ep.obs.timed_out = True
            return ep.obs

        # --- 5. Advance to next step ---
        ep.obs.step = current_step + 1
        return ep.obs

    # -----------------------------------------------------------------------
    # Read-only state accessor (for OpenEnv-style polling)
    # -----------------------------------------------------------------------

    def state(self) -> Observation:
        if self._episode is None:
            raise RuntimeError("must call reset() before state()")
        return self._episode.obs

    # =======================================================================
    # Internal: pre-sampling at reset time
    # =======================================================================

    def _presample_interviews(
        self, card: dict[str, Any], rng: random.Random
    ) -> tuple[dict[str, str], dict[str, str]]:
        """For every question id in the catalog, decide what THIS patient
        will say if asked.

        Two-stage draw:
          (a) Bernoulli(prob)  — does the patient have this symptom at all?
          (b) sample one string from `responses` (yes) or `denial_responses` (no)

        Returns (responses, polarities) where polarities[qid] is the str
        value of the Polarity enum ("yes" / "no") indicating which pool the
        response was drawn from. Polarity is consumed by the IdentityParser-
        Client (static mode) so reward shaping has the same data shape it
        would get from the LLM parser in dynamic mode — making static and
        dynamic directly comparable.

        Cards are validated to contain all 25 question ids in
        symptom_distribution, so missing entries are unexpected — we fall
        back to a generic denial just to keep the env running.
        """
        out: dict[str, str] = {}
        polarities: dict[str, str] = {}
        symptoms = card.get("symptom_distribution", {})
        # Sort for determinism — dict iteration order is insertion-defined,
        # but YAML round-tripping plus dict() construction can shuffle it.
        for qid in sorted(self.catalogs.questions.keys()):
            entry = symptoms.get(qid)
            if not entry:
                out[qid] = "I haven't really noticed anything like that."
                polarities[qid] = Polarity.NO.value
                continue
            prob = float(entry.get("prob", 0.0))
            if rng.random() < prob:
                pool = entry.get("responses") or ["(no detail given)"]
                polarities[qid] = Polarity.YES.value
            else:
                pool = entry.get("denial_responses") or ["No, nothing like that."]
                polarities[qid] = Polarity.NO.value
            out[qid] = rng.choice(pool)
        return out, polarities

    def _presample_exams(
        self, card: dict[str, Any], rng: random.Random
    ) -> dict[str, str]:
        """For every exam id, pre-sample one finding string. No Bernoulli —
        an exam always returns a finding (even if it's 'unremarkable')."""
        out: dict[str, str] = {}
        exam_dist = card.get("exam_distribution", {})
        for eid in sorted(self.catalogs.exams.keys()):
            entry = exam_dist.get(eid)
            if not entry:
                out[eid] = "Exam unremarkable."
                continue
            findings = entry.get("findings") or ["Exam unremarkable."]
            out[eid] = rng.choice(findings)
        return out

    def _presample_tests(
        self, card: dict[str, Any], rng: random.Random
    ) -> dict[str, tuple[str, str]]:
        """For every test id, pre-sample (value_text, flag).

        Disease cards only list test_sensitivities for tests CLINICALLY
        RELEVANT to that disease. For everything else we return a generic
        normal — this is what makes ordering ct_head on a UTI patient simply
        wasteful (you pay $400 for "Within normal limits") rather than a hard
        error. Cost discipline is a learned signal, not a coding constraint.
        """
        out: dict[str, tuple[str, str]] = {}
        sens = card.get("test_sensitivities", {})
        for tid in sorted(self.catalogs.tests.keys()):
            entry = sens.get(tid)
            if not entry:
                out[tid] = ("Within normal limits.", "N")
                continue
            prob_abnormal = float(entry.get("prob_abnormal", 0.0))
            if rng.random() < prob_abnormal:
                pool = entry.get("values_abnormal") or [{"value": "Abnormal", "flag": "H"}]
                chosen = rng.choice(pool)
                out[tid] = (str(chosen.get("value", "Abnormal")), str(chosen.get("flag", "H")))
            else:
                normal_value = entry.get("value_normal", "Within normal limits.")
                out[tid] = (str(normal_value), "N")
        return out

    # =======================================================================
    # Internal: action handlers
    # =======================================================================

    def _bump_used(self, action_type: ActionType, argument: str) -> bool:
        """Increment the duplicate counter and return True iff this is
        the SECOND-or-later use of (action_type, argument). Used to flag
        repeat actions on the LogEntry."""
        ep = self._episode
        key = (action_type.value, argument)
        prev = ep.hidden.used_arguments.get(key, 0)
        ep.hidden.used_arguments[key] = prev + 1
        return prev > 0

    def _append_log(self, entry: LogEntry) -> None:
        """Push a LogEntry and update aggregate cost/time on the obs."""
        ep = self._episode
        ep.obs.action_log.append(entry)
        ep.obs.cost_so_far += entry.cost
        ep.obs.time_elapsed_min += entry.time_min

    # ---- INTERVIEW ----

    def _handle_interview(self, action: Action, step: int) -> None:
        """Run one INTERVIEW step.

        Flow (PLAN_PATIENT_LLM.md §3):
          1. Validate qid (unchanged).
          2. Patient client produces an utterance. In static mode this is
             just the canned response; in dynamic mode it's an LLM call with
             the canned string available as a fallback.
          3. Parser client produces a `ParsedReply`. In static mode this is
             a no-LLM identity parse using `interview_polarities[qid]`; in
             dynamic mode the parser LLM is invoked.
          4. Confidence gate: low-confidence polarity is collapsed to UNCLEAR
             so noisy parses don't earn / cost reward shaping unfairly.
          5. Append a LogEntry. The doctor-visible `text` field is identical
             to the static-mode format ("Q (qid): qtext\\nA: utt") — the
             ParsedReply lives on the server-only `parsed` field.

        Invariants:
          - The doctor's `action.rationale` is NEVER passed to either client.
            The client APIs don't accept it. (Tested in test_dynamic_patient.)
          - The parser sees only `(qid, qtext, utterance)` — never history.
        """
        ep = self._episode
        qid = action.argument
        if not self.catalogs.is_valid_question(qid):
            self._append_log(LogEntry(
                step=step, kind="action", action=action,
                text=f"(invalid INTERVIEW: '{qid}' is not a known question id)",
                invalid=True,
                error=f"unknown question id: {qid!r}",
            ))
            return

        duplicate = self._bump_used(action.type, qid)
        canned = ep.hidden.interview_responses.get(qid, "No response available.")
        canned_polarity = Polarity(
            ep.hidden.interview_polarities.get(qid, Polarity.NO.value)
        )
        question_text = self.catalogs.questions[qid].text

        # 1. Patient utterance.
        patient_resp = self._patient_client.respond(
            qid=qid,
            qtext=question_text,
            canned=canned,
            canned_polarity=canned_polarity,
            persona=self._variant.patient_card if self._variant else "",
            history=self._recent_interview_history(),
            episode_seed=ep.hidden.seed,
            step=step,
        )

        # 2. Parser.
        parsed = self._parser_client.parse(
            qid=qid,
            qtext=question_text,
            utterance=patient_resp.text,
            canned_polarity=canned_polarity,
            episode_seed=ep.hidden.seed,
            step=step,
        )

        # 3. Confidence gate. Below threshold -> UNCLEAR (kills R3 credit
        #    for that step without rewarding NO either).
        if parsed.polarity_confidence < self.patient_runtime.parser_min_confidence:
            parsed = parsed.model_copy(update={"polarity": Polarity.UNCLEAR})

        # 4. Render the doctor-visible string. Identical format to the
        #    static-mode env so prompt rendering doesn't need to change.
        text = f"Q ({qid}): {question_text}\nA: {patient_resp.text}"
        self._append_log(LogEntry(
            step=step, kind="action", action=action,
            text=text,
            cost=0,
            time_min=_INTERVIEW_TIME_MIN,
            duplicate=duplicate,
            parsed=parsed,
            parser_failed=patient_resp.parser_failed,
            patient_source=patient_resp.source,
        ))

    def _recent_interview_history(self) -> list[tuple[str, str, str]]:
        """Return the most recent INTERVIEW (qid, qtext, utterance) tuples
        from the action log. Used to keep the patient LLM consistent with
        what it has already said in this episode.

        The doctor's `rationale` and any non-INTERVIEW log entries are
        deliberately filtered out — the patient must not see test results
        or differential reasoning.
        """
        ep = self._episode
        if ep is None:
            return []
        out: list[tuple[str, str, str]] = []
        for entry in ep.obs.action_log:
            if (
                entry.kind != "action"
                or entry.action is None
                or entry.action.type != ActionType.INTERVIEW
                or entry.invalid
            ):
                continue
            qid = entry.action.argument
            qtext = ""
            if self.catalogs.is_valid_question(qid):
                qtext = self.catalogs.questions[qid].text
            # entry.text is "Q (qid): qtext\nA: utterance" — the utterance
            # is everything after the first "A: ".
            marker = "\nA: "
            idx = entry.text.find(marker)
            utterance = entry.text[idx + len(marker):] if idx != -1 else ""
            out.append((qid, qtext, utterance))
        return out

    # ---- EXAMINE ----

    def _handle_examine(self, action: Action, step: int) -> None:
        ep = self._episode
        eid = action.argument
        if not self.catalogs.is_valid_exam(eid):
            self._append_log(LogEntry(
                step=step, kind="action", action=action,
                text=f"(invalid EXAMINE: '{eid}' is not a known exam id)",
                invalid=True,
                error=f"unknown exam id: {eid!r}",
            ))
            return

        duplicate = self._bump_used(action.type, eid)
        finding = ep.hidden.exam_findings.get(eid, "No findings recorded.")
        exam = self.catalogs.exams[eid]
        text = f"Exam ({eid} — {exam.text}):\n{finding}"
        self._append_log(LogEntry(
            step=step, kind="action", action=action,
            text=text,
            cost=exam.cost,
            time_min=exam.time_min,
            duplicate=duplicate,
        ))

    # ---- ORDER_TEST ----

    def _handle_order_test(self, action: Action, step: int) -> None:
        ep = self._episode
        tid = action.argument
        if not self.catalogs.is_valid_test(tid):
            self._append_log(LogEntry(
                step=step, kind="action", action=action,
                text=f"(invalid ORDER_TEST: '{tid}' is not a known test id)",
                invalid=True,
                error=f"unknown test id: {tid!r}",
            ))
            return

        duplicate = self._bump_used(action.type, tid)
        test = self.catalogs.tests[tid]
        value, flag = ep.hidden.test_results.get(tid, ("Result unavailable.", "N"))
        turnaround_steps = 0 if self.instant_tests else test.turnaround_steps

        # Charge the cost AT ORDER TIME (matches real billing). Wait time is
        # captured by deliver_at_step, not by minutes-on-clock.
        deliver_at = step + turnaround_steps
        ep.obs.pending_tests.append(PendingTest(
            test_id=tid,
            deliver_at_step=deliver_at,
            result_text=value,
            flag=flag,
            cost_paid=test.cost,
        ))

        if turnaround_steps == 0:
            wait_msg = "result available this step"
        elif turnaround_steps == 1:
            wait_msg = "result expected next step"
        else:
            wait_msg = f"result expected in {turnaround_steps} steps"

        text = f"Ordered: {test.name} (${test.cost}) — {wait_msg}."
        self._append_log(LogEntry(
            step=step, kind="action", action=action,
            text=text,
            cost=test.cost,
            time_min=0,
            duplicate=duplicate,
        ))

    # ---- UPDATE_DIFFERENTIAL ----

    def _handle_update_differential(self, action: Action, step: int) -> None:
        ep = self._episode
        board = action.board or []

        # Validate: every disease must be in the corpus, probs in [0,1],
        # total must be near 1.0. Invalid boards still get accepted (no
        # crash) but flagged so R8 (format reward) can penalize.
        invalid = False
        error = ""
        if not isinstance(board, list) or not board:
            invalid = True
            error = "board must be a non-empty list of {disease, prob} entries"
        else:
            total = 0.0
            for entry in board:
                if not isinstance(entry, dict):
                    invalid = True
                    error = "each board entry must be a dict"
                    break
                did = entry.get("disease")
                prob = entry.get("prob")
                if did not in self.catalogs.diseases:
                    invalid = True
                    error = f"unknown disease in board: {did!r}"
                    break
                try:
                    prob_f = float(prob)
                except (TypeError, ValueError):
                    invalid = True
                    error = f"prob must be numeric, got {prob!r}"
                    break
                if not (0.0 <= prob_f <= 1.0):
                    invalid = True
                    error = f"prob out of range [0,1]: {prob_f}"
                    break
                total += prob_f
            if not invalid and abs(total - 1.0) > 0.05:
                invalid = True
                error = f"board probabilities sum to {total:.3f}, expected ~1.0"

        # Always store what the agent sent (even if invalid) — the prompt
        # should reflect the agent's most recent claim. Format reward
        # punishes invalid boards but the env doesn't reject them.
        ep.obs.differential_board = board

        # Build a one-line summary for the log: "top: disease (p=...)".
        if board and not invalid:
            top = max(board, key=lambda e: float(e.get("prob", 0.0)))
            summary = f"Differential updated: top={top['disease']} (p={float(top['prob']):.2f})"
        else:
            summary = "Differential updated (malformed)"

        self._append_log(LogEntry(
            step=step, kind="action", action=action,
            text=summary,
            cost=0,
            time_min=0,
            invalid=invalid,
            error=error,
        ))

    # ---- DIAGNOSE ----

    def _handle_diagnose(self, action: Action, step: int) -> None:
        """Terminal action. Records the agent's final answer and ends the
        episode — even if the answer is wrong or out-of-vocab. The env does
        NOT reveal correctness here; that's R1's job at episode-end."""
        ep = self._episode
        did = action.argument
        invalid = not self.catalogs.is_valid_disease(did)

        if invalid:
            text = f"Final diagnosis: '{did}' (NOT a known disease id)"
        else:
            text = f"Final diagnosis: {did}"

        self._append_log(LogEntry(
            step=step, kind="action", action=action,
            text=text,
            cost=0,
            time_min=0,
            invalid=invalid,
            error="" if not invalid else f"unknown disease id: {did!r}",
        ))

        ep.obs.terminal = True
        # Store whatever the agent said (even if invalid). R1 will compare
        # against true_disease and award 0 for mismatch / unknown.
        ep.obs.diagnosis = did

    # =======================================================================
    # Internal: pending-test resolution
    # =======================================================================

    def _resolve_pending(self, current_step: int) -> None:
        """Walk pending_tests; for each whose deliver_at_step <= current_step,
        emit a "result" LogEntry and remove from the queue.

        The cost was already charged when the test was ORDERED, so this
        block adds neither cost nor time — it just surfaces the result.
        """
        ep = self._episode
        still_pending: list[PendingTest] = []
        for pt in ep.obs.pending_tests:
            if pt.deliver_at_step <= current_step:
                test = self.catalogs.tests[pt.test_id]
                text = f"Result — {test.name}: {pt.result_text} [flag: {pt.flag}]"
                self._append_log(LogEntry(
                    step=current_step,
                    kind="result",
                    action=None,
                    text=text,
                    cost=0,
                    time_min=0,
                ))
            else:
                still_pending.append(pt)
        ep.obs.pending_tests = still_pending
