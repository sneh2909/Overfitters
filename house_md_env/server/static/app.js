/* ===========================================================================
 * House M.D. — ER Simulation frontend
 *
 * State machine
 * -------------
 * 1. boot()           → fetch catalogs + populate selectors
 * 2. newCase()        → POST /episodes → first paint, patient walks in
 * 3. step(action)     → POST /actions or /agent_step → animate the result
 * 4. terminal         → freeze, show end overlay with rewards
 *
 * Animation timing is centralised in TIMINGS so a "speed" multiplier can
 * scale every delay simultaneously without micro-managing each await.
 * =========================================================================== */

const API = '/api';

const TIMINGS = {
  speed: 2,
  bubbleDwell: 2600,
  walkPatientIn: 1300,
  walkToCash: 700,
  walkToMachine: 700,
  machineWork: 1100,
  cashWork: 700,
  doctorAdvance: 700,
  resultDelay: 600,
  betweenSteps: 500,
  typeCharMs: 34,
};

const speedAdjust = (ms) => ms / TIMINGS.speed;
const sleep = (ms) => new Promise((r) => setTimeout(r, speedAdjust(ms)));

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
const state = {
  catalogs: null,
  sessionId: null,
  obs: null,
  truth: null,
  policy: 'oracle',
  mode: 'auto',
  isPlaying: false,
  isStepping: false,
  lastBoardSnapshot: null,
  loggedActionCount: 0,
};

const ttsState = {
  enabled: localStorage.getItem('hmAudioEnabled') !== 'false',
  queue: Promise.resolve(),
  currentAudio: null,
};

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`${res.status}: ${err}`);
  }
  return res.json();
}

async function speakText(speaker, text) {
  const preparedSpeech = prepareSpeech(speaker, text);
  await playPreparedSpeech(preparedSpeech);
}

function prepareSpeech(speaker, text) {
  if (!ttsState.enabled || !text || !String(text).trim()) {
    return Promise.resolve(null);
  }

  return fetchSpeechClip(speaker, text);
}

async function playPreparedSpeech(preparedSpeech) {
  const run = () => playSpeechClip(preparedSpeech);
  ttsState.queue = ttsState.queue.then(run, run);
  await ttsState.queue;
}

async function waitForSpeechIdle() {
  await ttsState.queue;
}

async function fetchSpeechClip(speaker, text) {
  try {
    const res = await fetch(`${API}/tts/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        speaker,
        text: String(text),
        patient_sex: state.obs?.sex || null,
      }),
    });

    if (res.status === 204) return null;
    if (!res.ok) throw new Error(await res.text());

    const audioBlob = await res.blob();
    return URL.createObjectURL(audioBlob);
  } catch (e) {
    console.warn('TTS skipped', e);
    return null;
  }
}

async function playSpeechClip(preparedSpeech) {
  const audioUrl = await preparedSpeech;
  if (!audioUrl || !ttsState.enabled) {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    return;
  }

  try {
    await playAudio(audioUrl);
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

function playAudio(audioUrl) {
  return new Promise((resolve) => {
    const audio = new Audio(audioUrl);
    ttsState.currentAudio = audio;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      if (ttsState.currentAudio === audio) ttsState.currentAudio = null;
      resolve();
    };

    audio.addEventListener('ended', finish, { once: true });
    audio.addEventListener('error', finish, { once: true });

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(finish);
    }
  });
}

function setAudioEnabled(enabled) {
  ttsState.enabled = enabled;
  localStorage.setItem('hmAudioEnabled', String(enabled));
  const checkbox = $('ctrl-audio');
  const label = $('audio-toggle-label');
  if (checkbox) checkbox.checked = enabled;
  if (label) label.textContent = enabled ? 'Audio On' : 'Audio Off';
  if (!enabled && ttsState.currentAudio) {
    ttsState.currentAudio.pause();
    ttsState.currentAudio.currentTime = 0;
    ttsState.currentAudio = null;
  }
}

// ---------------------------------------------------------------------------
// Catalog lookup helpers (by id)
// ---------------------------------------------------------------------------
const lookups = {
  question: (id) => state.catalogs.questions.find((q) => q.id === id),
  exam: (id) => state.catalogs.exams.find((e) => e.id === id),
  test: (id) => state.catalogs.tests.find((t) => t.id === id),
  disease: (id) => state.catalogs.diseases.find((d) => d.id === id),
};

const TEST_ZONE = {
  lab_basic: 'lab',
  lab_targeted: 'lab',
  imaging: 'imaging',
  bedside: 'bedside',
  specialty: 'bedside',
};

// ---------------------------------------------------------------------------
// Inline SVG cloud silhouette — used as the thought-bubble backdrop.
// preserveAspectRatio="none" stretches it to the wrapper div's dimensions.
// ---------------------------------------------------------------------------
const CLOUD_SVG = `
  <svg class="cloud-shape" viewBox="0 0 320 160" preserveAspectRatio="none"
       xmlns="http://www.w3.org/2000/svg">
    <path d="M 50,135
             C 14,135 4,95 28,80
             C 12,55 35,22 65,32
             C 70,8 120,2 140,28
             C 160,4 215,8 225,40
             C 265,30 305,60 285,95
             C 310,115 285,150 250,140
             C 235,158 70,158 50,135 Z"
          fill="#f5f0ff" stroke="#8a5fff" stroke-width="4"
          stroke-linejoin="round" stroke-linecap="round"/>
  </svg>
`;

// ---------------------------------------------------------------------------
// Patient sprite classifier — picks visual variant from age + sex
// ---------------------------------------------------------------------------
function patientClasses(obs) {
  const classes = [];
  classes.push(obs.sex === 'female' ? 'female' : 'male');
  if (obs.age < 18) classes.push('kid');
  else if (obs.age >= 60) classes.push('elderly');
  else classes.push('adult');
  return classes;
}

function applyPatientLook(obs) {
  const p = $('patient');
  ['female', 'male', 'kid', 'adult', 'elderly'].forEach((c) => p.classList.remove(c));
  patientClasses(obs).forEach((c) => p.classList.add(c));
}

// ---------------------------------------------------------------------------
// Typewriter — types `text` into `el` char-by-char so bubbles feel alive.
// Returns a Promise that resolves when typing finishes (or is skipped).
// ---------------------------------------------------------------------------
function typeText(el, text, opts = {}) {
  const charMs = opts.charMs || TIMINGS.typeCharMs;
  const safe = String(text);
  el.textContent = '';
  el.classList.add('typing');
  return new Promise((resolve) => {
    let i = 0;
    const tick = () => {
      if (!el.isConnected) { resolve(); return; }
      if (i >= safe.length) {
        el.classList.add('done');
        resolve();
        return;
      }
      el.textContent += safe[i++];
      setTimeout(tick, speedAdjust(charMs));
    };
    tick();
  });
}

// ---------------------------------------------------------------------------
// Bubbles — append to a character's anchor and auto-clear after dwell
// ---------------------------------------------------------------------------
function showBubble(anchorId, html, opts = {}) {
  const anchor = $(anchorId);
  // Drop any existing bubble immediately so the new one never visually
  // stacks on top of the old (the old fade-out animation caused 300ms
  // overlaps when actions or results came back-to-back).
  anchor.querySelectorAll('.bubble').forEach((b) => b.remove());

  const bubble = el('div', `bubble ${opts.cls || ''}`, html);
  anchor.appendChild(bubble);

  if (!opts.persist) {
    const removeTimer = setTimeout(() => {
      bubble.classList.add('fade-out');
      setTimeout(() => bubble.remove(), 300);
    }, speedAdjust(opts.dwell || TIMINGS.bubbleDwell));
    bubble.dataset.removeTimer = String(removeTimer);
  }
  return bubble;
}

function releaseBubbleAfter(bubble, dwell) {
  if (!bubble || !bubble.isConnected) return;
  const removeTimer = setTimeout(() => {
    bubble.classList.add('fade-out');
    setTimeout(() => bubble.remove(), 300);
  }, speedAdjust(dwell || TIMINGS.bubbleDwell));
  bubble.dataset.removeTimer = String(removeTimer);
}

async function waitForTypingAndSpeech(bubble, startTyping, preparedSpeech, dwell) {
  const run = () => playSpeechWithTyping(startTyping, preparedSpeech);
  ttsState.queue = ttsState.queue.then(run, run);
  await ttsState.queue;
  releaseBubbleAfter(bubble, dwell);
}

async function playSpeechWithTyping(startTyping, preparedSpeech) {
  const audioUrl = await preparedSpeech;
  if (!audioUrl || !ttsState.enabled) {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    await startTyping();
    return;
  }

  try {
    const speechDone = playAudio(audioUrl);
    const typingDone = startTyping();
    await Promise.all([speechDone, typingDone]);
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

function clearBubbles(anchorId) {
  if (anchorId) {
    $(anchorId).querySelectorAll('.bubble').forEach((b) => b.remove());
  } else {
    document.querySelectorAll('.bubble').forEach((b) => b.remove());
  }
}

// ---------------------------------------------------------------------------
// HUD updates
// ---------------------------------------------------------------------------
function paintHud(obs) {
  $('hud-step').textContent = Math.min(obs.step, obs.step_cap);
  $('hud-step-cap').textContent = obs.step_cap;
  $('hud-cost').textContent = obs.cost_so_far;
  $('hud-time').textContent = obs.time_elapsed_min;
  const sevEl = $('hud-severity');
  sevEl.textContent = obs.severity_signal;
  sevEl.className = `badge ${obs.severity_signal}`;
  $('hud-policy').textContent = state.policy;
  $('patient-tag').textContent = `Patient · ${obs.age}yo ${obs.sex}`;

  const patient = $('patient');
  patient.classList.toggle('deteriorating', obs.severity_signal === 'deteriorating');
}

function paintPending(obs) {
  const pendBox = $('pending-list');
  pendBox.innerHTML = '';
  if (!obs.pending_tests.length) {
    pendBox.appendChild(el('div', 'empty', 'none'));
    return;
  }
  obs.pending_tests.forEach((p) => {
    const pill = el('div', 'pending-pill');
    pill.innerHTML = `${p.test_id} <span class="countdown">${p.steps_left}</span>`;
    pendBox.appendChild(pill);
  });
}

function paintLog(obs) {
  const log = $('action-log');
  log.innerHTML = '';
  obs.action_log.forEach((entry) => {
    const cls = `log-entry ${entry.kind} ${entry.invalid ? 'invalid' : ''}`;
    const div = el('div', cls);
    if (entry.kind === 'action' && entry.action) {
      div.innerHTML = `
        <div class="log-step">STEP ${entry.step}</div>
        <div><span class="log-action">${entry.action.type}</span> ${entry.action.argument}</div>
        <div class="log-text">${escapeHtml(entry.text).replace(/\n/g, '<br>')}</div>
      `;
    } else {
      div.innerHTML = `
        <div class="log-step">RESULT · STEP ${entry.step}</div>
        <div class="log-text">${escapeHtml(entry.text)}</div>
      `;
    }
    log.appendChild(div);
  });
  log.scrollTop = log.scrollHeight;
}

function paintAll(obs) {
  paintHud(obs);
  paintPending(obs);
  paintLog(obs);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------
// Animation primitives
// ---------------------------------------------------------------------------
async function patientWalkIn() {
  const p = $('patient');
  p.classList.add('walking');
  await sleep(50);
  p.classList.add('entered');
  await sleep(TIMINGS.walkPatientIn);
  p.classList.remove('walking');
}

async function patientToTarget(targetSelector, dwellMs, activeClass = 'active') {
  const p = $('patient');
  const target = document.querySelector(targetSelector);
  if (!target) return;

  const targetRect = target.getBoundingClientRect();
  const stageRect = $('stage').getBoundingClientRect();
  const targetX = targetRect.left - stageRect.left + targetRect.width / 2;
  const targetPct = (targetX / stageRect.width) * 100;

  p.classList.add('walking');
  p.style.left = `${targetPct}%`;
  p.style.transform = 'translateX(-50%)';
  await sleep(TIMINGS.walkToMachine);
  p.classList.remove('walking');

  target.classList.add(activeClass);
  await sleep(dwellMs);
  target.classList.remove(activeClass);
}

async function patientToCenter() {
  const p = $('patient');
  p.classList.add('walking');
  p.style.left = '50%';
  await sleep(TIMINGS.walkToMachine);
  p.classList.remove('walking');
}

async function doctorAdvance() {
  const d = $('doctor');
  d.classList.add('walking', 'advancing');
  await sleep(TIMINGS.doctorAdvance);
  d.classList.remove('walking');
}

async function doctorRetreat() {
  const d = $('doctor');
  d.classList.add('walking');
  d.classList.remove('advancing');
  await sleep(TIMINGS.doctorAdvance);
  d.classList.remove('walking');
}

// ---------------------------------------------------------------------------
// Per-action animations — what makes each turn feel different
// ---------------------------------------------------------------------------

async function animateInitial(obs) {
  await patientWalkIn();
  const bubble = showBubble('patient-bubbles', `
    <span class="bub-label">Chief Complaint</span>
    <div class="type-target"></div>
    <div class="vitals-line" style="margin-top:8px;font-size:11px;color:#5a6b8a">
      <span style="color:#2a3a55;font-weight:600;">Vitals:</span> <span class="vitals-target"></span>
    </div>
  `, { persist: true });
  const speech = prepareSpeech('patient', `${obs.chief_complaint}. Vitals: ${obs.intake_vitals}`);
  await waitForTypingAndSpeech(bubble, async () => {
    await typeText(bubble.querySelector('.type-target'), obs.chief_complaint, { charMs: 30 });
    await typeText(bubble.querySelector('.vitals-target'), obs.intake_vitals, { charMs: 22 });
  }, speech, 1400);
  await sleep(1400);
}

async function animateInterview(action, resultEntry) {
  const q = lookups.question(action.argument);
  const questionText = q ? q.text : action.argument;

  const dBubble = showBubble('doctor-bubbles', `
    <span class="bub-label">Dr. House asks</span>
    <div class="bub-q type-target"></div>
  `, { persist: true });
  const questionSpeech = prepareSpeech('doctor', questionText);
  await waitForTypingAndSpeech(
    dBubble,
    () => typeText(dBubble.querySelector('.type-target'), questionText),
    questionSpeech,
    900
  );
  await sleep(500);

  let answer = resultEntry.text;
  const idx = answer.indexOf('A: ');
  if (idx !== -1) answer = answer.slice(idx + 3);

  const pBubble = showBubble('patient-bubbles', `
    <span class="bub-label">Patient</span>
    <div class="bub-a type-target"></div>
    ${resultEntry.duplicate ? '<div style="font-size:10px;color:#d68a32;margin-top:4px;">[DUP — already asked]</div>' : ''}
  `, { persist: true });
  const answerSpeech = prepareSpeech('patient', answer);
  await waitForTypingAndSpeech(
    pBubble,
    () => typeText(pBubble.querySelector('.type-target'), answer),
    answerSpeech,
    1200
  );
  await sleep(1200);
}

async function animateExamine(action, resultEntry) {
  const exam = lookups.exam(action.argument);
  const examText = exam ? exam.text : action.argument;

  await doctorAdvance();
  const dBubble = showBubble('doctor-bubbles', `
    <span class="bub-label">Examining</span>
    <div class="type-target"></div>
  `, { cls: 'action', persist: true });
  const examSpeech = examText.replace(/[.:]$/, '');
  const preparedExamSpeech = prepareSpeech('doctor', examSpeech);
  await waitForTypingAndSpeech(
    dBubble,
    () => typeText(dBubble.querySelector('.type-target'), `… ${examSpeech} …`),
    preparedExamSpeech,
    300
  );
  await sleep(900);

  clearBubbles('doctor-bubbles');

  let finding = resultEntry.text;
  const colonNL = finding.indexOf(':\n');
  if (colonNL !== -1) finding = finding.slice(colonNL + 2);

  const pBubble = showBubble('patient-bubbles', `
    <span class="bub-label">Finding · $${resultEntry.cost}</span>
    <div class="type-target"></div>
  `, { cls: 'finding', persist: true });
  const findingSpeech = prepareSpeech('patient', finding);
  await waitForTypingAndSpeech(
    pBubble,
    () => typeText(pBubble.querySelector('.type-target'), finding),
    findingSpeech,
    1500
  );

  await sleep(1500);
  await doctorRetreat();
}

async function animateOrderTest(action, resultEntry, postObs) {
  const test = lookups.test(action.argument);
  const testName = test ? test.name : action.argument;
  const cost = test ? test.cost : 0;
  const turnaround = test ? test.turnaround_steps : 0;
  const zone = test ? TEST_ZONE[test.category] || 'lab' : 'lab';

  const orderBubble = showBubble('doctor-bubbles', `
    <span class="bub-label">Order Placed</span>
    <div>📋 <span class="type-target"></span></div>
    <div style="font-size:11px;margin-top:4px;color:#2da45c">
      Cost: $${cost} · Turnaround: ${turnaround}-step
    </div>
  `, { cls: 'order', persist: true });
  const orderSpeech = prepareSpeech('doctor', `Order placed. ${testName}. Cost ${cost} dollars. Turnaround ${turnaround} step${turnaround !== 1 ? 's' : ''}.`);
  await waitForTypingAndSpeech(
    orderBubble,
    () => typeText(orderBubble.querySelector('.type-target'), testName, { charMs: 24 }),
    orderSpeech,
    800
  );
  await sleep(800);

  // 1) Patient walks to the cash counter to pay
  await patientToTarget('#cash-counter', TIMINGS.cashWork);
  const cashBubble = showBubble('patient-bubbles', `
    <span class="bub-label">Pays</span>
    <div class="type-target"></div>
  `, { cls: 'cash', persist: true });
  const cashSpeech = prepareSpeech('patient', `Paying ${cost} dollars.`);
  await waitForTypingAndSpeech(
    cashBubble,
    () => typeText(cashBubble.querySelector('.type-target'), `$${cost} 💵`, { charMs: 30 }),
    cashSpeech,
    700
  );
  await sleep(700);

  // 2) Then walks to the corresponding test machine
  await patientToTarget(`.machine[data-zone="${zone}"]`, TIMINGS.machineWork);

  // 3) Returns to center
  await patientToCenter();

  if (turnaround === 0 && postObs) {
    const resultLog = postObs.action_log
      .filter((e) => e.kind === 'result')
      .reverse()
      .find((e) => e.text.includes(test ? test.name : action.argument));
    if (resultLog) {
      await showResultReport(resultLog);
      await sleep(1800);
    }
  } else {
    const pendBubble = showBubble('doctor-bubbles', `
      <span class="bub-label">Pending</span>
      <div class="type-target"></div>
    `, { persist: true });
    const pendingText = `Awaiting ${testName} — ${turnaround} step${turnaround !== 1 ? 's' : ''}`;
    const pendingSpeech = prepareSpeech('doctor', pendingText);
    await waitForTypingAndSpeech(
      pendBubble,
      () => typeText(pendBubble.querySelector('.type-target'), pendingText),
      pendingSpeech,
      800
    );
    await sleep(800);
  }
}

async function showResultReport(resultEntry) {
  let text = resultEntry.text;
  if (text.startsWith('Result — ')) text = text.slice('Result — '.length);

  const flagMatch = text.match(/\[flag:\s*([A-Z]+)\]\s*$/);
  const flag = flagMatch ? flagMatch[1] : 'N';
  let body = flagMatch ? text.slice(0, flagMatch.index).trim() : text;

  const colonIdx = body.indexOf(':');
  const testName = colonIdx !== -1 ? body.slice(0, colonIdx) : 'Result';
  const value = colonIdx !== -1 ? body.slice(colonIdx + 1).trim() : body;

  const bubble = showBubble('doctor-bubbles', `
    <div class="wb-header">
      <span>📋 LAB RESULT</span>
      <span class="flag ${flag}">${flag}</span>
    </div>
    <div class="wb-body">
      <div class="wb-test type-test"></div>
      <div class="wb-value type-value"></div>
    </div>
    <div class="wb-tray"></div>
  `, { cls: 'report', persist: true });

  const reportSpeech = prepareSpeech('doctor', `Lab result. ${testName}. ${value}. Flag ${flag}.`);
  await waitForTypingAndSpeech(bubble, async () => {
    await typeText(bubble.querySelector('.type-test'), testName, { charMs: 14 });
    await typeText(bubble.querySelector('.type-value'), value, { charMs: 16 });
  }, reportSpeech, 1400);
}

async function animateUpdateDifferential(action) {
  const board = (action.board || []).slice().sort((a, b) => b.prob - a.prob).slice(0, 5);
  const rowsHtml = board.map((b, i) => {
    const name = lookups.disease(b.disease)?.name || b.disease;
    const pct = Math.round(b.prob * 100);
    return `
      <div class="diff-row" data-idx="${i}" style="opacity:0;transform:translateY(4px);transition:all .25s ease;">
        <span class="diff-name">${escapeHtml(name)}</span>
        <span class="diff-bar"><span class="diff-fill" style="width:0%" data-target="${pct}"></span></span>
        <span class="diff-prob">0%</span>
      </div>
    `;
  }).join('');

  const bubble = showBubble('doctor-bubbles', `
    ${CLOUD_SVG}
    <div class="thought-content">
      <span class="bub-label">💭 Differential</span>
      ${rowsHtml}
    </div>
    <span class="puff p1"></span>
    <span class="puff p2"></span>
    <span class="puff p3"></span>
  `, { cls: 'thought', dwell: 3800 });

  // Reveal each row in sequence and animate bar fill + percent count-up
  const rows = bubble.querySelectorAll('.diff-row');
  for (const row of rows) {
    row.style.opacity = '1';
    row.style.transform = 'translateY(0)';
    const fill = row.querySelector('.diff-fill');
    const probEl = row.querySelector('.diff-prob');
    const target = parseInt(fill.dataset.target, 10);
    fill.style.width = `${target}%`;
    let v = 0;
    const dur = speedAdjust(420);
    const start = performance.now();
    await new Promise((resolve) => {
      const tick = (now) => {
        const t = Math.min(1, (now - start) / dur);
        v = Math.round(target * t);
        probEl.textContent = `${v}%`;
        if (t < 1) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
    await sleep(120);
  }
  await sleep(1600);
}

async function animateDiagnose(action) {
  const dx = lookups.disease(action.argument);
  const dxName = dx ? dx.name : action.argument;
  await doctorAdvance();

  const bubble = showBubble('doctor-bubbles', `
    <span class="bub-label">Final Diagnosis</span>
    <div class="type-target"></div>
  `, { cls: 'diagnosis', persist: true });
  const diagnosisSpeech = prepareSpeech('doctor', `Final diagnosis: ${dxName}.`);
  await waitForTypingAndSpeech(
    bubble,
    () => typeText(bubble.querySelector('.type-target'), dxName, { charMs: 30 }),
    diagnosisSpeech,
    1500
  );
  await sleep(1500);
}

// ---------------------------------------------------------------------------
// One step lifecycle: send request → animate → repaint state
// ---------------------------------------------------------------------------

async function performStep(actionPayload) {
  if (state.isStepping || !state.sessionId) return;
  if (state.obs && state.obs.terminal) return;

  state.isStepping = true;
  try {
    const before = state.obs;
    const beforeLogLen = before ? before.action_log.length : 0;

    let after;
    if (actionPayload) {
      after = await api(`/episodes/${state.sessionId}/actions`, {
        method: 'POST',
        body: JSON.stringify(actionPayload),
      });
    } else {
      after = await api(`/episodes/${state.sessionId}/agent_step`, {
        method: 'POST',
      });
    }

    // Find the action entry that was just appended (first new action entry).
    const newEntries = after.action_log.slice(beforeLogLen);
    const actionEntry = newEntries.find((e) => e.kind === 'action');
    const resultEntries = newEntries.filter((e) => e.kind === 'result');

    if (actionEntry && actionEntry.action) {
      const a = actionEntry.action;
      switch (a.type) {
        case 'INTERVIEW':
          await animateInterview(a, actionEntry);
          break;
        case 'EXAMINE':
          await animateExamine(a, actionEntry);
          break;
        case 'ORDER_TEST':
          await animateOrderTest(a, actionEntry, after);
          break;
        case 'UPDATE_DIFFERENTIAL':
          await animateUpdateDifferential(a);
          break;
        case 'DIAGNOSE':
          await animateDiagnose(a);
          break;
        default:
          await sleep(500);
      }
    }

    // Show any DELAYED test results that arrived this step (for non-zero
    // turnaround tests ordered earlier). Each report fully replaces the
    // previous one so multiple results never visually stack.
    if (actionEntry && actionEntry.action?.type !== 'ORDER_TEST') {
      for (const r of resultEntries) {
        clearBubbles('doctor-bubbles');
        await showResultReport(r);
        await sleep(TIMINGS.resultDelay + 1400);
      }
    }

    state.obs = after;
    paintAll(after);

    if (after.terminal) {
      await handleTerminal();
    }
  } catch (e) {
    console.error('step failed', e);
  } finally {
    state.isStepping = false;
  }
}

// ---------------------------------------------------------------------------
// Terminal handling
// ---------------------------------------------------------------------------

async function handleTerminal() {
  state.isPlaying = false;
  $('btn-play').textContent = '▶ Play';
  $('btn-play').classList.remove('playing');

  await sleep(1200);

  let rewards = null;
  let truth = null;
  try {
    [rewards, truth] = await Promise.all([
      api(`/episodes/${state.sessionId}/rewards`),
      api(`/episodes/${state.sessionId}/truth`),
    ]);
  } catch (e) {
    console.error('terminal fetch failed', e);
  }

  const obs = state.obs;
  state.truth = truth;
  refreshTruthPanel();

  $('end-cost').textContent = obs.cost_so_far;
  $('end-steps').textContent = obs.step;
  $('end-dx').textContent = obs.diagnosis
    ? lookups.disease(obs.diagnosis)?.name || obs.diagnosis
    : '—';
  $('end-truth').textContent = truth ? truth.true_disease_name : '—';

  const correct = obs.diagnosis && truth && obs.diagnosis === truth.true_disease_id;
  const verdictEl = $('end-verdict');
  if (obs.timed_out) {
    verdictEl.textContent = '⏰ TIMED OUT — patient left without diagnosis';
    verdictEl.className = 'end-verdict timeout';
  } else if (correct) {
    verdictEl.textContent = '✅ CORRECT DIAGNOSIS';
    verdictEl.className = 'end-verdict';
  } else {
    verdictEl.textContent = '❌ MISDIAGNOSIS';
    verdictEl.className = 'end-verdict wrong';
  }

  if (rewards) {
    paintRewards(rewards);
  }

  $('end-overlay').classList.remove('hidden');
}

function paintRewards(rewards) {
  const panel = $('rewards-panel');
  panel.innerHTML = '';
  const order = ['r1_accuracy', 'r2_cost', 'r6_anchoring', 'r7_safety', 'r8_format'];
  const labels = {
    r1_accuracy: 'R1 Accuracy',
    r2_cost: 'R2 Cost',
    r6_anchoring: 'R6 Anchoring',
    r7_safety: 'R7 Safety',
    r8_format: 'R8 Format',
  };
  order.forEach((k) => {
    const v = rewards[k] ?? 0;
    const pct = Math.min(100, Math.abs(v) * 100);
    const row = el('div', 'reward-row');
    row.innerHTML = `
      <span class="reward-name">${labels[k]}</span>
      <span class="reward-bar"><span class="reward-fill ${v >= 0 ? 'pos' : 'neg'}" style="width:${pct}%"></span></span>
      <span class="reward-val">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
    `;
    panel.appendChild(row);
  });

  const total = rewards.total ?? 0;
  const totalRow = el('div', 'reward-row total');
  totalRow.innerHTML = `
    <span class="reward-name">Total</span>
    <span class="reward-bar"></span>
    <span class="reward-val">${total >= 0 ? '+' : ''}${total.toFixed(2)}</span>
  `;
  panel.appendChild(totalRow);
}

// ---------------------------------------------------------------------------
// Auto-play loop
// ---------------------------------------------------------------------------
async function playLoop() {
  while (state.isPlaying && state.obs && !state.obs.terminal) {
    await performStep(null);
    if (!state.obs.terminal) await sleep(TIMINGS.betweenSteps);
  }
  state.isPlaying = false;
  $('btn-play').textContent = '▶ Play';
  $('btn-play').classList.remove('playing');
}

// ---------------------------------------------------------------------------
// Truth panel
// ---------------------------------------------------------------------------
async function refreshTruthPanel() {
  if (!state.truth) {
    try {
      state.truth = await api(`/episodes/${state.sessionId}/truth`);
    } catch (e) { return; }
  }
  const t = state.truth;
  const panel = $('truth-panel');
  if (panel.classList.contains('hidden')) return;
  panel.innerHTML = `
    <div class="truth-row"><span class="k">Disease</span><span class="v">${escapeHtml(t.true_disease_name)}</span></div>
    <div class="truth-row"><span class="k">Severity</span><span class="v">${escapeHtml(t.severity)}</span></div>
    <div class="truth-row"><span class="k">Family</span><span class="v">${escapeHtml(t.family)}</span></div>
    <div class="truth-row"><span class="k">Variant</span><span class="v">${escapeHtml(t.variant_id)}</span></div>
    <div class="truth-row"><span class="k">Det. rate</span><span class="v">${t.deterioration_rate.toFixed(2)}</span></div>
    <div class="truth-row"><span class="k">Seed</span><span class="v">${t.seed}</span></div>
  `;
}

// ---------------------------------------------------------------------------
// New case lifecycle
// ---------------------------------------------------------------------------
async function newCase() {
  if (state.isStepping) return;
  state.isStepping = true;
  await waitForSpeechIdle();

  // Reset visual state
  state.isPlaying = false;
  state.obs = null;
  state.truth = null;
  state.loggedActionCount = 0;
  $('end-overlay').classList.add('hidden');
  $('btn-play').textContent = '▶ Play';
  $('btn-play').classList.remove('playing');
  clearBubbles();
  $('patient').classList.remove('entered', 'walking', 'deteriorating',
    'female', 'male', 'kid', 'adult', 'elderly');
  $('patient').style.left = '';
  $('patient').style.transform = '';
  $('doctor').classList.remove('walking', 'advancing');
  $('action-log').innerHTML = '';
  $('pending-list').innerHTML = '';
  $('truth-panel').classList.add('hidden');
  $('truth-panel').innerHTML = '<div>Hidden — toggle to reveal</div>';
  $('truth-toggle').textContent = 'reveal';

  const disease = $('ctrl-disease').value || null;
  const policy = $('ctrl-policy').value;
  state.policy = policy;

  try {
    const data = await api('/episodes', {
      method: 'POST',
      body: JSON.stringify({ disease, policy }),
    });
    state.sessionId = data.session_id;
    state.obs = data;
    applyPatientLook(data);
    paintAll(data);

    await animateInitial(data);
  } catch (e) {
    console.error('new case failed', e);
    alert('Could not start a new case: ' + e.message);
  } finally {
    state.isStepping = false;
  }
}

// ---------------------------------------------------------------------------
// Manual action panel population
// ---------------------------------------------------------------------------
function populateManualArgs() {
  const type = $('manual-type').value;
  const arg = $('manual-arg');
  arg.innerHTML = '';

  if (!state.catalogs) return;
  let items = [];
  if (type === 'INTERVIEW') items = state.catalogs.questions.map((q) => ({ id: q.id, label: q.text }));
  else if (type === 'EXAMINE') items = state.catalogs.exams.map((e) => ({ id: e.id, label: `${e.text} ($${e.cost})` }));
  else if (type === 'ORDER_TEST') items = state.catalogs.tests.map((t) => ({ id: t.id, label: `${t.name} ($${t.cost}, ${t.turnaround_steps}-step)` }));
  else if (type === 'DIAGNOSE') items = state.catalogs.diseases.map((d) => ({ id: d.id, label: `${d.name} (${d.severity})` }));
  else if (type === 'UPDATE_DIFFERENTIAL') {
    arg.innerHTML = '<option>(uses current top-3 split)</option>';
    return;
  }

  items.forEach((it) => {
    const opt = document.createElement('option');
    opt.value = it.id;
    opt.textContent = `${it.id} — ${it.label}`;
    arg.appendChild(opt);
  });
}

async function executeManual() {
  const type = $('manual-type').value;
  const argument = $('manual-arg').value;
  let board = null;
  if (type === 'UPDATE_DIFFERENTIAL') {
    const all = state.catalogs.diseases.map((d) => d.id);
    const sample = [all[0], all[1], all[2]];
    board = sample.map((d, i) => ({ disease: d, prob: [0.5, 0.3, 0.2][i] }));
  }
  await performStep({ type, argument, rationale: 'manual', board });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
async function boot() {
  state.catalogs = await api('/catalogs');

  // Populate disease selector
  const sel = $('ctrl-disease');
  state.catalogs.diseases.forEach((d) => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `${d.name} (${d.severity})`;
    sel.appendChild(opt);
  });

  populateManualArgs();
  $('manual-type').addEventListener('change', populateManualArgs);
  $('manual-execute').addEventListener('click', executeManual);
  setAudioEnabled(ttsState.enabled);
  $('ctrl-audio').addEventListener('change', (e) => {
    setAudioEnabled(e.target.checked);
  });

  $('btn-new').addEventListener('click', newCase);
  $('btn-end-next').addEventListener('click', newCase);
  $('btn-end-close').addEventListener('click', () => $('end-overlay').classList.add('hidden'));

  $('btn-step').addEventListener('click', () => performStep(null));

  $('btn-play').addEventListener('click', () => {
    if (state.isPlaying) {
      state.isPlaying = false;
      $('btn-play').textContent = '▶ Play';
      $('btn-play').classList.remove('playing');
    } else {
      if (!state.obs || state.obs.terminal) return;
      state.isPlaying = true;
      $('btn-play').textContent = '⏸ Pause';
      $('btn-play').classList.add('playing');
      playLoop();
    }
  });

  $('ctrl-speed').addEventListener('change', (e) => {
    TIMINGS.speed = parseFloat(e.target.value) || 1;
  });

  $('ctrl-mode').addEventListener('change', (e) => {
    state.mode = e.target.value;
    document.querySelector('.controls').classList.toggle('manual-mode', state.mode === 'manual');
  });

  $('ctrl-policy').addEventListener('change', (e) => {
    state.policy = e.target.value;
    $('hud-policy').textContent = state.policy;
  });

  $('truth-toggle').addEventListener('click', async () => {
    const panel = $('truth-panel');
    const hidden = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    $('truth-toggle').textContent = hidden ? 'hide' : 'reveal';
    if (hidden) {
      if (!state.truth && state.sessionId) {
        try { state.truth = await api(`/episodes/${state.sessionId}/truth`); } catch (e) {}
      }
      refreshTruthPanel();
    }
  });

  // Auto-start a first case
  await newCase();
}

document.addEventListener('DOMContentLoaded', boot);
