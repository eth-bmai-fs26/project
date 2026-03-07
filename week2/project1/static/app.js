'use strict';

const APPROVE_COLOR  = '#2ecc71';
const REJECT_COLOR   = '#e74c3c';
const TIMER_SECONDS  = 8;
const HISTORY_MAX    = 10;

let tiles      = [];
let mode       = 'challenge';
let timedMode  = false;
let timerId    = null;
let timerLeft  = TIMER_SECONDS;

// ── State ────────────────────────────────────────────────────────
const challenge = {
  seenIndices: new Set(),
  current:     null,
  revealed:    false,
  userChoice:  null,
  score:       { correct: 0, total: 0 },
  streak:      0,
  bestStreak:  0,
  history:     [],   // booleans, last HISTORY_MAX results
  answers:     [],   // {tile, userChoice, correct, timedOut} for summary
  gameOver:    false,
};

const explorer = {};

// ── Bootstrap ────────────────────────────────────────────────────
async function init() {
  const res = await fetch('/api/tiles');
  tiles = await res.json();

  challenge.current = pickRandom(tiles, challenge.seenIndices);
  challenge.seenIndices.add(challenge.current.index);

  document.getElementById('btn-challenge').addEventListener('click', () => switchMode('challenge'));
  document.getElementById('btn-explorer').addEventListener('click',  () => switchMode('explorer'));
  document.getElementById('btn-reset').addEventListener('click', resetChallenge);

  document.getElementById('timer-toggle-cb').addEventListener('change', e => {
    timedMode = e.target.checked;
    if (!timedMode) stopTimer();
    else if (mode === 'challenge' && !challenge.revealed && !challenge.gameOver) startTimer();
  });

  document.getElementById('lightbox-backdrop').addEventListener('click', closeLightbox);
  document.addEventListener('keydown', onKeydown);

  render();
}

// ── Keyboard shortcuts ───────────────────────────────────────────
function onKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  if (e.key === 'Escape') { closeLightbox(); return; }

  if (mode !== 'challenge' || challenge.gameOver) return;

  if (!challenge.revealed) {
    if (e.key === 'a' || e.key === 'A') handleChoice('APPROVE');
    if (e.key === 'r' || e.key === 'R') handleChoice('REJECT');
  } else {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); handleNext(); }
  }
}

// ── Lightbox ─────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
  document.getElementById('lightbox').style.display = 'none';
}

function attachLightbox(container) {
  container.querySelectorAll('.clickable-img').forEach(img => {
    img.addEventListener('click', () => openLightbox(img.src));
  });
}

// ── Timer ────────────────────────────────────────────────────────
function startTimer() {
  stopTimer();
  timerLeft = TIMER_SECONDS;
  tickTimer();
  timerId = setInterval(tickTimer, 1000);
}

function stopTimer() {
  if (timerId) { clearInterval(timerId); timerId = null; }
}

function tickTimer() {
  const el = document.getElementById('timer-display');
  if (el) {
    el.textContent = timerLeft;
    el.classList.toggle('urgent', timerLeft <= 3);
  }
  if (timerLeft <= 0) {
    stopTimer();
    // Auto-submit the wrong answer on timeout
    const wrong = challenge.current.status === 'APPROVE' ? 'REJECT' : 'APPROVE';
    handleChoice(wrong, true);
    return;
  }
  timerLeft--;
}

// ── Helpers ──────────────────────────────────────────────────────
function pickRandom(pool, seenIndices) {
  const available = pool.filter(t => !seenIndices.has(t.index));
  const src = available.length > 0 ? available : pool;
  return src[Math.floor(Math.random() * src.length)];
}

function coverageBarHTML(pct) {
  const color = pct >= 0.40 ? APPROVE_COLOR : REJECT_COLOR;
  return `
    <div class="cov-bar-wrapper">
      <div class="cov-bar-label">
        Coverage: <strong style="color:${color}">${(pct * 100).toFixed(1)}%</strong>
        &middot; threshold: 40%
      </div>
      <div class="cov-bar-track">
        <div class="cov-bar-fill" data-target="${Math.round(pct * 100)}" style="width:0;background:${color}"></div>
        <div class="cov-bar-marker"></div>
      </div>
    </div>`;
}

function animateBars() {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll('.cov-bar-fill[data-target]').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  }));
}

// ── Mode switching ────────────────────────────────────────────────
function switchMode(newMode) {
  stopTimer();
  mode = newMode;
  render();
}

function render() {
  document.getElementById('btn-challenge').classList.toggle('active', mode === 'challenge');
  document.getElementById('btn-explorer').classList.toggle('active',  mode === 'explorer');
  document.getElementById('challenge-hint').style.display = mode === 'challenge' ? '' : 'none';
  document.getElementById('explorer-hint').style.display  = mode === 'explorer'  ? '' : 'none';
  document.getElementById('btn-reset').style.display      = mode === 'challenge' ? '' : 'none';
  document.getElementById('timer-section').style.display  = mode === 'challenge' ? '' : 'none';

  if (mode === 'challenge') {
    document.getElementById('main-title').textContent   = 'Challenge Mode';
    document.getElementById('main-caption').textContent =
      'The algorithm uses U-Net segmentation to measure intact tile coverage. ' +
      'It approves batches with > 40% intact area. Can you match it?';
    renderChallenge();
  } else {
    document.getElementById('main-title').textContent   = 'Explorer Mode';
    document.getElementById('main-caption').textContent =
      'Browse the dataset — drag the slider on any tile to compare original vs mask.';
    renderExplorer();
  }
}

// ── Challenge mode ────────────────────────────────────────────────
function renderChallenge() {
  if (challenge.gameOver) { renderSummary(); return; }

  const t   = challenge.current;
  const sc  = challenge.score;
  const pct = sc.total > 0 ? `${Math.round(sc.correct / sc.total * 100)}%` : '—';

  const histDots = challenge.history.slice(-HISTORY_MAX)
    .map(c => `<span class="hist-dot" style="background:${c ? APPROVE_COLOR : REJECT_COLOR}"></span>`)
    .join('');

  const timerBadge = (timedMode && !challenge.revealed)
    ? `<div class="timer-display" id="timer-display">${timerLeft}</div>`
    : '';

  document.getElementById('main-content').innerHTML = `
    <div class="score-bar fade-in">
      <div class="score-left">
        <span class="score-tile-info">Tile ${challenge.seenIndices.size} of ${tiles.length}</span>
        <div class="hist-dots">${histDots}</div>
      </div>
      <div class="score-right">
        <div class="score-badge">
          <div class="score-badge-label">Score</div>
          <div class="score-badge-value">${pct}</div>
          <div class="score-badge-sub">${sc.correct} / ${sc.total}</div>
        </div>
        <div class="score-badge">
          <div class="score-badge-label">Streak</div>
          <div class="score-badge-value">${challenge.streak}</div>
          <div class="score-badge-sub">best ${challenge.bestStreak}</div>
        </div>
      </div>
    </div>

    <div class="challenge-grid fade-in">
      <div class="card">
        <div class="img-wrapper">
          <img class="clickable-img" src="/images/original/${t.filename}" alt="Original tile batch">
          ${timerBadge}
        </div>
        <div class="card-body">
          ${!challenge.revealed
            ? `<div class="kbd-hint">Press <kbd>A</kbd> Approve &nbsp; <kbd>R</kbd> Reject</div>`
            : `<div class="kbd-hint">Press <kbd>Space</kbd> for next tile</div>`
          }
          ${verdictAreaHTML(t)}
        </div>
      </div>
      <div id="right-panel">${rightPanelHTML(t)}</div>
    </div>`;

  const content = document.getElementById('main-content');
  attachLightbox(content);

  if (!challenge.revealed) {
    document.getElementById('btn-approve').addEventListener('click', () => handleChoice('APPROVE'));
    document.getElementById('btn-reject').addEventListener('click',  () => handleChoice('REJECT'));
    if (timedMode) startTimer();
  } else {
    document.getElementById('btn-next').addEventListener('click', handleNext);
    animateBars();
  }
}

function verdictAreaHTML(t) {
  if (!challenge.revealed) {
    return `
      <div class="verdict-label">What's your verdict?</div>
      <div class="verdict-btns">
        <button class="btn-approve" id="btn-approve">APPROVE</button>
        <button class="btn-reject"  id="btn-reject">REJECT</button>
      </div>`;
  }
  const correct = challenge.userChoice === t.status;
  const color   = challenge.userChoice === 'APPROVE' ? APPROVE_COLOR : REJECT_COLOR;
  return `
    <div class="result-text fade-in">
      You chose: <strong style="color:${color}">${challenge.userChoice}</strong>
      &mdash; ${correct ? 'Correct!' : 'Wrong!'}
    </div>
    <button class="btn-next" id="btn-next">Next Tile &rarr;</button>`;
}

function rightPanelHTML(t) {
  if (!challenge.revealed) {
    return `<div class="placeholder-panel">
      Make your decision &mdash; the mask will be revealed after you choose.
    </div>`;
  }
  const algoColor = t.status === 'APPROVE' ? APPROVE_COLOR : REJECT_COLOR;
  return `
    <div class="card fade-in">
      <img class="clickable-img" src="/images/segmented/${t.filename}" alt="U-Net segmentation mask">
      <div class="card-body">
        ${coverageBarHTML(t.coverage_pct)}
        <div class="algo-verdict">
          Algorithm verdict: <span style="color:${algoColor}">${t.status}</span>
        </div>
      </div>
    </div>`;
}

function handleChoice(choice, timedOut = false) {
  stopTimer();
  const correct = choice === challenge.current.status;
  challenge.userChoice  = choice;
  challenge.revealed    = true;
  challenge.score.correct += correct ? 1 : 0;
  challenge.score.total   += 1;
  challenge.streak      = correct ? challenge.streak + 1 : 0;
  challenge.bestStreak  = Math.max(challenge.bestStreak, challenge.streak);
  challenge.history.push(correct);
  challenge.answers.push({ tile: challenge.current, userChoice: choice, correct, timedOut });
  renderChallenge();
}

function handleNext() {
  if (challenge.seenIndices.size >= tiles.length) {
    challenge.gameOver = true;
    renderChallenge();
    return;
  }
  const next = pickRandom(tiles, challenge.seenIndices);
  challenge.seenIndices.add(next.index);
  challenge.current    = next;
  challenge.revealed   = false;
  challenge.userChoice = null;
  renderChallenge();
}

function resetChallenge() {
  stopTimer();
  challenge.seenIndices = new Set();
  challenge.current     = pickRandom(tiles, challenge.seenIndices);
  challenge.seenIndices.add(challenge.current.index);
  challenge.revealed    = false;
  challenge.userChoice  = null;
  challenge.score       = { correct: 0, total: 0 };
  challenge.streak      = 0;
  challenge.bestStreak  = 0;
  challenge.history     = [];
  challenge.answers     = [];
  challenge.gameOver    = false;
  renderChallenge();
}

// ── End-of-deck summary ───────────────────────────────────────────
function renderSummary() {
  const sc  = challenge.score;
  const pct = sc.total > 0 ? Math.round(sc.correct / sc.total * 100) : 0;
  const wrong = challenge.answers.filter(a => !a.correct);

  const wrongGrid = wrong.length === 0
    ? `<p style="color:var(--text-muted);font-size:14px;">No mistakes — perfect round!</p>`
    : wrong.slice(0, 6).map(a => {
        const ac = a.tile.status === 'APPROVE' ? APPROVE_COLOR : REJECT_COLOR;
        const uc = a.userChoice  === 'APPROVE' ? APPROVE_COLOR : REJECT_COLOR;
        return `
          <div class="summary-tile">
            <img class="clickable-img" src="/images/original/${a.tile.filename}" alt="${a.tile.filename}">
            <div class="summary-tile-info">
              <span>You: <strong style="color:${uc}">${a.userChoice}</strong></span>
              <span>Algo: <strong style="color:${ac}">${a.tile.status}</strong></span>
              ${a.timedOut ? '<span style="color:var(--text-muted);font-size:11px">Timed out</span>' : ''}
            </div>
          </div>`;
      }).join('');

  document.getElementById('main-content').innerHTML = `
    <div class="summary fade-in">
      <div class="summary-header">
        <div class="summary-score">${pct}%</div>
        <div class="summary-sub">
          ${sc.correct} of ${sc.total} correct &nbsp;&middot;&nbsp; Best streak: ${challenge.bestStreak}
        </div>
      </div>
      ${wrong.length > 0 ? `
        <div class="summary-section-title">Tiles you got wrong (${wrong.length})</div>
        <div class="summary-wrong-grid">${wrongGrid}</div>` : wrongGrid}
      <button class="btn-next" id="btn-play-again" style="max-width:220px;margin-top:24px">
        Play Again
      </button>
    </div>`;

  document.getElementById('btn-play-again').addEventListener('click', resetChallenge);
  attachLightbox(document.getElementById('main-content'));
}

// ── Explorer mode ─────────────────────────────────────────────────
function renderExplorer() {
  document.getElementById('main-content').innerHTML = `
    <div id="dist-chart"></div>
    <div id="explorer-results"></div>`;
  updateExplorer();
}

function updateExplorer() {
  const shown = tiles.slice(0, 50);

  renderDistChart(tiles);

  document.getElementById('explorer-results').innerHTML = `
    <div class="explorer-count">Showing ${shown.length} of ${tiles.length} tiles</div>
    <div class="explorer-grid">
      ${shown.map(tileCardHTML).join('')}
    </div>`;

  shown.forEach(t => initCompareSlider(t.index));
  animateBars();
}

// ── Distribution chart ────────────────────────────────────────────
function renderDistChart(filtered) {
  const buckets = Array(10).fill(0);
  filtered.forEach(t => {
    buckets[Math.min(9, Math.floor(t.coverage_pct * 10))]++;
  });
  const maxCount = Math.max(...buckets, 1);

  const bars = buckets.map((count, i) => {
    const color = (i + 1) / 10 >= 0.40 ? APPROVE_COLOR : REJECT_COLOR;
    const h     = Math.round(count / maxCount * 100);
    return `
      <div class="dist-bar-col">
        <div class="dist-bar-count">${count || ''}</div>
        <div class="dist-bar" style="height:${h}%;background:${color}"></div>
        <div class="dist-bar-label">${i * 10}</div>
      </div>`;
  }).join('');

  document.getElementById('dist-chart').innerHTML = `
    <div class="dist-chart fade-in">
      <div class="dist-title">Coverage distribution (% of intact area)</div>
      <div class="dist-bars">${bars}</div>
    </div>`;
}

// ── Explorer tile card ────────────────────────────────────────────
function tileCardHTML(t) {
  const borderColor = t.status === 'APPROVE' ? APPROVE_COLOR : REJECT_COLOR;
  return `
    <div class="tile-card" style="border-color:${borderColor}">
      <div class="img-compare" id="compare-${t.index}">
        <img class="compare-base" src="/images/original/${t.filename}" alt="${t.filename}" draggable="false">
        <img class="compare-mask" id="mask-${t.index}" src="/images/segmented/${t.filename}" alt="mask" draggable="false">
        <div class="compare-handle" id="handle-${t.index}">
          <div class="compare-knob">&#8596;</div>
        </div>
        <div class="compare-labels">
          <span class="compare-label-orig">Original</span>
          <span class="compare-label-mask">Mask</span>
        </div>
      </div>
      <div class="tile-card-footer">
        ${coverageBarHTML(t.coverage_pct)}
        <div class="tile-status" style="color:${borderColor}">${t.status}</div>
      </div>
    </div>`;
}

// ── Compare slider ────────────────────────────────────────────────
function initCompareSlider(index) {
  const container = document.getElementById(`compare-${index}`);
  const maskImg   = document.getElementById(`mask-${index}`);
  const handle    = document.getElementById(`handle-${index}`);
  if (!container || !maskImg || !handle) return;

  let dragging = false;

  function setPos(clientX) {
    const rect = container.getBoundingClientRect();
    let pct = (clientX - rect.left) / rect.width;
    pct = Math.max(0.02, Math.min(0.98, pct));
    maskImg.style.clipPath  = `inset(0 ${((1 - pct) * 100).toFixed(1)}% 0 0)`;
    handle.style.left       = `${(pct * 100).toFixed(1)}%`;
  }

  // Mouse
  handle.addEventListener('mousedown', e => {
    dragging = true;
    e.preventDefault();
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup',   onMouseUp);
  });

  function onMouseMove(e) { if (dragging) setPos(e.clientX); }
  function onMouseUp()    { dragging = false; document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); }

  // Touch
  handle.addEventListener('touchstart', e => {
    dragging = true;
    document.addEventListener('touchmove', onTouchMove, { passive: false });
    document.addEventListener('touchend',  onTouchEnd);
  });

  function onTouchMove(e) { e.preventDefault(); if (dragging) setPos(e.touches[0].clientX); }
  function onTouchEnd()   { dragging = false; document.removeEventListener('touchmove', onTouchMove); document.removeEventListener('touchend', onTouchEnd); }
}

init();
