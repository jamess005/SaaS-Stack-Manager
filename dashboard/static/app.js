// ── Utilities ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function confColor(p) {
  if (p === null || p === undefined) return '#94a3b8';
  return p >= 0.80 ? '#22c55e' : p >= 0.60 ? '#f59e0b' : '#ef4444';
}

function verdictBadge(v) {
  const map = {
    SWITCH: ['badge-sw', '⇄ SWITCH'],
    HOLD:   ['badge-ho', '⏸ HOLD'],
    STAY:   ['badge-st', '✓ STAY'],
  };
  const [cls, label] = map[v] || ['badge-st', v];
  return `<span class="verdict-badge ${cls}">${label}</span>`;
}

// ── Navigation ────────────────────────────────────────────────────────────────

function showView(name, btn) {
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  $(`view-${name}`).classList.add('active');
  btn.classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'competitors') loadCompetitors();
  if (name === 'health') loadHealth();
}

// ── Status polling ────────────────────────────────────────────────────────────

let _statusTimer = null;

function startStatusPoll() {
  if (_statusTimer) return;
  _statusTimer = setInterval(pollStatus, 3000);
}

function stopStatusPoll() {
  clearInterval(_statusTimer);
  _statusTimer = null;
}

async function pollStatus() {
  const r = await fetch('/api/status');
  const d = await r.json();
  const bar = $('status-bar');
  if (d.busy) {
    bar.classList.add('visible');
    $('status-text').textContent = `Model running: ${d.task}…`;
    $('btn-eval').disabled = true;
    $('btn-summarise').disabled = true;
  } else {
    if (bar.classList.contains('visible')) {
      bar.classList.remove('visible');
      $('btn-eval').disabled = false;
      $('btn-summarise').disabled = false;
      loadDashboard();
    }
    stopStatusPoll();
  }
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

async function loadDashboard() {
  const [statsR, verdictsR, healthR] = await Promise.all([
    fetch('/api/stats'),
    fetch('/api/verdicts?n=10'),
    fetch('/api/health'),
  ]);
  const stats = await statsR.json();
  const verdicts = await verdictsR.json();
  const health = await healthR.json();

  renderStats(stats);
  renderDonut(stats);
  renderVerdictGrid(verdicts);
  renderConfBars(verdicts);
  renderMonitor(verdicts);
  renderFeedbackHistory(health);
  renderRegressionTests(health);
}

function renderStats(s) {
  $('stat-sw').textContent = s.switch;
  $('stat-ho').textContent = s.hold;
  $('stat-st').textContent = s.stay;
  $('stat-total').textContent = s.total_tracked;
  $('stat-eval-sub').textContent = `${s.total_evaluated} evaluated`;
}

function renderDonut(s) {
  const total = s.switch + s.hold + s.stay || 1;
  const swDeg = (s.switch / total) * 360;
  const hoDeg = (s.hold / total) * 360;
  const swPct = Math.round((s.switch / total) * 100);
  const hoPct = Math.round((s.hold / total) * 100);
  const stPct = 100 - swPct - hoPct;
  $('donut').style.background =
    `conic-gradient(#22c55e 0deg ${swDeg}deg, #f97316 ${swDeg}deg ${swDeg + hoDeg}deg, #3b82f6 ${swDeg + hoDeg}deg 360deg)`;
  $('donut').innerHTML = `<div class="donut-hole">${total}<br>evals</div>`;
  $('donut-legend').innerHTML = `
    <div class="leg"><div class="leg-dot" style="background:#22c55e"></div>Switch ${swPct}%</div>
    <div class="leg"><div class="leg-dot" style="background:#f97316"></div>Hold ${hoPct}%</div>
    <div class="leg"><div class="leg-dot" style="background:#3b82f6"></div>Stay ${stPct}%</div>`;
}

function renderVerdictGrid(verdicts) {
  const grid = $('verdict-grid');
  if (!verdicts.length) {
    grid.innerHTML = '<div style="color:#94a3b8;font-size:.8rem;padding:12px">No verdicts logged yet. Run the agent first.</div>';
    return;
  }

  // Sort: SWITCH first (ranked by confidence), then HOLD, then STAY
  const order = {SWITCH: 0, HOLD: 1, STAY: 2};
  const sorted = [...verdicts].sort((a, b) => {
    const oa = order[a.verdict] ?? 3;
    const ob = order[b.verdict] ?? 3;
    if (oa !== ob) return oa - ob;
    // within same verdict, higher confidence first
    return (b.verdict_token_prob ?? 0) - (a.verdict_token_prob ?? 0);
  });

  grid.innerHTML = sorted.map(v => {
    const isStay = v.verdict === 'STAY';
    const prob = v.verdict_token_prob;
    const probTag = prob != null
      ? `<span style="font-size:.65rem;color:${confColor(prob)};font-family:monospace">P=${prob.toFixed(2)}</span>`
      : '';

    const accBody = v.summary
      ? `<button class="acc-toggle" onclick="toggleAcc(this)"><span class="acc-arrow">▶</span> Why ${v.verdict.toLowerCase()}?</button>
         <div class="acc-body">
           <div class="acc-meta">Qwen2.5-3B · ${v.ts?.slice(0,10) || ''}</div>
           ${v.summary}
         </div>`
      : (isStay ? '' : `<div class="acc-pending">▸ Run <strong style="color:#16a34a;margin:0 3px">Generate Summaries</strong> for context</div>`);

    const route = v.memo_excerpt
      ? `<div class="vcard-route">${v.memo_excerpt.slice(0, 60)}…</div>`
      : '';

    return `
      <div class="vcard ${isStay ? 'stay' : ''}">
        <div class="vcard-top">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            ${verdictBadge(v.verdict)}
            ${probTag}
          </div>
          <div class="vcard-name">${v.competitor}</div>
          ${route}
          <div class="vcard-meta">${v.category} · ${v.ts?.slice(0,10) || '—'}</div>
        </div>
        ${accBody}
        ${feedbackRow(v.memo_filename, v.verdict)}
      </div>`;
  }).join('');
}

function feedbackRow(filename, verdict) {
  const id = filename.replace(/[^a-z0-9]/gi, '_');
  return `
    <div class="fb-row" id="fb-${id}">
      <span class="fb-label">Correct?</span>
      <div class="fb-btns">
        <button class="fb-btn" id="yes-${id}" onclick="voteFb('${filename}','${verdict}',true,'${id}')">👍</button>
        <button class="fb-btn" id="no-${id}" onclick="openFbNote('${filename}','${verdict}','${id}')">👎</button>
      </div>
    </div>
    <div class="fb-note" id="note-${id}">
      <textarea id="note-text-${id}" rows="2" placeholder="What was wrong? (optional)"></textarea>
      <button class="btn btn-ghost submit-btn" style="font-size:.72rem;padding:4px 10px"
              onclick="submitFbNote('${filename}','${verdict}','${id}')">Submit</button>
    </div>`;
}

function toggleAcc(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

function openFbNote(filename, verdict, id) {
  $(`yes-${id}`)?.classList.remove('yes');
  $(`no-${id}`)?.classList.add('no');
  $(`note-${id}`).style.display = 'block';
}

async function voteFb(filename, verdict, correct, id) {
  $(`yes-${id}`)?.classList.toggle('yes', correct);
  $(`no-${id}`)?.classList.toggle('no', !correct);
  if (correct) {
    $(`note-${id}`).style.display = 'none';
    await submitFeedback(filename, verdict, correct, verdict, '');
    disableFbRow(id);
  }
}

async function submitFbNote(filename, verdict, id) {
  const note = $(`note-text-${id}`)?.value || '';
  const actualInput = prompt('What should the verdict have been? (SWITCH / STAY / HOLD)', verdict);
  const actual = ['SWITCH','STAY','HOLD'].includes(actualInput) ? actualInput : verdict;
  await submitFeedback(filename, verdict, false, actual, note);
  disableFbRow(id);
}

async function submitFeedback(filename, stated, correct, actual, note) {
  await fetch('/api/feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({memo_filename: filename, correct, stated_verdict: stated, actual_verdict: actual, note}),
  });
}

function disableFbRow(id) {
  const row = $(`fb-${id}`);
  if (row) row.innerHTML = '<span class="fb-done">✓ Feedback saved</span>';
  const note = $(`note-${id}`);
  if (note) note.remove();
}

function renderConfBars(verdicts) {
  const withConf = verdicts.filter(v => v.verdict_token_prob != null);
  if (!withConf.length) {
    $('conf-bars').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No live model data yet</span>';
    return;
  }
  $('conf-bars').innerHTML = withConf.slice(0, 6).map(v => {
    const p = v.verdict_token_prob;
    const pct = Math.round(p * 100);
    const col = confColor(p);
    return `<div class="conf-item">
      <div class="conf-name">${v.competitor}</div>
      <div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${col}"></div></div>
      <div class="conf-val" style="color:${col}">${p.toFixed(2)}</div>
    </div>`;
  }).join('');
}

function renderMonitor(verdicts) {
  const holds = verdicts.filter(v => v.verdict === 'HOLD');
  if (!holds.length) {
    $('monitor-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No active HOLDs</span>';
    return;
  }
  $('monitor-list').innerHTML = holds.map(v => `
    <div class="mon-item">
      <div class="mon-name">${v.competitor}</div>
      <div class="mon-cond">${v.category} · review due ${v.ts?.slice(0,10) || '—'}</div>
    </div>`).join('');
}

function renderFeedbackHistory(health) {
  $('fb-history').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">Feedback appears here after you vote on cards above</span>';
}

function renderRegressionTests(health) {
  const last = health.last_accuracy;
  if (!last) {
    $('reg-score').textContent = '—';
    $('reg-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No regression tests run yet. Run: python scripts/drift_check.py</span>';
    return;
  }
  $('reg-score').textContent = `${last.correct}/${last.total}`;
  const results = (last.results || []).filter(r => r.status === 'ok');
  if (results.length) {
    $('reg-list').innerHTML = results.map(r => `
      <div class="reg-item">
        <div><div class="reg-name">${r.file.replace('.json','')}</div><div class="reg-exp">Expected ${r.expected}</div></div>
        <div class="${r.correct ? 'pass' : 'fail'}">${r.actual} ${r.correct ? '✓' : '✗'}</div>
      </div>`).join('');
  } else {
    $('reg-list').innerHTML = `<div style="color:#94a3b8;font-size:.75rem">${last.correct}/${last.total} canaries passed</div>`;
  }
}

// ── Competitors view ──────────────────────────────────────────────────────────

async function loadCompetitors() {
  const [compR, vR] = await Promise.all([
    fetch('/api/competitors'),
    fetch('/api/verdicts?n=100'),
  ]);
  const comps = await compR.json();
  const verdicts = await vR.json();

  const lastVerdict = {};
  for (const v of verdicts) { lastVerdict[v.competitor] = {verdict: v.verdict, ts: v.ts?.slice(0,10)}; }

  const tbody = $('comp-tbody');
  tbody.innerHTML = comps.map(c => {
    const lv = lastVerdict[c.slug];
    return `<tr>
      <td>${c.category}</td>
      <td><strong>${c.name}</strong></td>
      <td>£${c.monthly_cost_gbp ?? '—'}</td>
      <td>${lv ? verdictBadge(lv.verdict) : '<span style="color:#94a3b8">not run</span>'}</td>
      <td style="color:#94a3b8;font-size:.75rem">${lv?.ts || '—'}</td>
      <td><input class="url-input" value="${c.scraper_url || ''}" placeholder="https://…"
           data-slug="${c.slug}" onblur="saveUrl(this)"></td>
    </tr>`;
  }).join('');
}

async function saveUrl(input) {
  // Scraper URL persistence is read-only for now — field is editable in the UI
  // but a PATCH /api/competitors/:slug endpoint would be needed to persist
}

function toggleAddForm() { $('add-form').classList.toggle('open'); }

function autoSlug() {
  $('fc-slug').value = $('fc-name').value.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

async function submitAddCompetitor() {
  const body = {
    category: $('fc-category').value,
    name: $('fc-name').value.trim(),
    slug: $('fc-slug').value.trim(),
    monthly_cost_gbp: parseFloat($('fc-cost').value) || 0,
    scraper_url: $('fc-url').value.trim(),
  };
  if (!body.name || !body.slug) { alert('Name and slug are required'); return; }
  const r = await fetch('/api/competitors', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (r.ok) { toggleAddForm(); loadCompetitors(); }
  else { alert('Failed to add competitor'); }
}

// ── Model Health view ─────────────────────────────────────────────────────────

async function loadHealth() {
  const r = await fetch('/api/health');
  const h = await r.json();

  const trend = h.confidence_trend || [];
  if (!trend.length) {
    $('health-conf').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No live model confidence data. Run the agent without --dry-run.</span>';
  } else {
    $('health-conf').innerHTML = trend.map(t => {
      const p = t.prob;
      const col = confColor(p);
      return `<div class="conf-item">
        <div class="conf-name" style="width:90px">${t.ts} ${t.competitor}</div>
        <div class="conf-track"><div class="conf-fill" style="width:${Math.round(p*100)}%;background:${col}"></div></div>
        <div class="conf-val" style="color:${col}">${p?.toFixed(3) ?? '—'}</div>
      </div>`;
    }).join('');
  }

  const history = h.accuracy_history || [];
  if (!history.length) {
    $('health-reg-score').textContent = '—';
    $('health-reg-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No accuracy checks. Run: python scripts/drift_check.py</span>';
  } else {
    const last = history[history.length - 1];
    $('health-reg-score').textContent = `${last.correct}/${last.total}`;
    $('health-reg-list').innerHTML = [...history].reverse().map(entry => {
      const col = entry.accuracy >= 0.9 ? '#16a34a' : entry.accuracy >= 0.75 ? '#d97706' : '#dc2626';
      return `<div class="reg-item">
        <div><div class="reg-name">${entry.ts.slice(0,10)}</div><div class="reg-exp">${entry.correct}/${entry.total} correct</div></div>
        <div style="color:${col};font-weight:700;font-size:.78rem">${Math.round(entry.accuracy*100)}%</div>
      </div>`;
    }).join('');
  }
}

// ── Run actions ───────────────────────────────────────────────────────────────

async function runEval() {
  const inbox = prompt('Inbox file path (leave blank for dry-run):');
  const dry = !inbox;
  const r = await fetch('/api/run-eval', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({inbox_file: inbox || '', dry_run: dry}),
  });
  if (r.status === 409) { alert('Model is already running'); return; }
  if (r.ok) startStatusPoll();
}

async function runSummarise() {
  const r = await fetch('/api/run-summaries', {method: 'POST'});
  if (r.status === 409) { alert('Model is already running'); return; }
  if (r.ok) startStatusPoll();
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadDashboard();
pollStatus();
