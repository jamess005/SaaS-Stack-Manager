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

function fmtDate(ts) { return ts ? ts.slice(0, 10) : '—'; }

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

// ── Status polling (model lock) ───────────────────────────────────────────────

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
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const chip = $('status-chip');
    if (d.busy) {
      chip.style.display = 'inline-flex';
      $('status-text').textContent = `${d.task || 'model'} running…`;
    } else {
      if (chip.style.display !== 'none') loadDashboard();
      chip.style.display = 'none';
      stopStatusPoll();
    }
  } catch (_) {}
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

async function loadDashboard() {
  const [statsR, verdictsR, healthR] = await Promise.all([
    fetch('/api/stats'),
    fetch('/api/verdicts?n=10'),
    fetch('/api/health'),
  ]);
  const stats    = await statsR.json();
  const verdicts = await verdictsR.json();
  const health   = await healthR.json();

  renderStats(stats);
  renderDonut(stats);
  renderVerdictGrid(verdicts);
  renderConfBars(verdicts);
  renderMonitor(verdicts);
  renderFeedbackSidebar(health);
}

function renderStats(s) {
  $('stat-sw').textContent    = s.switch;
  $('stat-ho').textContent    = s.hold;
  $('stat-st').textContent    = s.stay;
  $('stat-total').textContent = s.total_tracked;
  $('stat-eval-sub').textContent = `${s.total_evaluated} evaluated`;
}

function renderDonut(s) {
  const total = s.switch + s.hold + s.stay || 1;
  const swDeg = (s.switch / total) * 360;
  const hoDeg = (s.hold   / total) * 360;
  const swPct = Math.round((s.switch / total) * 100);
  const hoPct = Math.round((s.hold   / total) * 100);
  const stPct = 100 - swPct - hoPct;
  $('donut').style.background =
    `conic-gradient(#22c55e 0deg ${swDeg}deg, #f97316 ${swDeg}deg ${swDeg+hoDeg}deg, #3b82f6 ${swDeg+hoDeg}deg 360deg)`;
  $('donut').innerHTML = `<div class="donut-hole">${total}<br>evals</div>`;
  $('donut-legend').innerHTML = `
    <div class="leg"><div class="leg-dot" style="background:#22c55e"></div>Switch ${swPct}%</div>
    <div class="leg"><div class="leg-dot" style="background:#f97316"></div>Hold ${hoPct}%</div>
    <div class="leg"><div class="leg-dot" style="background:#3b82f6"></div>Stay ${stPct}%</div>`;
}

function renderVerdictGrid(verdicts) {
  const grid = $('verdict-grid');
  if (!verdicts.length) {
    grid.innerHTML = '<div style="color:#94a3b8;font-size:.8rem;padding:12px;grid-column:1/-1">No verdicts logged yet.</div>';
    return;
  }

  // SWITCH first (ranked by confidence), then HOLD, then STAY (faded)
  const order = {SWITCH: 0, HOLD: 1, STAY: 2};
  const sorted = [...verdicts].sort((a, b) => {
    const oa = order[a.verdict] ?? 3, ob = order[b.verdict] ?? 3;
    if (oa !== ob) return oa - ob;
    return (b.verdict_token_prob ?? 0) - (a.verdict_token_prob ?? 0);
  });

  grid.innerHTML = sorted.map(v => {
    const isStay = v.verdict === 'STAY';
    const prob   = v.verdict_token_prob;
    const probTag = prob != null
      ? `<span style="font-size:.65rem;color:${confColor(prob)};font-family:monospace;margin-left:auto">P=${prob.toFixed(2)}</span>`
      : '';

    const whySection = v.summary
      ? `<button class="acc-toggle" onclick="toggleAcc(this)">
           <span class="acc-arrow">▶</span> Why ${v.verdict.toLowerCase()}?
         </button>
         <div class="acc-body">
           <div class="acc-meta">AI summary · ${fmtDate(v.ts)}</div>
           ${v.summary}
         </div>`
      : (isStay
          ? ''
          : `<div class="acc-pending">▸ Summary generates automatically on next eval run</div>`);

    return `
      <div class="vcard ${isStay ? 'stay' : ''}">
        <div class="vcard-top">
          <div style="display:flex;align-items:center;gap:4px;margin-bottom:7px">
            ${verdictBadge(v.verdict)}${probTag}
          </div>
          <div class="vcard-name">${v.competitor}</div>
          <div class="vcard-meta">${v.category} · ${fmtDate(v.ts)}</div>
        </div>
        ${whySection}
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
        <button class="fb-btn" id="no-${id}"  onclick="openFbNote('${filename}','${verdict}','${id}')">👎</button>
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
  const note        = $(`note-text-${id}`)?.value || '';
  const actualInput = prompt('What should the verdict have been? (SWITCH / STAY / HOLD)', verdict);
  const actual      = ['SWITCH','STAY','HOLD'].includes(actualInput) ? actualInput : verdict;
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
  $(`note-${id}`)?.remove();
}

function renderConfBars(verdicts) {
  const withConf = verdicts.filter(v => v.verdict_token_prob != null);
  if (!withConf.length) {
    $('conf-bars').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No live model data yet</span>';
    return;
  }
  $('conf-bars').innerHTML = withConf.slice(0, 6).map(v => {
    const p   = v.verdict_token_prob;
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
      <div class="mon-cond">${v.category} · ${fmtDate(v.ts)}</div>
    </div>`).join('');
}

function renderFeedbackSidebar(health) {
  const fb  = (health.feedback_log || []).slice(0, 5);
  const el  = $('fb-history-side');
  if (!fb.length) {
    el.innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No feedback yet — use 👍/👎 on cards</span>';
    return;
  }
  el.innerHTML = fb.map(f => {
    const icon = f.correct ? '👍' : '👎';
    const col  = f.correct ? 'var(--sw)' : 'var(--conf-lo)';
    return `<div class="fb-history-item">
      <span class="fb-filename">${f.memo_filename?.replace(/^\d{4}-\d{2}-\d{2}-/,'') || '—'}</span>
      <span style="color:${col}">${icon}</span>
    </div>`;
  }).join('');
}

// ── Competitors view ──────────────────────────────────────────────────────────

async function loadCompetitors() {
  const [compR, vR] = await Promise.all([
    fetch('/api/competitors'),
    fetch('/api/verdicts?n=100'),
  ]);
  const comps    = await compR.json();
  const verdicts = await vR.json();

  const lastVerdict = {};
  for (const v of verdicts) lastVerdict[v.competitor] = {verdict: v.verdict, ts: fmtDate(v.ts)};

  $('comp-tbody').innerHTML = comps.map(c => {
    const lv = lastVerdict[c.slug];
    return `<tr>
      <td>${c.category}</td>
      <td><strong>${c.name}</strong></td>
      <td>£${c.monthly_cost_gbp ?? '—'}</td>
      <td>${lv ? verdictBadge(lv.verdict) : '<span style="color:#94a3b8">not run</span>'}</td>
      <td style="color:#94a3b8;font-size:.75rem">${lv?.ts || '—'}</td>
      <td><input class="url-input" value="${c.scraper_url || ''}" placeholder="https://…"
           data-slug="${c.slug}"></td>
    </tr>`;
  }).join('');
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
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  if (r.ok) { toggleAddForm(); loadCompetitors(); }
  else { alert('Failed to add competitor'); }
}

// ── Model Health view ─────────────────────────────────────────────────────────

async function loadHealth() {
  const [healthR, statsR, verdictsR] = await Promise.all([
    fetch('/api/health'),
    fetch('/api/stats'),
    fetch('/api/verdicts?n=100'),
  ]);
  const h       = await healthR.json();
  const stats   = await statsR.json();
  const allV    = await verdictsR.json();

  // Stat tiles
  $('h-total-runs').textContent = stats.total_evaluated;

  const withConf = allV.filter(v => v.verdict_token_prob != null);
  if (withConf.length) {
    const avg = withConf.reduce((s, v) => s + v.verdict_token_prob, 0) / withConf.length;
    const low = withConf.filter(v => v.verdict_token_prob < 0.60).length;
    $('h-avg-conf').textContent = avg.toFixed(2);
    $('h-low-conf').textContent = low;
  } else {
    $('h-avg-conf').textContent = 'N/A';
    $('h-low-conf').textContent = 'N/A';
  }

  const last = h.last_accuracy;
  $('h-regression').textContent = last ? `${last.correct}/${last.total}` : '—';
  $('health-reg-score').textContent = last ? `${last.correct}/${last.total}` : '—';

  // Confidence trend
  const trend = h.confidence_trend || [];
  if (!trend.length) {
    $('health-conf').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No live model confidence data. Run the agent without --dry-run.</span>';
  } else {
    $('health-conf').innerHTML = trend.map(t => {
      const col = confColor(t.prob);
      const pct = t.prob != null ? Math.round(t.prob * 100) : 0;
      return `<div class="conf-item">
        <div class="conf-name" style="width:110px">${t.ts} ${t.competitor}</div>
        <div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${col}"></div></div>
        <div class="conf-val" style="color:${col}">${t.prob?.toFixed(3) ?? '—'}</div>
      </div>`;
    }).join('');
  }

  // Regression history
  const history = h.accuracy_history || [];
  if (!history.length) {
    $('health-reg-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No checks recorded. Run: python scripts/drift_check.py</span>';
  } else {
    $('health-reg-list').innerHTML = [...history].reverse().map(entry => {
      const col = entry.accuracy >= 0.9 ? '#16a34a' : entry.accuracy >= 0.75 ? '#d97706' : '#dc2626';
      const detail = (entry.results || []).map(r =>
        `<span style="font-size:.65rem;color:${r.correct?'var(--sw)':'var(--conf-lo)'}">${r.file.replace('.json','')}: ${r.actual} ${r.correct?'✓':'✗'}</span>`
      ).join('  ');
      return `<div class="reg-item" style="flex-direction:column;align-items:flex-start;gap:4px;padding:10px 0">
        <div style="display:flex;justify-content:space-between;width:100%">
          <div><div class="reg-name">${entry.ts.slice(0,10)}</div><div class="reg-exp">${entry.correct}/${entry.total} canaries correct</div></div>
          <div style="color:${col};font-weight:800;font-size:1rem">${Math.round(entry.accuracy*100)}%</div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">${detail}</div>
      </div>`;
    }).join('');
  }

  // Validation attempts
  const attempts = allV.filter(v => v.validation_attempts != null);
  if (!attempts.length) {
    $('health-validation').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No validation data yet.</span>';
  } else {
    const onePass = attempts.filter(v => v.validation_attempts === 1).length;
    const twoPass = attempts.filter(v => v.validation_attempts === 2).length;
    const total   = attempts.length;
    $('health-validation').innerHTML = `
      <div style="display:flex;gap:20px;align-items:center">
        <div style="flex:1">
          <div style="display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:4px">
            <span>Pass on first attempt</span><span style="color:var(--sw);font-weight:700">${onePass}/${total}</span>
          </div>
          <div class="conf-track" style="height:8px">
            <div class="conf-fill" style="width:${Math.round(onePass/total*100)}%;background:var(--sw)"></div>
          </div>
        </div>
        <div style="flex:1">
          <div style="display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:4px">
            <span>Needed retry</span><span style="color:${twoPass>0?'var(--ho)':'var(--subtle)'};font-weight:700">${twoPass}/${total}</span>
          </div>
          <div class="conf-track" style="height:8px">
            <div class="conf-fill" style="width:${Math.round(twoPass/total*100)}%;background:var(--ho)"></div>
          </div>
        </div>
      </div>`;
  }

  // Human feedback log
  const fb = h.feedback_log || [];
  if (!fb.length) {
    $('health-feedback').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No feedback recorded yet. Use 👍/👎 on verdict cards.</span>';
  } else {
    $('health-feedback').innerHTML = `
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="font-size:.68rem;color:var(--subtle);text-transform:uppercase">
          <th style="text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">Memo</th>
          <th style="padding:4px 8px;border-bottom:1px solid var(--border)">Stated</th>
          <th style="padding:4px 8px;border-bottom:1px solid var(--border)">Actual</th>
          <th style="padding:4px 8px;border-bottom:1px solid var(--border)">Result</th>
          <th style="text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)">Note</th>
        </tr></thead>
        <tbody>${fb.map(f => `<tr style="font-size:.78rem">
          <td style="padding:6px 8px;font-family:monospace;font-size:.7rem;color:var(--muted)">${f.memo_filename?.replace(/^\d{4}-\d{2}-\d{2}-/,'') || '—'}</td>
          <td style="text-align:center;padding:6px 8px">${verdictBadge(f.stated_verdict)}</td>
          <td style="text-align:center;padding:6px 8px">${verdictBadge(f.actual_verdict)}</td>
          <td style="text-align:center;padding:6px 8px">${f.correct ? '<span style="color:var(--sw);font-weight:700">👍</span>' : '<span style="color:var(--conf-lo);font-weight:700">👎</span>'}</td>
          <td style="padding:6px 8px;color:var(--muted)">${f.note || '—'}</td>
        </tr>`).join('')}</tbody>
      </table>`;
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadDashboard();
pollStatus();
