// ── Utilities ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

// Competitor name cache — loaded once from /api/competitors
let _compNames = {};
let _compLastUpdated = {};
async function _ensureCompNames() {
  if (Object.keys(_compNames).length) return;
  try {
    const r = await fetch('/api/competitors');
    const comps = await r.json();
    for (const c of comps) {
      _compNames[c.slug] = c.name;
      if (c.last_updated) _compLastUpdated[c.slug] = c.last_updated;
    }
  } catch (_) {}
}

function displayName(slug) {
  return _compNames[slug] || slug.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function displayNameVersion(slug) {
  const name = displayName(slug);
  const lu = _compLastUpdated[slug];
  if (!lu) return name;
  // "2024-10-20" → "v24.10"
  const ver = 'v' + lu.slice(2, 4) + '.' + lu.slice(5, 7);
  return `${name} <span class="ver-tag">${ver}</span>`;
}

function fmtCategory(cat) {
  const map = {project_mgmt: 'Project Management', crm: 'CRM', hr: 'HR'};
  return map[cat] || (cat ? cat.charAt(0).toUpperCase() + cat.slice(1) : '');
}

function confColor(p) {
  if (p === null || p === undefined) return '#94a3b8';
  return p >= 0.80 ? '#22c55e' : p >= 0.65 ? '#f59e0b' : '#ef4444';
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

function fmtDate(ts) {
  if (!ts) return '—';
  const d = ts.slice(0, 10);
  const t = ts.slice(11, 16);
  return t && t !== '00:00' ? `${d} ${t}` : d;
}

// ── Navigation ────────────────────────────────────────────────────────────────

function showView(name, btn) {
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  $(`view-${name}`).classList.add('active');
  btn.classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'history') loadHistory();
  if (name === 'competitors') loadCompetitors();
  if (name === 'health') loadHealth();
  if (name === 'review') loadReview();
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
  await _ensureCompNames();
  const [statsR, verdictsR, healthR, rankingsR, holdsR] = await Promise.all([
    fetch('/api/stats'),
    fetch('/api/verdicts?n=6'),
    fetch('/api/health'),
    fetch('/api/rankings'),
    fetch('/api/holds'),
  ]);
  const stats    = await statsR.json();
  const verdicts = await verdictsR.json();
  const health   = await healthR.json();
  const rankings = await rankingsR.json();
  const holds    = await holdsR.json();

  renderStats(stats);
  renderDonut(stats);
  renderRankings(rankings);
  renderVerdictGrid(verdicts);
  renderConfBars(verdicts);
  renderMonitor(holds);
  renderFeedbackSidebar(health);
  updateReviewBadge();
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
           <span class="acc-arrow">▶</span> Summary
         </button>
         <div class="acc-body">
           <div class="acc-meta">AI summary · ${fmtDate(v.ts)}</div>
           ${v.summary}
         </div>`
      : (isStay
          ? ''
          : `<div class="acc-pending">▸ Summary pending</div>`);

    return `
      <div class="vcard ${isStay ? 'stay' : ''}">
        <div class="vcard-top">
          <div style="display:flex;align-items:center;gap:4px;margin-bottom:7px">
            ${verdictBadge(v.verdict)}${probTag}
          </div>
          <div class="vcard-name">${displayName(v.competitor)}</div>
          <div class="vcard-meta"><span class="rank-cat">${fmtCategory(v.category)}</span> · ${fmtDate(v.ts)}</div>
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
        <button class="fb-btn" id="no-${id}"  onclick="voteFb('${filename}','${verdict}',false,'${id}')">👎</button>
      </div>
    </div>
    <div class="fb-note" id="note-${id}">
      <textarea id="note-text-${id}" rows="2" placeholder="What was wrong? (optional)"></textarea>
      <div style="display:flex;gap:5px;align-items:center;margin-top:5px;flex-wrap:wrap">
        <label style="font-size:.68rem;color:var(--subtle)">Should be:</label>
        <select id="actual-${id}" style="font-size:.72rem;padding:3px 6px;border:1px solid var(--border);border-radius:5px;background:var(--card)">
          <option value="SWITCH"${verdict==='SWITCH'?' selected':''}>SWITCH</option>
          <option value="STAY"${verdict==='STAY'?' selected':''}>STAY</option>
          <option value="HOLD"${verdict==='HOLD'?' selected':''}>HOLD</option>
        </select>
        <button class="btn btn-primary" style="font-size:.72rem;padding:4px 10px"
                onclick="submitFbNote('${filename}','${verdict}','${id}')">Submit</button>
        <button class="btn btn-ghost" style="font-size:.72rem;padding:4px 10px"
                onclick="cancelFb('${id}')">Cancel</button>
      </div>
    </div>`;
}

function toggleAcc(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

function cancelFb(id) {
  $(`yes-${id}`)?.classList.remove('yes');
  $(`no-${id}`)?.classList.remove('no');
  const yesBtn = $(`yes-${id}`);
  const noBtn = $(`no-${id}`);
  if (yesBtn) yesBtn.disabled = false;
  if (noBtn) noBtn.disabled = false;
  $(`note-${id}`).style.display = 'none';
}

async function voteFb(filename, verdict, correct, id) {
  const yesBtn = $(`yes-${id}`);
  const noBtn = $(`no-${id}`);
  if (correct) {
    // Thumbs up: save immediately, lock row
    yesBtn?.classList.add('yes');
    noBtn?.classList.remove('no');
    if (yesBtn) yesBtn.disabled = true;
    if (noBtn) noBtn.disabled = true;
    $(`note-${id}`).style.display = 'none';
    await submitFeedback(filename, verdict, true, verdict, '');
    disableFbRow(id);
  } else {
    // Thumbs down: show note form, do NOT save yet
    noBtn?.classList.add('no');
    yesBtn?.classList.remove('yes');
    if (yesBtn) yesBtn.disabled = true;
    if (noBtn) noBtn.disabled = true;
    $(`note-${id}`).style.display = 'block';
  }
}

async function submitFbNote(filename, verdict, id) {
  const note = $(`note-text-${id}`)?.value || '';
  const actual = $(`actual-${id}`)?.value || verdict;
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
  const noteEl = $(`note-${id}`);
  if (noteEl) noteEl.remove();
  // If this card is in the review queue, fade it out and update the badge
  const reviewCard = $(`review-card-${id}`);
  if (reviewCard) {
    setTimeout(() => reviewCard.remove(), 900);
    const badge = $('review-badge');
    if (badge) {
      const n = parseInt(badge.textContent || '0') - 1;
      if (n <= 0) badge.style.display = 'none';
      else badge.textContent = n;
    }
  }
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
      <div class="conf-name">${displayName(v.competitor)}</div>
      <div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${col}"></div></div>
      <div class="conf-val" style="color:${col}">${p.toFixed(2)}</div>
    </div>`;
  }).join('');
}

function renderMonitor(holds) {
  if (!holds.length) {
    $('monitor-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No active HOLDs</span>';
    return;
  }
  $('monitor-list').innerHTML = holds.map(h => {
    const condition = h.reassess_condition
      ? `<div class="mon-cond">${h.reassess_condition}</div>`
      : '';
    const review = h.review_by
      ? `<div class="mon-chips"><span class="chip chip-watch">Review by ${h.review_by}</span></div>`
      : '';
    return `<div class="mon-item">
      <div class="mon-name">${displayName(h.competitor)}</div>
      <div style="font-size:.68rem;color:var(--subtle)"><span class="rank-cat">${fmtCategory(h.category)}</span> · ${fmtDate(h.ts)}</div>
      ${condition}
      ${review}
    </div>`;
  }).join('');
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
    const name = _extractDisplayName(f.memo_filename);
    return `<div class="fb-history-item">
      <span class="fb-filename">${name}</span>
      <span style="color:${col}">${icon}</span>
    </div>`;
  }).join('');
}

function _extractDisplayName(memo_filename) {
  if (!memo_filename) return '—';
  // Strip date prefix and .md extension: "2026-04-13-project_mgmt-flowboard.md" → "flowboard"
  const stripped = memo_filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
  // Split on first hyphen: "project_mgmt-flowboard" → category + slug
  const dashIdx = stripped.indexOf('-');
  if (dashIdx >= 0) {
    const slug = stripped.slice(dashIdx + 1);
    const cat = stripped.slice(0, dashIdx);
    return `${displayName(slug)} <span class="rank-cat">${fmtCategory(cat)}</span>`;
  }
  return displayName(stripped);
}

// ── History view ──────────────────────────────────────────────────────────────

async function loadHistory() {
  await _ensureCompNames();
  const r = await fetch('/api/history');
  const all = await r.json();
  const filter = $('history-filter')?.value || 'all';
  const filtered = filter === 'all' ? all : all.filter(v => v.verdict === filter);

  // Update total count in header
  const countEl = $('history-count');
  if (countEl) {
    countEl.textContent = filter === 'all'
      ? `${all.length} verdict${all.length !== 1 ? 's' : ''}`
      : `${filtered.length} of ${all.length}`;
  }

  const el = $('history-list');
  if (!filtered.length) {
    el.innerHTML = '<div style="color:#94a3b8;text-align:center;padding:20px">No verdicts found.</div>';
    return;
  }

  el.innerHTML = filtered.map(v => {
    const prob = v.verdict_token_prob;
    const probStr = prob != null ? prob.toFixed(4) : '—';
    const col = confColor(prob);
    const summary = v.summary
      ? `<div class="rank-summary">${v.summary}</div>`
      : '<div class="rank-summary" style="color:#94a3b8;font-style:italic">Summary pending</div>';
    const fb = v.memo_filename ? feedbackRow(v.memo_filename, v.verdict) : '';
    return `<div class="rank-item-card">
      <div class="rank-item">
        <div style="min-width:80px">${verdictBadge(v.verdict)}</div>
        <div class="rank-body">
          <div class="rank-header">
            <span class="rank-name">${displayNameVersion(v.competitor)}</span>
            <span class="rank-cat">${fmtCategory(v.category)}</span>
            <span class="rank-conf" style="color:${col}">P=${probStr}</span>
          </div>
          ${summary}
          <div class="rank-ts">${fmtDate(v.ts)}</div>
        </div>
      </div>
      ${fb}
    </div>`;
  }).join('');
}

// ── Competitors view ──────────────────────────────────────────────────────────

async function loadCompetitors() {
  await _ensureCompNames();
  const [compR, vR] = await Promise.all([
    fetch('/api/competitors'),
    fetch('/api/verdicts?n=100'),
  ]);
  const comps    = await compR.json();
  const verdicts = await vR.json();

  const lastVerdict = {};
  for (const v of verdicts) lastVerdict[v.competitor] = {verdict: v.verdict, ts: fmtDate(v.ts)};

  // Group by category, sorted alphabetically
  const groups = {};
  for (const c of comps) {
    const cat = c.category || 'other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(c);
  }

  let rows = '';
  for (const cat of Object.keys(groups).sort()) {
    rows += `<tr class="cat-header-row"><td colspan="5">${fmtCategory(cat)}</td></tr>`;
    rows += groups[cat].map(c => {
      const lv = lastVerdict[c.slug];
      return `<tr>
        <td><strong>${c.name}</strong></td>
        <td>£${c.monthly_cost_gbp ?? '—'}</td>
        <td>${lv ? verdictBadge(lv.verdict) : '<span style="color:#94a3b8">not run</span>'}</td>
        <td style="color:#94a3b8;font-size:.75rem">${lv?.ts || '—'}</td>
        <td><input class="url-input" value="${c.scraper_url || ''}" placeholder="https://…"
             data-slug="${c.slug}"></td>
      </tr>`;
    }).join('');
  }
  $('comp-tbody').innerHTML = rows || '<tr><td colspan="5" style="color:#94a3b8;text-align:center;padding:20px">No competitors loaded.</td></tr>';
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
  await _ensureCompNames();
  const [healthR, statsR, verdictsR] = await Promise.all([
    fetch('/api/health'),
    fetch('/api/stats'),
    fetch('/api/verdicts?n=100'),
  ]);
  const h       = await healthR.json();
  const stats   = await statsR.json();
  const allV    = await verdictsR.json();

  loadFeedbackQueue();

  $('h-total-runs').textContent = stats.total_evaluated;

  const withConf = allV.filter(v => v.verdict_token_prob != null);
  if (withConf.length) {
    const avg = withConf.reduce((s, v) => s + v.verdict_token_prob, 0) / withConf.length;
    const low = withConf.filter(v => v.verdict_token_prob < 0.65).length;
    $('h-avg-conf').textContent = avg.toFixed(2);
    $('h-low-conf').textContent = low;
  } else {
    $('h-avg-conf').textContent = 'N/A';
    $('h-low-conf').textContent = 'N/A';
  }

  const last = h.last_accuracy;
  $('h-regression').textContent = last ? `${last.correct}/${last.total}` : '—';
  $('health-reg-score').textContent = last ? `${last.correct}/${last.total}` : '—';

  const trend = h.confidence_trend || [];
  if (!trend.length) {
    $('health-conf').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No live model confidence data. Run the agent without --dry-run.</span>';
  } else {
    $('health-conf').innerHTML = trend.slice(0, 15).map(t => {
      const col = confColor(t.prob);
      const pct = t.prob != null ? Math.round(t.prob * 100) : 0;
      return `<div class="conf-item">
        <div class="conf-name" style="width:110px">${t.ts} ${displayName(t.competitor)}</div>
        <div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${col}"></div></div>
        <div class="conf-val" style="color:${col}">${t.prob?.toFixed(3) ?? '—'}</div>
      </div>`;
    }).join('');
  }

  const history = h.accuracy_history || [];
  if (!history.length) {
    $('health-reg-list').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No checks recorded. Run: python scripts/drift_check.py</span>';
  } else {
    $('health-reg-list').innerHTML = [...history].reverse().map(entry => {
      const pct = Math.round(entry.accuracy * 100);
      const col = pct >= 90 ? 'var(--sw)' : pct >= 75 ? 'var(--ho)' : 'var(--conf-lo)';
      const bg = pct >= 90 ? 'var(--sw-bg)' : pct >= 75 ? 'var(--ho-bg)' : '#fef2f2';
      const canaries = (entry.results || []).map(r => {
        const name = r.file.replace('.json','').replace(/_/g,' ');
        const icon = r.correct ? '✓' : '✗';
        const c = r.correct ? 'var(--sw)' : 'var(--conf-lo)';
        return `<div style="display:flex;align-items:center;gap:5px;padding:4px 8px;border-radius:5px;background:${r.correct?'var(--sw-bg)':'#fef2f2'};border:1px solid ${r.correct?'var(--sw-border)':'#fca5a5'}">
          <span style="font-size:.7rem;font-weight:700;color:${c}">${icon}</span>
          <span style="font-size:.68rem;color:var(--text)">${name}</span>
          <span style="font-size:.62rem;color:var(--subtle);margin-left:auto">${r.actual}</span>
        </div>`;
      }).join('');
      return `<div style="padding:12px 0;border-bottom:1px solid var(--border)">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div>
            <div style="font-size:.78rem;font-weight:700;color:var(--text)">${entry.ts.slice(0,10)} ${entry.ts.slice(11,16)}</div>
            <div style="font-size:.68rem;color:var(--subtle)">${entry.correct}/${entry.total} canaries correct</div>
          </div>
          <div style="background:${bg};color:${col};font-weight:800;font-size:.9rem;padding:4px 10px;border-radius:6px">${pct}%</div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">${canaries}</div>
      </div>`;
    }).join('');
  }

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

  const fb = h.feedback_log || [];
  if (!fb.length) {
    $('health-feedback').innerHTML = '<span style="color:#94a3b8;font-size:.75rem">No feedback recorded yet. Use 👍/👎 on verdict cards.</span>';
  } else {
    $('health-feedback').innerHTML = `
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="font-size:.68rem;color:var(--subtle);text-transform:uppercase">
          <th style="text-align:left;padding:6px 10px;border-bottom:1px solid var(--border)">Tool</th>
          <th style="padding:6px 10px;border-bottom:1px solid var(--border)">Stated</th>
          <th style="padding:6px 10px;border-bottom:1px solid var(--border)">Actual</th>
          <th style="padding:6px 10px;border-bottom:1px solid var(--border)">Result</th>
          <th style="text-align:left;padding:6px 10px;border-bottom:1px solid var(--border)">Note</th>
        </tr></thead>
        <tbody>${fb.map(f => `<tr style="font-size:.82rem">
          <td style="padding:8px 10px;font-size:.76rem;color:var(--muted)">${_extractDisplayName(f.memo_filename)}</td>
          <td style="text-align:center;padding:8px 10px">${verdictBadge(f.stated_verdict)}</td>
          <td style="text-align:center;padding:8px 10px">${verdictBadge(f.actual_verdict)}</td>
          <td style="text-align:center;padding:8px 10px">${f.correct ? '<span style="color:var(--sw);font-weight:700">👍</span>' : '<span style="color:var(--conf-lo);font-weight:700">👎</span>'}</td>
          <td style="padding:8px 10px;color:var(--muted)">${f.note || '—'}</td>
        </tr>`).join('')}</tbody>
      </table>`;
  }
}

// ── Rankings ──────────────────────────────────────────────────────────────────

function renderRankings(rankings) {
  const el = $('rankings-list');
  if (!rankings.length) {
    el.innerHTML = `<div style="text-align:center;padding:18px 12px">
      <div style="font-size:.8rem;font-weight:700;color:var(--text)">No switch recommendations</div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:4px">All tools are currently rated Stay or Hold</div>
      <div style="font-size:.68rem;color:var(--subtle);margin-top:8px;line-height:1.6">A switch appears here when a competitor resolves a key push issue and the ROI threshold is met.</div>
    </div>`;
    return;
  }
  el.innerHTML = rankings.map(r => {
    const conf = r.confidence;
    const confStr = conf != null ? conf.toFixed(4) : '—';
    const col = confColor(conf);
    const summary = r.summary
      ? `<div class="rank-summary">${r.summary}</div>`
      : '<div class="rank-summary" style="color:#94a3b8;font-style:italic">Summary pending</div>';
    const fb = r.memo_filename ? feedbackRow(r.memo_filename, 'SWITCH') : '';
    return `<div class="rank-item-card">
      <div class="rank-item">
        <div class="rank-num">#${r.rank}</div>
        <div class="rank-body">
          <div class="rank-header">
            <span class="rank-name">${displayName(r.competitor)}</span>
            <span class="rank-cat">${fmtCategory(r.category)}</span>
            <span class="rank-conf" style="color:${col}">P=${confStr}</span>
          </div>
          ${summary}
          <div class="rank-ts">${fmtDate(r.ts)}</div>
        </div>
      </div>
      ${fb}
    </div>`;
  }).join('');
}

// ── Feedback Learning (DPO) ──────────────────────────────────────────────────

async function loadFeedbackQueue() {
  try {
    const r = await fetch('/api/feedback-queue');
    const q = await r.json();
    $('rt-human').textContent = q.human_corrections;
    $('rt-canary').textContent = q.canary_failures;
    $('rt-total').textContent = q.total_corrections;

    const btn = $('btn-retrain');
    const badge = $('retrain-badge');
    if (q.total_corrections > 0) {
      btn.disabled = false;
      badge.textContent = `${q.total_corrections} pending`;
      badge.style.background = 'var(--ho-bg)';
      badge.style.color = 'var(--ho)';
    } else {
      btn.disabled = true;
      badge.textContent = 'up to date';
      badge.style.background = 'var(--sw-bg)';
      badge.style.color = 'var(--sw)';
    }
    if (q.has_dpo_adapter) {
      $('retrain-status').textContent = 'DPO adapter available at training/checkpoints_dpo/';
    }
  } catch (e) {
    console.error('Failed to load feedback queue', e);
  }
}

async function triggerRetrain() {
  const btn = $('btn-retrain');
  btn.disabled = true;
  btn.textContent = 'Training…';
  $('retrain-status').textContent = 'Harvesting pairs and running DPO training…';
  try {
    const r = await fetch('/api/retrain', { method: 'POST' });
    const d = await r.json();
    if (r.ok) {
      $('retrain-status').textContent = 'Training started. Canary gate will run automatically after.';
    } else {
      $('retrain-status').textContent = d.error || 'Failed to start training';
      btn.disabled = false;
    }
  } catch (e) {
    $('retrain-status').textContent = 'Error: ' + e.message;
    btn.disabled = false;
  }
  btn.textContent = 'Retrain Model';
}

async function triggerCanary() {
  const btn = $('btn-canary');
  btn.disabled = true;
  btn.textContent = 'Running…';
  $('retrain-status').textContent = 'Running canary regression check…';
  try {
    const r = await fetch('/api/run-canary', { method: 'POST' });
    const d = await r.json();
    if (r.ok) {
      $('retrain-status').textContent = 'Canary check started. Refresh Model Health when complete.';
    } else {
      $('retrain-status').textContent = d.error || 'Failed to start canaries';
    }
  } catch (e) {
    $('retrain-status').textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Run Canaries';
}

// ── Review Queue ─────────────────────────────────────────────────────────────

async function loadReview() {
  await _ensureCompNames();
  const r = await fetch('/api/review-queue');
  const data = await r.json();
  const queue = data.queue || [];

  const badge = $('review-badge');
  if (badge) {
    if (queue.length > 0) {
      badge.textContent = queue.length;
      badge.style.display = 'inline';
    } else {
      badge.style.display = 'none';
    }
  }

  const el = $('review-list');
  if (!queue.length) {
    el.innerHTML = '<div style="color:#94a3b8;text-align:center;padding:28px">No low-confidence predictions pending review.</div>';
    return;
  }

  el.innerHTML = queue.map(item => {
    const id = item.memo_filename.replace(/[^a-z0-9]/gi, '_');
    const prob = item.verdict_token_prob;
    const col = confColor(prob);
    const margin = item.verdict_margin != null ? ` margin=${item.verdict_margin.toFixed(3)}` : '';
    const summarySection = item.summary
      ? `<div class="review-summary">${item.summary}</div>`
      : `<div class="review-summary review-summary-pending">Summary pending</div>`;
    return `
      <div class="review-card" id="review-card-${id}">
        <div class="review-card-top">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
            ${verdictBadge(item.verdict)}
            <span style="font-size:.68rem;font-family:monospace;color:${col};font-weight:700">P=${prob.toFixed(3)}${margin}</span>
          </div>
          <div class="vcard-name">${displayName(item.competitor)}</div>
          <div class="vcard-meta">
            <span class="rank-cat">${fmtCategory(item.category)}</span> · ${item.ts}
          </div>
          ${summarySection}
        </div>
        ${feedbackRow(item.memo_filename, item.verdict)}
      </div>`;
  }).join('');
}

async function updateReviewBadge() {
  try {
    const r = await fetch('/api/review-queue');
    const data = await r.json();
    const badge = $('review-badge');
    if (!badge) return;
    const n = data.count || 0;
    if (n > 0) { badge.textContent = n; badge.style.display = 'inline'; }
    else badge.style.display = 'none';
  } catch (_) {}
}

// ── Init ──────────────────────────────────────────────────────────────────────

loadDashboard();
pollStatus();

setInterval(() => {
  const active = document.querySelector('.view.active');
  if (active && active.id === 'view-dashboard') loadDashboard();
}, 30000);
