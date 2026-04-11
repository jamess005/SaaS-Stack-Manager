"""
Drift report — reads drift_log.jsonl and generates a self-contained HTML dashboard.

Usage:
    python scripts/drift_report.py
    python scripts/drift_report.py --open        # open in browser after generating
    python scripts/drift_report.py --last 50     # show last N live runs only
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agent.drift_tracker import load_records  # noqa: E402

_DRIFT_LOG = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"
_REPORT_OUT = _PROJECT_ROOT / "outputs" / "drift_dashboard.html"

_ALERT_FORMAT_VALID_THRESHOLD = 0.90   # < 90% format valid → alert
_ALERT_ACCURACY_THRESHOLD     = 0.75   # < 75% canary accuracy → alert
_ALERT_CITATION_THRESHOLD     = 1.5    # avg citations < 1.5 → warning


def _build_html(live_runs: list[dict], accuracy_checks: list[dict], last_n: int) -> str:
    runs = live_runs[-last_n:] if last_n else live_runs

    # ── Summary stats ──────────────────────────────────────────────────────────
    total_runs = len(runs)
    fmt_valid_rate = sum(1 for r in runs if r.get("format_valid")) / total_runs if total_runs else 0
    avg_citations = sum(r.get("citation_count", 0) for r in runs) / total_runs if total_runs else 0
    avg_push = sum(r.get("push_signal_count", 0) for r in runs) / total_runs if total_runs else 0
    last_accuracy = accuracy_checks[-1]["accuracy"] if accuracy_checks else None
    last_accuracy_ts = accuracy_checks[-1]["ts"][:10] if accuracy_checks else "—"

    verdict_counts = {"SWITCH": 0, "STAY": 0, "HOLD": 0}
    for r in runs:
        v = r.get("verdict", "")
        if v in verdict_counts:
            verdict_counts[v] += 1

    # ── Alerts ─────────────────────────────────────────────────────────────────
    alerts = []
    if total_runs > 0 and fmt_valid_rate < _ALERT_FORMAT_VALID_THRESHOLD:
        alerts.append(("error", f"Format validity {fmt_valid_rate:.0%} — below 90% threshold. Model may be producing malformed outputs."))
    if last_accuracy is not None and last_accuracy < _ALERT_ACCURACY_THRESHOLD:
        alerts.append(("error", f"Canary accuracy {last_accuracy:.0%} — below 75% threshold. Consider retraining."))
    if total_runs > 0 and avg_citations < _ALERT_CITATION_THRESHOLD:
        alerts.append(("warning", f"Average citations {avg_citations:.1f} — below 1.5. Evidence depth may be degrading."))
    if not accuracy_checks:
        alerts.append(("info", "No accuracy checks recorded yet. Run scripts/drift_check.py to baseline canary performance."))

    # ── Chart data ─────────────────────────────────────────────────────────────
    run_labels   = [r["ts"][:10] for r in runs]
    citations    = [r.get("citation_count", 0) for r in runs]
    push_counts  = [r.get("push_signal_count", 0) for r in runs]
    pull_counts  = [r.get("pull_signal_count", 0) for r in runs]
    fmt_series   = [1 if r.get("format_valid") else 0 for r in runs]
    val_attempts = [r.get("validation_attempts", 1) for r in runs]

    acc_labels   = [r["ts"][:10] for r in accuracy_checks]
    acc_values   = [round(r["accuracy"] * 100, 1) for r in accuracy_checks]

    def js(v):
        return json.dumps(v)

    alerts_html = ""
    for level, msg in alerts:
        colour = {"error": "#dc2626", "warning": "#d97706", "info": "#2563eb"}[level]
        icon   = {"error": "⚠", "warning": "⚠", "info": "ℹ"}[level]
        alerts_html += f'<div class="alert" style="border-left:4px solid {colour};background:{colour}11;padding:10px 14px;margin-bottom:10px;border-radius:4px"><strong style="color:{colour}">{icon} {level.upper()}:</strong> {msg}</div>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SaaS Stack Manager — Drift Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1   {{ font-size: 1.4rem; font-weight: 700; margin: 0 0 4px; }}
  .sub {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .card .label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .chart-wrap {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .chart-wrap h2 {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin: 0 0 16px; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 700px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>SaaS Stack Manager — Drift Dashboard</h1>
<p class="sub">Last {total_runs} live runs shown · Last canary check: {last_accuracy_ts}</p>

{alerts_html}

<div class="grid">
  <div class="card">
    <div class="label">Total runs</div>
    <div class="value">{total_runs}</div>
  </div>
  <div class="card">
    <div class="label">Format valid</div>
    <div class="value" style="color:{'#22c55e' if fmt_valid_rate >= 0.9 else '#f59e0b' if fmt_valid_rate >= 0.75 else '#ef4444'}">{fmt_valid_rate:.0%}</div>
  </div>
  <div class="card">
    <div class="label">Avg citations</div>
    <div class="value" style="color:{'#22c55e' if avg_citations >= 2 else '#f59e0b' if avg_citations >= 1.5 else '#ef4444'}">{avg_citations:.1f}</div>
  </div>
  <div class="card">
    <div class="label">Avg push signals</div>
    <div class="value">{avg_push:.1f}</div>
  </div>
  <div class="card">
    <div class="label">Last canary accuracy</div>
    <div class="value" style="color:{'#22c55e' if last_accuracy is not None and last_accuracy >= 0.9 else '#f59e0b' if last_accuracy is not None and last_accuracy >= 0.75 else '#ef4444'}">{f'{last_accuracy:.0%}' if last_accuracy is not None else '—'}</div>
  </div>
  <div class="card">
    <div class="label">Verdict split</div>
    <div class="value" style="font-size:1rem;line-height:1.8">
      <span style="color:#22c55e">SW {verdict_counts['SWITCH']}</span> ·
      <span style="color:#94a3b8">ST {verdict_counts['STAY']}</span> ·
      <span style="color:#f59e0b">HO {verdict_counts['HOLD']}</span>
    </div>
  </div>
</div>

<div class="row">
  <div class="chart-wrap">
    <h2>Evidence citations per run</h2>
    <canvas id="citChart"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Signal counts per run</h2>
    <canvas id="sigChart"></canvas>
  </div>
</div>

<div class="row">
  <div class="chart-wrap">
    <h2>Format validity &amp; validation attempts</h2>
    <canvas id="fmtChart"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Canary accuracy over time</h2>
    <canvas id="accChart"></canvas>
  </div>
</div>

<script>
const labels = {js(run_labels)};
const cfg = (data, label, color) => ({{
  type: 'line',
  data: {{ labels, datasets: [{{ label, data, borderColor: color, backgroundColor: color + '22',
    tension: 0.3, pointRadius: 3, fill: true }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ color:'#94a3b8', maxTicksLimit: 8 }}, grid: {{ color:'#ffffff11' }} }},
               y: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#ffffff11' }} }} }} }}
}});

new Chart(document.getElementById('citChart'), cfg({js(citations)}, 'Citations', '#818cf8'));

new Chart(document.getElementById('sigChart'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'Push', data: {js(push_counts)}, borderColor: '#f87171', backgroundColor: '#f8717122', tension: 0.3, pointRadius: 3, fill: false }},
    {{ label: 'Pull', data: {js(pull_counts)}, borderColor: '#34d399', backgroundColor: '#34d39922', tension: 0.3, pointRadius: 3, fill: false }},
  ]}},
  options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color:'#e2e8f0' }} }} }},
    scales: {{ x: {{ ticks: {{ color:'#94a3b8', maxTicksLimit: 8 }}, grid: {{ color:'#ffffff11' }} }},
               y: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#ffffff11' }} }} }} }}
}});

new Chart(document.getElementById('fmtChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [
    {{ label: 'Format valid', data: {js(fmt_series)}, backgroundColor: '#22c55e88', yAxisID: 'y' }},
    {{ label: 'Validation attempts', data: {js(val_attempts)}, backgroundColor: '#f59e0b88', yAxisID: 'y2', type: 'line', tension: 0.3 }},
  ]}},
  options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color:'#e2e8f0' }} }} }},
    scales: {{
      x:  {{ ticks: {{ color:'#94a3b8', maxTicksLimit: 8 }}, grid: {{ color:'#ffffff11' }} }},
      y:  {{ ticks: {{ color:'#94a3b8', stepSize: 1 }}, grid: {{ color:'#ffffff11' }}, min: 0, max: 1.2 }},
      y2: {{ position: 'right', ticks: {{ color:'#94a3b8', stepSize: 1 }}, grid: {{ display: false }}, min: 1, max: 3 }},
    }} }}
}});

new Chart(document.getElementById('accChart'), {{
  type: 'line',
  data: {{ labels: {js(acc_labels)}, datasets: [
    {{ label: 'Canary accuracy %', data: {js(acc_values)}, borderColor: '#38bdf8', backgroundColor: '#38bdf822',
      tension: 0.3, pointRadius: 5, fill: true }},
  ]}},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#ffffff11' }} }},
      y: {{ ticks: {{ color:'#94a3b8', callback: v => v + '%' }}, grid: {{ color:'#ffffff11' }},
           min: 0, max: 100,
           afterDataLimits(scale) {{ scale.max = 100; }},
      }},
    }},
    plugins: {{
      annotation: {{ annotations: {{ threshold: {{
        type: 'line', yMin: 75, yMax: 75, borderColor: '#ef444466',
        borderDash: [6,3], label: {{ content: '75% threshold', enabled: true, color: '#ef4444' }}
      }} }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drift monitoring dashboard.")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="Open dashboard in browser after generating.")
    parser.add_argument("--last", type=int, default=0, metavar="N",
                        help="Show last N live runs (default: all).")
    args = parser.parse_args()

    records = load_records(_DRIFT_LOG)
    if not records:
        print(f"No drift log found at {_DRIFT_LOG}.")
        print("Run the agent at least once to start collecting data.")
        sys.exit(0)

    live_runs       = [r for r in records if r.get("type") == "live_run"]
    accuracy_checks = [r for r in records if r.get("type") == "accuracy_check"]

    html = _build_html(live_runs, accuracy_checks, args.last)
    _REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {_REPORT_OUT}")

    if args.open_browser:
        webbrowser.open(_REPORT_OUT.as_uri())


if __name__ == "__main__":
    main()
