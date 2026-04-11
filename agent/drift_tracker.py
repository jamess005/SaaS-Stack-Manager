"""
Drift tracker — logs quality signals after each agent run.

Appends a JSON record to outputs/drift_log.jsonl after each verdict.
Two record types:
  "live_run"       — quality signals extracted from the verdict memo
  "accuracy_check" — canary eval results (written by scripts/drift_check.py)

Quality signals per run:
  format_valid         — structural completeness (all 9 sections present)
  validation_attempts  — 1 = passed first time, 2 = needed a retry
  citation_count       — evidence depth (quoted strings in EVIDENCE section)
  push_signal_count    — reasoning completeness (bullets in PUSH SIGNALS)
  pull_signal_count    — reasoning completeness (bullets in PULL SIGNALS)
  roi_threshold_met    — financial signal outcome
  roi_margin_gbp       — annual_net - £1200 threshold (positive = above)
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DRIFT_LOG = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"

# Advisory: how many live runs between accuracy checks
ACCURACY_CHECK_INTERVAL = 10

_bullet_re = re.compile(r"^\s{2,}-\s", re.MULTILINE)
_quoted_re = re.compile(r'"[^"\n]+"')


def _extract_section(memo: str, header: str) -> str:
    """Return the body of a named section (up to the next ALL-CAPS header or end)."""
    pattern = re.compile(
        rf"^\s*{re.escape(header)}\s*:(.+?)(?=^\s*[A-Z][A-Z\s]+\s*:|$)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(memo)
    return m.group(1) if m else ""


def extract_quality_signals(
    memo: str,
    roi_result: dict,
    format_valid: bool,
    validation_attempts: int,
) -> dict:
    """Extract quality proxy signals from a verdict memo and ROI result."""
    push_text = _extract_section(memo, "PUSH SIGNALS")
    pull_text = _extract_section(memo, "PULL SIGNALS")
    evidence_text = _extract_section(memo, "EVIDENCE")

    annual_net = roi_result.get("annual_net_gbp", 0.0)

    return {
        "format_valid": format_valid,
        "validation_attempts": validation_attempts,
        "citation_count": len(_quoted_re.findall(evidence_text)),
        "push_signal_count": len(_bullet_re.findall(push_text)),
        "pull_signal_count": len(_bullet_re.findall(pull_text)),
        "roi_threshold_met": bool(roi_result.get("roi_threshold_met", False)),
        "roi_margin_gbp": round(annual_net - 1200.0, 2),
    }


def log_live_run(
    category: str,
    competitor: str,
    verdict: str,
    quality: dict,
    log_path: Path | None = None,
) -> None:
    """Append a live_run record to the drift log."""
    log_path = log_path or _DRIFT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "live_run",
        "category": category,
        "competitor": competitor,
        "verdict": verdict,
        **quality,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_accuracy_due(
    log_path: Path | None = None,
    interval: int = ACCURACY_CHECK_INTERVAL,
) -> bool:
    """Return True if an accuracy check is due (every N live runs since last check)."""
    log_path = log_path or _DRIFT_LOG
    if not log_path.exists():
        return False
    records = load_records(log_path)
    last_check = max(
        (i for i, r in enumerate(records) if r.get("type") == "accuracy_check"),
        default=-1,
    )
    live_since = sum(1 for r in records[last_check + 1:] if r.get("type") == "live_run")
    return live_since >= interval


def load_records(log_path: Path | None = None) -> list[dict]:
    """Load all records from the drift log. Returns [] if file does not exist."""
    log_path = log_path or _DRIFT_LOG
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
