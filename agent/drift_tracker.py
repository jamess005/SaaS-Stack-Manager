"""
Drift tracker — logs real model-internal signals after each agent run.

Appends a JSON record to outputs/drift_log.jsonl after each verdict.
Three record types:
  "live_run"       — confidence signals from the verdict token's probability distribution
  "accuracy_check" — canary eval results (written by scripts/drift_check.py)
  "human_feedback" — manual correctness labels (written by scripts/feedback.py)

Confidence signals per run (from model logits — not text proxies):
  format_valid         — structural completeness (all required sections present)
  validation_attempts  — 1 = passed first time, 2 = needed a retry
  verdict_token_prob   — P(chosen verdict token) at that position
  verdict_entropy_bits — Shannon entropy over SWITCH/STAY/HOLD distribution
  verdict_margin       — gap between top-1 and top-2 verdict probabilities
  verdict_probs        — full {SWITCH, STAY, HOLD} probability dict
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DRIFT_LOG = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"

# Advisory: how many live runs between accuracy checks
ACCURACY_CHECK_INTERVAL = 10


def log_live_run(
    category: str,
    competitor: str,
    verdict: str,
    format_valid: bool,
    validation_attempts: int,
    confidence: dict | None,
    log_path: Path | None = None,
) -> None:
    """Append a live_run record to the drift log."""
    log_path = log_path or _DRIFT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "live_run",
        "category": category,
        "competitor": competitor,
        "verdict": verdict,
        "format_valid": format_valid,
        "validation_attempts": validation_attempts,
    }
    if confidence:
        record.update(confidence)
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
