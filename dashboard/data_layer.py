import json
import re
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DRIFT_LOG = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"
_SUMMARIES = _PROJECT_ROOT / "outputs" / "summaries.json"
_LOCK_FILE = _PROJECT_ROOT / "outputs" / ".model_lock"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_COMPETITORS_DIR = _PROJECT_ROOT / "data" / "competitors"


def _load_drift_records(log_path: Path) -> list[dict]:
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


def _load_summaries(summaries_path: Path) -> dict:
    if not summaries_path.exists():
        return {}
    try:
        return json.loads(summaries_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_stats(
    log_path: Path | None = None,
    competitors_dir: Path | None = None,
) -> dict:
    log_path = log_path if log_path is not None else _DRIFT_LOG
    competitors_dir = competitors_dir if competitors_dir is not None else _COMPETITORS_DIR
    records = _load_drift_records(log_path)
    live_runs = [r for r in records if r.get("type") == "live_run"]

    # Use latest verdict per competitor (not raw counts of all historical runs)
    latest: dict[str, dict] = {}
    for r in live_runs:
        key = r.get("competitor", "")
        if not key:
            continue
        prev = latest.get(key)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            latest[key] = r

    counts: dict[str, int] = {"SWITCH": 0, "HOLD": 0, "STAY": 0}
    for r in latest.values():
        v = r.get("verdict", "")
        if v in counts:
            counts[v] += 1

    total_tracked = 0
    if competitors_dir.exists():
        for cat_dir in competitors_dir.iterdir():
            if cat_dir.is_dir():
                total_tracked += len(list(cat_dir.glob("*.json")))

    return {
        "switch": counts["SWITCH"],
        "hold": counts["HOLD"],
        "stay": counts["STAY"],
        "total_evaluated": sum(counts.values()),
        "total_tracked": total_tracked,
    }


def get_recent_verdicts(
    n: int = 10,
    log_path: Path | None = None,
    outputs_dir: Path | None = None,
    summaries_path: Path | None = None,
) -> list[dict]:
    """Return last N live_run records, most recent first, with summary."""
    log_path = log_path if log_path is not None else _DRIFT_LOG
    outputs_dir = outputs_dir if outputs_dir is not None else _OUTPUTS_DIR
    summaries_path = summaries_path if summaries_path is not None else _SUMMARIES
    records = _load_drift_records(log_path)
    live_runs = [r for r in records if r.get("type") == "live_run"]
    live_runs.sort(key=lambda r: r.get("ts", ""))
    recent = live_runs[-n:]
    summaries = _load_summaries(summaries_path)

    result = []
    for run in reversed(recent):
        ts = run.get("ts", "")[:10]
        category = run.get("category", "")
        competitor = run.get("competitor", "")
        memo_filename = f"{ts}-{category}-{competitor}.md"
        memo_path = outputs_dir / memo_filename

        memo_excerpt = ""
        if memo_path.exists():
            text = memo_path.read_text(encoding="utf-8")
            m = re.search(r"EVIDENCE:\s*(.+?)(?=\n[A-Z][A-Z\s]+:|$)", text, re.DOTALL)
            if m:
                memo_excerpt = m.group(1).strip()[:300]

        result.append({
            **run,
            "memo_filename": memo_filename,
            "memo_excerpt": memo_excerpt,
            "summary": summaries.get(memo_filename),
        })

    return result


def get_competitors(competitors_dir: Path | None = None) -> list[dict]:
    """Return all competitor JSON files as dicts, with slug and category injected."""
    competitors_dir = competitors_dir if competitors_dir is not None else _COMPETITORS_DIR
    result = []
    if not competitors_dir.exists():
        return result
    for cat_dir in sorted(competitors_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for comp_file in sorted(cat_dir.glob("*.json")):
            try:
                data = json.loads(comp_file.read_text(encoding="utf-8"))
                data.setdefault("slug", comp_file.stem)
                data.setdefault("category", cat_dir.name)
                data.setdefault("scraper_url", "")
                result.append(data)
            except (json.JSONDecodeError, OSError):
                pass
    return result


def add_competitor(
    category: str,
    slug: str,
    name: str,
    monthly_cost_gbp: float,
    scraper_url: str,
    competitors_dir: Path | None = None,
) -> dict:
    """Create a new competitor JSON file from template. Returns the created data."""
    competitors_dir = competitors_dir if competitors_dir is not None else _COMPETITORS_DIR
    cat_dir = competitors_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    data = {
        "name": name,
        "based_on": "user-added",
        "category": category,
        "monthly_cost_gbp": monthly_cost_gbp,
        "seat_count_assumed": None,
        "pricing_model": "unknown",
        "cost_per_seat_gbp": None,
        "pricing_tiers": [],
        "compliance": {"sso": None, "soc2_type2": None, "gdpr": None, "data_residency_uk": None},
        "features": [],
        "known_limitations": [],
        "integration_compatibility": [],
        "recent_trajectory": "newly added",
        "last_updated": today,
        "scraper_url": scraper_url,
    }
    out = cat_dir / f"{slug}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    data["slug"] = slug
    return data


def record_feedback(
    memo_filename: str,
    correct: bool,
    stated_verdict: str,
    actual_verdict: str,
    note: str,
    log_path: Path | None = None,
) -> None:
    log_path = log_path if log_path is not None else _DRIFT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "human_feedback",
        "memo_filename": memo_filename,
        "correct": correct,
        "stated_verdict": stated_verdict,
        "actual_verdict": actual_verdict,
    }
    if note:
        record["note"] = note
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_health(log_path: Path | None = None) -> dict:
    log_path = log_path if log_path is not None else _DRIFT_LOG
    records = _load_drift_records(log_path)
    live_runs = [r for r in records if r.get("type") == "live_run"]
    accuracy_checks = [r for r in records if r.get("type") == "accuracy_check"]
    feedback = [r for r in records if r.get("type") == "human_feedback"]

    # Confidence trend: last 20 live runs that have confidence scores
    conf_runs = [
        {"ts": r["ts"][:10], "competitor": r.get("competitor", ""),
         "verdict": r.get("verdict", ""), "prob": r.get("verdict_token_prob"),
         "margin": r.get("verdict_margin"), "entropy": r.get("verdict_entropy_bits")}
        for r in live_runs[-20:]
        if r.get("verdict_token_prob") is not None
    ]

    last_accuracy = accuracy_checks[-1] if accuracy_checks else None

    return {
        "confidence_trend": conf_runs,
        "last_accuracy": last_accuracy,
        "accuracy_history": accuracy_checks[-10:],
        "feedback_log": list(reversed(feedback[-20:])),
    }


def get_rankings(
    log_path: Path | None = None,
    summaries_path: Path | None = None,
) -> list[dict]:
    """Return current SWITCH verdicts ranked by confidence, most recent per competitor."""
    log_path = log_path or _DRIFT_LOG
    summaries_path = summaries_path or _SUMMARIES
    records = _load_drift_records(log_path)
    summaries = _load_summaries(summaries_path)

    # Latest verdict per competitor
    latest: dict[str, dict] = {}
    for r in records:
        if r.get("type") != "live_run":
            continue
        key = r.get("competitor", "")
        if not key:
            continue
        prev = latest.get(key)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            latest[key] = r

    switches = [r for r in latest.values() if r.get("verdict") == "SWITCH"]
    switches.sort(key=lambda r: r.get("verdict_token_prob") or 0, reverse=True)

    result = []
    for rank, r in enumerate(switches, 1):
        ts_date = r.get("ts", "")[:10]
        cat = r.get("category", "")
        comp = r.get("competitor", "")
        memo_fn = f"{ts_date}-{cat}-{comp}.md"
        result.append({
            "rank": rank,
            "category": cat,
            "competitor": comp,
            "confidence": r.get("verdict_token_prob"),
            "margin": r.get("verdict_margin"),
            "ts": r.get("ts", ""),
            "summary": summaries.get(memo_fn),
            "memo_filename": memo_fn,
        })
    return result


def get_review_queue(
    log_path: Path | None = None,
    summaries_path: Path | None = None,
    conf_threshold: float = 0.65,
) -> dict:
    """Return low-confidence live_run records not yet reviewed by a human.

    Returns the latest run per competitor where verdict_token_prob < conf_threshold
    and no human_feedback record exists for that memo filename yet.
    """
    log_path = log_path if log_path is not None else _DRIFT_LOG
    summaries_path = summaries_path if summaries_path is not None else _SUMMARIES
    records = _load_drift_records(log_path)
    live_runs = [r for r in records if r.get("type") == "live_run"]
    summaries = _load_summaries(summaries_path)

    reviewed = {
        r["memo_filename"]
        for r in records
        if r.get("type") == "human_feedback" and r.get("memo_filename")
    }

    # Latest run per competitor
    latest: dict[str, dict] = {}
    for r in live_runs:
        key = r.get("competitor", "")
        if not key:
            continue
        prev = latest.get(key)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            latest[key] = r

    queue = []
    for r in latest.values():
        prob = r.get("verdict_token_prob")
        if prob is None or prob >= conf_threshold:
            continue
        ts = r.get("ts", "")[:10]
        cat = r.get("category", "")
        comp = r.get("competitor", "")
        memo_fn = f"{ts}-{cat}-{comp}.md"
        if memo_fn in reviewed:
            continue
        queue.append({
            "memo_filename": memo_fn,
            "competitor": comp,
            "category": cat,
            "verdict": r.get("verdict", ""),
            "verdict_token_prob": prob,
            "verdict_margin": r.get("verdict_margin"),
            "ts": ts,
            "summary": summaries.get(memo_fn),
        })

    queue.sort(key=lambda x: x["verdict_token_prob"] or 1.0)
    return {"queue": queue, "count": len(queue)}


def get_active_holds(
    log_path: Path | None = None,
    summaries_path: Path | None = None,
    outputs_dir: Path | None = None,
) -> list[dict]:
    """Return latest HOLD verdicts (most recent per competitor)."""
    log_path = log_path or _DRIFT_LOG
    summaries_path = summaries_path or _SUMMARIES
    outputs_dir = outputs_dir or _OUTPUTS_DIR
    records = _load_drift_records(log_path)
    summaries = _load_summaries(summaries_path)

    latest: dict[str, dict] = {}
    for r in records:
        if r.get("type") != "live_run":
            continue
        key = r.get("competitor", "")
        if not key:
            continue
        prev = latest.get(key)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            latest[key] = r

    holds = [r for r in latest.values() if r.get("verdict") == "HOLD"]
    holds.sort(key=lambda r: r.get("ts", ""), reverse=True)

    result = []
    for r in holds:
        ts_date = r.get("ts", "")[:10]
        cat = r.get("category", "")
        comp = r.get("competitor", "")
        memo_fn = f"{ts_date}-{cat}-{comp}.md"

        # Try to extract REASSESS CONDITION from the memo
        reassess = ""
        review_by = ""
        memo_path = outputs_dir / memo_fn
        if memo_path.exists():
            text = memo_path.read_text(encoding="utf-8")
            import re as _re
            m = _re.search(r"REASSESS CONDITION:\s*(.+)", text)
            if m:
                reassess = m.group(1).strip()
            m2 = _re.search(r"REVIEW BY:\s*(\S+)", text)
            if m2:
                review_by = m2.group(1).strip()

        result.append({
            "category": cat,
            "competitor": comp,
            "confidence": r.get("verdict_token_prob"),
            "ts": r.get("ts", ""),
            "memo_filename": memo_fn,
            "summary": summaries.get(memo_fn),
            "reassess_condition": reassess,
            "review_by": review_by,
        })
    return result


def get_all_verdicts(
    log_path: Path | None = None,
    summaries_path: Path | None = None,
) -> list[dict]:
    """Return ALL live_run records, most recent first, with summary."""
    log_path = log_path or _DRIFT_LOG
    summaries_path = summaries_path or _SUMMARIES
    records = _load_drift_records(log_path)
    summaries = _load_summaries(summaries_path)
    live_runs = [r for r in records if r.get("type") == "live_run"]
    live_runs.sort(key=lambda r: r.get("ts", ""), reverse=True)

    result = []
    for run in live_runs:
        ts = run.get("ts", "")[:10]
        category = run.get("category", "")
        competitor = run.get("competitor", "")
        memo_fn = f"{ts}-{category}-{competitor}.md"
        result.append({
            **run,
            "memo_filename": memo_fn,
            "summary": summaries.get(memo_fn),
        })
    return result


def clean_outputs(
    outputs_dir: Path | None = None,
    log_path: Path | None = None,
    summaries_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict:
    """Remove generated memos, reset drift log and summaries. Returns counts."""
    outputs_dir = outputs_dir or _OUTPUTS_DIR
    log_path = log_path or _DRIFT_LOG
    summaries_path = summaries_path or _SUMMARIES
    lock_path = lock_path or _LOCK_FILE

    removed = 0
    for md in outputs_dir.glob("*.md"):
        md.unlink()
        removed += 1
    if log_path.exists():
        log_path.write_text("", encoding="utf-8")
    if summaries_path.exists():
        summaries_path.write_text("{}", encoding="utf-8")
    lock_path.unlink(missing_ok=True)

    return {"memos_removed": removed}


def is_model_busy(lock_path: Path | None = None) -> bool:
    lock_path = lock_path if lock_path is not None else _LOCK_FILE
    return lock_path.exists()


def set_model_busy(task: str, lock_path: Path | None = None) -> None:
    lock_path = lock_path if lock_path is not None else _LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(task, encoding="utf-8")


def clear_model_busy(lock_path: Path | None = None) -> None:
    lock_path = lock_path if lock_path is not None else _LOCK_FILE
    lock_path.unlink(missing_ok=True)


_FEEDBACK_PAIRS = _PROJECT_ROOT / "training" / "feedback_pairs.jsonl"
_DPO_ADAPTER = _PROJECT_ROOT / "training" / "checkpoints_dpo"


def write_training_checkpoint(pairs_trained: int, log_path: Path | None = None) -> None:
    """Append a dpo_checkpoint record so the pending-correction count resets."""
    import json as _json
    from datetime import datetime, timezone
    log_path = log_path or _DRIFT_LOG
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "dpo_checkpoint",
        "pairs_trained": pairs_trained,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")


def get_feedback_queue(log_path: Path | None = None) -> dict:
    """Count actionable corrections awaiting retraining (since the last training run)."""
    log_path = log_path or _DRIFT_LOG
    records = _load_drift_records(log_path)

    # Anchor to the last successful training run — only count NEW corrections since then
    checkpoints = [r for r in records if r.get("type") == "dpo_checkpoint"]
    cutoff_ts = checkpoints[-1].get("ts", "") if checkpoints else ""

    # Human feedback corrections since last training
    human_corrections = [
        r for r in records
        if r.get("type") == "human_feedback"
        and not r.get("correct", True)
        and r.get("stated_verdict", "") != r.get("actual_verdict", "")
        and r.get("ts", "") > cutoff_ts
    ]
    # Deduplicate by memo_filename (latest wins)
    seen: dict[str, dict] = {}
    for fb in human_corrections:
        key = fb["memo_filename"]
        if key not in seen or fb.get("ts", "") > seen[key].get("ts", ""):
            seen[key] = fb
    human_count = len(seen)

    # Canary failures from the most recent accuracy_check that is itself after the cutoff
    canary_count = 0
    accuracy_checks = [
        r for r in records
        if r.get("type") == "accuracy_check" and r.get("ts", "") > cutoff_ts
    ]
    if accuracy_checks:
        last = accuracy_checks[-1]
        canary_count = sum(
            1 for r in last.get("results", [])
            if r.get("status") == "ok" and not r.get("correct", True)
        )

    total = human_count + canary_count
    has_dpo_adapter = _DPO_ADAPTER.exists()
    has_pending_pairs = _FEEDBACK_PAIRS.exists() and _FEEDBACK_PAIRS.stat().st_size > 0

    return {
        "human_corrections": human_count,
        "canary_failures": canary_count,
        "total_corrections": total,
        "ready_to_train": total >= 1,
        "has_pending_pairs": has_pending_pairs,
        "has_dpo_adapter": has_dpo_adapter,
    }
