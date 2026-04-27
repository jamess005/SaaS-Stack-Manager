import json
from pathlib import Path

import scripts.evaluate_model as evaluate_model
from scripts.evaluate_model import _collect_samples, _load_signal_case, _persist_dashboard_run


def _write_signal(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_signal_case_prefers_explicit_metadata(tmp_path: Path):
    signal_path = _write_signal(
        tmp_path,
        "heldout_case.json",
        {
            "category": "analytics",
            "competitor_slug": "datalens",
            "scenario": "embedded_reporting_rollout",
            "expected_verdict": "switch",
            "signal": {
                "competitor": "DataLens",
                "competitor_changes": ["Embedded client reporting now ships on all paid plans"],
                "current_tool_status": ["InsightDeck still requires manual PDF exports for client reports"],
                "notes": ["Migration requires a one-off connector remap"],
                "compliance_changes": "No compliance change",
            },
        },
    )

    case = _load_signal_case(signal_path)

    assert case["category"] == "analytics"
    assert case["competitor"] == "datalens"
    assert case["scenario"] == "embedded_reporting_rollout"
    assert case["expected"] == "SWITCH"


def test_load_signal_case_falls_back_to_generated_filename(tmp_path: Path):
    signal_path = _write_signal(
        tmp_path,
        "finance_ledgerflow_pull_dominant.json",
        {
            "competitor": "LedgerFlow",
            "competitor_changes": ["Barclays bank feeds now sync every 15 minutes"],
            "current_tool_status": ["VaultLedger still syncs Barclays overnight"],
            "notes": ["Rollout is GA for all UK customers"],
            "compliance_changes": "No compliance change",
        },
    )

    case = _load_signal_case(signal_path)

    assert case["category"] == "finance"
    assert case["competitor"] == "ledgerflow"
    assert case["scenario"] == "pull_dominant"
    assert case["expected"] == "SWITCH"


def test_collect_samples_all_files_mode_returns_all_matching_cases(tmp_path: Path):
    _write_signal(
        tmp_path,
        "case_one.json",
        {
            "category": "crm",
            "competitor_slug": "closerhub",
            "scenario": "pipeline_cleanup",
            "expected_verdict": "SWITCH",
            "signal": {
                "competitor_changes": ["Sequence builder expanded to unlimited steps"],
                "current_tool_status": ["NexusCRM remains capped at 5 steps"],
                "notes": ["Migration team available next month"],
                "compliance_changes": "No compliance change",
            },
        },
    )
    _write_signal(
        tmp_path,
        "case_two.json",
        {
            "category": "crm",
            "competitor_slug": "dealstream",
            "scenario": "pipeline_cleanup",
            "expected_verdict": "STAY",
            "signal": {
                "competitor_changes": ["Minor dashboard polish"],
                "current_tool_status": ["Support response remains under 2 hours on NexusCRM"],
                "notes": ["No material workflow change"],
                "compliance_changes": "No compliance change",
            },
        },
    )
    _write_signal(tmp_path, "broken.json", ["not", "a", "dict"])

    samples = _collect_samples("pipeline_cleanup", 1, tmp_path, all_files=True)

    assert [path.name for path in samples] == ["case_one.json", "case_two.json"]


def test_persist_dashboard_run_writes_memo_and_drift_record(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    log_path = tmp_path / "drift_log.jsonl"

    memo_filename = _persist_dashboard_run(
        {
            "category": "analytics",
            "competitor": "datalens",
            "scenario": "embedded_reporting_cutover",
            "actual": "SWITCH",
            "format_valid": True,
            "validation_attempts": 2,
            "confidence": {"verdict_token_prob": 0.73},
            "memo_text": "VERDICT: SWITCH\n",
        },
        outputs_dir=outputs_dir,
        log_path=log_path,
        run_date="2026-04-25",
    )

    assert memo_filename == "2026-04-25-analytics-datalens-embedded_reporting_cutover.md"
    assert (outputs_dir / memo_filename).read_text(encoding="utf-8") == "VERDICT: SWITCH\n"

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["type"] == "live_run"
    assert record["memo_filename"] == memo_filename
    assert record["validation_attempts"] == 2
    assert record["verdict_token_prob"] == 0.73


def test_evaluate_one_uses_voting_pipeline_when_requested(tmp_path: Path, monkeypatch):
    signal_path = _write_signal(
        tmp_path,
        "heldout_case.json",
        {
            "category": "crm",
            "competitor_slug": "velocitycrm",
            "scenario": "security_block",
            "expected_verdict": "STAY",
            "signal": {
                "competitor_changes": ["SSO rollout slipped to next year"],
                "current_tool_status": ["Current stack still meets security needs"],
                "notes": [],
                "compliance_changes": "No compliance change",
            },
        },
    )

    monkeypatch.setattr(evaluate_model, "load_context", lambda *args, **kwargs: {"category": "crm"})
    monkeypatch.setattr(evaluate_model, "extract_pass1_vars", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluate_model, "calculate_roi", lambda *args, **kwargs: {"annual_net_gbp": 0})
    monkeypatch.setattr(evaluate_model, "parse_signal_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluate_model, "run_lean", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_lean should not be used")))
    monkeypatch.setattr(
        evaluate_model,
        "run_voting",
        lambda *args, **kwargs: (
            "EVIDENCE:\n\"quoted\"\nPUSH SIGNALS:\n  - none\nPULL SIGNALS:\n  - none\nANALYSIS:\nStable\nVERDICT: STAY\n",
            {"verdict_token_prob": 0.61},
        ),
    )
    monkeypatch.setattr(evaluate_model, "validate_verdict", lambda memo: (True, []))
    monkeypatch.setattr(evaluate_model, "extract_verdict_class", lambda memo: "STAY")

    result = evaluate_model._evaluate_one(signal_path, tokenizer=None, model=None, pipeline="voting")

    assert result["status"] == "ok"
    assert result["actual"] == "STAY"
    assert result["validation_attempts"] == 1
    assert result["confidence"]["verdict_token_prob"] == 0.61