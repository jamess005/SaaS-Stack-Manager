"""Tests for the feedback harvester and DPO pipeline data layer."""

from collections import Counter
import json
from pathlib import Path

import pytest


# ── Harvester tests ────────────────────────────────────────────────────────────

@pytest.fixture
def drift_log_with_feedback(tmp_path):
    """Drift log containing human feedback corrections and canary failures."""
    log = tmp_path / "drift_log.jsonl"
    records = [
        # Live runs
        {"ts": "2026-04-13T10:00:00+00:00", "type": "live_run", "category": "crm",
         "competitor": "dealstream", "verdict": "SWITCH", "format_valid": True,
         "validation_attempts": 1, "verdict_token_prob": 0.85},
        {"ts": "2026-04-13T10:30:00+00:00", "type": "live_run", "category": "project_mgmt",
         "competitor": "opscanvas", "verdict": "SWITCH", "format_valid": True,
         "validation_attempts": 1, "verdict_token_prob": 0.72},
        # Human feedback: user says SWITCH was wrong, should be STAY
        {"ts": "2026-04-13T14:00:00+00:00", "type": "human_feedback",
         "memo_filename": "2026-04-13-crm-dealstream.md",
         "correct": False, "stated_verdict": "SWITCH", "actual_verdict": "STAY"},
        # Human feedback: correct=true (no pair needed)
        {"ts": "2026-04-13T14:01:00+00:00", "type": "human_feedback",
         "memo_filename": "2026-04-13-project_mgmt-opscanvas.md",
         "correct": True, "stated_verdict": "SWITCH", "actual_verdict": "SWITCH"},
        # Human feedback where stated==actual (UI bug, skip)
        {"ts": "2026-04-13T14:02:00+00:00", "type": "human_feedback",
         "memo_filename": "2026-04-13-project_mgmt-opscanvas.md",
         "correct": False, "stated_verdict": "SWITCH", "actual_verdict": "SWITCH"},
        # Canary accuracy check with one failure
        {"ts": "2026-04-13T15:00:00+00:00", "type": "accuracy_check",
         "correct": 3, "total": 4, "accuracy": 0.75,
         "results": [
             {"status": "ok", "file": "crm_leadsphere_competitor_nearly_ready.json",
              "expected": "HOLD", "actual": "HOLD", "correct": True},
             {"status": "ok", "file": "hr_teamrise_shelfware_case.json",
              "expected": "SWITCH", "actual": "STAY", "correct": False},
             {"status": "ok", "file": "project_mgmt_opscanvas_contract_renewal_hold.json",
              "expected": "HOLD", "actual": "HOLD", "correct": True},
             {"status": "ok", "file": "finance_brightbooks_pull_dominant.json",
              "expected": "SWITCH", "actual": "SWITCH", "correct": True},
         ]},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return log


def test_parse_memo_filename():
    from training.feedback_harvester import _parse_memo_filename
    assert _parse_memo_filename("2026-04-13-crm-dealstream.md") == ("crm", "dealstream")
    assert _parse_memo_filename("2026-04-13-project_mgmt-opscanvas.md") == ("project_mgmt", "opscanvas")
    assert _parse_memo_filename("invalid.md") is None


def test_parse_canary_stem():
    from training.feedback_harvester import _parse_canary_stem
    result = _parse_canary_stem("hr_teamrise_shelfware_case")
    assert result == ("hr", "teamrise", "shelfware_case")
    result2 = _parse_canary_stem("crm_leadsphere_competitor_nearly_ready")
    assert result2 == ("crm", "leadsphere", "competitor_nearly_ready")
    assert _parse_canary_stem("invalid") is None


def test_extract_human_feedback_filters_correctly(drift_log_with_feedback):
    from training.feedback_harvester import _extract_human_feedback_pairs, _load_drift_records
    records = _load_drift_records(drift_log_with_feedback)

    # Mock to avoid needing actual competitor data files
    import training.feedback_harvester as fh
    original_find = fh._find_signal_for_competitor
    original_load = fh.load_context

    # Patch to return dummy data
    fh._find_signal_for_competitor = lambda cat, comp: {
        "competitor": comp,
        "competitor_changes": ["new API integration"],
        "current_tool_status": [],
        "compliance_changes": "",
        "notes": [],
    }

    def mock_context(cat, comp, data_root):
        return {
            "category": cat,
            "competitor_slug": comp,
            "current_stack_entry": {
                "tool": "TestTool",
                "monthly_cost_gbp": 200,
                "known_issues": ["slow performance"],
                "seat_count": 10,
            },
            "competitor_data": {
                "name": comp.title(),
                "monthly_cost_gbp": 180,
                "compliance": {"sso": True, "soc2_type2": True, "gdpr": True, "data_residency_uk": True},
                "features": [],
            },
            "usage_metrics_entry": None,
            "business_rules_text": "Standard rules apply.",
            "global_rules_text": "Global rules.",
        }
    fh.load_context = mock_context

    try:
        pairs = _extract_human_feedback_pairs(records)
        # Should find exactly 1: the CRM/dealstream correction (SWITCH→STAY)
        assert len(pairs) == 1
        assert pairs[0]["category"] == "crm"
        assert pairs[0]["competitor"] == "dealstream"
        assert pairs[0]["wrong_verdict"] == "SWITCH"
        assert pairs[0]["correct_verdict"] == "STAY"
        assert pairs[0]["source"] == "human_feedback"
        # Chosen trace should end with STAY, rejected with SWITCH
        assert "VERDICT: STAY" in pairs[0]["chosen"]
        assert "VERDICT: SWITCH" in pairs[0]["rejected"]
    finally:
        fh._find_signal_for_competitor = original_find
        fh.load_context = original_load


def test_extract_canary_pairs(drift_log_with_feedback):
    from training.feedback_harvester import _extract_canary_pairs, _load_drift_records
    records = _load_drift_records(drift_log_with_feedback)

    import training.feedback_harvester as fh
    original_load = fh.load_context
    original_dir = fh._GENERATED_DIR

    # Create a temporary signal file
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        gen_dir = Path(td)
        signal_file = gen_dir / "hr_teamrise_shelfware_case.json"
        signal_data = {
            "competitor": "teamrise",
            "competitor_changes": ["new onboarding module"],
            "current_tool_status": ["inactive seats detected"],
            "compliance_changes": "",
            "notes": [],
        }
        signal_file.write_text(json.dumps(signal_data), encoding="utf-8")
        fh._GENERATED_DIR = gen_dir

        def mock_context(cat, comp, data_root):
            return {
                "category": cat,
                "competitor_slug": comp,
                "current_stack_entry": {
                    "tool": "PeoplePulse",
                    "monthly_cost_gbp": 300,
                    "known_issues": ["outdated performance reviews"],
                    "seat_count": 20,
                },
                "competitor_data": {
                    "name": "TeamRise",
                    "monthly_cost_gbp": 250,
                    "compliance": {"sso": True, "soc2_type2": True, "gdpr": True, "data_residency_uk": True},
                    "features": ["new onboarding module"],
                },
                "usage_metrics_entry": {"shelfware_flag": True, "inactive_seats": 8},
                "business_rules_text": "HR rules.",
                "global_rules_text": "Global rules.",
            }
        fh.load_context = mock_context

        try:
            pairs = _extract_canary_pairs(records)
            # Should find exactly 1: hr_teamrise STAY→SWITCH
            assert len(pairs) == 1
            assert pairs[0]["category"] == "hr"
            assert pairs[0]["competitor"] == "teamrise"
            assert pairs[0]["wrong_verdict"] == "STAY"
            assert pairs[0]["correct_verdict"] == "SWITCH"
            assert pairs[0]["source"] == "canary"
            assert "VERDICT: SWITCH" in pairs[0]["chosen"]
            assert "VERDICT: STAY" in pairs[0]["rejected"]
        finally:
            fh.load_context = original_load
            fh._GENERATED_DIR = original_dir


def test_harvest_excludes_eval_canaries_by_default(tmp_path, monkeypatch):
    import training.feedback_harvester as fh

    monkeypatch.setattr(fh, "_load_drift_records", lambda log_path=None: [{"type": "stub"}])
    monkeypatch.setattr(fh, "_extract_human_feedback_pairs", lambda records: [{"source": "human_feedback"}])
    monkeypatch.setattr(fh, "_extract_canary_pairs", lambda records: [{"source": "canary"}])
    monkeypatch.setattr(fh, "_extract_golden_canary_pairs", lambda: [{"source": "golden_canary"}])
    monkeypatch.setattr(fh, "_sample_sft_traces", lambda: [{"source": "sft_sample"}])

    pairs = fh.harvest(
        log_path=tmp_path / "drift_log.jsonl",
        output_path=tmp_path / "feedback_pairs.jsonl",
        dry_run=True,
    )

    assert [pair["source"] for pair in pairs] == ["human_feedback", "sft_sample"]


def test_harvest_can_opt_into_eval_canaries(tmp_path, monkeypatch):
    import training.feedback_harvester as fh

    monkeypatch.setattr(fh, "_load_drift_records", lambda log_path=None: [{"type": "stub"}])
    monkeypatch.setattr(fh, "_extract_human_feedback_pairs", lambda records: [{"source": "human_feedback"}])
    monkeypatch.setattr(fh, "_extract_canary_pairs", lambda records: [{"source": "canary"}])
    monkeypatch.setattr(fh, "_extract_golden_canary_pairs", lambda: [{"source": "golden_canary"}])
    monkeypatch.setattr(fh, "_sample_sft_traces", lambda: [{"source": "sft_sample"}])

    pairs = fh.harvest(
        log_path=tmp_path / "drift_log.jsonl",
        output_path=tmp_path / "feedback_pairs.jsonl",
        dry_run=True,
        include_canary_failures=True,
        include_golden_canaries=True,
    )

    assert Counter(pair["source"] for pair in pairs) == Counter({
        "human_feedback": 1,
        "canary": 1,
        "golden_canary": 1,
        "sft_sample": 1,
    })


def test_build_trace_has_required_sections():
    from training.feedback_harvester import _build_trace

    context = {
        "category": "finance",
        "competitor_slug": "ledgerflow",
        "current_stack_entry": {
            "tool": "QuickLedger",
            "monthly_cost_gbp": 200,
            "known_issues": ["no multi-currency"],
            "seat_count": 10,
        },
        "competitor_data": {"name": "LedgerFlow", "monthly_cost_gbp": 180},
        "usage_metrics_entry": None,
    }
    signal = {
        "competitor": "ledgerflow",
        "competitor_changes": ["native multi-currency support"],
        "current_tool_status": [],
        "compliance_changes": "",
        "notes": [],
    }
    roi = {
        "migration_cost_one_time": 720,
        "annual_net_gbp": 1500,
        "roi_threshold_met": True,
    }

    trace = _build_trace(context, signal, roi, "SWITCH")
    assert "PUSH SIGNALS:" in trace
    assert "PULL SIGNALS:" in trace
    assert "COMPLIANCE: PASSED" in trace
    assert "ROI:" in trace
    assert "HOLD CONDITION:" in trace
    assert "ANALYSIS:" in trace
    assert "VERDICT: SWITCH" in trace


# ── Data layer feedback queue tests ────────────────────────────────────────────


def test_get_feedback_queue_counts(drift_log_with_feedback):
    from dashboard.data_layer import get_feedback_queue
    q = get_feedback_queue(log_path=drift_log_with_feedback)
    # 1 human correction (dealstream SWITCH→STAY, the opscanvas one has stated==actual)
    assert q["human_corrections"] == 1
    # 1 canary failure (hr_teamrise)
    assert q["canary_failures"] == 1
    assert q["total_corrections"] == 2
    assert q["ready_to_train"] is True


def test_get_feedback_queue_no_corrections(tmp_path):
    from dashboard.data_layer import get_feedback_queue
    log = tmp_path / "drift_log.jsonl"
    records = [
        {"ts": "2026-04-13T10:00:00+00:00", "type": "live_run", "category": "finance",
         "competitor": "ledgerflow", "verdict": "STAY"},
        {"ts": "2026-04-13T14:00:00+00:00", "type": "human_feedback",
         "memo_filename": "2026-04-13-finance-ledgerflow.md",
         "correct": True, "stated_verdict": "STAY", "actual_verdict": "STAY"},
        {"ts": "2026-04-13T15:00:00+00:00", "type": "accuracy_check",
         "correct": 4, "total": 4, "accuracy": 1.0,
         "results": [
             {"status": "ok", "file": "test.json", "expected": "STAY", "actual": "STAY", "correct": True},
         ]},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    q = get_feedback_queue(log_path=log)
    assert q["human_corrections"] == 0
    assert q["canary_failures"] == 0
    assert q["total_corrections"] == 0
    assert q["ready_to_train"] is False


def test_get_feedback_queue_empty_log(tmp_path):
    from dashboard.data_layer import get_feedback_queue
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    q = get_feedback_queue(log_path=log)
    assert q["total_corrections"] == 0


# ── API route tests ────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, drift_log_with_feedback):
    """Flask test client with isolated paths."""
    import dashboard.data_layer as dl
    from dashboard import create_app

    old_drift = dl._DRIFT_LOG
    old_lock = dl._LOCK_FILE
    dl._DRIFT_LOG = drift_log_with_feedback
    dl._LOCK_FILE = tmp_path / ".model_lock"

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

    dl._DRIFT_LOG = old_drift
    dl._LOCK_FILE = old_lock


def test_api_feedback_queue(client):
    r = client.get("/api/feedback-queue")
    assert r.status_code == 200
    data = r.get_json()
    assert "human_corrections" in data
    assert "canary_failures" in data
    assert "total_corrections" in data


def test_api_retrain_when_busy(client, tmp_path):
    import dashboard.data_layer as dl
    dl.set_model_busy("test", lock_path=dl._LOCK_FILE)
    r = client.post("/api/retrain")
    assert r.status_code == 409
    dl.clear_model_busy(lock_path=dl._LOCK_FILE)


def test_api_run_canary_when_busy(client, tmp_path):
    import dashboard.data_layer as dl
    dl.set_model_busy("test", lock_path=dl._LOCK_FILE)
    r = client.post("/api/run-canary")
    assert r.status_code == 409
    dl.clear_model_busy(lock_path=dl._LOCK_FILE)
