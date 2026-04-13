"""
Integration tests for the full agent pipeline.

All tests run with AGENT_DRY_RUN=true — no model is loaded.
tmp_path fixtures are used for outputs_dir and register_path to keep test runs isolated.

Fixture files in fixtures/ use human-friendly names (inbox_switch.md etc.).
Integration tests copy them to a tmp_path directory with the {category}_{competitor}.md
naming convention required by parse_inbox_filename.
"""

import json
import os
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

# Force dry-run for the entire test module before any agent imports
os.environ["AGENT_DRY_RUN"] = "true"

from agent.agent import run_agent  # noqa: E402 — must come after env var is set
from agent.output_validator import validate_verdict  # noqa: E402

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _inbox(tmp_path: Path, fixture_name: str, canonical_name: str) -> Path:
    """Copy a fixture inbox file to tmp_path with the {category}_{competitor}.md name."""
    src = _FIXTURES_DIR / fixture_name
    dst = tmp_path / canonical_name
    shutil.copy(src, dst)
    return dst


# ── SWITCH pipeline ───────────────────────────────────────────────────────────


def test_full_pipeline_switch(tmp_path):
    inbox = _inbox(tmp_path, "inbox_switch.md", "finance_ledgerflow.md")
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
    assert result["verdict"] == "SWITCH"
    assert result["hold_registered"] is False
    assert result["output_path"].exists()


def test_switch_output_file_content_passes_validation(tmp_path):
    inbox = _inbox(tmp_path, "inbox_switch.md", "finance_ledgerflow.md")
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
    content = result["output_path"].read_text(encoding="utf-8")
    is_valid, errors = validate_verdict(content)
    assert is_valid is True, f"Validation errors: {errors}"


# ── STAY pipeline ─────────────────────────────────────────────────────────────


def test_full_pipeline_stay(tmp_path):
    inbox = _inbox(tmp_path, "inbox_stay.md", "crm_velocitycrm.md")
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
    assert result["verdict"] == "STAY"
    assert result["hold_registered"] is False


def test_stay_no_hold_register_entry(tmp_path):
    inbox = _inbox(tmp_path, "inbox_stay.md", "crm_velocitycrm.md")
    register_path = tmp_path / "hold.json"
    run_agent(inbox, outputs_dir=tmp_path / "out", register_path=register_path, log_path=tmp_path / "drift.jsonl")
    assert not register_path.exists()


# ── HOLD pipeline ─────────────────────────────────────────────────────────────


def test_full_pipeline_hold(tmp_path):
    inbox = _inbox(tmp_path, "inbox_hold.md", "hr_workforge.md")
    register_path = tmp_path / "hold.json"
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=register_path, log_path=tmp_path / "drift.jsonl")
    assert result["verdict"] == "HOLD"
    assert result["hold_registered"] is True


def test_hold_register_schema(tmp_path):
    inbox = _inbox(tmp_path, "inbox_hold.md", "hr_workforge.md")
    register_path = tmp_path / "hold.json"
    run_agent(inbox, outputs_dir=tmp_path / "out", register_path=register_path, log_path=tmp_path / "drift.jsonl")
    register = json.loads(register_path.read_text(encoding="utf-8"))
    assert len(register) == 1
    entry = register[0]
    required_keys = {"category", "current_tool", "competitor", "hold_reason", "reassess_condition", "issued_date", "review_by"}
    assert required_keys.issubset(entry.keys()), f"Missing keys: {required_keys - set(entry.keys())}"


# ── Output filename format ────────────────────────────────────────────────────


def test_output_file_written_with_correct_name_switch(tmp_path):
    inbox = _inbox(tmp_path, "inbox_switch.md", "finance_ledgerflow.md")
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
    # Pattern: YYYY-MM-DD-finance-ledgerflow.md
    assert re.match(r"\d{4}-\d{2}-\d{2}-finance-ledgerflow\.md", result["output_path"].name)


def test_output_file_written_with_correct_name_hold(tmp_path):
    inbox = _inbox(tmp_path, "inbox_hold.md", "hr_workforge.md")
    result = run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
    assert re.match(r"\d{4}-\d{2}-\d{2}-hr-workforge\.md", result["output_path"].name)


# ── Error handling ────────────────────────────────────────────────────────────


def test_invalid_inbox_filename_raises(tmp_path):
    bad_file = tmp_path / "badname.md"
    bad_file.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse category"):
        run_agent(bad_file, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")


def test_inbox_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_agent(tmp_path / "nonexistent.md", outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")


# ── Dry-run behaviour ─────────────────────────────────────────────────────────


def test_load_model_not_called_in_dry_run(tmp_path):
    inbox = _inbox(tmp_path, "inbox_switch.md", "finance_ledgerflow.md")
    with patch("agent.agent.load_model") as mock_load:
        mock_load.return_value = (None, None)
        run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
        mock_load.assert_called_once()


def test_calculate_roi_called_with_correct_payload(tmp_path):
    inbox = _inbox(tmp_path, "inbox_switch.md", "finance_ledgerflow.md")
    from agent.roi_calculator import calculate_roi as real_roi

    with patch("agent.agent.calculate_roi", wraps=real_roi) as mock_roi:
        run_agent(inbox, outputs_dir=tmp_path / "out", register_path=tmp_path / "hold.json", log_path=tmp_path / "drift.jsonl")
        mock_roi.assert_called_once()
        payload = mock_roi.call_args[0][0]
        required_payload_keys = {
            "current_monthly_cost",
            "competitor_monthly_cost",
            "migration_hours",
            "staff_hourly_rate",
            "annual_saving",
            "roi_threshold",
        }
        assert required_payload_keys.issubset(payload.keys()), f"Missing payload keys: {required_payload_keys - set(payload.keys())}"
        # Verify values come from structured JSON, not model inference
        assert payload["current_monthly_cost"] == 420.0   # VaultLedger from current_stack.json
        assert payload["competitor_monthly_cost"] == 380.0  # LedgerFlow from competitors/finance/ledgerflow.json
        assert payload["staff_hourly_rate"] == 48.0
        assert payload["roi_threshold"] == 1200.0
