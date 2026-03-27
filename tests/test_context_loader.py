"""Unit tests for context_loader — reads from data/ at project root."""

import pytest
from pathlib import Path
from agent.context_loader import VALID_CATEGORIES, load_context, parse_inbox_filename

# data/ lives at project root, which is two levels above this test file
_DATA_ROOT = Path(__file__).parent.parent / "data"


# ── parse_inbox_filename ──────────────────────────────────────────────────────


def test_parse_inbox_filename_finance():
    cat, slug = parse_inbox_filename("finance_ledgerflow.md")
    assert cat == "finance"
    assert slug == "ledgerflow"


def test_parse_inbox_filename_project_mgmt():
    # category name contains an underscore — greedy prefix match must handle this
    cat, slug = parse_inbox_filename("project_mgmt_flowboard.md")
    assert cat == "project_mgmt"
    assert slug == "flowboard"


def test_parse_inbox_filename_crm():
    cat, slug = parse_inbox_filename("crm_pipelineiq.md")
    assert cat == "crm"
    assert slug == "pipelineiq"


def test_parse_inbox_filename_hr():
    cat, slug = parse_inbox_filename("hr_workforge.md")
    assert cat == "hr"
    assert slug == "workforge"


def test_parse_inbox_filename_analytics():
    cat, slug = parse_inbox_filename("analytics_datalens.md")
    assert cat == "analytics"
    assert slug == "datalens"


def test_parse_inbox_filename_full_path():
    cat, slug = parse_inbox_filename("/some/path/market_inbox/finance_novapay.md")
    assert cat == "finance"
    assert slug == "novapay"


def test_parse_inbox_filename_invalid_raises():
    with pytest.raises(ValueError, match="Cannot parse category"):
        parse_inbox_filename("unknown_tool.md")


def test_parse_inbox_filename_no_slug_raises():
    with pytest.raises(ValueError):
        parse_inbox_filename("finance_.md")


# ── load_context — structure checks ──────────────────────────────────────────


def test_load_context_returns_all_keys():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    required_keys = {
        "category",
        "competitor_slug",
        "current_stack_entry",
        "usage_metrics_entry",
        "business_rules_text",
        "global_rules_text",
        "competitor_data",
        "company_profile_text",
    }
    assert required_keys == set(ctx.keys())


def test_load_context_category_and_slug_preserved():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert ctx["category"] == "finance"
    assert ctx["competitor_slug"] == "ledgerflow"


# ── load_context — data correctness ──────────────────────────────────────────


def test_load_context_current_stack_entry_correct():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert ctx["current_stack_entry"]["tool"] == "VaultLedger"
    assert ctx["current_stack_entry"]["monthly_cost_gbp"] == 420


def test_load_context_competitor_data_correct():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert ctx["competitor_data"]["name"] == "LedgerFlow"
    assert ctx["competitor_data"]["category"] == "finance"


def test_load_context_business_rules_non_empty():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert len(ctx["business_rules_text"]) > 50


def test_load_context_global_rules_contains_soc2():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert "SOC2 Type II" in ctx["global_rules_text"]


def test_load_context_global_rules_contains_roi_threshold():
    ctx = load_context("hr", "workforge", _DATA_ROOT)
    assert "1,200" in ctx["global_rules_text"]


def test_load_context_usage_metrics_entry_present():
    ctx = load_context("finance", "ledgerflow", _DATA_ROOT)
    assert ctx["usage_metrics_entry"] is not None
    assert "utilisation_pct" in ctx["usage_metrics_entry"]


def test_load_context_company_profile_non_empty():
    ctx = load_context("crm", "pipelineiq", _DATA_ROOT)
    assert len(ctx["company_profile_text"]) > 50


# ── load_context — error handling ────────────────────────────────────────────


def test_load_context_invalid_competitor_raises():
    with pytest.raises(FileNotFoundError, match="Competitor file not found"):
        load_context("finance", "nonexistent_tool", _DATA_ROOT)


def test_load_context_invalid_category_raises():
    with pytest.raises(ValueError, match="Invalid category"):
        load_context("logistics", "sometool", _DATA_ROOT)


# ── load_context — smoke tests for all categories ────────────────────────────


def test_load_context_crm_velocitycrm():
    ctx = load_context("crm", "velocitycrm", _DATA_ROOT)
    assert ctx["current_stack_entry"]["tool"] == "NexusCRM"


def test_load_context_hr_hrnest():
    # Hard-block competitor file — should load normally (agent interprets values, not loader)
    ctx = load_context("hr", "hrnest", _DATA_ROOT)
    assert ctx["competitor_data"]["name"] is not None


def test_load_context_analytics_datalens():
    ctx = load_context("analytics", "datalens", _DATA_ROOT)
    assert ctx["current_stack_entry"]["tool"] == "InsightDeck"


def test_load_context_project_mgmt_flowboard():
    ctx = load_context("project_mgmt", "flowboard", _DATA_ROOT)
    assert ctx["current_stack_entry"]["tool"] == "TaskBridge"


# ── VALID_CATEGORIES sanity ───────────────────────────────────────────────────


def test_valid_categories_contains_all_five():
    assert VALID_CATEGORIES == {"crm", "hr", "finance", "project_mgmt", "analytics"}
