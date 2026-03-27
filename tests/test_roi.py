"""Unit tests for roi_calculator — runs in CI without GPU."""

import pytest
from agent.roi_calculator import calculate_roi, extract_pass1_vars


# ── extract_pass1_vars ────────────────────────────────────────────────────────


def test_extract_pass1_vars_returns_correct_values():
    context = {
        "current_stack_entry": {"monthly_cost_gbp": 420},
        "competitor_data": {"monthly_cost_gbp": 380},
    }
    result = extract_pass1_vars(context)
    assert result["current_monthly_cost"] == 420.0
    assert result["competitor_monthly_cost"] == 380.0
    assert result["annual_saving"] == pytest.approx(480.0)
    assert result["staff_hourly_rate"] == 48.0
    assert result["migration_hours"] == 15.0
    assert result["roi_threshold"] == 1200.0


def test_extract_pass1_vars_negative_saving_when_competitor_costs_more():
    context = {
        "current_stack_entry": {"monthly_cost_gbp": 300},
        "competitor_data": {"monthly_cost_gbp": 450},
    }
    result = extract_pass1_vars(context)
    assert result["annual_saving"] == pytest.approx(-1800.0)


# ── calculate_roi ─────────────────────────────────────────────────────────────


def test_switch_direct_saving():
    result = calculate_roi({
        "current_monthly_cost": 420,
        "competitor_monthly_cost": 380,
        "migration_hours": 10,
        "staff_hourly_rate": 48,
        "annual_saving": 480,
        "roi_threshold": 1200,
    })
    assert result["migration_cost_one_time"] == 480.0
    assert result["amortised_migration_cost_per_year"] == 160.0
    assert result["annual_net_gbp"] == 320.0
    assert result["roi_threshold_met"] is False  # 320 < 1200


def test_roi_threshold_met():
    result = calculate_roi({
        "current_monthly_cost": 510,
        "competitor_monthly_cost": 390,
        "migration_hours": 8,
        "staff_hourly_rate": 48,
        "annual_saving": 1440,
        "roi_threshold": 1200,
    })
    assert result["annual_net_gbp"] == pytest.approx(1440 - (384 / 3), rel=1e-3)
    assert result["roi_threshold_met"] is True


def test_zero_saving_short_migration():
    result = calculate_roi({
        "current_monthly_cost": 420,
        "competitor_monthly_cost": 420,
        "migration_hours": 5,
        "staff_hourly_rate": 48,
        "annual_saving": 0,
        "roi_threshold": 1200,
    })
    assert result["migration_cost_one_time"] == 240.0
    assert result["annual_net_gbp"] < 0
    assert result["roi_threshold_met"] is False



def test_high_migration_cost_erodes_saving():
    result = calculate_roi({
        "current_monthly_cost": 510,
        "competitor_monthly_cost": 460,
        "migration_hours": 15,
        "staff_hourly_rate": 65,
        "annual_saving": 600,
        "roi_threshold": 1200,
    })
    assert result["migration_cost_one_time"] == 975.0
    assert result["amortised_migration_cost_per_year"] == 325.0
    assert result["annual_net_gbp"] == pytest.approx(275.0)
    assert result["roi_threshold_met"] is False
