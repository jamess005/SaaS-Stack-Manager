"""
ROI Calculator — pure Python, no model involvement.

Extracts financial variables directly from structured context data and computes
the migration ROI. All inputs come from JSON files — no model inference required.
"""

_STAFF_HOURLY_RATE_GBP = 48.0     # global.md: blended staff rate
_ROI_THRESHOLD_GBP = 1200.0       # global.md: minimum annual net to justify switch
_MIGRATION_HOURS_DEFAULT = 15.0   # global.md: maximum migration budget (conservative default)


def extract_pass1_vars(context: dict) -> dict:
    """
    Extract financial variables directly from loaded context.

    Reads monthly costs from structured JSON fields and applies the constants
    defined in global.md. No model inference — all values are deterministic.

    Args:
        context: dict returned by context_loader.load_context()

    Returns:
        {
            "current_monthly_cost": float,
            "competitor_monthly_cost": float,
            "migration_hours": float,
            "staff_hourly_rate": float,
            "annual_saving": float,
            "roi_threshold": float,
        }
    """
    current = float(context["current_stack_entry"]["monthly_cost_gbp"])
    competitor = float(context["competitor_data"]["monthly_cost_gbp"])
    return {
        "current_monthly_cost": current,
        "competitor_monthly_cost": competitor,
        "migration_hours": _MIGRATION_HOURS_DEFAULT,
        "staff_hourly_rate": _STAFF_HOURLY_RATE_GBP,
        "annual_saving": round((current - competitor) * 12, 2),
        "roi_threshold": _ROI_THRESHOLD_GBP,
    }


def calculate_roi(payload: dict) -> dict:
    """
    Calculate migration cost and annualised ROI from a financial payload.

    Args:
        payload: {
            "current_monthly_cost": float,
            "competitor_monthly_cost": float,
            "migration_hours": float,
            "staff_hourly_rate": float,
            "annual_saving": float,
            "roi_threshold": float,
        }

    Returns:
        {
            "migration_cost_one_time": float,
            "annual_direct_saving": float,
            "amortised_migration_cost_per_year": float,
            "annual_net_gbp": float,
            "roi_threshold_gbp": float,
            "roi_threshold_met": bool,
            "note": str
        }
    """
    migration_cost = payload["migration_hours"] * payload["staff_hourly_rate"]
    amortised_migration = migration_cost / 3  # spread over 3 years

    annual_direct_saving = payload["annual_saving"]
    annual_net = annual_direct_saving - amortised_migration
    roi_met = annual_net >= payload["roi_threshold"]

    return {
        "migration_cost_one_time": round(migration_cost, 2),
        "annual_direct_saving": round(annual_direct_saving, 2),
        "amortised_migration_cost_per_year": round(amortised_migration, 2),
        "annual_net_gbp": round(annual_net, 2),
        "roi_threshold_gbp": payload["roi_threshold"],
        "roi_threshold_met": roi_met,
        "note": (
            "Direct cost saving only. Operational gains (e.g. unblocked revenue, "
            "compliance risk removal) are not captured here — flag qualitatively in memo."
        ),
    }
