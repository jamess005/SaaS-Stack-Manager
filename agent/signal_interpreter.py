"""Helpers for interpreting structured market-signal deltas.

These functions are intentionally conservative. If a signal cannot be parsed
reliably, callers should fall back to the baseline structured context.
"""

from __future__ import annotations

import json
import re
from typing import Any


_PRICE_RE = re.compile(r"£\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_COMPETITOR_HINT_RE = re.compile(
    r"competitor|challenger|new tool|alternative|candidate",
    re.IGNORECASE,
)


def parse_signal_payload(inbox_text: str) -> dict[str, Any] | None:
    """Parse a structured signal payload from inbox text when available."""
    try:
        payload = json.loads(inbox_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalise_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    return []


def signal_current_tool_status(signal: dict[str, Any] | None) -> list[str]:
    return _normalise_lines((signal or {}).get("current_tool_status"))


def signal_competitor_changes(signal: dict[str, Any] | None) -> list[str]:
    return _normalise_lines((signal or {}).get("competitor_changes"))


def signal_notes(signal: dict[str, Any] | None) -> list[str]:
    return _normalise_lines((signal or {}).get("notes"))


def signal_compliance_changes(signal: dict[str, Any] | None) -> str:
    if signal is None:
        return ""
    value = signal.get("compliance_changes")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value.strip()
    return ""


def infer_competitor_monthly_cost(context: dict[str, Any], signal: dict[str, Any] | None) -> float:
    """Infer competitor monthly cost from the signal if it explicitly changed."""
    baseline = float(context["competitor_data"]["monthly_cost_gbp"])
    if signal is None:
        return baseline

    pricing_delta = str(signal.get("pricing_delta", "")).strip()
    if not pricing_delta:
        return baseline

    lower = pricing_delta.lower()
    if "unchanged" in lower or lower == "competitive":
        return baseline

    current_cost = float(context["current_stack_entry"]["monthly_cost_gbp"])
    amounts = [float(value) for value in _PRICE_RE.findall(pricing_delta)]

    from_to = re.search(
        r"from\s*£\s*([0-9]+(?:\.[0-9]+)?)\s*to\s*£\s*([0-9]+(?:\.[0-9]+)?)",
        pricing_delta,
        re.IGNORECASE,
    )
    if from_to:
        return float(from_to.group(2))

    explicit_reduce = re.search(
        r"(?:reduced|decreased|down)\s+by\s*£\s*([0-9]+(?:\.[0-9]+)?)",
        pricing_delta,
        re.IGNORECASE,
    )
    if explicit_reduce:
        return max(0.0, baseline - float(explicit_reduce.group(1)))

    explicit_increase = re.search(
        r"(?:increased|increase|up)\s+by\s*£\s*([0-9]+(?:\.[0-9]+)?)",
        pricing_delta,
        re.IGNORECASE,
    )
    if explicit_increase:
        return baseline + float(explicit_increase.group(1))

    if len(amounts) == 1:
        if _COMPETITOR_HINT_RE.search(pricing_delta) or abs(amounts[0] - current_cost) > 0.01:
            return amounts[0]
        return baseline

    if len(amounts) >= 2:
        for amount in amounts:
            if abs(amount - current_cost) < 0.01:
                continue
            return amount

    if "slightly reduced" in lower:
        return max(0.0, baseline - 10.0)
    if "slightly increased" in lower:
        return baseline + 10.0

    return baseline