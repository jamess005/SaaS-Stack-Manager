"""Unit tests for output_validator — pure string operations, no file I/O."""

import pytest
from agent.output_validator import extract_hold_metadata, extract_verdict_class, validate_verdict

# ── Shared memo fixtures ───────────────────────────────────────────────────────

_SWITCH_MEMO = """\
CATEGORY: Finance
CURRENT TOOL: VaultLedger (£420/mo)
COMPETITOR: LedgerFlow (£380/mo)
DATE: 2024-11-20

PUSH SIGNALS:
  - No multi-currency support [HIGH — blocking EU billing]
  - IFRS 15 workaround in use [HIGH — compliance risk]

PULL SIGNALS:
  - "multi-currency invoicing (40+ currencies including EUR native)" [HIGH — resolves #1 push]
  - "IFRS 15 revenue recognition module" [HIGH — resolves #2 push]

FINANCIAL ANALYSIS:
  Migration cost: 15hrs × £48 = £720 one-time
  Annual saving: £480
  Annual net: £240
  ROI threshold met: YES (direct + operational combined)

VERDICT: SWITCH

EVIDENCE:
  "multi-currency invoicing (40+ currencies including EUR native)"
  "IFRS 15 revenue recognition module"
"""

_STAY_MEMO = """\
CATEGORY: CRM
CURRENT TOOL: NexusCRM (£510/mo)
COMPETITOR: VelocityCRM (£389/mo)
DATE: 2024-11-10

PUSH SIGNALS:
  - Mobile app crashes on iOS 17+ [MEDIUM — reported issues]

PULL SIGNALS:
  - "SSO and audit logging are available on Advanced tier at £28/seat" [HARD BLOCK — Core tier lacks SSO]

FINANCIAL ANALYSIS:
  Migration cost: 8hrs × £48 = £384 one-time
  Annual saving: £1,452
  Annual net: £1,324
  ROI threshold met: NO (compliance blocks supersede)

VERDICT: STAY

EVIDENCE:
  "SSO and audit logging are available on Advanced tier at £28/seat"
"""

_HOLD_MEMO = """\
CATEGORY: HR
CURRENT TOOL: PeoplePulse (£385/mo)
COMPETITOR: WorkForge (£420/mo)
DATE: 2024-11-12

PUSH SIGNALS:
  - GDPR erasure must be handled manually [HIGH — compliance risk]

PULL SIGNALS:
  - "automated GDPR right-to-erasure processing" [HIGH — resolves compliance gap]

FINANCIAL ANALYSIS:
  Migration cost: 12hrs × £48 = £576 one-time
  Annual saving: -£420
  Annual net: -£612
  ROI threshold met: NO (net negative)

VERDICT: HOLD

REASSESS CONDITION: PeoplePulse contract expires November 2025.
REVIEW BY: 2025-10-01

EVIDENCE:
  "automated GDPR right-to-erasure processing"
"""


# ── validate_verdict — valid memos ────────────────────────────────────────────


def test_valid_switch_memo():
    valid, errors = validate_verdict(_SWITCH_MEMO)
    assert valid is True
    assert errors == []


def test_valid_stay_memo():
    valid, errors = validate_verdict(_STAY_MEMO)
    assert valid is True
    assert errors == []


def test_valid_hold_memo():
    valid, errors = validate_verdict(_HOLD_MEMO)
    assert valid is True
    assert errors == []


# ── validate_verdict — missing section headers ─────────────────────────────────


@pytest.mark.parametrize(
    "section",
    ["CATEGORY", "CURRENT TOOL", "COMPETITOR", "DATE", "PUSH SIGNALS", "PULL SIGNALS", "VERDICT", "EVIDENCE"],
)
def test_missing_section_header(section):
    # Remove the offending line from the SWITCH memo
    lines = [ln for ln in _SWITCH_MEMO.splitlines() if not ln.strip().startswith(section + ":")]
    broken = "\n".join(lines)
    valid, errors = validate_verdict(broken)
    assert valid is False
    assert any(section in e for e in errors), f"Expected error mentioning {section!r}, got: {errors}"


def test_missing_financial_analysis_section():
    lines = [ln for ln in _SWITCH_MEMO.splitlines() if "FINANCIAL ANALYSIS" not in ln]
    broken = "\n".join(lines)
    valid, errors = validate_verdict(broken)
    assert valid is False
    assert any("FINANCIAL ANALYSIS" in e for e in errors)


# ── validate_verdict — bad verdict value ──────────────────────────────────────


def test_invalid_verdict_value():
    memo = _SWITCH_MEMO.replace("VERDICT: SWITCH", "VERDICT: MAYBE")
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("MAYBE" in e or "Invalid verdict" in e for e in errors)


# ── validate_verdict — no quoted citations ────────────────────────────────────


def test_no_quoted_citations_in_evidence():
    memo = _SWITCH_MEMO.replace(
        '"multi-currency invoicing (40+ currencies including EUR native)"',
        "multi-currency invoicing",
    ).replace('"IFRS 15 revenue recognition module"', "IFRS 15 revenue recognition module")
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("citation" in e.lower() or "quote" in e.lower() for e in errors)


# ── validate_verdict — empty signal sections ──────────────────────────────────


def test_empty_push_signals():
    memo = _SWITCH_MEMO.replace(
        "  - No multi-currency support [HIGH — blocking EU billing]\n  - IFRS 15 workaround in use [HIGH — compliance risk]",
        "",
    )
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("PUSH" in e for e in errors)


def test_empty_pull_signals():
    memo = _SWITCH_MEMO.replace(
        '  - "multi-currency invoicing (40+ currencies including EUR native)" [HIGH — resolves #1 push]\n  - "IFRS 15 revenue recognition module" [HIGH — resolves #2 push]',
        "",
    )
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("PULL" in e for e in errors)


# ── validate_verdict — missing ROI threshold line ─────────────────────────────


def test_roi_threshold_line_required():
    memo = _SWITCH_MEMO.replace("ROI threshold met: YES (direct + operational combined)", "")
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("ROI threshold" in e for e in errors)


# ── validate_verdict — HOLD without reassess condition ───────────────────────


def test_hold_without_reassess_condition():
    memo = _HOLD_MEMO.replace(
        "REASSESS CONDITION: PeoplePulse contract expires November 2025.\nREVIEW BY: 2025-10-01",
        "",
    )
    valid, errors = validate_verdict(memo)
    assert valid is False
    assert any("REASSESS" in e.upper() for e in errors)


# ── extract_verdict_class ─────────────────────────────────────────────────────


def test_extract_verdict_class_switch():
    assert extract_verdict_class(_SWITCH_MEMO) == "SWITCH"


def test_extract_verdict_class_stay():
    assert extract_verdict_class(_STAY_MEMO) == "STAY"


def test_extract_verdict_class_hold():
    assert extract_verdict_class(_HOLD_MEMO) == "HOLD"


def test_extract_verdict_class_none_for_garbage():
    assert extract_verdict_class("this is not a memo") is None


def test_extract_verdict_class_invalid_value():
    assert extract_verdict_class("VERDICT: UNKNOWN") is None


# ── extract_hold_metadata ─────────────────────────────────────────────────────


def test_extract_hold_metadata_complete():
    result = extract_hold_metadata(_HOLD_MEMO)
    assert result is not None
    assert result["category"] == "hr"
    assert result["current_tool"] == "PeoplePulse"
    assert result["competitor"] == "WorkForge"
    assert "November 2025" in result["reassess_condition"]
    assert result["review_by"] == "2025-10-01"
    assert "hold_reason" in result


def test_extract_hold_metadata_returns_none_for_switch():
    assert extract_hold_metadata(_SWITCH_MEMO) is None


def test_extract_hold_metadata_returns_none_for_stay():
    assert extract_hold_metadata(_STAY_MEMO) is None


def test_extract_hold_metadata_partial_returns_none():
    # Missing REVIEW BY line
    memo = _HOLD_MEMO.replace("REVIEW BY: 2025-10-01", "")
    result = extract_hold_metadata(memo)
    assert result is None
