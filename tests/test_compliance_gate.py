# tests/test_compliance_gate.py
import os

import pytest
from agent.model_runner import _compliance_pass_python


def _ctx(category, seats, compliance):
    return {
        "category": category,
        "current_stack_entry": {"seat_count": seats},
        "competitor_data": {"compliance": compliance},
    }


FULL_PASS = {"soc2_type2": True, "sso_saml": True,
             "uk_residency": True, "gdpr_eu_residency": True, "audit_log": True}


def test_all_pass():
    passed, failures = _compliance_pass_python(_ctx("crm", 22, FULL_PASS))
    assert passed
    assert failures == []


def test_soc2_fail():
    c = {**FULL_PASS, "soc2_type2": False}
    passed, failures = _compliance_pass_python(_ctx("analytics", 12, c))
    assert not passed
    assert any("SOC2" in f for f in failures)


def test_sso_fail_high_seats():
    c = {**FULL_PASS, "sso_saml": False}
    passed, failures = _compliance_pass_python(_ctx("crm", 22, c))
    assert not passed
    assert any("SSO" in f for f in failures)


def test_sso_pass_low_seats():
    # SSO not required when seat_count <= 10
    c = {**FULL_PASS, "sso_saml": False}
    passed, failures = _compliance_pass_python(_ctx("analytics", 8, c))
    assert passed


def test_residency_fail_neither():
    c = {**FULL_PASS, "uk_residency": False, "gdpr_eu_residency": False}
    passed, failures = _compliance_pass_python(_ctx("finance", 8, c))
    assert not passed
    assert any("residency" in f.lower() for f in failures)


def test_eu_residency_accepted():
    c = {**FULL_PASS, "uk_residency": False, "gdpr_eu_residency": True}
    passed, failures = _compliance_pass_python(_ctx("finance", 8, c))
    assert passed


def test_audit_log_required_finance():
    c = {**FULL_PASS, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("finance", 8, c))
    assert not passed
    assert any("audit" in f.lower() for f in failures)


def test_audit_log_required_hr():
    c = {**FULL_PASS, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("hr", 8, c))
    assert not passed


def test_audit_log_required_crm():
    c = {**FULL_PASS, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("crm", 8, c))
    assert not passed


def test_audit_log_not_required_analytics():
    c = {**FULL_PASS, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("analytics", 8, c))
    assert passed


def test_audit_log_not_required_project_mgmt():
    c = {**FULL_PASS, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("project_mgmt", 8, c))
    assert passed


def test_multiple_failures_all_reported():
    c = {"soc2_type2": False, "sso_saml": False,
         "uk_residency": False, "gdpr_eu_residency": False, "audit_log": False}
    passed, failures = _compliance_pass_python(_ctx("crm", 22, c))
    assert not passed
    assert len(failures) >= 4  # SOC2, SSO, residency, audit log (all four)


def test_sso_pass_exact_boundary():
    # seat_count == 10 is NOT above the >10 threshold — SSO should not be required
    c = {**FULL_PASS, "sso_saml": False}
    passed, _ = _compliance_pass_python(_ctx("analytics", 10, c))
    assert passed


def test_uk_residency_accepted():
    c = {**FULL_PASS, "gdpr_eu_residency": False, "uk_residency": True}
    passed, _ = _compliance_pass_python(_ctx("finance", 8, c))
    assert passed


def test_parse_compliance_changes_noop_when_already_passing():
    """If all required fields already True, returns unchanged dict."""
    from agent.model_runner import _parse_compliance_changes
    os.environ["AGENT_DRY_RUN"] = "true"
    compliance = {"soc2_type2": True, "sso_saml": True,
                  "uk_residency": True, "audit_log": True}
    result = _parse_compliance_changes("acquired SOC2", compliance, "crm", 22, None, None)
    assert result == compliance
    del os.environ["AGENT_DRY_RUN"]


def test_parse_compliance_changes_noop_empty_text():
    """Empty changes text returns unchanged dict."""
    from agent.model_runner import _parse_compliance_changes
    os.environ["AGENT_DRY_RUN"] = "true"
    compliance = {"soc2_type2": False}
    result = _parse_compliance_changes("", compliance, "analytics", 12, None, None)
    assert result == compliance
    del os.environ["AGENT_DRY_RUN"]
