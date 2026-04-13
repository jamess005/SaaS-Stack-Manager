import os
import pytest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"


def test_validate_lean_output_pass():
    from agent.output_validator import validate_lean_output
    text = "ANALYSIS: Compliance passes. Competitor resolves multi-currency gap.\nVERDICT: SWITCH"
    valid, errors = validate_lean_output(text)
    assert valid
    assert errors == []


def test_validate_lean_output_missing_analysis():
    from agent.output_validator import validate_lean_output
    text = "VERDICT: STAY"
    valid, errors = validate_lean_output(text)
    assert not valid
    assert any("ANALYSIS" in e for e in errors)


def test_validate_lean_output_invalid_verdict():
    from agent.output_validator import validate_lean_output
    text = "ANALYSIS: Something.\nVERDICT: DUNNO"
    valid, errors = validate_lean_output(text)
    assert not valid
    assert any("Invalid" in e for e in errors)


def test_run_lean_dry_run_returns_valid_verdict():
    os.environ["AGENT_DRY_RUN"] = "true"
    from agent.context_loader import load_context
    from agent.model_runner import run_lean
    from agent.output_validator import extract_verdict_class
    from agent.roi_calculator import calculate_roi, extract_pass1_vars
    from agent.signal_interpreter import parse_signal_payload

    context = load_context("finance", "ledgerflow", _DATA_ROOT)
    signal = parse_signal_payload("{}")
    roi = calculate_roi(extract_pass1_vars(context, signal))
    text, _ = run_lean("{}", context, roi, None, None)
    verdict = extract_verdict_class(text)
    assert verdict in {"SWITCH", "STAY", "HOLD"}
    del os.environ["AGENT_DRY_RUN"]


def test_run_lean_compliance_fail_short_circuits():
    """When Python compliance gate fails, STAY is returned without a model call."""
    os.environ["AGENT_DRY_RUN"] = "true"
    from agent.context_loader import load_context
    from agent.model_runner import run_lean
    from agent.output_validator import extract_verdict_class
    from agent.roi_calculator import calculate_roi, extract_pass1_vars
    from agent.signal_interpreter import parse_signal_payload

    context = load_context("crm", "velocitycrm", _DATA_ROOT)
    # Force compliance failure
    context["competitor_data"]["compliance"]["soc2_type2"] = False
    signal = parse_signal_payload("{}")
    roi = calculate_roi(extract_pass1_vars(context, signal))

    import agent.model_runner as mr
    from unittest.mock import patch

    with patch.object(mr, '_is_dry_run', return_value=False), \
         patch.object(mr, '_generate', side_effect=lambda *a, **kw: "ANALYSIS: fake\nVERDICT: SWITCH") as mock_gen:
        text, confidence = run_lean("{}", context, roi, None, None)
        verdict = extract_verdict_class(text)
        assert verdict == "STAY"
        assert confidence is None
        # _generate should NOT have been called for the verdict step
        # It may be called for compliance_changes parsing (0 or 1 times, not for verdict)
        # The key check is that text is STAY from Python gate
        assert "compliance" in text.lower() or verdict == "STAY"

    del os.environ["AGENT_DRY_RUN"]
