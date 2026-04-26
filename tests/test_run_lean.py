import os
import pytest
from pathlib import Path

import torch

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


def test_run_lean_compliance_block_included_in_user_message():
    """Compliance profile is passed as raw data in the user message for model reasoning."""
    os.environ["AGENT_DRY_RUN"] = "true"
    from agent.context_loader import load_context
    from agent.model_runner import _build_lean_user, _format_compliance_block
    from agent.roi_calculator import calculate_roi, extract_pass1_vars
    from agent.signal_interpreter import parse_signal_payload

    context = load_context("crm", "velocitycrm", _DATA_ROOT)
    context["competitor_data"]["compliance"]["soc2_type2"] = False
    signal = parse_signal_payload("{}")
    roi = calculate_roi(extract_pass1_vars(context, signal))

    block = _format_compliance_block(context, signal)
    assert "SOC2=No" in block

    user_msg = _build_lean_user(context, roi, signal)
    assert "Competitor compliance:" in user_msg

    del os.environ["AGENT_DRY_RUN"]


class _FakeTokenizer:
    def __init__(self):
        self._encodings = {
            "SWITCH": [14],
            "STAY": [16],
            "HOLD": [13],
            " SWITCH": [15],
            " STAY": [17],
            " HOLD": [12],
        }
        self._pieces = {
            10: "VERDICT",
            11: ":",
            12: " HOLD",
            13: "HOLD",
            14: "SWITCH",
            15: " SWITCH",
            16: "STAY",
            17: " STAY",
        }

    def encode(self, text, add_special_tokens=False):
        return self._encodings[text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self._pieces[token_id] for token_id in token_ids)


def test_extract_verdict_confidence_uses_softmax_renormalized():
    from agent.model_runner import _extract_verdict_confidence

    tokenizer = _FakeTokenizer()
    generated_ids = torch.tensor([10, 11, 12])  # decodes to "VERDICT: HOLD"

    vocab_size = 32
    scores = [torch.zeros((1, vocab_size)) for _ in range(3)]
    scores[2][0, 12] = 4.0   # " HOLD" token — should dominate
    scores[2][0, 15] = 1.5   # " SWITCH"
    scores[2][0, 17] = 0.5   # " STAY"

    # New API: no transition_scores argument
    confidence = _extract_verdict_confidence(tokenizer, scores, generated_ids)

    assert confidence is not None
    # After renorm across 3 verdict tokens: HOLD ≈ 0.899, SWITCH ≈ 0.074, STAY ≈ 0.027
    assert confidence["verdict_token_prob"] == pytest.approx(0.899, abs=5e-3)
    assert confidence["verdict_probs"]["HOLD"] == pytest.approx(0.899, abs=5e-3)
    assert confidence["verdict_probs"]["SWITCH"] == pytest.approx(0.074, abs=5e-3)
    assert confidence["verdict_probs"]["STAY"] == pytest.approx(0.027, abs=5e-3)
    # Renormalized probs must sum to 1
    total = sum(confidence["verdict_probs"].values())
    assert total == pytest.approx(1.0, abs=1e-5)
