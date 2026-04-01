from agent.roi_calculator import extract_pass1_vars
from agent.signal_interpreter import infer_competitor_monthly_cost, parse_signal_payload


def _context() -> dict:
    return {
        "current_stack_entry": {"monthly_cost_gbp": 510},
        "competitor_data": {"monthly_cost_gbp": 390},
    }


def test_parse_signal_payload_returns_none_for_non_json():
    assert parse_signal_payload("market update: plain text") is None


def test_infer_competitor_monthly_cost_from_from_to_price():
    signal = {"pricing_delta": "from £390 to £345"}
    assert infer_competitor_monthly_cost(_context(), signal) == 345.0


def test_infer_competitor_monthly_cost_from_competitor_vs_current():
    signal = {"pricing_delta": "£390 (competitor) vs £510 (NexusCRM)"}
    assert infer_competitor_monthly_cost(_context(), signal) == 390.0


def test_extract_pass1_vars_uses_signal_price_override():
    signal = {"pricing_delta": "reduced by £40"}
    payload = extract_pass1_vars(_context(), signal)
    assert payload["competitor_monthly_cost"] == 350.0
    assert payload["annual_saving"] == 1920.0