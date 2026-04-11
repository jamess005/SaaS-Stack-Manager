# tests/test_analysis_templates.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.generate_cot_traces import _analysis

_NSB_SIGNAL = {
    "competitor": "MetricFlux",
    "competitor_changes": ["real-time dashboards with sub-minute refresh rates"],
    "current_tool_status": [],
    "notes": ["feature in preview — GA date TBD"],
    "compliance_changes": "",
}
_NSB_CONTEXT = {
    "category": "analytics",
    "current_stack_entry": {
        "tool": "InsightDeck",
        "known_issues": ["no mobile view"],
    },
}
_NSB_ROI = {"annual_net_gbp": -50.0, "roi_threshold_met": False, "migration_cost_one_time": 500.0}

_CNR_SIGNAL = {
    "competitor": "TeamSync Projects",
    "competitor_changes": ["feature in beta addresses main gap", "mobile app available"],
    "current_tool_status": ["shelfware rate 52/150 seats", "inactive seats 52"],
    "notes": ["migration would mean losing dedicated PM depth"],
    "compliance_changes": "",
}
_CNR_CONTEXT = {
    "category": "project_mgmt",
    "current_stack_entry": {
        "tool": "TaskBridge",
        "known_issues": ["no time-tracking built in"],
    },
}
_CNR_ROI = {"annual_net_gbp": 1500.0, "roi_threshold_met": True, "migration_cost_one_time": 800.0}


def test_nsb_all_variants_reference_hold_signal_none():
    """Every NSB variant must explicitly name 'Hold signal: NONE' so the model learns to trust that field."""
    for v in range(3):
        text = _analysis("negative_signal_buried", _NSB_SIGNAL, _NSB_CONTEXT, _NSB_ROI, variant=v)
        assert "Hold signal: NONE" in text, (
            f"NSB variant {v} does not reference 'Hold signal: NONE'.\nGot: {text}"
        )


def test_nsb_all_variants_reference_notes_position():
    """Every NSB variant must explain the pre-GA signal is in notes/buried signals, not competitor changes."""
    for v in range(3):
        text = _analysis("negative_signal_buried", _NSB_SIGNAL, _NSB_CONTEXT, _NSB_ROI, variant=v)
        assert any(kw in text.lower() for kw in ("notes", "buried")), (
            f"NSB variant {v} does not reference notes/buried position.\nGot: {text}"
        )


def test_cnr_all_variants_reference_hold_signal_field():
    """Every CNR variant must reference the 'Hold signal:' field by name."""
    for v in range(4):
        text = _analysis("competitor_nearly_ready", _CNR_SIGNAL, _CNR_CONTEXT, _CNR_ROI, variant=v)
        assert "Hold signal" in text, (
            f"CNR variant {v} does not reference 'Hold signal' field.\nGot: {text}"
        )


def test_cnr_all_variants_reference_competitor_changes_position():
    """Every CNR variant must explain the hold signal comes from competitor changes."""
    for v in range(4):
        text = _analysis("competitor_nearly_ready", _CNR_SIGNAL, _CNR_CONTEXT, _CNR_ROI, variant=v)
        assert "competitor changes" in text.lower(), (
            f"CNR variant {v} does not reference 'competitor changes'.\nGot: {text}"
        )


def test_sys_prompt_contains_positional_rule():
    """SYS_VERDICT_LEAN must contain the positional reasoning rule."""
    from agent.prompts import SYS_VERDICT_LEAN
    assert "competitor changes" in SYS_VERDICT_LEAN.lower(), (
        "SYS_VERDICT_LEAN missing 'competitor changes' in positional rule"
    )
    assert any(kw in SYS_VERDICT_LEAN.lower() for kw in ("buried notes", "buried signals")), (
        "SYS_VERDICT_LEAN missing 'buried notes'/'buried signals' in positional rule"
    )


def test_detect_disqualifier_fires_on_preview_in_notes():
    """Disqualifier is detected when 'preview' is in notes and hold signal is NONE."""
    from agent.model_runner import _detect_disqualifier
    notes = ["feature in preview — GA date TBD", "SSO on Enterprise only"]
    assert _detect_disqualifier(notes, "NONE") == "feature in preview — GA date TBD"


def test_detect_disqualifier_no_leak_when_hold_active():
    """Disqualifier does not fire when a hold signal is already active."""
    from agent.model_runner import _detect_disqualifier
    notes = ["feature in preview — GA date TBD"]
    assert _detect_disqualifier(notes, "contract renews in 60 days") == "NONE"


def test_detect_disqualifier_no_leak_for_beta_in_notes():
    """'beta' in notes is caught by hold signal detection, NOT disqualifier (disjoint keyword sets)."""
    from agent.model_runner import _detect_disqualifier
    notes = ["feature in beta — not shipped yet"]
    # beta is in _HOLD_NOTE_KW so hold_signal would be non-NONE; disqualifier won't fire
    assert _detect_disqualifier(notes, "feature in beta — not shipped yet") == "NONE"


def test_build_lean_user_emits_disqualifier_field():
    """_build_lean_user emits 'Disqualifier:' when preview is in notes and hold signal is NONE."""
    import os
    os.environ["AGENT_DRY_RUN"] = "true"
    from agent.context_loader import load_context
    from agent.model_runner import _build_lean_user
    from agent.roi_calculator import calculate_roi, extract_pass1_vars
    from pathlib import Path

    _DATA_ROOT = Path("/home/james/ml-proj/saasmanager/data")
    context = load_context("analytics", "metricflux", _DATA_ROOT)

    signal = {
        "competitor": "MetricFlux",
        "competitor_changes": ["real-time dashboards"],
        "current_tool_status": [],
        "notes": ["feature in preview — GA date TBD"],
        "compliance_changes": "",
    }
    roi = calculate_roi(extract_pass1_vars(context, signal))
    user_msg = _build_lean_user(context, roi, signal)
    assert "Disqualifier: feature in preview — GA date TBD" in user_msg
    assert "Hold signal: NONE" in user_msg
    del os.environ["AGENT_DRY_RUN"]


def test_nsb_analysis_references_disqualifier_label():
    """NSB ANALYSIS templates reference 'Disqualifier' by label name."""
    from training.generate_cot_traces import _analysis
    for v in range(3):
        text = _analysis("negative_signal_buried", _NSB_SIGNAL, _NSB_CONTEXT, _NSB_ROI, variant=v)
        assert "Disqualifier" in text, (
            f"NSB variant {v} does not reference 'Disqualifier'.\nGot: {text}"
        )
