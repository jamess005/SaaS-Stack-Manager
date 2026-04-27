"""
Generate Chain-of-Thought SFT training traces for the SaaS Stack Manager model.

For each sampled signal file, builds a structured reasoning trace that walks
through push signals → pull signals → compliance → ROI → hold condition → verdict.
These traces teach the model HOW to reason rather than what pattern to match.

Usage:
    python training/generate_cot_traces.py              # 8 per scenario (~144 total)
    python training/generate_cot_traces.py --limit 5    # 5 per scenario (~90 total)
    python training/generate_cot_traces.py --limit 12   # 12 per scenario (~216 total)
    python training/generate_cot_traces.py --balance --val-ratio 0.15
"""

import argparse
from collections import Counter, defaultdict
import json
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_OUTPUT_PATH = _PROJECT_ROOT / "training" / "sft_cot_traces.jsonl"

sys.path.insert(0, str(_PROJECT_ROOT))

from training.drift_canaries import DRIFT_CANARY_FILENAMES  # noqa: E402
from training.reasoning_alignment import build_semantic_view  # noqa: E402
from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.prompts import SYS_VERDICT_LEAN, _CATEGORY_RULES_COMPACT  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import (  # noqa: E402
    signal_compliance_changes,
    signal_competitor_changes,
    signal_current_tool_status,
    signal_notes,
)

# ── Scenario metadata ──────────────────────────────────────────────────────────

SCENARIO_TYPES: list[str] = [
    "hard_compliance_failure",
    "fluff_update",
    "irrelevant_change",
    "negative_signal_buried",
    "current_tool_rally",
    "pull_dominant",
    "push_dominant",
    "shelfware_case",
    "hold_resolved",
    "compliance_newly_met",
    "competitor_nearly_ready",
    "roadmap_confirmed_hold",
    "contract_renewal_hold",
    "vendor_acquisition_hold",
    "pilot_in_progress_hold",
    "both_signals",
    "price_hike_only",
    "dual_improvement",
]

EXPECTED_VERDICT: dict[str, str | None] = {
    "hard_compliance_failure":  "STAY",
    "fluff_update":             "STAY",
    "irrelevant_change":        "STAY",
    "negative_signal_buried":   "STAY",
    "current_tool_rally":       "STAY",
    "pull_dominant":            "SWITCH",
    "push_dominant":            "SWITCH",
    "shelfware_case":           "SWITCH",
    "hold_resolved":            "SWITCH",
    "compliance_newly_met":     "SWITCH",
    "competitor_nearly_ready":  "HOLD",
    "roadmap_confirmed_hold":   "HOLD",
    "contract_renewal_hold":    "HOLD",
    "vendor_acquisition_hold":  "HOLD",
    "pilot_in_progress_hold":   "HOLD",
    "both_signals":             None,
    "price_hike_only":          None,
    "dual_improvement":         None,
}

_HOLD_SCENARIOS = {
    "competitor_nearly_ready",
    "roadmap_confirmed_hold",
    "contract_renewal_hold",
    "vendor_acquisition_hold",
    "pilot_in_progress_hold",
}

_SKIP_SCENARIOS: set[str] = set()

# ── Filename parsing ───────────────────────────────────────────────────────────

def _parse_filename(stem: str) -> tuple[str, str, str] | None:
    """Parse {category}_{competitor}_{scenario} stem. Returns None if unparseable."""
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat + "_"):
            remainder = stem[len(cat) + 1:]
            for scenario in sorted(SCENARIO_TYPES, key=len, reverse=True):
                if remainder.endswith("_" + scenario):
                    slug = remainder[: -(len(scenario) + 1)]
                    if slug:
                        return cat, slug, scenario
    return None


# ── Signal quality assessment ─────────────────────────────────────────────────

_CONCRETE_KW = {
    "native", "integration", "api", "connector", "sso", "saml", "soc2", "gdpr",
    "iso", "invoic", "currency", "export", "import", "sync", "bulk", "offline",
    "certif", "hipaa", "pci", "audit", "mfa", "2fa", "webhook", "module",
    "dashboard", "report", "analytic", "automat", "migrat", "rest",
    "bank", "feed", "built-in", "scheduling", "tracking", "sequenc",
    "unlimited", "pipeline", "portal", "embedded", "compliance",
}

_VAGUE_KW = {
    "enhanced", "improved", "streamlined", "modernised", "modernized", "revamped",
    "better", "easier", "intuitive", "user-friendly", "experience", "workflow",
    "optimised", "optimized", "performance", "stability", "reliability",
    "refreshed", "updated", "polished", "smoother", "cleaner",
}

_HIGH_SEVERITY_KW = {
    "sso", "saml", "mfa", "compliance", "gdpr", "soc2", "iso", "hipaa", "pci",
    "down", "crash", "fail", "error", "broken", "missing", "no ", "lack",
    "cannot", "can't", "doesn't", "not available", "unavailable", "critical",
    "block", "prevent", "required", "mandatory",
    "shelfware", "inactive seats", "unused capacity", "underutilised", "underutilized",
}

_MEDIUM_SEVERITY_KW = {
    "slow", "manual", "complex", "difficult", "limited", "basic", "outdated",
    "workaround", "friction", "inconsistent", "unreliable", "clunky",
}


def _substance(text: str) -> str:
    lower = text.lower()
    c = sum(1 for kw in _CONCRETE_KW if kw in lower)
    v = sum(1 for kw in _VAGUE_KW if kw in lower)
    if c > v:
        return "CONCRETE"
    if v > c:
        return "VAGUE"
    return "VAGUE"  # Default to VAGUE for ambiguous cases


def _severity(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _HIGH_SEVERITY_KW):
        return "HIGH"
    if any(kw in lower for kw in _MEDIUM_SEVERITY_KW):
        return "MEDIUM"
    return "LOW"


_HOLD_NOTE_KW = frozenset([
    "hold:", "acquisition", "roadmap", "renews", "renewal", "pilot",
])

_HOLD_COMP_KW = frozenset(["beta", "roadmap", "not ga", "preview", "early access"])

_ADVISORY_PREFIXES = ("consider", "suggest", "recommend", "note:", "fyi")

_NEGATION_PATTERNS = re.compile(
    r"no\s+(beta|roadmap|caveats|hold)"
    r"|all\b.*\bga\b"
    r"|now\s+ga"
    r"|without\s+(beta|roadmap|caveats)",
    re.IGNORECASE,
)


def _detect_hold_signal(notes: list[str], comp_changes: list[str] | None = None) -> str:
    """Return the first hold condition found in notes (or competitor_changes), or 'NONE'."""
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if _NEGATION_PATTERNS.search(note):
            continue
        if any(kw in note_lower for kw in _HOLD_NOTE_KW):
            return note
    for change in (comp_changes or []):
        change_lower = change.lower()
        if _NEGATION_PATTERNS.search(change_lower):
            continue
        if any(kw in change_lower for kw in _HOLD_COMP_KW):
            return change
    return "NONE"


_DISQUALIFIER_NOTE_KW = frozenset([
    "preview", "early access", "beta", "not ga",
    "no relevance", "not relevant", "irrelevant to", "irrelevant", "unrelated",
    "poor fit", "wrong product", "not designed for",
    "hard requirement",
    "without relevance", "does not align", "does not address", "no impact on",
])

_DISQUALIFIER_NEGATION = re.compile(
    r"no\s+(beta|roadmap|caveats|hold)"
    r"|all\b.*\bga\b"
    r"|now\s+ga"
    r"|without\s+(beta|roadmap|caveats)",
    re.IGNORECASE,
)


def _detect_disqualifier(notes: list[str], hold_signal: str) -> str:
    """Return a disqualifier note if pre-GA language appears in notes and no hold is active.

    Uses the same negation check as hold detection to avoid false positives from
    phrases like 'no beta or roadmap caveats' which negate the keyword.
    """
    if hold_signal != "NONE":
        return "NONE"
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if _DISQUALIFIER_NEGATION.search(note):
            continue
        if any(kw in note_lower for kw in _DISQUALIFIER_NOTE_KW):
            return note
    return "NONE"


_SHELFWARE_KW = frozenset(["shelfware", "inactive seats", "inactive seat"])

_SHELFWARE_NEGATION = re.compile(
    r"no\s+shelfware|not\s+shelfware|no\s+inactive\s+seats?|no\s+unused"
    r"|reduced\s+to\s+0|seats\s+reduced|inactive\s+seats?\s+decrease"
    r"|remains?\s+stable|increase\s+by\s+[0-5]%",
    re.IGNORECASE,
)

_SHELFWARE_GLOBAL_NEGATION = re.compile(
    r"shelfware(?:\s+(?:flag|status))?(?:\s*(?::|=)\s*|\s+remains?\s+)(?:false|no|none)\b",
    re.IGNORECASE,
)

_SHELFWARE_COUNT = re.compile(r"inactive\s+seats?[^\d]*(\d+)", re.IGNORECASE)
_SHELFWARE_PERCENT = re.compile(r"(\d+)%", re.IGNORECASE)

_PRICE_HIKE_RE = re.compile(
    r"(?:price|pricing|cost|tier|per[- ]user|per[- ]active[- ]user)[^\n]{0,40}"
    r"(?:increase|increased|increasing|higher|hike|rise|rose|raised|more expensive)"
    r"|(?:increase|increased|higher|raised)[^\n]{0,20}(?:price|pricing|cost|tier)"
    r"|per[- ]active[- ]user\s+pricing",
    re.IGNORECASE,
)


def _detect_shelfware(tool_changes: list[str]) -> str:
    """Return the shelfware-flagging line from current_tool_status, or 'NONE'.

    Excludes negation phrases ('no shelfware', 'no inactive seats') to avoid
    false positives from signal files that write negation as a status line.
    A global negation ('Shelfware: False') anywhere in tool_changes overrides all
    detections.
    """
    for change in tool_changes:
        if _SHELFWARE_GLOBAL_NEGATION.search(change):
            return "NONE"
    for change in tool_changes:
        lower = change.lower()
        if any(kw in lower for kw in _SHELFWARE_KW):
            if _SHELFWARE_NEGATION.search(change):
                continue
            if "shelfware" in lower or "unused capacity" in lower or "underutilis" in lower:
                return change
            inactive_counts = [int(value) for value in _SHELFWARE_COUNT.findall(change)]
            if inactive_counts and max(inactive_counts) >= 10:
                return change
            percents = [int(value) for value in _SHELFWARE_PERCENT.findall(change)]
            if percents and max(percents) >= 30:
                return change
    return "NONE"


def _detect_price_hike(tool_changes: list[str]) -> str:
    for change in tool_changes:
        if _PRICE_HIKE_RE.search(change):
            return change
    return "NONE"


_TRAINING_PLACEHOLDER_REPLACEMENTS = {
    "state the contract renewal timing explicitly": "the contract renewal window is not yet open",
    "shallow_degradation": "a shallow incumbent degradation signal",
    "minor maintenance update only": "a maintenance-only competitor update",
    "issues persist": "the incumbent issues remain unresolved",
    "known issues persist": "the incumbent issues remain unresolved",
    "push issues persisting": "the incumbent issues remain unresolved",
    "surface cannot deliver": "the competitor cannot deliver",
    "acquisition certainty uncertain": "vendor acquisition remains uncertain",
    "shelfware case: no new changes": "a shelfware-driven switch case even without major new competitor changes",
}


def _normalize_training_text(text: str) -> str:
    cleaned = text
    for token, replacement in _TRAINING_PLACEHOLDER_REPLACEMENTS.items():
        cleaned = re.sub(re.escape(token), replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\.\.+", ".", cleaned)
    return cleaned.strip()


def _normalize_training_list(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        normalized = _normalize_training_text(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _primary_issue_for_boundary(context: dict, signal: dict) -> str:
    view = _semantic_view(context, signal)
    return view.primary_issue or "the incumbent gap"


def _primary_pull_for_boundary(context: dict, signal: dict, *, allow_pending: bool = False) -> str:
    view = _semantic_view(context, signal)
    if view.positive_pull is not None:
        return view.positive_pull.change
    if allow_pending and view.pending_pull is not None:
        return view.pending_pull.change
    return view.primary_pull or "the competitor change"


def _has_active_hold_signal(signal: dict) -> bool:
    notes = _normalize_training_list(signal_notes(signal))
    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    return _detect_hold_signal(notes, comp_changes) != "NONE"


def _active_hold_signal_text(signal: dict) -> str:
    notes = _normalize_training_list(signal_notes(signal))
    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    hold = _detect_hold_signal(notes, comp_changes)
    return _normalize_training_text(hold) if hold != "NONE" else "NONE"


def _disqualifier_text(signal: dict) -> str:
    notes = _normalize_training_list(signal_notes(signal))
    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    hold = _detect_hold_signal(notes, comp_changes)
    disqualifier = _detect_disqualifier(notes, hold)
    return _normalize_training_text(disqualifier) if disqualifier != "NONE" else "NONE"


def _shelfware_text(signal: dict) -> str:
    tool_changes = _normalize_training_list(signal_current_tool_status(signal))
    shelfware = _detect_shelfware(tool_changes)
    return _normalize_training_text(shelfware) if shelfware != "NONE" else "NONE"


def _boundary_rejections(verdict: str, scenario: str, signal: dict, context: dict, roi: dict) -> str:
    view = _semantic_view(context, signal)
    issue = _primary_issue_for_boundary(context, signal)
    pull = _primary_pull_for_boundary(context, signal, allow_pending=verdict == "HOLD")
    hold = _active_hold_signal_text(signal)
    disqualifier = _disqualifier_text(signal)
    shelfware = _shelfware_text(signal)
    roi_str = f"£{roi['annual_net_gbp']:+.0f}/yr"

    if verdict == "STAY":
        if disqualifier != "NONE":
            not_switch = (
                f"Not SWITCH: the disqualifier '{disqualifier}' makes the competitor non-viable today, "
                f"so there is no valid migration target."
            )
        elif view.positive_pull is None:
            not_switch = (
                f"Not SWITCH: there is no shipped competitor change that clearly addresses '{issue}', "
                f"so the switch case never starts."
            )
        elif not roi["roi_threshold_met"]:
            not_switch = f"Not SWITCH: the economics still fail the move, with ROI below threshold at {roi_str}."
        else:
            not_switch = f"Not SWITCH: the competitor change ('{pull}') does not solve the decisive gap ('{issue}') clearly enough to justify migration."

        if hold != "NONE":
            not_hold = f"Not HOLD: '{hold}' is not the decisive issue here; even after waiting, the switch case still lacks merit."
        elif disqualifier != "NONE":
            not_hold = "Not HOLD: this is not a wait-for-later case; the blocker is a real disqualifier, not a temporary delivery or timing gate."
        else:
            not_hold = "Not HOLD: no active contract, pilot, roadmap, or pre-GA gate is blocking the decision, so waiting would not create a stronger case."
        return f"{not_switch} {not_hold}"

    if verdict == "HOLD":
        not_switch = f"Not SWITCH: the active hold condition ('{hold}') still blocks commitment, so switching now would ignore a live temporal gate."
        if view.positive_pull is not None or view.pending_pull is not None:
            not_stay = (
                f"Not STAY: the competitor still shows a real switch case once the blocker clears, because '{pull}' "
                f"materially improves the incumbent gap."
            )
        else:
            not_stay = "Not STAY: the case is paused for timing, not rejected on merit; the decisive issue is when to move, not whether the competitor is viable at all."
        return f"{not_switch} {not_stay}"

    if shelfware != "NONE":
        not_stay = (
            f"Not STAY: staying would preserve known waste from underutilisation ('{shelfware}'), "
            "which is itself the decisive reason to move."
        )
    elif view.positive_pull is not None:
        not_stay = f"Not STAY: staying leaves '{issue}' unresolved even though the competitor now ships '{pull}'."
    else:
        not_stay = "Not STAY: staying leaves the decisive gap unresolved even though the source trace still presents a viable path away from the incumbent."

    if signal.get("previous_verdict") == "HOLD" and hold == "NONE":
        not_hold = "Not HOLD: Hold status is RESOLVED — the prior blocker is gone; re-issuing HOLD would ignore the resolved gate."
    else:
        not_hold = "Not HOLD: there is no active contract, pilot, roadmap, or pre-GA blocker left to wait on, so delaying would only postpone a justified move."
    return f"{not_stay} {not_hold}"


# ── Compliance status (matches _compliance_pass_python logic in model_runner.py) ─

_COMPLIANCE_POSITIVE_KW = frozenset({
    "achieved", "certified", "now available", "launched", "shipped",
    "enabled", "live", "now meets", "now compliant", "passed", "attained",
})


def _effective_compliance_status(context: dict, signal: dict) -> str:
    """Compute compliance status from baseline profile + signal text (no model call).

    Used internally for CoT generation and consistency checks — NOT shown to the model
    in the user message (the raw profile is shown instead via _format_compliance_block).
    """
    category = context["category"]
    seat_count = context["current_stack_entry"].get("seat_count", 0)
    profile = dict(context.get("competitor_data", {}).get("compliance", {}))

    cc = _normalize_training_text(signal_compliance_changes(signal))
    if cc and cc.lower().strip() not in ("unchanged", "no change", "none", ""):
        cc_l = cc.lower()
        positive = any(kw in cc_l for kw in _COMPLIANCE_POSITIVE_KW)

        if "soc2" in cc_l or "soc 2" in cc_l:
            if positive and "not soc2" not in cc_l and "no soc2" not in cc_l:
                profile["soc2_type2"] = True
            elif "not soc2" in cc_l or "no soc2" in cc_l:
                profile["soc2_type2"] = False

        if any(kw in cc_l for kw in ("sso", "saml", "oidc")):
            if positive and not any(kw in cc_l for kw in ("no sso", "lacks sso", "no saml", "lacks saml")):
                profile["sso_saml"] = True
            elif any(kw in cc_l for kw in ("no sso", "lacks sso", "no saml", "lacks saml")):
                profile["sso_saml"] = False

        if "uk" in cc_l and "residency" in cc_l:
            if positive:
                profile["uk_residency"] = True
                profile["gdpr_eu_residency"] = True
            else:
                profile["uk_residency"] = False

        if "audit" in cc_l:
            if any(kw in cc_l for kw in ("no audit log", "lacks audit log", "no audit trail", "no exportable audit")):
                profile["audit_log"] = False
                profile["audit_log_exportable"] = False
            elif positive:
                profile["audit_log"] = True
                profile["audit_log_exportable"] = True

    failures: list[str] = []
    if not profile.get("soc2_type2"):
        failures.append("No SOC2 Type II")
    if seat_count > 10 and not profile.get("sso_saml"):
        failures.append(f"No SSO/SAML ({seat_count} seats)")
    if not (profile.get("uk_residency") or profile.get("gdpr_eu_residency")):
        failures.append("No UK/EU data residency")
    if category in ("finance", "hr", "crm"):
        audit_ok = profile.get("audit_log") and profile.get("audit_log_exportable", True)
        if not audit_ok:
            failures.append("No exportable audit log")

    return f"BLOCKED — {'; '.join(failures)}" if failures else "PASSED"


def _format_compliance_block(context: dict, signal: dict) -> str:
    """Format raw competitor compliance profile + signal for the model to reason from.

    Mirrors _format_compliance_block in model_runner.py — must stay in sync.
    """
    profile = context.get("competitor_data", {}).get("compliance", {})
    soc2 = "Yes" if profile.get("soc2_type2") else "No"
    sso = "Yes" if profile.get("sso_saml") else "No"
    uk = profile.get("uk_residency", False)
    eu = profile.get("gdpr_eu_residency", False)
    residency = ("UK+EU" if (uk and eu) else "UK" if uk else "EU" if eu else "No")
    al = profile.get("audit_log", False)
    ale = profile.get("audit_log_exportable", True) if al else False
    audit = f"Yes (exportable: {'Yes' if ale else 'No'})" if al else "No"
    cc = _normalize_training_text(signal_compliance_changes(signal)) or "unchanged"
    return (
        f"Competitor compliance: SOC2={soc2} | SSO/SAML={sso} | Residency={residency} | Audit log={audit}\n"
        f"Compliance signal: {cc}"
    )


# ── User message builder (matches _build_lean_user in model_runner.py) ────────

def _build_user_message(context: dict, roi_result: dict, signal: dict, scenario: str = "") -> str:
    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    issues = _normalize_training_list(context["current_stack_entry"].get("known_issues", []))
    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none)"

    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    tool_changes = _normalize_training_list(signal_current_tool_status(signal))
    notes = _normalize_training_list(signal_notes(signal))

    comp_text = "\n".join(f"- {c}" for c in comp_changes) if comp_changes else "(none)"
    tool_change_text = (
        "\n".join(f"- {c}" for c in tool_changes) if tool_changes else "(unchanged this period)"
    )
    notes_text = "\n".join(f"- {n}" for n in notes) if notes else ""
    category_rules = _CATEGORY_RULES_COMPACT.get(category, "")

    roi_summary = (
        f"Migration: £{roi_result['migration_cost_one_time']:.0f}, "
        f"Annual net: £{roi_result['annual_net_gbp']:.0f}, "
        f"Threshold: {'MET' if roi_result['roi_threshold_met'] else 'NOT MET'}"
    )

    msg = (
        f"Category: {category} — current tool: {tool_name}\n"
        f"{category_rules}\n\n"
        f"Current tool known issues:\n{issues_text}\n\n"
        f"Changes this period:\n"
        f"  Current tool: {tool_change_text}\n"
        f"  Competitor: {comp_text}\n"
    )
    if notes_text:
        msg += f"\nBuried signals / notes:\n{notes_text}\n"
    hold_signal = _detect_hold_signal(notes, comp_changes)
    disqualifier = _detect_disqualifier(notes, hold_signal)
    msg += f"\nROI: {roi_summary}"
    msg += f"\n{_format_compliance_block(context, signal)}"
    msg += f"\nHold signal: {hold_signal}"
    if disqualifier != "NONE":
        msg += f"\nDisqualifier: {disqualifier}"
    # Add Shelfware flag whenever detected — matches inference behavior in model_runner.py.
    # The model must learn to handle Shelfware flag present in ALL verdict contexts.
    # Disqualifier takes priority over Shelfware (don't present both — contradictory).
    shelfware_signal = _detect_shelfware(tool_changes)
    if shelfware_signal != "NONE" and hold_signal == "NONE" and disqualifier == "NONE":
        msg += f"\nShelfware flag: {shelfware_signal}"
    prev_verdict = signal.get("previous_verdict")
    if prev_verdict == "HOLD" and hold_signal == "NONE":
        msg += "\nHold status: RESOLVED (prior verdict was HOLD — blocker has now cleared)"
    elif prev_verdict:
        msg += f"\nPrevious verdict: {prev_verdict}"
    return msg


_NO_CHANGE_MARKERS = {
    "no change",
    "no change — existing issues persist.",
    "no change this period",
    "stable",
    "unchanged",
}


def _semantic_view(context: dict, signal: dict):
    known = _normalize_training_list(context["current_stack_entry"].get("known_issues", []))
    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    tool_changes = [
        change for change in _normalize_training_list(signal_current_tool_status(signal))
        if change.lower().strip() not in _NO_CHANGE_MARKERS
    ]
    return build_semantic_view(known, comp_changes, tool_changes)


def _supports_switch_case(context: dict, signal: dict, roi: dict, *, allow_pending: bool = False) -> bool:
    if _shelfware_text(signal) != "NONE":
        return True

    view = _semantic_view(context, signal)
    match = view.positive_pull or (view.pending_pull if allow_pending else None)
    if match is None:
        return False

    if roi["roi_threshold_met"]:
        return True

    return _severity(match.issue) == "HIGH"


def _has_reassuring_current_tool_signal(signal: dict) -> bool:
    reassuring_terms = (
        "reliable",
        "stable",
        "consistent",
        "perform well",
        "no significant issues",
        "no major incidents",
        "responsive within",
        "deadline reminders",
        "intuitive",
    )
    tool_changes = _normalize_training_list(signal_current_tool_status(signal))
    return any(any(term in change.lower() for term in reassuring_terms) for change in tool_changes)


def _has_pending_hold_pull(context: dict, signal: dict) -> bool:
    view = _semantic_view(context, signal)
    return view.pending_pull is not None or view.has_pending_pull


def _consistency_failure_reason(scenario: str, signal: dict, context: dict, roi: dict, verdict: str) -> str | None:
    notes = _normalize_training_list(signal_notes(signal))
    comp_changes = _normalize_training_list(signal_competitor_changes(signal))
    tool_changes = _normalize_training_list(signal_current_tool_status(signal))
    hold_signal = _detect_hold_signal(notes, comp_changes)
    disqualifier = _detect_disqualifier(notes, hold_signal)
    shelfware_signal = _detect_shelfware(tool_changes)
    price_hike_signal = _detect_price_hike(tool_changes)
    view = _semantic_view(context, signal)

    if verdict == "HOLD" and hold_signal == "NONE":
        return "HOLD example has no active hold signal"

    if scenario in _HOLD_SCENARIOS and hold_signal == "NONE":
        return "HOLD scenario but no detectable hold signal"

    if scenario == "hold_resolved":
        if signal.get("previous_verdict") != "HOLD":
            return "hold_resolved scenario without Previous verdict: HOLD"
        if hold_signal != "NONE":
            return "hold_resolved scenario still has an active hold signal"
        if not _supports_switch_case(context, signal, roi):
            return "hold_resolved scenario lacks a concrete aligned switch case"

    if scenario == "shelfware_case" and shelfware_signal == "NONE":
        return "shelfware_case scenario lacks a real shelfware flag"

    if scenario == "price_hike_only" and price_hike_signal == "NONE":
        return "price_hike_only scenario lacks a real incumbent price signal"

    if verdict == "SWITCH" and shelfware_signal == "NONE" and not _supports_switch_case(context, signal, roi):
        return "SWITCH example lacks a concrete competitor change aligned to a key issue"

    if scenario == "current_tool_rally" and view.positive_tool is None and not _has_reassuring_current_tool_signal(signal):
        return "current_tool_rally scenario lacks a real current-tool improvement"

    if scenario == "dual_improvement":
        if view.positive_tool is None:
            return "dual_improvement scenario lacks a real current-tool improvement"
        if view.positive_pull is None:
            return "dual_improvement scenario lacks a competitor improvement aligned to a key issue"

    if scenario == "competitor_nearly_ready" and not _has_pending_hold_pull(context, signal):
        return "competitor_nearly_ready scenario lacks a pending pull aligned to a key issue"

    if scenario == "roadmap_confirmed_hold":
        if not (_supports_switch_case(context, signal, roi) or _has_pending_hold_pull(context, signal)):
            return "roadmap_confirmed_hold scenario lacks a roadmap pull that would justify waiting"

    if scenario in {"contract_renewal_hold", "vendor_acquisition_hold", "pilot_in_progress_hold"}:
        if not _supports_switch_case(context, signal, roi):
            return "hold scenario lacks a concrete underlying switch case once the hold clears"

    if scenario == "fluff_update" and view.positive_pull is not None:
        return "fluff_update scenario contains a concrete aligned competitor pull"

    if scenario == "irrelevant_change" and view.positive_pull is not None and disqualifier == "NONE":
        return "irrelevant_change scenario contains an aligned competitor pull"

    if scenario == "negative_signal_buried" and disqualifier == "NONE":
        return "negative_signal_buried scenario lacks a buried disqualifier"

    if shelfware_signal != "NONE" and hold_signal == "NONE" and verdict == "STAY" and disqualifier == "NONE":
        return "shelfware flag present with STAY verdict and no disqualifier"

    # Compliance consistency: blocked competitors only appear in hard_compliance_failure traces
    compliance_status = _effective_compliance_status(context, signal)
    compliance_blocked = compliance_status.startswith("BLOCKED")
    if scenario == "hard_compliance_failure" and not compliance_blocked:
        return "hard_compliance_failure scenario but compliance is not blocked"
    if compliance_blocked and scenario != "hard_compliance_failure":
        return f"compliance blocked ({compliance_status}) in non-compliance scenario"

    return None

# ── CoT trace section builders ────────────────────────────────────────────────

def _push_section(context: dict, signal: dict) -> str:
    known = _normalize_training_list(context["current_stack_entry"].get("known_issues", []))
    # Strip generic no-change placeholders — these are not push signals
    tool_changes = [
        c for c in _normalize_training_list(signal_current_tool_status(signal))
        if c.lower().strip() not in _NO_CHANGE_MARKERS
    ]

    lines: list[str] = []
    # Current-period degradation items first (most salient)
    for change in tool_changes[:3]:
        lines.append(f"  - {change} — Severity: {_severity(change)}")
    # Ongoing known issues (cap at 2 to avoid bloat)
    for issue in known[:2]:
        sev = _severity(issue)
        lines.append(f"  - {issue} — Severity: {sev} (ongoing)")

    if not lines:
        return "PUSH SIGNALS:\n  None flagged this period"
    return "PUSH SIGNALS:\n" + "\n".join(lines)


def _pull_section(signal: dict) -> str:
    changes = _normalize_training_list(signal_competitor_changes(signal))
    if not changes:
        return "PULL SIGNALS:\n  None identified"
    lines = [f"  - {c} — Substance: {_substance(c)}" for c in changes[:3]]
    return "PULL SIGNALS:\n" + "\n".join(lines)


_COMPLIANCE_HARD_TERMS = frozenset([
    "soc2", "soc 2", "sso", "saml", "oidc", "audit log", "audit trail",
    "uk residency", "eu residency", "gdpr residency", "data residency",
])


def _compliance_cot_line(context: dict, signal: dict) -> str:
    """Generate the COMPLIANCE line for the CoT trace.

    Uses the effective status (baseline + signal applied) to determine what the
    model should conclude, then formats it as the model would reason from the
    raw profile shown in the user message.
    """
    eff = _effective_compliance_status(context, signal)
    cc = _normalize_training_text(signal_compliance_changes(signal))
    has_positive_change = (
        cc
        and cc.lower().strip() not in ("unchanged", "no change", "none", "")
        and any(kw in cc.lower() for kw in _COMPLIANCE_POSITIVE_KW)
        and any(ht in cc.lower() for ht in _COMPLIANCE_HARD_TERMS)
    )
    if eff.startswith("BLOCKED"):
        return f"COMPLIANCE: {eff}"
    if has_positive_change:
        return "COMPLIANCE: Previously BLOCKED — now MET"
    return "COMPLIANCE: PASSED"


def _roi_line(roi: dict) -> str:
    met = "MET" if roi["roi_threshold_met"] else "NOT MET"
    return (
        f"ROI: Migration £{roi['migration_cost_one_time']:.0f} | "
        f"Annual net £{roi['annual_net_gbp']:+.0f} → Threshold {met}"
    )


def _hold_line(signal: dict, scenario: str) -> str:
    notes = _normalize_training_list(signal_notes(signal))
    comp_ch = _normalize_training_list(signal_competitor_changes(signal))
    # Use the same detection logic as _build_user_message / _build_lean_user
    hold = _detect_hold_signal(notes, comp_ch)
    if scenario in _HOLD_SCENARIOS:
        if hold != "NONE":
            return f"HOLD CONDITION: {hold.replace(chr(10), ' ')}"
        # Fallback to first note if present (shouldn't happen after signal repair)
        if notes:
            return f"HOLD CONDITION: {notes[0].replace(chr(10), ' ')}"
    return "HOLD CONDITION: NONE"


# ── ANALYSIS text templates (scenario-specific) ───────────────────────────────

def _analysis(scenario: str, signal: dict, context: dict, roi: dict, variant: int = 0) -> str:
    comp = signal.get("competitor", "Competitor")
    tool = context["current_stack_entry"]["tool"]
    known = _normalize_training_list(context["current_stack_entry"].get("known_issues", []))
    comp_ch = _normalize_training_list(signal_competitor_changes(signal))
    tool_ch = _normalize_training_list(signal_current_tool_status(signal))
    notes = _normalize_training_list(signal_notes(signal))
    cc = _normalize_training_text(signal_compliance_changes(signal))

    view = _semantic_view(context, signal)
    relevant_pull = view.positive_pull or view.pending_pull
    relevant_tool_negative = view.negative_tool
    relevant_tool_positive = view.positive_tool

    issue = _normalize_training_text(((relevant_pull.issue if relevant_pull else view.primary_issue) or "existing tool limitations").replace("\n", " "))
    top_comp = _normalize_training_text(((relevant_pull.change if relevant_pull else view.primary_pull) or "minor updates").replace("\n", " "))
    top_tool = _normalize_training_text(((relevant_tool_negative.change if relevant_tool_negative else view.primary_tool_change) or "no new changes").replace("\n", " "))
    rally_tool = _normalize_training_text(((relevant_tool_positive.change if relevant_tool_positive else view.primary_tool_change) or "active improvements").replace("\n", " "))
    top_note = _normalize_training_text((notes[0] if notes else "").replace("\n", " "))
    net = roi["annual_net_gbp"]
    met = roi["roi_threshold_met"]
    roi_str = f"ROI threshold {'met' if met else 'not met'} at £{net:+.0f}/yr net"
    issue_severity = _severity(issue)

    if scenario == "pull_dominant":
        templates = [
            # V0 — pull signal resolves push issue
            (
                f"{comp} delivers '{top_comp}' — a concrete new capability that directly resolves '{issue}'. "
                f"This is not a vague update; it addresses the specific push issue. "
                f"ROI gate: {roi_str}. Compliance: PASSED. Hold signal: NONE. All gates clear. SWITCH."
            ),
            # V1 — gate-check-first
            (
                f"Gate check: Compliance PASSED. Hold signal NONE. ROI {roi_str}. All clear. "
                f"Pull quality: CONCRETE — '{top_comp}' from {comp} resolves '{issue}'. "
                f"All gates clear. SWITCH."
            ),
            # V2 — delivered capability, derive from evidence
            (
                f"{comp} ships '{top_comp}'. This is a delivered capability, not a roadmap item — "
                f"it can be relied upon today. It addresses {tool}'s primary gap: {issue}. "
                f"When a concrete pull signal resolves the primary push driver and all blocking gates "
                f"clear (compliance PASSED, hold NONE, {roi_str}), the switch case is established. SWITCH."
            ),
            # V3 — decision framework reasoning
            (
                f"Step 1: Check blocking gates. Compliance: PASSED. Hold signal: NONE. No blocks. "
                f"Step 2: Evaluate pull substance. '{top_comp}' is CONCRETE — a specific, shipped capability. "
                f"Step 3: Match pull to push. Push driver: '{issue}'. Pull resolves it directly. "
                f"Step 4: ROI check. {roi_str}. "
                f"All four steps clear → SWITCH."
            ),
            # V4 — explicit alternatives ruled out
            (
                f"Gates: Compliance PASSED. Hold signal NONE. {roi_str}. No blocking conditions. "
                f"Pull substance: CONCRETE — '{top_comp}' is a shipped, releasable capability (not roadmap). "
                f"Pull resolves push: '{top_comp}' directly addresses '{issue}', the primary driver. "
                f"Not STAY: staying means accepting '{issue}' indefinitely — the competitor offers a concrete resolution. "
                f"Not HOLD: Hold signal is NONE — there is no condition waiting to be resolved. "
                f"All gates clear, pull is concrete and targeted. SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "push_dominant":
        push = _normalize_training_text(((relevant_tool_negative.change if relevant_tool_negative else top_tool) or issue).replace("\n", " "))
        templates = [
            (
                f"Push signal: HIGH — {tool} is actively harming operations ({push}). "
                f"{comp} delivers '{top_comp}', which addresses '{issue}'. "
                f"Compliance gate: PASSED. Hold signal: NONE. ROI gate: {roi_str}. SWITCH."
            ),
            (
                f"{tool} is degrading where Meridian actually feels it: {push}. "
                f"{comp}'s delivered change ('{top_comp}') addresses the same issue rather than a side concern. "
                f"With the decisive gap matched to a live alternative, and no compliance or hold blocker remaining, SWITCH."
            ),
            (
                f"Both signals point the same way. The incumbent gap is '{issue}', and {tool} is still degrading through '{push}'. "
                f"{comp} now ships '{top_comp}', which gives the business a concrete exit path on that exact problem. "
                f"{roi_str}. Compliance PASSED. Hold NONE. SWITCH."
            ),
            (
                f"This is not a vague escape-hatch case. {tool} is deteriorating at '{push}', and {comp} answers the same operational problem with '{top_comp}'. "
                f"Because the current pain is live, the competitor response is delivered, and the blocking gates are clear, the rational move is SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "shelfware_case":
        advisory_note = next((n for n in notes if n.lower().startswith(_ADVISORY_PREFIXES)), None)
        if advisory_note:
            notes_clarification = f" Advisory note '{advisory_note[:60]}' is NOT a hold condition."
        elif notes:
            notes_clarification = (
                " Competitor limitations or restrictions in notes do not block this"
                " SWITCH — the decision is driven by cost waste, not competitor quality."
            )
        else:
            notes_clarification = ""
        roi_override = (
            f"ROI: {roi_str} — shelfware exception applies; "
            f"the annual ROI figure does not block SWITCH when shelfware waste is the dominant signal."
        )
        templates = [
            # V0 — waste elimination with explicit ROI override
            (
                f"SWITCH CONFIRMED — shelfware case. {tool} is underutilised ({top_tool}). "
                f"Paying for unused capacity with no utilisation improvement justifies switching. "
                f"Shelfware waste elimination overrides the standard ROI threshold — "
                f"the cost of inaction is the ongoing waste. "
                f"{roi_override} "
                f"Compliance: PASSED. Hold signal: NONE. "
                f"Switch driven by waste elimination, not by competitor features.{notes_clarification} SWITCH."
            ),
            # V1 — cost framing with explicit ROI override
            (
                f"Decision driver: waste, not competitor quality. {tool} is {top_tool}. "
                f"The organisation pays for idle capacity with no utilisation plan. "
                f"The cost of staying (continued waste) exceeds migration cost. "
                f"{roi_override} "
                f"Compliance PASSED. Hold NONE.{notes_clarification} SWITCH."
            ),
            # V2 — cost of inaction reasoning with explicit ROI override
            (
                f"Shelfware case: {top_tool}. The cost of staying is not zero — it is the "
                f"ongoing waste of paying for unused capacity. When the cost of inaction "
                f"(continued waste) is certain and the cost of action (migration) is one-time, "
                f"the economics favour switching. {roi_override} "
                f"Compliance PASSED.{notes_clarification} SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "hold_resolved":
        if met:
            roi_note = f"ROI: {roi_str} — standard switch economics are satisfied."
        else:
            if issue_severity == "HIGH":
                roi_note = (
                    f"ROI: {roi_str} — below threshold, but '{issue}' remains a HIGH-severity blocker, "
                    f"so the delivered capability still creates clear operational gain."
                )
            else:
                roi_note = f"ROI: {roi_str}."
        templates = [
            (
                f"Hold status is RESOLVED — the prior blocker has cleared and HOLD is not available. "
                f"Re-evaluating on normal switch criteria: {comp} now delivers '{top_comp}', which addresses '{issue}'. "
                f"Compliance PASSED. {roi_note} With the blocker gone and the core issue now covered, SWITCH."
            ),
            (
                f"Hold status RESOLVED means the prior hold condition is gone. Once there is no active hold, "
                f"the decision reverts to the ordinary SWITCH/STAY test. Here it passes: {comp} delivers '{top_comp}', "
                f"which now covers '{issue}'. Compliance PASSED. {roi_note} No hold condition exists, so SWITCH."
            ),
            (
                f"RESOLVED hold status means the prior wait condition has cleared — HOLD is not the right verdict. "
                f"The competitor change that mattered is live: '{top_comp}' addresses '{issue}'. "
                f"Compliance PASSED. {roi_note} Reassessing on the merits now supports SWITCH."
            ),
            (
                f"Step 1: Hold status RESOLVED — the old blocker has cleared; HOLD is no longer valid. "
                f"Step 2: Re-evaluate the real decision. {comp} ships '{top_comp}', which addresses '{issue}'. "
                f"Step 3: Compliance PASSED. Step 4: {roi_note} The trace clears the ordinary switch test, so SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "compliance_newly_met":
        templates = [
            (
                f"Compliance gate: previously BLOCKED, now PASSED. "
                f"That removes the hard stop, but the competitor still needs to address a real push issue. "
                f"Here {comp} delivers '{top_comp}', which addresses '{issue}'. "
                f"ROI gate: {roi_str}. Hold signal: NONE. "
                f"All gates clear. SWITCH."
            ),
            (
                f"Compliance unblocked. The only previous barrier has been cleared. "
                f"That does not create a switch case by itself; it simply allows the real switch case to be evaluated. "
                f"'{top_comp}' from {comp} now addresses '{issue}'. ROI: {roi_str}. Hold: NONE. All gates now clear. SWITCH."
            ),
            (
                f"Compliance was the sole blocking gate. It is now cleared. "
                f"With the blocker removed, evaluate the remaining signals on their merits: "
                f"'{top_comp}' from {comp} directly addresses '{issue}' — a concrete pull. "
                f"ROI {roi_str}. Hold NONE. The switch case that compliance previously blocked "
                f"now goes through. SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "fluff_update":
        templates = [
            # V0 — vague pull = no basis
            (
                f"Pull signal: VAGUE — '{top_comp}' is marketing language with no concrete feature delivery. "
                f"VAGUE pull signals provide no basis for SWITCH regardless of push signal severity. "
                f"Notes about migration complexity or implementation risk are not HOLD conditions — "
                f"only contract locks, beta features, active pilots, or vendor acquisitions trigger HOLD. "
                f"Hold signal: NONE. No switch case, no hold case. STAY."
            ),
            # V1 — substance test
            (
                f"Pull signal fails the substance test: '{top_comp}' is vague. "
                f"No specific feature, integration, or capability is described — only marketing language. "
                f"VAGUE pull cannot justify SWITCH regardless of push severity. "
                f"Hold signal: NONE. STAY."
            ),
            # V2 — derive from what a valid pull signal requires
            (
                f"A SWITCH requires evidence that the competitor resolves the push driver. "
                f"'{top_comp}' provides no such evidence — it names no specific feature or "
                f"capability, only a vague improvement. Vague language cannot be evaluated "
                f"against a specific push problem. Without a verifiable resolution, the switch "
                f"case does not exist. Hold signal: NONE. STAY."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "irrelevant_change":
        hold_signal = _detect_hold_signal(notes, comp_ch)
        disqualifier = _detect_disqualifier(notes, hold_signal)
        if disqualifier != "NONE":
            templates = [
                # V0 — Disqualifier-driven reasoning
                (
                    f"Disqualifier: '{disqualifier}'. "
                    f"When a Disqualifier is present, the competitor fails a hard requirement — "
                    f"verdict is STAY regardless of pull signal strength or ROI. "
                    f"{comp}'s '{top_comp}' is irrelevant to Meridian's needs. "
                    f"Hold signal: NONE. Disqualifier forces STAY."
                ),
                # V1 — relevance test + Disqualifier confirmation
                (
                    f"Relevance check: '{top_comp}' from {comp} does not address the push driver ({issue}). "
                    f"Disqualifier confirms: '{disqualifier}'. "
                    f"A Disqualifier means the competitor is not viable — this overrides any pull signals. "
                    f"Hold signal: NONE. STAY."
                ),
                # V2 — full reasoning chain
                (
                    f"The push driver is '{issue}'. {comp}'s update ('{top_comp}') targets a different domain "
                    f"and does not resolve this gap. "
                    f"Disqualifier field: '{disqualifier}' — the competitor fails a hard requirement. "
                    f"When a Disqualifier is set, verdict is STAY unconditionally. "
                    f"Hold signal: NONE. STAY."
                ),
            ]
        else:
            templates = [
                # V0 — missed target (fallback, no disqualifier detected)
                (
                    f"Pull signal: IRRELEVANT — {comp} delivers {top_comp}, which does not address {issue}. "
                    f"A change that misses the primary push signal is worthless as a pull. "
                    f"Hold signal: NONE. No switch case. STAY."
                ),
                # V1 — relevance test
                (
                    f"Relevance check fails: '{top_comp}' from {comp} does not address the push driver ({issue}). "
                    f"A feature improvement in an unrelated area cannot justify switching. "
                    f"Hold signal: NONE. STAY."
                ),
                # V2 — derive from relevance requirement
                (
                    f"The push driver is '{issue}'. A valid pull signal must address this specific gap. "
                    f"'{top_comp}' from {comp} does not — it is an improvement in a different area. "
                    f"A feature update that misses the primary pain point cannot justify a switch, "
                    f"regardless of how significant that update is in its own domain. "
                    f"Hold signal: NONE. STAY."
                ),
                # V3 — explicit alternatives ruled out
                (
                    f"'{top_comp}' from {comp} is a change in a different domain than '{issue}'. "
                    f"Not SWITCH: a valid pull must resolve the PRIMARY push driver. A feature improvement "
                    f"that misses the known gap cannot justify migration cost, regardless of its own quality. "
                    f"Not HOLD: the competitor is fully GA — there is no blocking condition waiting to clear. "
                    f"Hold signal: NONE. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "negative_signal_buried":
        _NSB_DISQUAL_KW = frozenset({"preview", "early access", "beta"})
        hold_note = next(
            (n for n in notes if any(kw in n.lower() for kw in _NSB_DISQUAL_KW)), None
        )
        disqualifier = hold_note if hold_note else top_note
        templates = [
            # V0 — Disqualifier field present; hold signal NONE
            (
                f"Disqualifier: {disqualifier} — pre-GA language in buried notes, not competitor changes. "
                f"Hold signal: NONE — no hold condition is present. "
                f"A Disqualifier means the competitor is not viable: this is not a hold condition. "
                f"HOLD requires a concrete blocking lock (contract, acquisition, active pilot). "
                f"The buried note kills the pull case. STAY."
            ),
            # V1 — positional contrast via Disqualifier label
            (
                f"Surface reading: pull signal looks CONCRETE, suggests SWITCH. "
                f"Disqualifier field: '{disqualifier}' — found in buried notes, not competitor changes. "
                f"Pre-GA language in notes = Disqualifier (competitor not viable) → STAY. "
                f"Pre-GA language in competitor changes = Hold signal (competitor nearly ready) → HOLD. "
                f"Hold signal: NONE. Disqualifier forces STAY."
            ),
            # V2 — derive from Disqualifier semantics
            (
                f"The Disqualifier field identifies: '{disqualifier}' in buried notes. "
                f"Hold signal: NONE — no temporal gate is blocking the decision. "
                f"A Disqualifier is not a hold condition: it means the competitor itself is not ready, "
                f"not that we should wait for a named condition to clear. "
                f"Surface signals are overridden by the Disqualifier. STAY."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "current_tool_rally":
        rally = rally_tool if relevant_tool_positive is not None else top_tool
        incumbent_state = "actively improving" if relevant_tool_positive is not None else "operationally steady"
        templates = [
            (
                f"Push signal: WEAKENING — {tool} is {incumbent_state} ({rally}). "
                f"The urgency to switch is diminishing. {comp}'s changes ({top_comp}) do not outpace "
                f"the incumbent's recovery. Hold signal: NONE. Insufficient case for SWITCH or HOLD. STAY."
            ),
            (
                f"The incumbent is steadying: {tool} shows {rally}. "
                f"When the current tool actively resolves its own issues, the urgency to switch drops. "
                f"{comp}'s '{top_comp}' no longer provides a decisive advantage. STAY."
            ),
            (
                f"{tool} is {incumbent_state}: {rally}. The push signal weakens when the "
                f"incumbent is recovering or clearly stable. A switch requires the competitor to be "
                f"decisively better — when the current tool is closing the gap, that threshold "
                f"is harder to clear. {comp}'s '{top_comp}' does not create a sufficient delta "
                f"over an improving incumbent. Hold signal: NONE. STAY."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "competitor_nearly_ready":
        blocking = next(
            (n for n in notes if not n.lower().startswith(_ADVISORY_PREFIXES)
             and any(kw in n.lower() for kw in _HOLD_NOTE_KW)),
            None,
        )
        if blocking is None:
            blocking = next((c for c in comp_ch if any(kw in c.lower() for kw in _HOLD_COMP_KW)), top_note)
        has_shelfware = any(
            "shelfware" in s.lower() or "inactive" in s.lower()
            for s in tool_ch
        )
        roi_note = f"ROI assessment: {roi_str} — but ROI is irrelevant while the hold gate is active." if not met else f"ROI: {roi_str} — but holding regardless because the feature is not GA."
        templates = [
            # V0 — hold signal active from competitor changes
            (
                f"Hold signal: {blocking} — found in competitor changes, not buried notes. "
                f"Pre-GA language in competitor changes means the competitor is nearly ready, "
                f"not that it should be rejected. "
                f"The feature is not GA — it cannot be relied upon in production yet. "
                f"{roi_note} "
                f"HOLD — reassess when the feature ships GA."
            ),
            # V1 — positional contrast: competitor changes vs notes
            (
                f"Hold signal active: '{blocking}' — in competitor changes, not buried notes. "
                f"Pre-GA language in competitor changes = competitor nearly ready → HOLD (wait for GA). "
                f"This is different from pre-GA in buried notes, which would be a disqualifier → STAY. "
                f"Even a CONCRETE pull signal is irrelevant until the feature ships GA. "
                f"{roi_note} HOLD."
            ),
            # V2 — derive from GA requirement and positional reasoning
            (
                f"The blocking condition ('{blocking}') is in competitor changes — "
                f"the competitor is nearly ready, not absent. "
                f"Committing to a pre-GA feature risks relying on capability that may ship late or differently. "
                f"Hold signal active from competitor changes. {roi_note} "
                f"The prudent gate is to suspend the decision until GA is confirmed. "
                f"HOLD — reassess at GA confirmation."
            ),
            # V3 — competing signals (shelfware, advisory notes) do not override the hold gate
            (
                (
                    f"Shelfware pressure confirms motivation to move. "
                    if has_shelfware else
                    f"Genuine pressure signals are present. "
                ) +
                f"However, the binding constraint is the Hold signal: '{blocking}' — "
                f"in competitor changes, meaning the feature is pre-GA. "
                f"Pre-GA language in competitor changes = competitor nearly ready → HOLD, "
                f"not STAY and not SWITCH. {roi_note} "
                f"Advisory notes or cost pressure do not override the hold gate. "
                f"HOLD — reassess when the feature ships GA."
            ),
            # V4 — explicit alternatives ruled out
            (
                f"Hold signal active: '{blocking}'. The feature is pre-GA — delivery is not yet confirmed. "
                f"Not SWITCH: committing before GA confirmation means relying on a feature that may ship "
                f"late, incompletely, or not at all. The switch case requires CONFIRMED, releasable delivery. "
                f"Not STAY: this is not a poor-fit rejection — the competitor IS relevant and nearly ready. "
                f"STAY would imply the competitor is unsuitable; HOLD correctly captures 'suitable but not yet'. "
                f"{roi_note} HOLD — reassess when the feature ships GA."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "roadmap_confirmed_hold":
        roi_note = f"ROI assessment: {roi_str} — but ROI is irrelevant while the hold gate is active." if not met else f"ROI: {roi_str} — but holding regardless because the roadmap item has not shipped."
        templates = [
            # V0 — promise not delivery
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"Roadmap items are promises, not delivered capability — this gate is NOT cleared. "
                f"Pull signal quality is irrelevant until the feature ships. {roi_note} "
                f"Hold signal active. HOLD — reassess only when delivered."
            ),
            # V1 — promise vs delivery
            (
                f"HOLD. Blocking condition: {top_note}. "
                f"A roadmap commitment is not delivered capability. "
                f"The gate is not cleared until the feature ships. {roi_note} "
                f"Pull signals are irrelevant until delivery is confirmed. HOLD."
            ),
            # V2 — derive from the difference between roadmap and shipped capability
            (
                f"The blocking condition is '{top_note}'. A roadmap item is a future intention. "
                f"Switching on the basis of undelivered capability introduces real risk: the "
                f"feature may ship late, incompletely, or not at all. {roi_note} The correct approach is to "
                f"wait for confirmed delivery before committing. HOLD — reassess when the feature ships."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "contract_renewal_hold":
        roi_note = f"ROI assessment: {roi_str} — but ROI is irrelevant while the hold gate is active." if not met else f"ROI: {roi_str} — but holding regardless due to contract timing."
        templates = [
            # V0 — commercial gate
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"Contract terms prevent switching now without penalty. "
                f"The commercial gate is not cleared regardless of signal strength or ROI. {roi_note} "
                f"Hold signal active. HOLD — reassess at renewal."
            ),
            # V1 — penalty framing
            (
                f"HOLD. Commercial gate blocked: {top_note}. "
                f"The contract has not expired — switching now incurs penalties. "
                f"Competitor signal strength and ROI cannot override a hard commercial block. {roi_note} "
                f"HOLD until the renewal window. HOLD."
            ),
            # V2 — derive from commercial constraint logic
            (
                f"The contract '{top_note}' creates a hard commercial constraint. "
                f"Switching before the contract expires incurs penalties that the ROI calculation "
                f"does not account for. {roi_note} The signal strength is irrelevant — the commercial gate "
                f"overrides the technical decision until the renewal window opens. "
                f"HOLD — reassess at contract renewal."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "vendor_acquisition_hold":
        roi_note = f"ROI assessment: {roi_str} — but ROI is irrelevant while the hold gate is active." if not met else f"ROI: {roi_str} — but holding regardless due to acquisition uncertainty."
        templates = [
            # V0 — uncertainty
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"A vendor acquisition creates uncertainty — "
                f"pull signal strength and ROI are irrelevant when the vendor's future is unknown. "
                f"Even CONCRETE pull signals cannot override an active acquisition hold. {roi_note} "
                f"Hold signal active. HOLD — reassess once transition stabilises."
            ),
            # V1 — risk framing
            (
                f"HOLD. Acquisition uncertainty: {top_note}. "
                f"Vendor acquisitions introduce product, roadmap, and support unknowns. "
                f"Committing to an acquired competitor mid-transition is high risk. {roi_note} "
                f"HOLD until transition stabilises. HOLD."
            ),
            # V2 — derive from acquisition uncertainty logic
            (
                f"The acquisition '{top_note}' makes the competitor an unstable landing zone. "
                f"Product strategy, pricing, support, and roadmap are all subject to change "
                f"during a transition. Switching mid-acquisition means committing to a target "
                f"that may look different in six months. {roi_note} Even strong pull signals cannot overcome "
                f"this structural uncertainty. HOLD until the transition stabilises."
            ),
            # V3 — decision framework reasoning
            (
                f"Step 1: Check blocking gates. Hold signal: '{top_note}' — ACTIVE. "
                f"An active hold signal takes precedence over all other signal analysis including ROI. {roi_note} "
                f"Step 2: Hold type: vendor acquisition — roadmap, pricing, and support "
                f"stability are all unknown during a transition. "
                f"Step 3: The hold gate is binding. Pull, push, and ROI are not evaluated "
                f"until the hold condition clears. "
                f"HOLD — reassess when acquisition terms are finalised."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "pilot_in_progress_hold":
        roi_note = f"ROI assessment: {roi_str} — but ROI is irrelevant while the hold gate is active." if not met else f"ROI: {roi_str} — but holding regardless until pilot concludes."
        templates = [
            # V0 — incomplete evidence
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"A pilot is in progress — the switch case has not yet been validated. "
                f"Acting on incomplete evidence skips due diligence. {roi_note} "
                f"Hold signal active. HOLD — reassess after pilot results."
            ),
            # V1 — due diligence framing
            (
                f"HOLD. Active pilot: {top_note}. "
                f"The evaluation is ongoing — pilot results have not been reviewed. "
                f"Committing before the pilot concludes defeats the purpose of the evaluation. {roi_note} "
                f"HOLD until pilot data is available. HOLD."
            ),
            # V2 — derive from due diligence logic
            (
                f"A pilot is underway: {top_note}. The pilot exists precisely to validate "
                f"whether the switch is the right decision. Acting before it concludes means "
                f"ignoring the evidence the organisation is in the process of collecting. {roi_note} "
                f"The rational position is to wait for that evidence. HOLD — reassess after pilot results."
            ),
        ]
        return templates[variant % len(templates)]

    # Ambiguous scenarios — two variants (shorter, less critical to vary heavily)
    if scenario == "both_signals":
        if _supports_switch_case(context, signal, roi) and view.positive_pull is not None:
            templates = [
                (
                    f"Both push and pull signals are active. {comp} delivers {top_comp}, "
                    f"addressing {issue}. {roi_str.capitalize()}. "
                    f"When pull resolves the primary push driver and ROI is met, the switch case stands. "
                    f"Pull dominates — SWITCH."
                ),
                (
                    f"Push and pull both present. Pull dominates: '{top_comp}' from {comp} resolves '{issue}'. "
                    f"ROI met. All gates clear — the concrete pull addressing the primary push "
                    f"issue justifies moving. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Both push and pull signals are present but pull is insufficient. "
                    f"{comp} shows {top_comp}, but it does not clearly cover the decisive issue '{issue}', and {roi_str}. "
                    f"A pull signal that does not clearly address the primary push driver or clear the ROI gate "
                    f"cannot justify SWITCH. No clear dominant signal — STAY."
                ),
                (
                    f"Both signals active but the competitor delta is still weak. '{top_comp}' does not overcome "
                    f"{roi_str}. When push and pull are both present but neither dominates, "
                    f"the burden of proof for SWITCH is not met. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "price_hike_only":
        if met:
            templates = [
                (
                    f"Current tool pricing has increased and {roi_str}. "
                    f"The cost delta alone justifies SWITCH — when migration saves money and "
                    f"the competitor is at least comparable, the economics are sufficient. SWITCH."
                ),
                (
                    f"Price increase triggers SWITCH: {roi_str}. "
                    f"A cost-driven switch does not require a strong pull signal — the savings "
                    f"alone justify the move when no blocking gates are active. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Current tool pricing has increased, but {roi_str}. "
                    f"A price change alone cannot justify SWITCH when the cost delta is negative or negligible. "
                    f"Without a concrete pull signal resolving a push issue, the economics do not support moving. "
                    f"Hold signal: NONE. No blocking gate, but no switch case either. STAY."
                ),
                (
                    f"Price hike present but {roi_str}. "
                    f"ROI gate not cleared — the cost savings are insufficient to justify the migration "
                    f"effort and risk. A SWITCH requires either ROI threshold met OR a strong pull signal "
                    f"addressing a critical push issue. Neither is present here. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "dual_improvement":
        if _supports_switch_case(context, signal, roi) and view.positive_pull is not None and relevant_tool_positive is not None:
            templates = [
                (
                    f"Both tools improved this period. The delta favours {comp}: {top_comp} "
                    f"addresses {issue} more directly than {tool}'s {rally_tool}. "
                    f"{roi_str.capitalize()}. When both improve but the competitor's improvement "
                    f"resolves the primary push driver, the switch case is established. SWITCH."
                ),
                (
                    f"Both improved; competitor edge is decisive: '{top_comp}' resolves '{issue}'. "
                    f"{roi_str}. The competitor's improvement addresses the push driver while "
                    f"the incumbent's improvement ('{rally_tool}') does not close the gap enough. SWITCH."
                ),
            ]
        else:
            incumbent_clause = (
                f"{tool} improves with {rally_tool}"
                if relevant_tool_positive is not None
                else f"{tool} does not materially improve on the decisive gap"
            )
            templates = [
                (
                    f"{incumbent_clause} while "
                    f"{comp} delivers {top_comp}. {roi_str}. When both tools improve, a SWITCH "
                    f"requires the competitor to be decisively better — but neither the pull signal "
                    f"nor the ROI case creates a decisive advantage. STAY."
                ),
                (
                    f"Mutual-improvement test: {incumbent_clause}, while {comp} ships {top_comp}. "
                    f"Neither side establishes a decisive advantage. {roi_str}. Without a clear "
                    f"delta in the competitor's favour, switching during mutual improvement is premature. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "hard_compliance_failure":
        cat = context["category"]
        block_detail = _effective_compliance_status(context, signal)
        block_reason = block_detail[len("BLOCKED — "):] if block_detail.startswith("BLOCKED") else cc
        templates = [
            (
                f"The competitor cannot be adopted: {block_reason}. These are hard requirements for "
                f"{cat} tools — they are non-negotiable. No business signal (price, features, ROI) "
                f"overrides a compliance block. STAY — do not re-evaluate until compliance gaps are resolved."
            ),
            (
                f"Hard compliance block: {block_reason}. A {cat} tool must meet these requirements "
                f"before any business evaluation begins. The pull signals and ROI are irrelevant until "
                f"the compliance gate clears. STAY."
            ),
            (
                f"Compliance check fails first. The competitor is blocked on: {block_reason}. "
                f"These are mandatory for {cat}. Until formally resolved, the switch case does not exist. STAY."
            ),
        ]
        return templates[variant % len(templates)]

    return f"Signal evaluated for {comp} vs {tool}. {roi_str.capitalize()}."


# ── Ambiguous verdict inference ────────────────────────────────────────────────

def _infer_ambiguous_verdict(scenario: str, roi: dict, signal: dict, context: dict) -> str:
    if scenario == "price_hike_only":
        return "SWITCH" if roi["roi_threshold_met"] else "STAY"

    if scenario == "dual_improvement" and _semantic_view(context, signal).positive_tool is None:
        return "STAY"

    return "SWITCH" if _supports_switch_case(context, signal, roi) else "STAY"


# ── Full CoT trace assembly ───────────────────────────────────────────────────

def _build_cot_trace(scenario: str, signal: dict, context: dict, roi: dict, verdict: str, variant: int = 0) -> str:
    push = _push_section(context, signal)
    pull = _pull_section(signal)
    comp_line = _compliance_cot_line(context, signal)
    roi_line = _roi_line(roi)
    hold_line = _hold_line(signal, scenario)
    analysis_text = _normalize_training_text(_analysis(scenario, signal, context, roi, variant=variant))
    analysis_text = f"{analysis_text} {_boundary_rejections(verdict, scenario, signal, context, roi)}".strip()

    return (
        f"{push}\n\n"
        f"{pull}\n\n"
        f"{comp_line}\n"
        f"{roi_line}\n\n"
        f"{hold_line}\n\n"
        f"ANALYSIS: {analysis_text}\n"
        f"VERDICT: {verdict}"
    )


_POSITIVE_TOOL_TREND_KW = frozenset([
    "improved", "faster", "resolved", "available", "live", "launched", "shipped",
    "stable", "flexible", "covers", "white-labelling", "white-label", "ga",
])

_NEGATIVE_TOOL_TREND_KW = frozenset([
    "still", "absent", "manual", "delay", "slow", "outdated", "deprecated",
    "requires", "risk", "flagged", "crash", "error", "fails", "failing",
    "bottleneck", "non-functional", "unstable", "missing", "inadequate",
])

_IRRELEVANCE_KW = frozenset([
    "irrelevant", "unrelated", "no relevance", "poor fit", "wrong product",
    "does not address", "does not align", "not designed for", "no impact on",
])

_COMPLIANCE_PUSH_KW = frozenset([
    "audit", "compliance", "ifrs", "soc2", "soc 2", "sso", "saml",
    "gdpr", "residency", "hipaa", "pci", "iso",
])

_INTEGRATION_PUSH_KW = frozenset([
    "api", "sync", "integration", "connector", "barclays", "bank feed",
    "real-time", "real time", "middleware",
])

_COST_PUSH_KW = frozenset([
    "pricing", "price", "cost", "seat", "shelfware", "inactive", "unused capacity",
])

_OPERATIONAL_PUSH_KW = frozenset([
    "manual", "slow", "delay", "outdated", "crash", "error", "503",
    "deprecated", "workaround", "spreadsheet", "bottleneck", "non-functional",
    "unstable", "requires export", "requires manual",
])


def _roi_bucket(roi: dict) -> str:
    annual_net = roi["annual_net_gbp"]
    if roi["roi_threshold_met"]:
        return "met"
    if annual_net >= 0:
        return "positive_below_threshold"
    return "negative"


def _tool_trend(tool_changes: list[str]) -> str:
    active_changes = [
        change for change in tool_changes
        if change.lower().strip() not in _NO_CHANGE_MARKERS
    ]
    if not active_changes:
        return "stable"
    if _detect_shelfware(active_changes) != "NONE":
        return "shelfware"

    text = " ".join(active_changes).lower()
    pos_hits = sum(1 for kw in _POSITIVE_TOOL_TREND_KW if kw in text)
    neg_hits = sum(1 for kw in _NEGATIVE_TOOL_TREND_KW if kw in text)

    if pos_hits and not neg_hits:
        return "rally"
    if neg_hits and not pos_hits:
        return "degrading"
    if pos_hits and neg_hits:
        return "mixed"
    return "stable"


def _push_profile(context: dict, signal: dict) -> str:
    tool_changes = signal_current_tool_status(signal)
    known_issues = context["current_stack_entry"].get("known_issues", [])
    if _detect_shelfware(tool_changes) != "NONE":
        return "shelfware"

    text = " ".join([*tool_changes, *known_issues]).lower()
    if any(kw in text for kw in _COMPLIANCE_PUSH_KW):
        return "compliance_gap"
    if any(kw in text for kw in _INTEGRATION_PUSH_KW):
        return "integration_gap"
    if any(kw in text for kw in _COST_PUSH_KW):
        return "cost_pressure"
    if any(kw in text for kw in _OPERATIONAL_PUSH_KW):
        return "operational_friction"
    return "general_gap"


def _pull_profile(signal: dict, hold_signal: str, disqualifier: str) -> str:
    if disqualifier != "NONE":
        return "disqualified"
    if hold_signal != "NONE":
        return "blocked_pending"

    changes = signal_competitor_changes(signal)
    notes_text = " ".join(signal_notes(signal)).lower()
    if any(kw in notes_text for kw in _IRRELEVANCE_KW):
        return "irrelevant"
    if not changes:
        return "none"

    substances = {_substance(change) for change in changes[:3]}
    if substances == {"CONCRETE"}:
        return "concrete"
    if substances == {"VAGUE"}:
        return "vague"
    return "mixed"


def _gate_state(signal: dict, scenario: str, hold_signal: str, disqualifier: str, shelfware_signal: str) -> str:
    if disqualifier != "NONE":
        return "disqualifier"
    if hold_signal != "NONE":
        return "hold_active"
    if signal.get("previous_verdict") == "HOLD" and scenario == "hold_resolved":
        return "hold_resolved"
    if shelfware_signal != "NONE":
        return "shelfware_override"
    cc = _normalize_training_text(signal_compliance_changes(signal))
    has_positive_compliance = (
        cc
        and cc.lower().strip() not in ("unchanged", "no change", "none", "")
        and any(kw in cc.lower() for kw in _COMPLIANCE_POSITIVE_KW)
        and any(ht in cc.lower() for ht in _COMPLIANCE_HARD_TERMS)
    )
    if has_positive_compliance:
        return "compliance_unblocked"
    return "open"


def _build_trace_metadata(
    signal_path: Path,
    category: str,
    competitor_slug: str,
    scenario: str,
    context: dict,
    signal: dict,
    roi: dict,
    verdict: str,
    variant_index: int,
) -> dict:
    comp_changes = signal_competitor_changes(signal)
    tool_changes = signal_current_tool_status(signal)
    notes = signal_notes(signal)
    hold_signal = _detect_hold_signal(notes, comp_changes)
    disqualifier = _detect_disqualifier(notes, hold_signal)
    shelfware_signal = _detect_shelfware(tool_changes)
    gate_state = _gate_state(signal, scenario, hold_signal, disqualifier, shelfware_signal)
    push_profile = _push_profile(context, signal)
    pull_profile = _pull_profile(signal, hold_signal, disqualifier)
    roi_bucket = _roi_bucket(roi)
    tool_trend = _tool_trend(tool_changes)
    semantic_cell = "|".join([
        scenario,
        push_profile,
        pull_profile,
        gate_state,
        roi_bucket,
        tool_trend,
    ])

    return {
        "source_file": signal_path.name,
        "source_group": signal_path.stem,
        "category": category,
        "competitor_slug": competitor_slug,
        "scenario": scenario,
        "verdict": verdict,
        "variant_index": variant_index,
        "push_profile": push_profile,
        "pull_profile": pull_profile,
        "gate_state": gate_state,
        "roi_bucket": roi_bucket,
        "tool_trend": tool_trend,
        "semantic_cell": semantic_cell,
        "balance_cell": f"{verdict}:{category}",
    }


def _counter_to_dict(counter: Counter, top_n: int | None = None) -> dict[str, int]:
    items = counter.most_common(top_n) if top_n is not None else sorted(counter.items(), key=lambda item: str(item[0]))
    return {str(key): value for key, value in items}


def _dataset_stats(examples: list[dict]) -> dict:
    metadata = [example["metadata"] for example in examples]
    return {
        "total_traces": len(examples),
        "unique_sources": len({meta["source_group"] for meta in metadata}),
        "unique_competitors": len({meta["competitor_slug"] for meta in metadata}),
        "by_verdict": _counter_to_dict(Counter(meta["verdict"] for meta in metadata)),
        "by_category": _counter_to_dict(Counter(meta["category"] for meta in metadata)),
        "by_scenario": _counter_to_dict(Counter(meta["scenario"] for meta in metadata)),
        "by_balance_cell": _counter_to_dict(Counter(meta["balance_cell"] for meta in metadata)),
        "by_gate_state": _counter_to_dict(Counter(meta["gate_state"] for meta in metadata)),
        "by_push_profile": _counter_to_dict(Counter(meta["push_profile"] for meta in metadata)),
        "by_pull_profile": _counter_to_dict(Counter(meta["pull_profile"] for meta in metadata)),
        "by_roi_bucket": _counter_to_dict(Counter(meta["roi_bucket"] for meta in metadata)),
        "by_tool_trend": _counter_to_dict(Counter(meta["tool_trend"] for meta in metadata)),
        "by_variant_index": _counter_to_dict(Counter(meta["variant_index"] for meta in metadata)),
        "top_semantic_cells": _counter_to_dict(Counter(meta["semantic_cell"] for meta in metadata), top_n=20),
        "top_source_reuse": _counter_to_dict(Counter(meta["source_group"] for meta in metadata), top_n=20),
    }


def _select_diverse_examples(cell_examples: list[dict], target: int, rng: random.Random) -> list[dict]:
    pool = list(cell_examples)
    rng.shuffle(pool)

    source_counts: Counter = Counter()
    category_counts: Counter = Counter()
    competitor_counts: Counter = Counter()
    scenario_counts: Counter = Counter()
    semantic_counts: Counter = Counter()
    variant_counts: Counter = Counter()
    selected: list[dict] = []
    used_indexes: set[int] = set()

    while len(selected) < target and len(used_indexes) < len(pool):
        best_index: int | None = None
        best_score: tuple[int, int, int, int, int] | None = None

        for index, example in enumerate(pool):
            if index in used_indexes:
                continue
            meta = example["metadata"]
            score = (
                source_counts[meta["source_group"]],
                category_counts[meta["category"]],
                competitor_counts[meta["competitor_slug"]],
                scenario_counts[meta["scenario"]],
                semantic_counts[meta["semantic_cell"]],
                variant_counts[meta["variant_index"]],
            )
            if best_score is None or score < best_score:
                best_index = index
                best_score = score

        if best_index is None:
            break

        example = pool[best_index]
        meta = example["metadata"]
        selected.append(example)
        used_indexes.add(best_index)
        source_counts[meta["source_group"]] += 1
        category_counts[meta["category"]] += 1
        competitor_counts[meta["competitor_slug"]] += 1
        scenario_counts[meta["scenario"]] += 1
        semantic_counts[meta["semantic_cell"]] += 1
        variant_counts[meta["variant_index"]] += 1

    return selected


def _balance_examples(
    examples: list[dict],
    *,
    granularity: str = "verdict",
) -> tuple[list[dict], dict]:
    rng = random.Random(99)

    if granularity == "cell":
        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for example in examples:
            meta = example["metadata"]
            cells[(meta["verdict"], meta["category"])] .append(example)

        raw_cell_counts = {
            f"{verdict}:{category}": len(items)
            for (verdict, category), items in sorted(cells.items())
        }
        min_cell = min((len(items) for items in cells.values()), default=0)
        balanced: list[dict] = []

        for key in sorted(cells.keys()):
            balanced.extend(_select_diverse_examples(cells[key], min_cell, rng))

        rng.shuffle(balanced)
        balanced_verdict_counts = _counter_to_dict(
            Counter(example["metadata"]["verdict"] for example in balanced)
        )
        info = {
            "granularity": granularity,
            "raw_group_counts": raw_cell_counts,
            "target_per_group": min_cell,
            "group_count": len(cells),
            "total_after_balance": len(balanced),
            "balanced_verdict_counts": balanced_verdict_counts,
        }
        return balanced, info

    if granularity == "verdict":
        verdict_groups: dict[str, list[dict]] = defaultdict(list)
        for example in examples:
            meta = example["metadata"]
            verdict_groups[meta["verdict"]].append(example)

        raw_group_counts = {
            verdict: len(items)
            for verdict, items in sorted(verdict_groups.items())
        }
        min_verdict = min((len(items) for items in verdict_groups.values()), default=0)
        balanced: list[dict] = []

        for verdict in sorted(verdict_groups.keys()):
            balanced.extend(_select_diverse_examples(verdict_groups[verdict], min_verdict, rng))

        rng.shuffle(balanced)
        balanced_verdict_counts = _counter_to_dict(
            Counter(example["metadata"]["verdict"] for example in balanced)
        )
        info = {
            "granularity": granularity,
            "raw_group_counts": raw_group_counts,
            "target_per_group": min_verdict,
            "group_count": len(verdict_groups),
            "total_after_balance": len(balanced),
            "balanced_verdict_counts": balanced_verdict_counts,
        }
        return balanced, info

    raise ValueError(f"Unsupported balance granularity: {granularity}")


def _default_split_paths(output_path: str) -> tuple[Path, Path]:
    output = Path(output_path)
    train_path = output.with_name(f"{output.stem}.train{output.suffix}")
    val_path = output.with_name(f"{output.stem}.val{output.suffix}")
    return train_path, val_path


def _split_examples(examples: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    if val_ratio <= 0:
        return list(examples), []

    buckets: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        meta = example["metadata"]
        bucket_key = (meta["verdict"], meta["category"], meta["scenario"])
        buckets[bucket_key][meta["source_group"]].append(example)

    rng = random.Random(123)
    train_examples: list[dict] = []
    val_examples: list[dict] = []

    for bucket_key in sorted(buckets.keys()):
        groups = list(buckets[bucket_key].values())
        rng.shuffle(groups)
        if len(groups) < 2:
            train_examples.extend(group for source_group in groups for group in source_group)
            continue

        val_groups = round(len(groups) * val_ratio)
        if val_ratio > 0 and val_groups == 0 and len(groups) >= 3:
            val_groups = 1
        val_groups = min(val_groups, len(groups) - 1)

        for index, group in enumerate(groups):
            if index < val_groups:
                val_examples.extend(group)
            else:
                train_examples.extend(group)

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example["record"], ensure_ascii=False) + "\n")


# ── File sampling ─────────────────────────────────────────────────────────────

def _collect_samples(
    limit: int,
    *,
    include_drift_canaries: bool = False,
) -> list[tuple[Path, str, str, str]]:
    by_scenario: dict[str, list[Path]] = {s: [] for s in SCENARIO_TYPES}

    for path in _GENERATED_DIR.glob("*.json"):
        if not include_drift_canaries and path.name in DRIFT_CANARY_FILENAMES:
            continue
        parsed = _parse_filename(path.stem)
        if parsed is None:
            continue
        _, _, scenario = parsed
        if scenario in by_scenario:
            by_scenario[scenario].append(path)

    selected: list[tuple[Path, str, str, str]] = []
    rng = random.Random(42)

    for scenario in SCENARIO_TYPES:
        if scenario in _SKIP_SCENARIOS:
            continue
        pool = sorted(by_scenario.get(scenario, []))
        rng.shuffle(pool)
        for path in pool[:limit]:
            parsed = _parse_filename(path.stem)
            if parsed:
                cat, slug, sc = parsed
                selected.append((path, cat, slug, sc))

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT SFT training traces.")
    parser.add_argument("--limit", type=int, default=8,
                        help="Signal files to sample per scenario type (default: 8).")
    parser.add_argument("--variants", type=int, default=3,
                        help="ANALYSIS variants to generate per signal file (default: 3).")
    parser.add_argument("--output", default=str(_OUTPUT_PATH),
                        help="Output JSONL path.")
    parser.add_argument("--balance", action="store_true",
                        help="Downsample with diversity-aware selection.")
    parser.add_argument("--balance-granularity", choices=["verdict", "cell"], default="verdict",
                        help="Balancing scope when --balance is set. 'verdict' preserves more coverage; 'cell' equalises exact verdict/category cells.")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="Optional validation split ratio grouped by source signal within each scenario bucket.")
    parser.add_argument("--train-output", default=None,
                        help="Optional train split JSONL path. Defaults to '<output>.train.jsonl' when splitting.")
    parser.add_argument("--val-output", default=None,
                        help="Optional validation split JSONL path. Defaults to '<output>.val.jsonl' when splitting.")
    parser.add_argument("--stats-output", default=None,
                        help="Optional JSON path for dataset audit stats.")
    parser.add_argument("--include-drift-canaries", action="store_true",
                        help="Include drift-check canary fixtures in SFT sampling. Disabled by default to keep eval fixtures held out.")
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in the range [0.0, 1.0).")

    samples = _collect_samples(
        args.limit,
        include_drift_canaries=args.include_drift_canaries,
    )
    total = len(samples) * args.variants
    print(f"Processing {len(samples)} signal files × {args.variants} variants = {total} traces "
          f"({args.limit} per scenario, {len(SCENARIO_TYPES) - len(_SKIP_SCENARIOS)} scenarios)...")

    examples: list[dict] = []
    skipped = 0

    for signal_path, category, competitor_slug, scenario in samples:
        try:
            context = load_context(category, competitor_slug, _DATA_ROOT)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"  SKIP {signal_path.name}: {exc}")
            skipped += 1
            continue

        with signal_path.open(encoding="utf-8") as f:
            signal = json.load(f)

        try:
            pass1 = extract_pass1_vars(context, signal)
            roi = calculate_roi(pass1)
        except Exception as exc:
            print(f"  SKIP {signal_path.name} (ROI error): {exc}")
            skipped += 1
            continue

        expected = EXPECTED_VERDICT.get(scenario)
        verdict = expected if expected is not None else _infer_ambiguous_verdict(scenario, roi, signal, context)

        failure_reason = _consistency_failure_reason(scenario, signal, context, roi, verdict)
        if failure_reason is not None:
            print(f"  SKIP {signal_path.name}: {failure_reason}")
            skipped += 1
            continue

        user_msg = _build_user_message(context, roi, signal, scenario=scenario)
        for v in range(args.variants):
            cot_trace = _build_cot_trace(scenario, signal, context, roi, verdict, variant=v)
            examples.append({
                "record": {
                    "messages": [
                        {"role": "system", "content": SYS_VERDICT_LEAN},
                        {"role": "user",   "content": user_msg},
                        {"role": "assistant", "content": cot_trace},
                    ]
                },
                "metadata": _build_trace_metadata(
                    signal_path,
                    category,
                    competitor_slug,
                    scenario,
                    context,
                    signal,
                    roi,
                    verdict,
                    variant_index=v,
                ),
            })

    raw_stats = _dataset_stats(examples)

    if args.balance:
        examples, balance_info = _balance_examples(examples, granularity=args.balance_granularity)
        print(f"Pre-balance group counts ({args.balance_granularity}): {balance_info['raw_group_counts']}")
        print(f"Target per group: {balance_info['target_per_group']}")
        print(
            f"Balance ({args.balance_granularity}): {balance_info['target_per_group']} per group × "
            f"{balance_info['group_count']} groups = {balance_info['total_after_balance']} total "
            f"({balance_info['balanced_verdict_counts']} by verdict)"
        )
    else:
        balance_info = None

    out = Path(args.output)
    _write_jsonl(out, examples)
    print(f"Wrote {len(examples)} traces → {out}  ({skipped} skipped)")

    effective_val_ratio = args.val_ratio
    if effective_val_ratio <= 0 and (args.train_output or args.val_output):
        effective_val_ratio = 0.15
        print("Validation ratio not provided; defaulting to 0.15 because split outputs were requested.")

    train_examples: list[dict] = []
    val_examples: list[dict] = []
    if effective_val_ratio > 0:
        train_examples, val_examples = _split_examples(examples, effective_val_ratio)
        default_train, default_val = _default_split_paths(args.output)
        train_out = Path(args.train_output) if args.train_output else default_train
        val_out = Path(args.val_output) if args.val_output else default_val
        _write_jsonl(train_out, train_examples)
        _write_jsonl(val_out, val_examples)
        print(
            f"Wrote grouped split: {len(train_examples)} train / {len(val_examples)} val "
            f"→ {train_out}, {val_out}"
        )

    if args.stats_output:
        stats_payload = {
            "config": {
                "limit": args.limit,
                "variants": args.variants,
                "balance": args.balance,
                "balance_granularity": args.balance_granularity,
                "val_ratio": effective_val_ratio,
            },
            "skipped": skipped,
            "raw": raw_stats,
            "balanced": _dataset_stats(examples),
        }
        if balance_info is not None:
            stats_payload["balance_info"] = balance_info
        if train_examples or val_examples:
            stats_payload["train"] = _dataset_stats(train_examples)
            stats_payload["val"] = _dataset_stats(val_examples)
        stats_out = Path(args.stats_output)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        with stats_out.open("w", encoding="utf-8") as handle:
            json.dump(stats_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Wrote dataset stats → {stats_out}")

    if examples:
        print("\n─── Example trace (first record) ───────────────────────────────")
        ex = examples[0]["record"]["messages"]
        print(f"[USER]\n{ex[1]['content']}\n")
        print(f"[ASSISTANT]\n{ex[2]['content']}")
        print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
