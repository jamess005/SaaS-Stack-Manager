"""
Generate Chain-of-Thought SFT training traces for the SaaS Stack Manager model.

For each sampled signal file, builds a structured reasoning trace that walks
through push signals → pull signals → compliance → ROI → hold condition → verdict.
These traces teach the model HOW to reason rather than what pattern to match.

Usage:
    python training/generate_cot_traces.py              # 8 per scenario (~144 total)
    python training/generate_cot_traces.py --limit 5    # 5 per scenario (~90 total)
    python training/generate_cot_traces.py --limit 12   # 12 per scenario (~216 total)
"""

import argparse
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

# Excluded from training: Python compliance gate handles these, model is never called.
_SKIP_SCENARIOS = {"hard_compliance_failure"}

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
    r"shelfware\s*:\s*false|shelfware\s*:\s*no\b|shelfware\s*:\s*none\b",
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
        if any(kw in change.lower() for kw in _SHELFWARE_KW):
            if not _SHELFWARE_NEGATION.search(change):
                return change
    return "NONE"


# ── User message builder (matches _build_lean_user in model_runner.py) ────────

def _build_user_message(context: dict, roi_result: dict, signal: dict, scenario: str = "") -> str:
    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    issues = context["current_stack_entry"].get("known_issues", [])
    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none)"

    comp_changes = signal_competitor_changes(signal)
    tool_changes = signal_current_tool_status(signal)
    notes = signal_notes(signal)

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
    msg += "\nCompliance: PASSED"
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
    if prev_verdict:
        msg += f"\nPrevious verdict: {prev_verdict}"
    return msg


_NO_CHANGE_MARKERS = {
    "no change",
    "no change — existing issues persist.",
    "no change this period",
    "stable",
    "unchanged",
}

# ── CoT trace section builders ────────────────────────────────────────────────

def _push_section(context: dict, signal: dict) -> str:
    known = context["current_stack_entry"].get("known_issues", [])
    # Strip generic no-change placeholders — these are not push signals
    tool_changes = [
        c for c in signal_current_tool_status(signal)
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
    changes = signal_competitor_changes(signal)
    if not changes:
        return "PULL SIGNALS:\n  None identified"
    lines = [f"  - {c} — Substance: {_substance(c)}" for c in changes[:3]]
    return "PULL SIGNALS:\n" + "\n".join(lines)


_COMPLIANCE_HARD_TERMS = frozenset([
    "soc2", "soc 2", "sso", "saml", "oidc", "audit log", "audit trail",
    "uk residency", "eu residency", "gdpr residency", "data residency",
])


def _compliance_line(signal: dict) -> str:
    """Generate compliance status line for the CoT trace.

    Only reports BLOCKED for the four hard requirements (SOC2, SSO, residency,
    audit log).  Soft requirements like ISO27001 don't block the switch decision.
    """
    cc = signal_compliance_changes(signal)
    if not cc or cc.lower().strip() in ("unchanged", "no change", ""):
        return "COMPLIANCE: PASSED"
    cc_l = cc.lower()
    achieved = {
        "achieved", "certified", "added", "now available", "launched", "shipped",
        "compliant", "now meets", "now compliant", "met", "passed", "attained",
        "enabled", "live",
    }
    hard_blocked = {"not soc2", "no sso", "lacks sso", "no saml", "lacks saml",
                    "no uk", "no eu", "not gdpr", "no audit log", "lacks audit log",
                    "no audit trail", "lacks audit trail"}
    if any(kw in cc_l for kw in achieved) and any(ht in cc_l for ht in _COMPLIANCE_HARD_TERMS):
        return "COMPLIANCE: Previously BLOCKED — now MET"
    if any(kw in cc_l for kw in hard_blocked):
        return f"COMPLIANCE: BLOCKED — {cc}"
    return "COMPLIANCE: PASSED"


def _roi_line(roi: dict) -> str:
    met = "MET" if roi["roi_threshold_met"] else "NOT MET"
    return (
        f"ROI: Migration £{roi['migration_cost_one_time']:.0f} | "
        f"Annual net £{roi['annual_net_gbp']:+.0f} → Threshold {met}"
    )


def _hold_line(signal: dict, scenario: str) -> str:
    notes = signal_notes(signal)
    comp_ch = signal_competitor_changes(signal)
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
    known = context["current_stack_entry"].get("known_issues", [])
    comp_ch = signal_competitor_changes(signal)
    tool_ch = signal_current_tool_status(signal)
    notes = signal_notes(signal)
    cc = signal_compliance_changes(signal)

    issue = (known[0] if known else "existing tool limitations").replace("\n", " ")
    top_comp = (comp_ch[0] if comp_ch else "minor updates").replace("\n", " ")
    top_tool = (tool_ch[0] if tool_ch else "no new changes").replace("\n", " ")
    top_note = (notes[0] if notes else "").replace("\n", " ")
    net = roi["annual_net_gbp"]
    met = roi["roi_threshold_met"]
    roi_str = f"ROI threshold {'met' if met else 'not met'} at £{net:+.0f}/yr net"

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
        push = top_tool if tool_ch else issue
        pull_sub = _substance(top_comp)
        if pull_sub == "VAGUE":
            templates = [
                # V0 — push dominance explanation
                (
                    f"Push signal: HIGH — {tool} is critically degrading ({push}). "
                    f"Pull signal: VAGUE — {comp} delivers only '{top_comp}'. "
                    f"Push dominance: escalating failure rate justifies leaving a degrading tool "
                    f"even when pull signal is weak. ROI gate: {roi_str}. "
                    f"Compliance gate: PASSED. Hold signal: NONE. SWITCH."
                ),
                # V1 — urgency framing
                (
                    f"The tool is failing: {push}. At HIGH push severity the rule changes — "
                    f"a weak pull signal is sufficient when the incumbent is critically degrading. "
                    f"{comp}'s '{top_comp}' provides a viable exit path. "
                    f"Compliance: PASSED. Hold: NONE. ROI: {roi_str}. SWITCH."
                ),
                # V2 — derive from the severity principle
                (
                    f"{tool} is critically degrading: {push}. A tool in active failure cannot "
                    f"wait for a perfect pull signal. The pull signal ('{top_comp}') is vague, "
                    f"but vague pull is sufficient when push severity is HIGH — the priority is "
                    f"exiting the failing tool. All blocking gates clear: compliance PASSED, "
                    f"hold NONE, {roi_str}. SWITCH."
                ),
            ]
        else:
            templates = [
                # V0 — double signal
                (
                    f"Push signal: HIGH — {tool} is degrading ({push}). "
                    f"Pull signal: CONCRETE — {comp} resolves this with {top_comp}. "
                    f"ROI gate: {roi_str}. Compliance gate: PASSED. Hold signal: NONE. "
                    f"All gates clear. SWITCH."
                ),
                # V1 — aligned signals
                (
                    f"Both signals aligned: {tool} degrades ({push}) while {comp} delivers {top_comp}. "
                    f"Push and pull reinforce the same decision. Compliance PASSED. Hold NONE. "
                    f"ROI {roi_str}. SWITCH."
                ),
                # V2 — derive from both signal strengths
                (
                    f"{tool} is degrading: {push}. {comp} delivers a concrete resolution: '{top_comp}'. "
                    f"The ideal switch scenario — the push driver has a concrete answer from the competitor. "
                    f"Blocking gates: compliance PASSED, hold NONE, {roi_str}. SWITCH."
                ),
                # V3 — explicit alternatives ruled out
                (
                    f"{tool} is actively failing: {push}. This is ongoing operational harm, not a chronic gap. "
                    f"{comp} delivers '{top_comp}', a concrete capability that directly resolves this active failure. "
                    f"Not STAY: the cost of staying is ongoing operational damage — this cannot wait. "
                    f"Not HOLD: the competitor delivers a working GA solution; there is no condition to wait for. "
                    f"Gates: compliance PASSED, hold NONE, {roi_str}. SWITCH."
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
        templates = [
            # V0 — waste elimination
            (
                f"SWITCH CONFIRMED — shelfware case. {tool} is underutilised ({top_tool}). "
                f"Paying for unused capacity with no utilisation improvement justifies switching. "
                f"Shelfware waste elimination overrides the standard ROI threshold — "
                f"the cost of inaction is the ongoing waste. "
                f"Compliance: PASSED. Hold signal: NONE. "
                f"Switch driven by waste elimination, not by competitor features.{notes_clarification} SWITCH."
            ),
            # V1 — cost framing
            (
                f"Decision driver: waste, not competitor quality. {tool} is {top_tool}. "
                f"The organisation pays for idle capacity with no utilisation plan. "
                f"The cost of staying (continued waste) exceeds migration cost. "
                f"Compliance PASSED. Hold NONE.{notes_clarification} SWITCH."
            ),
            # V2 — cost of inaction reasoning
            (
                f"Shelfware case: {top_tool}. The cost of staying is not zero — it is the "
                f"ongoing waste of paying for unused capacity. When the cost of inaction "
                f"(continued waste) is certain and the cost of action (migration) is one-time, "
                f"the economics favour switching. The standard ROI threshold relaxes for shelfware. "
                f"Compliance PASSED.{notes_clarification} SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "hold_resolved":
        prev = signal.get("previous_verdict", "")
        prev_note = "Previous verdict: HOLD — " if prev == "HOLD" else ""
        if met:
            roi_note = f"ROI: {roi_str} — gate PASSED."
        else:
            roi_note = (
                f"ROI: {roi_str} — negative, but hold release is the binding gate; "
                f"ROI does NOT block when a prior HOLD is cleared."
            )
        templates = [
            # V0 — blocking condition resolved
            (
                f"{prev_note}Hold condition resolved — {comp} delivers '{top_comp}', "
                f"removing the blocking condition. "
                f"Hold signal: NONE (no new hold condition detected). "
                f"Compliance: PASSED. {roi_note} "
                f"Prior verdict was HOLD because a specific condition blocked the switch. "
                f"That condition is now gone. With no hold signal remaining and compliance passing, "
                f"the verdict is SWITCH — not HOLD (hold is cleared), not STAY (there is a pull signal). "
                f"SWITCH."
            ),
            # V1 — gate re-check framing
            (
                f"{prev_note}Prior HOLD is cleared. {comp} delivers '{top_comp}', "
                f"resolving the blocking condition. "
                f"Gate re-check: Hold signal NONE. Compliance PASSED. {roi_note} "
                f"With the hold released and no new blocking condition, verdict converts to SWITCH. SWITCH."
            ),
            # V2 — derive from hold-release logic
            (
                f"A prior HOLD was in place because a specific condition blocked the switch. "
                f"{comp} now delivers '{top_comp}', resolving that condition. "
                f"When the blocking condition that caused HOLD is removed, the switch case "
                f"reasserts — the pull signal still exists and compliance still passes. "
                f"{roi_note} SWITCH."
            ),
            # V3 — decision framework reasoning
            (
                f"Step 1: Check blocking gates. Hold signal: NONE — no active hold. "
                f"Previous verdict was HOLD — the prior blocking condition has been resolved. "
                f"Step 2: Evaluate pull. '{top_comp}' from {comp} is CONCRETE — a shipped, GA capability. "
                f"Step 3: Match to push driver. Resolves '{issue}'. "
                f"Step 4: {roi_note} "
                f"Hold cleared + concrete pull + compliance PASSED = SWITCH."
            ),
            # V4 — explicit alternatives ruled out
            (
                f"Previous verdict was HOLD. Hold signal now: NONE — the hold condition has cleared. "
                f"Not HOLD: the hold gate is no longer active. Re-issuing HOLD when hold_signal=NONE "
                f"means ignoring that the blocking condition is gone. HOLD requires an ACTIVE blocking signal. "
                f"Not STAY: {comp} delivers '{top_comp}', a concrete pull signal that resolves '{issue}'. "
                f"There is a concrete pull and the gates are clear. "
                f"Compliance PASSED. {roi_note} SWITCH."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "compliance_newly_met":
        templates = [
            # V0 — gate unblocked
            (
                f"Compliance gate: previously BLOCKED, now PASSED. "
                f"Pull signal: CONCRETE — {comp} delivers {top_comp} resolving {issue}. "
                f"ROI gate: {roi_str}. Hold signal: NONE. "
                f"All gates clear. SWITCH."
            ),
            # V1 — unblocking framing
            (
                f"Compliance unblocked. The only previous barrier has been cleared. "
                f"Pull signal CONCRETE: {top_comp} from {comp} resolves {issue}. "
                f"ROI: {roi_str}. Hold: NONE. All gates now clear. SWITCH."
            ),
            # V2 — derive from gate unblock logic
            (
                f"Compliance was the sole blocking gate. It is now cleared. "
                f"With the blocker removed, evaluate the remaining signals on their merits: "
                f"'{top_comp}' from {comp} directly resolves '{issue}' — a concrete pull. "
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
        rally = top_tool if tool_ch else "active improvements"
        templates = [
            # V0 — weakening push
            (
                f"Push signal: WEAKENING — {tool} is improving ({rally}). "
                f"The urgency to switch is diminishing. {comp}'s changes ({top_comp}) do not outpace "
                f"the incumbent's recovery. Hold signal: NONE. Insufficient case for SWITCH or HOLD. STAY."
            ),
            # V1 — incumbent recovery
            (
                f"The incumbent is recovering: {tool} delivers {rally}. "
                f"When the current tool actively resolves its own issues, the urgency to switch drops. "
                f"{comp}'s '{top_comp}' no longer provides a decisive advantage. STAY."
            ),
            # V2 — derive from delta between improving incumbent vs competitor
            (
                f"{tool} is actively improving: {rally}. The push signal weakens when the "
                f"incumbent resolves its own issues. A switch requires the competitor to be "
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
        if met and comp_ch:
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
                    f"{comp} shows {top_comp}, but {roi_str}. "
                    f"A pull signal that does not resolve the primary push driver or clear the ROI gate "
                    f"cannot justify SWITCH. No clear dominant signal — STAY."
                ),
                (
                    f"Both signals active but pull is weak. '{top_comp}' does not overcome "
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
        if met and comp_ch:
            templates = [
                (
                    f"Both tools improved this period. The delta favours {comp}: {top_comp} "
                    f"addresses {issue} more directly than {tool}'s {top_tool}. "
                    f"{roi_str.capitalize()}. When both improve but the competitor's improvement "
                    f"resolves the primary push driver, the switch case is established. SWITCH."
                ),
                (
                    f"Both improved; competitor edge is decisive: '{top_comp}' resolves '{issue}'. "
                    f"{roi_str}. The competitor's improvement addresses the push driver while "
                    f"the incumbent's improvement does not close the gap. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Both tools improved this period. {tool} rallies with {top_tool} while "
                    f"{comp} delivers {top_comp}. {roi_str}. When both tools improve, a SWITCH "
                    f"requires the competitor to be decisively better — but neither the pull signal "
                    f"nor the ROI case creates a decisive advantage. STAY."
                ),
                (
                    f"Dual improvement — {tool} ships {top_tool}, {comp} ships {top_comp}. "
                    f"Neither side establishes a decisive advantage. {roi_str}. Without a clear "
                    f"delta in the competitor's favour, switching during mutual improvement is premature. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    return f"Signal evaluated for {comp} vs {tool}. {roi_str.capitalize()}."


# ── Ambiguous verdict inference ────────────────────────────────────────────────

def _infer_ambiguous_verdict(roi: dict, signal: dict) -> str:
    return "SWITCH" if (roi["roi_threshold_met"] and signal_competitor_changes(signal)) else "STAY"


# ── Full CoT trace assembly ───────────────────────────────────────────────────

def _build_cot_trace(scenario: str, signal: dict, context: dict, roi: dict, verdict: str, variant: int = 0) -> str:
    push = _push_section(context, signal)
    pull = _pull_section(signal)
    comp_line = _compliance_line(signal)
    roi_line = _roi_line(roi)
    hold_line = _hold_line(signal, scenario)
    analysis_text = _analysis(scenario, signal, context, roi, variant=variant)

    return (
        f"{push}\n\n"
        f"{pull}\n\n"
        f"{comp_line}\n"
        f"{roi_line}\n\n"
        f"{hold_line}\n\n"
        f"ANALYSIS: {analysis_text}\n"
        f"VERDICT: {verdict}"
    )


# ── File sampling ─────────────────────────────────────────────────────────────

def _collect_samples(limit: int) -> list[tuple[Path, str, str, str]]:
    by_scenario: dict[str, list[Path]] = {s: [] for s in SCENARIO_TYPES}

    for path in _GENERATED_DIR.glob("*.json"):
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
                        help="Downsample to equal STAY/SWITCH/HOLD counts after generation.")
    args = parser.parse_args()

    samples = _collect_samples(args.limit)
    total = len(samples) * args.variants
    print(f"Processing {len(samples)} signal files × {args.variants} variants = {total} traces "
          f"({args.limit} per scenario, {len(SCENARIO_TYPES) - len(_SKIP_SCENARIOS)} scenarios)...")

    records: list[dict] = []
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
        verdict = expected if expected is not None else _infer_ambiguous_verdict(roi, signal)

        # ── Consistency gate: ensure structured fields match expected verdict ──
        # This prevents contradictory training examples (e.g. Hold signal: NONE
        # with VERDICT: HOLD) which confuse the model during training.
        notes = signal_notes(signal)
        comp_ch = signal_competitor_changes(signal)
        detected_hold = _detect_hold_signal(notes, comp_ch)

        if scenario in _HOLD_SCENARIOS and detected_hold == "NONE":
            print(f"  SKIP {signal_path.name}: HOLD scenario but no detectable hold signal")
            skipped += 1
            continue
        if expected == "SWITCH" and detected_hold != "NONE" and scenario != "hold_resolved":
            print(f"  SKIP {signal_path.name}: SWITCH scenario but stray hold signal: {detected_hold[:60]}")
            skipped += 1
            continue

        # Gate: Shelfware flag present but verdict is STAY — contradicts system prompt.
        tool_ch = signal_current_tool_status(signal)
        detected_shelfware = _detect_shelfware(tool_ch)
        if detected_shelfware != "NONE" and detected_hold == "NONE" and verdict == "STAY":
            disq = _detect_disqualifier(notes, detected_hold)
            if disq == "NONE":
                print(f"  SKIP {signal_path.name}: Shelfware flag with STAY verdict and no Disqualifier")
                skipped += 1
                continue
            continue

        user_msg = _build_user_message(context, roi, signal, scenario=scenario)
        for v in range(args.variants):
            cot_trace = _build_cot_trace(scenario, signal, context, roi, verdict, variant=v)
            records.append({
                "messages": [
                    {"role": "system", "content": SYS_VERDICT_LEAN},
                    {"role": "user",   "content": user_msg},
                    {"role": "assistant", "content": cot_trace},
                ]
            })

    if args.balance:
        # ── Stratified balancing: equal counts per (verdict, category) cell ──
        from collections import defaultdict

        def _extract_verdict(rec: dict) -> str:
            c = rec["messages"][2]["content"]
            if "VERDICT: SWITCH" in c:
                return "SWITCH"
            if "VERDICT: HOLD" in c:
                return "HOLD"
            return "STAY"

        def _extract_category(rec: dict) -> str:
            for line in rec["messages"][1]["content"].split("\n"):
                if line.startswith("Category:"):
                    return line.split("—")[0].replace("Category:", "").strip()
            return "unknown"

        # Group into (verdict, category) cells
        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for rec in records:
            v = _extract_verdict(rec)
            c = _extract_category(rec)
            cells[(v, c)].append(rec)

        raw_cell_counts = {k: len(v) for k, v in sorted(cells.items())}
        print(f"Pre-balance cell counts: {raw_cell_counts}")

        # Find the minimum cell size across all 15 cells
        min_cell = min(len(v) for v in cells.values()) if cells else 0
        target_per_cell = min_cell
        print(f"Target per (verdict, category) cell: {target_per_cell}")

        rng_b = random.Random(99)
        balanced: list[dict] = []
        for key in sorted(cells.keys()):
            cell_list = cells[key]
            rng_b.shuffle(cell_list)
            balanced.extend(cell_list[:target_per_cell])
        rng_b.shuffle(balanced)
        records = balanced
        total_per_verdict = target_per_cell * 5
        print(f"Balance: {target_per_cell} per cell × 15 cells = {len(records)} total "
              f"({total_per_verdict} per verdict class)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} traces → {out}  ({skipped} skipped)")

    if records:
        print("\n─── Example trace (first record) ───────────────────────────────")
        ex = records[0]["messages"]
        print(f"[USER]\n{ex[1]['content']}\n")
        print(f"[ASSISTANT]\n{ex[2]['content']}")
        print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
