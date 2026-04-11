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
    "hold:", "acquisition", "beta", "roadmap", "renews", "renewal", "pilot", "not ga",
])

_HOLD_COMP_KW = frozenset(["beta", "roadmap", "not ga", "preview"])

_ADVISORY_PREFIXES = ("consider", "suggest", "recommend", "note:", "fyi")


def _detect_hold_signal(notes: list[str], comp_changes: list[str] | None = None) -> str:
    """Return the first hold condition found in notes (or competitor_changes), or 'NONE'."""
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if any(kw in note_lower for kw in _HOLD_NOTE_KW):
            return note
    for change in (comp_changes or []):
        if any(kw in change.lower() for kw in _HOLD_COMP_KW):
            return change
    return "NONE"


_DISQUALIFIER_NOTE_KW = frozenset(["preview", "early access"])


def _detect_disqualifier(notes: list[str], hold_signal: str) -> str:
    """Return a disqualifier note if pre-GA language appears in notes and no hold is active."""
    if hold_signal != "NONE":
        return "NONE"
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if any(kw in note_lower for kw in _DISQUALIFIER_NOTE_KW):
            return note
    return "NONE"


# ── User message builder (matches _build_lean_user in model_runner.py) ────────

def _build_user_message(context: dict, roi_result: dict, signal: dict) -> str:
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
    # Use the first note that actually looks like a hold condition
    hold = next(
        (n for n in notes if not n.lower().startswith(_ADVISORY_PREFIXES)
         and any(kw in n.lower() for kw in _HOLD_NOTE_KW)),
        None,
    )
    # For competitor_nearly_ready, also check competitor_changes for beta/roadmap/preview
    if hold is None and scenario == "competitor_nearly_ready":
        hold = next((c for c in comp_ch if any(kw in c.lower() for kw in _HOLD_COMP_KW)), None)
    if scenario in _HOLD_SCENARIOS:
        if hold:
            return f"HOLD CONDITION: {hold}"
        # Fallback to first note if present
        if notes:
            return f"HOLD CONDITION: {notes[0]}"
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

    issue = known[0] if known else "existing tool limitations"
    top_comp = comp_ch[0] if comp_ch else "minor updates"
    top_tool = tool_ch[0] if tool_ch else "no new changes"
    top_note = notes[0] if notes else ""
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
                f"Hold condition resolved — {comp} delivers '{top_comp}', removing the blocking condition. "
                f"Hold signal: NONE (no new hold condition detected). "
                f"Compliance: PASSED. {roi_note} "
                f"Prior verdict was HOLD because a specific condition blocked the switch. "
                f"That condition is now gone. With no hold signal remaining and compliance passing, "
                f"the verdict is SWITCH — not HOLD (hold is cleared), not STAY (there is a pull signal). "
                f"SWITCH."
            ),
            # V1 — gate re-check framing
            (
                f"Prior HOLD is cleared. {comp} delivers '{top_comp}', resolving the blocking condition. "
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
        templates = [
            # V0 — missed target
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
        ]
        return templates[variant % len(templates)]

    if scenario == "negative_signal_buried":
        _NSB_DISQUAL_KW = frozenset({"preview", "early access"})
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
        templates = [
            # V0 — hold signal active from competitor changes
            (
                f"Hold signal: {blocking} — found in competitor changes, not buried notes. "
                f"Pre-GA language in competitor changes means the competitor is nearly ready, "
                f"not that it should be rejected. "
                f"The feature is not GA — it cannot be relied upon in production yet. "
                f"HOLD — reassess when the feature ships GA."
            ),
            # V1 — positional contrast: competitor changes vs notes
            (
                f"Hold signal active: '{blocking}' — in competitor changes, not buried notes. "
                f"Pre-GA language in competitor changes = competitor nearly ready → HOLD (wait for GA). "
                f"This is different from pre-GA in buried notes, which would be a disqualifier → STAY. "
                f"Even a CONCRETE pull signal is irrelevant until the feature ships GA. HOLD."
            ),
            # V2 — derive from GA requirement and positional reasoning
            (
                f"The blocking condition ('{blocking}') is in competitor changes — "
                f"the competitor is nearly ready, not absent. "
                f"Committing to a pre-GA feature risks relying on capability that may ship late or differently. "
                f"Hold signal active from competitor changes. "
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
                f"not STAY and not SWITCH. "
                f"Advisory notes or cost pressure do not override the hold gate. "
                f"HOLD — reassess when the feature ships GA."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "roadmap_confirmed_hold":
        templates = [
            # V0 — promise not delivery
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"Roadmap items are promises, not delivered capability — this gate is NOT cleared. "
                f"Pull signal quality is irrelevant until the feature ships. "
                f"Hold signal active. HOLD — reassess only when delivered."
            ),
            # V1 — promise vs delivery
            (
                f"HOLD. Blocking condition: {top_note}. "
                f"A roadmap commitment is not delivered capability. "
                f"The gate is not cleared until the feature ships. "
                f"Pull signals are irrelevant until delivery is confirmed. HOLD."
            ),
            # V2 — derive from the difference between roadmap and shipped capability
            (
                f"The blocking condition is '{top_note}'. A roadmap item is a future intention. "
                f"Switching on the basis of undelivered capability introduces real risk: the "
                f"feature may ship late, incompletely, or not at all. The correct approach is to "
                f"wait for confirmed delivery before committing. HOLD — reassess when the feature ships."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "contract_renewal_hold":
        templates = [
            # V0 — commercial gate
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"Contract terms prevent switching now without penalty. "
                f"The commercial gate is not cleared regardless of signal strength. "
                f"Hold signal active. HOLD — reassess at renewal."
            ),
            # V1 — penalty framing
            (
                f"HOLD. Commercial gate blocked: {top_note}. "
                f"The contract has not expired — switching now incurs penalties. "
                f"Competitor signal strength cannot override a hard commercial block. "
                f"HOLD until the renewal window. HOLD."
            ),
            # V2 — derive from commercial constraint logic
            (
                f"The contract '{top_note}' creates a hard commercial constraint. "
                f"Switching before the contract expires incurs penalties that the ROI calculation "
                f"does not account for. The signal strength is irrelevant — the commercial gate "
                f"overrides the technical decision until the renewal window opens. "
                f"HOLD — reassess at contract renewal."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "vendor_acquisition_hold":
        templates = [
            # V0 — uncertainty
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"A vendor acquisition creates uncertainty — "
                f"pull signal strength is irrelevant when the vendor's future is unknown. "
                f"Even CONCRETE pull signals cannot override an active acquisition hold. "
                f"Hold signal active. HOLD — reassess once transition stabilises."
            ),
            # V1 — risk framing
            (
                f"HOLD. Acquisition uncertainty: {top_note}. "
                f"Vendor acquisitions introduce product, roadmap, and support unknowns. "
                f"Committing to an acquired competitor mid-transition is high risk. "
                f"HOLD until transition stabilises. HOLD."
            ),
            # V2 — derive from acquisition uncertainty logic
            (
                f"The acquisition '{top_note}' makes the competitor an unstable landing zone. "
                f"Product strategy, pricing, support, and roadmap are all subject to change "
                f"during a transition. Switching mid-acquisition means committing to a target "
                f"that may look different in six months. Even strong pull signals cannot overcome "
                f"this structural uncertainty. HOLD until the transition stabilises."
            ),
        ]
        return templates[variant % len(templates)]

    if scenario == "pilot_in_progress_hold":
        templates = [
            # V0 — incomplete evidence
            (
                f"HOLD CONFIRMED. BLOCKED: {top_note}. "
                f"A pilot is in progress — the switch case has not yet been validated. "
                f"Acting on incomplete evidence skips due diligence. "
                f"Hold signal active. HOLD — reassess after pilot results."
            ),
            # V1 — due diligence framing
            (
                f"HOLD. Active pilot: {top_note}. "
                f"The evaluation is ongoing — pilot results have not been reviewed. "
                f"Committing before the pilot concludes defeats the purpose of the evaluation. "
                f"HOLD until pilot data is available. HOLD."
            ),
            # V2 — derive from due diligence logic
            (
                f"A pilot is underway: {top_note}. The pilot exists precisely to validate "
                f"whether the switch is the right decision. Acting before it concludes means "
                f"ignoring the evidence the organisation is in the process of collecting. "
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
                    f"addressing {issue}. {roi_str.capitalize()}. Pull dominates — SWITCH."
                ),
                (
                    f"Push and pull both present. Pull dominates: '{top_comp}' from {comp} resolves '{issue}'. "
                    f"ROI met. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Both push and pull signals are present but pull is insufficient. "
                    f"{comp} shows {top_comp}, but {roi_str}. No clear dominant signal — STAY."
                ),
                (
                    f"Both signals active but pull is weak. '{top_comp}' does not overcome {roi_str}. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "price_hike_only":
        if met:
            templates = [
                (
                    f"Current tool pricing has increased and {roi_str}. "
                    f"The cost delta alone meets the SWITCH threshold."
                ),
                (
                    f"Price increase triggers SWITCH: {roi_str}. Cost delta is sufficient. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Current tool pricing has increased, but {roi_str}. "
                    f"Price change without feature improvement does not clear the SWITCH threshold."
                ),
                (
                    f"Price hike present but {roi_str}. Insufficient alone to justify SWITCH. STAY."
                ),
            ]
        return templates[variant % len(templates)]

    if scenario == "dual_improvement":
        if met and comp_ch:
            templates = [
                (
                    f"Both tools improved this period. The delta favours {comp}: {top_comp} "
                    f"addresses {issue} more directly than {tool}'s {top_tool}. "
                    f"{roi_str.capitalize()}."
                ),
                (
                    f"Both improved; competitor edge is decisive: '{top_comp}' resolves '{issue}'. "
                    f"{roi_str}. SWITCH."
                ),
            ]
        else:
            templates = [
                (
                    f"Both tools improved this period. {tool} rallies with {top_tool} while "
                    f"{comp} delivers {top_comp}. {roi_str}. Without a decisive edge, STAY."
                ),
                (
                    f"Dual improvement, no decisive advantage. {roi_str}. STAY."
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

        user_msg = _build_user_message(context, roi, signal)
        for v in range(args.variants):
            cot_trace = _build_cot_trace(scenario, signal, context, roi, verdict, variant=v)
            records.append({
                "messages": [
                    {"role": "system", "content": SYS_VERDICT_LEAN},
                    {"role": "user",   "content": user_msg},
                    {"role": "assistant", "content": cot_trace},
                ]
            })

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
