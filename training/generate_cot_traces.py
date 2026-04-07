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
    "hold:", "acquisition", "beta", "roadmap", "renews", "renewal", "pilot", "not ga", "preview",
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
    msg += f"\nROI: {roi_summary}"
    msg += f"\nHold signal: {hold_signal}"
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
        return f"COMPLIANCE: Previously BLOCKED — now MET ({cc})"
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
    if scenario in _HOLD_SCENARIOS:
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
        if hold:
            return f"HOLD CONDITION: {hold}"
        # Fallback to first note if present
        if notes:
            return f"HOLD CONDITION: {notes[0]}"
    return "HOLD CONDITION: NONE"


# ── ANALYSIS text templates (scenario-specific) ───────────────────────────────

def _analysis(scenario: str, signal: dict, context: dict, roi: dict) -> str:
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
        return (
            f"{comp} delivers '{top_comp}' — a concrete new capability that directly resolves '{issue}'. "
            f"This is not a vague update; it addresses the specific push issue. "
            f"ROI gate: {roi_str}. Compliance: PASSED. Hold signal: NONE. All gates clear. SWITCH."
        )
    if scenario == "push_dominant":
        push = top_tool if tool_ch else issue
        return (
            f"Push signal: HIGH — {tool} is degrading ({push}). "
            f"Pull signal: CONCRETE — {comp} resolves this with {top_comp}. "
            f"ROI gate: {roi_str}. Compliance gate: PASSED. Hold signal: NONE. "
            f"All gates clear. SWITCH."
        )
    if scenario == "shelfware_case":
        return (
            f"SWITCH CONFIRMED — shelfware case. {tool} is underutilised ({top_tool}). "
            f"Paying for unused capacity with no improvement in utilisation justifies switching. "
            f"ROI gate: {roi_str}. Compliance: PASSED. Hold signal: NONE. "
            f"Switch driven by waste elimination, not by competitor features. SWITCH."
        )
    if scenario == "hold_resolved":
        return (
            f"SWITCH CONFIRMED. {comp} delivers '{top_comp}' — confirmed GA delivery, blocking condition cleared. "
            f"The prior hold no longer applies. "
            f"ROI gate: {roi_str}. Compliance: PASSED. Hold signal: NONE. All gates clear. SWITCH."
        )
    if scenario == "compliance_newly_met":
        return (
            f"Compliance gate: previously BLOCKED, now PASSED — {cc}. "
            f"Pull signal: CONCRETE — {comp} delivers {top_comp} resolving {issue}. "
            f"ROI gate: {roi_str}. Hold signal: NONE. "
            f"All gates clear. SWITCH."
        )
    if scenario == "fluff_update":
        return (
            f"Pull signal: VAGUE — '{top_comp}' is marketing language with no concrete feature delivery. "
            f"VAGUE pull signals provide no basis for SWITCH regardless of push signal severity. "
            f"Notes about migration complexity or implementation risk are not HOLD conditions — "
            f"only contract locks, beta features, active pilots, or vendor acquisitions trigger HOLD. "
            f"Hold signal: NONE. No switch case, no hold case. STAY."
        )
    if scenario == "irrelevant_change":
        return (
            f"Pull signal: IRRELEVANT — {comp} delivers {top_comp}, which does not address {issue}. "
            f"A change that misses the primary push signal is worthless as a pull. "
            f"Hold signal: NONE. No switch case. STAY."
        )
    if scenario == "negative_signal_buried":
        return (
            f"Buried signal: DISQUALIFYING — {top_note}. "
            f"Despite {comp} showing {top_comp}, this hidden negative cancels the pull case. "
            f"Reading only surface signals would suggest SWITCH; reading all signals gives STAY. "
            f"Hold signal: NONE. The case fails on evidence. STAY."
        )
    if scenario == "current_tool_rally":
        rally = top_tool if tool_ch else "active improvements"
        return (
            f"Push signal: WEAKENING — {tool} is improving ({rally}). "
            f"The urgency to switch is diminishing. {comp}'s changes ({top_comp}) do not outpace "
            f"the incumbent's recovery. Hold signal: NONE. Insufficient case for SWITCH or HOLD. STAY."
        )
    if scenario == "competitor_nearly_ready":
        # Prefer a real hold condition from notes; fall back to comp_changes "beta"/"roadmap" signal
        blocking = next(
            (n for n in notes if not n.lower().startswith(_ADVISORY_PREFIXES)
             and any(kw in n.lower() for kw in _HOLD_NOTE_KW)),
            None,
        )
        if blocking is None:
            blocking = next((c for c in comp_ch if any(kw in c.lower() for kw in _HOLD_COMP_KW)), top_note)
        return (
            f"HOLD CONFIRMED. BLOCKED: {blocking}. "
            f"The feature is not GA — it cannot be relied upon in production. "
            f"Pull signal quality is irrelevant until GA delivery is confirmed. "
            f"Hold signal active. HOLD — reassess when the block clears."
        )
    if scenario == "roadmap_confirmed_hold":
        return (
            f"HOLD CONFIRMED. BLOCKED: {top_note}. "
            f"Roadmap items are promises, not delivered capability — this gate is NOT cleared. "
            f"Pull signal quality is irrelevant until the feature ships. "
            f"Hold signal active. HOLD — reassess only when delivered."
        )
    if scenario == "contract_renewal_hold":
        return (
            f"HOLD CONFIRMED. BLOCKED: {top_note}. "
            f"Contract terms prevent switching now without penalty. "
            f"The commercial gate is not cleared regardless of signal strength. "
            f"Hold signal active. HOLD — reassess at renewal."
        )
    if scenario == "vendor_acquisition_hold":
        return (
            f"HOLD CONFIRMED. BLOCKED: {top_note}. "
            f"A vendor acquisition creates uncertainty — "
            f"pull signal strength is irrelevant when the vendor's future is unknown. "
            f"Even CONCRETE pull signals cannot override an active acquisition hold. "
            f"Hold signal active. HOLD — reassess once transition stabilises."
        )
    if scenario == "pilot_in_progress_hold":
        return (
            f"HOLD CONFIRMED. BLOCKED: {top_note}. "
            f"A pilot is in progress — the switch case has not yet been validated. "
            f"Acting on incomplete evidence skips due diligence. "
            f"Hold signal active. HOLD — reassess after pilot results."
        )
    # Ambiguous scenarios
    if scenario == "both_signals":
        if met and comp_ch:
            return (
                f"Both push and pull signals are active. {comp} delivers {top_comp}, "
                f"addressing {issue}. {roi_str.capitalize()}. Pull dominates — SWITCH."
            )
        return (
            f"Both push and pull signals are present but pull is insufficient. "
            f"{comp} shows {top_comp}, but {roi_str}. No clear dominant signal — STAY."
        )
    if scenario == "price_hike_only":
        if met:
            return (
                f"Current tool pricing has increased and {roi_str}. "
                f"The cost delta alone meets the SWITCH threshold."
            )
        return (
            f"Current tool pricing has increased, but {roi_str}. "
            f"Price change without feature improvement does not clear the SWITCH threshold."
        )
    if scenario == "dual_improvement":
        if met and comp_ch:
            return (
                f"Both tools improved this period. The delta favours {comp}: {top_comp} "
                f"addresses {issue} more directly than {tool}'s {top_tool}. "
                f"{roi_str.capitalize()}."
            )
        return (
            f"Both tools improved this period. {tool} rallies with {top_tool} while "
            f"{comp} delivers {top_comp}. {roi_str}. Without a decisive edge, STAY."
        )
    return f"Signal evaluated for {comp} vs {tool}. {roi_str.capitalize()}."


# ── Ambiguous verdict inference ────────────────────────────────────────────────

def _infer_ambiguous_verdict(roi: dict, signal: dict) -> str:
    return "SWITCH" if (roi["roi_threshold_met"] and signal_competitor_changes(signal)) else "STAY"


# ── Full CoT trace assembly ───────────────────────────────────────────────────

def _build_cot_trace(scenario: str, signal: dict, context: dict, roi: dict, verdict: str) -> str:
    push = _push_section(context, signal)
    pull = _pull_section(signal)
    comp_line = _compliance_line(signal)
    roi_line = _roi_line(roi)
    hold_line = _hold_line(signal, scenario)
    analysis_text = _analysis(scenario, signal, context, roi)

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
    parser.add_argument("--output", default=str(_OUTPUT_PATH),
                        help="Output JSONL path.")
    args = parser.parse_args()

    samples = _collect_samples(args.limit)
    print(f"Processing {len(samples)} signal files "
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
        cot_trace = _build_cot_trace(scenario, signal, context, roi, verdict)

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
