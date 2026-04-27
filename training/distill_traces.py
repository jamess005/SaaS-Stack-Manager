"""
Distillation pass: rewrite weak ANALYSIS sections with high-quality reasoning.

This acts as the "teacher model" step — the template generator produces bulk
traces with correct structures, and this script enhances reasoning quality
for any trace that falls below a reasoning-depth threshold.

Usage:
    python training/distill_traces.py                      # in-place rewrite
    python training/distill_traces.py --dry-run             # report only
    python training/distill_traces.py --output distilled.jsonl  # write to new file
"""

import argparse
import json
import re
from pathlib import Path

from training.reasoning_alignment import build_semantic_view

_INPUT = Path(__file__).parent / "sft_cot_traces.jsonl"

# ── Reasoning quality check ──────────────────────────────────────────────────

REASONING_CONNECTIVES = [
    "because", "therefore", "since", "however", "but", "although",
    "this means", "this is", "which means", "the reason", "the key",
    "importantly", "critically", "specifically", "directly",
    "cannot", "does not", "is not", "no basis", "no evidence",
    "overrides", "regardless", "insufficient", "decisive",
    "when", "if", "unless", "even", "despite",
]

MIN_INDICATORS = 2
MIN_LENGTH = 200
_BOUNDARY_MARKERS = ("Not STAY:", "Not HOLD:", "Not SWITCH:")
_PLACEHOLDER_REPLACEMENTS = {
    "state the contract renewal timing explicitly": "the contract renewal window is not yet open",
    "shallow_degradation": "a shallow incumbent degradation signal",
    "minor maintenance update only": "a maintenance-only competitor update",
    "issues persist": "the incumbent issues remain unresolved",
    "surface cannot deliver": "the competitor cannot deliver",
    "acquisition certainty uncertain": "vendor acquisition remains uncertain",
    "shelfware case: no new changes": "a shelfware-driven switch case even without major new competitor changes",
}


def _has_boundary_coverage(analysis: str) -> bool:
    return sum(marker in analysis for marker in _BOUNDARY_MARKERS) >= 2


def _contains_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in _PLACEHOLDER_REPLACEMENTS)


def _clean_distilled_text(text: str) -> str:
    cleaned = text
    for token, replacement in _PLACEHOLDER_REPLACEMENTS.items():
        cleaned = re.sub(re.escape(token), replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\.\.+", ".", cleaned)
    return cleaned.strip()


def _is_weak(analysis: str) -> bool:
    al = analysis.lower()
    indicators = sum(1 for c in REASONING_CONNECTIVES if c in al)
    return (
        indicators < MIN_INDICATORS
        or len(analysis) < MIN_LENGTH
        or not _has_boundary_coverage(analysis)
        or _contains_placeholder(analysis)
    )


# ── Extract structured data from a trace ─────────────────────────────────────

def _parse_trace(record: dict) -> dict:
    """Extract all structured fields from user + assistant messages."""
    user = record["messages"][1]["content"]
    assistant = record["messages"][2]["content"]

    info = {
        "verdict": "",
        "category": "",
        "tool": "",
        "competitor": "",
        "push_lines": [],
        "pull_lines": [],
        "compliance": "",
        "roi_line": "",
        "hold_condition": "",
        "analysis": "",
        "issues": [],
        "user_tool_changes": [],
        "user_competitor_changes": [],
        "user_notes": [],
        "user_previous_verdict": "",
        "user_roi": "",
        "user_hold": "",
        "user_shelfware": "",
        "user_disqualifier": "",
        "_user_lower": user.lower(),
    }

    # From user message
    for line in user.split("\n"):
        if line.startswith("Category:"):
            parts = line.split("—")
            info["category"] = parts[0].replace("Category:", "").strip()
            if len(parts) > 1:
                info["tool"] = parts[1].replace("current tool:", "").strip()
        elif line.startswith("ROI:"):
            info["user_roi"] = line
        elif line.startswith("Hold signal:"):
            info["user_hold"] = line.replace("Hold signal:", "").strip()
        elif line.startswith("Shelfware flag:"):
            info["user_shelfware"] = line.replace("Shelfware flag:", "").strip()
        elif line.startswith("Disqualifier:"):
            info["user_disqualifier"] = line.replace("Disqualifier:", "").strip()
        elif line.startswith("Previous verdict:"):
            info["user_previous_verdict"] = line.replace("Previous verdict:", "").strip()

    section = None
    for raw_line in user.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        lower = stripped.lower()

        if "known issues:" in lower:
            section = "issues"
            continue
        if stripped.startswith("Changes this period"):
            section = None
            continue
        if stripped.startswith("Buried signals / notes:"):
            section = "user_notes"
            continue
        if stripped.startswith("ROI:"):
            section = None
            continue
        if stripped.startswith("Current tool:"):
            section = "user_tool_changes"
            remainder = stripped.split("Current tool:", 1)[1].strip()
            if remainder and remainder not in {"(unchanged this period)", "(none)"}:
                info["user_tool_changes"].append(remainder[2:] if remainder.startswith("- ") else remainder)
            continue
        if stripped.startswith("Competitor:"):
            section = "user_competitor_changes"
            remainder = stripped.split("Competitor:", 1)[1].strip()
            if remainder and remainder not in {"(unchanged this period)", "(none)"}:
                info["user_competitor_changes"].append(remainder[2:] if remainder.startswith("- ") else remainder)
            continue

        if stripped.startswith("- "):
            if section == "issues":
                info["issues"].append(stripped[2:])
            elif section == "user_tool_changes":
                info["user_tool_changes"].append(stripped[2:])
            elif section == "user_competitor_changes":
                info["user_competitor_changes"].append(stripped[2:])
            elif section == "user_notes":
                info["user_notes"].append(stripped[2:])

    # From assistant message
    section = None
    for line in assistant.split("\n"):
        if line.startswith("PUSH SIGNALS:"):
            section = "push"
            continue
        elif line.startswith("PULL SIGNALS:"):
            section = "pull"
            continue
        elif line.startswith("COMPLIANCE:"):
            info["compliance"] = line.replace("COMPLIANCE:", "").strip()
            section = None
        elif line.startswith("ROI:"):
            info["roi_line"] = line
            section = None
        elif line.startswith("HOLD CONDITION:"):
            info["hold_condition"] = line.replace("HOLD CONDITION:", "").strip()
            section = None
        elif line.startswith("ANALYSIS:"):
            info["analysis"] = line[9:].strip()
            section = None
        elif line.startswith("VERDICT:"):
            info["verdict"] = line.replace("VERDICT:", "").strip()
            section = None

        if section == "push" and line.strip().startswith("- "):
            info["push_lines"].append(line.strip()[2:])
        elif section == "pull" and line.strip().startswith("- "):
            info["pull_lines"].append(line.strip()[2:])

    # Extract competitor name from pull lines
    for pl in info["pull_lines"]:
        # "PulseMetrics delivers ..." → competitor is PulseMetrics
        m = re.match(r"^(\w[\w\s]*?)\s+delivers\s+", pl)
        if m:
            info["competitor"] = m.group(1).strip()
            break
    if not info["competitor"]:
        # Try from analysis text
        for word in info["analysis"].split():
            if word[0].isupper() and len(word) > 3 and word not in (
                "SWITCH", "STAY", "HOLD", "CONCRETE", "VAGUE", "None",
                "Pull", "Push", "Step", "Gate", "ROI", "Compliance",
                "Migration", "Annual", "Threshold", "HIGH", "MEDIUM",
                "Previously", "BLOCKED", "PASSED", "NONE", "The",
                "Both", "When", "Decision", "Shelfware",
            ):
                info["competitor"] = word
                break

    return info


_NO_CHANGE_MARKERS = {
    "no change",
    "no change — existing issues persist.",
    "no change this period",
    "stable",
    "unchanged",
}


def _semantic_view(info: dict):
    tool_changes = [
        change for change in info["user_tool_changes"]
        if change.lower().strip() not in _NO_CHANGE_MARKERS
    ]
    return build_semantic_view(info["issues"], info["user_competitor_changes"], tool_changes)


# ── ROI parsing ──────────────────────────────────────────────────────────────

def _parse_roi(roi_line: str) -> dict:
    """Extract ROI numbers from the structured ROI line."""
    m = re.search(r"Migration £([\d,]+)", roi_line)
    migration = int(m.group(1).replace(",", "")) if m else 0
    m = re.search(r"Annual net £([+-]?[\d,]+)", roi_line)
    annual = int(m.group(1).replace(",", "").replace("+", "")) if m else 0
    met = "MET" in roi_line and "NOT MET" not in roi_line
    return {"migration": migration, "annual": annual, "met": met}


def _normalize_phrase(text: str, fallback: str) -> str:
    if not text:
        return fallback
    cleaned = _clean_distilled_text(text.strip())
    return cleaned if cleaned else fallback


def _primary_issue(info: dict) -> str:
    view = _semantic_view(info)
    if view.primary_issue:
        return _normalize_phrase(view.primary_issue, "the incumbent gap")
    if info["push_lines"]:
        return _normalize_phrase(info["push_lines"][0].split(" — ")[0], "the incumbent gap")
    return "the incumbent gap"


def _primary_pull(info: dict) -> str:
    view = _semantic_view(info)
    if view.positive_pull is not None:
        return _normalize_phrase(view.positive_pull.change, "the competitor change")
    if view.pending_pull is not None:
        return _normalize_phrase(view.pending_pull.change, "the competitor change")
    if view.primary_pull:
        return _normalize_phrase(view.primary_pull, "the competitor change")
    return "the competitor change"


def _primary_positive_tool(info: dict) -> str:
    view = _semantic_view(info)
    if view.positive_tool is not None:
        return _normalize_phrase(view.positive_tool.change, "active improvements")
    return "active improvements"


def _primary_negative_tool(info: dict) -> str:
    view = _semantic_view(info)
    if view.negative_tool is not None:
        return _normalize_phrase(view.negative_tool.change, "active degradation")
    return _normalize_phrase(view.primary_tool_change or "active degradation", "active degradation")


def _has_previous_hold(info: dict) -> bool:
    return info["user_previous_verdict"].strip().upper() == "HOLD"


def _issue_is_high(info: dict, issue: str) -> bool:
    for line in info["push_lines"]:
        if issue in line and "Severity: HIGH" in line:
            return True
    return any(token in issue.lower() for token in ("required", "missing", "manual", "outdated", "no ", "cannot", "compliance", "audit", "sso", "gdpr"))


def _resolution_clause(info: dict, comp: str) -> str:
    view = _semantic_view(info)
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    if view.positive_pull is not None:
        return f"{comp} delivers '{pull}', which addresses '{issue}'."
    return f"{comp} presents a live alternative, but this trace does not justify claiming that an unrelated feature resolves '{issue}'."


def _pull_substance(info: dict) -> str:
    if not info["pull_lines"]:
        return "NONE"
    if any("Substance: CONCRETE" in line for line in info["pull_lines"]):
        return "CONCRETE"
    if any("Substance: VAGUE" in line for line in info["pull_lines"]):
        return "VAGUE"
    return "UNKNOWN"


def _has_active_hold(info: dict) -> bool:
    hold = info["hold_condition"].strip()
    return bool(hold) and hold != "NONE"


def _has_disqualifier(info: dict) -> bool:
    disq = info["user_disqualifier"].strip()
    return bool(disq) and disq != "NONE"


def _has_shelfware(info: dict) -> bool:
    shelfware = info["user_shelfware"].strip()
    return bool(shelfware) and shelfware != "NONE"


def _reject_switch_for_stay(info: dict) -> str:
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    roi = _parse_roi(info["roi_line"])

    if _has_disqualifier(info):
        disqualifier = _normalize_phrase(info["user_disqualifier"], "the unresolved competitor gap")
        return (
            f"the disqualifier '{disqualifier}' makes the competitor non-viable today, "
            f"so there is no valid migration target."
        )
    if _pull_substance(info) == "NONE":
        return f"there is no shipped competitor change that resolves '{issue}', so the switch case never starts."
    if _pull_substance(info) == "VAGUE":
        return f"the competitor change ('{pull}') is too vague to prove it resolves '{issue}'."
    if not roi["met"]:
        return f"the economics still fail the move, with ROI below threshold at £{roi['annual']:+d}/yr."
    return f"the competitor change does not solve the decisive gap ('{issue}') clearly enough to justify migration."


def _reject_hold_for_stay(info: dict) -> str:
    if _has_disqualifier(info):
        return (
            "this is not a wait-for-later case; the blocker is a real disqualifier, "
            "not a temporary delivery or timing gate."
        )
    if _has_active_hold(info):
        hold = _normalize_phrase(info["hold_condition"], "the cited timing condition")
        return f"'{hold}' is not the decisive issue here; even after waiting, the switch case still lacks merit."
    return (
        "no active contract, pilot, roadmap, or pre-GA gate is blocking the decision, "
        "so waiting would not create a stronger case."
    )


def _reject_switch_for_hold(info: dict) -> str:
    hold = _normalize_phrase(info["hold_condition"], "the active hold condition")
    return f"the active hold condition ('{hold}') still blocks commitment, so switching now would ignore a live temporal gate."


def _reject_stay_for_hold(info: dict) -> str:
    pull = _primary_pull(info)
    if _pull_substance(info) != "NONE":
        return (
            f"the competitor still shows a real switch case once the blocker clears, because '{pull}' "
            "materially improves the incumbent gap."
        )
    return (
        "the case is paused for timing, not rejected on merit; the decisive issue is when to move, "
        "not whether the competitor is viable at all."
    )


def _reject_stay_for_switch(info: dict) -> str:
    issue = _primary_issue(info)
    pull = _primary_pull(info)

    if _has_shelfware(info):
        shelfware = _normalize_phrase(info["user_shelfware"], "the shelfware signal")
        return (
            f"staying would preserve known waste from underutilisation ('{shelfware}'), "
            "which is itself the decisive reason to move."
        )
    if _has_previous_hold(info):
        return (
            "the previous hold condition has already cleared, so returning to STAY would leave "
            "a now-actionable switch case on the table."
        )
    if _pull_substance(info) == "CONCRETE":
        return f"staying leaves '{issue}' unresolved even though the competitor now ships '{pull}'."
    return f"staying leaves the incumbent gap ('{issue}') unresolved despite a viable alternative."


def _reject_hold_for_switch(info: dict) -> str:
    if _has_previous_hold(info):
        return "the previous hold condition has already cleared, so re-issuing HOLD would ignore the resolved gate."
    if _has_active_hold(info):
        hold = _normalize_phrase(info["hold_condition"], "the cited hold condition")
        return f"'{hold}' is no longer a live blocker in this trace, so waiting is not justified."
    return (
        "there is no active contract, pilot, roadmap, or pre-GA blocker left to wait on, "
        "so delaying would only postpone a justified move."
    )


def _append_boundary_rejections(base_analysis: str, info: dict) -> str:
    verdict = info["verdict"]
    additions: list[str] = []

    if verdict == "STAY":
        additions.extend([
            f"Not SWITCH: {_reject_switch_for_stay(info)}",
            f"Not HOLD: {_reject_hold_for_stay(info)}",
        ])
    elif verdict == "HOLD":
        additions.extend([
            f"Not SWITCH: {_reject_switch_for_hold(info)}",
            f"Not STAY: {_reject_stay_for_hold(info)}",
        ])
    elif verdict == "SWITCH":
        additions.extend([
            f"Not STAY: {_reject_stay_for_switch(info)}",
            f"Not HOLD: {_reject_hold_for_switch(info)}",
        ])

    return " ".join(part.strip() for part in [base_analysis, *additions] if part).strip()


# ── Distilled analysis writers ───────────────────────────────────────────────
# Each function produces a high-quality reasoning chain for a specific pattern.
# The reasoning always includes explicit causal connectives and step-by-step logic.

def _distill_stay_irrelevant(info: dict) -> str:
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    comp = info["competitor"] or "Competitor"
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI threshold {'met' if roi['met'] else 'not met'} at £{roi['annual']:+d}/yr"

    return (
        f"The primary push driver is '{issue}'. A valid SWITCH requires the pull signal "
        f"to address this specific gap. However, {comp}'s change ('{pull}') targets a "
        f"different area entirely — it does not resolve the push driver. Because the pull "
        f"signal misses the actual pain point, it cannot justify migration regardless of "
        f"how significant the change is in its own domain. {roi_str}. "
        f"Hold signal: NONE — no temporal gate is active. Since there is no relevant pull "
        f"signal and no hold condition, the verdict is STAY."
    )


def _distill_stay_fluff(info: dict) -> str:
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    comp = info["competitor"] or "Competitor"

    return (
        f"The pull signal from {comp} ('{pull}') fails the substance test — it describes "
        f"improvements in vague, marketing-oriented language without naming a specific "
        f"feature or capability. Because a SWITCH decision commits the organisation to "
        f"migration cost and disruption, it requires concrete evidence that the competitor "
        f"resolves the push driver ('{issue}'). Vague language cannot be evaluated against "
        f"a specific problem, therefore it provides no basis for SWITCH. "
        f"Hold signal: NONE — this is not a hold case either, since no temporal blocking "
        f"condition exists. Without a verifiable pull signal, the verdict is STAY."
    )


def _distill_stay_rally(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    pull = _primary_pull(info)
    push = _primary_positive_tool(info)

    return (
        f"The incumbent ({tool}) is actively improving: '{push}'. This is significant "
        f"because it weakens the push signal — when the current tool addresses its own "
        f"issues, the urgency to switch diminishes. {comp}'s pull signal ('{pull}') "
        f"does not create a decisive enough advantage over an improving incumbent. "
        f"A SWITCH requires the competitor to be clearly better, but when both tools "
        f"are moving in the right direction, the burden of proof for switching increases. "
        f"Hold signal: NONE. Since the push signal is weakening and the pull signal "
        f"is insufficient to overcome an improving incumbent, the verdict is STAY."
    )


def _distill_stay_disqualifier(info: dict) -> str:
    disq = info["user_disqualifier"] or "pre-GA language in notes"
    comp = info["competitor"] or "Competitor"
    issue = _primary_issue(info)

    return (
        f"A Disqualifier is present: '{disq}' — found in buried notes, not in competitor "
        f"changes. This distinction is critical because pre-GA language in notes means the "
        f"competitor itself is not viable (Disqualifier → STAY), whereas pre-GA language "
        f"in competitor changes would mean the competitor is nearly ready (Hold signal → HOLD). "
        f"Because the Disqualifier indicates {comp} cannot deliver the capability in "
        f"production today, the pull signal is invalidated regardless of how concrete it "
        f"appears on the surface. Hold signal: NONE — no temporal gate exists. "
        f"The Disqualifier overrides the surface-level pull analysis. STAY."
    )


def _distill_stay_price(info: dict) -> str:
    roi = _parse_roi(info["roi_line"])
    tool = info["tool"]

    return (
        f"The current tool ({tool}) has increased pricing. However, a price change alone "
        f"cannot justify SWITCH when the cost delta is negative or negligible — "
        f"ROI threshold not met at £{roi['annual']:+d}/yr. Because migration carries "
        f"its own costs (£{roi['migration']} one-time) and disruption risk, the savings "
        f"must clear the ROI threshold to justify the move. Without a concrete pull signal "
        f"resolving a push issue, and with the economics not supporting the move, "
        f"the switch case does not exist. Hold signal: NONE. STAY."
    )


def _distill_stay_both(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    pull = _primary_pull(info)
    issue = _primary_issue(info)
    roi = _parse_roi(info["roi_line"])

    return (
        f"Both push and pull signals are present, but the pull signal is insufficient to "
        f"justify SWITCH. {comp} offers '{pull}', however the ROI threshold is not met "
        f"at £{roi['annual']:+d}/yr. Because the pull signal does not clearly resolve the "
        f"primary push driver ('{issue}') and the economics do not support migration, "
        f"the burden of proof for SWITCH is not met. When both signals are active but "
        f"neither dominates, the default is to stay — a SWITCH requires a decisive "
        f"advantage, not merely the presence of both signals. Hold signal: NONE. STAY."
    )


def _distill_switch_gate_check(info: dict) -> str:
    comp = info["competitor"] or "Competitor"
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI threshold {'met' if roi['met'] else 'not met'} at £{roi['annual']:+d}/yr"

    return (
        f"Step 1: Check blocking gates. Compliance: PASSED. Hold signal: NONE. No blocks. "
        f"Step 2: Evaluate pull substance. '{pull}' from {comp} is CONCRETE — this is a "
        f"specific, shipped capability, not a vague promise. "
        f"Step 3: Match pull to push. The push driver is '{issue}'. {_resolution_clause(info, comp)} "
        f"Step 4: ROI check. {roi_str}. "
        f"Since all four gates are clear and the concrete pull resolves the push driver, "
        f"the switch case is established. SWITCH."
    )


def _distill_switch_compliance(info: dict) -> str:
    comp = info["competitor"] or "Competitor"
    pull = _primary_pull(info)
    issue = _primary_issue(info)
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI: £{roi['annual']:+d}/yr"

    return (
        f"Compliance was the sole blocking gate — it is now cleared. This changes the "
        f"decision landscape because the switch case that compliance previously blocked "
        f"can now proceed on its merits. With the blocker removed: the pull signal from "
        f"{comp} ('{pull}') is CONCRETE. {_resolution_clause(info, comp)} "
        f"Because the pull addresses the specific gap and compliance now passes, the only "
        f"remaining question is whether any other gate blocks. Hold signal: NONE. "
        f"{roi_str}. Since compliance is cleared, the pull is concrete, and no "
        f"hold condition exists, the verdict converts from blocked to SWITCH."
    )


def _distill_switch_push_high(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    push = _primary_negative_tool(info)
    pull = _primary_pull(info)
    issue = _primary_issue(info)

    return (
        f"{tool} is critically degrading: '{push}'. At HIGH push severity, the calculus "
        f"changes — because the incumbent is actively failing, the cost of staying is not "
        f"zero but rather the ongoing damage from the degradation. {_resolution_clause(info, comp)} "
        f"That makes the alternative relevant to the same high-severity issue ('{issue}'), "
        f"not to some unrelated feature line. Compliance: PASSED. Hold: NONE. Therefore SWITCH."
    )


def _distill_switch_both_aligned(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    push = _primary_negative_tool(info)
    pull = _primary_pull(info)
    issue = _primary_issue(info)

    return (
        f"Both push and pull signals reinforce the same conclusion. {tool} is degrading "
        f"('{push}') while {comp} delivers '{pull}'. This is significant because when "
        f"push and pull are aligned — the tool is getting worse AND the competitor is "
        f"getting better at exactly the thing that matters — the switch case is at its "
        f"strongest. {_resolution_clause(info, comp)} "
        f"Compliance: PASSED. Hold signal: NONE. Since both signals point in the same "
        f"direction and all gates clear, the verdict is SWITCH."
    )


def _distill_switch_shelfware(info: dict) -> str:
    tool = info["tool"]
    shelfware = info["user_shelfware"] or info["push_lines"][0].split(" — ")[0] if info["push_lines"] else "unused capacity"

    return (
        f"This is a shelfware case: {tool} is underutilised ('{shelfware}'). The decision "
        f"driver here is waste elimination, not competitor features. Because the organisation "
        f"is paying for capacity it does not use, the cost of inaction is certain and ongoing. "
        f"The standard ROI threshold relaxes for shelfware — when you are paying for something "
        f"you do not use, the migration cost is offset by eliminating the waste. "
        f"Compliance: PASSED. Hold signal: NONE — advisory notes about competitor limitations "
        f"are not hold conditions. Since the switch is driven by cost waste rather than "
        f"competitor superiority, the verdict is SWITCH."
    )


def _distill_switch_both_signals(info: dict) -> str:
    comp = info["competitor"] or "Competitor"
    pull = _primary_pull(info)
    issue = _primary_issue(info)
    roi = _parse_roi(info["roi_line"])

    return (
        f"Both push and pull signals are active. {comp} delivers '{pull}', which directly "
        f"connects to '{issue}'. {_resolution_clause(info, comp)} The ROI threshold is met at £{roi['annual']:+d}/yr, meaning "
        f"the economics support the migration. Because the pull signal resolves the primary "
        f"push driver and the ROI gate clears, the switch case stands on both technical and "
        f"financial merits. Compliance: PASSED. Hold signal: NONE. When a concrete pull "
        f"addresses the push driver with positive ROI and no blocking gates, "
        f"the verdict is SWITCH."
    )


def _distill_hold_confirmed(info: dict) -> str:
    hold = info["hold_condition"]
    tool = info["tool"]
    roi = _parse_roi(info["roi_line"])
    roi_str = f"£{roi['annual']:+d}/yr"

    # Determine hold type from the condition text
    hold_lower = hold.lower()
    if "pilot" in hold_lower:
        reason = (
            f"A pilot is in progress: '{hold}'. The pilot exists precisely to validate "
            f"whether the switch is correct. Acting before it concludes means ignoring "
            f"evidence the organisation is actively collecting. Because the evaluation is "
            f"incomplete, committing to a decision now defeats the purpose of the pilot."
        )
        reassess = "pilot results are available"
    elif "acquisition" in hold_lower or "acquiring" in hold_lower:
        reason = (
            f"A vendor acquisition is underway: '{hold}'. Acquisitions create structural "
            f"uncertainty — product roadmap, pricing, and support are all subject to change. "
            f"Because the competitor's future state is unknown during a transition, committing "
            f"now risks landing on a platform that looks different in six months."
        )
        reassess = "the acquisition transition stabilises"
    elif "contract" in hold_lower or "renew" in hold_lower:
        reason = (
            f"A commercial constraint is active: '{hold}'. The contract has not expired — "
            f"switching before the renewal window incurs penalties that the ROI calculation "
            f"does not account for. Because the commercial gate creates a hard cost barrier, "
            f"it overrides the technical decision regardless of signal strength."
        )
        reassess = "the contract renewal window opens"
    elif "roadmap" in hold_lower or "scheduled" in hold_lower or "planned" in hold_lower:
        reason = (
            f"A roadmap commitment blocks the decision: '{hold}'. A roadmap item is a future "
            f"intention, not a delivered capability. Because committing on the basis of undelivered "
            f"capability introduces real risk (the feature may ship late, incompletely, or not at all), "
            f"the prudent gate is to wait for confirmed delivery."
        )
        reassess = "the feature ships and is confirmed GA"
    elif "beta" in hold_lower or "not ga" in hold_lower or "pre-ga" in hold_lower or "preview" in hold_lower:
        reason = (
            f"The competitor feature is pre-GA: '{hold}'. Pre-GA language in competitor changes "
            f"means the feature is nearly ready but cannot be relied upon in production. "
            f"Because pre-GA features may change scope, timeline, or quality before general "
            f"availability, the decision should wait for GA confirmation."
        )
        reassess = "the feature reaches general availability"
    else:
        reason = (
            f"A blocking condition is active: '{hold}'. This creates a temporal gate that "
            f"must clear before the switch decision can proceed. Because the blocking condition "
            f"introduces uncertainty or risk that the standard signal analysis cannot account for, "
            f"it takes precedence over all other factors."
        )
        reassess = "the blocking condition clears"

    return (
        f"{reason} "
        f"ROI ({roi_str}) is irrelevant while the hold gate is active — even positive ROI "
        f"cannot override a structural blocker. Hold signal active, therefore all other "
        f"signal analysis (push, pull, ROI) is suspended until the condition resolves. "
        f"HOLD — reassess when {reassess}."
    )


def _distill_hold_pilot(info: dict) -> str:
    return _distill_hold_confirmed(info)


def _distill_hold_resolved(info: dict) -> str:
    comp = info["competitor"] or "Competitor"
    issue = _primary_issue(info)
    pull = _primary_pull(info)
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI threshold {'met' if roi['met'] else 'not met'} at £{roi['annual']:+d}/yr"

    if roi['met']:
        roi_clause = f"{roi_str} — the economics now support the move."
    elif _issue_is_high(info, issue):
        roi_clause = f"{roi_str} — below threshold, but '{issue}' is still a HIGH-severity blocker, so the delivered capability creates clear operational gain."
    else:
        roi_clause = f"{roi_str}."

    return (
        f"Previous verdict: HOLD — the prior blocker has now cleared because Hold signal = NONE. "
        f"That does not make SWITCH automatic; it means the trace must be re-evaluated on the normal SWITCH/STAY rules. "
        f"Here {_resolution_clause(info, comp)} Compliance PASSED. {roi_clause} "
        f"With the hold removed and the core gap now covered, the verdict becomes SWITCH."
    )


# ── Pattern detection ────────────────────────────────────────────────────────

def _detect_pattern(info: dict) -> str:
    """Classify the trace by its reasoning pattern."""
    al = info["analysis"].lower()
    v = info["verdict"]

    if v == "STAY":
        if "irrelevant" in al or "relevance" in al or "misses" in al:
            return "stay_irrelevant"
        if "vague" in al or "substance" in al or "marketing" in al:
            return "stay_fluff"
        if "weakening" in al or "incumbent" in al or "recovering" in al or "improving" in al:
            return "stay_rally"
        if "disqualifier" in al or "pre-ga" in al:
            return "stay_disqualifier"
        if "price" in al or "pricing" in al or "cost" in al:
            return "stay_price"
        if "both" in al and ("insufficient" in al or "weak" in al or "present" in al):
            return "stay_both"
        return "stay_generic"

    if v == "SWITCH":
        # Check hold_resolved FIRST — must precede other SWITCH checks or falls through
        # to switch_generic which overwrites hold-resolution reasoning with gate-check logic.
        if _has_previous_hold(info):
            return "switch_hold_resolved"
        if "gate check" in al:
            return "switch_gate_check"
        if "compliance" in al and ("previously" in al or "unblocked" in al or "cleared" in al):
            return "switch_compliance"
        if "push signal: high" in al or "failing" in al or "degrading" in al:
            return "switch_push_high"
        if "aligned" in al:
            return "switch_both_aligned"
        if "shelfware" in al or "waste" in al or "underutilised" in al or "idle" in al:
            return "switch_shelfware"
        if "both" in al and "pull" in al:
            return "switch_both_signals"
        return "switch_generic"

    if v == "HOLD":
        if "pilot" in al:
            return "hold_pilot"
        return "hold_confirmed"

    return "unknown"


# ── Dispatch ─────────────────────────────────────────────────────────────────

DISTILLERS = {
    "switch_hold_resolved": _distill_hold_resolved,
    "stay_irrelevant": _distill_stay_irrelevant,
    "stay_fluff": _distill_stay_fluff,
    "stay_rally": _distill_stay_rally,
    "stay_disqualifier": _distill_stay_disqualifier,
    "stay_price": _distill_stay_price,
    "stay_both": _distill_stay_both,
    "stay_generic": _distill_stay_both,  # fallback to both-signals STAY logic
    "switch_gate_check": _distill_switch_gate_check,
    "switch_compliance": _distill_switch_compliance,
    "switch_push_high": _distill_switch_push_high,
    "switch_both_aligned": _distill_switch_both_aligned,
    "switch_shelfware": _distill_switch_shelfware,
    "switch_both_signals": _distill_switch_both_signals,
    "switch_generic": _distill_switch_gate_check,  # fallback to gate-check reasoning
    "hold_confirmed": _distill_hold_confirmed,
    "hold_pilot": _distill_hold_pilot,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Distill weak traces with high-quality reasoning.")
    parser.add_argument("--input", default=str(_INPUT), help="Input JSONL path.")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: overwrite input).")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write.")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = [json.loads(line) for line in input_path.read_text().strip().split("\n")]
    print(f"Loaded {len(records)} traces from {input_path}")

    rewritten = 0
    failed = 0

    for i, rec in enumerate(records):
        assistant = rec["messages"][2]["content"]

        # Extract analysis line
        old_analysis = ""
        for line in assistant.split("\n"):
            if line.startswith("ANALYSIS:"):
                old_analysis = line[9:].strip()
                break

        if not _is_weak(old_analysis):
            continue

        # Parse structured data
        info = _parse_trace(rec)

        # Detect pattern and distill
        pattern = _detect_pattern(info)
        distiller = DISTILLERS.get(pattern)

        if distiller is None:
            failed += 1
            continue

        new_analysis = _clean_distilled_text(_append_boundary_rejections(distiller(info), info))

        # Verify the distilled analysis is actually better
        if _is_weak(new_analysis):
            # Still weak after distillation — skip to avoid degrading
            print(f"  WARNING: distilled trace {i} ({pattern}) still weak, keeping original")
            failed += 1
            continue

        # Replace ANALYSIS line in assistant content
        new_assistant = assistant.replace(
            f"ANALYSIS: {old_analysis}",
            f"ANALYSIS: {new_analysis}",
            1,
        )
        rec["messages"][2]["content"] = new_assistant
        rewritten += 1

        if args.dry_run and rewritten <= 5:
            print(f"\n  --- Trace {i} ({pattern}) ---")
            print(f"  OLD: {old_analysis[:120]}...")
            print(f"  NEW: {new_analysis[:120]}...")

    print(f"\nDistillation complete: {rewritten} rewritten, {failed} unchanged, "
          f"{len(records) - rewritten - failed} already strong")

    if not args.dry_run:
        out_path = Path(args.output) if args.output else input_path
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} traces → {out_path}")


if __name__ == "__main__":
    main()
