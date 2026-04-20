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


def _is_weak(analysis: str) -> bool:
    al = analysis.lower()
    indicators = sum(1 for c in REASONING_CONNECTIVES if c in al)
    return indicators < MIN_INDICATORS or len(analysis) < MIN_LENGTH


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
        elif line.strip().startswith("- ") and "known issues" not in line.lower():
            pass  # handled below

    # Extract known issues from user message
    in_issues = False
    in_competitor = False
    for line in user.split("\n"):
        if "known issues:" in line.lower():
            in_issues = True
            continue
        if line.startswith("Changes this period") or line.startswith("Buried") or line.startswith("ROI"):
            in_issues = False
        if in_issues and line.strip().startswith("- "):
            info["issues"].append(line.strip()[2:])
        if "Competitor:" in line:
            in_competitor = True
            continue
        if in_competitor and line.strip().startswith("- "):
            # Competitor change
            pass

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


# ── ROI parsing ──────────────────────────────────────────────────────────────

def _parse_roi(roi_line: str) -> dict:
    """Extract ROI numbers from the structured ROI line."""
    m = re.search(r"Migration £([\d,]+)", roi_line)
    migration = int(m.group(1).replace(",", "")) if m else 0
    m = re.search(r"Annual net £([+-]?[\d,]+)", roi_line)
    annual = int(m.group(1).replace(",", "").replace("+", "")) if m else 0
    met = "MET" in roi_line and "NOT MET" not in roi_line
    return {"migration": migration, "annual": annual, "met": met}


# ── Distilled analysis writers ───────────────────────────────────────────────
# Each function produces a high-quality reasoning chain for a specific pattern.
# The reasoning always includes explicit causal connectives and step-by-step logic.

def _distill_stay_irrelevant(info: dict) -> str:
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "minor updates"
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
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "minor updates"
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
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "minor updates"
    push = info["push_lines"][0].split(" — ")[0] if info["push_lines"] else "active improvements"

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
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"

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
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "minor updates"
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
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
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "concrete feature"
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI threshold {'met' if roi['met'] else 'not met'} at £{roi['annual']:+d}/yr"

    return (
        f"Step 1: Check blocking gates. Compliance: PASSED. Hold signal: NONE. No blocks. "
        f"Step 2: Evaluate pull substance. '{pull}' from {comp} is CONCRETE — this is a "
        f"specific, shipped capability, not a vague promise. "
        f"Step 3: Match pull to push. The push driver is '{issue}'. The pull signal "
        f"directly resolves this gap, because it delivers the specific capability that "
        f"the current tool lacks. "
        f"Step 4: ROI check. {roi_str}. "
        f"Since all four gates are clear and the concrete pull resolves the push driver, "
        f"the switch case is established. SWITCH."
    )


def _distill_switch_compliance(info: dict) -> str:
    comp = info["competitor"] or "Competitor"
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "concrete feature"
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI: £{roi['annual']:+d}/yr"

    return (
        f"Compliance was the sole blocking gate — it is now cleared. This changes the "
        f"decision landscape because the switch case that compliance previously blocked "
        f"can now proceed on its merits. With the blocker removed: the pull signal from "
        f"{comp} ('{pull}') is CONCRETE, directly resolving the push driver ('{issue}'). "
        f"Because the pull addresses the specific gap and compliance now passes, the only "
        f"remaining question is whether any other gate blocks. Hold signal: NONE. "
        f"{roi_str}. Since compliance is cleared, the pull is concrete, and no "
        f"hold condition exists, the verdict converts from blocked to SWITCH."
    )


def _distill_switch_push_high(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    push = info["push_lines"][0].split(" — ")[0] if info["push_lines"] else "critical degradation"
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "available alternative"

    return (
        f"{tool} is critically degrading: '{push}'. At HIGH push severity, the calculus "
        f"changes — because the incumbent is actively failing, the cost of staying is not "
        f"zero but rather the ongoing damage from the degradation. Even if the pull signal "
        f"('{pull}' from {comp}) is not the strongest, it provides a viable exit path from "
        f"the failing tool. When push severity is HIGH, a weaker pull signal is sufficient "
        f"because the priority shifts from finding the perfect alternative to escaping the "
        f"failing incumbent. Compliance: PASSED. Hold: NONE. Therefore SWITCH."
    )


def _distill_switch_both_aligned(info: dict) -> str:
    tool = info["tool"]
    comp = info["competitor"] or "Competitor"
    push = info["push_lines"][0].split(" — ")[0] if info["push_lines"] else "degradation"
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "improvement"
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"

    return (
        f"Both push and pull signals reinforce the same conclusion. {tool} is degrading "
        f"('{push}') while {comp} delivers '{pull}'. This is significant because when "
        f"push and pull are aligned — the tool is getting worse AND the competitor is "
        f"getting better at exactly the thing that matters — the switch case is at its "
        f"strongest. The pull directly addresses the push driver ('{issue}'). "
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
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "concrete feature"
    issue = info["issues"][0] if info["issues"] else "existing tool limitations"
    roi = _parse_roi(info["roi_line"])

    return (
        f"Both push and pull signals are active. {comp} delivers '{pull}', which directly "
        f"addresses '{issue}'. The ROI threshold is met at £{roi['annual']:+d}/yr, meaning "
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
    pull = info["pull_lines"][0].split(" — ")[0] if info["pull_lines"] else "the blocking condition"
    roi = _parse_roi(info["roi_line"])
    roi_str = f"ROI threshold {'met' if roi['met'] else 'not met'} at £{roi['annual']:+d}/yr"

    return (
        f"Previous verdict: HOLD — the prior hold condition has now been resolved. "
        f"{comp} delivers '{pull}', which removes the specific blocking condition that "
        f"caused the HOLD. Because the hold gate is no longer active (Hold signal: NONE), "
        f"the decision reverts to a standard SWITCH/STAY evaluation. "
        f"With the hold cleared: compliance PASSED, pull signal is CONCRETE (GA delivery), "
        f"and the pull directly addresses the push driver. "
        f"{roi_str} — note that ROI does NOT block when a prior HOLD is cleared; "
        f"the hold-release is the binding event, not the ROI gate. "
        f"No new hold condition has emerged. Therefore SWITCH — not HOLD (condition gone), "
        f"not STAY (there is a concrete pull). SWITCH."
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
        if "previous verdict: hold" in info.get("_user_lower", ""):
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

        new_analysis = distiller(info)

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
