"""
Regenerate verdict step traces in the new ANALYSIS + VERDICT format.

Reads existing traces_fresh/traces.jsonl, extracts compliance/push/pull
outputs for each signal, and rewrites the verdict traces using:
  - scenario label to determine the correct verdict
  - actual signal content for the ANALYSIS text
  - new user message format (bullet list, not UNKNOWN labels)

Run from project root:
    python scripts/regenerate_verdict_traces.py [--traces-dir training/traces_fresh]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Scenario → expected verdict ────────────────────────────────────────────────
_SCENARIO_VERDICT = {
    "hard_compliance_failure": "STAY",
    "fluff_update":            "STAY",
    "irrelevant_change":       "STAY",
    "negative_signal_buried":  "STAY",
    "current_tool_rally":      "STAY",
    "pull_dominant":           "SWITCH",
    "push_dominant":           "SWITCH",
    "shelfware_case":          "SWITCH",
    "hold_resolved":           "SWITCH",
    "competitor_nearly_ready": "HOLD",
    # Ambiguous — derive from ROI + signal strength
    "both_signals":            None,
    "price_hike_only":         None,
    "dual_improvement":        None,
}


# ── Parsers ────────────────────────────────────────────────────────────────────

def _compliance_passed(result: str) -> bool:
    """True if compliance result contains PASS but not FAIL (case-insensitive)."""
    upper = result.upper()
    if "FAIL" in upper:
        return False
    return "PASS" in upper or "COMPLIANT" in upper


def _extract_high_signals(result: str, max_signals: int = 3) -> list[str]:
    """
    Extract HIGH-rated signal descriptions from Llama's numbered list format.
    Handles:  1. **Signal text**: HIGH ...
              1. **"Quoted feature"**: HIGH ...
    """
    signals = []
    for m in re.finditer(
        r'\d+\.\s+\*\*"?([^*\n"]{10,120})"?\*\*[:\s]+HIGH',
        result, re.IGNORECASE
    ):
        text = m.group(1).strip().rstrip("—–-").strip()
        if text:
            signals.append(text)
        if len(signals) >= max_signals:
            break
    return signals


def _extract_strength(result: str, signal_type: str) -> str:
    """Extract Overall push/pull: STRONG/MODERATE/WEAK."""
    m = re.search(
        rf"Overall\s+{signal_type}\s*[:\-–]\s*(STRONG|MODERATE|WEAK)",
        result, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    # Fallback: count HIGH signals
    high_count = len(re.findall(r'\bHIGH\b', result, re.IGNORECASE))
    if high_count >= 3:
        return "STRONG"
    if high_count >= 1:
        return "MODERATE"
    return "WEAK"


def _extract_bullets(result: str, max_lines: int = 5) -> str:
    """Extract up to max_lines bullets for the new user message format."""
    lines = []
    for m in re.finditer(
        r'\d+\.\s+\*\*("?[^*\n"]{5,120}"?)\*\*[:\s]+(?:HIGH|MEDIUM|LOW)',
        result, re.IGNORECASE
    ):
        text = m.group(1).strip().strip('"')
        rating_m = re.search(r'\b(HIGH|MEDIUM|LOW)\b', result[m.start():m.start()+200], re.IGNORECASE)
        rating = rating_m.group(1).upper() if rating_m else "MEDIUM"
        lines.append(f"  - {text} [{rating}]")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) if lines else "  (none)"


def _parse_roi(user_content: str) -> dict:
    """Parse ROI figures from the verdict user message."""
    roi = {"annual_saving": 0, "net": 0, "threshold_met": False, "roi_line": ""}
    m = re.search(r"(ROI:.+)", user_content)
    if m:
        roi["roi_line"] = m.group(1).strip()
    m = re.search(r"Annual saving:\s*£(-?[\d,]+)", user_content)
    if m:
        roi["annual_saving"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Net:\s*£(-?[\d,]+)", user_content)
    if m:
        roi["net"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Threshold met:\s*(YES|NO)", user_content)
    if m:
        roi["threshold_met"] = m.group(1).upper() == "YES"
    return roi


def _extract_compliance_block(result: str) -> str:
    """Extract the specific compliance failure reason (first FAIL bullet)."""
    for m in re.finditer(
        r'\*\*(No [^*\n]{5,80}|[^*\n]{5,80}outside[^*\n]{5,60})\*\*',
        result, re.IGNORECASE
    ):
        return m.group(1).strip()
    # Fallback: first line after FAIL
    for line in result.splitlines():
        if "fail" in line.lower() or "no soc2" in line.lower() or "residency" in line.lower():
            return line.strip().strip("*-").strip()
    return "a hard compliance requirement is not met"


def _header_line(user_content: str) -> str:
    """Extract '{category}: {tool} vs {competitor}' header from user message."""
    return user_content.split("\n")[0].strip()


# ── ANALYSIS + VERDICT generators ─────────────────────────────────────────────

def _make_verdict(
    scenario: str,
    compliance_result: str,
    push_result: str,
    pull_result: str,
    roi: dict,
) -> str:
    """
    Return 'ANALYSIS: ...\nVERDICT: X' string for this signal.
    Uses scenario label as primary determinant; signal content for ANALYSIS text.
    """
    comp_pass = _compliance_passed(compliance_result)
    push_strength = _extract_strength(push_result, "push")
    pull_strength = _extract_strength(pull_result, "pull")
    push_highs = _extract_high_signals(push_result)
    pull_highs = _extract_high_signals(pull_result)
    roi_met = roi["threshold_met"]
    annual_saving = roi["annual_saving"]
    net = roi["net"]

    verdict = _SCENARIO_VERDICT.get(scenario)

    # ── Deterministic: STAY scenarios ─────────────────────────────────────────
    if scenario == "hard_compliance_failure":
        block = _extract_compliance_block(compliance_result)
        analysis = (
            f"Compliance check fails: {block}. "
            "Hard compliance blocks are non-negotiable — no ROI or signal weighting "
            "applies when mandatory requirements are unmet. "
            "Competitor cannot be adopted regardless of push or pull signal strength."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "fluff_update":
        pull_desc = pull_highs[0] if pull_highs else "minor improvements"
        analysis = (
            f"Compliance passes but the signal describes only cosmetic or low-impact updates. "
            f"Pull signals are weak — {pull_desc[:80]} does not resolve meaningful push issues. "
            "Insufficient business case for migration; stay with the current tool."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "irrelevant_change":
        analysis = (
            "Compliance passes but the incoming signal describes changes outside "
            "Meridian's category requirements. "
            f"Push signals ({push_strength.lower()}) are not addressed by the competitor update. "
            "No trigger for switching — the signal has no bearing on Meridian's decision."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "negative_signal_buried":
        push_desc = push_highs[0] if push_highs else "a buried limitation"
        analysis = (
            "Compliance passes but the signal contains a buried negative: "
            f"{push_desc[:100]}. "
            "Despite apparent pull signals, this hidden limitation materially weakens the case. "
            "Signal weight does not justify the migration cost and disruption."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "current_tool_rally":
        push_desc = push_highs[0] if push_highs else "previously flagged issues"
        analysis = (
            "Compliance passes but the current tool has announced improvements that address "
            f"{push_desc[:80].lower()}. "
            "Push signals are diminished as the vendor is delivering on the roadmap. "
            "The competitor's pull case does not overcome a current tool now resolving its pain points."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    # ── Deterministic: SWITCH scenarios ───────────────────────────────────────
    if scenario == "pull_dominant":
        pull_desc = pull_highs[0] if pull_highs else "strong feature advantages"
        push_desc = push_highs[0] if push_highs else "current tool limitations"
        roi_note = (
            f"ROI threshold met (net £{net}/yr)." if roi_met
            else f"Direct saving does not meet threshold (net £{net}/yr) but operational gains from resolving {push_desc[:60].lower()} justify the switch."
        )
        analysis = (
            f"Compliance passes and pull signals are strong — {pull_desc[:80]} directly resolves "
            f"key push issues ({push_desc[:60].lower()}). "
            f"{roi_note} "
            "Switching is justified on pull signal strength alone."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "push_dominant":
        push_desc = push_highs[0] if push_highs else "critical current tool failures"
        pull_desc = pull_highs[0] if pull_highs else "competitor resolution of those issues"
        roi_note = (
            f"ROI threshold met (net £{net}/yr)." if roi_met
            else f"Operational gains from resolving {push_desc[:60].lower()} outweigh the direct cost picture (net £{net}/yr)."
        )
        analysis = (
            f"Compliance passes with strong push signals — {push_desc[:80]} is a high-priority issue. "
            f"Competitor directly addresses this: {pull_desc[:70]}. "
            f"{roi_note}"
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "shelfware_case":
        analysis = (
            f"Compliance passes and shelfware is flagged — inactive seats represent ongoing wasted spend. "
            f"Competitor offers equivalent capability at a lower per-seat cost. "
            f"{'ROI threshold met (net £' + str(net) + '/yr).' if roi_met else 'Eliminating shelfware spend justifies the switch even where direct net saving is limited (net £' + str(net) + '/yr).'}"
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "hold_resolved":
        pull_desc = pull_highs[0] if pull_highs else "previously missing requirements"
        analysis = (
            "Compliance passes and the conditions that previously warranted a HOLD have been resolved: "
            f"{pull_desc[:90]}. "
            f"{'ROI threshold now met (net £' + str(net) + '/yr).' if roi_met else 'Operational gains justify the switch despite the direct saving (net £' + str(net) + '/yr).'} "
            "The switch case that was pending is now actionable."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    # ── Deterministic: HOLD scenario ──────────────────────────────────────────
    if scenario == "competitor_nearly_ready":
        pull_desc = pull_highs[0] if pull_highs else "the target feature set"
        analysis = (
            f"Compliance passes and pull signals are promising — {pull_desc[:80]}. "
            "However, the competitor is not yet fully production-ready: key elements remain "
            "on the roadmap or in beta. "
            "Hold pending confirmation that the competitor has shipped and stabilised."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: HOLD"

    # ── Ambiguous scenarios: derive from signals + ROI ────────────────────────
    if not comp_pass:
        block = _extract_compliance_block(compliance_result)
        analysis = (
            f"Compliance check fails: {block}. "
            "Hard compliance blocks prevent adoption regardless of signal balance."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    # Both signals, price_hike_only, dual_improvement
    pull_desc = pull_highs[0] if pull_highs else "some competitor advantages"
    push_desc = push_highs[0] if push_highs else "current tool issues"

    if roi_met and pull_strength in ("STRONG", "MODERATE") and push_strength in ("STRONG", "MODERATE"):
        analysis = (
            f"Compliance passes with {push_strength.lower()} push ({push_desc[:70].lower()}) "
            f"and {pull_strength.lower()} pull ({pull_desc[:70].lower()}). "
            f"ROI threshold met (net £{net}/yr) — switch is justified."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if not roi_met and pull_strength == "STRONG" and push_strength == "STRONG":
        analysis = (
            f"Compliance passes with strong push ({push_desc[:70].lower()}) and strong pull "
            f"({pull_desc[:70].lower()}). "
            f"ROI threshold is not met on direct saving alone (net £{net}/yr) but signal "
            "strength is sufficient to justify the switch on operational grounds."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if pull_strength == "WEAK" or push_strength == "WEAK":
        weak_side = (
            "pull signals are weak — competitor does not resolve key issues"
            if pull_strength == "WEAK"
            else "push signals are weak — current tool is performing adequately"
        )
        analysis = (
            f"Compliance passes but {weak_side}. "
            f"{'ROI threshold not met (net £' + str(net) + '/yr) — ' if not roi_met else ''}"
            "signal weight does not justify migration cost and disruption."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    # Mixed moderate signals → HOLD
    analysis = (
        f"Compliance passes with {push_strength.lower()} push and {pull_strength.lower()} pull. "
        f"{'ROI threshold met (net £' + str(net) + '/yr) but ' if roi_met else 'ROI threshold not met (net £' + str(net) + '/yr) and '}"
        "signal balance is inconclusive — hold pending a clearer pull case or contract timing."
    )
    return f"ANALYSIS: {analysis}\nVERDICT: HOLD"


# ── New user message builder ───────────────────────────────────────────────────

def _build_new_user_msg(
    header: str,
    compliance_result: str,
    push_result: str,
    pull_result: str,
    roi_line: str,
) -> str:
    comp_line = compliance_result.split("\n")[0].strip()
    push_strength = _extract_strength(push_result, "push")
    pull_strength = _extract_strength(pull_result, "pull")
    push_bullets = _extract_bullets(push_result)
    pull_bullets = _extract_bullets(pull_result)

    return (
        f"{header}\n\n"
        f"Compliance: {comp_line}\n\n"
        f"Push signals [{push_strength}]:\n{push_bullets}\n\n"
        f"Pull signals [{pull_strength}]:\n{pull_bullets}\n\n"
        f"{roi_line}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate verdict traces with ANALYSIS+VERDICT format.")
    parser.add_argument("--traces-dir", default="training/traces_fresh",
                        help="Directory containing traces.jsonl")
    args = parser.parse_args()

    traces_path = _PROJECT_ROOT / args.traces_dir / "traces.jsonl"
    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load all traces
    traces = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    print(f"Loaded {len(traces)} traces from {traces_path}")

    # Group by source_file for cross-step access
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for t in traces:
        m = t["metadata"]
        by_source[m["source_file"]][m["step"]] = t

    # Rebuild traces, replacing verdict records
    new_traces = []
    updated = skipped = 0

    from agent.prompts import SYS_VERDICT  # noqa: E402 — import after path setup

    for trace in traces:
        meta = trace["metadata"]
        if meta["step"] != "verdict":
            new_traces.append(trace)
            continue

        source = meta["source_file"]
        scenario = meta["scenario"]
        group = by_source[source]

        comp_trace = group.get("compliance")
        push_trace = group.get("push")
        pull_trace = group.get("pull")

        if not all([comp_trace, push_trace, pull_trace]):
            new_traces.append(trace)
            skipped += 1
            continue

        compliance_result = comp_trace["messages"][-1]["content"]
        push_result = push_trace["messages"][-1]["content"]
        pull_result = pull_trace["messages"][-1]["content"]

        # ROI + header from existing verdict user message
        old_user = trace["messages"][1]["content"]
        roi = _parse_roi(old_user)
        header = _header_line(old_user)

        # Build new user message and ANALYSIS+VERDICT response
        new_user = _build_new_user_msg(
            header, compliance_result, push_result, pull_result, roi["roi_line"]
        )
        new_assistant = _make_verdict(
            scenario, compliance_result, push_result, pull_result, roi
        )

        new_traces.append({
            "messages": [
                {"role": "system", "content": SYS_VERDICT},
                {"role": "user",   "content": new_user},
                {"role": "assistant", "content": new_assistant},
            ],
            "metadata": meta,
        })
        updated += 1

    # Write back
    with traces_path.open("w", encoding="utf-8") as f:
        for t in new_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Updated {updated} verdict traces, skipped {skipped}")
    print(f"Written {len(new_traces)} total traces to {traces_path}")

    # Quick sanity check: show one updated verdict trace
    for t in new_traces:
        if t["metadata"]["step"] == "verdict":
            print("\n=== Sample updated verdict trace ===")
            print(f"Scenario: {t['metadata']['scenario']}")
            print(f"User msg (first 300): {t['messages'][1]['content'][:300]}")
            print(f"Assistant: {t['messages'][2]['content']}")
            break


if __name__ == "__main__":
    main()
