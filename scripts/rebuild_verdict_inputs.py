"""
Rebuild verdict training traces with inputs derived from structured signal JSON files.

Fixes the training/inference distribution mismatch: the Llama teacher model
produced empty push/pull outputs (~85-90% of verdict traces), so the model
learned to decide from compliance+ROI metadata alone.  At inference the
fine-tuned model generates real push/pull bullets — an input the verdict voter
was never trained on.

This script replaces the verdict step's *user messages* with content drawn
directly from training/generated/*.json (current_tool_status → push bullets,
competitor_changes → pull bullets, compliance_changes → compliance line).

The ANALYSIS text is also updated to reference real feature names from the
signal rather than generic fallback phrases.

Run from project root:
    python scripts/rebuild_verdict_inputs.py [--traces-dir training/traces]
"""

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"

# ── Scenario → signal strength labels ─────────────────────────────────────────

_PUSH_STRENGTH = {
    "hard_compliance_failure": "STRONG",
    "push_dominant":           "STRONG",
    "pull_dominant":           "MODERATE",
    "both_signals":            "STRONG",
    "current_tool_rally":      "MODERATE",
    "competitor_nearly_ready": "STRONG",
    "fluff_update":            "STRONG",
    "irrelevant_change":       "MODERATE",
    "negative_signal_buried":  "MODERATE",
    "hold_resolved":           "STRONG",
    "shelfware_case":          "MODERATE",
    "price_hike_only":         "STRONG",
    "dual_improvement":        "MODERATE",
}

_PULL_STRENGTH = {
    "hard_compliance_failure": "STRONG",
    "push_dominant":           "MODERATE",
    "pull_dominant":           "STRONG",
    "both_signals":            "STRONG",
    "current_tool_rally":      "WEAK",
    "competitor_nearly_ready": "MODERATE",
    "fluff_update":            "WEAK",
    "irrelevant_change":       "WEAK",
    "negative_signal_buried":  "MODERATE",
    "hold_resolved":           "STRONG",
    "shelfware_case":          "STRONG",
    "price_hike_only":         "WEAK",
    "dual_improvement":        "MODERATE",
}

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
    "both_signals":            None,
    "price_hike_only":         None,
    "dual_improvement":        None,
}

_ROADMAP_WORDS = frozenset([
    "beta", "roadmap", "q1", "q2", "q3", "q4",
    "planned", "scheduled", "upcoming", "preview",
    "not yet", "expected", "coming soon",
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_roi(user_content: str) -> dict:
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


def _has_roadmap_language(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _ROADMAP_WORDS)


# ── User message builder (from structured signal data) ─────────────────────────

def _build_verdict_user(signal: dict, scenario: str, header: str, roi_line: str) -> str:
    """Build a realistic verdict voter user message from structured signal JSON."""
    push_items = signal.get("current_tool_status", [])
    pull_items  = signal.get("competitor_changes", [])
    notes       = signal.get("notes", [])

    # --- Compliance line ---
    if scenario == "hard_compliance_failure":
        comp_changes = signal.get("compliance_changes", "compliance requirements not met")
        if isinstance(comp_changes, dict):
            # Extract fields that are False (failing requirements)
            fails = [k.replace("_", " ") for k, v in comp_changes.items() if v is False]
            reason = "; ".join(fails[:3]) if fails else "required compliance controls not met"
        else:
            reason = str(comp_changes)[:120]
        compliance_line = f"FAIL — {reason}"
    else:
        compliance_line = "PASS"

    # --- Push bullets ---
    push_strength = _PUSH_STRENGTH.get(scenario, "MODERATE")
    push_lines = [f"  - {s}" for s in push_items[:5]]
    push_bullets = "\n".join(push_lines) if push_lines else "  (none)"

    # --- Pull bullets ---
    pull_strength = _PULL_STRENGTH.get(scenario, "MODERATE")

    if scenario == "competitor_nearly_ready":
        # Annotate roadmap/beta items and surface not-ready notes.
        pull_lines = []
        has_explicit_roadmap = False
        for item in pull_items[:5]:
            if _has_roadmap_language(item):
                pull_lines.append(f"  - {item} [roadmap/beta — not yet GA]")
                has_explicit_roadmap = True
            else:
                pull_lines.append(f"  - {item}")
        # Surface any notes that call out not-ready status
        not_ready_notes = [n for n in notes if _has_roadmap_language(n) or "not ready" in n.lower()]
        if not_ready_notes:
            pull_lines.append(f"  NOTE: {not_ready_notes[0]}")
        elif not has_explicit_roadmap:
            pull_lines.append(
                "  NOTE: Key elements of the switch case remain on roadmap or in beta — not yet production-ready."
            )
        pull_bullets = "\n".join(pull_lines) if pull_lines else "  (none)"

    elif scenario == "hold_resolved":
        # Show that the previously blocking condition has now shipped.
        pull_lines = [f"  - {item}" for item in pull_items[:5]]
        pull_lines.append(
            "  NOTE: Features that previously blocked the switch decision have now shipped."
        )
        pull_bullets = "\n".join(pull_lines)

    elif scenario == "current_tool_rally":
        # Pull is weak because current tool is rallying — surface that
        pull_lines = [f"  - {item}" for item in pull_items[:5]]
        pull_lines.append(
            "  NOTE: Current tool vendor is actively addressing the gaps listed above."
        )
        pull_bullets = "\n".join(pull_lines) if pull_lines else "  (none)"

    else:
        pull_bullets = "\n".join(f"  - {item}" for item in pull_items[:5]) or "  (none)"

    return (
        f"{header}\n\n"
        f"Compliance: {compliance_line}\n\n"
        f"Push signals [{push_strength}]:\n{push_bullets}\n\n"
        f"Pull signals [{pull_strength}]:\n{pull_bullets}\n\n"
        f"{roi_line}"
    )


# ── ANALYSIS + VERDICT generator (from structured signal data) ─────────────────

def _make_verdict(signal: dict, scenario: str, roi: dict) -> str:
    """Return 'ANALYSIS: ...\nVERDICT: X' using actual signal content."""
    push_items = signal.get("current_tool_status", [])
    pull_items  = signal.get("competitor_changes", [])
    notes       = signal.get("notes", [])

    push_desc = push_items[0] if push_items else "current tool limitations"
    pull_desc = pull_items[0] if pull_items else "competitor improvements"
    net       = roi["net"]
    roi_met   = roi["threshold_met"]

    # ── STAY scenarios ────────────────────────────────────────────────────────
    if scenario == "hard_compliance_failure":
        raw_cc = signal.get("compliance_changes", "compliance requirements not met")
        if isinstance(raw_cc, dict):
            fails = [k.replace("_", " ") for k, v in raw_cc.items() if v is False]
            comp_changes = "; ".join(fails[:3]) if fails else "required compliance controls not met"
        else:
            comp_changes = str(raw_cc)[:120]
        analysis = (
            f"Compliance check fails: {comp_changes}. "
            "Hard compliance blocks are non-negotiable — no ROI or signal weighting "
            "applies when mandatory requirements are unmet. "
            "Competitor cannot be adopted regardless of push or pull signal strength."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "fluff_update":
        analysis = (
            f"Compliance passes but the competitor update is cosmetic: {pull_desc[:80]}. "
            f"This does not address the underlying push issue ({push_desc[:70].lower()}). "
            "Insufficient business case for migration; stay with the current tool."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "irrelevant_change":
        analysis = (
            f"Compliance passes but the incoming signal — {pull_desc[:80]} — "
            "describes changes outside Meridian's category requirements. "
            f"The push issue ({push_desc[:70].lower()}) is not addressed. "
            "No trigger for switching; the signal has no bearing on the decision."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "negative_signal_buried":
        # Find the buried negative from notes or the last pull item
        buried = next((n for n in notes if "no " in n.lower() or "lack" in n.lower()), None)
        if buried is None and len(pull_items) > 1:
            buried = pull_items[-1]
        buried = buried or "a hidden limitation"
        analysis = (
            f"Compliance passes but the signal contains a buried negative: {buried[:100]}. "
            f"Despite apparent pull signals ({pull_desc[:70].lower()}), "
            "this hidden limitation materially weakens the case. "
            "Signal weight does not justify the migration cost and disruption."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    if scenario == "current_tool_rally":
        rally_note = next((n for n in notes if "address" in n.lower() or "improv" in n.lower()), None)
        remedy_desc = rally_note or f"improvements to {push_desc[:60].lower()}"
        analysis = (
            f"Compliance passes but the current tool is actively addressing its gaps: {remedy_desc[:100]}. "
            f"The push case is diminishing as the vendor delivers on the roadmap. "
            f"Competitor pull ({pull_desc[:70].lower()}) does not overcome a current tool now resolving its issues."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: STAY"

    # ── SWITCH scenarios ──────────────────────────────────────────────────────
    if scenario == "pull_dominant":
        roi_note = (
            f"ROI threshold met (net £{net}/yr)." if roi_met
            else f"Direct saving does not meet threshold (net £{net}/yr) but operational gains from resolving {push_desc[:60].lower()} justify the switch."
        )
        analysis = (
            f"Compliance passes and pull signals are strong — {pull_desc[:80]} directly resolves "
            f"the key push issue ({push_desc[:70].lower()}). "
            f"{roi_note} Switching is justified on pull signal strength."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "push_dominant":
        roi_note = (
            f"ROI threshold met (net £{net}/yr)." if roi_met
            else f"Operational gains from resolving {push_desc[:60].lower()} outweigh the direct cost picture (net £{net}/yr)."
        )
        analysis = (
            f"Compliance passes with strong push — {push_desc[:90]} is a critical issue. "
            f"Competitor directly addresses this: {pull_desc[:80]}. "
            f"{roi_note}"
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "shelfware_case":
        analysis = (
            "Compliance passes and shelfware is flagged — inactive seats represent ongoing wasted spend. "
            f"Competitor offers equivalent capability ({pull_desc[:70]}) at a lower per-seat cost. "
            f"{'ROI threshold met (net £' + str(net) + '/yr).' if roi_met else 'Eliminating shelfware spend justifies the switch even where direct net saving is limited (net £' + str(net) + '/yr).'}"
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    if scenario == "hold_resolved":
        resolved_note = next(
            (n for n in notes if _has_roadmap_language(n) or "shipped" in n.lower() or "ga" in n.lower()),
            None
        )
        pull_shipped = resolved_note or pull_desc
        roi_note = (
            f"ROI threshold now met (net £{net}/yr)." if roi_met
            else f"Operational gains from resolving {push_desc[:60].lower()} justify the switch (net £{net}/yr)."
        )
        analysis = (
            f"Compliance passes and the conditions that previously blocked a switch have now been resolved: "
            f"{pull_shipped[:100]}. "
            f"{roi_note} "
            "The switch case that was on hold is now actionable."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: SWITCH"

    # ── HOLD scenario ─────────────────────────────────────────────────────────
    if scenario == "competitor_nearly_ready":
        not_ready = next(
            (n for n in notes if _has_roadmap_language(n) or "not ready" in n.lower()),
            None
        )
        roadmap_item = next(
            (c for c in pull_items if _has_roadmap_language(c)),
            pull_desc
        )
        not_ready_reason = not_ready or f"key feature ({roadmap_item[:70]}) remains on roadmap or in beta"
        analysis = (
            f"Compliance passes and the competitor shows promise — {pull_desc[:80]}. "
            f"However, {not_ready_reason[:100]}. "
            "The switch case is real but the competitor has not yet shipped the full feature set. "
            "Hold pending confirmation that roadmap items are GA and stable."
        )
        return f"ANALYSIS: {analysis}\nVERDICT: HOLD"

    # ── Ambiguous scenarios: derive from ROI and strength ─────────────────────
    push_strength = _PUSH_STRENGTH.get(scenario, "MODERATE")
    pull_strength = _PULL_STRENGTH.get(scenario, "MODERATE")

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
            f"ROI threshold is not met (net £{net}/yr) but signal strength justifies the switch on operational grounds."
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

    # Mixed moderate → HOLD
    analysis = (
        f"Compliance passes with {push_strength.lower()} push and {pull_strength.lower()} pull. "
        f"{'ROI threshold met (net £' + str(net) + '/yr) but ' if roi_met else 'ROI threshold not met (net £' + str(net) + '/yr) and '}"
        "signal balance is inconclusive — hold pending a clearer pull case or contract timing."
    )
    return f"ANALYSIS: {analysis}\nVERDICT: HOLD"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild verdict training traces from structured signal JSON files."
    )
    parser.add_argument("--traces-dir", default="training/traces",
                        help="Directory containing traces.jsonl (default: training/traces)")
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

    from agent.prompts import SYS_VERDICT  # noqa: E402

    updated = skipped = bad = 0
    new_traces = []

    for trace in traces:
        meta = trace["metadata"]
        if meta["step"] != "verdict":
            new_traces.append(trace)
            continue

        source_file = meta["source_file"]
        scenario    = meta["scenario"]

        # Load the structured signal JSON
        signal_path = _GENERATED_DIR / f"{source_file}.json"
        if not signal_path.exists():
            print(f"  WARN: Signal file not found: {signal_path}", file=sys.stderr)
            new_traces.append(trace)
            skipped += 1
            continue

        with signal_path.open(encoding="utf-8") as f:
            signal = json.load(f)

        # Extract ROI line and header from existing user message (calculated deterministically)
        old_user = trace["messages"][1]["content"]
        roi      = _parse_roi(old_user)
        header   = old_user.split("\n")[0].strip()

        if not roi["roi_line"]:
            print(f"  WARN: No ROI line in trace for {source_file}", file=sys.stderr)
            new_traces.append(trace)
            skipped += 1
            continue

        new_user      = _build_verdict_user(signal, scenario, header, roi["roi_line"])
        new_assistant = _make_verdict(signal, scenario, roi)

        # Sanity check: ANALYSIS + VERDICT format
        if not new_assistant.startswith("ANALYSIS:") or "VERDICT:" not in new_assistant:
            print(f"  BAD: Malformed output for {source_file}: {new_assistant[:80]}", file=sys.stderr)
            new_traces.append(trace)
            bad += 1
            continue

        new_traces.append({
            "messages": [
                {"role": "system",    "content": SYS_VERDICT},
                {"role": "user",      "content": new_user},
                {"role": "assistant", "content": new_assistant},
            ],
            "metadata": meta,
        })
        updated += 1

    print(f"\nUpdated {updated}  Skipped {skipped}  Bad {bad}")

    # Verdict distribution
    dist: dict[str, int] = {}
    for t in new_traces:
        if t["metadata"]["step"] == "verdict":
            v = t["messages"][-1]["content"].split("VERDICT:")[-1].strip().split()[0]
            dist[v] = dist.get(v, 0) + 1
    print(f"Verdict distribution: {dict(sorted(dist.items()))}")

    # Show one sample per key scenario
    shown = set()
    for t in new_traces:
        s = t["metadata"]["scenario"]
        if t["metadata"]["step"] == "verdict" and s not in shown and s in (
            "competitor_nearly_ready", "hold_resolved", "hard_compliance_failure",
            "push_dominant", "fluff_update"
        ):
            shown.add(s)
            print(f"\n=== {s} ===")
            print(f"USER:\n{t['messages'][1]['content'][:400]}")
            print(f"ASSISTANT:\n{t['messages'][2]['content']}")

    # Write back
    with traces_path.open("w", encoding="utf-8") as f:
        for t in new_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(new_traces)} traces to {traces_path}")


if __name__ == "__main__":
    main()
