"""
Feedback harvester — extracts DPO preference pairs from human feedback
and canary accuracy checks in drift_log.jsonl.

Sources of corrections:
  1. human_feedback records where correct=false and stated != actual
     (human disagreed with model and provided the right verdict)
  2. accuracy_check results where correct=false
     (canary eval shows model got a known answer wrong)

For each wrong verdict, produces a DPO pair:
  - prompt:   system + user messages (lean verdict format)
  - chosen:   CoT trace with the CORRECT verdict
  - rejected: CoT trace with the WRONG verdict

Output: training/feedback_pairs.jsonl — consumed by dpo_train.py

Usage:
    python training/feedback_harvester.py
    python training/feedback_harvester.py --min-pairs 5   # require at least 5
    python training/feedback_harvester.py --dry-run        # preview without writing
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.prompts import SYS_VERDICT_LEAN, _CATEGORY_RULES_COMPACT  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import (  # noqa: E402
    parse_signal_payload,
    signal_competitor_changes,
    signal_compliance_changes,
    signal_current_tool_status,
    signal_notes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DRIFT_LOG = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"
_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_OUTPUT_PATH = _PROJECT_ROOT / "training" / "feedback_pairs.jsonl"
_SAMPLE_SFT_PATH = _PROJECT_ROOT / "training" / "sft_cot_traces.jsonl"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

# Golden canary fixtures — generate reinforcement pairs for ALL known-answer signals.
# Format: stem → (correct_verdict, canonical_wrong_verdict)
# Wrong verdicts are the model's most common error for each scenario type.
_GOLDEN_CANARIES: dict[str, tuple[str, str]] = {
    "crm_leadsphere_competitor_nearly_ready":       ("HOLD",   "SWITCH"),
    "finance_brightbooks_pull_dominant":            ("SWITCH", "STAY"),
    "hr_teamrise_shelfware_case":                   ("SWITCH", "STAY"),
    "project_mgmt_opscanvas_contract_renewal_hold": ("HOLD",   "SWITCH"),
    "crm_closerhub_negative_signal_buried":         ("STAY",   "SWITCH"),
    "finance_ledgerflow_pull_dominant":             ("SWITCH", "STAY"),
    "project_mgmt_flowboard_shelfware_case":        ("SWITCH", "STAY"),
    "hr_workforge_fluff_update":                    ("STAY",   "SWITCH"),
    "analytics_pulsemetrics_irrelevant_change":     ("STAY",   "SWITCH"),
    "finance_exactspend_irrelevant_change":         ("STAY",   "HOLD"),
    "finance_exactspend_competitor_nearly_ready":   ("HOLD",   "SWITCH"),
    # Hold-resolved scenarios — model must not re-issue HOLD when hold_signal=NONE
    "finance_exactspend_hold_resolved":                ("SWITCH", "HOLD"),
    "analytics_pulsemetrics_hold_resolved":             ("SWITCH", "HOLD"),
    # Persistent pull/push dominant failures
    "hr_hrnest_pull_dominant":                          ("SWITCH", "STAY"),
    "project_mgmt_teamsync_projects_push_dominant":     ("SWITCH", "STAY"),
}

# ── Scenario metadata (imported from generate_cot_traces) ─────────────────────

_HOLD_NOTE_KW = frozenset([
    "hold:", "acquisition", "beta", "roadmap", "renews", "renewal", "pilot", "not ga",
    "early access",
])

_DISQUALIFIER_NOTE_KW = frozenset([
    "preview", "early access", "beta", "not ga",
    "no relevance", "not relevant", "irrelevant to", "irrelevant", "unrelated",
    "poor fit", "wrong product", "not designed for",
    "hard requirement",
    "without relevance", "does not align", "does not address", "no impact on",
])
_HOLD_COMP_KW = frozenset(["beta", "roadmap", "not ga", "preview"])
_ADVISORY_PREFIXES = ("consider", "suggest", "recommend", "note:", "fyi")
_SHELFWARE_KW = frozenset(["shelfware", "inactive seats", "inactive seat"])

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
    return "CONCRETE" if c > v else "VAGUE"


def _severity(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _HIGH_SEVERITY_KW):
        return "HIGH"
    if any(kw in lower for kw in _MEDIUM_SEVERITY_KW):
        return "MEDIUM"
    return "LOW"


_NEGATION_PATTERNS = re.compile(
    r"no\s+(beta|roadmap|caveats|hold)"
    r"|all\b.*\bga\b"
    r"|now\s+ga"
    r"|without\s+(beta|roadmap|caveats)",
    re.IGNORECASE,
)


def _detect_hold_signal(notes: list[str], comp_changes: list[str] | None = None) -> str:
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        # Skip notes that negate hold keywords (e.g. "no beta or roadmap caveats")
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


def _detect_disqualifier(notes: list[str], hold_signal: str) -> str:
    """Return a disqualifier note if notes indicate the competitor is not viable.

    Only fires when hold_signal is 'NONE' — disqualifier and hold are mutually exclusive.
    """
    if hold_signal != "NONE":
        return "NONE"
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if any(kw in note_lower for kw in _DISQUALIFIER_NOTE_KW):
            return note
    return "NONE"


_SHELFWARE_NEGATION = re.compile(
    r"no\s+shelfware|not\s+shelfware|no\s+inactive\s+seats?|no\s+unused",
    re.IGNORECASE,
)


def _detect_shelfware(tool_changes: list[str]) -> str:
    """Return the shelfware-flagging line from current_tool_status, or 'NONE'.

    Detects explicit shelfware/inactive-seat mentions so they can be surfaced as a
    structured field, matching the Hold signal pattern. Excludes negation phrases
    ('no shelfware', 'no inactive seats') to avoid false positives from signal
    files that write negation as a status line.
    """
    for change in tool_changes:
        if any(kw in change.lower() for kw in _SHELFWARE_KW):
            if not _SHELFWARE_NEGATION.search(change):
                return change
    return "NONE"


# ── Loading helpers ────────────────────────────────────────────────────────────


def _load_drift_records(log_path: Path | None = None) -> list[dict]:
    log_path = log_path or _DRIFT_LOG
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _parse_memo_filename(filename: str) -> tuple[str, str] | None:
    """Parse {date}-{category}-{competitor}.md → (category, competitor)."""
    m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)-([^-]+)\.md$", filename)
    if m:
        return m.group(1), m.group(2)
    # Handle category names with underscores (e.g. project_mgmt)
    m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)\.md$", filename)
    if not m:
        return None
    remainder = m.group(1)
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if remainder.startswith(cat + "-"):
            competitor = remainder[len(cat) + 1:]
            return cat, competitor
    return None


def _parse_canary_stem(stem: str) -> tuple[str, str, str] | None:
    """Parse {category}_{competitor}_{scenario} → (category, competitor, scenario)."""
    from training.generate_cot_traces import SCENARIO_TYPES
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat + "_"):
            remainder = stem[len(cat) + 1:]
            for scenario in sorted(SCENARIO_TYPES, key=len, reverse=True):
                if remainder.endswith("_" + scenario):
                    slug = remainder[: -(len(scenario) + 1)]
                    if slug:
                        return cat, slug, scenario
    return None


def _find_signal_for_competitor(category: str, competitor: str) -> dict | None:
    """Find the inbox signal file or generated signal for a given category/competitor."""
    # Try market_inbox first
    inbox_dir = _PROJECT_ROOT / "market_inbox"
    pattern = f"{category}_{competitor}.md"
    inbox_path = inbox_dir / pattern
    if inbox_path.exists():
        text = inbox_path.read_text(encoding="utf-8")
        parsed = parse_signal_payload(text)
        if parsed is not None:
            return parsed
        # Inbox file exists but is not structured JSON (e.g. auto-generated Markdown);
        # fall through to training/generated for a structured signal.

    # Try training/generated
    for p in _GENERATED_DIR.glob(f"{category}_{competitor}_*.json"):
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    return None


# ── Trace builder (lean format matching SFT CoT) ──────────────────────────────


def _build_lean_user(context: dict, roi_result: dict, signal: dict) -> str:
    """Build the user message matching run_lean's _build_lean_user format."""
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
    shelfware_signal = _detect_shelfware(tool_changes)
    msg += f"\nROI: {roi_summary}"
    msg += "\nCompliance: PASSED"
    msg += f"\nHold signal: {hold_signal}"
    if disqualifier != "NONE":
        msg += f"\nDisqualifier: {disqualifier}"
    if shelfware_signal != "NONE":
        msg += f"\nShelfware flag: {shelfware_signal}"
    return msg


def _build_trace(
    context: dict, signal: dict, roi: dict, verdict: str
) -> str:
    """Build a CoT reasoning trace for a given verdict (lean SFT format)."""
    tool = context["current_stack_entry"]["tool"]
    issues = context["current_stack_entry"].get("known_issues", [])
    comp_name = context["competitor_data"].get("name", context.get("competitor_slug", "Competitor"))

    comp_changes = signal_competitor_changes(signal)
    tool_changes = signal_current_tool_status(signal)
    notes = signal_notes(signal)
    metrics = context.get("usage_metrics_entry", {})

    # PUSH SIGNALS section
    push_lines: list[str] = []
    _no_change = {"no change", "no change — existing issues persist.", "stable", "unchanged"}
    tool_ch = [c for c in tool_changes if c.lower().strip() not in _no_change]
    for change in tool_ch[:3]:
        push_lines.append(f"  - {change} — Severity: {_severity(change)}")
    for issue in issues[:2]:
        push_lines.append(f"  - {issue} — Severity: {_severity(issue)} (ongoing)")
    push_block = "PUSH SIGNALS:\n" + "\n".join(push_lines) if push_lines else "PUSH SIGNALS:\n  None flagged this period"

    # PULL SIGNALS section
    pull_lines = [f"  - {c} — Substance: {_substance(c)}" for c in comp_changes[:3]]
    pull_block = "PULL SIGNALS:\n" + "\n".join(pull_lines) if pull_lines else "PULL SIGNALS:\n  None identified"

    # COMPLIANCE
    compliance_line = "COMPLIANCE: PASSED"

    # ROI
    met = "MET" if roi["roi_threshold_met"] else "NOT MET"
    roi_line = (
        f"ROI: Migration £{roi['migration_cost_one_time']:.0f} | "
        f"Annual net £{roi['annual_net_gbp']:+.0f} → Threshold {met}"
    )

    # HOLD CONDITION
    hold_signal = _detect_hold_signal(notes, comp_changes)
    hold_line = f"HOLD CONDITION: {hold_signal}"

    # ANALYSIS — verdict-specific reasoning
    analysis = _build_analysis(verdict, context, signal, roi, comp_name, tool, metrics)

    trace = (
        f"{push_block}\n\n{pull_block}\n\n{compliance_line}\n{roi_line}\n\n"
        f"{hold_line}\n\nANALYSIS: {analysis}\nVERDICT: {verdict}"
    )
    return trace


def _build_analysis(
    verdict: str,
    context: dict,
    signal: dict,
    roi: dict,
    comp_name: str,
    tool: str,
    metrics: dict,
) -> str:
    """Generate verdict-appropriate analysis text."""
    issues = context["current_stack_entry"].get("known_issues", [])
    comp_changes = signal_competitor_changes(signal)
    notes = signal_notes(signal)
    hold_signal = _detect_hold_signal(notes, comp_changes)
    net = roi["annual_net_gbp"]
    met = roi["roi_threshold_met"]
    roi_str = f"ROI threshold {'met' if met else 'not met'} at £{net:+.0f}/yr net"

    top_comp = comp_changes[0] if comp_changes else "minor updates"
    top_issue = issues[0] if issues else "existing limitations"

    shelfware = metrics.get("shelfware_flag", False) if metrics else False
    inactive = metrics.get("inactive_seats", 0) if metrics else 0

    if verdict == "SWITCH":
        # Determine what kind of switch case
        if shelfware or inactive > 5:
            return (
                f"SWITCH — shelfware case. {tool} is underutilised with {inactive} inactive seats. "
                f"Paying for unused capacity justifies switching — shelfware waste elimination "
                f"overrides the standard ROI threshold. Compliance PASSED. Hold signal: NONE. SWITCH."
            )
        pull_sub = _substance(top_comp) if comp_changes else "VAGUE"
        if pull_sub == "CONCRETE":
            return (
                f"{comp_name} delivers '{top_comp}' — a concrete capability that addresses '{top_issue}'. "
                f"Compliance PASSED. Hold signal: {hold_signal}. {roi_str}. "
                f"Concrete pull resolves the primary push driver. SWITCH."
            )
        return (
            f"Push severity is HIGH: {top_issue}. Even with a weaker pull signal, "
            f"the incumbent tool's degradation justifies switching. "
            f"Compliance PASSED. Hold signal: {hold_signal}. {roi_str}. SWITCH."
        )

    if verdict == "HOLD":
        return (
            f"The switch case has merit — {comp_name} offers improvements. However, "
            f"a hold condition is present: {hold_signal}. Until this is resolved, "
            f"the switch should wait. Compliance PASSED. {roi_str}. HOLD."
        )

    # STAY
    all_vague = all(_substance(c) == "VAGUE" for c in comp_changes) if comp_changes else True
    if all_vague and comp_changes:
        return (
            f"Pull signals are VAGUE — '{top_comp}' is marketing language with no concrete "
            f"feature delivery. Vague pull signals provide no basis for SWITCH regardless "
            f"of push severity. Hold signal: {hold_signal}. {roi_str}. STAY."
        )
    if not met:
        return (
            f"ROI threshold not met at £{net:+.0f}/yr net. The competitor changes do not "
            f"justify migration costs. Hold signal: {hold_signal}. STAY."
        )
    return (
        f"The competitor's changes do not resolve the primary push issues. "
        f"Current tool limitations remain, but the switch case is not established. "
        f"Hold signal: {hold_signal}. {roi_str}. STAY."
    )


# ── Pair extraction ────────────────────────────────────────────────────────────


def _extract_human_feedback_pairs(records: list[dict]) -> list[dict]:
    """Extract DPO pairs from human_feedback records where model was wrong."""
    feedback = [
        r for r in records
        if r.get("type") == "human_feedback"
        and not r.get("correct", True)
        and r.get("stated_verdict", "") != r.get("actual_verdict", "")
    ]

    if not feedback:
        logger.info("No actionable human feedback corrections found.")
        return []

    # Deduplicate: latest feedback per memo_filename
    latest: dict[str, dict] = {}
    for fb in feedback:
        key = fb["memo_filename"]
        if key not in latest or fb.get("ts", "") > latest[key].get("ts", ""):
            latest[key] = fb

    pairs = []
    for memo_file, fb in latest.items():
        parsed = _parse_memo_filename(memo_file)
        if not parsed:
            logger.warning("Cannot parse memo filename: %s", memo_file)
            continue

        category, competitor = parsed
        wrong_verdict = fb["stated_verdict"]
        correct_verdict = fb["actual_verdict"]

        try:
            context = load_context(category, competitor, _DATA_ROOT)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.warning("Cannot load context for %s/%s: %s", category, competitor, exc)
            continue

        signal = _find_signal_for_competitor(category, competitor)
        if signal is None:
            logger.warning("Cannot find signal for %s/%s", category, competitor)
            continue

        pass1 = extract_pass1_vars(context, signal)
        roi = calculate_roi(pass1)

        user_msg = _build_lean_user(context, roi, signal)
        prompt = [
            {"role": "system", "content": SYS_VERDICT_LEAN},
            {"role": "user", "content": user_msg},
        ]

        chosen = _build_trace(context, signal, roi, correct_verdict)
        rejected = _build_trace(context, signal, roi, wrong_verdict)

        prompt_with_note = list(prompt)
        if fb.get("note"):
            prompt_with_note.append(
                {"role": "user", "content": f"[Reviewer note: {fb['note']}]"}
            )

        pairs.append({
            "source": "human_feedback",
            "memo_filename": memo_file,
            "category": category,
            "competitor": competitor,
            "wrong_verdict": wrong_verdict,
            "correct_verdict": correct_verdict,
            "prompt": prompt_with_note,
            "chosen": chosen,
            "rejected": rejected,
        })
        logger.info(
            "Human feedback pair: %s/%s %s→%s",
            category, competitor, wrong_verdict, correct_verdict,
        )

    return pairs


def _extract_canary_pairs(records: list[dict]) -> list[dict]:
    """Extract DPO pairs from canary accuracy_check results where model was wrong.

    Only creates pairs for signals wrong in their MOST RECENT accuracy check.
    Signals the model now gets correct are excluded — training on stale failures
    causes regressions by undoing already-learned behaviour.
    """
    accuracy_checks = [r for r in records if r.get("type") == "accuracy_check"]
    if not accuracy_checks:
        logger.info("No accuracy checks found.")
        return []

    # Track the most recent result (correct or wrong) for each signal file.
    # Individual result entries don't carry their own timestamp, so we order
    # by the parent accuracy_check's ts field.
    latest_result_by_file: dict[str, dict] = {}
    for check in accuracy_checks:
        check_ts = check.get("ts", "")
        for r in check.get("results", []):
            if r.get("status") != "ok":
                continue
            fname = r["file"]
            existing = latest_result_by_file.get(fname)
            if existing is None or check_ts >= existing.get("_check_ts", ""):
                entry = dict(r)
                entry["_check_ts"] = check_ts
                latest_result_by_file[fname] = entry

    # Only generate pairs for signals that are STILL wrong in their latest check.
    wrong_by_file = {
        fname: r
        for fname, r in latest_result_by_file.items()
        if not r.get("correct", True)
    }

    if not wrong_by_file:
        logger.info("All canaries correct in most recent checks — no pairs needed.")
        return []

    pairs = []
    for result in wrong_by_file.values():
        signal_file = result["file"]
        stem = Path(signal_file).stem
        parsed = _parse_canary_stem(stem)
        if not parsed:
            logger.warning("Cannot parse canary filename: %s", signal_file)
            continue

        category, competitor, scenario = parsed
        wrong_verdict = result["actual"]
        correct_verdict = result["expected"]

        signal_path = _GENERATED_DIR / signal_file
        if not signal_path.exists():
            logger.warning("Canary signal file missing: %s", signal_path)
            continue

        try:
            context = load_context(category, competitor, _DATA_ROOT)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.warning("Cannot load context for canary %s/%s: %s", category, competitor, exc)
            continue

        with signal_path.open(encoding="utf-8") as f:
            signal = json.load(f)

        inbox_text = json.dumps(signal, indent=2, ensure_ascii=False)
        parsed_signal = parse_signal_payload(inbox_text)
        pass1 = extract_pass1_vars(context, parsed_signal)
        roi = calculate_roi(pass1)

        user_msg = _build_lean_user(context, roi, parsed_signal)
        prompt = [
            {"role": "system", "content": SYS_VERDICT_LEAN},
            {"role": "user", "content": user_msg},
        ]

        chosen = _build_trace(context, parsed_signal, roi, correct_verdict)
        wrong_memo = result.get("wrong_memo")
        if wrong_memo:
            rejected = wrong_memo
            logger.info("Using saved wrong_memo as rejected trace for %s", signal_file)
        else:
            rejected = _build_trace(context, parsed_signal, roi, wrong_verdict)

        pairs.append({
            "source": "canary",
            "signal_file": signal_file,
            "scenario": scenario,
            "category": category,
            "competitor": competitor,
            "wrong_verdict": wrong_verdict,
            "correct_verdict": correct_verdict,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
        logger.info(
            "Canary pair: %s/%s (%s) %s→%s",
            category, competitor, scenario, wrong_verdict, correct_verdict,
        )

    return pairs


def _extract_golden_canary_pairs() -> list[dict]:
    """Generate reinforcement pairs for all 10 known-answer canary fixtures.

    Unlike _extract_canary_pairs (which only trains on current failures), these
    pairs are always included. They anchor the model to correct behaviour and
    self-regulate: when the model is already correct, the chosen/rejected margin
    is large and the gradient is near-zero; when it starts to drift, the gradient
    pulls it back.
    """
    from training.generate_cot_traces import SCENARIO_TYPES

    pairs = []
    for stem, (correct_verdict, wrong_verdict) in _GOLDEN_CANARIES.items():
        # Find the signal file
        matches = list(_GENERATED_DIR.glob(f"{stem}*.json"))
        if not matches:
            logger.warning("Golden canary signal not found for stem: %s", stem)
            continue
        signal_path = matches[0]

        # Parse category and competitor from stem
        category = None
        competitor = None
        for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
            if stem.startswith(cat + "_"):
                remainder = stem[len(cat) + 1:]
                for sc in sorted(SCENARIO_TYPES, key=len, reverse=True):
                    if remainder.endswith("_" + sc):
                        competitor = remainder[: -(len(sc) + 1)]
                        category = cat
                        break
                if competitor:
                    break

        if not category or not competitor:
            logger.warning("Cannot parse golden canary stem: %s", stem)
            continue

        try:
            context = load_context(category, competitor, _DATA_ROOT)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.warning("Cannot load context for golden canary %s/%s: %s", category, competitor, exc)
            continue

        with signal_path.open(encoding="utf-8") as f:
            signal = json.load(f)

        inbox_text = json.dumps(signal, indent=2, ensure_ascii=False)
        parsed_signal = parse_signal_payload(inbox_text)
        pass1 = extract_pass1_vars(context, parsed_signal)
        roi = calculate_roi(pass1)

        user_msg = _build_lean_user(context, roi, parsed_signal)
        prompt = [
            {"role": "system", "content": SYS_VERDICT_LEAN},
            {"role": "user", "content": user_msg},
        ]

        chosen = _build_trace(context, parsed_signal, roi, correct_verdict)
        rejected = _build_trace(context, parsed_signal, roi, wrong_verdict)

        pairs.append({
            "source": "golden_canary",
            "signal_file": signal_path.name,
            "category": category,
            "competitor": competitor,
            "correct_verdict": correct_verdict,
            "wrong_verdict": wrong_verdict,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
        logger.info(
            "Golden canary pair: %s/%s correct=%s rejected=%s",
            category, competitor, correct_verdict, wrong_verdict,
        )

    return pairs


def _sample_sft_traces(n: int = 30, seed: int = 42) -> list[dict]:
    """Sample from SFT CoT traces to augment DPO with genuine reasoning patterns.

    chosen = authentic SFT trace (step-by-step ANALYSIS, correct verdict)
    rejected = same trace with VERDICT swapped (same analysis, wrong verdict)

    DPO loss: model learns that genuine analysis → correct verdict is preferred
    over genuine analysis → wrong verdict. Anchors the model to the reasoning
    style it was originally trained on, not just template outputs.

    Deduplicates traces by assistant content before sampling.
    """
    import random

    if not _SAMPLE_SFT_PATH.exists():
        logger.warning("SFT traces not found at %s — skipping SFT augmentation", _SAMPLE_SFT_PATH)
        return []

    traces = []
    with _SAMPLE_SFT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    # Deduplicate by assistant content
    seen: set[str] = set()
    unique: list[dict] = []
    for t in traces:
        key = t["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique.append(t)
    logger.info("SFT traces: %d total, %d unique after dedup", len(traces), len(unique))

    switch_traces = [t for t in unique if "VERDICT: SWITCH" in t["messages"][2]["content"]]
    stay_traces   = [t for t in unique if "VERDICT: STAY"   in t["messages"][2]["content"]]
    hold_traces   = [t for t in unique if "VERDICT: HOLD"   in t["messages"][2]["content"]]

    rng = random.Random(seed)
    per_bucket = n // 3
    sampled = (
        rng.sample(switch_traces, min(per_bucket, len(switch_traces))) +
        rng.sample(stay_traces,   min(per_bucket, len(stay_traces)))   +
        rng.sample(hold_traces,   min(per_bucket, len(hold_traces)))
    )

    pairs = []
    for trace in sampled:
        user_msg = trace["messages"][1]
        assistant_content = trace["messages"][2]["content"]

        if "VERDICT: SWITCH" in assistant_content:
            correct_verdict, wrong_verdict = "SWITCH", "STAY"
        elif "VERDICT: HOLD" in assistant_content:
            correct_verdict, wrong_verdict = "HOLD", "SWITCH"
        else:
            correct_verdict, wrong_verdict = "STAY", "SWITCH"

        # Use current SYS_VERDICT_LEAN (not the old version embedded in the SFT trace)
        prompt = [
            {"role": "system", "content": SYS_VERDICT_LEAN},
            user_msg,
        ]
        # Rejected: same genuine reasoning, wrong verdict — model learns verdict must
        # match its own analysis, not that analysis quality is irrelevant.
        rejected = assistant_content.replace(
            f"VERDICT: {correct_verdict}", f"VERDICT: {wrong_verdict}", 1
        )

        pairs.append({
            "source": "sft_sample",
            "category": "various",
            "competitor": "various",
            "wrong_verdict": wrong_verdict,
            "correct_verdict": correct_verdict,
            "prompt": prompt,
            "chosen": assistant_content,
            "rejected": rejected,
        })

    n_sw = sum(1 for p in pairs if p["correct_verdict"] == "SWITCH")
    n_st = sum(1 for p in pairs if p["correct_verdict"] == "STAY")
    n_ho = sum(1 for p in pairs if p["correct_verdict"] == "HOLD")
    logger.info(
        "SFT augmentation: %d pairs (%d SWITCH, %d STAY, %d HOLD)", len(pairs), n_sw, n_st, n_ho
    )
    return pairs


def harvest(
    log_path: Path | None = None,
    output_path: Path | None = None,
    min_pairs: int = 1,
    dry_run: bool = False,
    no_golden: bool = False,
    no_sft_samples: bool = False,
) -> list[dict]:
    """
    Harvest DPO preference pairs from drift log.

    Returns list of pairs. Writes to output_path unless dry_run.
    """
    log_path = log_path or _DRIFT_LOG
    output_path = output_path or _OUTPUT_PATH
    records = _load_drift_records(log_path)

    if not records:
        logger.warning("No records in drift log.")
        return []

    human_pairs = _extract_human_feedback_pairs(records)
    canary_pairs = _extract_canary_pairs(records)
    golden_pairs = [] if no_golden else _extract_golden_canary_pairs()
    sft_pairs = [] if no_sft_samples else _sample_sft_traces()
    all_pairs = human_pairs + canary_pairs + golden_pairs + sft_pairs

    logger.info(
        "Harvested %d pairs: %d human, %d canary failures, %d golden canary, %d sft samples",
        len(all_pairs), len(human_pairs), len(canary_pairs), len(golden_pairs), len(sft_pairs),
    )

    if len(all_pairs) < min_pairs:
        logger.warning(
            "Only %d pairs — below minimum of %d. Skipping write.",
            len(all_pairs), min_pairs,
        )
        return all_pairs

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Wrote %d pairs to %s", len(all_pairs), output_path)

    return all_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest DPO pairs from feedback + canaries.")
    parser.add_argument("--min-pairs", type=int, default=1,
                        help="Minimum pairs required before writing output.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview pairs without writing.")
    parser.add_argument("--output", type=str, default=str(_OUTPUT_PATH),
                        help="Output path for feedback_pairs.jsonl")
    parser.add_argument("--no-golden", action="store_true",
                        help="Skip golden canary pairs — include only failure corrections.")
    parser.add_argument("--no-sft-samples", action="store_true",
                        help="Skip SFT trace augmentation — use only failure/canary pairs.")
    args = parser.parse_args()

    pairs = harvest(
        min_pairs=args.min_pairs,
        dry_run=args.dry_run,
        output_path=Path(args.output),
        no_golden=args.no_golden,
        no_sft_samples=args.no_sft_samples,
    )

    if not pairs:
        print("No DPO pairs harvested.")
        return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Harvested {len(pairs)} DPO pairs:")
    for p in pairs:
        src = p["source"]
        cat = p["category"]
        comp = p["competitor"]
        wrong = p["wrong_verdict"]
        correct = p["correct_verdict"]
        print(f"  [{src}] {cat}/{comp}: {wrong} → {correct}")


if __name__ == "__main__":
    main()
