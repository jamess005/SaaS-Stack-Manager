"""
BCR (Batched Contextual Reinforcement) dataset builder.

Loads the 250 deterministic signal files from training/generated/, builds the full
context + ROI for each, and groups them into N=4 BCR batches where each batch
contains 4 different scenario types.

The resulting HuggingFace Dataset has two columns:
  prompt            — chat message list: [system, user] — used by GRPOTrainer
  expected_verdicts — list of n canonical verdict strings — passed to bcr_reward_fn

Ambiguous scenario types (both_signals, price_hike_only, dual_improvement) are
excluded because they have no clean canonical verdict for RL reward.

Usage:
    from training.bcr_dataset import build_bcr_dataset
    ds = build_bcr_dataset()
    print(len(ds), "batches")
"""

import json
import logging
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_BCR_N = 4

# Canonical verdict map — mirrors evaluate_model.py EXPECTED_VERDICT
_EXPECTED_VERDICT: dict[str, str | None] = {
    "hard_compliance_failure": "STAY",
    "fluff_update":            "STAY",
    "irrelevant_change":       "STAY",
    "negative_signal_buried":  "STAY",
    "current_tool_rally":      "STAY",
    "pull_dominant":           "SWITCH",
    "push_dominant":           "SWITCH",
    "shelfware_case":          "SWITCH",
    "hold_resolved":           "SWITCH",
    "compliance_newly_met":    "SWITCH",
    "competitor_nearly_ready": "HOLD",
    "roadmap_confirmed_hold":  "HOLD",
    "contract_renewal_hold":   "HOLD",
    "vendor_acquisition_hold": "HOLD",
    "pilot_in_progress_hold":  "HOLD",
    # Excluded — ambiguous
    "both_signals":            None,
    "price_hike_only":         None,
    "dual_improvement":        None,
}

_BCR_INSTRUCTION = (
    "Evaluate each case independently. "
    "For each case output exactly one line in this format:\n"
    "[N] ANALYSIS: <2-4 sentences reasoning> VERDICT: SWITCH|STAY|HOLD\n\n"
    "Example output for 2 cases:\n"
    "[1] ANALYSIS: The competitor ships EUR invoicing resolving the top push issue; "
    "ROI threshold met with £1,400 annual net saving. VERDICT: SWITCH\n"
    "[2] ANALYSIS: The competitor lacks SSO on all tiers — a hard compliance block "
    "for a 150-seat org. ROI and features are irrelevant while this block stands. VERDICT: STAY\n\n"
)


# ── Filename parsing ───────────────────────────────────────────────────────────

def _parse_stem(stem: str) -> tuple[str, str, str] | None:
    """Return (category, competitor_slug, scenario) or None if unparseable."""
    from agent.context_loader import VALID_CATEGORIES
    from training.generate_signals import SCENARIO_TYPES

    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat + "_"):
            remainder = stem[len(cat) + 1:]
            for scenario in sorted(SCENARIO_TYPES, key=len, reverse=True):
                if remainder.endswith("_" + scenario):
                    competitor_slug = remainder[:-(len(scenario) + 1)]
                    if competitor_slug:
                        return cat, competitor_slug, scenario
    return None


# ── Case text builder ──────────────────────────────────────────────────────────

def _build_case_text(context: dict, roi_result: dict, signal: dict) -> str:
    """
    Build the compact per-case text for a BCR batch entry.

    Mirrors the logic in model_runner._build_lean_user, adding compliance_changes
    so the model sees compliance information for hard_compliance_failure scenarios.
    """
    from agent.prompts import _CATEGORY_RULES_COMPACT
    from agent.signal_interpreter import (
        signal_compliance_changes,
        signal_competitor_changes,
        signal_current_tool_status,
        signal_notes,
    )

    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    issues = context["current_stack_entry"].get("known_issues", [])
    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none)"

    comp_changes = signal_competitor_changes(signal)
    tool_changes = signal_current_tool_status(signal)
    notes = signal_notes(signal)
    compliance_changes = signal_compliance_changes(signal)

    comp_text = "\n".join(f"- {c}" for c in comp_changes) if comp_changes else "(none)"
    tool_change_text = (
        "\n".join(f"- {c}" for c in tool_changes) if tool_changes else "(unchanged this period)"
    )

    category_rules = _CATEGORY_RULES_COMPACT.get(category, "")

    roi_summary = (
        f"Migration: £{roi_result['migration_cost_one_time']:.0f}, "
        f"Annual net: £{roi_result['annual_net_gbp']:.0f}, "
        f"Threshold: {'MET' if roi_result['roi_threshold_met'] else 'NOT MET'}"
    )

    text = (
        f"Category: {category} — current tool: {tool_name}\n"
        f"{category_rules}\n\n"
        f"Current tool known issues:\n{issues_text}\n\n"
        f"Changes this period:\n"
        f"  Current tool: {tool_change_text}\n"
        f"  Competitor: {comp_text}\n"
    )
    if notes:
        text += f"\nBuried signals / notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    if compliance_changes:
        text += f"\nCompliance changes: {compliance_changes}\n"
    text += f"\nROI: {roi_summary}"
    return text


# ── Example loading ────────────────────────────────────────────────────────────

def _load_deterministic_examples() -> list[dict]:
    """
    Load and process all deterministic signal files.

    Returns a list of dicts with keys:
        case_text — formatted string for one BCR case slot
        expected  — canonical verdict string
        scenario  — scenario type name
    """
    from agent.context_loader import load_context
    from agent.roi_calculator import calculate_roi, extract_pass1_vars
    from agent.signal_interpreter import parse_signal_payload

    examples = []
    skipped = 0

    for path in sorted(_GENERATED_DIR.glob("*.json")):
        parsed = _parse_stem(path.stem)
        if parsed is None:
            skipped += 1
            continue

        category, competitor_slug, scenario = parsed
        expected = _EXPECTED_VERDICT.get(scenario)
        if expected is None:
            continue  # ambiguous — skip

        try:
            signal_raw = path.read_text(encoding="utf-8")
            signal = json.loads(signal_raw)
            context = load_context(category, competitor_slug, _DATA_ROOT)
            pass1 = extract_pass1_vars(context, parse_signal_payload(signal_raw))
            roi = calculate_roi(pass1)
            case_text = _build_case_text(context, roi, signal)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            skipped += 1
            continue

        examples.append({
            "case_text": case_text,
            "expected": expected,
            "scenario": scenario,
        })

    logger.info(
        "Loaded %d deterministic examples (%d skipped)", len(examples), skipped
    )
    return examples


# ── Batch builder ──────────────────────────────────────────────────────────────

def _interleave(lists: list[list]) -> list:
    """Round-robin interleave: [a0,b0,c0,..., a1,b1,c1,..., ...]"""
    result = []
    max_len = max(len(l) for l in lists) if lists else 0
    for i in range(max_len):
        for lst in lists:
            if i < len(lst):
                result.append(lst[i])
    return result


def build_bcr_dataset(n: int = _BCR_N, seed: int = 42):
    """
    Build a HuggingFace Dataset of BCR batches.

    Each record:
        prompt            — [system_msg, user_msg] for GRPOTrainer
        expected_verdicts — list of n canonical verdicts for bcr_reward_fn

    Args:
        n:    Cases per batch (default 4).
        seed: Random seed for shuffling within scenario groups.

    Returns:
        datasets.Dataset with len = floor(total_examples / n).
    """
    from datasets import Dataset
    from agent.prompts import SYS_VERDICT_LEAN

    examples = _load_deterministic_examples()

    # Group by scenario type and shuffle within each group
    by_scenario: dict[str, list[dict]] = {}
    for ex in examples:
        by_scenario.setdefault(ex["scenario"], []).append(ex)

    rng = random.Random(seed)
    for grp in by_scenario.values():
        rng.shuffle(grp)

    # Round-robin interleave across scenario types so consecutive examples differ
    interleaved = _interleave(list(by_scenario.values()))

    # Split into batches of n; each batch spans n consecutive scenario types
    batches = [interleaved[i : i + n] for i in range(0, len(interleaved) - n + 1, n)]

    # Drop any batch where scenario types are not all distinct (shouldn't happen
    # with ≥n scenario types, but guards against edge cases)
    batches = [b for b in batches if len({ex["scenario"] for ex in b}) == n]

    records = []
    for batch in batches:
        case_blocks = "\n\n".join(
            f"--- CASE {i} ---\n{ex['case_text']}" for i, ex in enumerate(batch, 1)
        )
        user_content = _BCR_INSTRUCTION + case_blocks

        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYS_VERDICT_LEAN},
                    {"role": "user", "content": user_content},
                ],
                "expected_verdicts": [ex["expected"] for ex in batch],
            }
        )

    logger.info("Built %d BCR batches (N=%d) from %d examples", len(records), n, len(examples))
    return Dataset.from_list(records)
