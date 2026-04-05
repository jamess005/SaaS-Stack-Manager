"""
Phase 2 — Market signal generation script.

Uses local Llama 3.1 8B (4-bit NF4 quantisation) to generate market signal deltas
for each (category, competitor, scenario) combination.

Each signal represents a CHANGE — what's new about the competitor (pull signal) and/or
what's degrading about the current tool (push signal). The signal is NOT a full product
summary. The model already receives the full baseline context (current tool state,
competitor capabilities, usage metrics, business rules) via context_loader.py at
inference time. The signal is the delta that prompts a re-evaluation.

In a full system, a web scraper extracts release notes and condenses them into this
format; here we generate them synthetically.

Usage:
    # Generate all files (25 competitors × 13 scenarios)
    python training/generate_signals.py --all

    # Generate for one scenario across all competitors
    python training/generate_signals.py --scenario dual_improvement

    # Generate for one combination
    python training/generate_signals.py --category finance --competitor ledgerflow --scenario pull_dominant

    # Dry-run (prints prompt, no GPU)
    python training/generate_signals.py --dry-run --scenario dual_improvement

    # Limit to first N per category
    python training/generate_signals.py --all --limit 3

Environment (required for ROCm on RX 7800 XT):
    HSA_OVERRIDE_GFX_VERSION=11.0.0
    ROCR_VISIBLE_DEVICES=0
    HIP_VISIBLE_DEVICES=0
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# Project root is one level up from training/
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_OUTPUT_DIR = _PROJECT_ROOT / "training" / "generated"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.model_runner import load_model  # noqa: E402

# Use Qwen-1.5B at bf16 for signal generation — avoids ROCm bitsandbytes OOM
# that occurs during Llama-8B 4-bit NF4 weight conversion.
_SIGNAL_GEN_MODEL = "/home/james/ml-proj/models/qwen2.5-1.5b-instruct"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCENARIO_TYPES = [
    "pull_dominant",        # Competitor has strong feature pull — straightforward SWITCH case
    "push_dominant",        # Current tool has serious issues — even mediocre competitor looks good
    "both_signals",         # Both push and pull are present — well-balanced evaluation
    "dual_improvement",     # Both improving, competitor gaining faster — stalemate/HOLD
    "current_tool_rally",   # Current tool improving faster than competitor — STAY
    "competitor_nearly_ready",  # Competitor is close but missing one key feature — HOLD case
    "price_hike_only",      # Competitor only announces a price reduction — weak pull, ROI focus
    "shelfware_case",       # Current tool has high inactive seat count — push = shelfware
    "fluff_update",         # Competitor announces mostly marketing buzzwords, no substance
    "negative_signal_buried",  # Competitor buries a limitation in footnotes / tier caveat
    "hard_compliance_failure",  # Competitor fails a hard compliance requirement (SSO/SOC2/residency)
    "hold_resolved",        # Competitor resolves the condition that previously triggered a HOLD
    "irrelevant_change",    # Competitor ships real feature that's irrelevant to this business
    # ── New scenarios added for class balance (1:1:1 STAY/SWITCH/HOLD) ──────────
    "compliance_newly_met",     # SWITCH: competitor acquires cert that was the sole blocker
    "roadmap_confirmed_hold",   # HOLD: competitor commits to missing feature — confirmed Q-date
    "contract_renewal_hold",    # HOLD: current contract renews in 60 days — hold for window
    "vendor_acquisition_hold",  # HOLD: current vendor being acquired — wait for roadmap clarity
    "pilot_in_progress_hold",   # HOLD: 30-day POC underway — hold pending proof of delivery
]

# Scenario-specific framing instructions for the generation prompt.
# Each scenario instructs the LLM to produce a different mix of pull (competitor
# improvement) and push (current tool degradation) signals as DELTAS.
_SCENARIO_INSTRUCTIONS: dict[str, str] = {
    "pull_dominant": (
        "PULL-HEAVY. The competitor has just shipped 2-3 NEW features (pick exactly 2-3 "
        "from the feature list that directly address the current tool's known issues). "
        "Treat ALL OTHER features as pre-existing baseline — do NOT list them. "
        "Only list the 2-3 you picked as newly released. Price unchanged or slightly reduced. "
        "Migration tooling available. The CURRENT TOOL STATUS section: 'No change — "
        "existing issues persist.'"
    ),
    "push_dominant": (
        "PUSH-HEAVY. COMPETITOR CHANGES section: write ONE minor update (e.g. "
        "'Stability patch v3.2.1 released' or 'Minor UI refresh — no new features'). "
        "Do NOT list ANY baseline features as changes. "
        "CURRENT TOOL STATUS section MUST contain 2-3 bullet points describing how "
        "known issues have WORSENED recently — with concrete metrics (e.g. 'API failures "
        "increased from 50/day to 200/day', 'support response time degraded from 24h to "
        "72h'). Do NOT write 'No change' for this scenario. "
        "NOTES section: migration info or caveats only — NO degradation detail."
    ),
    "both_signals": (
        "BALANCED. The competitor has shipped one meaningful new feature AND reduced pricing "
        "slightly. Simultaneously, the current tool has developed a new problem or an existing "
        "issue has escalated. Both pull (competitor improvement) and push (current tool "
        "degradation) signals are present. List the competitor's specific new feature and "
        "price change, plus the current tool's new/worsened issue."
    ),
    "competitor_nearly_ready": (
        "HOLD SIGNAL. The competitor announces a key feature is 'in beta' or 'expected Q2 "
        "2025' — NOT yet GA. This feature directly addresses the current tool's main gap. "
        "Other baseline capabilities are unchanged. Price is competitive. The trigger creates "
        "a HOLD: compelling direction but not ready to switch today."
    ),
    "price_hike_only": (
        "PRICE-ONLY. The competitor is reducing its price by 10-15%. No new features shipped. "
        "State the price change as 'from £X to £Y'. The current tool's pricing is unchanged. "
        "This is a purely financial signal."
    ),
    "shelfware_case": (
        "SHELFWARE PUSH. COMPETITOR CHANGES section: write 'No new features. Minor "
        "maintenance update only.' Do NOT list ANY baseline features as changes. "
        "The main signal is in CURRENT TOOL STATUS: inactive seats have INCREASED "
        "(invent a specific number, e.g. 'inactive seats rose from 45 to 68'), feature "
        "adoption is declining, and the team is paying for unused capacity. The competitor "
        "offers per-active-user pricing that would eliminate the wasted seat cost. "
        "Put ALL utilisation detail in CURRENT TOOL STATUS — not NOTES."
    ),
    "fluff_update": (
        "NO SIGNAL. The competitor announces an update but nothing concrete has changed. "
        "In the COMPETITOR CHANGES section, use ONLY vague phrases: 'enhanced workflows', "
        "'improved performance', 'streamlined experience', 'optimised dashboards'. "
        "Do NOT reference any specific feature names from the competitor data. "
        "Pricing unchanged. No new compliance certifications. The current tool is stable — "
        "no changes. This trigger should produce no actionable signals."
    ),
    "negative_signal_buried": (
        "BURIED NEGATIVE. The competitor announces an attractive change (price cut or new "
        "feature) but a significant limitation applies: e.g. 'SSO available on Enterprise "
        "tier only', 'EU data residency requires Advanced plan', 'feature in preview — GA "
        "date TBD'. State the attractive change first, then the limitation in NOTES. "
        "The limitation is relevant to compliance requirements."
    ),
    "hard_compliance_failure": (
        "COMPLIANCE BLOCK. The competitor announces an attractive update (new feature, "
        "competitive pricing) but has an explicit, unavoidable compliance gap: e.g. "
        "'UK data residency not available', 'SOC2 certification in progress — expected H2 "
        "2025', 'no SSO on any tier', 'no audit log'. State the compliance block clearly "
        "in the COMPLIANCE section. The current tool is stable."
    ),
    "hold_resolved": (
        "HOLD RESOLVED. A feature that was previously roadmapped, in beta, or marked "
        "'coming soon' is now generally available. State it as a GA release with a specific "
        "date. This resolves a prior HOLD condition. Price is unchanged. The current tool "
        "has no new changes — its existing issues persist."
    ),
    "irrelevant_change": (
        "IRRELEVANT FEATURE. The competitor has shipped a genuine, concrete new feature, "
        "but it has NO relevance to a UK B2B management consultancy (150 staff, office-based, "
        "consulting projects). Invent a feature suited to a different industry — retail POS, "
        "manufacturing floor scheduling, healthcare patient records, restaurant table management, "
        "or construction site tracking. The feature must be specific and real-sounding (not vague), "
        "but completely useless for Meridian's business. Current tool is stable. Price unchanged. "
        "No compliance changes."
    ),
    "dual_improvement": (
        "BOTH IMPROVING — COMPETITOR EDGE. Both the current tool AND the competitor have "
        "shipped genuine improvements in this period, but the competitor has advanced more. "
        "CURRENT TOOL STATUS: pick 1-2 items from the KNOWN ISSUES list above and describe "
        "how they have been PARTIALLY addressed — improved but not fully resolved. Use "
        "specific metrics, version numbers, or technical details relevant to THIS tool in "
        "THIS category. Do NOT use generic phrases — every improvement must be specific to "
        "the product named above. "
        "COMPETITOR CHANGES: pick 2-3 items from the competitor's FEATURE list above that "
        "represent NEW capabilities shipped since the last evaluation. These should be "
        "substantive features that advance the competitor's position beyond the current tool. "
        "Use the exact feature names from the competitor data. "
        "The net result: the current tool is improving but the competitor has outpaced it. "
        "Pricing: unchanged for both sides. "
        "This scenario typically produces a HOLD — trajectory warrants reassessment in 3 months."
    ),
    "current_tool_rally": (
        "CURRENT TOOL RALLY — STAY CASE. The current tool has shipped significant improvements "
        "that directly address its biggest pain points, while the competitor has only shipped "
        "minor updates. "
        "CURRENT TOOL STATUS: pick 2-3 items from the KNOWN ISSUES list above and describe "
        "how they have been FULLY resolved or substantially improved. Use specific metrics, "
        "version numbers, or technical details relevant to THIS tool in THIS category. "
        "Do NOT use generic phrases — each improvement must name the specific issue that was "
        "fixed and how. These improvements materially reduce the case for switching. "
        "COMPETITOR CHANGES: write ONE minor update — a small maintenance patch, documentation "
        "refresh, or single minor feature that does not change the competitive picture. "
        "Do NOT list ANY baseline features as changes. "
        "The net result: the push signals that previously justified evaluating alternatives "
        "are now largely resolved. The current tool has closed the gap. "
        "Pricing: unchanged for both sides. "
        "This scenario produces a STAY verdict — the urgency to switch has been removed."
    ),
    # ── New scenarios for class balance ──────────────────────────────────────────
    "compliance_newly_met": (
        "COMPLIANCE UNLOCKED — SWITCH CASE. The competitor has just acquired a compliance "
        "certification that was previously missing and was the sole hard block preventing a "
        "switch. Name the specific certification acquired (e.g. 'SOC2 Type II certification "
        "achieved 2025-03-01', 'UK data residency now available in London region', 'SSO via "
        "SAML 2.0 now live on all tiers'). The certification must directly resolve one of the "
        "hard compliance requirements in the global business rules. "
        "Combine this with 1-2 existing features that now become accessible (previously blocked "
        "by compliance). "
        "CURRENT TOOL STATUS: existing push issues persist unchanged. "
        "PRICING: unchanged. "
        "The net result: the previously insurmountable compliance block is now resolved, and "
        "the existing ROI case or operational gains justify switching. Verdict should be SWITCH."
    ),
    "roadmap_confirmed_hold": (
        "HOLD — CONFIRMED ROADMAP COMMITMENT. The competitor announces that a specific missing "
        "feature (one that directly addresses the current tool's main push issue) is on the "
        "confirmed product roadmap with a named delivery quarter — e.g. 'confirmed for Q3 2025', "
        "'scheduled for release in August 2025'. This is NOT yet in beta — it is a firm "
        "commitment with a date. "
        "COMPETITOR CHANGES: state the roadmap commitment with the specific feature name and "
        "quarter. No other feature changes. "
        "CURRENT TOOL STATUS: existing issues persist, no improvement. "
        "PRICING: unchanged. "
        "The trigger creates a HOLD: the case for switching is real and the timeline is "
        "concrete — reassess in 60 days when the feature approaches its delivery date."
    ),
    "contract_renewal_hold": (
        "HOLD — CONTRACT RENEWAL WINDOW. The current tool's annual contract comes up for "
        "renewal in 60 days. The competitor has shipped meaningful improvements and the ROI "
        "case is borderline positive, but switching mid-contract would incur early termination "
        "fees that wipe out the saving. "
        "COMPETITOR CHANGES: 1-2 genuine new features that improve the competitive picture "
        "without being decisive on their own. "
        "CURRENT TOOL STATUS: existing push issues persist. "
        "NOTES: state the contract renewal timing explicitly — e.g. 'current contract renews "
        "2025-09-01 — early exit penalty of £X applies before that date'. Include the penalty "
        "amount (invent a plausible figure between £500 and £2,000). "
        "PRICING: unchanged. "
        "The verdict is HOLD: wait for the renewal window to switch cleanly without penalty."
    ),
    "vendor_acquisition_hold": (
        "HOLD — VENDOR ACQUISITION UNCERTAINTY. The current tool's vendor has just announced "
        "it is being acquired by a larger company. The acquisition introduces roadmap uncertainty "
        "— it is unclear which features will be retained, discontinued, or repriced. "
        "CURRENT TOOL STATUS: the acquisition announcement is the main signal — state it "
        "clearly with acquirer name (invent a plausible corporate name) and expected close "
        "date. Existing issues persist. "
        "COMPETITOR CHANGES: the competitor has shipped a minor improvement (1 feature) — "
        "not decisive on its own. "
        "NOTES: state that the acquiring company has a track record of sunsetting niche "
        "products or significantly increasing prices post-acquisition. "
        "PRICING: unchanged for now but flagged as 'under review post-acquisition'. "
        "The verdict is HOLD: the acquisition risk is real but the competitor isn't ready "
        "enough to switch today — reassess in 60 days when acquisition terms are clearer."
    ),
    "pilot_in_progress_hold": (
        "HOLD — PROOF OF CONCEPT IN PROGRESS. A 30-day pilot of the competitor is currently "
        "underway following last quarter's evaluation. The pilot has not yet concluded. "
        "COMPETITOR CHANGES: the competitor has shipped 1 improvement during the pilot period "
        "(a minor enhancement — not a decisive new feature). "
        "CURRENT TOOL STATUS: existing issues persist. No deterioration during the pilot. "
        "NOTES: state that the pilot is ongoing — e.g. 'Pilot started 2025-02-01, concludes "
        "2025-03-03. Integration testing with PayAxis/Azure AD in progress. Data migration "
        "dry-run scheduled for week 3.' Include one specific unresolved concern from the pilot "
        "(e.g. 'API rate limits under load not yet validated', 'GDPR data transfer mechanism "
        "under legal review'). "
        "PRICING: pilot pricing in effect — full pricing TBD on contract. "
        "The verdict is HOLD: do not switch until the pilot concludes and the open items "
        "are resolved."
    ),
}

# Known competitors per category (derived from data/competitors/)
_COMPETITORS: dict[str, list[str]] = {
    "crm": ["pipelineiq", "closerhub", "velocitycrm", "clientpulse", "dealstream"],
    "hr": ["workforge", "teamledger_hr", "hrnest", "crewplan", "shiftcore"],
    "finance": ["ledgerflow", "novapay", "clearbooks_pro", "exactspend", "paytrek"],
    "project_mgmt": ["flowboard", "sprintdesk", "teamsync_projects", "projectaxis", "sprintloop"],
    "analytics": ["datalens", "clearview_analytics", "pulsemetrics", "metricflux", "prism_bi"],
}


def _build_inbox_generation_prompt(context: dict, scenario: str) -> list[dict]:
    """
    Build the message list for inbox trigger generation.

    The trigger is a DELTA — what changed about the competitor (pull) and/or
    what changed about the current tool (push). The model already has the full
    baseline from context_loader. The trigger is the new information that
    prompts a re-evaluation.
    """
    tool_name = context["current_stack_entry"]["tool"]
    competitor = context["competitor_data"]
    competitor_name = competitor["name"]
    competitor_cost = competitor.get("monthly_cost_gbp", "?")
    current_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    category = context["category"]

    features_text = "\n".join(f"  - {f}" for f in competitor.get("features", []))
    limitations_text = "\n".join(f"  - {l}" for l in competitor.get("known_limitations", []))
    compliance = competitor.get("compliance", {})
    compliance_text = json.dumps(compliance, indent=2)

    known_issues_text = "\n".join(
        f"  - {issue}" for issue in context["current_stack_entry"].get("known_issues", [])
    )
    positive_signals_text = "\n".join(
        f"  - {sig}" for sig in context["current_stack_entry"].get("positive_signals", [])
    )

    # Usage metrics for push-side context
    metrics = context.get("usage_metrics_entry", {})
    metrics_text = ""
    if metrics:
        metrics_text = (
            f"  Active users: {metrics.get('avg_monthly_active_users', '?')} / "
            f"{metrics.get('total_seats', '?')} seats\n"
            f"  Utilisation: {metrics.get('utilisation_pct', '?')}%\n"
            f"  Inactive seats: {metrics.get('inactive_seats', '?')}\n"
            f"  Shelfware: {metrics.get('shelfware_flag', False)}"
        )

    scenario_instruction = _SCENARIO_INSTRUCTIONS[scenario]

    system_content = (
        "You generate market signal deltas as valid JSON for a SaaS stack evaluation pipeline. "
        "Output ONLY a valid JSON object — no markdown, no explanation, no text before or after. "
        "Each delta describes ONLY what has CHANGED — new features shipped, "
        "price changes, compliance updates, and/or current tool degradation. "
        "Do NOT restate existing baseline capabilities. "
        "No marketing language. No emotive words. Facts only."
    )

    user_content = f"""\
SCENARIO TYPE: {scenario}
SCENARIO INSTRUCTION: {scenario_instruction}

CURRENT TOOL BASELINE (what is already known — do NOT restate):
  Name: {tool_name}
  Monthly cost: £{current_cost}
  Known issues (existing — only reference if they WORSENED):
{known_issues_text}
  Positive signals (existing):
{positive_signals_text}
  Usage metrics:
{metrics_text}

COMPETITOR BASELINE (what is already known — do NOT restate):
  Name: {competitor_name}
  Category: {category}
  Monthly cost: £{competitor_cost}
  Known features (existing — only reference if they are NEW or CHANGED):
{features_text}
  Known limitations:
{limitations_text}
  Compliance:
{compliance_text}

Output ONLY this JSON, filling in the values. No other text before or after:

{{
  "scenario_type": "{scenario}",
  "category": "{category}",
  "competitor": "{competitor_name}",
  "current_tool": "{tool_name}",
  "date": "YYYY-MM-DD",
  "competitor_changes": ["change 1", "change 2"],
  "current_tool_status": ["status 1"],
  "pricing_delta": "describe any price change or 'unchanged'",
  "compliance_changes": "describe any compliance change or 'unchanged'",
  "notes": ["caveat or note"]
}}

Rules:
1. Output ONLY valid JSON — no markdown, no commentary, no text outside the braces
2. DELTA ONLY — do not restate baseline features or capabilities
3. No marketing language — no "excited", "game-changing", "revolutionary"
4. Push signals must reference specific degradation with concrete metrics
5. Pull signals must use exact feature names from the competitor data
6. Empty arrays [] when no items apply for that field
7. Follow the scenario instruction precisely
"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _generate_text(tokenizer, model, messages: list[dict], max_new_tokens: int = 350) -> str:
    """Run inference on the 8B model and return the generated text."""
    import torch  # type: ignore

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a BatchEncoding or a plain tensor
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_REQUIRED_FIELDS = {
    "scenario_type": str,
    "category": str,
    "competitor": str,
    "current_tool": str,
    "date": str,
    "competitor_changes": list,
    "current_tool_status": list,
    "pricing_delta": str,
    "compliance_changes": str,
    "notes": list,
}


def _parse_json_output(raw: str) -> dict:
    """Extract and parse a JSON object from LLM output, handling common issues."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")
    candidate = raw[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Common LLM repairs: trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON: {exc}") from exc


def _validate_trigger(data: dict) -> list[str]:
    """Check that a parsed trigger has all required fields with correct types."""
    errors = []
    for field, expected_type in _REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(
                f"Wrong type for {field}: expected {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )
    return errors


def generate_one(
    category: str,
    competitor_slug: str,
    scenario: str,
    tokenizer,
    model,
    output_dir: Path,
    dry_run: bool = False,
) -> Path | None:
    """
    Generate one inbox trigger JSON file and save it.

    Returns:
        Path to the saved file, or None if skipped/failed.
    """
    output_path = output_dir / f"{category}_{competitor_slug}_{scenario}.json"
    if output_path.exists():
        logger.info("Skipping (already exists): %s", output_path.name)
        return None

    try:
        context = load_context(category, competitor_slug, _DATA_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot load context for %s/%s: %s", category, competitor_slug, exc)
        return None

    messages = _build_inbox_generation_prompt(context, scenario)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY-RUN PROMPT: {category}/{competitor_slug}/{scenario}")
        print(f"{'='*60}")
        for msg in messages:
            print(f"\n[{msg['role'].upper()}]\n{msg['content'][:800]}...")
        return None

    logger.info("Generating: %s/%s/%s", category, competitor_slug, scenario)
    raw_text = _generate_text(tokenizer, model, messages)

    # Parse and validate JSON
    try:
        trigger = _parse_json_output(raw_text)
    except ValueError as exc:
        logger.error("JSON parse failed for %s/%s/%s: %s", category, competitor_slug, scenario, exc)
        return None

    # Force correct metadata (LLM may misspell names or omit scenario_type)
    trigger["scenario_type"] = scenario
    trigger["category"] = category
    trigger["competitor"] = context["competitor_data"]["name"]
    trigger["current_tool"] = context["current_stack_entry"]["tool"]

    errors = _validate_trigger(trigger)
    if errors:
        logger.warning("Validation issues for %s/%s/%s: %s", category, competitor_slug, scenario, errors)
        trigger.setdefault("date", "2025-01-15")
        trigger.setdefault("competitor_changes", [])
        trigger.setdefault("current_tool_status", [])
        trigger.setdefault("pricing_delta", "unchanged")
        trigger.setdefault("compliance_changes", "unchanged")
        trigger.setdefault("notes", [])
        # scenario_type is always authoritative from the caller — not left to setdefault
        trigger["scenario_type"] = scenario

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(trigger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved: %s", output_path.name)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate market signal files using Llama 3.1 8B on ROCm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/generate_signals.py --all\n"
            "  python training/generate_signals.py --scenario dual_improvement\n"
            "  python training/generate_signals.py --category finance\n"
            "  python training/generate_signals.py --category finance --competitor ledgerflow --scenario pull_dominant\n"
            "  python training/generate_signals.py --all --limit 3 --dry-run\n"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Generate all combinations.")
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES), help="Filter by category (use with --all or single run).")
    parser.add_argument("--competitor", help="Competitor slug (single run only — requires --category and --scenario).")
    parser.add_argument("--scenario", choices=SCENARIO_TYPES, help="Scenario type (filter with --all, or specify for single run).")
    parser.add_argument("--limit", type=int, default=0, help="Max files per category (0 = no limit).")
    parser.add_argument("--output-dir", default=str(_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts; do not load model or generate.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Single run requires all three specifics; batch run requires at least --all
    single_run = args.category and args.competitor and args.scenario
    batch_run = args.all or (args.scenario and not args.competitor) or (args.category and not args.competitor)
    if not single_run and not batch_run:
        parser.error(
            "Provide --all to generate everything, or filters like --scenario dual_improvement, "
            "or all three of --category, --competitor, --scenario for a single file."
        )

    tokenizer, model = None, None
    if not args.dry_run:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        tokenizer, model = load_model(model_path=_SIGNAL_GEN_MODEL, quantize_bits=0, adapter_path=None)

    if single_run and not batch_run:
        generate_one(args.category, args.competitor, args.scenario, tokenizer, model, output_dir, args.dry_run)
    else:
        # Batch: iterate all categories/competitors, applying any filters provided
        scenarios_to_run = [args.scenario] if args.scenario else SCENARIO_TYPES
        categories_to_run = {args.category: _COMPETITORS[args.category]} if args.category else _COMPETITORS
        total = 0
        for category, competitors in categories_to_run.items():
            count = 0
            for competitor_slug in competitors:
                for scenario in scenarios_to_run:
                    if args.limit and count >= args.limit:
                        break
                    result = generate_one(
                        category, competitor_slug, scenario, tokenizer, model, output_dir, args.dry_run
                    )
                    if result is not None:
                        count += 1
                        total += 1
                if args.limit and count >= args.limit:
                    break
        if not args.dry_run:
            logger.info("Generation complete. %d files written.", total)


if __name__ == "__main__":
    main()
