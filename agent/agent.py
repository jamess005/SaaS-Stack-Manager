"""
SaaS Stack Manager — CLI entrypoint and pipeline orchestrator.

Usage:
    python -m agent.agent <inbox_file>
    python -m agent.agent --dry-run <inbox_file>

The inbox_file must follow the naming convention {category}_{competitor}.md,
e.g. finance_ledgerflow.md or market_inbox/project_mgmt_flowboard.md.

The agent runs the full pipeline:
  1. Parse inbox filename for category + competitor
  2. Load scoped context files
  3. Load model (no-op in dry-run)
  4. Extract financial variables from context (Python — no model)
  5. ROI calculation (Python)
  6. Pass 2 — verdict memo generation
  7. Validate output, retry once on failure
  8. Write memo to outputs/
  9. Append to hold_register.json if HOLD verdict
 10. Auto-generate plain-English summary using loaded model → outputs/summaries.json
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from agent.context_loader import load_context, parse_inbox_filename
from agent.drift_tracker import check_accuracy_due, log_live_run
from agent.model_runner import load_model, run_voting
from agent.output_validator import extract_hold_metadata, extract_verdict_class, validate_verdict
from agent.roi_calculator import calculate_roi, extract_pass1_vars
from agent.signal_interpreter import parse_signal_payload

# Project root = parent of this file's directory (agent/agent.py → project root)
_PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_SUMMARIES_PATH = _PROJECT_ROOT / "outputs" / "summaries.json"

_SUMMARY_SYSTEM_PROMPT = (
    "You are helping a business team understand a software recommendation. "
    "Read the analyst memo and write 2-3 warm, plain-English sentences explaining "
    "why the recommendation makes sense for them. Focus on what the new tool does "
    "better — not technical details, compliance codes, or business rules. "
    "Be conversational and direct. No bullet points."
)


def _auto_summarise(
    memo_text: str,
    memo_filename: str,
    tokenizer,
    model,
    summaries_path: Path = _SUMMARIES_PATH,
) -> None:
    """Generate a plain-English summary for memo_text using the already-loaded model."""
    if tokenizer is None or model is None:
        return  # dry-run — no model available

    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if summaries_path.exists():
        try:
            existing = json.loads(summaries_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if memo_filename in existing:
        return  # already summarised

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": memo_text[:2000]},
    ]
    try:
        import torch
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        existing[memo_filename] = summary
        summaries_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Summary written for: %s", memo_filename)
    except Exception as exc:
        logger.warning("Summary generation skipped: %s", exc)


def _resolve_data_root() -> Path:
    return _PROJECT_ROOT / "data"


def _resolve_outputs_dir() -> Path:
    return _PROJECT_ROOT / "outputs"


def _resolve_register_path() -> Path:
    return _PROJECT_ROOT / "hold_register.json"


def _write_output(memo_text: str, category: str, competitor_slug: str, outputs_dir: Path) -> Path:
    """Write verdict memo to outputs/{date}-{category}-{competitor_slug}.md."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    filename = f"{today}-{category}-{competitor_slug}.md"
    output_path = outputs_dir / filename
    output_path.write_text(memo_text, encoding="utf-8")
    logger.info("Verdict memo written to: %s", output_path)
    return output_path


def _append_hold_register(hold_data: dict, register_path: Path) -> None:
    """
    Append a hold entry to hold_register.json.
    Creates the file initialised as [] if it does not exist.
    """
    if register_path.exists():
        with register_path.open(encoding="utf-8") as f:
            register = json.load(f)
    else:
        register = []

    # Add issued_date if not present
    if "issued_date" not in hold_data:
        hold_data["issued_date"] = datetime.date.today().isoformat()

    register.append(hold_data)
    with register_path.open("w", encoding="utf-8") as f:
        json.dump(register, f, indent=2, ensure_ascii=False)
    logger.info("Hold entry appended to: %s", register_path)


def _build_minimal_hold_entry(category: str, context: dict, competitor_slug: str) -> dict:
    """
    Build a minimal hold_register entry when extract_hold_metadata returns None.
    Uses today + 92 days as review_by (approximately 3 months).
    """
    tool_name = context["current_stack_entry"]["tool"]
    review_by = (datetime.date.today() + datetime.timedelta(days=92)).isoformat()
    return {
        "category": category,
        "current_tool": tool_name,
        "competitor": competitor_slug,
        "hold_reason": "See verdict memo for push signal details.",
        "reassess_condition": "Review memo for stated reassessment condition.",
        "issued_date": datetime.date.today().isoformat(),
        "review_by": review_by,
    }


def run_agent(
    inbox_path: Path,
    dry_run: bool = False,
    outputs_dir: Path | None = None,
    register_path: Path | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    log_path: Path | None = None,
) -> dict:
    """
    Run the full pipeline for one inbox file.

    Args:
        inbox_path: Path to the inbox trigger .md file.
        dry_run: Force dry-run mode (also activated by AGENT_DRY_RUN env var).
        outputs_dir: Override the output directory (used in tests for isolation).
        register_path: Override hold_register.json path (used in tests for isolation).
        model_path: Local path to the base model directory. If omitted, falls back
                    to the default HuggingFace model ID in load_model().
        adapter_path: Path to a fine-tuned LoRA adapter directory. Optional.
        log_path: Override drift log path (used in tests for isolation).

    Returns:
        {
            "verdict": str,               # "SWITCH", "STAY", or "HOLD"
            "output_path": Path,          # path where memo was written
            "hold_registered": bool,      # True if hold_register was updated
            "validation_errors": list,    # empty on success
        }

    Raises:
        FileNotFoundError: if inbox_path does not exist.
        ValueError: if filename cannot be parsed for category/competitor.
    """
    # Activate dry-run if requested via CLI flag or env var
    if dry_run:
        os.environ["AGENT_DRY_RUN"] = "true"

    inbox_path = Path(inbox_path).resolve()
    if not inbox_path.exists():
        raise FileNotFoundError(f"Inbox file not found: {inbox_path}")

    outputs_dir = outputs_dir or _resolve_outputs_dir()
    register_path = register_path or _resolve_register_path()
    data_root = _resolve_data_root()

    # ── Step 1: Parse filename ─────────────────────────────────────────────────
    category, competitor_slug = parse_inbox_filename(inbox_path.name)
    logger.info("Evaluating: %s / %s", category, competitor_slug)

    # ── Step 2: Load scoped context ────────────────────────────────────────────
    context = load_context(category, competitor_slug, data_root)
    inbox_text = inbox_path.read_text(encoding="utf-8")

    # ── Step 3: Load model ─────────────────────────────────────────────────────
    tokenizer, model = load_model(model_path=model_path, adapter_path=adapter_path)

    # ── Step 4: Extract financial variables from context (Python, no model) ───
    signal = parse_signal_payload(inbox_text)
    pass1_payload = extract_pass1_vars(context, signal)
    logger.info("Financial variables: %s", pass1_payload)

    # ── Step 5: ROI calculation ────────────────────────────────────────────────
    roi_result = calculate_roi(pass1_payload)
    logger.info(
        "ROI: annual_net=£%.2f, threshold_met=%s",
        roi_result["annual_net_gbp"],
        roi_result["roi_threshold_met"],
    )

    # ── Step 6: Voting pipeline — independent micro-decisions ────────────────
    logger.info("Running independent voting pipeline...")
    memo_text, confidence = run_voting(inbox_text, context, roi_result, tokenizer, model)

    # ── Step 7: Validate; retry once on failure ────────────────────────────────
    validation_attempts = 1
    is_valid, errors = validate_verdict(memo_text)
    if not is_valid:
        logger.warning("Validation failed (%d errors). Retrying...", len(errors))
        for err in errors:
            logger.warning("  - %s", err)
        memo_text, confidence = run_voting(
            inbox_text,
            context,
            roi_result,
            tokenizer,
            model,
            retry_hint=errors,
        )
        validation_attempts = 2
        is_valid, errors = validate_verdict(memo_text)
        if not is_valid:
            logger.error("Retry failed. Writing memo with [VALIDATION FAILED] prefix.")
            for err in errors:
                logger.error("  - %s", err)
            memo_text = "[VALIDATION FAILED — manual review required]\n\n" + memo_text
        else:
            logger.info("Retry succeeded.")
    else:
        errors = []

    # ── Step 8: Write output ───────────────────────────────────────────────────
    output_path = _write_output(memo_text, category, competitor_slug, outputs_dir)

    # ── Step 9: Auto-generate plain-English summary ────────────────────────────
    _auto_summarise(memo_text, output_path.name, tokenizer, model)

    # ── Step 10: Verdict + hold register ──────────────────────────────────────
    verdict = extract_verdict_class(memo_text) or "UNKNOWN"
    logger.info("Verdict: %s", verdict)

    hold_registered = False
    if verdict == "HOLD":
        hold_data = extract_hold_metadata(memo_text)
        if hold_data is None:
            logger.warning("Could not extract hold metadata from memo; building minimal entry.")
            hold_data = _build_minimal_hold_entry(category, context, competitor_slug)
        _append_hold_register(hold_data, register_path)
        hold_registered = True

    # ── Step 11: Drift tracking ────────────────────────────────────────────────
    log_live_run(category, competitor_slug, verdict, is_valid, validation_attempts, confidence,
                 log_path=log_path)
    if check_accuracy_due(log_path=log_path):
        logger.info(
            "Drift advisory: 10+ live runs since last accuracy check — "
            "run: python scripts/drift_check.py"
        )

    return {
        "verdict": verdict,
        "output_path": output_path,
        "hold_registered": hold_registered,
        "validation_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SaaS Stack Manager — evaluate a market inbox trigger file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m agent.agent market_inbox/finance_ledgerflow.md\n"
            "  python -m agent.agent --dry-run fixtures/inbox_switch.md\n"
        ),
    )
    parser.add_argument("inbox_file", help="Path to the inbox trigger .md file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use fixture responses instead of loading the model (no GPU required).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local path to the base model directory (e.g. /path/to/qwen2.5-3b-instruct).",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to a fine-tuned LoRA adapter directory (e.g. training/checkpoints/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_agent(
            Path(args.inbox_file),
            dry_run=args.dry_run,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
        )
        print(f"\nVerdict:  {result['verdict']}")
        print(f"Output:   {result['output_path']}")
        if result["hold_registered"]:
            print("Hold register updated.")
        if result["validation_errors"]:
            print(f"WARNING: {len(result['validation_errors'])} validation error(s) — manual review required.")
            sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
