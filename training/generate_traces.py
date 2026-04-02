"""
Phase 2 — Lean single-step training trace generation script.

Reads generated signal JSON files from training/generated/ and runs the lean
pipeline teacher to produce compact outputs in this format:

    ANALYSIS: ...
    VERDICT: SWITCH|STAY|HOLD

For each valid signal file, one JSONL training record is produced:
    [SYS_VERDICT_LEAN + compact user prompt] -> [lean assistant output]

Only traces that pass validate_lean_output() are saved.
Invalid outputs are logged and skipped.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_TRACES_DIR = _PROJECT_ROOT / "training" / "traces"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.model_runner import LOCAL_LLAMA_8B, _build_lean_user, _compliance_pass_python, _generate, _parse_compliance_changes, load_model, run_lean  # noqa: E402
from agent.output_validator import validate_lean_output  # noqa: E402
from agent.prompts import SYS_VERDICT_LEAN  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import parse_signal_payload, signal_compliance_changes  # noqa: E402
from training.generate_signals import SCENARIO_TYPES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Authoritative verdict per deterministic scenario.  The teacher (untuned Llama)
# is consulted only for reasoning; the final VERDICT label is always overridden
# with the canonical value so training labels are consistent.
_CANONICAL_VERDICT: dict[str, str] = {
    "pull_dominant":           "SWITCH",
    "push_dominant":           "SWITCH",
    "shelfware_case":          "SWITCH",
    "hold_resolved":           "SWITCH",
    "fluff_update":            "STAY",
    "current_tool_rally":      "STAY",
    "irrelevant_change":       "STAY",
    "negative_signal_buried":  "STAY",
    "hard_compliance_failure": "STAY",
    "competitor_nearly_ready": "HOLD",
    # Ambiguous — verdict depends on signal magnitude; teacher output kept as-is.
    # "both_signals":          None,
    # "price_hike_only":       None,
    # "dual_improvement":      None,
}


# Per-scenario reasoning hints injected into Llama teacher prompt.
# Guides Llama to produce ANALYSIS that supports the canonical verdict,
# preventing the ANALYSIS-VERDICT incoherence that confuses fine-tuning.
_REASONING_HINTS: dict[str, str] = {
    "competitor_nearly_ready": (
        "The key competitor feature is in beta or not yet GA. "
        "Issue a HOLD verdict and explicitly state what must happen (GA release) before switching."
    ),
    "negative_signal_buried": (
        "Look carefully in the notes section for limitations, tier restrictions, or maturity caveats "
        "that negate the apparent gains. These buried signals justify a STAY verdict."
    ),
    "hard_compliance_failure": (
        "A hard compliance block exists. Compliance blocks are non-negotiable — "
        "the switch is not possible regardless of features. Issue a STAY verdict."
    ),
    "irrelevant_change": (
        "The competitor's changes are not relevant to this business's use case. "
        "No compelling reason to switch. Issue a STAY verdict."
    ),
    "fluff_update": (
        "The competitor update is vague or superficial — no concrete new features. "
        "No pull signal. Issue a STAY verdict."
    ),
    "current_tool_rally": (
        "The current tool has improved and addressed the key push factors. "
        "The urgency to switch has been removed. Issue a STAY verdict."
    ),
    "pull_dominant": (
        "The competitor has shipped a feature that resolves a key push issue. "
        "Issue a SWITCH verdict and state the resolved issue clearly."
    ),
    "push_dominant": (
        "The current tool's reliability or compliance has degraded significantly. "
        "Issue a SWITCH verdict and name the specific degradation."
    ),
    "shelfware_case": (
        "The current tool has become shelfware — seat utilisation is very low. "
        "Issue a SWITCH verdict and quantify the waste."
    ),
    "hold_resolved": (
        "The previous blocking condition for a HOLD verdict has now been resolved. "
        "Issue a SWITCH verdict and name the resolved condition."
    ),
}


def _enforce_verdict(lean_output: str, canonical_verdict: str) -> str:
    """Replace the VERDICT: line in lean_output with the canonical verdict."""
    import re
    corrected = re.sub(
        r"VERDICT:\s*(SWITCH|STAY|HOLD)[^\n]*",
        f"VERDICT: {canonical_verdict}",
        lean_output,
        count=1,
    )
    if corrected == lean_output:
        # No VERDICT line found — append one
        corrected = lean_output.rstrip() + f"\nVERDICT: {canonical_verdict}"
    return corrected


def _parse_generated_filename(stem: str) -> tuple[str, str, str] | None:
    """
    Parse a generated signal filename stem into (category, competitor_slug, scenario).

    Files follow: {category}_{competitor}_{scenario}.json
    where scenario may contain underscores (e.g. 'hard_compliance_failure').
    Category is matched greedily (longest first to handle 'project_mgmt_').
    """
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        prefix = cat + "_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            for scenario in sorted(SCENARIO_TYPES, key=len, reverse=True):
                suffix = "_" + scenario
                if remainder.endswith(suffix):
                    competitor_slug = remainder[: -len(suffix)]
                    if competitor_slug:
                        return cat, competitor_slug, scenario
    return None


def _load_processed_keys(traces_path: Path) -> set[tuple[str, str]]:
    """Return the set of (source_file, step) keys already present in traces.jsonl."""
    if not traces_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                metadata = record.get("metadata", {})
                stem = metadata.get("source_file")
                step = metadata.get("step")
                if stem and step:
                    keys.add((stem, step))
            except json.JSONDecodeError:
                pass
    return keys


def _build_trace(
    messages: list[dict],
    assistant_output: str,
    category: str,
    competitor_slug: str,
    scenario: str,
    source_file: str,
    step: str,
) -> dict:
    """Format one micro-decision trace as a JSONL training record."""
    full_messages = list(messages)
    full_messages.append({"role": "assistant", "content": assistant_output})
    return {
        "messages": full_messages,
        "metadata": {
            "category": category,
            "competitor": competitor_slug,
            "scenario": scenario,
            "source_file": source_file,
            "step": step,
        },
    }


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_one(
    signal_path: Path,
    tokenizer: Any,
    model: Any,
    traces_path: Path,
    processed_keys: set[tuple[str, str]],
    dry_run: bool = False,
) -> dict:
    """
    Run the verdict memo pipeline on one signal file and save the training trace.

    Returns:
        {"status": "ok" | "skip" | "error", "reason": str | None}
    """
    parsed = _parse_generated_filename(signal_path.stem)
    if parsed is None:
        return {"status": "skip", "reason": f"Cannot parse filename: {signal_path.name}"}

    category, competitor_slug, scenario = parsed

    try:
        context = load_context(category, competitor_slug, _DATA_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "error", "reason": f"Context load failed: {exc}"}

    with signal_path.open(encoding="utf-8") as f:
        trigger = json.load(f)
    inbox_text = json.dumps(trigger, indent=2, ensure_ascii=False)

    expected_steps = {"lean"}
    if expected_steps.issubset({step for stem, step in processed_keys if stem == signal_path.stem}):
        return {"status": "skip", "reason": f"Already processed: {signal_path.stem}"}

    # Extract financial variables from structured context + signal delta (no model)
    signal = parse_signal_payload(inbox_text)
    pass1_payload = extract_pass1_vars(context, signal)
    roi_result = calculate_roi(pass1_payload)

    if dry_run:
        os.environ["AGENT_DRY_RUN"] = "true"

    hint = _REASONING_HINTS.get(scenario)
    if hint and not dry_run:
        _signal = parse_signal_payload(inbox_text)
        _category = context["category"]
        _seat_count = context["current_stack_entry"].get("seat_count", 0)
        _compliance_text = signal_compliance_changes(_signal)
        _compliance = dict(context["competitor_data"].get("compliance", {}))
        if _compliance_text:
            _compliance = _parse_compliance_changes(
                _compliance_text, _compliance, _category, _seat_count, tokenizer, model
            )
        _effective_context = {
            **context,
            "competitor_data": {**context["competitor_data"], "compliance": _compliance},
        }
        _passed, _failures = _compliance_pass_python(_effective_context)
        if not _passed:
            _failure_str = "; ".join(_failures)
            lean_result = (
                f"ANALYSIS: Compliance blocks present — {_failure_str}. "
                f"Hard compliance blocks are non-negotiable — STAY.\n"
                f"VERDICT: STAY"
            )
        else:
            _hint_system = SYS_VERDICT_LEAN + f"\n\nNOTE: {hint}"
            _user_content = _build_lean_user(_effective_context, roi_result, _signal)
            _messages = [
                {"role": "system", "content": _hint_system},
                {"role": "user", "content": _user_content},
            ]
            lean_result = _generate(
                tokenizer, model, _messages, max_new_tokens=200, temperature=0.1
            ).strip()
    else:
        lean_result = run_lean(inbox_text, context, roi_result, tokenizer, model)

    # Validate lean output; retry once on malformed output.
    is_valid, errors = validate_lean_output(lean_result)
    if not is_valid:
        logger.warning(
            "Validation failed for %s (%d errors), retrying: %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        lean_result = run_lean(inbox_text, context, roi_result, tokenizer, model)
        is_valid, errors = validate_lean_output(lean_result)

    if not is_valid:
        logger.warning(
            "Skipping %s — retry also failed (%d errors): %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        return {"status": "skip", "reason": f"Validation failed after retry: {errors[0]}"}
    # Override VERDICT with canonical label for all deterministic scenarios.
    if scenario in _CANONICAL_VERDICT:
        lean_result = _enforce_verdict(lean_result, _CANONICAL_VERDICT[scenario])
    if dry_run:
        signal = parse_signal_payload(inbox_text)
        user_content = _build_lean_user(context, roi_result, signal)
        print(f"\n{'='*60}")
        print(f"TRACE: {signal_path.name}")
        print(json.dumps({
            "messages": [
                {"role": "system", "content": SYS_VERDICT_LEAN},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": lean_result},
            ]
        }, ensure_ascii=False, indent=2))
        return {"status": "ok", "reason": None}

    signal = parse_signal_payload(inbox_text)
    user_content = _build_lean_user(context, roi_result, signal)
    step_record = _build_trace(
        [
            {"role": "system", "content": SYS_VERDICT_LEAN},
            {"role": "user", "content": user_content},
        ],
        lean_result,
        category,
        competitor_slug,
        scenario,
        signal_path.stem,
        "lean",
    )
    _append_jsonl(traces_path, step_record)

    logger.info("Saved trace for: %s", signal_path.name)
    return {"status": "ok", "reason": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training traces from signal JSON files using Llama 3.1 8B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/generate_traces.py\n"
            "  python training/generate_traces.py --fresh\n"
            "  python training/generate_traces.py --dry-run --limit 5\n"
            "  python training/generate_traces.py --category finance\n"
        ),
    )
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES), help="Filter to one category.")
    parser.add_argument("--limit", type=int, default=0, help="Max signal files to process (0 = no limit).")
    parser.add_argument("--generated-dir", default=str(_GENERATED_DIR), help="Directory of generated signal files.")
    parser.add_argument("--traces-dir", default=str(_TRACES_DIR), help="Output directory for traces.")
    parser.add_argument("--fresh", action="store_true", help="Delete existing traces.jsonl and start from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline but print traces; do not save.")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    traces_path = Path(args.traces_dir) / "traces.jsonl"
    report_path = Path(args.traces_dir) / "traces_report.json"

    if not generated_dir.exists():
        logger.error("Generated directory not found: %s. Run generate_signals.py first.", generated_dir)
        sys.exit(1)

    # Fresh start: wipe the traces file so we don't mix old and new format traces
    if args.fresh and not args.dry_run and traces_path.exists():
        logger.info("Fresh start: removing existing %s", traces_path)
        traces_path.unlink()

    signal_files = sorted(generated_dir.glob("*.json"))
    if args.category:
        signal_files = [f for f in signal_files if f.stem.startswith(args.category + "_")]

    # Always deduplicate: skip files already present in traces.jsonl
    if not args.dry_run:
        already_done = _load_processed_keys(traces_path)
        if already_done:
            before = len(signal_files)
            signal_files = [
                f for f in signal_files
                if not {"lean"}.issubset(
                    {step for stem, step in already_done if stem == f.stem}
                )
            ]
            logger.info("Skipping %d already-processed files (%d remaining).", before - len(signal_files), len(signal_files))

    if args.limit:
        signal_files = signal_files[: args.limit]

    if not signal_files:
        logger.info("No signal files to process.")
        sys.exit(0)

    logger.info("Processing %d signal files...", len(signal_files))

    tokenizer, model = None, None
    if not args.dry_run:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        tokenizer, model = load_model(model_path=LOCAL_LLAMA_8B, quantize_bits=4)
    else:
        os.environ["AGENT_DRY_RUN"] = "true"

    stats: dict[str, int] = {"ok": 0, "skip": 0, "error": 0}
    scenario_counts: dict[str, int] = {}

    for signal_path in signal_files:
        result = process_one(signal_path, tokenizer, model, traces_path, already_done if not args.dry_run else set(), args.dry_run)
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1

        if status == "ok":
            parsed = _parse_generated_filename(signal_path.stem)
            if parsed:
                scenario = parsed[2]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        elif status == "error":
            logger.error("Error on %s: %s", signal_path.name, result["reason"])

    report = {
        "signal_files_processed": len(signal_files),
        "passed": stats["ok"],
        "skipped": stats["skip"],
        "errors": stats["error"],
        "total_step_traces_written": stats["ok"],
        "signal_files_per_scenario": scenario_counts,
    }

    if not args.dry_run:
        Path(args.traces_dir).mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Post-run consistency check: verify no duplicates in final traces file
        final_keys = _load_processed_keys(traces_path)
        with traces_path.open(encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
        if total_lines != len(final_keys):
            logger.warning(
                "Consistency check: %d lines but %d unique stems — %d duplicate(s) detected.",
                total_lines, len(final_keys), total_lines - len(final_keys),
            )

    logger.info(
        "Done. %d/%d signal files → %d lean traces. %d skipped, %d errors.",
        stats["ok"],
        len(signal_files),
        stats["ok"],
        stats["skip"],
        stats["error"],
    )
    if not args.dry_run:
        logger.info("Traces: %s", traces_path)
        logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
