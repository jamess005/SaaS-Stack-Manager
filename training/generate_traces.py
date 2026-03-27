"""
Phase 2 — Training trace generation script.

Reads all generated signal JSON files from training/generated/ and runs the
verdict memo pipeline using the local Llama 3.1 8B (4-bit NF4) as the teacher model.

Financial variable extraction is handled by Python (extract_pass1_vars) — no model
call required for that step. The model generates only the verdict memo (Pass 2).

For each valid signal file, one JSONL training record is produced:
  [system prompt + context + inbox + Python-computed ROI] → [verdict memo]

Only traces where Pass 2 output passes validate_verdict() are saved.
Invalid outputs are logged and skipped — no malformed training data written.

Output:
  training/traces/traces.jsonl   — one JSON object per line
  training/traces/traces_report.json — summary (total, passed, failed, by scenario)

Usage:
    # Generate traces from all files in training/generated/
    python training/generate_traces.py

    # Resume a previous run (skip files already in traces.jsonl)
    python training/generate_traces.py --resume

    # Dry-run: run pipeline but print traces to stdout instead of saving
    python training/generate_traces.py --dry-run

    # Filter to one category or limit count
    python training/generate_traces.py --category finance
    python training/generate_traces.py --limit 10

Environment (required for ROCm on RX 7800 XT):
    HSA_OVERRIDE_GFX_VERSION=11.0.0
    ROCR_VISIBLE_DEVICES=0
    HIP_VISIBLE_DEVICES=0
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
from agent.model_runner import LOCAL_LLAMA_8B, _assemble_pass2_prompt, load_model, run_pass2  # noqa: E402
from agent.output_validator import validate_verdict  # noqa: E402
from agent.prompts import PASS2_FEW_SHOT, SYSTEM_PROMPT  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from training.generate_signals import SCENARIO_TYPES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def _load_processed_stems(traces_path: Path) -> set[str]:
    """Return the set of file stems already present in traces.jsonl."""
    if not traces_path.exists():
        return set()
    stems: set[str] = set()
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                stem = record.get("metadata", {}).get("source_file")
                if stem:
                    stems.add(stem)
            except json.JSONDecodeError:
                pass
    return stems


def _build_trace(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    pass2_output: str,
    category: str,
    competitor_slug: str,
    scenario: str,
    source_file: str,
) -> dict:
    """Format a Pass 2 verdict memo as a JSONL training record."""
    messages = _assemble_pass2_prompt(inbox_text, context, roi_result, SYSTEM_PROMPT, PASS2_FEW_SHOT)
    messages.append({"role": "assistant", "content": pass2_output})
    return {
        "messages": messages,
        "metadata": {
            "category": category,
            "competitor": competitor_slug,
            "scenario": scenario,
            "source_file": source_file,
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

    # Extract financial variables from structured context (no model)
    pass1_payload = extract_pass1_vars(context)
    roi_result = calculate_roi(pass1_payload)

    # Generate verdict memo
    pass2_output = run_pass2(
        inbox_text, context, roi_result, SYSTEM_PROMPT, PASS2_FEW_SHOT, tokenizer, model
    )

    # Validate; retry once on failure with validation errors as hints
    is_valid, errors = validate_verdict(pass2_output)
    if not is_valid:
        logger.warning(
            "Validation failed for %s (%d errors), retrying: %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        pass2_output = run_pass2(
            inbox_text, context, roi_result, SYSTEM_PROMPT, PASS2_FEW_SHOT,
            tokenizer, model, retry_hint=errors,
        )
        is_valid, errors = validate_verdict(pass2_output)

    if not is_valid:
        logger.warning(
            "Skipping %s — retry also failed (%d errors): %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        return {"status": "skip", "reason": f"Validation failed after retry: {errors[0]}"}

    trace = _build_trace(
        inbox_text, context, roi_result, pass2_output,
        category, competitor_slug, scenario, signal_path.stem,
    )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"TRACE: {signal_path.name}")
        print(f"Verdict memo preview: {pass2_output[:300]}...")
        return {"status": "ok", "reason": None}

    _append_jsonl(traces_path, trace)
    logger.info("Saved trace for: %s", signal_path.name)
    return {"status": "ok", "reason": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training traces from signal JSON files using Llama 3.1 8B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/generate_traces.py\n"
            "  python training/generate_traces.py --resume\n"
            "  python training/generate_traces.py --dry-run --limit 5\n"
            "  python training/generate_traces.py --category finance\n"
        ),
    )
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES), help="Filter to one category.")
    parser.add_argument("--limit", type=int, default=0, help="Max signal files to process (0 = no limit).")
    parser.add_argument("--generated-dir", default=str(_GENERATED_DIR), help="Directory of generated signal files.")
    parser.add_argument("--traces-dir", default=str(_TRACES_DIR), help="Output directory for traces.")
    parser.add_argument("--resume", action="store_true", help="Skip files already present in traces.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline but print traces; do not save.")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    traces_path = Path(args.traces_dir) / "traces.jsonl"
    report_path = Path(args.traces_dir) / "traces_report.json"

    if not generated_dir.exists():
        logger.error("Generated directory not found: %s. Run generate_signals.py first.", generated_dir)
        sys.exit(1)

    signal_files = sorted(generated_dir.glob("*.json"))
    if args.category:
        signal_files = [f for f in signal_files if f.stem.startswith(args.category + "_")]

    # Resume: skip files already processed
    if args.resume and not args.dry_run:
        already_done = _load_processed_stems(traces_path)
        if already_done:
            before = len(signal_files)
            signal_files = [f for f in signal_files if f.stem not in already_done]
            logger.info("Resume: skipping %d already-processed files (%d remaining).", before - len(signal_files), len(signal_files))

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
        result = process_one(signal_path, tokenizer, model, traces_path, args.dry_run)
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
        "total_traces_written": stats["ok"],
        "traces_per_scenario": scenario_counts,
    }

    if not args.dry_run:
        Path(args.traces_dir).mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "Done. %d/%d signal files → %d traces. %d skipped, %d errors.",
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
