"""
Phase 2 — Training trace generation script.

Reads all generated signal JSON files from training/generated/ and runs the
voting pipeline using the local Llama 3.1 8B (4-bit NF4) as the teacher model.

Financial variable extraction is handled by Python (extract_pass1_vars) — no model
call required for that step. The model generates four micro-decisions:
compliance, push, pull, and final verdict.

For each valid signal file, four JSONL training records are produced:
    [system + step-specific user prompt] → [assistant step output]

Only traces where the assembled memo passes validate_verdict() are saved.
Invalid outputs are logged and skipped so malformed reasoning is not written.

Output:
  training/traces/traces.jsonl   — one JSON object per line
  training/traces/traces_report.json — summary (total, passed, failed, by scenario)

Usage:
    # Generate traces (automatically skips files already in traces.jsonl)
    python training/generate_traces.py

    # Start fresh (delete existing traces.jsonl first)
    python training/generate_traces.py --fresh

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
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_TRACES_DIR = _PROJECT_ROOT / "training" / "traces"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.model_runner import LOCAL_LLAMA_8B, _assemble_verdict_memo, _build_compliance_user, _build_pull_user, _build_push_user, _build_verdict_user, _generate, load_model  # noqa: E402
from agent.output_validator import validate_verdict  # noqa: E402
from agent.prompts import SYS_COMPLIANCE, SYS_PULL, SYS_PUSH, SYS_VERDICT  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import parse_signal_payload  # noqa: E402
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


def _generate_vote_outputs(inbox_text: str, context: dict, roi_result: dict, tokenizer: Any, model: Any) -> dict:
    signal = parse_signal_payload(inbox_text)

    compliance_messages = [
        {"role": "system", "content": SYS_COMPLIANCE},
        {"role": "user", "content": _build_compliance_user(context, signal)},
    ]
    compliance_result = _generate(tokenizer, model, compliance_messages, max_new_tokens=100, temperature=0.1).strip()

    push_messages = [
        {"role": "system", "content": SYS_PUSH},
        {"role": "user", "content": _build_push_user(context, signal)},
    ]
    push_result = _generate(tokenizer, model, push_messages, max_new_tokens=300, temperature=0.3).strip()

    pull_messages = [
        {"role": "system", "content": SYS_PULL},
        {"role": "user", "content": _build_pull_user(context, inbox_text)},
    ]
    pull_result = _generate(tokenizer, model, pull_messages, max_new_tokens=400, temperature=0.3).strip()

    verdict_messages = [
        {"role": "system", "content": SYS_VERDICT},
        {
            "role": "user",
            "content": _build_verdict_user(
                context, roi_result, compliance_result, push_result, pull_result
            ),
        },
    ]
    verdict_result = _generate(tokenizer, model, verdict_messages, max_new_tokens=250, temperature=0.1).strip()

    memo = _assemble_verdict_memo(
        context, roi_result, compliance_result, push_result, pull_result, verdict_result
    )

    return {
        "compliance_messages": compliance_messages,
        "compliance_result": compliance_result,
        "push_messages": push_messages,
        "push_result": push_result,
        "pull_messages": pull_messages,
        "pull_result": pull_result,
        "verdict_messages": verdict_messages,
        "verdict_result": verdict_result,
        "memo": memo,
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

    expected_steps = {"compliance", "push", "pull", "verdict"}
    if expected_steps.issubset({step for stem, step in processed_keys if stem == signal_path.stem}):
        return {"status": "skip", "reason": f"Already processed: {signal_path.stem}"}

    # Extract financial variables from structured context + signal delta (no model)
    signal = parse_signal_payload(inbox_text)
    pass1_payload = extract_pass1_vars(context, signal)
    roi_result = calculate_roi(pass1_payload)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"TRACE: {signal_path.name}")
        print(json.dumps({"pass1_payload": pass1_payload, "roi_result": roi_result}, indent=2))
        return {"status": "ok", "reason": None}

    vote_outputs = _generate_vote_outputs(inbox_text, context, roi_result, tokenizer, model)

    # Validate assembled memo; retry once on malformed output.
    is_valid, errors = validate_verdict(vote_outputs["memo"])
    if not is_valid:
        logger.warning(
            "Validation failed for %s (%d errors), retrying: %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        vote_outputs = _generate_vote_outputs(inbox_text, context, roi_result, tokenizer, model)
        is_valid, errors = validate_verdict(vote_outputs["memo"])

    if not is_valid:
        logger.warning(
            "Skipping %s — retry also failed (%d errors): %s",
            signal_path.name,
            len(errors),
            "; ".join(errors[:2]),
        )
        return {"status": "skip", "reason": f"Validation failed after retry: {errors[0]}"}

    if dry_run:
        print(f"\n{'='*60}")
        print(f"TRACE: {signal_path.name}")
        print(f"Verdict memo preview: {vote_outputs['memo'][:300]}...")
        return {"status": "ok", "reason": None}

    step_records = [
        _build_trace(
            vote_outputs["compliance_messages"],
            vote_outputs["compliance_result"],
            category,
            competitor_slug,
            scenario,
            signal_path.stem,
            "compliance",
        ),
        _build_trace(
            vote_outputs["push_messages"],
            vote_outputs["push_result"],
            category,
            competitor_slug,
            scenario,
            signal_path.stem,
            "push",
        ),
        _build_trace(
            vote_outputs["pull_messages"],
            vote_outputs["pull_result"],
            category,
            competitor_slug,
            scenario,
            signal_path.stem,
            "pull",
        ),
        _build_trace(
            vote_outputs["verdict_messages"],
            vote_outputs["verdict_result"],
            category,
            competitor_slug,
            scenario,
            signal_path.stem,
            "verdict",
        ),
    ]
    for trace in step_records:
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
                if not {"compliance", "push", "pull", "verdict"}.issubset(
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
        "total_step_traces_written": stats["ok"] * 4,
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
        "Done. %d/%d signal files → %d step traces. %d skipped, %d errors.",
        stats["ok"],
        len(signal_files),
        stats["ok"] * 4,
        stats["skip"],
        stats["error"],
    )
    if not args.dry_run:
        logger.info("Traces: %s", traces_path)
        logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
