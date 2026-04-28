"""
Model evaluation script — tests the fine-tuned SaaS Stack Manager against
either the synthetic training corpus or a terminal-only held-out fixture set.

Usage:
    # Dry-run (no GPU, uses fixture responses)
    python scripts/evaluate_model.py --dry-run

    # Full evaluation, 2 samples per scenario type (~11 min with GPU)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/evaluate_model.py \\
        --adapter-path training/checkpoints_sft_cot/

    # Terminal-only held-out set with explicit expected verdicts
    python scripts/evaluate_model.py \\
        --signals-dir fixtures/eval_signals \\
        --all-files --dry-run
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"
_DEFAULT_HOLDOUT_DIR = _PROJECT_ROOT / "fixtures" / "eval_signals"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_DASHBOARD_LOG = _OUTPUTS_DIR / "drift_log.jsonl"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.drift_tracker import log_live_run  # noqa: E402
from agent.model_runner import load_base_model, load_model, run_lean, run_voting, unload_model  # noqa: E402
from agent.output_validator import extract_verdict_class, validate_lean_output, validate_verdict  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import parse_signal_payload  # noqa: E402
from agent.agent import _auto_summarise  # noqa: E402
from config import MODEL_PATH  # noqa: E402
from training.generate_signals import SCENARIO_TYPES  # noqa: E402

console = Console()

# ── Expected verdict map ───────────────────────────────────────────────────────
# None = ambiguous (reported but not counted in accuracy)
EXPECTED_VERDICT: dict[str, str | None] = {
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
    # Ambiguous — verdict depends on signal magnitude, not scenario type alone
    "both_signals":            None,
    "price_hike_only":         None,
    "dual_improvement":        None,
}


# ── Filename parsing ───────────────────────────────────────────────────────────

def _parse_generated_filename(stem: str) -> tuple[str, str, str] | None:
    """Parse {category}_{competitor}_{scenario} stem. Returns None if unparseable."""
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


def _coerce_expected_verdict(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None
    verdict = raw_value.strip().upper()
    return verdict if verdict in {"SWITCH", "STAY", "HOLD"} else None


def _load_signal_case(signal_path: Path) -> dict:
    with signal_path.open(encoding="utf-8") as f:
        raw_payload = json.load(f)

    if not isinstance(raw_payload, dict):
        raise ValueError(f"Signal file must contain a JSON object: {signal_path.name}")

    signal = raw_payload.get("signal") if isinstance(raw_payload.get("signal"), dict) else raw_payload
    if not isinstance(signal, dict):
        raise ValueError(f"Signal payload must be a JSON object: {signal_path.name}")

    parsed = _parse_generated_filename(signal_path.stem)
    category = raw_payload.get("category") if isinstance(raw_payload.get("category"), str) else None
    competitor_slug = (
        raw_payload.get("competitor_slug")
        if isinstance(raw_payload.get("competitor_slug"), str)
        else None
    )
    scenario = raw_payload.get("scenario") if isinstance(raw_payload.get("scenario"), str) else None

    if parsed is not None:
        category = category or parsed[0]
        competitor_slug = competitor_slug or parsed[1]
        scenario = scenario or parsed[2]

    scenario = (scenario or signal_path.stem).strip()
    if not category or not competitor_slug:
        raise ValueError(
            f"Cannot determine category and competitor_slug for {signal_path.name}"
        )

    expected = _coerce_expected_verdict(raw_payload.get("expected_verdict"))
    if expected is None and parsed is not None:
        expected = EXPECTED_VERDICT.get(parsed[2])

    return {
        "signal": signal,
        "category": category,
        "competitor": competitor_slug,
        "scenario": scenario,
        "expected": expected,
    }


def _confidence_pct(confidence: dict | None) -> float | None:
    if not confidence:
        return None
    value = confidence.get("verdict_token_prob")
    if isinstance(value, (int, float)):
        return float(value) * 100.0
    return None


def _format_confidence(confidence: dict | None) -> str:
    if not confidence:
        return "—"
    value = confidence.get("verdict_token_prob")
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "—"


def _dashboard_memo_filename(result: dict, run_date: str | None = None) -> str:
    date_prefix = run_date or datetime.now().date().isoformat()
    return (
        f"{date_prefix}-{result['category']}-{result['competitor']}-{result['scenario']}.md"
    )


def _persist_dashboard_run(
    result: dict,
    outputs_dir: Path | None = None,
    log_path: Path | None = None,
    run_date: str | None = None,
) -> str:
    outputs_dir = outputs_dir or _OUTPUTS_DIR
    log_path = log_path or _DASHBOARD_LOG
    memo_text = result.get("memo_text")
    if not isinstance(memo_text, str):
        raise ValueError("memo_text is required to persist dashboard runs")

    memo_filename = _dashboard_memo_filename(result, run_date=run_date)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / memo_filename).write_text(memo_text, encoding="utf-8")

    log_live_run(
        category=result["category"],
        competitor=result["competitor"],
        verdict=result.get("actual") or "UNKNOWN",
        format_valid=bool(result.get("format_valid")),
        validation_attempts=int(result.get("validation_attempts", 1)),
        confidence=result.get("confidence"),
        log_path=log_path,
        memo_filename=memo_filename,
    )
    return memo_filename


def _serialisable_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "memo_text"}


def _pipeline_components(pipeline: str):
    if pipeline == "voting":
        return run_voting, validate_verdict
    return run_lean, validate_lean_output


# ── Signal sampling ────────────────────────────────────────────────────────────

def _collect_samples(
    scenario_filter: str | None,
    limit: int,
    signals_dir: Path,
    all_files: bool = False,
) -> list[Path]:
    """
    Return a reproducible sample of signal files grouped by scenario type.
    Uses a fixed seed so the same files are always selected.
    """
    by_scenario: dict[str, list[Path]] = {}

    for path in sorted(signals_dir.glob("*.json")):
        try:
            case = _load_signal_case(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        scenario = case["scenario"]
        by_scenario.setdefault(scenario, []).append(path)

    if all_files:
        if scenario_filter:
            return list(by_scenario.get(scenario_filter, []))
        selected: list[Path] = []
        for scenario in sorted(by_scenario):
            selected.extend(by_scenario[scenario])
        return selected

    selected: list[Path] = []
    rng = random.Random(42)
    if scenario_filter:
        scenarios_to_run = [scenario_filter]
    else:
        scenarios_to_run = [s for s in SCENARIO_TYPES if s in by_scenario]
        extras = sorted(s for s in by_scenario if s not in SCENARIO_TYPES)
        scenarios_to_run.extend(extras)
    for scenario in scenarios_to_run:
        pool = sorted(by_scenario.get(scenario, []))
        rng.shuffle(pool)
        selected.extend(pool[:limit])

    return selected


# ── Single evaluation ──────────────────────────────────────────────────────────

def _evaluate_one(
    signal_path: Path,
    tokenizer,
    model,
    pipeline: str = "lean",
) -> dict:
    """Run the pipeline on one signal file and return a result record."""
    try:
        case = _load_signal_case(signal_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "skip",
            "reason": str(exc),
            "source_file": signal_path.name,
        }

    category = case["category"]
    competitor_slug = case["competitor"]
    scenario = case["scenario"]

    try:
        context = load_context(category, competitor_slug, _DATA_ROOT)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "error", "reason": str(exc), "scenario": scenario,
                "category": category, "competitor": competitor_slug}

    signal = case["signal"]
    inbox_text = json.dumps(signal, indent=2, ensure_ascii=False)

    pass1_payload = extract_pass1_vars(context, parse_signal_payload(inbox_text))
    roi_result = calculate_roi(pass1_payload)

    runner, validator = _pipeline_components(pipeline)
    validation_attempts = 1
    memo, confidence = runner(inbox_text, context, roi_result, tokenizer, model)

    # Retry once on validation failure
    is_valid, _errors = validator(memo)
    if not is_valid:
        validation_attempts = 2
        memo, confidence = runner(inbox_text, context, roi_result, tokenizer, model)
        is_valid, _ = validator(memo)

    verdict = extract_verdict_class(memo)
    expected = case["expected"]

    result: dict = {
        "status": "ok",
        "scenario": scenario,
        "category": category,
        "competitor": competitor_slug,
        "expected": expected,
        "actual": verdict,
        "confidence": confidence,
        "format_valid": is_valid,
        "validation_attempts": validation_attempts,
        "memo_text": memo,
        "source_file": signal_path.name,
    }
    if not is_valid:
        result["raw_output"] = memo
    return result


# ── Reporting ──────────────────────────────────────────────────────────────────

def _print_results(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]

    # ── Per-result table ──────────────────────────────────────────────────────
    detail = Table(title="Evaluation Results", show_lines=False)
    detail.add_column("Scenario", style="cyan", min_width=26)
    detail.add_column("Category / Competitor", style="dim", min_width=24)
    detail.add_column("Expected", justify="center", min_width=8)
    detail.add_column("Actual", justify="center", min_width=8)
    detail.add_column("Confidence", justify="right", min_width=10)
    detail.add_column("Match", justify="center", min_width=6)
    detail.add_column("Format", justify="center", min_width=7)

    for r in ok:
        expected = r["expected"] or "—"
        actual = r["actual"] or "?"
        ambiguous = r["expected"] is None

        if ambiguous:
            match_str = "[dim]n/a[/dim]"
            match_style = ""
        elif r["actual"] == r["expected"]:
            match_str = "[bold green]✓[/bold green]"
            match_style = "green"
        else:
            match_str = "[bold red]✗[/bold red]"
            match_style = "red"

        fmt_str = "[green]✓[/green]" if r["format_valid"] else "[red]✗[/red]"

        detail.add_row(
            r["scenario"],
            f"{r['category']} / {r['competitor']}",
            expected,
            f"[{match_style}]{actual}[/{match_style}]" if match_style else actual,
            _format_confidence(r.get("confidence")),
            match_str,
            fmt_str,
        )

    console.print()
    console.print(detail)

    # ── Summary table ─────────────────────────────────────────────────────────
    by_scenario: dict[str, list[dict]] = {}
    for r in ok:
        by_scenario.setdefault(r["scenario"], []).append(r)

    summary = Table(title="Accuracy by Scenario", show_lines=False)
    summary.add_column("Scenario", style="cyan", min_width=26)
    summary.add_column("Tested", justify="right", min_width=7)
    summary.add_column("Correct", justify="right", min_width=8)
    summary.add_column("Format OK", justify="right", min_width=10)
    summary.add_column("Accuracy", justify="right", min_width=9)

    total_clear, total_correct = 0, 0

    for scenario in SCENARIO_TYPES:
        rows = by_scenario.get(scenario, [])
        if not rows:
            continue
        ambiguous = EXPECTED_VERDICT.get(scenario) is None
        fmt_ok = sum(1 for r in rows if r["format_valid"])

        if ambiguous:
            summary.add_row(
                f"[dim]{scenario} (ambiguous)[/dim]",
                str(len(rows)),
                "[dim]n/a[/dim]",
                str(fmt_ok),
                "[dim]n/a[/dim]",
            )
        else:
            correct = sum(1 for r in rows if r["actual"] == r["expected"])
            pct = correct / len(rows) * 100
            total_clear += len(rows)
            total_correct += correct
            colour = "green" if pct == 100 else "yellow" if pct >= 50 else "red"
            summary.add_row(
                scenario,
                str(len(rows)),
                str(correct),
                str(fmt_ok),
                f"[{colour}]{pct:.0f}%[/{colour}]",
            )

    console.print()
    console.print(summary)

    if total_clear > 0:
        overall_pct = total_correct / total_clear * 100
        colour = "green" if overall_pct >= 80 else "yellow" if overall_pct >= 60 else "red"
        console.print(
            f"\n[bold]Overall accuracy (clear-cut scenarios): "
            f"[{colour}]{total_correct}/{total_clear} = {overall_pct:.1f}%[/{colour}][/bold]"
        )

    scored_conf = [_confidence_pct(r.get("confidence")) for r in ok]
    scored_conf = [pct for pct in scored_conf if pct is not None]
    correct_conf = [
        _confidence_pct(r.get("confidence"))
        for r in ok
        if r.get("expected") is not None and r.get("actual") == r.get("expected")
    ]
    correct_conf = [pct for pct in correct_conf if pct is not None]
    wrong_conf = [
        _confidence_pct(r.get("confidence"))
        for r in ok
        if r.get("expected") is not None and r.get("actual") != r.get("expected")
    ]
    wrong_conf = [pct for pct in wrong_conf if pct is not None]
    if scored_conf:
        console.print(
            f"[bold]Average verdict confidence:[/bold] {sum(scored_conf) / len(scored_conf):.1f}%"
        )
    if correct_conf and wrong_conf:
        console.print(
            f"[bold]Correct vs wrong confidence:[/bold] "
            f"{sum(correct_conf) / len(correct_conf):.1f}% / {sum(wrong_conf) / len(wrong_conf):.1f}%"
        )

    skipped = [r for r in results if r["status"] != "ok"]
    if skipped:
        console.print(f"\n[yellow]⚠  {len(skipped)} file(s) skipped or errored:[/yellow]")
        for r in skipped:
            console.print(f"  [dim]{r.get('source_file', '?')} — {r.get('reason', '')}[/dim]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SaaS Stack Manager model accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path",
                        default=str(MODEL_PATH),
                        help="Local path to base model directory.")
    parser.add_argument("--adapter-path",
                        default=str(_PROJECT_ROOT / "training" / "checkpoints_sft_cot"),
                        help="Path to fine-tuned LoRA adapter directory. Pass 'none' to run base model only.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Load base model with no LoRA adapter (diagnostic baseline).")
    parser.add_argument("--per-scenario", type=int, default=2, dest="per_scenario",
                        help="Signal files to test per scenario type (default: 2 → 36 total tests).")
    parser.add_argument("--scenario", default=None,
                        help="Evaluate one scenario label only.")
    parser.add_argument("--signals-dir",
                        default=str(_GENERATED_DIR),
                    help=("Directory of JSON signal files. Defaults to training/generated; "
                        f"use {_DEFAULT_HOLDOUT_DIR.relative_to(_PROJECT_ROOT)} for the realistic hold-out set."))
    parser.add_argument("--all-files", action="store_true",
                        help="Evaluate every JSON file in --signals-dir instead of sampling per scenario.")
    parser.add_argument("--holdout", action="store_true",
                help="Shortcut for --signals-dir fixtures/eval_signals --all-files.")
    parser.add_argument("--pipeline", choices=["lean", "voting"], default="lean",
                        help="Inference pipeline to evaluate. Use 'voting' to mirror the normal production route.")
    parser.add_argument("--populate-dashboard", action="store_true",
                        help="Write memos and live_run drift records so the dashboard reflects this evaluation run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use fixture responses — no GPU required.")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Evaluate specific signal files only (paths relative to project root "
                             "or absolute). Bypasses --scenario and --per-scenario sampling.")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["AGENT_DRY_RUN"] = "true"

    if args.holdout and not args.files:
        args.signals_dir = str(_DEFAULT_HOLDOUT_DIR)
        args.all_files = True

    if args.files:
        samples = [Path(f) if Path(f).is_absolute() else _PROJECT_ROOT / f for f in args.files]
        missing = [p for p in samples if not p.exists()]
        if missing:
            for p in missing:
                console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)
        mode_label = f"{len(samples)} explicit file(s)"
    else:
        signals_dir = Path(args.signals_dir)
        if not signals_dir.is_absolute():
            signals_dir = _PROJECT_ROOT / signals_dir
        samples = _collect_samples(args.scenario, args.per_scenario, signals_dir, args.all_files)
        if args.all_files:
            mode_label = f"all files in {signals_dir}"
        else:
            mode_label = f"{args.per_scenario} per scenario from {signals_dir}"

    if not samples:
        console.print("[red]No signal files found. Check the selected signals directory.[/red]")
        sys.exit(1)

    console.print(
        f"\n[bold]Evaluating {len(samples)} signal(s)[/bold] "
        f"({mode_label}, {'dry-run' if args.dry_run else 'live model'})"
    )

    adapter = None if (args.no_adapter or args.adapter_path.lower() == "none") else args.adapter_path
    tokenizer, model = load_model(
        model_path=args.model_path,
        adapter_path=adapter,
    )

    results: list[dict] = []
    correct = 0
    total_scored = 0

    console.print()
    for i, signal_path in enumerate(samples, 1):
        parsed = _parse_generated_filename(signal_path.stem)
        scenario = parsed[2] if parsed else "?"
        cat_comp = f"{parsed[0]}/{parsed[1]}" if parsed else signal_path.stem

        console.print(
            f"[dim][{i:3d}/{len(samples)}][/dim] [cyan]{scenario:<28}[/cyan] {cat_comp:<30}",
            end="",
        )

        result = _evaluate_one(signal_path, tokenizer, model, pipeline=args.pipeline)
        result.setdefault("source_file", signal_path.name)
        if args.populate_dashboard and result["status"] == "ok":
            result["memo_filename"] = _persist_dashboard_run(result)
        results.append(result)
        conf_text = _format_confidence(result.get("confidence"))

        if result["status"] == "ok":
            expected = result["expected"]
            actual = result["actual"] or "?"
            ambiguous = expected is None
            if ambiguous:
                console.print(f"→ [dim]{actual}[/dim]  [dim](ambiguous, p={conf_text})[/dim]")
            elif actual == expected:
                correct += 1
                total_scored += 1
                console.print(
                    f"→ [bold green]{actual}[/bold green] [green]✓[/green]"
                    f"  [dim]p={conf_text} · {correct}/{total_scored} correct[/dim]"
                )
            else:
                total_scored += 1
                console.print(
                    f"→ [bold red]{actual}[/bold red] [red]✗[/red]"
                    f"  expected [yellow]{expected}[/yellow]"
                    f"  [dim]p={conf_text} · {correct}/{total_scored} correct[/dim]"
                )
        else:
            console.print(f"→ [yellow]SKIP: {result.get('reason', '')}[/yellow]")

    _print_results(results)

    # Save results JSON
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = _OUTPUTS_DIR / f"eval_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "results": [_serialisable_result(result) for result in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")

    # ── Summarise dashboard memos (base model, no adapter) ───────────────────
    # After all verdicts are done we unload the verdict model (already out of scope
    # here) and load the untuned base model to produce plain-English summaries for
    # every memo that was written during this run.
    dashboard_results = [
        r for r in results
        if r["status"] == "ok" and args.populate_dashboard and r.get("memo_filename")
    ]
    if dashboard_results and not args.dry_run:
        console.print(f"\n[dim]Generating summaries for {len(dashboard_results)} memo(s) …[/dim]")
        unload_model(model)  # release verdict model before loading base
        base_tok, base_model = load_base_model(model_path=args.model_path)
        for r in dashboard_results:
            memo_path = _OUTPUTS_DIR / r["memo_filename"]
            if memo_path.exists():
                memo_text = memo_path.read_text(encoding="utf-8")
                _auto_summarise(memo_text, r["memo_filename"], base_tok, base_model)
        unload_model(base_model)
        console.print("[dim]Summaries written to outputs/summaries.json[/dim]")


if __name__ == "__main__":
    main()
