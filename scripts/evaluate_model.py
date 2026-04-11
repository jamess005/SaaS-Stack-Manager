"""
Model evaluation script — tests the fine-tuned SaaS Stack Manager against
the generated signal corpus and reports accuracy per scenario type.

Usage:
    # Dry-run (no GPU, uses fixture responses)
    python scripts/evaluate_model.py --dry-run

    # Full evaluation, 2 samples per scenario type (~11 min with GPU)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/evaluate_model.py \\
        --model-path /home/james/ml-proj/models/qwen2.5-3b-instruct \\
        --adapter-path training/checkpoints/ \\
        --limit 2

    # One scenario type only
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/evaluate_model.py \\
        --model-path /home/james/ml-proj/models/qwen2.5-3b-instruct \\
        --adapter-path training/checkpoints/ \\
        --scenario hard_compliance_failure --limit 5
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
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES, load_context  # noqa: E402
from agent.model_runner import load_model, run_lean  # noqa: E402
from agent.output_validator import extract_verdict_class, validate_lean_output  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import parse_signal_payload  # noqa: E402
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


# ── Signal sampling ────────────────────────────────────────────────────────────

def _collect_samples(scenario_filter: str | None, limit: int) -> list[Path]:
    """
    Return a reproducible sample of signal files grouped by scenario type.
    Uses a fixed seed so the same files are always selected.
    """
    by_scenario: dict[str, list[Path]] = {s: [] for s in SCENARIO_TYPES}

    for path in _GENERATED_DIR.glob("*.json"):
        parsed = _parse_generated_filename(path.stem)
        if parsed is None:
            continue
        _, _, scenario = parsed
        if scenario in by_scenario:
            by_scenario[scenario].append(path)

    selected: list[Path] = []
    rng = random.Random(42)
    scenarios_to_run = [scenario_filter] if scenario_filter else SCENARIO_TYPES
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
) -> dict:
    """Run the pipeline on one signal file and return a result record."""
    parsed = _parse_generated_filename(signal_path.stem)
    if parsed is None:
        return {
            "status": "skip",
            "reason": f"Cannot parse filename: {signal_path.name}",
        }

    category, competitor_slug, scenario = parsed

    try:
        context = load_context(category, competitor_slug, _DATA_ROOT)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "error", "reason": str(exc), "scenario": scenario,
                "category": category, "competitor": competitor_slug}

    with signal_path.open(encoding="utf-8") as f:
        signal = json.load(f)
    inbox_text = json.dumps(signal, indent=2, ensure_ascii=False)

    pass1_payload = extract_pass1_vars(context, parse_signal_payload(inbox_text))
    roi_result = calculate_roi(pass1_payload)

    memo = run_lean(inbox_text, context, roi_result, tokenizer, model)

    # Retry once on validation failure
    is_valid, _errors = validate_lean_output(memo)
    if not is_valid:
        memo = run_lean(inbox_text, context, roi_result, tokenizer, model)
        is_valid, _ = validate_lean_output(memo)

    verdict = extract_verdict_class(memo)
    expected = EXPECTED_VERDICT.get(scenario)

    result: dict = {
        "status": "ok",
        "scenario": scenario,
        "category": category,
        "competitor": competitor_slug,
        "expected": expected,
        "actual": verdict,
        "format_valid": is_valid,
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
                        default="/home/james/ml-proj/models/qwen2.5-3b-instruct",
                        help="Local path to base model directory.")
    parser.add_argument("--adapter-path",
                        default=str(_PROJECT_ROOT / "training" / "checkpoints_sft_cot"),
                        help="Path to fine-tuned LoRA adapter directory. Pass 'none' to run base model only.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Load base model with no LoRA adapter (diagnostic baseline).")
    parser.add_argument("--per-scenario", type=int, default=2, dest="per_scenario",
                        help="Signal files to test per scenario type (default: 2 → 36 total tests).")
    parser.add_argument("--scenario", default=None, choices=SCENARIO_TYPES,
                        help="Evaluate one scenario type only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use fixture responses — no GPU required.")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Evaluate specific signal files only (paths relative to project root "
                             "or absolute). Bypasses --scenario and --per-scenario sampling.")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["AGENT_DRY_RUN"] = "true"

    if args.files:
        samples = [Path(f) if Path(f).is_absolute() else _PROJECT_ROOT / f for f in args.files]
        missing = [p for p in samples if not p.exists()]
        if missing:
            for p in missing:
                console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)
        mode_label = f"{len(samples)} explicit file(s)"
    else:
        samples = _collect_samples(args.scenario, args.per_scenario)
        mode_label = f"{args.per_scenario} per scenario"

    if not samples:
        console.print("[red]No signal files found. Check training/generated/ exists.[/red]")
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

        result = _evaluate_one(signal_path, tokenizer, model)
        result.setdefault("source_file", signal_path.name)
        results.append(result)

        if result["status"] == "ok":
            expected = result["expected"]
            actual = result["actual"] or "?"
            ambiguous = expected is None
            if ambiguous:
                console.print(f"→ [dim]{actual}[/dim]  [dim](ambiguous)[/dim]")
            elif actual == expected:
                correct += 1
                total_scored += 1
                console.print(
                    f"→ [bold green]{actual}[/bold green] [green]✓[/green]"
                    f"  [dim]{correct}/{total_scored} correct[/dim]"
                )
            else:
                total_scored += 1
                console.print(
                    f"→ [bold red]{actual}[/bold red] [red]✗[/red]"
                    f"  expected [yellow]{expected}[/yellow]"
                    f"  [dim]{correct}/{total_scored} correct[/dim]"
                )
        else:
            console.print(f"→ [yellow]SKIP: {result.get('reason', '')}[/yellow]")

    _print_results(results)

    # Save results JSON
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = _OUTPUTS_DIR / f"eval_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
