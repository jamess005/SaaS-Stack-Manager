"""
Drift check — runs the canary eval set and appends an accuracy_check record
to outputs/drift_log.jsonl.

Run this periodically (or when the agent advises it) to track whether the
fine-tuned model is still making correct decisions on known-answer signals.

Usage:
    # Live model (AMD GPU required)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/drift_check.py

    # Dry-run (no GPU — uses fixture responses for pipeline testing)
    python scripts/drift_check.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from training.drift_canaries import DRIFT_CANARY_EXPECTED, DRIFT_CANARY_FILES  # noqa: E402
from agent.context_loader import load_context  # noqa: E402
from agent.drift_tracker import _DRIFT_LOG, load_records  # noqa: E402
from agent.model_runner import load_model, run_lean  # noqa: E402
from agent.output_validator import extract_verdict_class, validate_lean_output  # noqa: E402
from agent.roi_calculator import calculate_roi, extract_pass1_vars  # noqa: E402
from agent.signal_interpreter import parse_signal_payload  # noqa: E402
from config import MODEL_PATH  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()

_DATA_ROOT = _PROJECT_ROOT / "data"

# Canary signals — known-answer fixtures not used in training
_CANARY_FILES = DRIFT_CANARY_FILES
_EXPECTED = DRIFT_CANARY_EXPECTED


def _run_canary(signal_path: Path, tokenizer, model) -> dict:
    # Re-parse using longest-prefix match
    from agent.context_loader import VALID_CATEGORIES
    from training.generate_signals import SCENARIO_TYPES

    competitor_slug = None
    scenario = None
    category = None
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if signal_path.stem.startswith(cat + "_"):
            remainder = signal_path.stem[len(cat) + 1:]
            for sc in sorted(SCENARIO_TYPES, key=len, reverse=True):
                if remainder.endswith("_" + sc):
                    competitor_slug = remainder[: -len(sc) - 1]
                    scenario = sc
                    category = cat
                    break
            if competitor_slug:
                break

    if not competitor_slug:
        return {"status": "skip", "file": signal_path.name, "reason": "unrecognised filename"}

    try:
        context = load_context(category, competitor_slug, _DATA_ROOT)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "error", "file": signal_path.name, "reason": str(exc)}

    with signal_path.open(encoding="utf-8") as f:
        signal = json.load(f)
    inbox_text = json.dumps(signal, indent=2, ensure_ascii=False)

    pass1 = extract_pass1_vars(context, parse_signal_payload(inbox_text))
    roi = calculate_roi(pass1)
    memo, confidence = run_lean(inbox_text, context, roi, tokenizer, model)
    is_valid, _ = validate_lean_output(memo)
    if not is_valid:
        memo, confidence = run_lean(inbox_text, context, roi, tokenizer, model)
        is_valid, _ = validate_lean_output(memo)

    actual = extract_verdict_class(memo)
    expected = _EXPECTED.get(signal_path.stem)

    # Retry once if the verdict is wrong — generation is stochastic and a
    # single bad sample shouldn't always count as a failure
    retried = False
    if actual != expected:
        retry_memo, retry_conf = run_lean(inbox_text, context, roi, tokenizer, model)
        retry_valid, _ = validate_lean_output(retry_memo)
        actual_retry = extract_verdict_class(retry_memo)
        if actual_retry == expected:
            memo, confidence, actual, is_valid = retry_memo, retry_conf, actual_retry, retry_valid
            retried = True

    result: dict = {
        "status": "ok",
        "file": signal_path.name,
        "expected": expected,
        "actual": actual,
        "correct": actual == expected,
        "format_valid": is_valid,
    }
    if retried:
        result["retried"] = True
    if confidence:
        result["confidence"] = confidence
    # Save actual model output when canary fails — used as DPO rejected trace on next retrain.
    # At this point memo is still the initial wrong attempt (retry only overwrites on success).
    if not result["correct"]:
        result["wrong_memo"] = memo
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canary eval and log accuracy check.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use fixture responses — no GPU required.")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--adapter-path",
                        default=str(_PROJECT_ROOT / "training" / "checkpoints_sft_cot"))
    parser.add_argument("--dpo-adapter-path", default=None,
                        help="Optional path to a DPO adapter to stack on top of the SFT adapter. "
                             "Disabled by default so canary runs measure the SFT adapter alone.")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["AGENT_DRY_RUN"] = "true"

    missing = [p for p in _CANARY_FILES if not p.exists()]
    if missing:
        console.print("[red]Missing canary signal files — regenerate with generate_signals.py:[/red]")
        for p in missing:
            console.print(f"  [dim]{p}[/dim]")
        sys.exit(1)

    console.print(f"\n[bold]Running canary eval[/bold] ({len(_CANARY_FILES)} signals, "
                  f"{'dry-run' if args.dry_run else 'live model'})\n")
    if args.dpo_adapter_path:
        console.print(f"[dim]Stacking DPO adapter: {args.dpo_adapter_path}[/dim]\n")
    else:
        console.print("[dim]Running SFT-only canary check (no DPO adapter stacked).[/dim]\n")

    adapter = None if args.dry_run else args.adapter_path
    dpo_adapter = None if args.dry_run else args.dpo_adapter_path
    tokenizer, model = load_model(model_path=args.model_path, adapter_path=adapter,
                                  dpo_adapter_path=dpo_adapter)

    results = []
    for path in _CANARY_FILES:
        console.print(f"  [dim]{path.name}[/dim] ... ", end="")
        r = _run_canary(path, tokenizer, model)
        results.append(r)
        if r["status"] == "ok":
            mark = "[green]✓[/green]" if r["correct"] else "[red]✗[/red]"
            retry_tag = " [yellow](retried)[/yellow]" if r.get("retried") else ""
            conf_info = ""
            if "confidence" in r:
                c = r["confidence"]
                conf_info = f"  prob={c.get('verdict_token_prob', '?'):.3f} margin={c.get('verdict_margin', '?'):.3f}"
            console.print(f"{r['actual']} {mark}{retry_tag}  (expected {r['expected']}){conf_info}")
        else:
            console.print(f"[yellow]SKIP: {r.get('reason', '')}[/yellow]")

    ok = [r for r in results if r["status"] == "ok"]
    correct = sum(1 for r in ok if r["correct"])
    total = len(ok)
    accuracy = correct / total if total else 0.0

    colour = "green" if accuracy >= 0.9 else "yellow" if accuracy >= 0.75 else "red"
    console.print(f"\n[bold]Canary accuracy: [{colour}]{correct}/{total} = {accuracy:.0%}[/{colour}][/bold]")

    # Append accuracy_check record
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "accuracy_check",
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "results": results,
    }
    _DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DRIFT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"\n[dim]Logged to {_DRIFT_LOG}[/dim]")

    if accuracy < 0.75:
        console.print("\n[bold red]⚠  Accuracy below 75% — model may have drifted. "
                      "Review outputs and consider retraining.[/bold red]")


if __name__ == "__main__":
    main()
