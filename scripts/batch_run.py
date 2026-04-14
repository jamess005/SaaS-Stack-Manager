"""Batch-run the agent pipeline on all market_inbox files that don't have a
corresponding output memo yet. Skips already-processed files.

Usage:
    # Use DPO adapter (default if training/checkpoints_dpo/ exists)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/batch_run.py

    # Explicit adapter
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/batch_run.py --adapter-path training/checkpoints_sft_cot

    # Dry-run (no GPU)
    python scripts/batch_run.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_INBOX_DIR = _PROJECT_ROOT / "market_inbox"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_DPO_ADAPTER = _PROJECT_ROOT / "training" / "checkpoints_dpo"
_SFT_ADAPTER = _PROJECT_ROOT / "training" / "checkpoints_sft_cot"


def _has_output(category: str, slug: str) -> bool:
    pattern = f"*-{category}-{slug}.md"
    return any(_OUTPUTS_DIR.glob(pattern))


def _default_adapter() -> str | None:
    """Return the SFT adapter path if it exists."""
    if _SFT_ADAPTER.exists():
        return str(_SFT_ADAPTER)
    return None


def _default_dpo_adapter() -> str | None:
    """Return the DPO adapter path if it exists (stacked on top of SFT)."""
    if _DPO_ADAPTER.exists() and (_DPO_ADAPTER / "adapter_config.json").exists():
        return str(_DPO_ADAPTER)
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch-run agent on all inbox files.")
    parser.add_argument("--adapter-path", default=None,
                        help="SFT LoRA adapter path. Defaults to training/checkpoints_sft_cot.")
    parser.add_argument("--dpo-adapter-path", default=None,
                        help="DPO adapter path (stacked on SFT). Auto-detected if present.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use fixture responses — no GPU required.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    args = parser.parse_args()

    adapter = args.adapter_path or _default_adapter()
    dpo_adapter = args.dpo_adapter_path or _default_dpo_adapter()

    inbox_files = sorted(_INBOX_DIR.glob("*.md"))
    total = len(inbox_files)
    done = 0
    skipped = 0
    failed = 0

    print(f"Found {total} inbox files.")
    print(f"Adapter: {adapter or 'none (base model)'}")
    if dpo_adapter:
        print(f"DPO adapter: {dpo_adapter}")
    print()

    for f in inbox_files:
        stem = f.stem  # e.g. "finance_ledgerflow"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            print(f"SKIP (bad name): {f.name}")
            skipped += 1
            continue
        category, slug = parts[0], parts[1]

        if not args.force and _has_output(category, slug):
            print(f"SKIP (output exists): {f.name}")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING [{done+skipped+failed+1}/{total}]: {f.name}")
        print(f"{'='*60}")

        cmd = [sys.executable, "-m", "agent.agent", str(f)]
        if args.dry_run:
            cmd.append("--dry-run")
        if adapter:
            cmd += ["--adapter-path", adapter]
        if dpo_adapter:
            cmd += ["--dpo-adapter-path", dpo_adapter]

        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))

        if result.returncode == 0:
            done += 1
            print(f"✓ DONE: {f.name}")
        else:
            failed += 1
            print(f"✗ FAILED: {f.name} (exit code {result.returncode})")

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {done} succeeded, {skipped} skipped, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
