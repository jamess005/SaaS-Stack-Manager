"""Batch-run the agent pipeline on all market_inbox files that don't have a
corresponding output memo for today's date yet. Skips already-processed files."""

import subprocess
import sys
import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_INBOX_DIR = _PROJECT_ROOT / "market_inbox"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

today = datetime.date.today().isoformat()


def _has_output(category: str, slug: str) -> bool:
    pattern = f"*-{category}-{slug}.md"
    return any(_OUTPUTS_DIR.glob(pattern))


def main():
    inbox_files = sorted(_INBOX_DIR.glob("*.md"))
    total = len(inbox_files)
    done = 0
    skipped = 0
    failed = 0

    print(f"Found {total} inbox files. Checking for existing outputs...\n")

    for f in inbox_files:
        # Parse category_slug from filename
        stem = f.stem  # e.g. "finance_ledgerflow"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            print(f"SKIP (bad name): {f.name}")
            skipped += 1
            continue
        category, slug = parts[0], parts[1]

        if _has_output(category, slug):
            print(f"SKIP (output exists): {f.name}")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING [{done+skipped+failed+1}/{total}]: {f.name}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, "-m", "agent.agent", str(f)],
            cwd=str(_PROJECT_ROOT),
        )

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
