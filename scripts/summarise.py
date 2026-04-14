"""
Summariser — generates plain-English verdict summaries using base Qwen2.5-3B.

Reads recent verdict memos from outputs/*.md, generates a 2-3 sentence summary
for each one that doesn't yet have a summary in outputs/summaries.json, then
writes the updated summaries.json.

Must not run while the agent is running. Creates outputs/.model_lock while running.

Usage:
    python scripts/summarise.py
    python scripts/summarise.py --dry-run   # prints summaries, does not write
"""

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.data_layer import (  # noqa: E402
    _LOCK_FILE, _OUTPUTS_DIR, _SUMMARIES,
    clear_model_busy, is_model_busy, set_model_busy,
)

_SYSTEM_PROMPT = (
    "You are an internal analyst writing a brief for a decision-maker. "
    "Read the verdict memo and write 2-3 factual sentences that state: "
    "what the key push issue(s) are with the current tool, what the competitor "
    "change(s) address, whether the ROI threshold was met, and what the verdict is. "
    "Use neutral, precise language. Do not use promotional or persuasive phrasing. "
    "No bullet points."
)


def _extract_memo_context(memo_text: str) -> dict:
    """Pull key fields from a verdict memo."""
    def _find(pattern):
        m = re.search(pattern, memo_text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    verdict = _find(r"^VERDICT:\s*(\w+)")
    if not verdict:
        m = re.search(r"VERDICT:\s*(\w+)", memo_text)
        verdict = m.group(1).strip() if m else ""

    competitor = _find(r"^COMPETITOR:\s*(.+)")
    current = _find(r"^CURRENT TOOL:\s*(.+)")
    evidence_block = _find(r"EVIDENCE:\s*(.+?)(?=\n[A-Z][A-Z\s]+:|$)")
    quotes = re.findall(r'"([^"]+)"', evidence_block)

    return {
        "verdict": verdict,
        "competitor": competitor,
        "current_tool": current,
        "evidence": quotes[:4],
    }


def _build_prompt(memo_text: str, ctx: dict) -> str:
    lines = [
        f"Verdict: {ctx['verdict']}",
        f"Current tool: {ctx['current_tool']}",
        f"Recommended: {ctx['competitor']}",
        "",
        "Key evidence from memo:",
    ]
    for q in ctx["evidence"]:
        lines.append(f'- "{q}"')
    lines += ["", "Full memo:", memo_text[:1200]]
    return "\n".join(lines)


def _summarise_one(memo_text: str, tokenizer, model) -> str:
    ctx = _extract_memo_context(memo_text)
    user_content = _build_prompt(memo_text, ctx)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plain-English verdict summaries.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summaries without writing to disk.")
    parser.add_argument("--model-path", default=None,
                        help="Override base model path.")
    args = parser.parse_args()

    if is_model_busy():
        print("Model is busy (lock file exists). Try again later.", file=sys.stderr)
        sys.exit(1)

    # Find memos that need a summary
    existing = {}
    if _SUMMARIES.exists():
        try:
            existing = json.loads(_SUMMARIES.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    memos = sorted(_OUTPUTS_DIR.glob("*.md"))
    pending = [m for m in memos if m.name not in existing]

    if not pending:
        print("All memos already have summaries.")
        return

    print(f"Generating summaries for {len(pending)} memo(s)…")

    if args.dry_run:
        for m in pending:
            print(f"\n--- {m.name} ---")
            ctx = _extract_memo_context(m.read_text(encoding="utf-8"))
            print(f"  [dry-run] Would summarise: {ctx['verdict']} — {ctx['competitor']}")
        return

    try:
        set_model_busy("summarise")
        from agent.model_runner import load_model  # noqa: E402
        from config import MODEL_PATH  # noqa: E402
        model_path = args.model_path or str(MODEL_PATH)
        tokenizer, model = load_model(model_path=model_path, adapter_path=None)

        for memo_path in pending:
            print(f"  {memo_path.name} … ", end="", flush=True)
            try:
                memo_text = memo_path.read_text(encoding="utf-8")
                summary = _summarise_one(memo_text, tokenizer, model)
                existing[memo_path.name] = summary
                print("done")
            except Exception as exc:
                print(f"ERROR: {exc}")

        _SUMMARIES.parent.mkdir(parents=True, exist_ok=True)
        _SUMMARIES.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSummaries written to {_SUMMARIES}")

    finally:
        clear_model_busy()


if __name__ == "__main__":
    main()
