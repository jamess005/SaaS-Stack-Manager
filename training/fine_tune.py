"""
Phase 3 — QLoRA fine-tuning script for SaaS Stack Manager.

Target model: Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct.
Hardware: AMD RX 7800 XT — 16GB VRAM, ROCm 7.2.

Training data: JSONL reasoning traces in training/traces/
Each trace has the format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...],
   "metadata": {"scenario": "...", ...}}

Usage:
    python training/fine_tune.py --model qwen2.5-3b --traces-dir training/traces/

Environment:
    ROCR_VISIBLE_DEVICES=0
    HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7800 XT is gfx1101
    HIP_VISIBLE_DEVICES=0
"""

import argparse
import json
import os
import warnings
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# ── Model path registry ────────────────────────────────────────────────────────
_MODEL_PATHS: dict[str, str] = {
    "qwen2.5-3b": "/home/james/ml-proj/models/qwen2.5-3b-instruct",
    "llama-3.2-3b": "/home/james/ml-proj/models/llama-3.2-3b-instruct",
}


def build_model_path(model_name: str) -> str:
    if model_name not in _MODEL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid choices: {list(_MODEL_PATHS)}"
        )
    return _MODEL_PATHS[model_name]


# ── Trace loading ──────────────────────────────────────────────────────────────


def load_traces(traces_dir: str) -> list[dict]:
    """
    Walk traces_dir for all .jsonl files, parse each line as JSON.
    Skips malformed lines with a warning.
    Each valid record must have a 'messages' key whose first three roles are
    system / user / assistant (multi-turn conversations are allowed).
    """
    records: list[dict] = []
    scenario_counts: Counter = Counter()
    skipped = 0

    for path in sorted(Path(traces_dir).rglob("*.jsonl")):
        # Skip backup files (e.g. traces_backup.jsonl, *.pre_split, etc.)
        if "backup" in path.stem or path.stem != "traces":
            warnings.warn(f"Skipping non-primary trace file: {path.name}")
            continue
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    warnings.warn(f"{path}:{lineno}: malformed JSON — {exc}")
                    skipped += 1
                    continue

                if "messages" not in record:
                    warnings.warn(f"{path}:{lineno}: missing 'messages' key — skipped")
                    skipped += 1
                    continue

                msgs = record["messages"]
                roles = [m["role"] for m in msgs]
                assert len(msgs) >= 3 and roles[:3] == ["system", "user", "assistant"], (
                    f"{path}:{lineno}: expected messages[0:3] to be "
                    f"[system, user, assistant], got {roles[:3]}"
                )

                scenario = record.get("metadata", {}).get("scenario", "unknown")
                scenario_counts[scenario] += 1
                records.append(record)

    if skipped:
        console.print(f"[yellow]⚠  Skipped {skipped} malformed lines.[/yellow]")

    console.print(f"\n[bold green]✓ Loaded {len(records)} traces[/bold green] from [cyan]{traces_dir}[/cyan]\n")

    table = Table(title="Scenario breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Count", justify="right")
    for scenario, count in sorted(scenario_counts.items()):
        table.add_row(scenario, str(count))
    console.print(table)

    return records


# ── Training ───────────────────────────────────────────────────────────────────


def train(model_name: str, traces_dir: str, output_dir: str) -> None:
    # Set ROCm env vars before importing torch / HIP drivers
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


    import torch  # type: ignore[import-untyped]
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-untyped]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-untyped]

    model_path = build_model_path(model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = load_traces(traces_dir)

    # Shuffle with a fixed seed so results are reproducible.
    import random
    rng = random.Random(42)
    rng.shuffle(records)

    console.print(f"\n[bold]Loading model:[/bold] [cyan]{model_path}[/cyan]")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},  # explicit placement — device_map="auto" breaks ROCm 7.x
        trust_remote_code=True,
    )
    console.print("[green]✓ Model loaded[/green]")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def to_prompt_completion(recs):
        prompt_completion: list[dict[str, str]] = []
        eos_token = tokenizer.eos_token or ""
        for rec in recs:
            prompt = tokenizer.apply_chat_template(
                rec["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            completion = rec["messages"][-1]["content"]
            if eos_token and not completion.endswith(eos_token):
                completion = completion + eos_token
            prompt_completion.append({"prompt": prompt, "completion": completion})
        return prompt_completion

    dataset = Dataset.from_list(to_prompt_completion(records))
    console.print(f"[green]✓ Dataset built:[/green] {len(dataset)} train examples")
    # Eval excluded from trainer — eval forward passes run without gradient checkpointing
    # and OOM on 16 GB VRAM with pull examples up to 1,474 tokens.
    # Use evaluate_model.py after training for accuracy evaluation.

    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        report_to="none",
        max_length=4096,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    console.print("\n[bold yellow]▶ Training started…[/bold yellow]")
    train_result = trainer.train()

    final_loss = train_result.training_loss
    console.print(f"\n[bold green]✓ Training complete[/bold green]  final loss: [cyan]{final_loss:.4f}[/cyan]")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"[green]✓ Adapter saved to:[/green] [cyan]{output_dir}[/cyan]")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SaaS Stack Manager model with QLoRA on ROCm (Phase 3)."
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-3b",
        choices=list(_MODEL_PATHS),
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--traces-dir",
        default="training/traces/",
        help="Directory containing JSONL trace files.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/",
        help="Output directory for LoRA adapter weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate traces, print stats, then exit without loading any model.",
    )
    args = parser.parse_args()

    if args.dry_run:
        records = load_traces(args.traces_dir)

        # Token-length stats using a fast approximation (chars / 4)
        lengths = [
            (
                sum(len(m["content"]) for m in r["messages"][:-1])
                + len(r["messages"][-1]["content"])
            ) // 4
            for r in records
        ]
        lengths.sort()
        n = len(lengths)
        console.print(
            f"\n[bold]Approx token-length stats (chars÷4):[/bold]\n"
            f"  min={lengths[0]}  median={lengths[n//2]}  "
            f"p95={lengths[int(n*0.95)]}  max={lengths[-1]}"
        )
        console.print("\n[bold green]Dry run complete — no model loaded.[/bold green]")
        return

    train(args.model, args.traces_dir, args.output_dir)


if __name__ == "__main__":
    main()
