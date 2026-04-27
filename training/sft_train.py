"""
SFT training script — trains Qwen2.5-3B-Instruct on CoT reasoning traces.

Produces a LoRA adapter that has learned the explicit reasoning process:
push signals → pull signals → compliance → ROI → hold condition → verdict.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/sft_train.py

    # Override defaults
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/sft_train.py \\
        --model-path /path/to/model --epochs 3 --lr 2e-4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from config import MODEL_PATH as _cfg_model_path  # noqa: E402
_DEFAULT_MODEL_PATH = str(_cfg_model_path)
_DEFAULT_DATA_PATH  = str(_PROJECT_ROOT / "training" / "sft_cot_traces.jsonl")
_DEFAULT_ADAPTER_OUT = str(_PROJECT_ROOT / "training" / "checkpoints_sft_cot")


def _load_records(data_path: str) -> list[dict]:
    with open(data_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _infer_val_data_path(train_data_path: str) -> str | None:
    train_path = Path(train_data_path)
    if train_path.name.endswith(".train.jsonl"):
        candidate = train_path.with_name(train_path.name.replace(".train.jsonl", ".val.jsonl"))
        if candidate.exists():
            return str(candidate)
    return None


def _prefer_split_data_paths(data_path: str) -> tuple[str, str | None]:
    data_path_obj = Path(data_path)

    if data_path_obj.name.endswith(".train.jsonl"):
        return data_path, _infer_val_data_path(data_path)

    train_candidate = data_path_obj.with_name(f"{data_path_obj.stem}.train{data_path_obj.suffix}")
    val_candidate = data_path_obj.with_name(f"{data_path_obj.stem}.val{data_path_obj.suffix}")

    if train_candidate.exists() and val_candidate.exists():
        return str(train_candidate), str(val_candidate)

    return data_path, None


def _resolve_eval_data_path(
    *,
    enable_eval: bool,
    explicit_val_data_path: str | None,
    inferred_val_data_path: str | None,
) -> str | None:
    if not enable_eval:
        return None
    return explicit_val_data_path or inferred_val_data_path


def _validate_sequence_lengths(
    records: list[dict],
    tokenizer,
    *,
    max_seq_len: int,
    dataset_name: str,
) -> None:
    if not records:
        return

    lengths: list[int] = []
    too_long: list[tuple[int, int]] = []

    for index, record in enumerate(records):
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        token_count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        lengths.append(token_count)
        if token_count > max_seq_len:
            too_long.append((index, token_count))

    sorted_lengths = sorted(lengths)
    p95_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))
    logger.info(
        "%s token lengths: max=%d, p95=%d, over_limit=%d",
        dataset_name,
        sorted_lengths[-1],
        sorted_lengths[p95_index],
        len(too_long),
    )

    if too_long:
        examples = ", ".join(f"#{idx}={length}" for idx, length in too_long[:5])
        raise ValueError(
            f"{dataset_name} has {len(too_long)} record(s) longer than max_seq_len={max_seq_len}. "
            f"Examples: {examples}. Increase --max-seq-len or shorten the traces."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT CoT training for SaaS decision model.")
    parser.add_argument("--model-path",   default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path",    default=_DEFAULT_DATA_PATH)
    parser.add_argument("--train-data-path", default=None,
                        help="Optional explicit train split JSONL path. Overrides --data-path.")
    parser.add_argument("--val-data-path", default=None,
                        help="Optional validation split JSONL path for epoch-level evaluation.")
    parser.add_argument("--adapter-out",  default=_DEFAULT_ADAPTER_OUT)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max-seq-len",  type=int,   default=2048)
    parser.add_argument(
        "--enable-eval",
        action="store_true",
        help=(
            "Run epoch-level validation on the val split. Disabled by default because "
            "evaluation forward passes can OOM on 16GB ROCm GPUs with long traces."
        ),
    )
    args = parser.parse_args()

    if args.train_data_path:
        train_data_path = args.train_data_path
        inferred_val_path = _infer_val_data_path(train_data_path)
    else:
        train_data_path, inferred_val_path = _prefer_split_data_paths(args.data_path)

    val_data_path = _resolve_eval_data_path(
        enable_eval=args.enable_eval,
        explicit_val_data_path=args.val_data_path,
        inferred_val_data_path=inferred_val_path,
    )

    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION not set. "
            "For RX 7800 XT (gfx1101) set HSA_OVERRIDE_GFX_VERSION=11.0.0."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────────
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    device_map = {"": 0} if is_rocm else "auto"
    if is_rocm:
        logger.info("ROCm backend (HIP %s) — explicit device_map.", torch.version.hip)

    logger.info("Loading model at bf16 (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded: %.0fM parameters, dtype=%s", n_params, model.dtype)

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info("LoRA config: r=%d, alpha=%d", lora_config.r, lora_config.lora_alpha)

    # ── Dataset ────────────────────────────────────────────────────────────────
    logger.info("Loading training dataset from: %s", train_data_path)
    if train_data_path != args.data_path and not args.train_data_path:
        logger.info("Detected adjacent train/val split next to %s — training will use the split files by default.", args.data_path)
    records = _load_records(train_data_path)
    logger.info("Training dataset: %d examples", len(records))
    if not records:
        raise ValueError(f"Training dataset is empty: {train_data_path}")

    # Verify format: each record must have a 'messages' key with 3 turns
    sample = records[0] if records else {}
    assert "messages" in sample and len(sample["messages"]) == 3, (
        "Expected records with {\"messages\": [system, user, assistant]}"
    )
    _validate_sequence_lengths(
        records,
        tokenizer,
        max_seq_len=args.max_seq_len,
        dataset_name="Training dataset",
    )

    train_dataset = Dataset.from_list(records)

    eval_dataset = None
    if inferred_val_path and not args.enable_eval:
        logger.info(
            "Detected validation split at %s but leaving eval disabled by default for ROCm safety. "
            "Pass --enable-eval to opt in.",
            inferred_val_path,
        )
    if val_data_path:
        logger.info("Loading validation dataset from: %s", val_data_path)
        eval_records = _load_records(val_data_path)
        if eval_records:
            eval_sample = eval_records[0]
            assert "messages" in eval_sample and len(eval_sample["messages"]) == 3, (
                "Expected validation records with {\"messages\": [system, user, assistant]}"
            )
            _validate_sequence_lengths(
                eval_records,
                tokenizer,
                max_seq_len=args.max_seq_len,
                dataset_name="Validation dataset",
            )
            eval_dataset = Dataset.from_list(eval_records)
            logger.info("Validation dataset: %d examples", len(eval_records))
        else:
            logger.warning("Validation dataset is empty: %s", val_data_path)

    # ── SFT config ─────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.adapter_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        bf16=True,
        max_length=args.max_seq_len,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        do_eval=eval_dataset is not None,
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )
    logger.info(
        "SFT config: epochs=%d, lr=%.0e, batch=1×ga8, max_seq=%d, eval=%s",
        sft_config.num_train_epochs,
        sft_config.learning_rate,
        sft_config.max_length,
        "on" if eval_dataset is not None else "off",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    trainer.save_model(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    logger.info("Adapter and tokenizer saved to: %s", args.adapter_out)


if __name__ == "__main__":
    main()
