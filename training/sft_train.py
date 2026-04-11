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


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT CoT training for SaaS decision model.")
    parser.add_argument("--model-path",   default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path",    default=_DEFAULT_DATA_PATH)
    parser.add_argument("--adapter-out",  default=_DEFAULT_ADAPTER_OUT)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max-seq-len",  type=int,   default=1024)
    args = parser.parse_args()

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
    logger.info("Loading dataset from: %s", args.data_path)
    with open(args.data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Dataset: %d examples", len(records))

    # Verify format: each record must have a 'messages' key with 3 turns
    sample = records[0] if records else {}
    assert "messages" in sample and len(sample["messages"]) == 3, (
        "Expected records with {\"messages\": [system, user, assistant]}"
    )

    dataset = Dataset.from_list(records)

    # ── SFT config ─────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.adapter_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        max_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    logger.info(
        "SFT config: epochs=%d, lr=%.0e, batch=2×ga4, max_seq=%d",
        sft_config.num_train_epochs,
        sft_config.learning_rate,
        sft_config.max_length,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    trainer.save_model(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    logger.info("Adapter and tokenizer saved to: %s", args.adapter_out)


if __name__ == "__main__":
    main()
