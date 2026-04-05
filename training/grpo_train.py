"""
GRPO training script — Batched Contextual Reinforcement for SaaS decision model.

Trains Qwen2.5-1.5B-Instruct with:
  - LoRA r=16, alpha=32 across all attention + MLP projections
  - GRPO reward = mean per-instance VERDICT exact-match across N=4 BCR cases
  - bf16, no quantization (~3 GB VRAM on AMD RX 7800 XT)
  - 5 epochs, ~40–75 min on ROCm 7.2

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/grpo_train.py

    # Override defaults
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/grpo_train.py \\
        --model-path /path/to/model --epochs 3 --lr 2e-5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Restrict to the discrete RX 7800 XT (GPU 0) — prevents DataParallel from
# pulling in the integrated Radeon (GPU 1) which causes a hang on ROCm.
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "/home/james/ml-proj/models/qwen2.5-1.5b-instruct"
_DEFAULT_ADAPTER_OUT = str(_PROJECT_ROOT / "training" / "checkpoints_grpo")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO BCR training for SaaS decision model.")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-out", default=_DEFAULT_ADAPTER_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-generations", type=int, default=2,
                        help="G rollouts per prompt for policy gradient estimate.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.02,
                        help="KL penalty coefficient.")
    args = parser.parse_args()

    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION not set. "
            "For RX 7800 XT (gfx1101) set HSA_OVERRIDE_GFX_VERSION=11.0.0."
        )

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from agent.grpo_rewards import bcr_reward_fn
    from training.bcr_dataset import build_bcr_dataset

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    device_map = {"": 0} if is_rocm else "auto"
    if is_rocm:
        logger.info("ROCm backend detected (HIP %s) — using explicit device_map.", torch.version.hip)

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
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
            "gate_proj", "up_proj", "down_proj",        # MLP projections
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info("LoRA config: r=%d, alpha=%d", lora_config.r, lora_config.lora_alpha)

    # ── Dataset ────────────────────────────────────────────────────────────────
    logger.info("Building BCR training dataset...")
    train_dataset = build_bcr_dataset()
    logger.info("Dataset: %d BCR batches", len(train_dataset))

    # ── GRPO config ────────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.adapter_out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,      # one BCR context per device step
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        temperature=0.8,
        max_completion_length=args.max_new_tokens,
        beta=args.beta,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        # Disable vLLM — use native HF generation (ROCm-safe)
        use_vllm=False,
    )
    logger.info(
        "GRPO config: epochs=%d, lr=%.0e, G=%d, max_new_tokens=%d, beta=%.3f",
        grpo_config.num_train_epochs,
        grpo_config.learning_rate,
        grpo_config.num_generations,
        grpo_config.max_completion_length,
        grpo_config.beta,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=bcr_reward_fn,
        peft_config=lora_config,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    trainer.save_model(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    logger.info("Adapter and tokenizer saved to: %s", args.adapter_out)


if __name__ == "__main__":
    main()
