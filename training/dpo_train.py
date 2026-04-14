"""
DPO training script — Direct Preference Optimization on feedback pairs.

Trains the existing SFT CoT adapter further using DPO, learning from:
  - Human feedback corrections (dashboard 👍/👎)
  - Canary regression failures (drift_check.py)

The harvester (feedback_harvester.py) produces preference pairs; this script
trains the model to prefer the correct verdicts over the wrong ones.

Usage:
    # Harvest pairs first, then train
    python training/feedback_harvester.py
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/dpo_train.py

    # Override defaults
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python training/dpo_train.py \\
        --epochs 2 --lr 5e-5 --beta 0.1

    # Dry-run (validate data only, no GPU)
    python training/dpo_train.py --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from config import MODEL_PATH as _cfg_model_path  # noqa: E402

_DEFAULT_MODEL_PATH = str(_cfg_model_path)
_DEFAULT_SFT_ADAPTER = str(_PROJECT_ROOT / "training" / "checkpoints_sft_cot")
_DEFAULT_DPO_ADAPTER_OUT = str(_PROJECT_ROOT / "training" / "checkpoints_dpo")
_DEFAULT_DATA_PATH = str(_PROJECT_ROOT / "training" / "feedback_pairs.jsonl")
_CANARY_SCRIPT = str(_PROJECT_ROOT / "scripts" / "drift_check.py")


def _load_pairs(data_path: str) -> list[dict]:
    """Load DPO pairs from feedback_pairs.jsonl."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_prompt(messages: list[dict], tokenizer) -> str:
    """Apply chat template to prompt messages (system + user)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _run_canary_gate(model_path: str, adapter_path: str) -> tuple[bool, float]:
    """
    Run canary eval after training. Returns (passed, accuracy).

    The gate passes if accuracy >= baseline (doesn't regress).
    """
    logger.info("Running canary gate validation...")
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable, _CANARY_SCRIPT,
            "--model-path", model_path,
            "--adapter-path", adapter_path,
        ],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    logger.info("Canary gate stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("Canary gate failed: %s", result.stderr)
        return False, 0.0

    # Parse accuracy from drift_log — last accuracy_check record
    drift_log = _PROJECT_ROOT / "outputs" / "drift_log.jsonl"
    if not drift_log.exists():
        return False, 0.0

    last_check = None
    with drift_log.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "accuracy_check":
                    last_check = rec
            except json.JSONDecodeError:
                pass

    if last_check is None:
        return False, 0.0

    accuracy = last_check.get("accuracy", 0.0)
    return accuracy >= 0.75, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training on feedback pairs.")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--sft-adapter", default=_DEFAULT_SFT_ADAPTER,
                        help="Path to the base SFT LoRA adapter to build on.")
    parser.add_argument("--adapter-out", default=_DEFAULT_DPO_ADAPTER_OUT)
    parser.add_argument("--data-path", default=_DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.3,
                        help="DPO beta — controls deviation from reference policy. "
                             "Higher values keep the model closer to the reference.")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data format only — no training.")
    parser.add_argument("--skip-canary", action="store_true",
                        help="Skip post-training canary validation.")
    args = parser.parse_args()

    # ── Load and validate data ─────────────────────────────────────────────
    if not Path(args.data_path).exists():
        logger.error("No feedback pairs found at %s. Run feedback_harvester.py first.", args.data_path)
        sys.exit(1)

    pairs = _load_pairs(args.data_path)
    logger.info("Loaded %d DPO pairs from %s", len(pairs), args.data_path)

    if not pairs:
        logger.error("No pairs to train on.")
        sys.exit(1)

    # Validate format
    for i, p in enumerate(pairs):
        assert "prompt" in p, f"Pair {i} missing 'prompt'"
        assert "chosen" in p, f"Pair {i} missing 'chosen'"
        assert "rejected" in p, f"Pair {i} missing 'rejected'"
        assert isinstance(p["prompt"], list), f"Pair {i} 'prompt' must be list of messages"

    logger.info("Data validation passed.")

    if args.dry_run:
        logger.info("[DRY RUN] %d pairs validated. Exiting.", len(pairs))
        for p in pairs:
            src = p.get("source", "?")
            cat = p.get("category", "?")
            comp = p.get("competitor", "?")
            logger.info("  [%s] %s/%s: %s → %s",
                        src, cat, comp, p.get("wrong_verdict"), p.get("correct_verdict"))
        return

    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION not set. "
            "For RX 7800 XT (gfx1101) set HSA_OVERRIDE_GFX_VERSION=11.0.0."
        )

    # ── Imports (deferred to skip in dry-run) ──────────────────────────────
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model + SFT adapter ───────────────────────────────────────────────
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    device_map = {"": 0} if is_rocm else "auto"
    if is_rocm:
        logger.info("ROCm backend (HIP %s) — explicit device_map.", torch.version.hip)

    logger.info("Loading base model at bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Apply existing SFT adapter as the starting point
    if Path(args.sft_adapter).exists():
        logger.info("Loading SFT adapter from: %s", args.sft_adapter)
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()  # Merge SFT weights into base for DPO
        logger.info("SFT adapter merged into base model.")
    else:
        logger.warning("No SFT adapter found at %s — training from base model.", args.sft_adapter)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model ready: %.0fM parameters", n_params)

    # ── LoRA for DPO ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    logger.info("Building DPO dataset...")

    dpo_records = []
    for p in pairs:
        prompt_text = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True,
        )
        dpo_records.append({
            "prompt": prompt_text,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })

    dataset = Dataset.from_list(dpo_records)
    logger.info("Dataset: %d preference pairs", len(dataset))
    logger.info("DPO sequence budget: max_length=%d", args.max_length)
    logger.info("DPO hyperparams: lr=%.0e, beta=%.2f, epochs=%d, lora_r=8",
                args.lr, args.beta, args.epochs)

    # ── DPO config ─────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=args.adapter_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )
    logger.info(
        "DPO config: epochs=%d, lr=%.0e, beta=%.2f, batch=1×ga4",
        dpo_config.num_train_epochs,
        dpo_config.learning_rate,
        args.beta,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # Force LoRA disable-trick: no second model copy in VRAM
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    trainer.save_model(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    logger.info("DPO adapter saved to: %s", args.adapter_out)

    # ── Canary gate ────────────────────────────────────────────────────────
    # The canary gate loads the model in a subprocess.  The DPOTrainer
    # holds circular refs (model → optimizer → accelerator → model) that
    # Python's del/gc cannot reliably break while the process is alive,
    # so the training model stays resident on the GPU.  Loading a second
    # copy OOMs the 16 GB card.
    #
    # Fix: skip the in-process canary entirely.  The dashboard (routes.py)
    # and CLI callers should run drift_check.py as a *separate* invocation
    # after this process has exited and released all VRAM.
    if not args.skip_canary:
        logger.info(
            "Canary gate skipped automatically — GPU memory cannot be "
            "reliably freed while this process is alive.  Run the canary "
            "as a separate step:\n"
            "  python scripts/drift_check.py --adapter-path %s",
            args.adapter_out,
        )
    else:
        logger.info("Canary gate skipped (--skip-canary).")


if __name__ == "__main__":
    main()
