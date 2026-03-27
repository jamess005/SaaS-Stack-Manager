"""
Phase 3 — QLoRA fine-tuning script via Unsloth on ROCm.

Target model: Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct (determined after Phase 1 baseline).
Hardware: AMD RX 7800 XT — 16GB VRAM, ROCm 7.1.

Training data: hand-written JSONL reasoning traces in training/traces/
Each trace has the format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage (Phase 3 only):
    python training/fine_tune.py --model qwen2.5-3b --traces-dir training/traces/

Environment:
    ROCR_VISIBLE_DEVICES=0
    HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7800 XT is gfx1101
    HIP_VISIBLE_DEVICES=0
"""

import argparse


def load_traces(traces_dir: str) -> list[dict]:
    """
    Load JSONL training traces from traces_dir.
    Each line must have a 'messages' key with system/user/assistant turns.

    Phase 3 implementation — not yet written.
    """
    raise NotImplementedError(
        "Phase 3 fine-tuning not yet implemented. "
        "Complete Phase 1 evaluation and Phase 2 trace generation first."
    )


def train(model_name: str, traces_dir: str, output_dir: str) -> None:
    """
    Fine-tune the target model using QLoRA via Unsloth on ROCm.

    Phase 3 implementation — not yet written.
    """
    raise NotImplementedError(
        "Phase 3 fine-tuning not yet implemented. "
        "Complete Phase 1 evaluation and Phase 2 trace generation first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SaaS Stack Manager model with QLoRA on ROCm (Phase 3)."
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-3b",
        choices=["qwen2.5-3b", "llama-3.2-3b"],
        help="Base model to fine-tune (determined by Phase 1 baseline comparison).",
    )
    parser.add_argument("--traces-dir", default="training/traces/", help="Directory containing JSONL trace files.")
    parser.add_argument("--output-dir", default="training/checkpoints/", help="Output directory for LoRA weights.")
    args = parser.parse_args()
    train(args.model, args.traces_dir, args.output_dir)


if __name__ == "__main__":
    main()
