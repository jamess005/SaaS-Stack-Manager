"""
Central configuration — reads machine-specific paths from environment variables.
Copy .env.example to .env and set values for your local setup.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; export vars in your shell instead

# Base directory where model weights are stored locally
# Example: /home/user/models  or  /mnt/data/models
MODELS_DIR = Path(os.getenv("MODELS_DIR", "~/models")).expanduser()

# Student model — Qwen2.5-3B-Instruct (base for fine-tuning and inference)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "qwen2.5-3b-instruct")))

# Teacher model — Llama 3.1 8B Instruct (used only during training trace generation)
# HuggingFace snapshot paths may include a hash subdirectory — override via TEACHER_MODEL_PATH.
TEACHER_MODEL_PATH = Path(os.getenv("TEACHER_MODEL_PATH", str(MODELS_DIR / "llama-3.1-8b-instruct")))

# Smaller base model variant used in GRPO training experiments
SMALL_MODEL_PATH = Path(os.getenv("SMALL_MODEL_PATH", str(MODELS_DIR / "qwen2.5-1.5b-instruct")))
