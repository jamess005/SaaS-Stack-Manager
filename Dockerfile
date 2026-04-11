# SaaS Stack Manager — inference image
# Requires AMD GPU with ROCm 7.x on the host for live model inference.
# Dry-run mode (AGENT_DRY_RUN=true) works without GPU for pipeline testing.

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ROCm PyTorch wheel — must be installed before other ML deps to avoid conflicts.
# This makes the image ~3 GB; it is required for GPU inference.
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm7.2

# Remaining inference dependencies
RUN pip install --no-cache-dir \
    python-dotenv>=1.0.0 \
    transformers>=4.45.0 \
    peft>=0.13.0 \
    accelerate>=0.34.0 \
    bitsandbytes>=0.44.0 \
    huggingface-hub>=0.24.0 \
    rich>=13.7.0

# Application code — training/ included for generate_signals imports
COPY agent/       agent/
COPY data/        data/
COPY fixtures/    fixtures/
COPY training/    training/
COPY scripts/     scripts/
COPY config.py    config.py

# Runtime mount points (populated via docker-compose volumes)
RUN mkdir -p market_inbox outputs training/checkpoints_sft_cot

# Model weights are mounted at /models at runtime — see docker-compose.yml
# Adapter weights are mounted at training/checkpoints_sft_cot/ at runtime
ENV MODELS_DIR=/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "agent.agent"]
