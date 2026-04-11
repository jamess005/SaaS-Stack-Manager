# saas-stack-manager

A fine-tuned 3B language model agent that manages the SaaS stack for a fictional UK consulting company. The agent monitors tool health over time — detecting when tools degrade, fall behind competitors, or stop earning their cost — and produces justified **Stay**, **Switch**, or **Hold** recommendations with cited evidence.

This is not a one-time purchasing tool. It's an ongoing management loop: every time a competitor ships an update, a pricing change lands, or a reliability issue surfaces, the agent evaluates whether the balance has shifted and whether action is warranted.

> **Portfolio project** — all data is synthetic and fictional. No real SaaS brands used.

---

## How It Works

```
market_inbox/signal.json  →  agent.py  →  outputs/verdict.md
```

1. Drop a competitor market signal (JSON) into `market_inbox/`
2. The agent loads only the relevant category's business rules, current stack data, and competitor baseline
3. Python extracts financial variables directly from structured context (no model call)
4. ROI wrapper calculates migration cost and annualised net
5. Fine-tuned 3B model generates structured verdict memo with cited evidence
6. Output written to `outputs/`, Hold register updated if applicable

## Verdict Classes

| Verdict | Meaning |
|---|---|
| `SWITCH` | Net signal weight crosses threshold, competitor is ready now |
| `STAY` | Current tool is adequate, no competitor clears the bar |
| `HOLD` | Switch may be warranted but conditions not right yet |

## Results

The fine-tuned model achieves **100% accuracy (30/30)** across 18 distinct scenario types in held-out evaluation, and **4/4** on unseen competitor fixtures not present in training.

Scenario types tested include: compliance failures, shelfware detection, competitor hold conditions, roadmap signals, pilot-in-progress, contract renewal gates, and dual-signal ambiguous cases.

## Setup

### Requirements

- Python 3.11+
- ROCm 7.x (Linux, AMD GPU) — or adapt `device_map` for CUDA
- Model weights downloaded locally (Qwen2.5-3B-Instruct)

### Install

```bash
git clone https://github.com/jamess005/SaaS-Stack-Manager
cd saas-stack-manager
pip install -r requirements.txt
```

### ROCm PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

### Configuration

Copy `.env.example` to `.env` and set `MODELS_DIR` to wherever your model weights live:

```bash
cp .env.example .env
# edit .env — set MODELS_DIR=/path/to/your/models
```

AMD GPU users: set the following in your shell before running:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7800 XT (gfx1101)
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

### Run

```bash
python -m agent.agent market_inbox/your_signal.json
```

### Evaluate

```bash
# Dry-run (no GPU required)
python scripts/evaluate_model.py --dry-run

# Full evaluation against held-out signals
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/evaluate_model.py \
    --adapter-path training/checkpoints_sft_cot/
```

## Project Structure

```
agent/          # Inference pipeline (context loader, model runner, ROI calculator, validator)
data/           # Synthetic ecosystem (company profile, stack, competitors, business rules)
training/       # Signal generation, CoT trace generation, SFT fine-tuning scripts
scripts/        # Evaluation and data utilities
tests/          # 100 unit + integration tests for the production pipeline
fixtures/       # Dry-run fixture responses
```

## Hardware

Developed on: AMD RX 7800 XT (16GB VRAM), Ryzen 7 7800X3D, 32GB RAM, Linux, ROCm 7.1

## Status

- [x] Synthetic ecosystem (company profile, stack, competitors, business rules)
- [x] Inference pipeline (single-pass, Python ROI extraction, structured citations)
- [x] Signal generation (1,275 CoT training traces across 18 scenario types)
- [x] SFT fine-tuning (QLoRA, Qwen2.5-3B-Instruct, positional reasoning fix)
- [x] Evaluation: 30/30 (100%) on 18 clear-cut scenarios, 4/4 on unseen competitors
