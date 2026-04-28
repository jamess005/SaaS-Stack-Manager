# saas-stack-manager

A fine-tuned 3B language model agent that manages the SaaS stack for a fictional UK consulting company. The agent monitors tool health continuously — detecting when tools degrade, fall behind competitors, or stop earning their cost — and produces justified **Stay**, **Switch**, or **Hold** recommendations with cited evidence.

This is not a one-shot purchasing tool. It evaluates every incoming signal in context: incumbent weaknesses, competitor changes, ROI, compliance status, and hold conditions (beta software, acquisition freeze, contract lock-in). The output is a structured verdict memo written to the company's outputs log, with automatic plain-English summaries generated for the dashboard.

> **Portfolio project** — all data is synthetic and fictional. No real SaaS brands used.

---

## How It Works

```
market_inbox/signal.json  →  agent.py  →  outputs/verdict.md
                                       →  outputs/summaries.json
```

1. Drop a market signal (JSON) into `market_inbox/`
2. Agent loads scoped context for that category: business rules, current stack entry, competitor baseline
3. Python extracts financial variables from structured context (no model call)
4. ROI calculator computes migration cost and annualised net savings
5. Fine-tuned LoRA adapter generates a structured verdict memo (ANALYSIS + VERDICT format)
6. Output is validated; retried once if format check fails
7. Memo written to `outputs/`, Hold register updated if verdict is HOLD
8. Base Qwen2.5-3B (no adapter) generates a 2–3 sentence plain-English summary → `outputs/summaries.json` for the dashboard
9. Drift tracker logs the run; advisory raised after 10 live runs without a canary check

## Verdict Classes

| Verdict | Meaning |
|---|---|
| `SWITCH` | Net signal weight crosses threshold — competitor is ready, ROI met |
| `STAY` | Pull signals are vague or ROI not met — current tool holds |
| `HOLD` | Switch case has merit but a blocking condition prevents it (beta, pilot, contract) |

---

## Model Training

**Base model**: Qwen2.5-3B-Instruct (3B, locally cached, never fine-tuned for inference summaries)

**SFT adapter** (`training/checkpoints_sft_cot/`): QLoRA fine-tune on 1,275 chain-of-thought traces across 18 scenario types. Traces are structured ANALYSIS → VERDICT pairs: push signals, pull signals, compliance, ROI, hold conditions, reasoning, verdict. Training used a positional-reasoning system prompt to eliminate the base model's tendency to default SWITCH regardless of hold conditions.

**DPO adapter** (`training/checkpoints_dpo/`): stacked on the SFT adapter via accumulative training. Each run merges the previous DPO correction into the model before adding new preference pairs, so corrections compound without overwriting prior learning. Beta 0.3 keeps the model close to the SFT reference while correcting specific failure modes.

### Continuous improvement loop

```
live run → drift_log.jsonl
                ↓
   feedback_harvester.py
   (human feedback + canary failures + SFT augmentation)
                ↓
   feedback_pairs.jsonl
                ↓
   dpo_train.py   →   checkpoints_dpo/
                ↓
   drift_check.py  (canary gate: ≥75% accuracy required)
```

Human feedback is captured via the dashboard (thumbs up/down on any verdict memo). Drift canaries are known-answer signals run periodically by `scripts/drift_check.py`. The harvester only trains on signals the model is *currently* getting wrong — stale failures are excluded to avoid undoing already-learned behaviour. SFT trace augmentation (30 balanced samples per DPO run) prevents catastrophic forgetting.

---

## Results

The fine-tuned model achieves **100% accuracy (30/30)** across 30 distinct scenario types in held-out evaluation, balanced evenly across Stay / Switch / Hold verdict classes.

Scenario types tested include: compliance failures, shelfware detection, hold conditions (beta / pilot / acquisition / contract), roadmap signals, fluff-update disqualification, dual-signal ambiguous cases, and post-acquisition pause scenarios across all five tool categories.

Accuracy progression during training: 70% (SFT baseline) → 93% (positional reasoning + compliance gate) → 100% (final eval).

---

## Setup

### Requirements

- Python 3.12+
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

```bash
cp .env.example .env
# edit .env — set MODELS_DIR=/path/to/your/models
```

AMD GPU users — set in your shell before running:

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
# Dry-run (no GPU required — uses fixture responses)
python scripts/evaluate_model.py --dry-run

# Full evaluation against held-out signals
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/evaluate_model.py \
    --adapter-path training/checkpoints_sft_cot/
```

### Docker

The Docker image runs inference only. Adapter weights and outputs are mounted at runtime:

```bash
# Build
docker build -t saas-manager .

# Run a single signal (AMD GPU required for live inference)
docker compose run agent market_inbox/finance_ledgerflow.md

# Dry-run (no GPU)
AGENT_DRY_RUN=true docker compose run agent market_inbox/finance_ledgerflow.md
```

### Dashboard

```bash
python -m dashboard
# → http://localhost:5000
```

Displays verdict history, confidence trends, drift monitoring, and a review queue for human feedback capture.

---

## Project Structure

```
agent/          # Inference pipeline (context loader, model runner, ROI calculator, validator)
data/           # Synthetic ecosystem (company profile, stack, competitors, business rules)
training/       # Signal generation, CoT trace generation, SFT + DPO fine-tuning scripts
scripts/        # Evaluation harness, drift checker, batch runner
tests/          # 148 unit + integration tests for the full pipeline
fixtures/       # Dry-run fixture responses and held-out eval signals
dashboard/      # Flask monitoring server — drift tracker, verdict review queue, confidence charts
```

---

## Hardware

Developed on: AMD RX 7800 XT (16GB VRAM), Ryzen 7 7800X3D, 32GB RAM, Linux, ROCm 7.2

---

## Status

- [x] Synthetic ecosystem (company profile, stack, competitors, business rules)
- [x] Inference pipeline (single-pass, Python ROI extraction, structured citations)
- [x] Signal generation (1,275 CoT training traces across 18 scenario types)
- [x] SFT fine-tuning (QLoRA, Qwen2.5-3B-Instruct, positional reasoning fix)
- [x] DPO fine-tuning (accumulative preference learning from human feedback + canary failures)
- [x] Flask dashboard (drift monitoring, confidence trends, verdict review queue)
- [x] Evaluation: 30/30 (100%) across 30 held-out scenarios, balanced Stay / Switch / Hold
