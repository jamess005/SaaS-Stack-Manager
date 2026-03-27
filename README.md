# saas-stack-manager

A fine-tuned 3B language model agent that manages the SaaS stack for a fictional UK consulting company. The agent monitors tool health over time — detecting when tools degrade, fall behind competitors, or stop earning their cost — and produces justified **Stay**, **Switch**, or **Hold** recommendations with cited evidence.

This is not a one-time purchasing tool. It's an ongoing management loop: every time a competitor ships an update, a pricing change lands, or a reliability issue surfaces, the agent evaluates whether the balance has shifted and whether action is warranted.

> **Portfolio project** — all data is synthetic and fictional. No real SaaS brands used.

---

## How It Works

```
market_inbox/release_note.md  →  agent.py  →  outputs/verdict.md
```

1. Drop a competitor market signal (JSON) into `market_inbox/`
2. The agent loads only the relevant category's business rules, current stack data, and competitor baseline
3. Python extracts financial variables directly from structured context (no model call)
4. ROI wrapper calculates migration cost and annualised net
5. Model generates structured verdict memo with Quote-to-Claim citations
6. Output written to `outputs/`, Hold register updated if applicable

## Verdict Classes

| Verdict | Meaning |
|---|---|
| `SWITCH` | Net signal weight crosses threshold, competitor is ready now |
| `STAY` | Current tool is adequate, no competitor clears the bar |
| `HOLD` | Switch may be warranted but conditions not right yet |

## Setup

### Requirements

- Python 3.11+
- ROCm 7.1 (Linux, AMD GPU)
- Ollama (for inbox generation)

### Install

```bash
git clone https://github.com/jamess005/SaaS-Stack-Manager
cd saas-stack-manager
pip install -r requirements.txt
```

### ROCm PyTorch

```bash
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### Run

```bash
python -m agent.agent market_inbox/your_trigger_file.md
```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for full directory layout and build phases.

## Hardware

Developed on: AMD RX 7800 XT (16GB VRAM), Ryzen 7 7800X3D, 32GB RAM, Linux, ROCm 7.1

## Status

- [x] Synthetic ecosystem generated (company profile, stack, competitors, business rules)
- [x] Inference pipeline (single-pass, Python ROI extraction, Quote-to-Claim citations)
- [x] Signal generation (325 synthetic market signals across 13 scenario types)
- [x] Reasoning traces (325 JSONL entries, 8B teacher model, 4-bit quantised)
- [ ] Fine-tuning (QLoRA via Unsloth, target: Qwen2.5-3B or Llama-3.2-3B)
