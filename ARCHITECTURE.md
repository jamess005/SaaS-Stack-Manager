# Architecture & Build Roadmap: saas-stack-manager

---

## Repository Setup

**Repo name:** `saas-stack-manager`
**Visibility:** Public
**Description:** *Fine-tuned 3B LLM agent that manages a company's SaaS stack — monitoring tool health, detecting when tools stop earning their cost, and recommending Stay/Switch/Hold decisions using push/pull signal reasoning.*

---

## Directory Structure (Full)

```
saas-stack-manager/
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Tests + lint on push/PR
│       └── lint.yml             # Pre-commit lint check
│
├── data/                        # ✅ GENERATED — drop ecosystem.zip contents here
│   ├── company_profile.md
│   ├── current_stack.json
│   ├── usage_metrics.json
│   ├── business_rules/
│   │   ├── global.md
│   │   ├── crm.md
│   │   ├── hr.md
│   │   ├── finance.md
│   │   ├── project_mgmt.md
│   │   └── analytics.md
│   └── competitors/
│       ├── crm/
│       ├── hr/
│       ├── finance/
│       ├── project_mgmt/
│       └── analytics/
│
├── market_inbox/                # Drop .md trigger files here at runtime
│   └── .gitkeep
│
├── outputs/                     # Verdict memos written here at runtime
│   └── .gitkeep
│
├── hold_register.json           # Initialise as empty array []
│
├── agent/
│   ├── __init__.py
│   ├── agent.py                 # CLI entrypoint — python -m agent.agent <inbox_file>
│   ├── context_loader.py        # Loads scoped JSON + MD files for a given category
│   ├── roi_calculator.py        # Pure Python ROI calculation — no model involvement
│   ├── output_validator.py      # Checks verdict fields + citation presence
│   └── model_runner.py          # Handles Pass 1 (extraction) and Pass 2 (verdict)
│
├── training/
│   ├── traces/                  # Hand-written JSONL CoT examples — gold standard
│   │   └── .gitkeep
│   ├── generated/               # Llama 8B-generated market signal delta .md files
│   │   └── .gitkeep
│   ├── generate_inbox.py        # Llama 8B (4-bit NF4) delta trigger generator
│   ├── generate_traces.py       # Reasoning trace generator
│   └── fine_tune.py             # QLoRA training script via Unsloth
│
├── scripts/
│   └── validate_ecosystem.py    # Sanity-checks all JSON/MD files in data/
│
├── tests/
│   ├── __init__.py
│   ├── test_roi.py              # Unit tests for roi_calculator
│   ├── test_validator.py        # Unit tests for output_validator
│   ├── test_context_loader.py   # Unit tests for context_loader
│   └── test_agent_integration.py # End-to-end with fixture inbox files
│
├── fixtures/                    # Static test data for CI — never changes
│   ├── inbox_switch.md
│   ├── inbox_stay.md
│   ├── inbox_hold.md
│   └── expected_outputs/
│       ├── switch_verdict.json
│       ├── stay_verdict.json
│       └── hold_verdict.json
│
├── requirements.txt             # Runtime deps
├── requirements-dev.txt         # Dev/test deps
├── pyproject.toml               # Black, isort, pytest config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .gitignore
└── README.md
```

---

## Build Phases

### Phase 0 — Repository Bootstrap *(do this now)*
- [ ] Create repo `saas-stack-manager` on GitHub
- [ ] Clone locally
- [ ] Copy ecosystem.zip contents into `data/`
- [ ] Add boilerplate files (this repo provides them)
- [ ] Initial commit + push

### Phase 1 — Core Pipeline (no fine-tuning)
- [ ] `context_loader.py` — load and assemble scoped context from data files
- [ ] `roi_calculator.py` — pure function, fully unit-tested
- [ ] `output_validator.py` — parse and validate verdict memo format
- [ ] `model_runner.py` — load quantised 3B model, run Pass 1 + Pass 2
- [ ] `agent.py` — orchestrate full loop, write outputs, update hold_register
- [ ] System prompt + 3–5 few-shot examples in context
- [ ] Manual test on 5 fixture scenarios (SWITCH, STAY, HOLD, fluff, compliance fail)
- [ ] Identify and document failure modes

### Phase 2 — Inbox Generation
- [x] `training/generate_inbox.py` — Local Llama 8B (4-bit NF4) generation script
  - Produces structured JSON market signal deltas — ONLY what has changed
  - Push signals (current tool degradation) and pull signals (competitor improvement)
  - JSON schema: category, competitor, current_tool, date, competitor_changes[], current_tool_status[], pricing_delta, compliance_changes, notes[]
  - Validated on output: JSON parsing + required field checks + type enforcement
  - Parameterised by: category, scenario type, competitor
  - 11 scenario types × 25 competitors = 275 .json files
  - Scenarios: pull_dominant, push_dominant, both_signals, competitor_nearly_ready, price_hike_only, shelfware_case, fluff_update, negative_signal_buried, hard_compliance_failure, hold_resolved, irrelevant_change
- [ ] Review generated files — reject shallow/repetitive ones
- [ ] Manually write 80–100 JSONL reasoning traces in `training/traces/`

### Phase 3 — Fine-Tuning (conditional on Phase 1 failures)
- [ ] Set up Unsloth + QLoRA on ROCm 7.1
- [ ] `fine_tune.py` — training script
- [ ] Run training on 80–100 JSONL traces (~1–3hr on 7800 XT)
- [ ] Evaluate on held-out test set
- [ ] Compare Phase 1 baseline vs fine-tuned — document delta

### Phase 4 — Polish + Portfolio
- [ ] README with architecture diagram, example verdict output, setup instructions
- [ ] Benchmark table (Phase 1 prompt-only vs Phase 2 fine-tuned)
- [ ] Tidy CI/CD, ensure all tests green

---

## Hardware Notes (ROCm)

```
GPU:  AMD RX 7800 XT — 16GB VRAM
CPU:  Ryzen 7 7800X3D
RAM:  32GB
OS:   Linux
ROCm: 7.2
```

### ROCm Package Strategy

Using **ROCm 7.2** with PyTorch 2.11.0. bitsandbytes 4-bit NF4 quantization works
on ROCm with explicit device_map={"":0} (~5 GB VRAM for 8B model). Key packages:

```
torch==2.11.0+rocm7.2       # via PyTorch ROCm wheels
torchvision==0.26.0+rocm7.2
torchaudio==2.11.0+rocm7.2
unsloth[rocm]               # QLoRA fine-tuning — has ROCm-specific install path
transformers>=4.45.0
peft>=0.13.0
datasets>=2.20.0
accelerate>=0.34.0
bitsandbytes>=0.44.0        # Native ROCm support from 0.44.0+
```

### Environment Variables (add to .env, never commit)

```bash
ROCR_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.0.0    # RX 7800 XT is gfx1101 — may need this
HIP_VISIBLE_DEVICES=0
```

---

## CI/CD Pipeline Overview

### `ci.yml` — triggers on push and PR to main

```
1. Checkout code
2. Set up Python 3.11
3. Install requirements-dev.txt
4. Run pre-commit hooks (black, isort, flake8)
5. Run pytest tests/ (CPU-only, no model loaded)
   - test_roi.py       — pure Python, always passes in CI
   - test_validator.py — pure Python, always passes in CI
   - test_context_loader.py — reads from fixtures/, always passes
   - test_agent_integration.py — uses mock model runner in CI
6. Upload coverage report
```

**Note:** CI never loads the actual model. `model_runner.py` has a `--dry-run` flag that returns a fixture response, allowing full pipeline integration tests without GPU.

### `lint.yml` — triggers on push to any branch

```
1. Checkout
2. Python 3.11
3. pip install pre-commit
4. pre-commit run --all-files
```

---

## Key Design Decisions (Recorded)

| Decision | Rationale |
|---|---|
| 3B not 7B+ | Scoped context loading keeps each call small — capacity ceiling not reached |
| Two-pass inference | Pass 1 extracts variables cleanly; Python handles maths; Pass 2 produces verdict with known ROI |
| Python ROI wrapper | 3B models make arithmetic errors — never let the model calculate |
| Quote-to-Claim rule | Prevents hallucinated features — every assertion needs a source sentence |
| Scoped context loading | Never load full knowledge base — only category-relevant files per call |
| Phase 1 before fine-tuning | Earn the fine-tuning with evidence; failure cases become training data |
| Fictional SaaS ecosystem | Avoids scraping cost and prevents parametric memory bleed from real brand names |
| ROCm 7.2 with explicit device_map | device_map="auto" (accelerate dispatch) triggers HIP kernel failures on ROCm 7.x. Using device_map={"":0} with 4-bit NF4 quantization (~5 GB VRAM). |
| Hand-written traces only | Generated traces may have subtle reasoning errors — gold standard must be verified |
