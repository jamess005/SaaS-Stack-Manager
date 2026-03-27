# Project Blueprint: SaaS Stack Manager
> Last updated: March 2025 — reflects full ecosystem generation and architecture decisions

---

## 1. Project Overview

A fine-tuned small language model (3B parameters) acting as an ongoing **SaaS Stack Manager** for a fictional UK consulting business. Rather than handling one-off purchasing decisions, the agent continuously monitors the health and value of the company's existing tool stack — detecting when tools are degrading, falling behind competitors, or no longer earning their cost — and recommends action when the balance tips.

The agent does not summarise data. It reasons across competing push and pull signals, weighs migration friction against potential gains, and produces an evidence-backed recommendation memo with cited evidence for every claim.

**Scope boundary:** This project builds the reasoning engine only. Web scraping, live data ingestion, and real-time SaaS monitoring are explicitly out of scope. All data is synthetic and fictional.

---

## 2. Core Concept: Push and Pull Factors

The decision logic mirrors a real-world question: *is it time to replace the car, or keep maintaining it?*

**Pull Factors** — competition becoming more attractive:
- Competitor adds a must-have feature the current tool lacks
- Competitor reduces pricing or changes to a more favourable model
- Competitor resolves a known reliability or compliance gap
- Competitor ecosystem integrations improve

**Push Factors** — current tool becoming less viable:
- Persistent unfixed issues eroding operational efficiency
- Price increases without feature improvements
- EOL signals, deprecation notices, declining support quality
- Staff training burden growing on an ageing system
- Tech debt accumulating; tool failing to modernise

The agent holds both simultaneously and determines whether their combined weight crosses the action threshold.

---

## 3. The Synthetic World: Meridian Consulting Group

All data is fictional. The company is **Meridian Consulting Group** — a UK B2B management consultancy, 150 staff, London HQ + Bristol office, EU clients. Chosen to justify: multi-currency requirements, GDPR/UK GDPR compliance, SOC2 mandates, and realistic consulting-shaped SaaS needs.

### Current Stack (5 categories)

| Category | Tool | Monthly Cost | Key Known Issues |
|---|---|---|---|
| CRM | NexusCRM | £510/mo | No LinkedIn enrichment, 5-step email sequence cap, API rate limits breaking analytics sync |
| HR | PeoplePulse | £385/mo | Manual GDPR erasure (compliance risk), broken PayAxis middleware, no onboarding templates |
| Finance | VaultLedger | £420/mo | No multi-currency (blocking EU invoicing), overnight bank feed only, no IFRS 15 |
| Project Mgmt | TaskBridge | £290/mo | No time-tracking (patched by ChronoLog at £95/mo extra), no client portal, 52 shelfware seats |
| Analytics | InsightDeck | £340/mo | No data lineage (ISO audit risk), no mobile, missed ML roadmap commitments x2 |

### Competitors (3 per category)

Each category has one strong contender, one premium option, and one that fails a hard compliance check — ensuring the agent encounters all verdict types.

| Category | Strong | Premium | Hard Block |
|---|---|---|---|
| CRM | PipelineIQ | CloserHub | VelocityCRM (no SSO, no audit log) |
| HR | WorkForge | TeamLedger HR | HRNest (no SOC2) |
| Finance | LedgerFlow | NovaPay | ClearBooks Pro (no SOC2) |
| Project Mgmt | FlowBoard | SprintDesk | TeamSync Projects (no time-tracking, no client portal) |
| Analytics | DataLens | ClearView Analytics | PulseMetrics (product analytics — wrong category fit) |

---

## 4. File Structure

```
saas-stack-manager/
├── data/
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
├── market_inbox/            # Drop trigger .md files here
├── outputs/                 # Generated verdict memos written here
├── hold_register.json       # Active Hold verdicts
├── agent/
│   ├── agent.py             # Orchestration entrypoint
│   ├── context_loader.py    # Scoped file loader
│   ├── roi_calculator.py    # Pure Python ROI wrapper
│   ├── output_validator.py  # Format + citation checker
│   └── model_runner.py      # Inference calls (Pass 1 + Pass 2)
├── training/
│   ├── traces/              # Hand-written CoT JSONL training examples
│   ├── generated/           # Llama 8B-generated market signal delta .md files
│   ├── generate_inbox.py    # Llama 8B (4-bit NF4) delta trigger generator
│   ├── generate_traces.py   # Reasoning trace generator
│   └── fine_tune.py         # QLoRA training script
├── scripts/
│   └── validate_ecosystem.py    # Sanity-checks all JSON/MD files in data/
├── tests/
│   ├── test_roi.py
│   ├── test_validator.py
│   └── test_agent_integration.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── lint.yml
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

---

## 5. Verdict Classes

| Verdict | Meaning |
|---|---|
| **SWITCH** | Net signal weight crosses threshold, best available competitor is ready now |
| **STAY** | Current tool is adequate, no competitor clears the bar |
| **HOLD** | Switch may be warranted but conditions not right yet — reassess trigger and date defined |

### Hold Register

When a Hold verdict is issued, `agent.py` appends to `hold_register.json`:

```json
{
  "category": "finance",
  "current_tool": "VaultLedger",
  "competitor": "NovaPay",
  "hold_reason": "NovaPay lacks IFRS 15 support. Required per finance.md.",
  "reassess_condition": "NovaPay ships IFRS 15 revenue recognition module",
  "issued_date": "2024-11-15",
  "review_by": "2025-02-15"
}
```

**Scope note:** Hold register is informational in v1. Operator drops a new trigger file to trigger reassessment. Automated scheduling is v2.

**Scope note:** Each call compares one competitor against one current tool. Multi-tool consolidation scenarios are v2.

---

## 6. The Reasoning Loop

For every trigger file in `market_inbox/`, the agent runs a 7-step chain:

| Step | Name | What Happens |
|---|---|---|
| 1 | Ingestion | Parse release note — extract feature changes, pricing, deprecations, compliance updates, buried negatives |
| 2 | Context Mapping | Identify category and current tool. Load scoped files only. |
| 3 | Push Audit | Review current tool's known issues and usage metrics. Shelfware? Unresolved pain points? |
| 4 | Pull Audit | Compare competitor update against business rules. Gaps addressed? New negatives introduced? |
| 5 | Variable Extraction | Extract financial variables into JSON payload. Model does not calculate. |
| 6 | ROI Calculation | Python wrapper computes migration cost and annualised net. Returns result to model. |
| 7 | Verdict + Memo | Structured memo with signal weighting and Quote-to-Claim citations for every asserted feature. |

---

## 7. Inference Pipeline

```
market_inbox/ (.md file dropped)
       │
       ▼
agent.py — extracts category + competitor from filename convention
       │
       ▼
context_loader.py — loads scoped files only:
  current_stack.json (category entry)
  usage_metrics.json (current tool entry)
  business_rules/{category}.md
  competitors/{category}/{competitor}.json
       │
       ▼
model_runner.py — Pass 1
  Reasoning chain + variable extraction
  Outputs JSON: {current_monthly_cost, competitor_monthly_cost,
                 migration_hours, staff_hourly_rate, annual_saving, roi_threshold}
       │
       ▼
roi_calculator.py — pure Python
  migration_cost = hours × rate
  annual_net = saving − (migration_cost / 3)   # amortised 3yr
  roi_met = annual_net >= threshold
  Returns: {migration_cost, annual_net, roi_threshold_met}
       │
       ▼
model_runner.py — Pass 2
  ROI result appended to context
  Generates final verdict memo
       │
       ▼
output_validator.py
  Checks: all fields present, at least one Quote-to-Claim citation
  On fail: retry once → flag for manual review
       │
       ▼
outputs/{date}-{category}-{competitor}.md
hold_register.json updated if HOLD
```

**Triggering (v1):** Manual. Operator drops file into `market_inbox/` and runs `agent.py`. Watchdog automation is a clean v2 extension.

---

## 8. Output Format

```
CATEGORY: Finance
CURRENT TOOL: VaultLedger (£420/mo)
COMPETITOR: LedgerFlow (£380/mo)
DATE: 2024-11-20

PUSH SIGNALS:
  - Multi-currency gap blocking EU client invoicing [HIGH — must-have per finance.md]
  - IFRS 15 workaround in use — audit risk [HIGH — compliance requirement]
  - Overnight bank feed only — no intraday visibility [MEDIUM]

PULL SIGNALS:
  - "multi-currency invoicing (40+ currencies including EUR native)" [HIGH — resolves #1 push]
  - "IFRS 15 revenue recognition module" [HIGH — resolves #2 push]
  - "real-time bank feeds (Barclays)" [MEDIUM — resolves #3 push]
  - "audit trail export: PDF with digital signature" [HIGH — auditor requirement met]
  - Price reduction: £420 → £380/mo [POSITIVE]

FINANCIAL ANALYSIS:
  Migration cost: 15hrs × £48 = £720 one-time
  Annual saving: £480
  Amortised migration: £240/yr over 3 years
  Annual net: £240
  ROI threshold (£1,200/yr): NOT MET on direct saving alone
  Operational gain: Multi-currency and IFRS 15 compliance unblock EU billing
    and remove audit risk — qualitative value exceeds threshold
  ROI threshold met: YES (operational + direct combined)

VERDICT: SWITCH

EVIDENCE:
  "multi-currency invoicing (40+ currencies including EUR native)"
  "IFRS 15 revenue recognition module"
  "real-time bank feeds (Barclays, HSBC, Lloyds, Starling, Monzo)"
  "audit trail export: PDF with digital signature"
```

---

## 9. Python ROI Wrapper

```python
def calculate_roi(payload: dict) -> dict:
    migration_cost = payload["migration_hours"] * payload["staff_hourly_rate"]
    amortised_migration = migration_cost / 3
    annual_net = payload["annual_saving"] - amortised_migration
    roi_met = annual_net >= payload["roi_threshold"]

    return {
        "migration_cost_one_time": round(migration_cost, 2),
        "annual_net_gbp": round(annual_net, 2),
        "roi_threshold_gbp": payload["roi_threshold"],
        "roi_threshold_met": roi_met,
        "note": "Excludes operational gains. Flag qualitatively if applicable."
    }
```

Some switches have financial impact that cannot be expressed as direct saving (e.g. unblocking EU billing). The model flags these qualitatively rather than fabricating a number.

---

## 10. Fine-Tuning Strategy

### What Fine-Tuning Is Doing

The base 3B model already knows English, pricing concepts, and cost-benefit reasoning. Fine-tuning conditions three specific behaviours:

1. **Reasoning chain adherence** — 7-step loop executed in order, reliably
2. **Quote-to-Claim discipline** — every asserted feature must cite the exact source sentence
3. **Output format consistency** — structured memo produced every time

### Training Data Format (JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are a SaaS stack management agent for Meridian Consulting Group..."},
    {"role": "user", "content": "[scoped context + inbox trigger + ROI result from Pass 1]"},
    {"role": "assistant", "content": "[full 7-step reasoning chain + verdict memo]"}
  ]
}
```

### Scenario Taxonomy (training coverage required)

| Scenario | Target Verdict | Examples Needed |
|---|---|---|
| Pull-dominant | SWITCH | 8–12 |
| Push-dominant | SWITCH to best available | 8–12 |
| Both signals active | STAY or SWITCH (magnitude-dependent) | 8–12 |
| Competitor nearly ready | HOLD | 8–12 |
| Price hike only | ROI-dependent | 8–12 |
| Shelfware case | SWITCH | 6–8 |
| Fluff update | STAY | 8–12 |
| Negative signal buried | STAY | 8–12 |
| Hard compliance failure | STAY (regardless) | 8–12 |
| Hold resolved | SWITCH | 6–8 |

**Total target: ~80–100 examples across all categories.**

### Data Generation (Zero Cost)

| Tier | What | How |
|---|---|---|
| 1 | Ecosystem files (JSON, rules) | ✅ Already generated — Python script |
| 2 | Market inbox trigger files | Local Llama 8B (4-bit NF4) — market signal deltas (push/pull changes only, no baseline restating) |
| 3 | Reasoning traces (labels) | Hand-written — non-negotiable, these are your gold standard |

### Phased Approach

**Phase 1 — Prompt Engineering Baseline**
Full pipeline with strong system prompt + 3–5 few-shot examples in context. Test held-out scenarios. Identify repeatable failure modes.

**Phase 2 — Fine-Tuning (Conditional)**
Fine-tune only if Phase 1 shows consistent failures. Failure cases become highest-priority training examples. Toolchain: Unsloth + QLoRA on ROCm.

---

## 11. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated features | Quote-to-Claim enforced in training — no citation = invalid verdict |
| Arithmetic errors | Python wrapper handles all calculation — model extracts variables only |
| ROCm training issues | bitsandbytes 4-bit NF4 works on ROCm 7.2 with device_map={"":0} (~5 GB VRAM for 8B). accelerate's device_map="auto" crashes on ROCm — always use explicit. |
| Scenario undercoverage | Follow taxonomy in Section 10 — every type needs examples |
| Output format drift | output_validator.py rejects non-conforming outputs and retries |
| Hold verdict underuse | Explicit Hold examples in training data — without them model defaults to binary |
| Knowledge base staleness | Operator updates competitor baselines manually in v1 |

---

## 12. Target Model

**Candidates: Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct**

Both fit in 16GB VRAM at 4-bit quantisation. Both supported by Unsloth for QLoRA on ROCm.

- **Qwen2.5-3B** — slight edge on structured JSON output consistency
- **Llama-3.2-3B** — broader community tooling, more ROCm fine-tuning guides available

Run Phase 1 baseline with both. Whichever produces more consistent output format without fine-tuning is the better candidate to fine-tune.

---

## 13. Status

| Component | Status |
|---|---|
| Company profile + global rules | ✅ Complete |
| current_stack.json (5 categories) | ✅ Complete |
| usage_metrics.json | ✅ Complete |
| Business rules (6 files) | ✅ Complete |
| Competitor baselines (25 files) | ✅ Complete |
| market_inbox trigger files | 🔄 Generating — 250 delta files (25 competitors × 10 scenarios) |
| Reasoning traces (JSONL) | ⏳ Pending Phase 1 |
| agent.py + pipeline | ⏳ Pending |
| Fine-tuning run | ⏳ Pending Phase 2 |
