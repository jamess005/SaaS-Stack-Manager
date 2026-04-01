# Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ~1.1 GB of stale training artefacts, dead one-off scripts, and intermediate checkpoints; consolidate the two trace directories into one canonical location; prevent recurrence with .gitignore additions.

**Architecture:** Pure filesystem cleanup — no logic changes to any source file. The active model adapter (root files in `training/checkpoints/`) and the final checkpoint (`checkpoint-489/`) are preserved. Fresh traces (`training/traces_fresh/`) replace the stale `training/traces/` content, then `traces_fresh/` is removed. All dead fix/migration scripts are deleted.

**Tech Stack:** bash (rm, cp, git rm), Python (pre-deletion integrity check), git (commit cleanup)

---

### Task 1: Verify fresh traces are sound before touching anything

**Files:**
- Read: `training/traces_fresh/traces.jsonl`
- Read: `training/traces/traces.jsonl`

- [ ] **Step 1: Confirm fresh trace count and ANALYSIS format**

```bash
python -c "
import json
fresh = sum(1 for l in open('training/traces_fresh/traces.jsonl') if l.strip())
old   = sum(1 for l in open('training/traces/traces.jsonl') if l.strip())
ok, bad = 0, 0
for l in open('training/traces_fresh/traces.jsonl'):
    t = json.loads(l)
    if t['metadata']['step'] == 'verdict':
        asst = t['messages'][-1]['content']
        if asst.startswith('ANALYSIS:') and 'VERDICT:' in asst:
            ok += 1
        else:
            bad += 1
print(f'Fresh: {fresh} traces  |  Old: {old} traces')
print(f'Verdict traces — OK: {ok}, BAD: {bad}')
"
```

Expected output:
```
Fresh: 1300 traces  |  Old: 1300 traces
Verdict traces — OK: 325, BAD: 0
```

Do not proceed to Step 2 if `BAD > 0`.

- [ ] **Step 2: Replace old traces with fresh ones**

```bash
cp training/traces_fresh/traces.jsonl      training/traces/traces.jsonl
cp training/traces_fresh/traces_report.json training/traces/traces_report.json
```

- [ ] **Step 3: Delete stale backup and log files from training/traces/**

```bash
rm -f training/traces/traces_backup.jsonl
rm -f training/traces/traces.jsonl.bak
rm -f training/traces/traces.jsonl.pre_restructure
rm -f training/traces/traces.jsonl.pre_split
rm -f training/traces/fix_log.json
rm -f training/traces/fix_log_pass2.json
rm -f training/traces/audit_report.json
```

- [ ] **Step 4: Delete traces_fresh/ directory**

```bash
rm -rf training/traces_fresh/
```

- [ ] **Step 5: Verify training/traces/ is clean**

```bash
ls training/traces/
```

Expected: `traces.jsonl  traces_report.json` — nothing else.

---

### Task 2: Delete one-off fix and migration scripts

These scripts (`fix_traces`, `fix_traces_pass2`, `restructure_traces`, `split_traces`, `audit_traces`) were created during earlier trace format migrations and are not part of the active workflow. `regenerate_traces.py` targeted the old Pass 2 full-memo format and is superseded by `regenerate_verdict_traces.py`.

**Files:**
- Delete: `scripts/fix_traces.py` (untracked)
- Delete: `scripts/fix_traces_pass2.py` (untracked)
- Delete: `scripts/restructure_traces.py` (untracked)
- Delete: `scripts/split_traces.py` (untracked)
- Delete: `scripts/audit_traces.py` (untracked)
- Delete: `scripts/regenerate_traces.py` (tracked — requires `git rm`)

- [ ] **Step 1: Delete untracked fix scripts**

```bash
rm scripts/fix_traces.py
rm scripts/fix_traces_pass2.py
rm scripts/restructure_traces.py
rm scripts/split_traces.py
rm scripts/audit_traces.py
```

- [ ] **Step 2: Remove regenerate_traces.py from git and disk**

```bash
git rm scripts/regenerate_traces.py
```

Expected: `rm 'scripts/regenerate_traces.py'`

- [ ] **Step 3: Verify scripts/ contains only active files**

```bash
ls scripts/
```

Expected: `evaluate_model.py  regenerate_verdict_traces.py  validate_ecosystem.py`

---

### Task 3: Prune intermediate training checkpoints (~900 MB)

**Keep:** root adapter files in `training/checkpoints/` (the deployed model) + `checkpoint-489/` (final checkpoint from last run).

**Delete:** all earlier checkpoint-N directories, plus `qwen2.5-3b-lora/` (a stale previous run, 481 MB).

- [ ] **Step 1: Delete early large checkpoints (contain full optimizer state, ~183 MB each)**

```bash
rm -rf training/checkpoints/checkpoint-41/
rm -rf training/checkpoints/checkpoint-82/
rm -rf training/checkpoints/checkpoint-123/
```

- [ ] **Step 2: Delete mid-training checkpoints (~33 MB each)**

```bash
rm -rf training/checkpoints/checkpoint-162/
rm -rf training/checkpoints/checkpoint-163/
rm -rf training/checkpoints/checkpoint-324/
rm -rf training/checkpoints/checkpoint-326/
rm -rf training/checkpoints/checkpoint-486/
```

- [ ] **Step 3: Delete stale qwen2.5-3b-lora run directory (481 MB)**

```bash
rm -rf training/checkpoints/qwen2.5-3b-lora/
```

- [ ] **Step 4: Confirm what remains**

```bash
ls training/checkpoints/
du -sh training/checkpoints/
```

Expected listing: `adapter_config.json  adapter_model.safetensors  chat_template.jinja  checkpoint-489/  README.md  tokenizer_config.json  tokenizer.json`

Expected size: ~35 MB (one checkpoint + root adapter).

---

### Task 4: Trim stale eval outputs

Keep the 3 most recent eval JSON files. The two markdown outputs are deliberate artefacts, leave them.

- [ ] **Step 1: Delete all but the 3 newest eval JSONs**

```bash
cd outputs && ls -t eval_*.json | tail -n +4 | xargs rm -f && cd ..
```

- [ ] **Step 2: Verify outputs/**

```bash
ls outputs/
```

Expected: two `.md` files + three most recent `eval_*.json` files.

---

### Task 5: Tighten .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add training/traces_fresh/ to .gitignore**

The current `.gitignore` covers `training/traces/` and `training/checkpoints/` but not `training/traces_fresh/`. Add it so a future regeneration run doesn't accidentally show as untracked.

Edit `.gitignore` — in the "Training outputs" block, add one line after `training/traces/`:

```
training/traces_fresh/
```

The block should read:
```
# Training outputs — generated at runtime, not versioned
training/generated/
training/traces/
training/traces_fresh/
training/checkpoints/
training/runs/
training/logs/
```

- [ ] **Step 2: Remove coverage.xml from the repo**

`coverage.xml` is a test artefact. It's sitting untracked in the root.

```bash
rm -f coverage.xml
```

---

### Task 6: Commit and verify

- [ ] **Step 1: Stage all changes**

```bash
git add .gitignore
git status
```

Review the status output — the only staged change should be `.gitignore`. All deleted scripts were either untracked (no git action needed) or already handled by `git rm` in Task 2.

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: clean up stale traces, dead scripts, and intermediate checkpoints

- Replace training/traces with fresh ANALYSIS-format traces (1300 traces)
- Delete 7 stale backup/log files from training/traces/
- Delete 6 one-off migration scripts from scripts/
- Prune 9 intermediate checkpoints (~900 MB freed)
- Delete stale qwen2.5-3b-lora run (~481 MB freed)
- Trim outputs/ to last 3 eval files
- Add training/traces_fresh/ to .gitignore"
```

- [ ] **Step 3: Run the test suite to confirm nothing broke**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass. This cleanup touches no source files, so any failures indicate a pre-existing issue.

- [ ] **Step 4: Verify fine_tune.py can read the canonical traces**

```bash
python training/fine_tune.py --traces-dir training/traces/ --dry-run
```

Expected: prints the 1300-trace scenario breakdown table, exits without loading any model.

- [ ] **Step 5: Final tree check**

```bash
tree -L 3 --dirsfirst -I '__pycache__|*.pyc'
```

Expected directory count: well under 20 (was 39). The `training/` tree should show only `checkpoints/checkpoint-489/` as a subdirectory, plus the flat `traces/` and `generated/` directories.
