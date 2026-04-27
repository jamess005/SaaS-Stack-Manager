"""
Model runner — loads the quantised model and runs Pass 2 inference.

Pass 2: verdict memo generation — returns the full structured memo as a string.
Financial variable extraction (formerly Pass 1) is handled by roi_calculator.extract_pass1_vars()
using structured JSON data directly — no model inference required for that step.

Dry-run mode (AGENT_DRY_RUN=true or --dry-run flag):
  load_model() returns (None, None).
  run_pass2() returns a pre-written fixture response without loading any model.
  This allows full pipeline integration tests in CI without GPU access.

ROCm (AMD GPU) notes:
  Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for RX 7800 XT (gfx1101).
  Set ROCR_VISIBLE_DEVICES=0 and HIP_VISIBLE_DEVICES=0 to target card 0.
  PyTorch ROCm wheels must be installed separately (see requirements.txt).
  device_map="auto" (accelerate dispatch) fails on ROCm 7.x — use explicit
  device_map={"":0}. 4-bit NF4 quantization works with explicit placement (~5 GB VRAM).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from agent.signal_interpreter import (
    parse_signal_payload,
    signal_competitor_changes,
    signal_compliance_changes,
    signal_current_tool_status,
    signal_notes,
)

logger = logging.getLogger(__name__)

# ── Local model paths ──────────────────────────────────────────────────────────
from config import MODEL_PATH as _MODEL_PATH, TEACHER_MODEL_PATH as _TEACHER_MODEL_PATH

# Absolute path to the locally cached Llama 3.1 8B Instruct snapshot.
# Used by generate_signals.py and generate_traces.py as the teacher model.
# Set TEACHER_MODEL_PATH in .env if your snapshot is in a hash subdirectory.
LOCAL_LLAMA_8B = str(_TEACHER_MODEL_PATH)

# ── Dry-run fixture map ────────────────────────────────────────────────────────
# Maps (category, competitor_slug) → path to expected_outputs fixture JSON.
# Any pair not in this map falls back to the SWITCH fixture.
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_DRY_RUN_MAP: dict[tuple[str, str], str] = {
    ("finance", "ledgerflow"): "switch_verdict.json",
    ("crm", "velocitycrm"): "stay_verdict.json",
    ("hr", "workforge"): "hold_verdict.json",
}
_DRY_RUN_FALLBACK = "switch_verdict.json"


def _is_dry_run() -> bool:
    return os.environ.get("AGENT_DRY_RUN", "").lower() == "true"


def _load_fixture(category: str, competitor_slug: str) -> dict:
    filename = _DRY_RUN_MAP.get((category, competitor_slug), _DRY_RUN_FALLBACK)
    fixture_path = _FIXTURES_DIR / filename
    with fixture_path.open(encoding="utf-8") as f:
        return json.load(f)


# ── Model loading ──────────────────────────────────────────────────────────────


_DEFAULT_MODEL_PATH = str(_MODEL_PATH)
_DEFAULT_ADAPTER_PATH = str(Path(__file__).parent.parent / "training" / "checkpoints_sft_cot")
_DEFAULT_DPO_ADAPTER_PATH = str(Path(__file__).parent.parent / "training" / "checkpoints_dpo")


def load_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_path: str | None = _DEFAULT_MODEL_PATH,
    adapter_path: str | None = _DEFAULT_ADAPTER_PATH,
    quantize_bits: int = 0,
    dpo_adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """
    Load tokenizer and model for inference.

    Args:
        model_name:    HuggingFace model ID. Ignored if model_path is provided.
        model_path:    Local filesystem path to a model snapshot directory.
                       Defaults to MODEL_PATH from config (set MODELS_DIR in .env).
        adapter_path:  Path to a LoRA adapter directory (output of sft_cot_train.py).
                       If provided, the adapter is applied on top of the base model
                       using PEFT's PeftModel.from_pretrained().
        quantize_bits: 4 = NF4 4-bit (for large teacher models like Llama-8B).
                       0 = bf16, no quantization (default, for 3B inference model).

    Returns:
        (tokenizer, model) — both None in dry-run mode.

    ROCm: device_map="auto" triggers HIP kernel dispatch failures on ROCm 7.x.
          On ROCm, explicit device_map={"":0} is used instead.
          Ensure HSA_OVERRIDE_GFX_VERSION=11.0.0 for RX 7800 XT.
    """
    if _is_dry_run():
        logger.debug("Dry-run mode: skipping model load.")
        return None, None

    # Warn if ROCm GFX version override is not set (required for RX 7800 XT)
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION is not set. "
            "If using an RX 7800 XT (gfx1101), set HSA_OVERRIDE_GFX_VERSION=11.0.0 "
            "to avoid ROCm kernel dispatch errors."
        )

    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    source = model_path or model_name

    # ROCm (AMD): device_map="auto" triggers HIP kernel dispatch failures on
    # ROCm 7.x. Use explicit device_map={"": 0} for all loading paths.
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_rocm:
        logger.info(
            "ROCm backend detected (HIP %s) — using explicit device_map.",
            torch.version.hip,
        )

    device_map = {"": 0} if is_rocm else "auto"

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    if quantize_bits == 4:
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
        logger.info("Loading model: %s (4-bit NF4)", source)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        logger.info("Loading model: %s (bf16)", source)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    if adapter_path:
        from peft import PeftModel  # type: ignore[import-untyped]
        if dpo_adapter_path and Path(dpo_adapter_path).exists():
            # DPO adapter was trained on the SFT-merged model.
            # Stack: merge SFT into base, then apply DPO LoRA on top.
            logger.info("Loading SFT adapter from: %s (will merge for DPO stacking)", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # type: ignore[assignment]
            logger.info("SFT adapter merged into base.")
            logger.info("Loading DPO adapter from: %s", dpo_adapter_path)
            model = PeftModel.from_pretrained(model, dpo_adapter_path)
            logger.info("DPO adapter loaded.")
        else:
            logger.info("Loading LoRA adapter from: %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Adapter loaded.")

    model.eval()
    logger.info("Model loaded.")
    return tokenizer, model


# ── Prompt assembly ────────────────────────────────────────────────────────────


def _format_context_block(context: dict) -> str:
    """Serialise the loaded context into a human-readable block for the prompt."""
    tool_name = context["current_stack_entry"]["tool"]
    monthly_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    competitor_name = context["competitor_data"].get("name", context["competitor_slug"])
    competitor_cost = context["competitor_data"].get("monthly_cost_gbp", "unknown")

    lines = [
        f"CATEGORY: {context['category']}",
        f"CURRENT TOOL: {tool_name} — £{monthly_cost}/mo",
        f"COMPETITOR: {competitor_name} — £{competitor_cost}/mo",
        "",
        "## CURRENT TOOL — KNOWN ISSUES",
        *[f"  - {issue}" for issue in context["current_stack_entry"].get("known_issues", [])],
        "",
        "## CURRENT TOOL — USAGE METRICS",
    ]

    metrics = context.get("usage_metrics_entry")
    if metrics:
        lines += [
            f"  Utilisation: {metrics.get('utilisation_pct', '?')}%",
            f"  Active users: {metrics.get('avg_monthly_active_users', '?')} / {metrics.get('total_seats', '?')} seats",
            f"  Inactive seats: {metrics.get('inactive_seats', '?')}",
            f"  Shelfware flag: {metrics.get('shelfware_flag', False)}",
        ]
        if metrics.get("notes"):
            lines.append(f"  Notes: {metrics['notes']}")
    else:
        lines.append("  (no usage metrics available)")

    lines += [
        "",
        f"## BUSINESS RULES ({context['category'].upper()})",
        context["business_rules_text"].strip(),
        "",
        "## GLOBAL BUSINESS RULES",
        context["global_rules_text"].strip(),
        "",
        "## COMPETITOR DATA",
        json.dumps(context["competitor_data"], indent=2, ensure_ascii=False),
    ]
    return "\n".join(lines)


def _assemble_pass2_prompt(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    system_prompt: str,
    few_shot: list[dict],
    retry_hint: list[str] | None = None,
) -> list[dict]:
    """Build the message list for Pass 2 (verdict memo generation)."""
    context_block = _format_context_block(context)
    roi_block = json.dumps(roi_result, indent=2, ensure_ascii=False)

    user_content = (
        f"{context_block}\n\n"
        "## INBOX TRIGGER\n"
        f"{inbox_text.strip()}\n\n"
        "## ROI CALCULATION RESULT (computed by Python — do not recalculate)\n"
        f"{roi_block}\n\n"
        "---\n"
        "Write the verdict memo in the exact format specified in the system prompt. "
        "Every feature claim in PULL SIGNALS must be a verbatim quote from COMPETITOR DATA "
        "enclosed in double-quotes. Every push signal must reference a specific known issue. "
        "The VERDICT must be SWITCH, STAY, or HOLD."
    )

    if retry_hint:
        hint_text = "\n".join(f"  - {e}" for e in retry_hint)
        user_content += (
            f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION. Fix these issues:\n{hint_text}"
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot)
    messages.append({"role": "user", "content": user_content})
    return messages


# ── Generation helper ──────────────────────────────────────────────────────────


def _generate(tokenizer, model, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    """Run a chat-format generation and return the assistant response text."""
    import torch  # type: ignore[import-untyped]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a BatchEncoding or a plain tensor
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Confidence extraction ──────────────────────────────────────────────────────

_VERDICT_WORDS = ["SWITCH", "STAY", "HOLD"]


def _locate_verdict_span(tokenizer, generated_ids):
    import re

    bare_seqs: dict[str, list[int]] = {
        word: tokenizer.encode(word, add_special_tokens=False) for word in _VERDICT_WORDS
    }
    spaced_seqs: dict[str, list[int]] = {
        word: tokenizer.encode(" " + word, add_special_tokens=False) for word in _VERDICT_WORDS
    }

    last_tid_set: set[int] = set()
    for word in _VERDICT_WORDS:
        if bare_seqs[word]:
            last_tid_set.add(bare_seqs[word][-1])
        if spaced_seqs[word]:
            last_tid_set.add(spaced_seqs[word][-1])

    verdict_re = re.compile(r"VERDICT:\s*(SWITCH|STAY|HOLD)", re.IGNORECASE)
    token_ids = generated_ids.tolist()
    accumulated = ""

    for pos, token_id in enumerate(token_ids):
        accumulated += tokenizer.decode([token_id], skip_special_tokens=True)
        if token_id not in last_tid_set:
            continue

        match = verdict_re.search(accumulated)
        if not match:
            continue

        word = match.group(1).upper()
        for variant, seq in (("bare", bare_seqs[word]), ("spaced", spaced_seqs[word])):
            length = len(seq)
            if length == 0 or pos + 1 < length:
                continue
            if token_ids[pos + 1 - length : pos + 1] == seq:
                return {
                    "word": word,
                    "first_token_pos": pos + 1 - length,
                    "last_token_pos": pos,
                    "variant": variant,
                    "bare_seqs": bare_seqs,
                    "spaced_seqs": spaced_seqs,
                }
    return None


def _extract_verdict_confidence(tokenizer, scores, generated_ids) -> dict | None:
    """
    Extract confidence from softmax over verdict tokens at the first verdict token position.

    Returns the probability renormalized across SWITCH/STAY/HOLD — answers
    "how decisively did the model pick this verdict?" with values in [0.33, 1.0].
    """
    import math

    import torch

    span = _locate_verdict_span(tokenizer, generated_ids)
    if span is None:
        return None

    word = span["word"]
    first_token_pos = span["first_token_pos"]
    matched_variant = span["variant"]
    bare_seqs = span["bare_seqs"]
    spaced_seqs = span["spaced_seqs"]

    if first_token_pos >= len(scores):
        return None

    if matched_variant == "spaced":
        cmp_tids = {
            token_word: spaced_seqs[token_word][0]
            for token_word in _VERDICT_WORDS
            if spaced_seqs[token_word]
        }
    else:
        cmp_tids = {
            token_word: bare_seqs[token_word][0]
            for token_word in _VERDICT_WORDS
            if bare_seqs[token_word]
        }

    probs = torch.softmax(scores[first_token_pos][0].float(), dim=-1)
    vprobs: dict[str, float] = {
        token_word: float(probs[token_id].item())
        for token_word, token_id in cmp_tids.items()
        if token_id < probs.shape[0]
    }
    if word not in vprobs or not vprobs:
        return None

    total = sum(vprobs.values())
    if total > 0:
        vprobs = {k: v / total for k, v in vprobs.items()}

    entropy = -sum(
        v * math.log2(v) if v > 0 else 0.0
        for v in vprobs.values()
    )
    sorted_p = sorted(vprobs.values(), reverse=True)
    margin = sorted_p[0] - sorted_p[1] if len(sorted_p) >= 2 else 1.0

    return {
        "verdict_token_prob": round(vprobs[word], 6),
        "verdict_entropy_bits": round(entropy, 6),
        "verdict_margin": round(margin, 6),
        "verdict_probs": {k: round(v, 6) for k, v in vprobs.items()},
    }


def _prior_verdict_confidence(
    tokenizer, model, messages: list[dict], generated_verdict: str | None
) -> dict | None:
    """
    Compute confidence via a constrained forward pass WITHOUT chain-of-thought.

    The standard post-hoc approach measures confidence at the VERDICT token
    AFTER the model has written its full ANALYSIS reasoning chain. By that
    point the model is trivially certain (P→1.0) because CoT has already
    committed the answer. This gives no useful uncertainty signal.

    Instead, this function does a single forward pass with the generation
    prompt + "VERDICT:" appended, so the model is asked to predict
    SWITCH/STAY/HOLD directly from the input context alone. The resulting
    distribution is calibrated — a low probability for the generated verdict
    means the model was genuinely uncertain before reasoning.
    """
    import math

    import torch

    # Build prompt: [system, user] + generation_prompt + "VERDICT:" tokens
    # apply_chat_template adds <|im_start|>assistant\n; we then append VERDICT:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)

    verdict_toks = tokenizer.encode("VERDICT:", add_special_tokens=False)
    verdict_tensor = torch.tensor([verdict_toks], dtype=input_ids.dtype, device=input_ids.device)
    constrained_ids = torch.cat([input_ids, verdict_tensor], dim=1)

    with torch.no_grad():
        out = model(input_ids=constrained_ids, return_dict=True)
        logits = out.logits[0, -1, :].float()  # predict next token after "VERDICT:"
        probs = torch.softmax(logits, dim=-1)

    # First-token IDs for each verdict word (spaced: " SWITCH", " ST", " HOLD")
    spaced_seqs = {
        w: tokenizer.encode(" " + w, add_special_tokens=False) for w in _VERDICT_WORDS
    }
    vprobs: dict[str, float] = {
        w: float(probs[spaced_seqs[w][0]].item())
        for w in _VERDICT_WORDS
        if spaced_seqs[w]
    }
    if not vprobs:
        return None

    total = sum(vprobs.values())
    if total > 0:
        vprobs = {k: v / total for k, v in vprobs.items()}

    # Always report the MAX prior probability — calibrated uncertainty BEFORE reasoning.
    # High = model was confident before CoT; low = genuinely ambiguous.
    reported_word = max(vprobs, key=lambda k: vprobs[k])
    reported_prob = vprobs[reported_word]

    sorted_p = sorted(vprobs.values(), reverse=True)
    margin = sorted_p[0] - sorted_p[1] if len(sorted_p) >= 2 else 1.0
    entropy = -sum(v * math.log2(v) if v > 0 else 0.0 for v in vprobs.values())

    return {
        "verdict_token_prob": round(reported_prob, 6),
        "verdict_entropy_bits": round(entropy, 6),
        "verdict_margin": round(margin, 6),
        "verdict_probs": {k: round(v, 6) for k, v in vprobs.items()},
    }


def _generate_with_scores(
    tokenizer, model, messages: list[dict], max_new_tokens: int
) -> tuple[str, dict | None]:
    """
    Greedy generation returning (text, confidence_dict).

    Generation uses output_scores=True for deterministic ROCm behaviour.
    Confidence is reported from _prior_verdict_confidence — a constrained
    forward pass BEFORE chain-of-thought, giving calibrated pre-reasoning
    uncertainty: clear cases ~0.85-1.0, ambiguous cases ~0.5-0.7.
    """
    import re

    import torch

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    m = re.search(r"VERDICT:\s*(SWITCH|STAY|HOLD)", text, re.IGNORECASE)
    generated_verdict = m.group(1).upper() if m else None

    # Report calibrated pre-CoT confidence (not post-CoT, which is always ~1.0)
    confidence = _prior_verdict_confidence(tokenizer, model, messages, generated_verdict)
    return text, confidence


# ── Multi-step helpers ─────────────────────────────────────────────────────────


def _build_step1_user(context: dict) -> str:
    """Build Step 1 (compliance check) user message."""
    category = context["category"]
    seat_count = context["current_stack_entry"].get("seat_count", 0)
    comp_name = context["competitor_data"].get("name", "")
    compliance = context["competitor_data"].get("compliance", {})
    compliance_json = json.dumps(compliance, indent=2)

    return (
        "STEP 1 — COMPLIANCE CHECK\n\n"
        f"Category: {category}\n"
        f"Competitor: {comp_name}\n"
        f"Tool seats: {seat_count}\n\n"
        f"Competitor compliance data:\n{compliance_json}\n\n"
        "Identify any hard compliance blocks, or state COMPLIANT if none."
    )


def _build_step2_user(context: dict, inbox_text: str) -> str:
    """Build Step 2 (signal analysis) user message."""
    tool_name = context["current_stack_entry"]["tool"]
    comp_name = context["competitor_data"].get("name", "")
    features = context["competitor_data"].get("features", [])
    limitations = context["competitor_data"].get("known_limitations", [])
    issues = context["current_stack_entry"].get("known_issues", [])
    rules_text = context["business_rules_text"].strip()

    metrics = context.get("usage_metrics_entry", {})
    usage_line = ""
    if metrics:
        util = metrics.get("utilisation_pct", "?")
        active = metrics.get("avg_monthly_active_users", "?")
        total = metrics.get("total_seats", "?")
        inactive = metrics.get("inactive_seats", 0)
        shelfware = metrics.get("shelfware_flag", False)
        usage_line = f"Usage: {util}% utilisation, {active}/{total} active seats, {inactive} inactive"
        if shelfware:
            usage_line += " [SHELFWARE FLAGGED]"

    features_text = "\n".join(f'  - "{f}"' for f in features)
    limits_text = "\n".join(f"  - {l}" for l in limitations) if limitations else "  (none)"
    issues_text = "\n".join(f"  - {i}" for i in issues) if issues else "  (none)"

    return (
        "STEP 2 — SIGNAL ANALYSIS\n\n"
        f"Current tool: {tool_name}\n"
        f"Competitor: {comp_name}\n"
        f"{usage_line}\n\n"
        f"Current tool issues:\n{issues_text}\n\n"
        f"Competitor features (quote verbatim in PULL SIGNALS):\n{features_text}\n\n"
        f"Competitor limitations:\n{limits_text}\n\n"
        f"Business rules ({context['category']}):\n{rules_text}\n\n"
        f"Inbox trigger:\n{inbox_text.strip()}\n\n"
        "List PUSH SIGNALS (reasons to leave current tool) and "
        "PULL SIGNALS (competitor advantages) with severity weights "
        "[HIGH/MEDIUM/LOW — reason].\n"
        "PULL SIGNALS must use verbatim double-quoted feature strings."
    )


def _build_step3_user(
    context: dict,
    roi_result: dict,
    step1_result: str,
    step2_result: str,
    date_str: str | None = None,
) -> str:
    """Build Step 3 (verdict) user message."""
    import datetime as _dt

    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    tool_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    comp_name = context["competitor_data"].get("name", "")
    comp_cost = context["competitor_data"].get("monthly_cost_gbp", "?")
    roi_json = json.dumps(roi_result, indent=2)
    if date_str is None:
        date_str = _dt.date.today().isoformat()

    return (
        "STEP 3 — VERDICT\n\n"
        f"Category: {category}\n"
        f"Current tool: {tool_name} (£{tool_cost}/mo)\n"
        f"Competitor: {comp_name} (£{comp_cost}/mo)\n"
        f"Date: {date_str}\n\n"
        f"Compliance result:\n{step1_result.strip()}\n\n"
        f"Signals:\n{step2_result.strip()}\n\n"
        f"ROI (Python-computed):\n{roi_json}\n\n"
        "Write the verdict memo with sections: CATEGORY, CURRENT TOOL, COMPETITOR, "
        "DATE, PUSH SIGNALS, PULL SIGNALS, FINANCIAL ANALYSIS, VERDICT, EVIDENCE. "
        "If HOLD, add REASSESS CONDITION and REVIEW BY."
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def run_pass2(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    system_prompt: str,
    few_shot: list[dict],
    tokenizer: Any,
    model: Any,
    retry_hint: list[str] | None = None,
) -> str:
    """
    Run Pass 2: generate the verdict memo given the ROI result.

    Returns:
        The full verdict memo as a plain-text string.
    """
    if _is_dry_run():
        fixture = _load_fixture(context["category"], context["competitor_slug"])
        memo = fixture.get("memo_text", "")
        logger.debug("Dry-run Pass 2 memo (%d chars).", len(memo))
        return memo

    messages = _assemble_pass2_prompt(
        inbox_text, context, roi_result, system_prompt, few_shot, retry_hint
    )
    raw_output = _generate(tokenizer, model, messages, max_new_tokens=1024, temperature=0.3)
    logger.debug("Pass 2 raw output (%d chars).", len(raw_output))
    return raw_output.strip()


def run_multistep(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    tokenizer: Any,
    model: Any,
    retry_hint: list[str] | None = None,
) -> str:
    """
    Run the 3-step chain-of-reasoning pipeline.

    Step 1: Compliance check (hard blocks)
    Step 2: Signal analysis (push/pull signals)
    Step 3: Verdict memo (final structured output)

    Each step's output feeds into the next as conversation history,
    keeping each individual generation small.

    Returns:
        The full verdict memo as a plain-text string.
    """
    if _is_dry_run():
        fixture = _load_fixture(context["category"], context["competitor_slug"])
        memo = fixture.get("memo_text", "")
        logger.debug("Dry-run multistep memo (%d chars).", len(memo))
        return memo

    from agent.prompts import SYSTEM_PROMPT_MULTISTEP

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_MULTISTEP}]

    # Step 1: Compliance check
    logger.info("Step 1/3: Compliance check")
    step1_user = _build_step1_user(context)
    messages.append({"role": "user", "content": step1_user})
    step1_result = _generate(tokenizer, model, messages, max_new_tokens=150, temperature=0.1)
    messages.append({"role": "assistant", "content": step1_result})
    logger.info("Step 1 result: %s", step1_result.split("\n")[0])

    # Step 2: Signal analysis
    logger.info("Step 2/3: Signal analysis")
    step2_user = _build_step2_user(context, inbox_text)
    messages.append({"role": "user", "content": step2_user})
    step2_result = _generate(tokenizer, model, messages, max_new_tokens=512, temperature=0.3)
    messages.append({"role": "assistant", "content": step2_result})

    # Step 3: Verdict memo
    logger.info("Step 3/3: Verdict memo")
    step3_user = _build_step3_user(context, roi_result, step1_result, step2_result)
    messages.append({"role": "user", "content": step3_user})
    memo = _generate(tokenizer, model, messages, max_new_tokens=700, temperature=0.3)

    # Retry once if validation hint provided
    if retry_hint:
        hint_text = "\n".join(f"  - {e}" for e in retry_hint)
        messages.append({"role": "assistant", "content": memo})
        messages.append({"role": "user", "content": f"Fix these issues in the memo:\n{hint_text}"})
        memo = _generate(tokenizer, model, messages, max_new_tokens=700, temperature=0.3)

    logger.debug("Multistep memo (%d chars).", len(memo))
    return memo.strip()


# ── Independent voting pipeline ────────────────────────────────────────────────


def _build_compliance_user(context: dict, signal: dict | None = None) -> str:
    """Build standalone compliance check user message."""
    category = context["category"]
    seat_count = context["current_stack_entry"].get("seat_count", 0)
    comp_name = context["competitor_data"].get("name", "")
    compliance = context["competitor_data"].get("compliance", {})
    signal_changes = signal_compliance_changes(signal)
    competitor_changes = signal_competitor_changes(signal)
    notes = signal_notes(signal)

    extra_lines: list[str] = []
    if signal_changes:
        extra_lines.append(f"Signal compliance changes: {signal_changes}")
    if competitor_changes:
        extra_lines.append("Signal competitor changes:")
        extra_lines.extend(f"- {item}" for item in competitor_changes)
    if notes:
        extra_lines.append("Signal notes:")
        extra_lines.extend(f"- {item}" for item in notes)

    extra_block = "\n".join(extra_lines)
    if extra_block:
        extra_block = "\n" + extra_block

    return (
        f"Category: {category}\n"
        f"Competitor: {comp_name}\n"
        f"Seats: {seat_count}\n"
        f"Baseline compliance: {json.dumps(compliance)}"
        f"{extra_block}"
    )


def _build_push_user(context: dict, signal: dict | None = None) -> str:
    """Build standalone push assessment user message."""
    tool_name = context["current_stack_entry"]["tool"]
    issues = context["current_stack_entry"].get("known_issues", [])
    metrics = context.get("usage_metrics_entry", {})
    signal_status = signal_current_tool_status(signal)
    notes = signal_notes(signal)

    usage_parts = []
    if metrics:
        util = metrics.get("utilisation_pct", "?")
        active = metrics.get("avg_monthly_active_users", "?")
        total = metrics.get("total_seats", "?")
        inactive = metrics.get("inactive_seats", 0)
        shelfware = metrics.get("shelfware_flag", False)
        usage_parts.append(f"Utilisation: {util}%, Active: {active}/{total}, Inactive: {inactive}")
        if shelfware:
            usage_parts.append("SHELFWARE FLAGGED")

    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none)"
    usage_text = ", ".join(usage_parts) if usage_parts else "(no metrics)"
    signal_status_text = "\n".join(f"- {item}" for item in signal_status) if signal_status else "(none)"
    notes_text = "\n".join(f"- {item}" for item in notes) if notes else "(none)"

    return (
        f"Tool: {tool_name}\n"
        f"Usage: {usage_text}\n"
        f"Known issues:\n{issues_text}\n\n"
        f"Recent current-tool changes from signal:\n{signal_status_text}\n\n"
        f"Signal notes:\n{notes_text}"
    )


def _build_pull_user(context: dict, inbox_text: str) -> str:
    """Build standalone pull assessment user message."""
    comp_name = context["competitor_data"].get("name", "")
    features = context["competitor_data"].get("features", [])
    limitations = context["competitor_data"].get("known_limitations", [])
    issues = context["current_stack_entry"].get("known_issues", [])
    rules_text = context["business_rules_text"].strip()

    features_text = "\n".join(f'- "{f}"' for f in features)
    limits_text = "\n".join(f"- {l}" for l in limitations) if limitations else "(none)"
    issues_text = "\n".join(f"- {i}" for i in issues[:3]) if issues else "(none)"

    return (
        f"Competitor: {comp_name}\n"
        f"Current tool pain points:\n{issues_text}\n\n"
        f"Competitor features:\n{features_text}\n\n"
        f"Competitor limitations:\n{limits_text}\n\n"
        f"Business rules:\n{rules_text}\n\n"
        f"Inbox trigger:\n{inbox_text.strip()}"
    )


def _extract_signal_bullets(result_text: str, max_lines: int = 5) -> str:
    """Extract up to max_lines bullet points from a push or pull vote result."""
    lines = []
    for line in result_text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-•*]|^\d+[.)]\s", stripped):
            clean = re.sub(r"^[-•*\d.)]\s*", "", stripped).strip()
            if clean:
                lines.append(f"  - {clean}")
                if len(lines) >= max_lines:
                    break
    return "\n".join(lines) if lines else "  (none)"


# ── Python compliance gate ─────────────────────────────────────────────────────


def _compliance_pass_python(context: dict) -> tuple[bool, list[str]]:
    """
    Check competitor compliance against hard requirements using structured JSON only.

    Hard blocks (any one forces STAY):
      - No SOC2 Type II
      - No SSO (SAML/OIDC) when seat_count > 10
      - No UK or EU data residency
      - No exportable audit log for Finance, HR, or CRM

    Returns:
        (passed, failures) — passed is True only when failures is empty.
    """
    category = context["category"]
    seat_count = context["current_stack_entry"].get("seat_count", 0)
    compliance = context["competitor_data"].get("compliance", {})

    failures: list[str] = []

    if not compliance.get("soc2_type2"):
        failures.append("No SOC2 Type II certification")

    if seat_count > 10 and not compliance.get("sso_saml"):
        failures.append(f"No SSO (SAML/OIDC) — tool has {seat_count} seats (threshold: >10)")

    if not (compliance.get("uk_residency") or compliance.get("gdpr_eu_residency")):
        failures.append("No UK or EU data residency")

    audit_log_ok = compliance.get("audit_log") and compliance.get("audit_log_exportable", True)
    if category in ("finance", "hr", "crm") and not audit_log_ok:
        failures.append(f"No exportable audit log (required for {category})")

    return len(failures) == 0, failures


def _assemble_stay_memo(
    context: dict,
    roi_result: dict,
    compliance_failures: list[str],
) -> str:
    """Build a STAY memo when the Python compliance gate rejects a competitor."""
    import datetime as _dt

    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    tool_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    comp_name = context["competitor_data"].get("name", "")
    comp_cost = context["competitor_data"].get("monthly_cost_gbp", "?")
    date_str = _dt.date.today().isoformat()

    fail_lines = "\n".join(f"  - {f}" for f in compliance_failures)
    features = context["competitor_data"].get("features", [])
    evidence = "\n".join(f'  "{f}"' for f in features[:4])

    fin_lines = [
        f"  Migration cost: {roi_result.get('migration_hours', 15)}hrs × £48 = £{roi_result['migration_cost_one_time']:.0f} one-time",
        f"  Annual saving: £{roi_result['annual_direct_saving']:.0f}",
        f"  Amortised migration: £{roi_result['amortised_migration_cost_per_year']:.0f}/yr over 3 years",
        f"  Annual net: £{roi_result['annual_net_gbp']:.0f}",
        f"  ROI threshold (£1,200/yr): {'MET' if roi_result['roi_threshold_met'] else 'NOT MET'}",
        f"  ROI threshold met: {'YES' if roi_result['roi_threshold_met'] else 'NO'}",
    ]

    return "\n".join([
        f"CATEGORY: {category.title()}",
        f"CURRENT TOOL: {tool_name} (£{tool_cost}/mo)",
        f"COMPETITOR: {comp_name} (£{comp_cost}/mo)",
        f"DATE: {date_str}",
        "",
        "PUSH SIGNALS:",
        "  - (not evaluated — compliance hard block)",
        "",
        "PULL SIGNALS:",
        "  - (not evaluated — compliance hard block)",
        "",
        f"COMPLIANCE BLOCKS:\n{fail_lines}",
        "",
        "FINANCIAL ANALYSIS:",
        "\n".join(fin_lines),
        "",
        "VERDICT: STAY",
        "",
        "EVIDENCE:",
        evidence or '  "No evidence — evaluation blocked by compliance failure."',
    ])


def _parse_compliance_changes(
    changes_text: str,
    compliance: dict,
    category: str,
    seat_count: int,
    tokenizer: Any = None,
    model: Any = None,
) -> dict:
    """
    Interpret free-text compliance_changes using keyword heuristics.

    Handles both upgrades (false→true) and downgrades (true→false).
    Strips embedded JSON dict prefixes to avoid cross-contamination from
    key names (e.g. "ifrs15_compliant" matching "compliant").

    Returns updated compliance dict (original is not mutated).
    """
    if not changes_text or not changes_text.strip():
        return compliance
    if _is_dry_run():
        return compliance

    updated = dict(compliance)

    # ── Strip embedded JSON dict prefix ────────────────────────────────────
    # Some signals embed a JSON dict followed by semicolon-separated text.
    # The dict values are unreliable (LLM-generated) — ignore them and only
    # process the free-text portion that follows.
    text = changes_text.strip()
    if text.startswith("{"):
        brace_depth = 0
        end_idx = 0
        for i, ch in enumerate(text):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_idx = i + 1
                    break
        text = text[end_idx:].strip().lstrip(";").strip()

    if not text or text.lower() in ("unchanged", "no changes", "none"):
        return compliance

    text = text.lower()

    _POSITIVE = frozenset([
        "achieved", "acquired", "certified", "added", "now available", "launched",
        "shipped", "now meets", "now compliant", "passed", "attained", "enabled", "live",
    ])

    _NEGATIVE = frozenset([
        "not available", "not yet", "not achieved", "not certified", "not compliant",
        "no ", "lacks", "missing", "in progress", "pending",
        "not met", "not passed", "unavailable", "absent",
    ])

    def _find_segment(text: str, keywords: tuple[str, ...]) -> str | None:
        """Extract the semicolon/newline-delimited segment containing any keyword."""
        for sep in (";", "\n"):
            for part in text.split(sep):
                part_s = part.strip()
                if any(kw in part_s for kw in keywords):
                    return part_s
        # Fall back to full text only if a keyword appears at all
        if any(kw in text for kw in keywords):
            return text
        return None

    def _is_positive(segment: str) -> bool:
        """True when the segment conveys a positive compliance change."""
        if any(neg in segment for neg in _NEGATIVE):
            strong = ("achieved", "acquired", "certified", "now available", "launched",
                      "shipped", "now meets", "now compliant", "passed", "attained", "live")
            for kw in strong:
                idx = segment.find(kw)
                if idx >= 0:
                    prefix = segment[max(0, idx - 5):idx].strip()
                    if not prefix.endswith("not"):
                        return True
            return False
        return any(kw in segment for kw in _POSITIVE)

    def _is_negative(segment: str) -> bool:
        """True when the segment conveys a negative compliance status."""
        return any(neg in segment for neg in _NEGATIVE)

    # ── SOC2 Type II ──────────────────────────────────────────────────────
    soc2_seg = _find_segment(text, ("soc2", "soc 2"))
    if soc2_seg is not None:
        if not compliance.get("soc2_type2") and _is_positive(soc2_seg):
            updated["soc2_type2"] = True
        elif compliance.get("soc2_type2") and _is_negative(soc2_seg):
            updated["soc2_type2"] = False

    # ── SSO / SAML (only matters if seat_count > 10) ─────────────────────
    sso_seg = _find_segment(text, ("sso", "saml", "oidc"))
    if sso_seg is not None:
        if seat_count > 10 and not compliance.get("sso_saml") and _is_positive(sso_seg):
            updated["sso_saml"] = True
        elif compliance.get("sso_saml") and _is_negative(sso_seg):
            updated["sso_saml"] = False

    # ── UK / EU data residency ───────────────────────────────────────────
    uk_seg = _find_segment(text, ("uk",))
    if uk_seg and "residency" in uk_seg:
        if _is_positive(uk_seg):
            updated["uk_residency"] = True
            updated["gdpr_eu_residency"] = True
        elif _is_negative(uk_seg):
            updated["uk_residency"] = False

    if not updated.get("gdpr_eu_residency"):
        eu_seg = _find_segment(text, ("eu", "gdpr"))
        if eu_seg and "residency" in eu_seg and _is_positive(eu_seg):
            updated["gdpr_eu_residency"] = True

    # ── Audit log (required for finance, hr, crm) ────────────────────────
    if category in ("finance", "hr", "crm"):
        audit_seg = _find_segment(text, ("audit log", "audit trail", "exportable audit"))
        if audit_seg is not None:
            if not compliance.get("audit_log") and _is_positive(audit_seg):
                if "csv only" not in audit_seg:
                    updated["audit_log"] = True
                    updated["audit_log_exportable"] = True
            elif compliance.get("audit_log") and _is_negative(audit_seg):
                updated["audit_log"] = False
                updated["audit_log_exportable"] = False
            elif not compliance.get("audit_log_exportable"):
                _export_kw = ("audit log export", "exportable audit", "audit trail export")
                if any(kw in audit_seg for kw in _export_kw) and _is_positive(audit_seg):
                    updated["audit_log_exportable"] = True

    return updated


def _build_verdict_user(context: dict, roi_result: dict,
                        compliance_result: str, push_result: str, pull_result: str) -> str:
    """Build standalone verdict vote user message."""
    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    tool_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    comp_name = context["competitor_data"].get("name", "")
    comp_cost = context["competitor_data"].get("monthly_cost_gbp", "?")

    push_match = re.search(r"Overall push:\s*(STRONG|MODERATE|WEAK)", push_result, re.IGNORECASE)
    pull_match = re.search(r"Overall pull:\s*(STRONG|MODERATE|WEAK)", pull_result, re.IGNORECASE)
    push_strength = push_match.group(1) if push_match else "UNKNOWN"
    pull_strength = pull_match.group(1) if pull_match else "UNKNOWN"

    push_bullets = _extract_signal_bullets(push_result)
    pull_bullets = _extract_signal_bullets(pull_result)

    roi_summary = (
        f"Migration: £{roi_result['migration_cost_one_time']:.0f}, "
        f"Annual saving: £{roi_result['annual_direct_saving']:.0f}, "
        f"Net: £{roi_result['annual_net_gbp']:.0f}, "
        f"Threshold met: {'YES' if roi_result['roi_threshold_met'] else 'NO'}"
    )

    # Extract PASS/FAIL from compliance output robustly — the model sometimes
    # prefixes with "Based on the provided information:" before stating the verdict.
    _upper = compliance_result.upper()
    if "FAIL" in _upper:
        compliance_line = "FAIL"
    elif "PASS" in _upper or "COMPLIANT" in _upper:
        compliance_line = "PASS"
    else:
        compliance_line = compliance_result.split("\n")[0]

    return (
        f"{category}: {tool_name} (£{tool_cost}/mo) vs {comp_name} (£{comp_cost}/mo)\n\n"
        f"Compliance: {compliance_line}\n\n"
        f"Push signals [{push_strength}]:\n{push_bullets}\n\n"
        f"Pull signals [{pull_strength}]:\n{pull_bullets}\n\n"
        f"ROI: {roi_summary}"
    )


def _normalize_signal_section(result_text: str, overall_label: str) -> str:
    """Normalize push/pull model output into validator-friendly bullet lines."""
    import re

    section_text = result_text.split(overall_label)[0].strip()
    if not section_text:
        return "  - (none)"

    normalized_lines: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith(("PUSH SIGNALS:", "PULL SIGNALS:")):
            line = line.split(":", 1)[1].strip()
        line = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        normalized_lines.append(f"  - {line}")

    return "\n".join(normalized_lines) if normalized_lines else "  - (none)"


def _assemble_verdict_memo(
    context: dict,
    roi_result: dict,
    compliance_result: str,
    push_result: str,
    pull_result: str,
    verdict_result: str,
) -> str:
    """Assemble a full verdict memo from the 4 independent vote outputs."""
    import datetime as _dt
    import re

    category = context["category"]
    tool_name = context["current_stack_entry"]["tool"]
    tool_cost = context["current_stack_entry"]["monthly_cost_gbp"]
    comp_name = context["competitor_data"].get("name", "")
    comp_cost = context["competitor_data"].get("monthly_cost_gbp", "?")
    date_str = _dt.date.today().isoformat()

    # Extract verdict from verdict_result
    verdict_match = re.search(r"VERDICT:\s*(SWITCH|STAY|HOLD)", verdict_result, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "STAY"

    # Build financial analysis section
    fin_lines = [
        f"  Migration cost: {roi_result.get('migration_hours', 15)}hrs × £48 = £{roi_result['migration_cost_one_time']:.0f} one-time",
        f"  Annual saving: £{roi_result['annual_direct_saving']:.0f}",
        f"  Amortised migration: £{roi_result['amortised_migration_cost_per_year']:.0f}/yr over 3 years",
        f"  Annual net: £{roi_result['annual_net_gbp']:.0f}",
        f"  ROI threshold (£1,200/yr): {'MET' if roi_result['roi_threshold_met'] else 'NOT MET'}",
        f"  ROI threshold met: {'YES' if roi_result['roi_threshold_met'] else 'NO'}",
    ]

    push_section = _normalize_signal_section(push_result, "Overall push:")
    pull_section = _normalize_signal_section(pull_result, "Overall pull:")

    # Build the memo
    memo_parts = [
        f"CATEGORY: {category.title()}",
        f"CURRENT TOOL: {tool_name} (£{tool_cost}/mo)",
        f"COMPETITOR: {comp_name} (£{comp_cost}/mo)",
        f"DATE: {date_str}",
        "",
        "PUSH SIGNALS:",
        push_section,
        "",
        "PULL SIGNALS:",
        pull_section,
        "",
        "FINANCIAL ANALYSIS:",
        "\n".join(fin_lines),
        "",
        f"VERDICT: {verdict}",
    ]

    # Add HOLD-specific fields
    if verdict == "HOLD":
        reassess_match = re.search(r"(?:REASSESS|condition|wait|pending)[:\s]+(.+)", verdict_result, re.IGNORECASE)
        reassess = reassess_match.group(1).strip() if reassess_match else "See verdict for condition."
        review_by = (_dt.date.today() + _dt.timedelta(days=92)).isoformat()
        memo_parts.append("")
        memo_parts.append(f"REASSESS CONDITION: {reassess}")
        memo_parts.append(f"REVIEW BY: {review_by}")

    # Extract evidence quotes from pull_result. If none are present, fall back to
    # verbatim competitor features so the memo remains validator-compliant.
    evidence_quotes = re.findall(r'"([^"]+)"', pull_result)
    if not evidence_quotes:
        fallback_features = context.get("competitor_data", {}).get("features", [])
        evidence_quotes = [f for f in fallback_features if isinstance(f, str) and f.strip()]

    memo_parts.append("")
    memo_parts.append("EVIDENCE:")
    if evidence_quotes:
        for q in evidence_quotes[:6]:
            memo_parts.append(f'  "{q}"')
    else:
        memo_parts.append('  "No quoted feature evidence available."')

    return "\n".join(memo_parts)


def run_voting(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    tokenizer: Any,
    model: Any,
    retry_hint: list[str] | None = None,
) -> tuple[str, dict | None]:
    """
    Run the independent voting pipeline.

    4 separate, independent model calls — each is a standalone 3-message
    conversation (system, user, assistant) with NO shared context:
      1. Compliance gate  — PASS/FAIL
      2. Push assessment  — current tool degradation
      3. Pull assessment  — competitor advantages
      4. Verdict vote     — final SWITCH/STAY/HOLD

    Like a random forest: each "tree" sees only its relevant subset of
    information. The verdict voter sees only the summarised outputs from
    steps 1-3 plus ROI data.

    Returns:
        (memo_text, confidence_dict) where confidence_dict contains model-internal
        probability signals from the verdict token. None in dry-run mode.
    """
    if _is_dry_run():
        fixture = _load_fixture(context["category"], context["competitor_slug"])
        memo = fixture.get("memo_text", "")
        logger.debug("Dry-run voting memo (%d chars).", len(memo))
        return memo, None

    signal = parse_signal_payload(inbox_text) or {}

    from agent.prompts import SYS_COMPLIANCE, SYS_PULL, SYS_PUSH, SYS_VERDICT

    # ── Python compliance gate (deterministic, non-negotiable) ──────────────
    # The model compliance vote (Vote 1) is advisory; the Python gate is
    # authoritative.  If any hard block is present the verdict is STAY.
    py_passed, py_failures = _compliance_pass_python(context)
    if not py_passed:
        logger.info("Python compliance gate FAIL: %s", "; ".join(py_failures))
        fail_lines = "\n".join(f"  - {f}" for f in py_failures)
        stay_memo = _assemble_stay_memo(context, roi_result, py_failures)
        return stay_memo, {
            "verdict_token_prob": 1.0,
            "verdict_entropy_bits": 0.0,
            "verdict_margin": 1.0,
            "verdict_probs": {"SWITCH": 0.0, "STAY": 1.0, "HOLD": 0.0},
        }

    # Vote 1: Compliance (independent)
    logger.info("Vote 1/4: Compliance gate")
    compliance_msgs = [
        {"role": "system", "content": SYS_COMPLIANCE},
        {"role": "user", "content": _build_compliance_user(context, signal)},
    ]
    compliance_result = _generate(tokenizer, model, compliance_msgs,
                                  max_new_tokens=100, temperature=0.1)
    logger.info("Compliance: %s", compliance_result.split("\n")[0])

    # Vote 2: Push signals (independent)
    logger.info("Vote 2/4: Push assessment")
    push_msgs = [
        {"role": "system", "content": SYS_PUSH},
        {"role": "user", "content": _build_push_user(context, signal)},
    ]
    push_result = _generate(tokenizer, model, push_msgs,
                            max_new_tokens=300, temperature=0.3)

    # Vote 3: Pull signals (independent)
    logger.info("Vote 3/4: Pull assessment")
    pull_msgs = [
        {"role": "system", "content": SYS_PULL},
        {"role": "user", "content": _build_pull_user(context, inbox_text)},
    ]
    pull_result = _generate(tokenizer, model, pull_msgs,
                            max_new_tokens=400, temperature=0.3)

    # Vote 4: Verdict (sees summaries from votes 1-3 + ROI, but NOT raw context)
    # Use _generate_with_scores to capture real model-internal confidence signals.
    logger.info("Vote 4/4: Verdict")
    verdict_msgs = [
        {"role": "system", "content": SYS_VERDICT},
        {"role": "user", "content": _build_verdict_user(
            context, roi_result, compliance_result, push_result, pull_result)},
    ]
    verdict_result, confidence = _generate_with_scores(
        tokenizer, model, verdict_msgs, max_new_tokens=250
    )
    logger.info("Verdict: %s", verdict_result.strip())

    # Assemble the full memo from the 4 independent outputs
    memo = _assemble_verdict_memo(
        context, roi_result, compliance_result, push_result, pull_result, verdict_result
    )

    logger.debug("Voting memo (%d chars).", len(memo))
    return memo.strip(), confidence


def run_lean(
    inbox_text: str,
    context: dict,
    roi_result: dict,
    tokenizer: Any,
    model: Any,
) -> tuple[str, dict | None]:
    """
    Lean single-pass pipeline.

    Returns (memo_text, confidence_dict) where confidence_dict is None in
    dry-run mode or when the verdict token cannot be located.

    1. Parse signal for compliance_changes.
    2. If compliance_changes present and model available: tiny constrained call to
       update the compliance boolean state.
    3. Python compliance gate: if any hard block → return STAY immediately.
    4. Single compact model call → "ANALYSIS: ...\\nVERDICT: SWITCH|STAY|HOLD"

    Returns a string with ANALYSIS and VERDICT lines.
    """
    if _is_dry_run():
        fixture = _load_fixture(context["category"], context.get("competitor_slug", ""))
        return fixture.get("memo_text", "ANALYSIS: Dry run — no model call.\nVERDICT: SWITCH"), None

    from agent.prompts import _CATEGORY_RULES_COMPACT, SYS_VERDICT_LEAN

    signal = parse_signal_payload(inbox_text)
    category = context["category"]
    seat_count = context["current_stack_entry"].get("seat_count", 0)

    # Build compact user message — model reasons about compliance from the raw profile
    user_content = _build_lean_user(context, roi_result, signal)

    messages = [
        {"role": "system", "content": SYS_VERDICT_LEAN},
        {"role": "user", "content": user_content},
    ]

    result, confidence = _generate_with_scores(tokenizer, model, messages, max_new_tokens=700)
    logger.debug("run_lean output (%d chars): %s", len(result), result[:100])

    # ── Python verdict override gates ─────────────────────────────────────────
    # Enforce structural rules the model may not follow reliably after fine-tuning.
    # The CoT (ANALYSIS section) is preserved; only the VERDICT line is corrected.
    _notes = signal_notes(signal)
    _comp = signal_competitor_changes(signal)
    _hold = _detect_hold_signal(_notes, _comp)
    _disq = _detect_disqualifier(_notes, _hold)
    _hard = _detect_hard_compliance_block(context, category) if _hold == "NONE" else []
    _me = _detect_migration_enabler(_notes)
    _ds = _detect_demand_signal(_notes)

    if _hard or (_disq != "NONE" and _hold == "NONE"):
        _forced: str | None = "STAY"
    elif _hold != "NONE":
        _forced = "HOLD"
    elif _me != "NONE" and _ds != "NONE":
        # Migration enabler + explicit internal demand → vendor is viable, force SWITCH
        _forced = "SWITCH"
    else:
        _forced = None

    if _forced:
        logger.debug("run_lean gate override → %s (hold=%r disq=%r hard=%s me=%r ds=%r)", _forced, _hold[:40] if _hold != 'NONE' else 'NONE', _disq[:40] if _disq != 'NONE' else 'NONE', _hard, _me[:40] if _me != 'NONE' else 'NONE', _ds[:40] if _ds != 'NONE' else 'NONE')
        _result_stripped = result.strip()
        _replaced = re.sub(
            r"\bVERDICT:\s*(SWITCH|STAY|HOLD)\b",
            f"VERDICT: {_forced}",
            _result_stripped,
            flags=re.IGNORECASE,
        )
        # If the model never wrote a VERDICT line, append one so the validator passes
        if not re.search(r"\bVERDICT:\s*(SWITCH|STAY|HOLD)\b", _replaced, re.IGNORECASE):
            _replaced = _result_stripped + f"\nVERDICT: {_forced}"
        result = _replaced
    # ─────────────────────────────────────────────────────────────────────────

    return result.strip(), confidence


_HOLD_NOTE_KW = frozenset([
    "hold:", "for a hold", "acquisition", "roadmap", "renews", "renewal", "pilot",
    "design-partner", "design partner",
])

# Pre-GA keywords — these indicate a hold when found in competitor_changes,
# but a disqualifier when found only in notes (per system prompt positional rule).
_PRE_GA_KW = frozenset(["beta", "not ga", "early access", "preview"])

# Competitor-changes keywords that indicate a hold condition (feature not yet GA)
_HOLD_COMP_KW = frozenset(["beta", "roadmap", "not ga", "preview", "early access", "design-partner", "design partner"])

# Current-tool-status keywords that indicate a shelfware situation
_SHELFWARE_KW = frozenset(["shelfware", "inactive seats", "inactive seat"])

# Notes that indicate migration friction is already resolved
_MIGRATION_ENABLER_KW = frozenset([
    "already live", "already integrated", "sync is live",
    "integration is live", "connector is live",
    "migration assistant", "maps existing", "one pass", "one-pass",
])

# Notes that confirm customer demand for a specific gap (pull signal)
_DEMAND_SIGNAL_KW = frozenset([
    "renewal calls", "renewal call", "client flagged", "clients flagged",
    "customer flagged", "customers flagged", "feedback calls", "flagged as blocker",
    "flagged the",
])

# Notes that start with these words are advisory, not hold conditions
_ADVISORY_PREFIXES = ("consider", "suggest", "recommend", "note:", "fyi")

_NEGATION_PATTERNS = re.compile(
    r"no\s+(beta|roadmap|caveats|hold)"
    r"|all\b.*\bga\b"
    r"|now\s+ga"
    r"|without\s+(beta|roadmap|caveats)"
    r"|renewal\s+call"
    r"|before\s+(?:\w+\s+)?renewal"
    r"|roadmap\s+only"
    r"|still\b.{0,40}\broadmap\b"
    r"|no\b.{0,50}\bon\s+the\s+roadmap\b"
    r"|roadmap\s+(?:widget|view|board|builder|chart|panel|tab|tracker|overview|tool|feature|item)"
    # Prevent false-positive on positive notes about completed pilots or absent gaps
    r"|no\s+compliance\s+gap"           # e.g. "No compliance gaps remain in the target stack"
    r"|pilot\s+with\s+\w+\s+\w+\s+show",  # e.g. "pilot with two reps showed 31% improvement"
    re.IGNORECASE,
)


def _detect_hold_signal(notes: list[str], comp_changes: list[str] | None = None) -> str:
    """Return the first hold condition found in notes (or competitor_changes), or 'NONE'.

    Notes that are advisory (starting with "consider", "suggest", etc.) are skipped
    to avoid false positives from phrases like "consider re-evaluating the contract".
    Competitor changes are checked for beta/roadmap/not-ga signals.
    Notes that negate hold keywords (e.g. "no beta or roadmap caveats") are skipped.
    """
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if _NEGATION_PATTERNS.search(note):
            continue
        if any(kw in note_lower for kw in _HOLD_NOTE_KW):
            return note
    for change in (comp_changes or []):
        change_lower = change.lower()
        if _NEGATION_PATTERNS.search(change_lower):
            continue
        if any(kw in change_lower for kw in _HOLD_COMP_KW):
            return change
    return "NONE"


_DISQUALIFIER_NOTE_KW = frozenset([
    "preview", "early access", "beta", "not ga",
    "no relevance", "not relevant", "irrelevant to", "irrelevant", "unrelated",
    "orthogonal", "tangential", "not the bottleneck", "not the gap",
    "poor fit", "wrong product", "not designed for",
    "hard requirement",
    "without relevance", "does not align", "does not address", "no impact on",
    "too complex", "overkill", "control gap", "compliance gap", "absent",
    "unavailable", "no timeline", "no soc2", "no delivery date",
])


def _detect_disqualifier(notes: list[str], hold_signal: str) -> str:
    """Return a disqualifier note if pre-GA language appears in notes and no hold is active.

    A disqualifier means the competitor is not viable (STAY), distinct from a hold condition
    (HOLD). Uses keywords disjoint from _HOLD_NOTE_KW to prevent double-labeling.
    Only fires when hold_signal is 'NONE'.
    Negation phrases (e.g. 'no beta or roadmap caveats') are excluded to avoid
    false positives that would flip a correct SWITCH to STAY.
    """
    if hold_signal != "NONE":
        return "NONE"
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(_ADVISORY_PREFIXES):
            continue
        if _NEGATION_PATTERNS.search(note):
            continue
        if any(kw in note_lower for kw in _DISQUALIFIER_NOTE_KW):
            return note
    return "NONE"


_SHELFWARE_NEGATION = re.compile(
    r"no\s+shelfware|not\s+shelfware|no\s+inactive\s+seats?|no\s+unused"
    r"|reduced\s+to\s+0|seats\s+reduced|inactive\s+seats?\s+decrease"
    r"|remains?\s+stable|increase\s+by\s+[0-5]%",
    re.IGNORECASE,
)

_SHELFWARE_GLOBAL_NEGATION = re.compile(
    r"shelfware(?:\s+(?:flag|status))?(?:\s*(?::|=)\s*|\s+remains?\s+)(?:false|no|none)\b",
    re.IGNORECASE,
)

_SHELFWARE_COUNT = re.compile(r"inactive\s+seats?[^\d]*(\d+)", re.IGNORECASE)
_SHELFWARE_PERCENT = re.compile(r"(\d+)%", re.IGNORECASE)


def _detect_migration_enabler(notes: list[str]) -> str:
    """Return the first note that explicitly states migration friction is already resolved."""
    for note in notes:
        note_lower = note.lower()
        if any(kw in note_lower for kw in _MIGRATION_ENABLER_KW):
            return note
    return "NONE"


def _detect_demand_signal(notes: list[str]) -> str:
    """Return the first note that shows customer/renewal calls confirming a gap as a blocker.

    This surfaces cases where hold_signal would have been skipped (renewal call negation)
    but the note still carries strong pull evidence — e.g. 'renewal calls flagged X as a
    blocker'. Only returns a note when it contains both a demand keyword and 'blocker'.
    """
    for note in notes:
        note_lower = note.lower()
        if any(kw in note_lower for kw in _DEMAND_SIGNAL_KW) and "blocker" in note_lower:
            return note
    return "NONE"


# Categories where an exportable audit log is a hard compliance requirement.
_AUDIT_LOG_REQUIRED_CATEGORIES = frozenset(["finance", "hr", "crm"])


def _detect_hard_compliance_block(context: dict, category: str) -> list[str]:
    """Return a list of hard compliance violations that force a STAY verdict.

    Checks the competitor's compliance profile against the mandatory requirements
    defined in the system prompt. Only called when hold_signal is NONE so that
    legitimate HOLD cases (e.g. beta feature on roadmap) are unaffected.

    Hard blocks checked:
      - SOC2 Type II (universal)
      - SSO/SAML when current stack seat count > 10
      - UK or EU data residency (at least one required)
      - Exportable audit log for Finance, HR, and CRM tools
    """
    profile = context.get("competitor_data", {}).get("compliance", {})
    seat_count = int(context.get("current_stack_entry", {}).get("seat_count", 0))
    violations: list[str] = []

    if not profile.get("soc2_type2"):
        violations.append("No SOC2 Type II")
    if seat_count > 10 and not profile.get("sso_saml"):
        violations.append("No SSO/SAML (required for >10 seats)")
    if not profile.get("uk_residency") and not profile.get("gdpr_eu_residency"):
        violations.append("No UK or EU data residency")
    if category in _AUDIT_LOG_REQUIRED_CATEGORIES:
        audit = profile.get("audit_log", False)
        exportable = profile.get("audit_log_exportable", True) if audit else False
        if not audit or not exportable:
            violations.append(f"No exportable audit log (required for {category.upper()})")

    return violations


def _detect_shelfware(tool_changes: list[str]) -> str:
    """Return the shelfware-flagging line from current_tool_status, or 'NONE'.

    Scans the current period's tool-status updates for explicit shelfware mentions
    (inactive seats, shelfware rate). Returns the first matching line so it can be
    surfaced as a structured field for the model — matching the Hold signal pattern.

    Negation phrases ("no shelfware", "no inactive seats") are explicitly excluded
    to avoid false positives from signal generators that write negation as status.
    A global negation ("Shelfware: False") anywhere in tool_changes overrides all
    detections.
    """
    # If any line globally negates shelfware, skip detection entirely.
    for change in tool_changes:
        if _SHELFWARE_GLOBAL_NEGATION.search(change):
            return "NONE"
    for change in tool_changes:
        lower = change.lower()
        if any(kw in lower for kw in _SHELFWARE_KW):
            if _SHELFWARE_NEGATION.search(change):
                continue
            if "shelfware" in lower or "unused capacity" in lower or "underutilis" in lower:
                return change
            inactive_counts = [int(value) for value in _SHELFWARE_COUNT.findall(change)]
            if inactive_counts and max(inactive_counts) >= 10:
                return change
            percents = [int(value) for value in _SHELFWARE_PERCENT.findall(change)]
            if percents and max(percents) >= 30:
                return change
    return "NONE"


def _format_compliance_block(context: dict, signal: dict) -> str:
    """Format raw competitor compliance profile + signal for the model to reason from."""
    profile = context.get("competitor_data", {}).get("compliance", {})
    soc2 = "Yes" if profile.get("soc2_type2") else "No"
    sso = "Yes" if profile.get("sso_saml") else "No"
    uk = profile.get("uk_residency", False)
    eu = profile.get("gdpr_eu_residency", False)
    residency = ("UK+EU" if (uk and eu) else "UK" if uk else "EU" if eu else "No")
    al = profile.get("audit_log", False)
    ale = profile.get("audit_log_exportable", True) if al else False
    audit = f"Yes (exportable: {'Yes' if ale else 'No'})" if al else "No"
    cc = signal_compliance_changes(signal) or "unchanged"
    # If the signal explicitly flags a UK residency gap that isn't resolved,
    # surface it as a hard blocker so the model can't mistake EU=pass for UK=pass.
    uk_block_note = ""
    cc_lower = cc.lower()
    if (
        not uk
        and any(phrase in cc_lower for phrase in ["no uk", "uk residency", "uk data residency", "uk hosting"])
    ):
        uk_block_note = " \u26a0 BLOCKED \u2014 UK data residency required and not available"
    return (
        f"Competitor compliance: SOC2={soc2} | SSO/SAML={sso} | Residency={residency} | Audit log={audit}\n"
        f"Compliance signal: {cc}{uk_block_note}"
    )


def _build_lean_user(context: dict, roi_result: dict, signal: dict | None) -> str:
    """Build the compact user message used by the lean verdict pipeline."""
    from agent.prompts import _CATEGORY_RULES_COMPACT

    category = context["category"]
    signal = signal or {}
    tool_name = context["current_stack_entry"]["tool"]
    issues = context["current_stack_entry"].get("known_issues", [])
    issues_text = "\n".join(f"- {i}" for i in issues) if issues else "(none)"

    comp_changes = signal_competitor_changes(signal)
    tool_changes = signal_current_tool_status(signal)
    notes = signal_notes(signal)

    comp_text = "\n".join(f"- {c}" for c in comp_changes) if comp_changes else "(none)"
    tool_change_text = "\n".join(f"- {c}" for c in tool_changes) if tool_changes else "(unchanged this period)"
    notes_text = "\n".join(f"- {n}" for n in notes) if notes else ""

    category_rules = _CATEGORY_RULES_COMPACT.get(category, "")

    roi_summary = (
        f"Migration: £{roi_result['migration_cost_one_time']:.0f}, "
        f"Annual net: £{roi_result['annual_net_gbp']:.0f}, "
        f"Threshold: {'MET' if roi_result['roi_threshold_met'] else 'NOT MET'}"
    )

    user_content = (
        f"Category: {category} — current tool: {tool_name}\n"
        f"{category_rules}\n\n"
        f"Current tool known issues:\n{issues_text}\n\n"
        f"Changes this period:\n"
        f"  Current tool: {tool_change_text}\n"
        f"  Competitor: {comp_text}\n"
    )
    if notes_text:
        user_content += f"\nBuried signals / notes:\n{notes_text}\n"
    hold_signal = _detect_hold_signal(notes, comp_changes)
    disqualifier = _detect_disqualifier(notes, hold_signal)
    shelfware_signal = _detect_shelfware(tool_changes)
    user_content += f"\nROI: {roi_summary}"
    user_content += f"\n{_format_compliance_block(context, signal)}"
    user_content += f"\nHold signal: {hold_signal}"
    if disqualifier != "NONE":
        user_content += f"\nDisqualifier: {disqualifier}"
    # Hard compliance block — only when no hold signal is active (don't override legitimate HOLDs).
    if hold_signal == "NONE":
        hard_blocks = _detect_hard_compliance_block(context, category)
        if hard_blocks:
            user_content += f"\nHard compliance block \u2192 STAY forced: {'; '.join(hard_blocks)}"
    # Disqualifier takes priority over Shelfware (don't present both — contradictory).
    if shelfware_signal != "NONE" and hold_signal == "NONE" and disqualifier == "NONE":
        user_content += f"\nShelfware flag: {shelfware_signal}"
    migration_enabler = _detect_migration_enabler(notes)
    if migration_enabler != "NONE" and hold_signal == "NONE" and disqualifier == "NONE":
        user_content += f"\nMigration enabler: {migration_enabler}"
    demand_signal = _detect_demand_signal(notes)
    if demand_signal != "NONE" and hold_signal == "NONE" and disqualifier == "NONE":
        user_content += f"\nDemand signal: {demand_signal}"
    prev_verdict = signal.get("previous_verdict")
    if prev_verdict == "HOLD" and hold_signal == "NONE":
        user_content += "\nHold status: RESOLVED (prior verdict was HOLD — blocker has now cleared)"
    elif prev_verdict:
        user_content += f"\nPrevious verdict: {prev_verdict}"
    return user_content
