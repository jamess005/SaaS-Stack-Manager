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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Local model paths ──────────────────────────────────────────────────────────
# Absolute path to the locally cached Llama 3.1 8B Instruct snapshot.
# Used by generate_signals.py and generate_traces.py as the teacher model.
LOCAL_LLAMA_8B = (
    "/home/james/ml-proj/models/llama-3.1-8b-instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

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


def load_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_path: str | None = None,
    quantize_bits: int = 4,
) -> tuple[Any, Any]:
    """
    Load tokenizer and quantised model for inference.

    Args:
        model_name: HuggingFace model ID. Ignored if model_path is provided.
        model_path: Local filesystem path to a model snapshot directory.
                    Takes priority over model_name. Use LOCAL_LLAMA_8B for the
                    locally cached Llama 3.1 8B teacher model.
        quantize_bits: Quantisation level. 4 = NF4 4-bit (~4-5GB VRAM).
                       8 = INT8 (~8-10GB VRAM).

    Returns:
        (tokenizer, model) — both None in dry-run mode.

    ROCm: device_map="auto" triggers HIP kernel dispatch failures on ROCm 7.x.
          On ROCm, explicit device_map={{"":0}} is used instead.
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-untyped]

    source = model_path or model_name
    logger.info("Loading model: %s (quantize_bits=%d)", source, quantize_bits)

    # ROCm (AMD): device_map="auto" triggers HIP kernel dispatch failures on
    # ROCm 7.x. Use explicit device_map={"": 0} for all loading paths.
    # 4-bit NF4 via bitsandbytes works on ROCm with explicit device placement.
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_rocm:
        logger.info(
            "ROCm backend detected (HIP %s) — using explicit device_map.",
            torch.version.hip,
        )

    if quantize_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # On ROCm, always use explicit device placement to avoid accelerate dispatch.
    device_map = {"": 0} if is_rocm else "auto"

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )

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
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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
    raw_output = _generate(tokenizer, model, messages, max_new_tokens=768, temperature=0.3)
    logger.debug("Pass 2 raw output (%d chars).", len(raw_output))
    return raw_output.strip()
