"""
Reward functions for GRPO BCR training.

parse_numbered_verdicts — extracts [N] VERDICT tags from BCR completions.
bcr_reward_fn          — reward function signature expected by GRPOTrainer.reward_funcs.

TRL (0.29+) calls reward functions as:
    fn(prompts, completions, completion_ids, **dataset_columns) -> list[float]

For conversational prompts each completion is [{"role": "assistant", "content": "..."}].
The reward function returns one float per completion (range 0.0–1.0).
"""

import re

_VALID_VERDICTS = {"SWITCH", "STAY", "HOLD"}

# Matches VERDICT (with any non-alpha chars between colon and word, e.g. markdown bold **).
# Order-based: we extract all occurrences and assign them to positions 0..n-1.
# This is robust to the model using **VERDICT:** or VERDICT: or VERDICT — SWITCH etc.
_VERDICT_ORDER_RE = re.compile(
    r"VERDICT\b[^A-Z\n]{0,10}(SWITCH|STAY|HOLD)",
    re.IGNORECASE,
)

# Also try [N]-tagged format as a secondary pass for exact placement.
_NUMBERED_VERDICT_RE = re.compile(
    r"\[(\d+)\][^[]*?VERDICT\b[^A-Z\n]{0,10}(SWITCH|STAY|HOLD)",
    re.IGNORECASE | re.DOTALL,
)


def parse_numbered_verdicts(text: str, n: int) -> list[str | None]:
    """
    Extract per-case verdicts from BCR output. Two strategies:

    1. Try [N]-tagged extraction first (exact placement).
    2. Fall back to order-based extraction (Nth VERDICT occurrence = case N).
       Robust to the model using alternative formatting (markdown bold, etc.).

    Args:
        text: Model completion text.
        n:    Expected number of cases (e.g. 4 for BCR N=4).

    Returns:
        List of n verdict strings ("SWITCH"/"STAY"/"HOLD") or None where missing.
    """
    # Strategy 1: [N]-tagged
    tagged: dict[int, str] = {}
    for match in _NUMBERED_VERDICT_RE.finditer(text):
        idx = int(match.group(1)) - 1
        word = match.group(2).upper()
        if 0 <= idx < n and word in _VALID_VERDICTS:
            tagged[idx] = word

    if len(tagged) == n:
        return [tagged.get(i) for i in range(n)]

    # Strategy 2: order-based — extract all VERDICT occurrences in sequence
    ordered = [m.group(1).upper() for m in _VERDICT_ORDER_RE.finditer(text)
               if m.group(1).upper() in _VALID_VERDICTS]

    results: list[str | None] = [None] * n
    for i, word in enumerate(ordered[:n]):
        results[i] = word

    # If tagged had partial hits, prefer them over order-based for those slots
    for idx, word in tagged.items():
        results[idx] = word

    return results


def bcr_reward_fn(
    prompts: list,
    completions: list,
    expected_verdicts: list[list[str]],
    **kwargs,
) -> list[float]:
    """
    Per-completion reward for BCR GRPO training.

    Mean fraction of per-instance verdicts that match the canonical answer.
    A batch where all 4 cases are correct returns 1.0; all wrong returns 0.0.

    Args:
        prompts:           Prompt message lists (unused — included for TRL signature).
        completions:       One per generation. For conversational models each element is
                           [{"role": "assistant", "content": "..."}]; for non-conversational
                           it is a plain string.
        expected_verdicts: One list of n canonical verdicts per completion
                           (TRL repeats the dataset row for each generation, so these are
                           identical across the G completions for one prompt).

    Returns:
        List of float rewards, one per completion.
    """
    rewards = []
    for completion, expected in zip(completions, expected_verdicts):
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        else:
            text = str(completion)

        n = len(expected)
        parsed = parse_numbered_verdicts(text, n)
        correct = sum(
            1 for p, e in zip(parsed, expected) if p is not None and p == e
        )
        rewards.append(correct / n if n > 0 else 0.0)
    return rewards
