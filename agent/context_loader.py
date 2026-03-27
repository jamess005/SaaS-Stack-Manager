"""
Context loader — loads scoped data files for a single category + competitor evaluation.

Loads only the files relevant to one inference call:
  - data/current_stack.json (category slice only)
  - data/usage_metrics.json (current tool entry only)
  - data/business_rules/{category}.md
  - data/business_rules/global.md
  - data/competitors/{category}/{competitor_slug}.json
  - data/company_profile.md

No model involvement. Pure file I/O returning structured dicts and strings.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"crm", "hr", "finance", "project_mgmt", "analytics"}


def parse_inbox_filename(filename: str) -> tuple[str, str]:
    """
    Extract category and competitor_slug from an inbox filename.

    Args:
        filename: e.g. "finance_ledgerflow.md" or a full path like
                  "/path/to/market_inbox/project_mgmt_flowboard.md"

    Returns:
        (category, competitor_slug) — e.g. ("finance", "ledgerflow")

    Raises:
        ValueError: if the filename stem does not start with a known category prefix.
    """
    stem = Path(filename).stem  # strip path and extension
    # Greedy prefix match — handles "project_mgmt_flowboard" correctly
    # because "project_mgmt" contains an underscore
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        prefix = cat + "_"
        if stem.startswith(prefix):
            competitor_slug = stem[len(prefix):]
            if not competitor_slug:
                raise ValueError(
                    f"Filename {filename!r} matched category {cat!r} but has no competitor slug."
                )
            return cat, competitor_slug
    raise ValueError(
        f"Cannot parse category from filename: {filename!r}. "
        f"Expected format: {{category}}_{{competitor}}.md "
        f"where category is one of {sorted(VALID_CATEGORIES)}."
    )


def load_context(category: str, competitor_slug: str, data_root: Path) -> dict:
    """
    Load all scoped context files for one category + competitor evaluation.

    Args:
        category: One of the VALID_CATEGORIES strings.
        competitor_slug: Filename stem of the competitor JSON, e.g. "ledgerflow".
        data_root: Absolute path to the data/ directory.

    Returns:
        {
            "category": str,
            "competitor_slug": str,
            "current_stack_entry": dict,      # single category entry from current_stack.json
            "usage_metrics_entry": dict|None, # tool entry from usage_metrics.json (None if absent)
            "business_rules_text": str,       # contents of data/business_rules/{category}.md
            "global_rules_text": str,         # contents of data/business_rules/global.md
            "competitor_data": dict,          # parsed competitor JSON
            "company_profile_text": str,      # contents of data/company_profile.md
        }

    Raises:
        ValueError: if category is not in VALID_CATEGORIES.
        FileNotFoundError: if the competitor JSON does not exist.
        KeyError: if current_stack.json is missing the requested category key.
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category {category!r}. Must be one of {sorted(VALID_CATEGORIES)}."
        )

    data_root = Path(data_root)

    # Load current_stack.json and extract only the relevant category entry
    current_stack_path = data_root / "current_stack.json"
    with current_stack_path.open(encoding="utf-8") as f:
        full_stack = json.load(f)
    if category not in full_stack:
        raise KeyError(
            f"Category {category!r} not found in {current_stack_path}. "
            f"Available keys: {list(full_stack.keys())}"
        )
    current_stack_entry = full_stack[category]

    # Load usage_metrics.json — keyed by tool name, not category
    tool_name = current_stack_entry["tool"]
    usage_metrics_path = data_root / "usage_metrics.json"
    usage_metrics_entry = None
    if usage_metrics_path.exists():
        with usage_metrics_path.open(encoding="utf-8") as f:
            all_metrics = json.load(f)
        if tool_name in all_metrics:
            usage_metrics_entry = all_metrics[tool_name]
        else:
            logger.warning(
                "No usage metrics entry for tool %r in %s. Proceeding without metrics.",
                tool_name,
                usage_metrics_path,
            )
    else:
        logger.warning("usage_metrics.json not found at %s.", usage_metrics_path)

    # Load category-specific business rules
    rules_path = data_root / "business_rules" / f"{category}.md"
    business_rules_text = rules_path.read_text(encoding="utf-8")

    # Load global business rules (always included)
    global_rules_path = data_root / "business_rules" / "global.md"
    global_rules_text = global_rules_path.read_text(encoding="utf-8")

    # Load competitor JSON — raise clearly if missing
    competitor_path = data_root / "competitors" / category / f"{competitor_slug}.json"
    if not competitor_path.exists():
        available = sorted(p.stem for p in (data_root / "competitors" / category).glob("*.json"))
        raise FileNotFoundError(
            f"Competitor file not found: {competitor_path}. "
            f"Available competitors for {category!r}: {available}"
        )
    with competitor_path.open(encoding="utf-8") as f:
        competitor_data = json.load(f)

    # Load company profile (small, always relevant)
    company_profile_path = data_root / "company_profile.md"
    company_profile_text = company_profile_path.read_text(encoding="utf-8")

    return {
        "category": category,
        "competitor_slug": competitor_slug,
        "current_stack_entry": current_stack_entry,
        "usage_metrics_entry": usage_metrics_entry,
        "business_rules_text": business_rules_text,
        "global_rules_text": global_rules_text,
        "competitor_data": competitor_data,
        "company_profile_text": company_profile_text,
    }
