"""
Ecosystem validator — sanity-checks all JSON and Markdown files in data/.

Run from project root:
    python scripts/validate_ecosystem.py

Checks:
  - current_stack.json has all 5 categories with required keys
  - usage_metrics.json has an entry for each tool in current_stack.json
  - All 15 competitor JSON files exist and have the required schema
  - All 6 business rule files exist (global.md + 5 category files)

Exits with code 1 if any errors are found.
"""

import json
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table

    _USE_RICH = True
except ImportError:
    _USE_RICH = False

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"

_REQUIRED_CATEGORIES = {"crm", "hr", "finance", "project_mgmt", "analytics"}

_REQUIRED_STACK_KEYS = {"tool", "monthly_cost_gbp", "seat_count", "known_issues", "integrations", "positive_signals"}

_REQUIRED_COMPETITOR_KEYS = {
    "name",
    "category",
    "monthly_cost_gbp",
    "compliance",
    "features",
    "known_limitations",
    "integration_compatibility",
}

_REQUIRED_COMPLIANCE_KEYS = {"soc2_type2", "gdpr_eu_residency", "sso_saml", "audit_log"}

_EXPECTED_RULES_FILES = {"global.md", "crm.md", "hr.md", "finance.md", "project_mgmt.md", "analytics.md"}

# Expected competitor slugs per category
_EXPECTED_COMPETITORS: dict[str, set[str]] = {
    "crm": {"pipelineiq", "closerhub", "velocitycrm", "clientpulse", "dealstream"},
    "hr": {"workforge", "teamledger_hr", "hrnest", "crewplan", "shiftcore"},
    "finance": {"ledgerflow", "novapay", "clearbooks_pro", "exactspend", "paytrek"},
    "project_mgmt": {"flowboard", "sprintdesk", "teamsync_projects", "projectaxis", "sprintloop"},
    "analytics": {"datalens", "clearview_analytics", "pulsemetrics", "metricflux", "prism_bi"},
}


def _pass(msg: str) -> None:
    if _USE_RICH:
        Console().print(f"[green]PASS[/green] {msg}")
    else:
        print(f"PASS  {msg}")


def _fail(msg: str) -> None:
    if _USE_RICH:
        Console().print(f"[red]FAIL[/red] {msg}")
    else:
        print(f"FAIL  {msg}")


def validate_current_stack(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"{path} does not exist."]
    with path.open(encoding="utf-8") as f:
        stack = json.load(f)
    for cat in _REQUIRED_CATEGORIES:
        if cat not in stack:
            errors.append(f"current_stack.json missing category: {cat!r}")
            continue
        entry = stack[cat]
        missing = _REQUIRED_STACK_KEYS - set(entry.keys())
        if missing:
            errors.append(f"current_stack.json [{cat}] missing keys: {sorted(missing)}")
    return errors


def validate_usage_metrics(path: Path, current_stack: dict) -> list[str]:
    errors = []
    if not path.exists():
        return [f"{path} does not exist."]
    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    for cat, entry in current_stack.items():
        tool_name = entry.get("tool")
        if tool_name and tool_name not in metrics:
            errors.append(f"usage_metrics.json missing entry for tool: {tool_name!r} (category: {cat})")
    return errors


def validate_competitor_json(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"{path} does not exist."]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    missing = _REQUIRED_COMPETITOR_KEYS - set(data.keys())
    if missing:
        errors.append(f"{path.name} missing keys: {sorted(missing)}")
    compliance = data.get("compliance", {})
    if isinstance(compliance, dict):
        missing_compliance = _REQUIRED_COMPLIANCE_KEYS - set(compliance.keys())
        if missing_compliance:
            errors.append(f"{path.name} compliance block missing keys: {sorted(missing_compliance)}")
    else:
        errors.append(f"{path.name} compliance field is not a dict.")
    if not isinstance(data.get("features"), list) or len(data["features"]) == 0:
        errors.append(f"{path.name} features list is empty or missing.")
    return errors


def validate_business_rules(rules_dir: Path) -> list[str]:
    errors = []
    for filename in _EXPECTED_RULES_FILES:
        p = rules_dir / filename
        if not p.exists():
            errors.append(f"Missing business rules file: {p}")
        elif p.stat().st_size == 0:
            errors.append(f"Business rules file is empty: {p}")
    return errors


def main() -> None:
    all_errors: list[tuple[str, str]] = []  # (label, error_message)

    # Load current_stack.json once for reuse
    stack_path = _DATA_ROOT / "current_stack.json"
    current_stack = {}
    if stack_path.exists():
        with stack_path.open(encoding="utf-8") as f:
            current_stack = json.load(f)

    # Validate current_stack.json
    errs = validate_current_stack(stack_path)
    label = str(stack_path.relative_to(_PROJECT_ROOT))
    if errs:
        for e in errs:
            all_errors.append((label, e))
        _fail(label)
    else:
        _pass(label)

    # Validate usage_metrics.json
    metrics_path = _DATA_ROOT / "usage_metrics.json"
    errs = validate_usage_metrics(metrics_path, current_stack)
    label = str(metrics_path.relative_to(_PROJECT_ROOT))
    if errs:
        for e in errs:
            all_errors.append((label, e))
        _fail(label)
    else:
        _pass(label)

    # Validate business rules
    rules_dir = _DATA_ROOT / "business_rules"
    errs = validate_business_rules(rules_dir)
    label = str(rules_dir.relative_to(_PROJECT_ROOT))
    if errs:
        for e in errs:
            all_errors.append((label, e))
        _fail(label)
    else:
        _pass(label)

    # Validate competitor JSON files
    for category, slugs in sorted(_EXPECTED_COMPETITORS.items()):
        for slug in sorted(slugs):
            comp_path = _DATA_ROOT / "competitors" / category / f"{slug}.json"
            errs = validate_competitor_json(comp_path)
            label = str(comp_path.relative_to(_PROJECT_ROOT))
            if errs:
                for e in errs:
                    all_errors.append((label, e))
                _fail(f"{label}")
                for e in errs:
                    print(f"       {e}")
            else:
                _pass(label)

    # Summary
    total_checks = 2 + 1 + sum(len(v) for v in _EXPECTED_COMPETITORS.values())
    failed = len(set(label for label, _ in all_errors))
    passed = total_checks - failed

    print(f"\nSummary: {passed} passed, {failed} failed")
    if all_errors:
        print("\nErrors:")
        for label, msg in all_errors:
            print(f"  [{label}] {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
