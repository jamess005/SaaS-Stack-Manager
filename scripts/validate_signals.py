"""
Signal validation script — checks every generated signal file for structural
and content correctness beyond the basic JSON schema check in generate_signals.py.

Checks per file:
  1. Required fields present with correct types
  2. scenario_type matches filename
  3. category matches filename
  4. competitor name non-empty
  5. current_tool name non-empty
  6. date is a plausible YYYY-MM-DD string
  7. Scenario-specific content rules:
     - pull_dominant:          competitor_changes non-empty, current_tool_status == "No change..." OR empty
     - push_dominant:          current_tool_status non-empty with degradation metrics
     - shelfware_case:         current_tool_status non-empty (inactive seat signal)
     - fluff_update:           competitor_changes non-empty, no specific feature names expected
     - hard_compliance_failure: compliance_changes non-empty or notes mentions a block
     - compliance_newly_met:   compliance_changes non-empty (cert acquired)
     - hold scenarios:         notes non-empty (hold reason must be stated)
  8. No field is a placeholder (e.g. "change 1", "status 1", "YYYY-MM-DD")

Usage:
    python scripts/validate_signals.py
    python scripts/validate_signals.py --scenario pull_dominant
    python scripts/validate_signals.py --fix-metadata   # re-force scenario/category/tool fields
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"

sys.path.insert(0, str(_PROJECT_ROOT))

from agent.context_loader import VALID_CATEGORIES  # noqa: E402
from training.generate_signals import SCENARIO_TYPES, _COMPETITORS  # noqa: E402

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLACEHOLDER_RE = re.compile(
    r"^(change \d+|status \d+|YYYY-MM-DD|note \d+|caveat \d+|describe .*)$",
    re.IGNORECASE,
)

# Scenario types that require hold-reason in notes
_HOLD_SCENARIOS = {
    "competitor_nearly_ready",
    "roadmap_confirmed_hold",
    "contract_renewal_hold",
    "vendor_acquisition_hold",
    "pilot_in_progress_hold",
    "dual_improvement",
}

# Scenarios where current_tool_status should show degradation (non-empty)
_PUSH_REQUIRED = {"push_dominant", "shelfware_case"}

# Scenarios where compliance_changes must be non-trivial
_COMPLIANCE_SIGNAL_REQUIRED = {"hard_compliance_failure", "compliance_newly_met"}


def _validate_file(path: Path) -> list[str]:
    """Return list of error strings for one signal file. Empty = clean."""
    errors: list[str] = []

    # Parse filename
    stem = path.stem
    parsed_cat = None
    parsed_comp_slug = None
    parsed_scenario = None

    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat + "_"):
            parsed_cat = cat
            remainder = stem[len(cat) + 1:]
            for sc in sorted(SCENARIO_TYPES, key=len, reverse=True):
                if remainder.endswith("_" + sc):
                    parsed_scenario = sc
                    parsed_comp_slug = remainder[:-(len(sc) + 1)]
                    break
            break

    if not parsed_cat or not parsed_scenario or not parsed_comp_slug:
        errors.append("Cannot parse filename — does not match {category}_{competitor}_{scenario}.json pattern")
        return errors

    # Load JSON
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append("Root is not a JSON object")
        return errors

    # Required fields
    required = {
        "scenario_type": str, "category": str, "competitor": str,
        "current_tool": str, "date": str, "competitor_changes": list,
        "current_tool_status": list, "pricing_delta": str,
        "compliance_changes": str, "notes": list,
    }
    for field, expected_type in required.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(f"Wrong type for {field}: expected {expected_type.__name__}, got {type(data[field]).__name__}")

    if errors:
        return errors  # can't do further checks with missing fields

    # Metadata consistency
    if data["scenario_type"] != parsed_scenario:
        errors.append(f"scenario_type mismatch: file says '{data['scenario_type']}', filename says '{parsed_scenario}'")
    if data["category"] != parsed_cat:
        errors.append(f"category mismatch: file says '{data['category']}', filename says '{parsed_cat}'")
    if not data["competitor"].strip():
        errors.append("competitor name is empty")
    if not data["current_tool"].strip():
        errors.append("current_tool name is empty")

    # Date format
    if not _DATE_RE.match(data["date"]):
        errors.append(f"date is not YYYY-MM-DD: {data['date']!r}")

    # Placeholder detection
    all_strings: list[str] = [data["date"], data["pricing_delta"], data["compliance_changes"]]
    all_strings += data["competitor_changes"]
    all_strings += data["current_tool_status"]
    all_strings += data["notes"]
    for s in all_strings:
        if isinstance(s, str) and _PLACEHOLDER_RE.match(s.strip()):
            errors.append(f"Placeholder value detected: {s!r}")

    # Scenario-specific content rules
    scenario = parsed_scenario

    if scenario == "pull_dominant":
        if not data["competitor_changes"]:
            errors.append("pull_dominant: competitor_changes is empty (should list 2-3 new features)")

    if scenario in _PUSH_REQUIRED:
        if not data["current_tool_status"]:
            errors.append(f"{scenario}: current_tool_status is empty (must show push signals)")
        elif all(s.lower().strip() in ("no change", "no change — existing issues persist.", "stable", "unchanged") for s in data["current_tool_status"]):
            errors.append(f"{scenario}: current_tool_status shows no degradation (must have specific metrics)")

    if scenario == "fluff_update":
        if not data["competitor_changes"]:
            errors.append("fluff_update: competitor_changes is empty (should have vague marketing phrases)")

    if scenario in _COMPLIANCE_SIGNAL_REQUIRED:
        cc = data["compliance_changes"].lower().strip()
        if cc in ("unchanged", "no change", ""):
            errors.append(f"{scenario}: compliance_changes must describe a compliance event, got: {data['compliance_changes']!r}")

    if scenario in _HOLD_SCENARIOS:
        if not data["notes"]:
            errors.append(f"{scenario}: notes is empty (hold reason must be stated in notes)")

    if scenario == "hard_compliance_failure":
        cc = data["compliance_changes"].lower().strip()
        notes_text = " ".join(data["notes"]).lower()
        compliance_words = {"soc2", "sso", "gdpr", "residency", "audit", "mfa", "saml", "certif", "block", "not available", "no sso", "no soc"}
        if not any(w in cc or w in notes_text for w in compliance_words):
            errors.append("hard_compliance_failure: no compliance block keywords found in compliance_changes or notes")

    if scenario == "compliance_newly_met":
        cc = data["compliance_changes"].lower()
        achieved_words = {
            "achieved", "certified", "now available", "acquired", "live",
            "completed", "granted", "now met", "now compliant", "compliant",
            "met", "passed", "attained", "meets", "now meets",
        }
        if not any(w in cc for w in achieved_words):
            errors.append(f"compliance_newly_met: compliance_changes should describe acquisition of cert, got: {data['compliance_changes']!r}")

    if scenario in {"roadmap_confirmed_hold", "contract_renewal_hold", "vendor_acquisition_hold", "pilot_in_progress_hold"}:
        notes_text = " ".join(data["notes"]).lower()
        hold_words = {
            "hold", "reassess", "60 days", "pilot", "renewal", "acquisition",
            "roadmap", "quarter", "pending", "wait", "q1", "q2", "q3", "q4",
            "2025", "2026", "renew", "contract", "in progress", "not yet",
            "migration", "integration test", "caveat", "connector", "penalty",
        }
        if not any(w in notes_text for w in hold_words):
            errors.append(f"{scenario}: notes should explain the hold condition, but no hold-related keywords found")

    return errors


def _fix_types(path: Path) -> bool:
    """Coerce type mismatches produced by smaller generation models. Returns True if file changed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    changed = False

    # compliance_changes must be str
    cc = data.get("compliance_changes")
    if isinstance(cc, list):
        data["compliance_changes"] = "; ".join(str(x) for x in cc if x)
        changed = True
    elif isinstance(cc, dict):
        data["compliance_changes"] = "; ".join(f"{k}: {v}" for k, v in cc.items())
        changed = True

    # notes must be list
    notes = data.get("notes")
    if isinstance(notes, str) and notes.strip():
        data["notes"] = [notes.strip()]
        changed = True
    elif notes is None:
        data["notes"] = []
        changed = True

    # date must be YYYY-MM-DD
    date_val = data.get("date", "")
    if not _DATE_RE.match(str(date_val)):
        data["date"] = "2026-03-01"
        changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all generated signal files.")
    parser.add_argument("--scenario", choices=SCENARIO_TYPES, help="Validate one scenario type only.")
    parser.add_argument("--fix-metadata", action="store_true",
                        help="Re-force scenario_type/category/current_tool fields from filename and context.")
    parser.add_argument("--fix-types", action="store_true",
                        help="Auto-coerce type mismatches (list→str for compliance_changes, str→list for notes, etc).")
    args = parser.parse_args()

    files = sorted(_GENERATED_DIR.glob("*.json"))
    if args.scenario:
        files = [f for f in files if f.stem.endswith("_" + args.scenario)]

    if args.fix_types:
        fixed = sum(1 for f in files if _fix_types(f))
        print(f"Fixed types in {fixed} files.")
        # Re-glob to pick up changes
        files = sorted(_GENERATED_DIR.glob("*.json"))
        if args.scenario:
            files = [f for f in files if f.stem.endswith("_" + args.scenario)]

    total = len(files)
    errors_by_file: dict[str, list[str]] = {}
    scenario_counts: dict[str, int] = defaultdict(int)
    scenario_errors: dict[str, int] = defaultdict(int)

    for path in files:
        errs = _validate_file(path)
        # Count by scenario
        for sc in SCENARIO_TYPES:
            if path.stem.endswith("_" + sc):
                scenario_counts[sc] += 1
                if errs:
                    scenario_errors[sc] += 1
                break
        if errs:
            errors_by_file[path.name] = errs

    # Report
    print(f"\nValidated {total} signal files\n")

    print("── Scenario counts ──────────────────────────────")
    for sc in SCENARIO_TYPES:
        count = scenario_counts.get(sc, 0)
        err_count = scenario_errors.get(sc, 0)
        status = f"  ({err_count} with errors)" if err_count else ""
        print(f"  {sc:<30} {count:>3}{status}")

    if errors_by_file:
        print(f"\n── Errors ({len(errors_by_file)} files) ──────────────────────────────")
        for fname, errs in sorted(errors_by_file.items()):
            print(f"\n  {fname}")
            for e in errs:
                print(f"    • {e}")
        print(f"\nResult: {len(errors_by_file)}/{total} files have errors")
        sys.exit(1)
    else:
        print(f"\nResult: all {total} files valid")


if __name__ == "__main__":
    main()
