"""Generate market_inbox trigger files for all competitors that don't already have one.

Reads competitor JSON files and produces a realistic inbox .md file for each,
using the feature data, pricing, and compliance info from the JSON.
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_COMPETITORS_DIR = _PROJECT_ROOT / "data" / "competitors"
_INBOX_DIR = _PROJECT_ROOT / "market_inbox"


def _inbox_filename(category: str, slug: str) -> str:
    return f"{category}_{slug}.md"


def _generate_inbox(category: str, slug: str, data: dict) -> str:
    name = data.get("name", slug)
    cost = data.get("monthly_cost_gbp", "N/A")
    pricing_model = data.get("pricing_model", "per_seat")
    seat_count = data.get("seat_count_assumed")
    cost_per_seat = data.get("cost_per_seat_gbp")
    tiers = data.get("pricing_tiers", [])
    features = data.get("features", [])
    limitations = data.get("known_limitations", [])
    integrations = data.get("integration_compatibility", [])
    trajectory = data.get("recent_trajectory", "stable")
    compliance = data.get("compliance", {})
    last_updated = data.get("last_updated", "2026-04-01")

    # Build pricing section
    pricing_lines = [f"- **Base**: £{cost}/mo"]
    if pricing_model == "per_seat" and cost_per_seat:
        pricing_lines.append(f"- **Per seat**: £{cost_per_seat}/seat/mo")
    if seat_count:
        pricing_lines.append(f"- **Assumed seats**: {seat_count}")
    for tier in tiers:
        tier_name = tier.get("name", "Tier")
        tier_cost = tier.get("monthly_cost_gbp", "N/A")
        tier_note = tier.get("note", "")
        line = f"- **{tier_name}**: £{tier_cost}/mo"
        if tier_note:
            line += f" ({tier_note})"
        pricing_lines.append(line)
    pricing_block = "\n".join(pricing_lines)

    # Features
    feature_lines = "\n".join(f"- {f}" for f in features) if features else "- No notable features listed"

    # Limitations
    limit_lines = "\n".join(f"- ⚠️ {l}" for l in limitations) if limitations else "- None noted"

    # Integrations
    int_lines = ", ".join(integrations) if integrations else "None listed"

    # Compliance
    sso = compliance.get("sso")
    soc2 = compliance.get("soc2_type2")
    gdpr = compliance.get("gdpr")
    residency = compliance.get("data_residency_uk")

    def _flag(val):
        if val is True:
            return "✅ Yes"
        if val is False:
            return "❌ No"
        return "⚠️ Unknown"

    compliance_block = f"""- **SOC2 Type II**: {_flag(soc2)}
- **GDPR compliant**: {_flag(gdpr)}
- **UK data residency**: {_flag(residency)}
- **SSO (SAML 2.0)**: {_flag(sso)}"""

    # Trajectory description
    trajectory_desc = trajectory.replace("_", " ").title()

    return f"""# {name} — Product Update (April 2026)

**Category**: {category}
**Last updated**: {last_updated}
**Market trajectory**: {trajectory_desc}

## PRICING

{pricing_block}

## CURRENT FEATURES

{feature_lines}

## KNOWN LIMITATIONS

{limit_lines}

## INTEGRATIONS

{int_lines}

## COMPLIANCE

{compliance_block}

## SUPPORT

- Standard SLA: business-hours support
- Documentation and knowledge base available

## NOTE

This inbox file was auto-generated from competitor registry data as of {last_updated}.
Evaluate against the current stack tool for the {category} category.
"""


def main():
    _INBOX_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0

    for cat_dir in sorted(_COMPETITORS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for comp_file in sorted(cat_dir.glob("*.json")):
            slug = comp_file.stem
            inbox_file = _INBOX_DIR / _inbox_filename(category, slug)
            if inbox_file.exists():
                print(f"SKIP (exists): {inbox_file.name}")
                skipped += 1
                continue
            try:
                data = json.loads(comp_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"ERROR reading {comp_file}: {e}")
                continue

            content = _generate_inbox(category, slug, data)
            inbox_file.write_text(content, encoding="utf-8")
            print(f"CREATED: {inbox_file.name}")
            generated += 1

    print(f"\nDone: {generated} created, {skipped} skipped")


if __name__ == "__main__":
    main()
