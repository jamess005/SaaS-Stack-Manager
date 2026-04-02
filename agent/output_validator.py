"""
Output validator — checks verdict memos for structural completeness and citation presence.

Pure string operations. No file I/O, no model involvement, no external dependencies.

Enforces:
  1. All 9 required section headers present
  2. VERDICT value is SWITCH, STAY, or HOLD
  3. At least one double-quoted citation in the EVIDENCE section
  4. PUSH SIGNALS has at least one bullet
  5. PULL SIGNALS has at least one bullet
  6. FINANCIAL ANALYSIS contains the ROI threshold line
  7. HOLD verdict includes a REASSESS CONDITION line
"""

import re

# Required section headers (matched case-insensitively at line start)
_REQUIRED_SECTIONS = [
    "CATEGORY",
    "CURRENT TOOL",
    "COMPETITOR",
    "DATE",
    "PUSH SIGNALS",
    "PULL SIGNALS",
    "FINANCIAL ANALYSIS",
    "VERDICT",
    "EVIDENCE",
]

_VALID_VERDICTS = {"SWITCH", "STAY", "HOLD"}

# Matches a line starting with the section header followed by a colon
_section_re = {s: re.compile(rf"^\s*{re.escape(s)}\s*:", re.IGNORECASE | re.MULTILINE) for s in _REQUIRED_SECTIONS}

# Matches the VERDICT line and captures the verdict word
_verdict_line_re = re.compile(r"^\s*VERDICT\s*:\s*(\w+)", re.IGNORECASE | re.MULTILINE)

# Matches a quoted string "..." (at least one character inside)
_quoted_string_re = re.compile(r'"[^"\n]+"')

# Matches a bullet point "  - " style
_bullet_re = re.compile(r"^\s{2,}-\s", re.MULTILINE)

# Matches the ROI threshold met line
_roi_threshold_re = re.compile(r"ROI threshold met\s*:", re.IGNORECASE)

# Matches the REASSESS CONDITION line
_reassess_re = re.compile(r"^\s*REASSESS CONDITION\s*:", re.IGNORECASE | re.MULTILINE)


def _get_section_text(memo_text: str, section_name: str) -> str:
    """
    Extract the text belonging to a section, from its header to the next header or end.
    Splits the memo on uppercase section header lines (e.g. 'VERDICT:', 'EVIDENCE:').
    Returns an empty string if the section is not found.
    """
    # Split on lines that look like a section header: ALL-CAPS words followed by colon
    segments = re.split(r"\n(?=[A-Z][A-Z\s]+\s*:)", memo_text)
    for segment in segments:
        first_line = segment.split("\n")[0]
        if re.match(rf"^\s*{re.escape(section_name)}\s*:", first_line, re.IGNORECASE):
            parts = segment.split("\n", 1)
            return parts[1] if len(parts) > 1 else ""
    return ""


def validate_verdict(memo_text: str) -> tuple[bool, list[str]]:
    """
    Validate a verdict memo string.

    Args:
        memo_text: Raw text output from Pass 2.

    Returns:
        (is_valid, errors) where errors is an empty list if is_valid is True.
    """
    errors: list[str] = []

    # 1. Check all required section headers are present
    for section in _REQUIRED_SECTIONS:
        if not _section_re[section].search(memo_text):
            errors.append(f"Missing required section: {section}:")

    # 2. Check VERDICT value is valid
    verdict_match = _verdict_line_re.search(memo_text)
    if verdict_match:
        verdict_word = verdict_match.group(1).upper()
        if verdict_word not in _VALID_VERDICTS:
            errors.append(
                f"Invalid verdict value: {verdict_word!r}. Must be one of {sorted(_VALID_VERDICTS)}."
            )
    # (missing VERDICT section already caught above)

    # 3. Check at least one quoted citation in EVIDENCE section
    evidence_text = _get_section_text(memo_text, "EVIDENCE")
    # Only check citations if EVIDENCE section is present (absence already caught above)
    if _section_re["EVIDENCE"].search(memo_text) and not _quoted_string_re.search(evidence_text):
        errors.append(
            'Missing quoted citation in EVIDENCE section. '
            'At least one "exact quote" is required (Quote-to-Claim rule).'
        )

    # 4. Check PUSH SIGNALS has at least one bullet
    push_text = _get_section_text(memo_text, "PUSH SIGNALS")
    if push_text and not _bullet_re.search(push_text):
        errors.append("PUSH SIGNALS section has no bullet points (expected '  - ' entries).")

    # 5. Check PULL SIGNALS has at least one bullet
    pull_text = _get_section_text(memo_text, "PULL SIGNALS")
    if pull_text and not _bullet_re.search(pull_text):
        errors.append("PULL SIGNALS section has no bullet points (expected '  - ' entries).")

    # 6. Check FINANCIAL ANALYSIS contains ROI threshold line
    financial_text = _get_section_text(memo_text, "FINANCIAL ANALYSIS")
    if financial_text and not _roi_threshold_re.search(financial_text):
        errors.append("FINANCIAL ANALYSIS section is missing the 'ROI threshold met:' line.")

    # 7. HOLD verdict requires REASSESS CONDITION
    if verdict_match and verdict_match.group(1).upper() == "HOLD":
        if not _reassess_re.search(memo_text):
            errors.append(
                "HOLD verdict requires a 'REASSESS CONDITION:' line stating when to re-evaluate."
            )

    return len(errors) == 0, errors


def extract_verdict_class(memo_text: str) -> str | None:
    """
    Extract the verdict class string from a memo.

    Returns:
        "SWITCH", "STAY", "HOLD", or None if not found or value is invalid.
    """
    match = _verdict_line_re.search(memo_text)
    if not match:
        return None
    word = match.group(1).upper()
    return word if word in _VALID_VERDICTS else None


def extract_hold_metadata(memo_text: str) -> dict | None:
    """
    For HOLD verdicts: extract fields needed for hold_register.json.

    Looks for lines:
      REASSESS CONDITION: <text>
      REVIEW BY: <YYYY-MM-DD>

    Also extracts CURRENT TOOL and COMPETITOR from the memo header.

    Returns:
        dict with keys: hold_reason, reassess_condition, review_by,
                        category, current_tool, competitor
        or None if any required field cannot be parsed.
    """
    if extract_verdict_class(memo_text) != "HOLD":
        return None

    def _extract_line_value(pattern: str) -> str | None:
        match = re.search(pattern, memo_text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None

    reassess = _extract_line_value(r"^\s*REASSESS CONDITION\s*:\s*(.+)$")
    review_by = _extract_line_value(r"^\s*REVIEW BY\s*:\s*(.+)$")
    category = _extract_line_value(r"^\s*CATEGORY\s*:\s*(.+)$")
    current_tool_raw = _extract_line_value(r"^\s*CURRENT TOOL\s*:\s*(.+)$")
    competitor_raw = _extract_line_value(r"^\s*COMPETITOR\s*:\s*(.+)$")

    # Strip cost annotation from tool names, e.g. "VaultLedger (£420/mo)" → "VaultLedger"
    def _strip_cost(s: str | None) -> str | None:
        if s is None:
            return None
        return re.sub(r"\s*\(.*\)", "", s).strip()

    current_tool = _strip_cost(current_tool_raw)
    competitor = _strip_cost(competitor_raw)

    # Build a brief hold_reason from push signals (first HIGH signal, or generic)
    push_text = _get_section_text(memo_text, "PUSH SIGNALS")
    hold_reason = None
    if push_text:
        first_bullet = re.search(r"^\s{2,}-\s(.+)$", push_text, re.MULTILINE)
        if first_bullet:
            hold_reason = first_bullet.group(1).strip()
    if not hold_reason:
        hold_reason = "See push signals in verdict memo."

    # All fields must be present for a valid hold entry
    if not all([reassess, review_by, category, current_tool, competitor]):
        return None

    return {
        "category": category.lower() if category else None,
        "current_tool": current_tool,
        "competitor": competitor,
        "hold_reason": hold_reason,
        "reassess_condition": reassess,
        "review_by": review_by,
    }


_lean_analysis_re = re.compile(r"^\s*ANALYSIS\s*:", re.IGNORECASE | re.MULTILINE)


def validate_lean_output(text: str) -> tuple[bool, list[str]]:
    """
    Validate lean pipeline output (ANALYSIS + VERDICT lines only).

    Returns (is_valid, errors).
    """
    errors: list[str] = []
    if not _lean_analysis_re.search(text):
        errors.append("Missing ANALYSIS: line")
    verdict_match = _verdict_line_re.search(text)
    if not verdict_match:
        errors.append("Missing VERDICT: line")
    elif verdict_match.group(1).upper() not in _VALID_VERDICTS:
        errors.append(f"Invalid verdict: {verdict_match.group(1)!r}")
    return len(errors) == 0, errors
