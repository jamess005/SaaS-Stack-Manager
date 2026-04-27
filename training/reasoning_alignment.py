from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

Polarity = Literal["positive", "negative", "pending", "neutral"]

_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "linkedin_enrichment": ("linkedin", "enrich"),
    "email_sequences": ("email sequence", "email sequences", "sequence builder", "sequence steps", "unlimited email"),
    "api_sync": ("api", "sync", "connector", "middleware", "rate limit", "webhook"),
    "mobile": ("mobile", "ios", "android", "phones", "tablets"),
    "embedded_reporting": ("embedded reporting", "self-serve embedded reporting", "white-label embedded reporting"),
    "dashboard_reporting": ("dashboard", "dashboards", "reporting", "reports", "analytics"),
    "data_lineage": ("data lineage", "lineage"),
    "anomaly_detection": ("anomaly detection", "anomaly", "natural language query", "nlq"),
    "performance_reviews": ("performance review", "continuous feedback"),
    "surveys": ("enps", "pulse survey", "survey tooling"),
    "onboarding": ("onboarding", "new hire", "workflow template", "workflow templates"),
    "payroll": ("payaxis", "payroll"),
    "gdpr_erasure": ("gdpr", "right-to-erasure", "erasure"),
    "time_tracking": ("time-tracking", "time tracking", "chronolog"),
    "client_portal": ("client portal", "deliverables", "comment", "approval"),
    "gantt": ("gantt", "dependency auto-update", "dependency changes"),
    "capacity_planning": ("capacity planning", "utilisation", "utilization"),
    "slack": ("slack", "zapier"),
    "multi_currency": ("multi-currency", "currencies", "currency", "eur invoices", "eur native", "invoicing"),
    "reconciliation": ("reconciliation", "month-end", "dataset", "datasets", "rows"),
    "ifrs15": ("ifrs 15", "revenue recognition"),
    "bank_feed": ("barclays", "bank feed", "overnight sync"),
    "audit_export": ("audit trail", "audit log", "digital signature", "pdf"),
    "support_sla": ("24/7", "sla"),
    "shelfware": ("inactive seats", "unused capacity", "underutilised", "underutilized", "per-active-user pricing", "wasted seats"),
    "pricing": ("price", "pricing", "tier", "cost", "per-user", "per active user"),
    "compliance_cert": ("soc2", "soc 2", "iso27001", "iso 27001", "sso", "saml", "residency", "data residency"),
}

_GENERIC_TOKENS = {
    "active", "added", "advanced", "all", "analytics", "annual", "app", "basic", "better",
    "bridge", "business", "capability", "change", "changes", "client", "clients", "company",
    "concrete", "current", "dashboard", "dashboards", "data", "deliver", "delivers", "enterprise",
    "feature", "features", "functionality", "ga", "general", "generally", "high", "improved",
    "integration", "management", "module", "native", "new", "now", "only", "platform", "pricing",
    "project", "real", "reporting", "review", "service", "shipped", "signal", "status", "support",
    "system", "tool", "tools", "tracking", "update", "updated", "user", "users", "via", "view",
    "workflows",
}

_POSITIVE_OVERRIDES = (
    "no major incidents",
    "no need for manual",
    "no longer requires",
    "updated to include",
    "now generally available",
    "generally available",
    "ga as of",
    "eliminates wasted seats",
)

_POSITIVE_PATTERNS = (
    " added ",
    " introduced ",
    " reintroduced ",
    " available",
    " shipped",
    " launched",
    " native ",
    " built-in",
    " unlimited",
    " real-time",
    " digital signature",
    " continuous feedback",
    " commenting",
    " approval",
    " auto-update",
    " auto-updates",
    " capacity planning added",
    " certified",
    " certification",
    " compliant",
    " per-active-user pricing",
    " price reduction",
    " stable",
    " improved",
    " enhanced",
    " faster",
    " live",
)

_PENDING_PATTERNS = (
    " in beta",
    " beta ",
    " preview",
    " roadmap",
    " early access",
    " in progress",
    " scheduled for",
    " expected ga",
    " ga date tbd",
    " q3 2025",
    " q4 2025",
)

_NEGATIVE_PATTERNS = (
    " without ",
    " missing",
    " lacks",
    " limited",
    " read-only",
    " manual",
    " requires",
    " deprecated",
    " unstable",
    " crash",
    " csv only",
    " only overnight",
    " workaround",
    " unresolved",
    " persists",
    " non-functional",
    " maintenance-only",
    " costs extra",
    " no new features",
)


@dataclass(frozen=True)
class Match:
    issue: str
    change: str
    score: int


@dataclass(frozen=True)
class SemanticView:
    primary_issue: str | None
    primary_pull: str | None
    primary_tool_change: str | None
    positive_pull: Match | None
    pending_pull: Match | None
    positive_tool: Match | None
    negative_tool: Match | None
    has_positive_pull: bool
    has_pending_pull: bool
    has_positive_tool: bool
    has_negative_tool: bool


@dataclass(frozen=True)
class _Signal:
    text: str
    topics: frozenset[str]
    tokens: frozenset[str]
    polarity: Polarity


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> frozenset[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _clean(text).lower())
        if len(token) > 2 and token not in _GENERIC_TOKENS
    }
    return frozenset(tokens)


def _topics(text: str) -> frozenset[str]:
    lower = _clean(text).lower()
    topics = {
        topic
        for topic, patterns in _TOPIC_PATTERNS.items()
        if any(pattern in lower for pattern in patterns)
    }
    return frozenset(topics)


def classify_change_polarity(text: str, source: Literal["issue", "competitor", "tool"]) -> Polarity:
    lower = f" {_clean(text).lower()} "

    if source == "issue":
        return "negative"

    if any(phrase in lower for phrase in _PENDING_PATTERNS):
        return "pending"

    positive_score = 0
    negative_score = 0
    topic_hits = bool(_topics(text))

    if any(phrase in lower for phrase in _POSITIVE_OVERRIDES):
        positive_score += 3

    positive_score += sum(1 for phrase in _POSITIVE_PATTERNS if phrase in lower)
    negative_score += sum(1 for phrase in _NEGATIVE_PATTERNS if phrase in lower)

    if lower.startswith(" no ") and not any(phrase in lower for phrase in _POSITIVE_OVERRIDES):
        negative_score += 2

    if source == "competitor":
        if positive_score == 0 and negative_score == 0:
            return "positive" if topic_hits else "neutral"
        if negative_score >= positive_score and negative_score > 0:
            return "negative"
        if positive_score > 0:
            return "positive"
        return "neutral"

    if positive_score > negative_score and positive_score > 0:
        return "positive"
    if negative_score > positive_score and negative_score > 0:
        return "negative"
    if positive_score > 0 and negative_score == 0:
        return "positive"
    return "neutral"


def _build_entries(items: Sequence[str], source: Literal["issue", "competitor", "tool"]) -> list[_Signal]:
    entries: list[_Signal] = []
    for item in items:
        text = _clean(item)
        if not text:
            continue
        entries.append(
            _Signal(
                text=text,
                topics=_topics(text),
                tokens=_tokenize(text),
                polarity=classify_change_polarity(text, source),
            )
        )
    return entries


def _match_score(issue: _Signal, change: _Signal) -> int:
    topic_overlap = issue.topics & change.topics
    token_overlap = issue.tokens & change.tokens

    score = len(topic_overlap) * 10
    if len(token_overlap) >= 2:
        score += len(token_overlap)

    if topic_overlap:
        score += 2

    return score


def _best_match(issues: Sequence[_Signal], changes: Sequence[_Signal], allowed: set[Polarity]) -> Match | None:
    best: Match | None = None
    for issue in issues:
        for change in changes:
            if change.polarity not in allowed:
                continue
            score = _match_score(issue, change)
            if score <= 0:
                continue
            candidate = Match(issue=issue.text, change=change.text, score=score)
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def _first_by_polarity(changes: Sequence[_Signal], allowed: set[Polarity]) -> str | None:
    for change in changes:
        if change.polarity in allowed:
            return change.text
    return None


def build_semantic_view(
    issues: Sequence[str],
    competitor_changes: Sequence[str],
    tool_changes: Sequence[str],
) -> SemanticView:
    issue_entries = _build_entries(issues, "issue")
    competitor_entries = _build_entries(competitor_changes, "competitor")
    tool_entries = _build_entries(tool_changes, "tool")

    positive_pull = _best_match(issue_entries, competitor_entries, {"positive"})
    pending_pull = _best_match(issue_entries, competitor_entries, {"pending"})
    positive_tool = _best_match(issue_entries, tool_entries, {"positive"})
    negative_tool = _best_match(issue_entries, tool_entries, {"negative"})

    primary_issue = None
    for match in (positive_pull, pending_pull, negative_tool, positive_tool):
        if match is not None:
            primary_issue = match.issue
            break
    if primary_issue is None:
        primary_issue = issue_entries[0].text if issue_entries else None

    primary_pull = None
    for text in (
        positive_pull.change if positive_pull else None,
        pending_pull.change if pending_pull else None,
        _first_by_polarity(competitor_entries, {"positive"}),
        _first_by_polarity(competitor_entries, {"pending"}),
        competitor_entries[0].text if competitor_entries else None,
    ):
        if text:
            primary_pull = text
            break

    primary_tool_change = None
    for text in (
        negative_tool.change if negative_tool else None,
        positive_tool.change if positive_tool else None,
        _first_by_polarity(tool_entries, {"negative"}),
        _first_by_polarity(tool_entries, {"positive"}),
        tool_entries[0].text if tool_entries else None,
    ):
        if text:
            primary_tool_change = text
            break

    return SemanticView(
        primary_issue=primary_issue,
        primary_pull=primary_pull,
        primary_tool_change=primary_tool_change,
        positive_pull=positive_pull,
        pending_pull=pending_pull,
        positive_tool=positive_tool,
        negative_tool=negative_tool,
        has_positive_pull=any(entry.polarity == "positive" for entry in competitor_entries),
        has_pending_pull=any(entry.polarity == "pending" for entry in competitor_entries),
        has_positive_tool=any(entry.polarity == "positive" for entry in tool_entries),
        has_negative_tool=any(entry.polarity == "negative" for entry in tool_entries),
    )