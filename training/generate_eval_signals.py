"""Generate a realistic held-out evaluation set for terminal-only model checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_COMPETITOR_ROOT = _PROJECT_ROOT / "data" / "competitors"
_OUTPUT_DIR = _PROJECT_ROOT / "fixtures" / "eval_signals"


def _case(
    case_id: str,
    category: str,
    competitor_slug: str,
    expected_verdict: str,
    summary: str,
    competitor_changes: list[str],
    current_tool_status: list[str],
    notes: list[str],
    pricing_delta: str,
    compliance_changes: str = "No compliance change",
) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "competitor_slug": competitor_slug,
        "expected_verdict": expected_verdict,
        "summary": summary,
        "competitor_changes": competitor_changes,
        "current_tool_status": current_tool_status,
        "notes": notes,
        "pricing_delta": pricing_delta,
        "compliance_changes": compliance_changes,
    }


CASES = [
    _case(
        "embedded_reporting_cutover",
        "analytics",
        "datalens",
        "SWITCH",
        "Migration-priced DataLens rollout closes embedded reporting and mobile access gaps.",
        [
            "Embedded client reporting is now included on Professional for migrated InsightDeck accounts",
            "Mobile dashboard viewer now supports offline snapshots for account managers",
        ],
        [
            "InsightDeck still requires manual PDF packs for client reporting",
            "Partners still cannot review live dashboards from phones during client visits",
        ],
        [
            "Two renewal calls this month flagged the manual reporting workflow as a blocker",
            "DataLens migration assistant maps existing InsightDeck dashboards in one pass",
        ],
        "from PS480 to PS355 for the first 12 seats on migrated InsightDeck accounts",
    ),
    _case(
        "builder_refresh_only",
        "analytics",
        "clearview_analytics",
        "STAY",
        "ClearView refreshed the UI, but the release does not actually close Meridian's client-reporting gap.",
        [
            "Dashboard builder got a visual refresh and AI chart layout suggestions",
            "Saved views now load slightly faster on tablets",
        ],
        [
            "InsightDeck's main pain remains client-facing reporting, not dashboard cosmetics",
            "The analytics team already likes the current drag-and-drop builder",
        ],
        [
            "White-label embeds remain Enterprise-only",
            "AI anomaly detection is still roadmap only",
        ],
        "PS390 (competitor) vs PS340 (InsightDeck)",
    ),
    _case(
        "product_analytics_residency_block",
        "analytics",
        "pulsemetrics",
        "STAY",
        "PulseMetrics added product analytics polish, but the platform is still the wrong compliance and use-case fit.",
        [
            "PulseMetrics launched board-level funnel benchmarking",
            "Mobile dashboards now refresh in near real time",
        ],
        [
            "Compliance still requires UK data residency for analytics shared with EU clients",
            "Meridian needs BI-style client reporting rather than product analytics",
        ],
        [
            "Vendor still has no UK hosting region on the roadmap",
            "White-label client reporting is still unavailable",
        ],
        "PS310 (competitor) vs PS340 (InsightDeck)",
        "Still no UK data residency",
    ),
    _case(
        "overkill_and_cost",
        "analytics",
        "prism_bi",
        "STAY",
        "Prism BI is strong technically, but the operating model and price remain misaligned to the team.",
        [
            "Prism BI expanded root-cause analysis to all Professional tenants",
            "The migration assistant can now convert more InsightDeck dashboard widgets",
        ],
        [
            "InsightDeck still lacks lineage and anomaly detection",
            "Meridian still has no dedicated analytics engineer to own a heavy BI stack",
        ],
        [
            "Vendor still estimates more than 20 hours of setup and training",
            "Self-serve reporting remains too complex for occasional users",
        ],
        "PS520 (competitor) vs PS340 (InsightDeck)",
    ),
    _case(
        "anomaly_design_partner",
        "analytics",
        "clearview_analytics",
        "HOLD",
        "ClearView is interesting, but the promised anomaly capability is still a design-partner rollout.",
        [
            "ClearView opened AI anomaly detection to ten design-partner accounts",
            "The mobile viewer now supports cached client dashboards",
        ],
        [
            "InsightDeck still lacks anomaly detection",
            "The ISO audit still wants better data lineage evidence",
        ],
        [
            "The anomaly feature is design-partner only until the vendor confirms GA timing",
            "Account team advised waiting for the first public release notes before migration",
        ],
        "PS390 (competitor) vs PS340 (InsightDeck)",
    ),
    _case(
        "analytics_post_acquisition_pause",
        "analytics",
        "prism_bi",
        "HOLD",
        "Prism BI could solve the analytics gaps, but a fresh acquisition has frozen implementation timing.",
        [
            "Prism BI launched a consulting-onboarding package with dashboard conversion support",
            "Live client embeds are now available with audit-grade lineage",
        ],
        [
            "Manual client exports still consume analyst time every week",
            "Compliance still wants lineage before the next audit prep cycle",
        ],
        [
            "The vendor announced an acquisition yesterday and paused new onboarding commitments for 60 days",
            "Finance asked the team to hold any analytics migration until the support model is confirmed",
        ],
        "PS520 (competitor) vs PS340 (InsightDeck)",
    ),
    _case(
        "linkedin_outlook_release",
        "crm",
        "clientpulse",
        "SWITCH",
        "ClientPulse closes NexusCRM's enrichment, Outlook, and mobile gaps at a credible total cost.",
        [
            "Native LinkedIn enrichment is now included on Professional for all migrated accounts",
            "Outlook sync and mobile parity both shipped in the same release",
        ],
        [
            "NexusCRM reps still copy LinkedIn contact data by hand",
            "The current mobile app remains unstable for field sellers",
        ],
        [
            "Sales ops wants to remove the Outlook plugin workarounds before the next quarter",
            "Migration runbook is already scoped with the vendor",
        ],
        "PS418 (competitor) vs PS510 (NexusCRM)",
    ),
    _case(
        "mobile_sync_release",
        "crm",
        "dealstream",
        "SWITCH",
        "DealStream now fixes the mobile and sync pain that keeps surfacing in the sales process.",
        [
            "DealStream's Android app reached full parity with iOS",
            "Activity sync now updates InsightDeck in real time without rate-limit delays",
        ],
        [
            "NexusCRM still crashes for some iOS users and lags on mobile follow-up",
            "InsightDeck dashboards still go stale when NexusCRM API limits are hit",
        ],
        [
            "The vendor migration tool now imports NexusCRM pipelines and activities in one pass",
            "Sales leadership wants cleaner field activity tracking before the summer push",
        ],
        "PS462 (competitor) vs PS510 (NexusCRM)",
    ),
    _case(
        "sequence_and_enrichment",
        "crm",
        "pipelineiq",
        "SWITCH",
        "PipelineIQ now offers the automation and enrichment breadth the CRM team is missing today.",
        [
            "Unlimited-step email sequences are now bundled with native LinkedIn enrichment credits",
            "Outlook and Gmail sync share the same automation rules engine",
        ],
        [
            "NexusCRM still caps sequences at five steps",
            "Business development reps still hand-build lead lists outside the CRM",
        ],
        [
            "PipelineIQ's high-rate-limit API should remove the current InsightDeck sync bottleneck",
            "No security exceptions are needed for the move",
        ],
        "PS460 (competitor) vs PS510 (NexusCRM)",
    ),
    _case(
        "security_block",
        "crm",
        "velocitycrm",
        "STAY",
        "VelocityCRM remains a hard no despite minor UX work because the security baseline is still missing.",
        [
            "VelocityCRM trimmed page-load times on the pipeline board",
            "The Gmail sidebar widget was refreshed for quicker note entry",
        ],
        [
            "Security review still requires UK residency, SSO, and exportable audit logs for the CRM estate",
            "NexusCRM may be clunky, but it is already cleared for those controls",
        ],
        [
            "Android remains beta and the vendor still has no UK hosting option",
            "Audit logging is still unavailable on the live tiers",
        ],
        "PS390 (competitor) vs PS510 (NexusCRM)",
        "SSO and audit log remain unavailable and UK residency is unchanged",
    ),
    _case(
        "conversation_ai_pilot",
        "crm",
        "closerhub",
        "HOLD",
        "CloserHub looks strong, but the feature leadership wants is still only in a controlled pilot.",
        [
            "CloserHub opened conversation intelligence with auto-summaries to twelve pilot accounts",
            "Sequence branching now reacts to call outcomes",
        ],
        [
            "Managers still review calls manually and want better coaching visibility",
            "The current CRM still slows follow-up when sequence logic gets complex",
        ],
        [
            "The conversation AI rollout is pilot-only until the vendor publishes stable call quality metrics",
            "Ops asked to wait for a general-release milestone before migrating recordings",
        ],
        "from PS680 to PS525 for the first year on a 22-seat migrated account",
    ),
    _case(
        "renewal_bakeoff",
        "crm",
        "clientpulse",
        "HOLD",
        "ClientPulse is a real contender, but the team wants to finish a renewal bake-off before committing.",
        [
            "ClientPulse reserved migration slots for late summer and offered a live sandbox with sample data",
            "The reporting console now mirrors quota tracking from the weekly sales pack",
        ],
        [
            "NexusCRM renewal talks start next month",
            "Leadership wants a side-by-side bake-off before locking the next annual CRM term",
        ],
        [
            "Commercial asked for a hold until the bake-off is complete and pricing is signed off",
            "No production migration should begin before the renewal checkpoint",
        ],
        "PS418 (competitor) vs PS510 (NexusCRM)",
    ),
    _case(
        "barclays_realtime_cutover",
        "finance",
        "ledgerflow",
        "SWITCH",
        "LedgerFlow turns two recurring close blockers into standard product behavior.",
        [
            "LedgerFlow's Barclays feed now refreshes every 15 minutes with exception alerts",
            "Signed PDF audit packs are now generated directly from month-end close",
        ],
        [
            "VaultLedger still syncs Barclays overnight",
            "Finance still assembles signed audit evidence by hand",
        ],
        [
            "The controller wants the overnight feed bottleneck removed before the next reporting cycle",
            "No compliance gaps remain in the target stack",
        ],
        "PS380 (competitor) vs PS420 (VaultLedger)",
    ),
    _case(
        "multi_currency_close",
        "finance",
        "brightbooks",
        "SWITCH",
        "BrightBooks now combines the multi-currency, revenue-recognition, and audit controls that finance is missing.",
        [
            "BrightBooks now posts native EUR invoices and IFRS 15 schedules from the same workflow",
            "Project billing dashboards now feed signed audit exports automatically",
        ],
        [
            "VaultLedger still needs manual EUR conversions and a separate IFRS 15 workaround",
            "Audit exports remain CSV-only for external reviewers",
        ],
        [
            "The finance team wants one close process instead of three stitched workarounds",
            "Migration services are available next month",
        ],
        "PS395 (competitor) vs PS420 (VaultLedger)",
    ),
    _case(
        "controls_gap",
        "finance",
        "clearbooks_pro",
        "STAY",
        "ClearBooks Pro still misses the basic controls the finance stack requires.",
        [
            "ClearBooks Pro refreshed cashflow charts and added receipt OCR",
            "The expense queue is slightly faster to review",
        ],
        [
            "Security still requires SOC2 and SSO for finance systems",
            "Finance also still needs IFRS 15 without a spreadsheet workaround",
        ],
        [
            "The vendor still has no SOC2 timeline and no IFRS 15 delivery date",
            "Support remains business-hours only",
        ],
        "PS290 (competitor) vs PS420 (VaultLedger)",
        "SOC2 and SSO remain unavailable",
    ),
    _case(
        "finance_us_residency_block",
        "finance",
        "paytrek",
        "STAY",
        "PayTrek is feature-rich, but the residency model still rules it out immediately.",
        [
            "PayTrek improved approval routing and foreign-exchange controls",
            "The AP console now flags payment anomalies before release",
        ],
        [
            "Meridian still needs UK or EU data residency for client finance data",
            "The current stack cannot trade away residency controls for feature breadth",
        ],
        [
            "All finance data remains US-hosted with no UK or EU region committed",
            "The vendor says only a GDPR DPA is available, not regional hosting",
        ],
        "PS400 (competitor) vs PS420 (VaultLedger)",
        "Finance data remains US-hosted only",
    ),
    _case(
        "ifrs15_pilot",
        "finance",
        "exactspend",
        "HOLD",
        "ExactSpend is close, but the one missing module is still only in pilot.",
        [
            "ExactSpend opened its IFRS 15 module to six pilot customers",
            "The bank-feed monitor now flags stale Barclays connections automatically",
        ],
        [
            "VaultLedger still relies on a manual IFRS 15 workaround",
            "Real-time bank feeds would help the month-end team immediately",
        ],
        [
            "The IFRS 15 module is still pilot-only and finance wants shipped evidence before migration",
            "The controller asked to hold until the first audited close on the new module",
        ],
        "PS365 (competitor) vs PS420 (VaultLedger)",
    ),
    _case(
        "finance_post_acquisition_pause",
        "finance",
        "brightbooks",
        "HOLD",
        "BrightBooks looks strong, but a fresh ownership change has paused implementation confidence.",
        [
            "BrightBooks launched a managed migration package for professional-services finance teams",
            "Multi-entity consolidation is now available on Professional",
        ],
        [
            "VaultLedger still slows close with manual currency handling",
            "The CFO is open to a move once vendor stability is clearer",
        ],
        [
            "BrightBooks announced a private-equity acquisition and paused onboarding commitments for 45 days",
            "Finance asked to wait until the new support model is confirmed",
        ],
        "PS395 (competitor) vs PS420 (VaultLedger)",
    ),
    _case(
        "feedback_and_surveys_rollout",
        "hr",
        "workforge",
        "SWITCH",
        "WorkForge now solves several manual HR workflows in one release package.",
        [
            "WorkForge moved continuous feedback and eNPS into the Core tier",
            "Onboarding template packs now cover consulting hires and contractors",
        ],
        [
            "PeoplePulse still only supports annual review cycles",
            "HR still builds onboarding workflows from scratch for each new hire",
        ],
        [
            "The HR lead wants one system for feedback, surveys, and onboarding before the next review cycle",
            "PayAxis sync is already live in the target platform",
        ],
        "PS420 (competitor) vs PS385 (PeoplePulse)",
    ),
    _case(
        "erasure_automation_release",
        "hr",
        "workforge",
        "SWITCH",
        "WorkForge now removes a manual compliance process that legal wants closed quickly.",
        [
            "WorkForge now automates right-to-erasure requests with an audit trail",
            "Payroll sync alerts now fire before quarterly connector failures",
        ],
        [
            "PeoplePulse still requires support tickets for GDPR erasure requests",
            "The current PayAxis connector still breaks on quarterly schema changes",
        ],
        [
            "Legal asked for automated erasure handling before contract renewals with EU clients",
            "The extra monthly spend is smaller than the manual compliance overhead",
        ],
        "PS420 (competitor) vs PS385 (PeoplePulse)",
    ),
    _case(
        "flat_rate_but_blocked",
        "hr",
        "hrnest",
        "STAY",
        "HRNest is cheaper, but it is still missing core enterprise controls.",
        [
            "HRNest lowered its flat-rate pricing and refreshed the self-service portal",
            "The org chart and leave screens now load faster",
        ],
        [
            "Security still requires SOC2, SSO, and audit logs for the 150-seat HR system",
            "PeoplePulse already handles leave and self-service reliably",
        ],
        [
            "HRNest remains CSV-only for payroll and still has no audit log",
            "The attractive flat rate does not remove the control gaps",
        ],
        "PS295 (competitor) vs PS385 (PeoplePulse)",
        "SOC2, SSO, and audit log remain unavailable",
    ),
    _case(
        "soc2_gap",
        "hr",
        "teamledger_hr",
        "STAY",
        "TeamLedger HR added nice workflow polish, but the security gate still does not clear.",
        [
            "TeamLedger HR added new survey templates and compensation bands",
            "The onboarding builder now supports reusable task blocks",
        ],
        [
            "The HR team wants better feedback tooling, but security still needs a SOC2-backed platform",
            "PeoplePulse already serves as the system of record for headcount",
        ],
        [
            "SOC2 remains absent in the vendor's live controls documentation",
            "GDPR erasure is still manual",
        ],
        "PS510 (competitor) vs PS385 (PeoplePulse)",
        "SOC2 remains unavailable",
    ),
    _case(
        "learning_hub_preview",
        "hr",
        "workforge",
        "HOLD",
        "WorkForge is promising, but the learning expansion HR wants is not generally available yet.",
        [
            "WorkForge previewed an expanded learning hub with manager-assigned curricula",
            "Compensation benchmarking is being opened to a small design-partner group",
        ],
        [
            "HR wants to unify learning and performance in one platform",
            "PeoplePulse still leaves learning workflows in separate tools",
        ],
        [
            "The expanded learning module is preview-only through next quarter",
            "HR asked to wait for a public rollout before changing the training stack",
        ],
        "PS420 (competitor) vs PS385 (PeoplePulse)",
    ),
    _case(
        "post_merger_pause",
        "hr",
        "workforge",
        "HOLD",
        "WorkForge is attractive, but vendor-change risk has temporarily paused onboarding confidence.",
        [
            "WorkForge announced a deeper BrightPay payroll integration and a consulting onboarding squad",
            "The offboarding workflow now adds IT and finance approvals automatically",
        ],
        [
            "PeoplePulse still forces manual coordination across HR and payroll",
            "Operations is interested but does not want to migrate during vendor uncertainty",
        ],
        [
            "The vendor announced a merger with a payroll partner and paused new enterprise onboarding until the integration plan lands",
            "Leadership asked for a hold until the roadmap freeze is lifted",
        ],
        "PS420 (competitor) vs PS385 (PeoplePulse)",
    ),
    _case(
        "portal_and_time_capture",
        "project_mgmt",
        "projectaxis",
        "SWITCH",
        "ProjectAxis folds approval and time capture into the delivery workflow Meridian already runs.",
        [
            "ProjectAxis now pushes approved time straight into billing exports",
            "Client portal approvals can sign off deliverables without email chases",
        ],
        [
            "TaskBridge still relies on ChronoLog for time capture",
            "Clients still approve deliverables over email threads outside the project system",
        ],
        [
            "PMO wants billing and approval evidence in one place before the next wave of fixed-fee projects",
            "The implementation plan keeps all active templates",
        ],
        "PS330 (competitor) vs PS385 (TaskBridge plus ChronoLog)",
    ),
    _case(
        "capacity_and_approvals_release",
        "project_mgmt",
        "flowboard",
        "SWITCH",
        "FlowBoard now closes two of the heaviest operational gaps in one move.",
        [
            "FlowBoard rolled resource heatmaps and client approvals into the Standard plan for migrated teams",
            "Native time-tracking now feeds budget-vs-actual alerts automatically",
        ],
        [
            "Managers still track capacity in spreadsheets",
            "TaskBridge still has no live client approval workflow",
        ],
        [
            "Operations wants to retire ChronoLog and the spreadsheet capacity model together",
            "The new FlowBoard release matches the consulting delivery workflow closely",
        ],
        "PS340 (competitor) vs PS385 (TaskBridge plus ChronoLog)",
    ),
    _case(
        "dev_team_mismatch",
        "project_mgmt",
        "sprintloop",
        "STAY",
        "SprintLoop remains better suited to engineering teams than consulting delivery.",
        [
            "SprintLoop added better GitHub issue linking and retrospective templates",
            "Milestone views now surface sprint velocity trends",
        ],
        [
            "Consulting delivery still needs client approvals, billing integration, and time tracking",
            "TaskBridge already works better for non-technical staff than dev-centric boards",
        ],
        [
            "UK residency and audit-log gaps remain unresolved on the live tiers",
            "The product is still designed for engineering teams, not client delivery",
        ],
        "PS420 (Business competitor) vs PS385 (TaskBridge plus ChronoLog)",
        "UK residency and audit-log gaps remain unresolved",
    ),
    _case(
        "reporting_polish_only",
        "project_mgmt",
        "sprintdesk",
        "STAY",
        "SprintDesk improved reporting depth, but not the collaboration and workflow gaps Meridian actually has.",
        [
            "SprintDesk shipped deeper burndown charts and roadmap widgets",
            "Automation rules can now chain more sprint-state changes",
        ],
        [
            "TaskBridge's real gap is client collaboration and built-in time tracking, not sprint analytics",
            "Most delivery teams are non-technical and do not want a heavier engineering tool",
        ],
        [
            "Client portal is still absent and time-tracking still needs Premium",
            "The Gantt view still shares TaskBridge's refresh problem",
        ],
        "PS410 (competitor) vs PS385 (TaskBridge plus ChronoLog)",
    ),
    _case(
        "ai_scheduler_beta",
        "project_mgmt",
        "flowboard",
        "HOLD",
        "FlowBoard is compelling, but the planning feature PMO wants is still beta-only.",
        [
            "FlowBoard opened its AI scheduling assistant to a closed beta",
            "Capacity views now propose staffing moves for delayed projects",
        ],
        [
            "Operations wants better forecasting for utilisation, but not at the cost of moving onto beta planning logic",
            "The current tool still needs better capacity planning",
        ],
        [
            "The scheduling assistant is closed-beta only and the vendor has not published GA timing",
            "PMO asked to hold until the first public release proves stable",
        ],
        "PS340 (competitor) vs PS385 (TaskBridge plus ChronoLog)",
    ),
    _case(
        "pm_post_acquisition_pause",
        "project_mgmt",
        "projectaxis",
        "HOLD",
        "ProjectAxis still looks operationally strong, but a fresh acquisition has paused start dates.",
        [
            "ProjectAxis announced a consulting onboarding squad and a new migration service for TaskBridge accounts",
            "The client portal now includes approval reminders and budget alerts",
        ],
        [
            "TaskBridge still needs separate tools for time capture and approvals",
            "The operational case for switching is strong but leadership wants vendor stability first",
        ],
        [
            "ProjectAxis announced an acquisition this week and froze implementation start dates for 30 days",
            "Finance asked the PMO to pause any migration until the support structure is confirmed",
        ],
        "PS330 (competitor) vs PS385 (TaskBridge plus ChronoLog)",
    ),
    # ── ANALYTICS new ─────────────────────────────────────────────────
    _case(
        "lineage_and_alerting_release",
        "analytics",
        "datalens",
        "SWITCH",
        "DataLens closes InsightDeck's two most disruptive operational gaps in one release.",
        [
            "DataLens shipped data lineage tracing and real-time pipeline alerting on all Professional plans",
            "Alerting rules can be set per data source with Slack and email channels",
        ],
        [
            "InsightDeck has no lineage view; broken pipeline root-cause takes the data team a morning to trace",
            "No alerting means failures surface on Monday after clients notice dashboard gaps",
        ],
        [
            "Both gaps were rated critical in the last data team health check",
            "DataLens migration assistant maps existing InsightDeck dashboards automatically",
        ],
        "PS355 (DataLens) vs PS340 (InsightDeck)",
    ),
    _case(
        "white_label_embed_rollout",
        "analytics",
        "metricflux",
        "SWITCH",
        "MetricFlux rolls white-label embedded dashboards onto all paid tiers, directly closing the client-reporting gap.",
        [
            "MetricFlux moved white-label embedded client dashboards to the Growth tier for all accounts",
            "Client-shareable links now support custom subdomain branding",
        ],
        [
            "InsightDeck client reports require manual PDF exports and Canva re-branding — three hours per monthly pack",
            "Clients cannot access live dashboards; all sharing is via static exports",
        ],
        [
            "Client reporting was the top pain in the last PM retrospective",
            "MetricFlux embed supports the same chart types InsightDeck uses today",
        ],
        "PS290 (MetricFlux) vs PS340 (InsightDeck)",
    ),
    _case(
        "ai_sql_chat_irrelevant",
        "analytics",
        "clearview_analytics",
        "STAY",
        "ClearView's AI SQL chat is impressive but does not address the client-facing reporting gap.",
        [
            "ClearView launched an AI-powered natural-language SQL chat for analysts",
            "Chat can query any connected data source and export results to CSV",
        ],
        [
            "InsightDeck's core gap is client-facing embedded reporting, not analyst SQL access",
            "The analytics team already uses Redash for ad-hoc SQL queries",
        ],
        [
            "White-label embeds remain Enterprise-only at ClearView with no pricing change announced",
            "SQL chat feature is orthogonal to the actual bottleneck",
        ],
        "PS390 (ClearView Analytics) vs PS340 (InsightDeck)",
    ),
    _case(
        "dashboard_speed_irrelevant",
        "analytics",
        "pulsemetrics",
        "STAY",
        "PulseMetrics cut load times to sub-second but the team's bottleneck is export formatting, not speed.",
        [
            "PulseMetrics reduced dashboard load time to under one second with edge caching",
            "New chart transition animations improve stakeholder presentation look and feel",
        ],
        [
            "InsightDeck dashboards load in 2-3 seconds which the team considers acceptable",
            "The weekly bottleneck is formatting and branding the exported report pack, not dashboard viewing speed",
        ],
        [
            "PulseMetrics still has no UK data residency and white-label reporting remains unavailable",
            "Speed improvements do not change the export workflow at all",
        ],
        "PS310 (PulseMetrics) vs PS340 (InsightDeck)",
        "Still no UK data residency",
    ),
    _case(
        "client_portal_early_access",
        "analytics",
        "metricflux",
        "HOLD",
        "MetricFlux's client portal is the feature that would trigger a switch, but it is early-access only.",
        [
            "MetricFlux announced a client portal with live embedded dashboards for end customers",
            "Early access opened to 20 design-partner accounts",
        ],
        [
            "InsightDeck still requires manual exports for client reporting",
            "The MetricFlux portal matches the team's workflow requirements exactly",
        ],
        [
            "Design-partner cohort is closed; general availability timeline has not been confirmed",
            "Team wants to evaluate the GA release notes before committing to migration",
        ],
        "PS290 (MetricFlux) vs PS340 (InsightDeck)",
    ),
    _case(
        "datalens_acquisition_pause",
        "analytics",
        "datalens",
        "HOLD",
        "DataLens released useful features but a fresh acquisition has frozen migration discussions.",
        [
            "DataLens shipped an improved mobile snapshot viewer and a consulting-segment onboarding pack",
            "Onboarding pack includes a dashboard-conversion tool for InsightDeck exports",
        ],
        [
            "InsightDeck still lacks offline mobile access for account managers visiting clients",
            "DataLens onboarding would ease the migration significantly",
        ],
        [
            "DataLens was acquired by Ascend Analytics Group last quarter",
            "Vendor paused migration engagements pending new product roadmap publication",
        ],
        "PS355 (DataLens) vs PS340 (InsightDeck)",
    ),
    # ── CRM new ────────────────────────────────────────────────────────
    _case(
        "contact_intelligence_consolidation",
        "crm",
        "leadsphere",
        "SWITCH",
        "LeadSphere bundles contact scoring and multi-step sequencing, removing the need for a separate Outreach subscription.",
        [
            "LeadSphere moved contact scoring and multi-step email sequencing into Professional for migrated accounts",
            "Sequence analytics now feed directly into pipeline risk flags",
        ],
        [
            "NexusCRM reps manually assess deal priority with no scoring model",
            "Email cadences require a separate Outreach subscription adding PS120 per month",
        ],
        [
            "Sales ops wants to consolidate CRM and sequencing before the next hiring wave",
            "LeadSphere migration includes a data-mapping tool for NexusCRM exports",
        ],
        "PS475 (LeadSphere) vs PS510 (NexusCRM)",
    ),
    _case(
        "ai_forecast_core_plan",
        "crm",
        "closerhub",
        "SWITCH",
        "CloserHub moves AI deal scoring onto Core, directly addressing the pipeline visibility OKR.",
        [
            "CloserHub moved AI deal scoring and automated pipeline risk flags to the Core plan",
            "Forecast accuracy charts now visible to ops without a dashboard licence",
        ],
        [
            "NexusCRM pipeline forecasts require manual rep updates every Friday; no early warning on slipping deals",
            "Finance OKR for this quarter is 12-week forecast accuracy improvement",
        ],
        [
            "CloserHub pilot with two reps showed 31% forecast accuracy improvement over 6 weeks",
            "Ops team has already scoped the migration runbook",
        ],
        "PS430 (CloserHub) vs PS510 (NexusCRM)",
    ),
    _case(
        "territory_management_irrelevant",
        "crm",
        "leadsphere",
        "STAY",
        "LeadSphere's territory management release does not address the forecasting and pipeline gaps.",
        [
            "LeadSphere launched territory management with automatic account assignment rules",
            "New account health score widget added to the default dashboard",
        ],
        [
            "The team runs a flat structure with no territory boundaries; territory management solves a problem we do not have",
            "NexusCRM's Apollo enrichment integration is still superior for contact data quality",
        ],
        [
            "Forecasting and pipeline visibility remain the actual gaps; LeadSphere did not ship anything in those areas",
            "Switching would lose the Apollo enrichment workflow already embedded in the sales process",
        ],
        "PS475 (LeadSphere) vs PS510 (NexusCRM)",
    ),
    _case(
        "smb_pivot_mismatch",
        "crm",
        "dealstream",
        "STAY",
        "DealStream's latest release targets SMB teams and moves further from the enterprise feature set we need.",
        [
            "DealStream launched automated SMB sequences and simplified the UI for faster contact creation",
            "New onboarding flow reduced time-to-first-deal for SMB teams by 40 percent",
        ],
        [
            "NexusCRM is well-configured for the enterprise deal cycle with account hierarchy and role-based approvals",
            "DealStream's SMB focus means the enterprise features we rely on remain on the roadmap",
        ],
        [
            "Account hierarchy and approval workflows are not included in the new DealStream release",
            "The product direction diverges from our use case with each release",
        ],
        "PS380 (DealStream) vs PS510 (NexusCRM)",
    ),
    _case(
        "soc2_audit_pending",
        "crm",
        "velocitycrm",
        "HOLD",
        "VelocityCRM adds intent signals but its SOC 2 Type II audit must complete before the compliance team approves migration.",
        [
            "VelocityCRM added buyer intent signals and improved Slack deal-room integration",
            "Intent data is sourced from G2 Buyer Intent and LinkedIn Sales Navigator APIs",
        ],
        [
            "NexusCRM lacks native intent data; Slack deal rooms currently require Zapier automation",
            "Both gaps are on the sales ops priority list for this half",
        ],
        [
            "VelocityCRM is completing a SOC 2 Type II audit with certificate expected in 8 weeks",
            "Compliance team will not approve any CRM migration before the Type II report is in hand",
        ],
        "PS445 (VelocityCRM) vs PS510 (NexusCRM)",
    ),
    _case(
        "uk_residency_closed_beta",
        "crm",
        "pipelineiq",
        "HOLD",
        "PipelineIQ expands enterprise features but UK data residency — a hard requirement — is still in closed beta.",
        [
            "PipelineIQ expanded its sequence builder and launched an account hierarchy view",
            "Account hierarchy now supports up to five levels for holding-company structures",
        ],
        [
            "NexusCRM lacks multi-level account hierarchy for the group structure",
            "Sequence automation in NexusCRM is basic compared to what sales ops needs",
        ],
        [
            "PipelineIQ UK data residency option is in closed beta with no confirmed GA date",
            "Legal team requires UK-hosted CRM data before migration approval",
        ],
        "PS490 (PipelineIQ) vs PS510 (NexusCRM)",
    ),
    # ── FINANCE new ────────────────────────────────────────────────────
    _case(
        "bank_recon_and_expense_categorisation",
        "finance",
        "novapay",
        "SWITCH",
        "NovaPay automates the two workflows that consume most of the finance team's manual time.",
        [
            "NovaPay added automated bank reconciliation and AI-based expense categorisation on Business",
            "Reconciliation runs nightly and flags discrepancies with a one-click correction workflow",
        ],
        [
            "VaultLedger bank reconciliation is a manual Friday process taking six hours per week",
            "Expense categorisation requires a mid-month manual review by the finance analyst",
        ],
        [
            "Both gaps were flagged as high-priority in the last finance team health review",
            "NovaPay cost matches VaultLedger so there is no budget barrier",
        ],
        "PS420 (NovaPay) vs PS420 (VaultLedger)",
    ),
    _case(
        "mtd_ir35_compliance_bundle",
        "finance",
        "exactspend",
        "SWITCH",
        "ExactSpend bundles MTD bridging and IR35 tracking onto the standard plan, closing two compliance gaps.",
        [
            "ExactSpend bundled MTD-compliant bridging software and IR35 determination tracking into the standard plan",
            "IR35 assessments are logged with audit trail and exported to the engagement letter workflow",
        ],
        [
            "VaultLedger requires a separate Avalara add-on for MTD bridging costing PS35 per month",
            "IR35 determination is done manually in a spreadsheet with no audit trail",
        ],
        [
            "HMRC MTD for ITSA mandate makes bridging non-negotiable by April next year",
            "IR35 automation would eliminate an estimated PS3000 annual manual review cost",
        ],
        "PS395 (ExactSpend) vs PS420 (VaultLedger)",
    ),
    _case(
        "fx_dashboard_cosmetics_only",
        "finance",
        "brightbooks",
        "STAY",
        "BrightBooks improves FX visuals but VaultLedger's multi-entity reporting advantage outweighs cosmetic gains.",
        [
            "BrightBooks shipped real-time FX rate feeds and visual hedging exposure summaries",
            "FX dashboard now highlights open positions against budget in a single view",
        ],
        [
            "VaultLedger multi-entity consolidation and management reporting are better structured for the group finance team",
            "FX exposure management is handled in the Treasury module; dashboard cosmetics do not change the workflow",
        ],
        [
            "BrightBooks still requires manual journal entries for intercompany eliminations",
            "The FX dashboard improvement is useful but does not justify migrating away from a better consolidation model",
        ],
        "PS395 (BrightBooks) vs PS420 (VaultLedger)",
    ),
    _case(
        "payroll_integration_redundant",
        "finance",
        "novapay",
        "STAY",
        "NovaPay's new payroll integration duplicates a capability VaultLedger already delivers.",
        [
            "NovaPay added a native payroll integration and bulk supplier payment scheduling",
            "Payroll sync supports 12 providers including PayAxis",
        ],
        [
            "VaultLedger already has a direct PayAxis sync that the team uses daily",
            "Bulk payments are handled via the Lloyds treasury module outside the accounting stack",
        ],
        [
            "Both new NovaPay features duplicate functionality already in place",
            "No active pain exists that NovaPay's new release addresses",
        ],
        "PS420 (NovaPay) vs PS420 (VaultLedger)",
    ),
    _case(
        "multi_entity_design_partner",
        "finance",
        "clearbooks_pro",
        "HOLD",
        "ClearBooks Pro announces multi-entity consolidation — the exact missing feature — but it is design-partner only.",
        [
            "ClearBooks Pro announced multi-entity consolidation for group reporting, opened to 15 design partners",
            "Consolidation handles intercompany eliminations automatically",
        ],
        [
            "VaultLedger requires manual journal adjustments for intercompany eliminations each month",
            "Multi-entity consolidation is the single feature that would trigger a switch",
        ],
        [
            "Design-partner cohort is closed with no Q3 GA confirmation from the vendor",
            "Finance team will not migrate to a feature in design-partner preview",
        ],
        "PS380 (ClearBooks Pro) vs PS420 (VaultLedger)",
    ),
    _case(
        "uk_data_residency_roadmap_only",
        "finance",
        "ledgerflow",
        "HOLD",
        "LedgerFlow ships bank feeds and a UK partner programme but UK data residency is still unachieved.",
        [
            "LedgerFlow added automated bank feeds for major UK banks and launched a UK accounting partner programme",
            "Partner programme includes a dedicated migration team for mid-market accounts",
        ],
        [
            "VaultLedger bank feeds require manual CSV upload for two of our three bank accounts",
            "A UK migration partner would significantly reduce the implementation risk",
        ],
        [
            "LedgerFlow UK data residency certification is on the roadmap but has not yet been granted",
            "Finance requires UK-hosted data for GDPR compliance; migration is blocked until certification is confirmed",
        ],
        "PS355 (LedgerFlow) vs PS420 (VaultLedger)",
    ),
    # ── HR new ─────────────────────────────────────────────────────────
    _case(
        "probation_and_review_templates",
        "hr",
        "teamrise",
        "SWITCH",
        "TeamRise adds automated probation tracking and structured review templates, closing two compliance-sensitive gaps.",
        [
            "TeamRise added automated probation milestone alerts and structured performance review templates for all tiers",
            "Review templates include a 90-day and 6-month probation workflow with manager sign-off checkpoints",
        ],
        [
            "PeoplePulse manages probation via manual calendar reminders; two compliance misses this year",
            "Review cycle templates must be rebuilt from scratch each quarter in PeoplePulse",
        ],
        [
            "HR wants both gaps closed before the next hiring surge which begins next quarter",
            "TeamRise includes a PeoplePulse data import tool for employee records and review history",
        ],
        "PS410 (TeamRise) vs PS385 (PeoplePulse)",
    ),
    _case(
        "rota_and_absence_suite",
        "hr",
        "crewplan",
        "SWITCH",
        "CrewPlan solves shift scheduling and absence management for the growing deskless workforce.",
        [
            "CrewPlan launched rota planning for deskless workers with auto-scheduling and PayAxis absence export",
            "Absence management feeds directly into the payroll cutoff with zero manual entry",
        ],
        [
            "PeoplePulse has no shift-scheduling module; warehouse rotas are built in a shared spreadsheet",
            "Absence is tracked manually with a CSV export to PayAxis every pay period",
        ],
        [
            "Warehouse and facilities headcount is growing 30 percent year-on-year; manual scheduling is no longer viable",
            "CrewPlan PayAxis export is already built and certified",
        ],
        "PS360 (CrewPlan) vs PS385 (PeoplePulse)",
    ),
    _case(
        "hourly_worker_mismatch",
        "hr",
        "shiftcore",
        "STAY",
        "ShiftCore is designed for hourly deskless workers; our workforce is 90 percent salaried hybrid.",
        [
            "ShiftCore expanded auto-scheduling accuracy and added real-time shift trade notifications",
            "New mobile app supports shift-swap approvals for frontline managers",
        ],
        [
            "PeoplePulse covers the professional headcount and structured review cycles well",
            "ShiftCore's operating model assumes hourly pay cycles and requires a different payroll integration",
        ],
        [
            "The workforce is 90 percent salaried hybrid; ShiftCore is built for hourly deskless workers",
            "Adopting ShiftCore would require a separate HR platform for the professional headcount",
        ],
        "PS330 (ShiftCore) vs PS385 (PeoplePulse)",
    ),
    _case(
        "engagement_surveys_no_gdpr_erasure",
        "hr",
        "teamrise",
        "STAY",
        "TeamRise adds engagement surveys but the GDPR erasure gap still blocks a switch.",
        [
            "TeamRise added 30 engagement survey templates and a rolling eNPS tracking dashboard",
            "Survey results can be segmented by department and tenure",
        ],
        [
            "PeoplePulse already supports structured quarterly surveys that the team runs reliably",
            "eNPS tracking is available through the existing PeoplePulse analytics module",
        ],
        [
            "Legal requires GDPR right-to-erasure automation before any HR platform change",
            "TeamRise still has no GDPR erasure automation; the gap that blocked the last evaluation remains",
        ],
        "PS410 (TeamRise) vs PS385 (PeoplePulse)",
    ),
    _case(
        "gdpr_erasure_eu_preview_only",
        "hr",
        "crewplan",
        "HOLD",
        "CrewPlan announces GDPR erasure automation but the preview is EU-only with no UK GA date.",
        [
            "CrewPlan announced GDPR right-to-erasure automation and released it in preview for EU customers",
            "Erasure requests are logged with a 72-hour SLA dashboard for DPOs",
        ],
        [
            "PeoplePulse handles erasure requests manually with an average 48-hour response time",
            "GDPR erasure automation is the primary feature driving the CrewPlan evaluation",
        ],
        [
            "EU-only preview excludes UK customers; vendor has not confirmed a UK GA date",
            "HR will not migrate until the feature is generally available in the UK region",
        ],
        "PS360 (CrewPlan) vs PS385 (PeoplePulse)",
    ),
    _case(
        "series_b_enterprise_rebuild",
        "hr",
        "shiftcore",
        "HOLD",
        "ShiftCore ships useful enterprise workflows but a recent Series B leaves the product roadmap unclear.",
        [
            "ShiftCore added manager escalation workflows and improved shift-swap approvals for enterprise accounts",
            "New reporting module shows shift-coverage gaps by site",
        ],
        [
            "PeoplePulse has no shift-swap workflows; manager escalations go by email today",
            "Site-level coverage reporting would be useful for the facilities team",
        ],
        [
            "ShiftCore closed a Series B last month and is rebuilding its enterprise pricing tier",
            "New pricing and support SLA have not been published; stability concern until Q4 roadmap is confirmed",
        ],
        "PS330 (ShiftCore) vs PS385 (PeoplePulse)",
    ),
    # ── PROJECT MANAGEMENT new ─────────────────────────────────────────
    _case(
        "gantt_and_client_approvals",
        "project_mgmt",
        "opscanvas",
        "SWITCH",
        "OpsCanvas closes both critical PM gaps — Gantt dependency tracking and client approvals — in one release.",
        [
            "OpsCanvas shipped Gantt dependency tracking and a client approval workflow on all paid plans",
            "Client approval links are shareable without a login and support inline comment threads",
        ],
        [
            "TaskBridge has no Gantt view; dependencies are tracked in a separate Confluence table",
            "Client approvals are managed over email with no audit trail; both rated critical in the PM survey",
        ],
        [
            "Both features were explicitly requested in the last all-hands prioritisation session",
            "OpsCanvas includes a TaskBridge export migration tool",
        ],
        "PS310 (OpsCanvas) vs PS385 (TaskBridge)",
    ),
    _case(
        "jira_sync_and_resource_forecast",
        "project_mgmt",
        "teamsync_projects",
        "SWITCH",
        "TeamSync Projects closes the dev-PM collaboration gap and removes the duplicate Confluence board.",
        [
            "TeamSync Projects added bidirectional Jira sync and resource forecasting for mixed dev and delivery teams",
            "Resource forecasts now show utilisation across both project and sprint workloads in one view",
        ],
        [
            "TaskBridge has no Jira integration; dev team maintains a separate Confluence board to track delivery risk",
            "Resource forecasting is done in a spreadsheet that is always a week out of date",
        ],
        [
            "Dev-PM integration friction was the most-cited issue in the last delivery retrospective",
            "Jira sync alone eliminates the duplicate board maintenance that consumes 2 hours per sprint",
        ],
        "PS220 (TeamSync Projects) vs PS385 (TaskBridge)",
    ),
    _case(
        "ai_summaries_not_the_gap",
        "project_mgmt",
        "flowboard",
        "STAY",
        "FlowBoard's AI summaries are a nice addition but do not address the custom status workflow gap.",
        [
            "FlowBoard added AI-generated project summaries and a weekly stakeholder digest email",
            "Summaries are auto-drafted from task completion data and editable before sending",
        ],
        [
            "TaskBridge integrates with the existing reporting stack and the team's reporting workflow is fine",
            "The PM team's core gap is custom status workflows for consulting delivery stages, not report generation",
        ],
        [
            "FlowBoard custom status workflows remain on the Enterprise plan at PS420 per month",
            "The AI summaries feature is orthogonal to the custom status requirement",
        ],
        "PS340 (FlowBoard) vs PS385 (TaskBridge)",
    ),
    _case(
        "price_cut_no_custom_statuses",
        "project_mgmt",
        "projectaxis",
        "STAY",
        "ProjectAxis cuts pricing 20 percent but custom status workflows — the core requirement — remain Enterprise-only.",
        [
            "ProjectAxis reduced pricing 20 percent across all tiers and added bulk task import",
            "Bulk import now supports CSV and JSON exports from TaskBridge",
        ],
        [
            "TaskBridge custom status workflows are central to how the consulting team tracks delivery stages",
            "ProjectAxis custom statuses remain restricted to the Enterprise plan which costs more than the current setup",
        ],
        [
            "Price reduction is welcome but does not change the feature gap on the standard tier",
            "Bulk import alone does not justify a migration that would lose the custom status model",
        ],
        "PS300 (ProjectAxis) vs PS385 (TaskBridge)",
    ),
    _case(
        "multi_workspace_early_access",
        "project_mgmt",
        "sprintdesk",
        "HOLD",
        "SprintDesk's multi-workspace feature is the migration trigger but it is early access only with no GA date.",
        [
            "SprintDesk announced multi-workspace support for managing separate client portfolios and opened early access",
            "Multi-workspace includes cross-portfolio resource view and consolidated billing",
        ],
        [
            "TaskBridge handles all client portfolios in a single flat workspace with no separation",
            "Multi-workspace is central to the business case: it enables per-client data isolation",
        ],
        [
            "Early access cohort is limited to 10 teams; general availability timeline has not been confirmed",
            "Team will not migrate without first evaluating the GA feature in a trial",
        ],
        "PS350 (SprintDesk) vs PS385 (TaskBridge)",
    ),
    _case(
        "parent_acquisition_roadmap_unclear",
        "project_mgmt",
        "flowboard",
        "HOLD",
        "FlowBoard adds useful capacity views but its parent company's acquisition leaves the product roadmap unclear.",
        [
            "FlowBoard released a portfolio health dashboard and improved capacity allocation charts",
            "Capacity charts now show over- and under-allocation per resource across all active projects",
        ],
        [
            "TaskBridge lacks a portfolio-level health view; capacity planning is maintained in spreadsheets",
            "Capacity allocation charts would reduce the weekly PM planning meeting by 30 minutes",
        ],
        [
            "FlowBoard's parent company completed an acquisition of SprintBase last quarter",
            "Integration and support model are unclear until the Q3 roadmap is published; team wants stability signal",
        ],
        "PS340 (FlowBoard) vs PS385 (TaskBridge)",
    ),
]


def _load_competitor_name(category: str, competitor_slug: str) -> str:
    path = _COMPETITOR_ROOT / category / f"{competitor_slug}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["name"]


def _validate_cases() -> None:
    if len(CASES) != 60:
        raise ValueError(f"Expected 60 held-out cases, found {len(CASES)}")

    case_ids = [case["case_id"] for case in CASES]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Held-out case_ids must be unique")

    verdict_counts = Counter(case["expected_verdict"] for case in CASES)
    expected_counts = {"SWITCH": 20, "STAY": 20, "HOLD": 20}
    if verdict_counts != expected_counts:
        raise ValueError(
            f"Expected balanced verdict counts {expected_counts}, found {dict(verdict_counts)}"
        )


def _build_payload(case: dict) -> dict:
    competitor_name = _load_competitor_name(case["category"], case["competitor_slug"])
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "competitor_slug": case["competitor_slug"],
        "scenario": case["case_id"],
        "expected_verdict": case["expected_verdict"],
        "summary": case["summary"],
        "signal": {
            "competitor": competitor_name,
            "competitor_changes": case["competitor_changes"],
            "current_tool_status": case["current_tool_status"],
            "notes": case["notes"],
            "compliance_changes": case["compliance_changes"],
            "pricing_delta": case["pricing_delta"],
        },
    }


def main() -> None:
    _validate_cases()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in _OUTPUT_DIR.glob("*.json"):
        path.unlink()

    for case in CASES:
        payload = _build_payload(case)
        filename = f"{case['category']}_{case['competitor_slug']}_{case['case_id']}.json"
        out_path = _OUTPUT_DIR / filename
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    verdict_counts = Counter(case["expected_verdict"] for case in CASES)
    print(f"Wrote {len(CASES)} held-out fixtures to {_OUTPUT_DIR}")
    print(
        "Verdict mix: "
        + ", ".join(f"{verdict}={verdict_counts[verdict]}" for verdict in ("SWITCH", "STAY", "HOLD"))
    )


if __name__ == "__main__":
    main()