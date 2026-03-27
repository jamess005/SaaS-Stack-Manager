# Company Profile: Meridian Consulting Group

## Overview
- **Type:** B2B Professional Services (management consulting, technology advisory)
- **Size:** ~150 employees
- **Locations:** London HQ (Clerkenwell), Bristol satellite office (20 staff), remote-first policy
- **Revenue band:** £8M–£12M ARR
- **Clients:** Mid-market and enterprise — mix of UK domestic and EU international
- **Finance:** GBP primary, EUR secondary (40% of revenue billed in EUR to EU clients)
- **Founded:** 2011

## Client Verticals
- **DACH region:** Automotive (tier-1 suppliers), financial services, manufacturing operations
- **Benelux region:** Professional services, logistics, retail transformation
- **UK domestic:** Financial services, public sector adjacent, tech scale-ups
- EU clients require EUR invoicing, EU-resident data storage, and GDPR-compliant tooling.
  Failure to invoice in EUR natively has caused 2 payment delays in the past 12 months.

## Staff Breakdown by Function
| Function              | Headcount |
|-----------------------|-----------|
| Delivery / Consulting | 95        |
| Admin / Support       | 20        |
| Operations / Finance  | 12        |
| Sales / BD            | 8         |
| Leadership            | 5         |
| IT                    | 6         |
| HR                    | 4         |
| **Total**             | **150**   |

Consulting delivery staff operate in billable project teams. Utilisation rate is tracked
monthly; project billing accuracy directly affects revenue recognition.

## IT Infrastructure
- **Identity provider:** Azure Active Directory (Microsoft Entra ID). All SaaS tools must
  support SAML 2.0 or OIDC SSO. Entra ID is the single source of truth for user provisioning
  and de-provisioning. Manual account management is not acceptable for tools with >10 users.
- **Cloud:** Azure (primary). No on-prem servers.
- **Collaboration:** Microsoft 365 (Teams, SharePoint, Outlook). CRM and project tools
  must integrate with Outlook natively; Gmail-only tools are not viable.
- **Integration layer:** No ESB or iPaaS in use. Tool integrations must be native API or
  Zapier. Zapier is acceptable only for non-critical, low-frequency flows. Core data flows
  (billing, HR, CRM sync) must be native integrations.
- **Payroll:** PayAxis (UK payroll provider). Finance and HR tools must integrate natively
  with PayAxis; middleware connectors have broken three times in 18 months.

## Consulting Billing Model
- **Time-and-materials:** ~60% of engagements. Billable hours logged weekly per consultant.
  Billing accuracy is audited quarterly. Errors result in revenue leakage or client disputes.
- **Fixed-fee retainers:** ~40% of engagements. Monthly invoices raised per client in GBP or
  EUR. Multi-currency invoicing accuracy is business-critical.
- Project management tools must support time tracking and client-facing deliverable approval
  to avoid billing disputes.

## Technology Philosophy
Meridian runs a lean, integrated stack. Tool sprawl is actively managed — any new SaaS must:
1. Integrate with the existing data layer without a custom middleware layer
2. Not duplicate functionality already covered by another tool
3. Support Microsoft Entra ID SSO (no exceptions for tools with >10 users)
4. Store data in UK or EU regions (no US-only hosting — GDPR hard block)

The CTO owns procurement decisions over £200/mo. CFO approves finance tool changes.
IT signs off on SSO integration before any tool goes live.

## Global Constraints (apply to ALL categories)
- **SOC2 Type II:** Mandatory for any tool handling client data. No exceptions.
- **GDPR / UK GDPR:** Data residency must be UK or EU. US-only storage is a hard block.
- **SSO / SAML:** Required for any tool with more than 10 user seats. Must work with
  Microsoft Entra ID (Azure AD). Tools that offer SSO only on a premium tier are evaluated
  at the premium tier price — if that erases the saving, verdict is STAY.
- **MFA enforcement:** Must be enforceable at admin level.
- **Audit logging:** Required for Finance, HR, and CRM. Logs must be exportable (PDF preferred).
- **Vendor Support SLA:** Business-hours (GMT) support required minimum; 24/7 for finance tools.
- **Maximum migration window:** 15 working hours per tool (staff time budget at £48/hr blended).
- **Minimum ROI to justify switching:** £1,200/yr net of amortised migration cost (3-yr basis).
- **Contract flexibility:** Monthly or annual preferred. 3-year lock-ins require board approval.

## Budget Cycles
- **Annual SaaS review:** October each year. New tool adoptions and cancellations take effect
  November 1st where possible (aligning with annual contract renewals).
- **Q1 freeze:** January–March is a budget freeze period. Non-critical procurement paused.
- **Ad-hoc approval:** CTO can approve tools up to £500/mo without board sign-off.
  Above £500/mo requires CFO co-approval.

## Staff Hourly Rate (for migration cost calculations)
- Operations / Admin: £38/hr
- Technical (IT/Dev): £65/hr
- Use blended rate of £48/hr unless role is specified
