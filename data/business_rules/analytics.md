# Business Rules — Analytics Category

## Must-Have Features
- Real-time or near-real-time data connectors (NexusCRM, VaultLedger, TaskBridge)
- Self-serve embedded reporting for client delivery (white-label)
- Mobile-responsive dashboards
- Data lineage and audit trail (ISO audit requirement — currently failing)
- Dashboard builder without SQL requirement for business users

## Nice-to-Have Features
- AI anomaly detection and alerting
- Natural language query interface
- Predictive pipeline and revenue forecasting
- Automated insight digests (weekly email to stakeholders)
- dbt or warehouse-native modelling layer

## Needs / Pain Points (active)
- Data lineage absent — compliance team has flagged for upcoming ISO audit
- NexusCRM API rate limit causing 4-hour staleness in sales dashboards
- No mobile view — client dashboards unusable on phones
- Embedded client reporting requires manual PDF export — time consuming

## Compliance
- Client-facing dashboards — SOC2 Type II mandatory
- Data lineage / audit trail required for ISO27001 compliance
- UK/EU data residency

## Pricing Sensitivity
- Current: £340/mo (12 seats)
- Acceptable range: £280–£500/mo
- High value placed on client-facing features — willingness to pay premium for embedded reporting

## Migration Notes
- Dashboard definitions must be exportable or re-buildable
- 3 white-label client dashboards must remain live during migration
- Connector rebuild time must be included in migration estimate
