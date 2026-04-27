from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_GENERATED_DIR = _PROJECT_ROOT / "training" / "generated"

DRIFT_CANARY_EXPECTED: dict[str, str] = {
    "crm_leadsphere_competitor_nearly_ready": "HOLD",
    "finance_brightbooks_pull_dominant": "SWITCH",
    "hr_teamrise_shelfware_case": "SWITCH",
    "project_mgmt_opscanvas_contract_renewal_hold": "HOLD",
    "crm_closerhub_negative_signal_buried": "STAY",
    "finance_ledgerflow_pull_dominant": "SWITCH",
    "project_mgmt_flowboard_shelfware_case": "SWITCH",
    "hr_workforge_fluff_update": "STAY",
    "analytics_pulsemetrics_irrelevant_change": "STAY",
    "finance_exactspend_competitor_nearly_ready": "HOLD",
}

DRIFT_CANARY_STEMS = frozenset(DRIFT_CANARY_EXPECTED)
DRIFT_CANARY_FILENAMES = frozenset(f"{stem}.json" for stem in DRIFT_CANARY_STEMS)
DRIFT_CANARY_FILES = tuple(_GENERATED_DIR / filename for filename in sorted(DRIFT_CANARY_FILENAMES))