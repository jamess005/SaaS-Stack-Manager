import json
import pytest
from pathlib import Path
from dashboard.data_layer import get_stats, get_recent_verdicts


@pytest.fixture
def drift_log(tmp_path):
    log = tmp_path / "drift_log.jsonl"
    records = [
        {"ts": "2026-04-12T10:00:00+00:00", "type": "live_run", "category": "finance",
         "competitor": "ledgerflow", "verdict": "SWITCH", "format_valid": True,
         "validation_attempts": 1, "verdict_token_prob": 0.91},
        {"ts": "2026-04-12T11:00:00+00:00", "type": "live_run", "category": "hr",
         "competitor": "workforge", "verdict": "HOLD", "format_valid": True,
         "validation_attempts": 1, "verdict_token_prob": 0.78},
        {"ts": "2026-04-12T12:00:00+00:00", "type": "live_run", "category": "crm",
         "competitor": "velocitycrm", "verdict": "STAY", "format_valid": True,
         "validation_attempts": 1, "verdict_token_prob": 0.55},
        {"ts": "2026-04-12T13:00:00+00:00", "type": "accuracy_check",
         "correct": 4, "total": 4, "accuracy": 1.0, "results": []},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return log


def test_get_stats_counts_verdicts(drift_log):
    stats = get_stats(log_path=drift_log)
    assert stats["switch"] == 1
    assert stats["hold"] == 1
    assert stats["stay"] == 1
    assert stats["total_evaluated"] == 3


def test_get_stats_empty_log(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    stats = get_stats(log_path=log)
    assert stats["switch"] == 0
    assert stats["hold"] == 0
    assert stats["stay"] == 0


def test_get_stats_missing_log(tmp_path):
    stats = get_stats(log_path=tmp_path / "nonexistent.jsonl")
    assert stats["total_evaluated"] == 0


def test_get_recent_verdicts_order(drift_log, tmp_path):
    verdicts = get_recent_verdicts(n=10, log_path=drift_log,
                                   outputs_dir=tmp_path, summaries_path=tmp_path / "s.json")
    # Most recent first
    assert verdicts[0]["competitor"] == "velocitycrm"
    assert verdicts[1]["competitor"] == "workforge"
    assert verdicts[2]["competitor"] == "ledgerflow"


def test_get_recent_verdicts_respects_n(drift_log, tmp_path):
    verdicts = get_recent_verdicts(n=2, log_path=drift_log,
                                   outputs_dir=tmp_path, summaries_path=tmp_path / "s.json")
    assert len(verdicts) == 2


def test_get_recent_verdicts_includes_confidence(drift_log, tmp_path):
    verdicts = get_recent_verdicts(n=10, log_path=drift_log,
                                   outputs_dir=tmp_path, summaries_path=tmp_path / "s.json")
    switch_v = next(v for v in verdicts if v["verdict"] == "SWITCH")
    assert switch_v["verdict_token_prob"] == 0.91


# ── Task 3 tests ──────────────────────────────────────────────────────────────

from dashboard.data_layer import (
    get_competitors, record_feedback, get_health,
    is_model_busy, set_model_busy, clear_model_busy,
)


@pytest.fixture
def competitors_dir(tmp_path):
    crm = tmp_path / "crm"
    crm.mkdir()
    (crm / "testcrm.json").write_text(json.dumps({
        "name": "TestCRM", "category": "crm", "monthly_cost_gbp": 200,
        "features": ["feature a", "feature b"],
        "known_limitations": ["no sso"],
        "compliance": {"sso": False, "soc2_type2": True},
        "scraper_url": "https://testcrm.io/changelog",
    }), encoding="utf-8")
    return tmp_path


def test_get_competitors_returns_list(competitors_dir):
    comps = get_competitors(competitors_dir=competitors_dir)
    assert len(comps) == 1
    assert comps[0]["name"] == "TestCRM"
    assert comps[0]["slug"] == "testcrm"
    assert comps[0]["category"] == "crm"


def test_record_feedback_appends_record(tmp_path):
    log = tmp_path / "drift_log.jsonl"
    record_feedback(
        memo_filename="2026-04-12-finance-ledgerflow.md",
        correct=True,
        stated_verdict="SWITCH",
        actual_verdict="SWITCH",
        note="",
        log_path=log,
    )
    records = json.loads(log.read_text().strip())
    assert records["type"] == "human_feedback"
    assert records["correct"] is True
    assert records["memo_filename"] == "2026-04-12-finance-ledgerflow.md"


def test_record_feedback_with_note(tmp_path):
    log = tmp_path / "drift_log.jsonl"
    record_feedback(
        memo_filename="2026-04-12-crm-velocitycrm.md",
        correct=False,
        stated_verdict="SWITCH",
        actual_verdict="STAY",
        note="Missed compliance block",
        log_path=log,
    )
    record = json.loads(log.read_text().strip())
    assert record["correct"] is False
    assert record["note"] == "Missed compliance block"
    assert record["actual_verdict"] == "STAY"


def test_get_health_last_accuracy_check(drift_log, tmp_path):
    health = get_health(log_path=drift_log)
    assert health["last_accuracy"]["accuracy"] == 1.0
    assert health["last_accuracy"]["correct"] == 4
    assert health["last_accuracy"]["total"] == 4


def test_get_health_no_checks(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text('{"type":"live_run","verdict":"SWITCH","ts":"2026-01-01T00:00:00+00:00","category":"crm","competitor":"x","format_valid":true,"validation_attempts":1}\n')
    health = get_health(log_path=log)
    assert health["last_accuracy"] is None


def test_model_lock(tmp_path):
    lock = tmp_path / ".model_lock"
    assert not is_model_busy(lock_path=lock)
    set_model_busy("eval", lock_path=lock)
    assert is_model_busy(lock_path=lock)
    clear_model_busy(lock_path=lock)
    assert not is_model_busy(lock_path=lock)


# ── Task 4 tests ──────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import create_app
import dashboard.data_layer as dl


@pytest.fixture
def app(tmp_path, drift_log, competitors_dir, monkeypatch):
    monkeypatch.setattr(dl, "_DRIFT_LOG", drift_log)
    monkeypatch.setattr(dl, "_OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dl, "_SUMMARIES", tmp_path / "summaries.json")
    monkeypatch.setattr(dl, "_LOCK_FILE", tmp_path / ".model_lock")
    monkeypatch.setattr(dl, "_COMPETITORS_DIR", competitors_dir)
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200


def test_api_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.get_json()
    assert data["switch"] == 1
    assert data["hold"] == 1
    assert data["stay"] == 1


def test_api_verdicts(client):
    r = client.get("/api/verdicts?n=10")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    assert data[0]["verdict"] == "STAY"   # most recent first


def test_api_competitors(client):
    r = client.get("/api/competitors")
    assert r.status_code == 200
    data = r.get_json()
    assert data[0]["name"] == "TestCRM"


def test_api_feedback_post(client):
    payload = {
        "memo_filename": "2026-04-12-finance-ledgerflow.md",
        "correct": True,
        "stated_verdict": "SWITCH",
        "actual_verdict": "SWITCH",
        "note": "",
    }
    r = client.post("/api/feedback", json=payload)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_api_feedback_missing_field(client):
    r = client.post("/api/feedback", json={"correct": True})
    assert r.status_code == 400


def test_api_status_not_busy(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.get_json()["busy"] is False


def test_api_run_eval_busy(client, tmp_path, monkeypatch):
    lock = tmp_path / ".model_lock"
    lock.write_text("eval")
    monkeypatch.setattr(dl, "_LOCK_FILE", lock)
    r = client.post("/api/run-eval", json={"dry_run": True})
    assert r.status_code == 409


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert "last_accuracy" in data
    assert data["last_accuracy"]["accuracy"] == 1.0
