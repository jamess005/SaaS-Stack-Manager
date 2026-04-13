import subprocess
import sys
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

import dashboard.data_layer as dl

bp = Blueprint("dashboard", __name__)

_PROJECT_ROOT = Path(__file__).parent.parent


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/stats")
def api_stats():
    return jsonify(dl.get_stats(log_path=dl._DRIFT_LOG, competitors_dir=dl._COMPETITORS_DIR))


@bp.route("/api/verdicts")
def api_verdicts():
    n = request.args.get("n", 10, type=int)
    return jsonify(dl.get_recent_verdicts(
        n=n,
        log_path=dl._DRIFT_LOG,
        outputs_dir=dl._OUTPUTS_DIR,
        summaries_path=dl._SUMMARIES,
    ))


@bp.route("/api/competitors")
def api_competitors():
    return jsonify(dl.get_competitors(competitors_dir=dl._COMPETITORS_DIR))


@bp.route("/api/competitors", methods=["POST"])
def api_add_competitor():
    body = request.get_json(silent=True) or {}
    required = ["category", "slug", "name", "monthly_cost_gbp"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    data = dl.add_competitor(
        category=body["category"],
        slug=body["slug"],
        name=body["name"],
        monthly_cost_gbp=float(body["monthly_cost_gbp"]),
        scraper_url=body.get("scraper_url", ""),
        competitors_dir=dl._COMPETITORS_DIR,
    )
    return jsonify(data), 201


@bp.route("/api/feedback", methods=["POST"])
def api_feedback():
    body = request.get_json(silent=True) or {}
    required = ["memo_filename", "correct", "stated_verdict", "actual_verdict"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    dl.record_feedback(
        memo_filename=body["memo_filename"],
        correct=bool(body["correct"]),
        stated_verdict=body["stated_verdict"],
        actual_verdict=body["actual_verdict"],
        note=body.get("note", ""),
        log_path=dl._DRIFT_LOG,
    )
    return jsonify({"ok": True})


@bp.route("/api/health")
def api_health():
    return jsonify(dl.get_health(log_path=dl._DRIFT_LOG))


@bp.route("/api/status")
def api_status():
    busy = dl.is_model_busy(lock_path=dl._LOCK_FILE)
    task = ""
    if busy:
        try:
            task = dl._LOCK_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            task = "unknown"
    return jsonify({"busy": busy, "task": task})


@bp.route("/api/run-eval", methods=["POST"])
def api_run_eval():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    body = request.get_json(silent=True) or {}
    inbox = body.get("inbox_file", "")
    dry_run = bool(body.get("dry_run", False))
    if not inbox and not dry_run:
        return jsonify({"error": "inbox_file required"}), 400
    dl.set_model_busy("eval", lock_path=dl._LOCK_FILE)
    cmd = [sys.executable, "-m", "agent.agent"]
    if dry_run:
        cmd.append("--dry-run")
    if inbox:
        cmd.append(inbox)
    subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=_subprocess_env(dry_run),
    )
    return jsonify({"ok": True, "task": "eval"})


@bp.route("/api/run-summaries", methods=["POST"])
def api_run_summaries():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    dl.set_model_busy("summarise", lock_path=dl._LOCK_FILE)
    subprocess.Popen(
        [sys.executable, str(_PROJECT_ROOT / "scripts" / "summarise.py")],
        cwd=str(_PROJECT_ROOT),
    )
    return jsonify({"ok": True, "task": "summarise"})


def _subprocess_env(dry_run: bool) -> dict:
    import os
    env = os.environ.copy()
    if dry_run:
        env["AGENT_DRY_RUN"] = "true"
    return env
