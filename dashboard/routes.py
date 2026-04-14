import subprocess
import sys
import threading
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


@bp.route("/api/review-queue")
def api_review_queue():
    return jsonify(dl.get_review_queue(log_path=dl._DRIFT_LOG, summaries_path=dl._SUMMARIES))


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


def _run_and_unlock(cmd, cwd, env, lock_path):
    """Run a subprocess and clear the model lock when done."""
    try:
        subprocess.run(cmd, cwd=cwd, env=env)
    finally:
        lock_path.unlink(missing_ok=True)


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
    threading.Thread(
        target=_run_and_unlock,
        args=(cmd, str(_PROJECT_ROOT), _subprocess_env(dry_run), dl._LOCK_FILE),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "task": "eval"})


@bp.route("/api/run-summaries", methods=["POST"])
def api_run_summaries():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    dl.set_model_busy("summarise", lock_path=dl._LOCK_FILE)
    threading.Thread(
        target=_run_and_unlock,
        args=(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "summarise.py")],
            str(_PROJECT_ROOT),
            _subprocess_env(False),
            dl._LOCK_FILE,
        ),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "task": "summarise"})


@bp.route("/api/rankings")
def api_rankings():
    return jsonify(dl.get_rankings(
        log_path=dl._DRIFT_LOG,
        summaries_path=dl._SUMMARIES,
    ))


@bp.route("/api/holds")
def api_holds():
    return jsonify(dl.get_active_holds(
        log_path=dl._DRIFT_LOG,
        summaries_path=dl._SUMMARIES,
        outputs_dir=dl._OUTPUTS_DIR,
    ))


@bp.route("/api/history")
def api_history():
    return jsonify(dl.get_all_verdicts(
        log_path=dl._DRIFT_LOG,
        summaries_path=dl._SUMMARIES,
    ))


@bp.route("/api/run-batch", methods=["POST"])
def api_run_batch():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    body = request.get_json(silent=True) or {}
    clean = bool(body.get("clean", False))
    inbox_dir = _PROJECT_ROOT / "market_inbox"
    inbox_files = sorted(inbox_dir.glob("*.md"))
    if not inbox_files:
        return jsonify({"error": "No inbox files in market_inbox/"}), 404
    dl.set_model_busy("batch-eval", lock_path=dl._LOCK_FILE)

    def _batch():
        try:
            if clean:
                dl.clean_outputs()
            for f in inbox_files:
                subprocess.run(
                    [sys.executable, "-m", "agent.agent", str(f)],
                    cwd=str(_PROJECT_ROOT),
                    env=_subprocess_env(False),
                )
        finally:
            dl._LOCK_FILE.unlink(missing_ok=True)

    threading.Thread(target=_batch, daemon=True).start()
    return jsonify({"ok": True, "task": "batch-eval", "count": len(inbox_files)})


def _subprocess_env(dry_run: bool) -> dict:
    import os
    env = os.environ.copy()
    if dry_run:
        env["AGENT_DRY_RUN"] = "true"
    return env


@bp.route("/api/feedback-queue")
def api_feedback_queue():
    return jsonify(dl.get_feedback_queue(log_path=dl._DRIFT_LOG))


@bp.route("/api/retrain", methods=["POST"])
def api_retrain():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    dl.set_model_busy("retrain", lock_path=dl._LOCK_FILE)

    def _retrain():
        try:
            env = _subprocess_env(False)
            # Step 1: Harvest feedback pairs
            subprocess.run(
                [sys.executable, str(_PROJECT_ROOT / "training" / "feedback_harvester.py")],
                cwd=str(_PROJECT_ROOT),
                env=env,
            )
            pairs_path = _PROJECT_ROOT / "training" / "feedback_pairs.jsonl"
            if not pairs_path.exists() or pairs_path.stat().st_size == 0:
                return  # No pairs to train on
            # Step 2: DPO training with canary gate
            subprocess.run(
                [sys.executable, str(_PROJECT_ROOT / "training" / "dpo_train.py")],
                cwd=str(_PROJECT_ROOT),
                env=env,
            )
        finally:
            dl._LOCK_FILE.unlink(missing_ok=True)

    threading.Thread(target=_retrain, daemon=True).start()
    return jsonify({"ok": True, "task": "retrain"})


@bp.route("/api/run-canary", methods=["POST"])
def api_run_canary():
    if dl.is_model_busy(lock_path=dl._LOCK_FILE):
        return jsonify({"error": "Model is busy"}), 409
    dl.set_model_busy("canary", lock_path=dl._LOCK_FILE)
    threading.Thread(
        target=_run_and_unlock,
        args=(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "drift_check.py")],
            str(_PROJECT_ROOT),
            _subprocess_env(False),
            dl._LOCK_FILE,
        ),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "task": "canary"})
