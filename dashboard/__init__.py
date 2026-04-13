from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    from dashboard.routes import bp
    app.register_blueprint(bp)

    _migrate_competitor_files()

    return app


def _migrate_competitor_files() -> None:
    """Add scraper_url field to any competitor JSON files that lack it."""
    import json
    from pathlib import Path
    competitors_dir = Path(__file__).parent.parent / "data" / "competitors"
    if not competitors_dir.exists():
        return
    for comp_file in competitors_dir.rglob("*.json"):
        try:
            data = json.loads(comp_file.read_text(encoding="utf-8"))
            if "scraper_url" not in data:
                data["scraper_url"] = ""
                comp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass
