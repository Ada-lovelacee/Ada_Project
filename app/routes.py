from flask import Blueprint, current_app, jsonify, render_template, send_from_directory

from .fl_dashboard import build_overview_payload


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template(
        "index.html",
        refresh_ms=current_app.config["DASHBOARD_REFRESH_MS"],
    )


@main_bp.route("/api/overview")
def api_overview():
    payload = build_overview_payload(
        sync_code_dir=current_app.config["SYNC_CODE_DIR"],
        results_dir=current_app.config["SYNC_RESULTS_DIR"],
    )
    return jsonify(payload)


@main_bp.route("/artifacts/<path:filename>")
def artifact_file(filename: str):
    return send_from_directory(current_app.config["SYNC_RESULTS_DIR"], filename)
