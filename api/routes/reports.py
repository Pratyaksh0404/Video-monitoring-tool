"""
api/routes/reports.py
───────────────────────
Report endpoints:

  GET  /api/v1/reports/generate  — generate and return HTML shift report
  POST /api/v1/reports/send      — email the shift report now
"""

from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key

reports_bp = Blueprint("reports_v1", __name__)


def register(app, send_session_report_fn, generate_report_fn, streamer, alert_log, alert_log_lock):

    @reports_bp.route("/api/v1/reports/generate", methods=["GET"])
    @require_api_key
    def generate_report():
        """
        Generate an HTML shift report for the current session and return it.
        Does NOT send an email — use /send for that.

        Response: HTML document (Content-Type: text/html)
        """
        import time
        from flask import Response

        with alert_log_lock:
            alerts = list(alert_log)

        stats = dict(streamer.stats)
        html  = generate_report_fn(
            alert_log=alerts,
            stats=stats,
            session_start=time.time() - 3600,   # approximate
        )
        return Response(html, mimetype="text/html")

    @reports_bp.route("/api/v1/reports/send", methods=["POST"])
    @require_api_key
    def send_report():
        """
        Trigger an email shift report for the current session.
        Same as clicking 'Send Report' in the dashboard.

        Body (JSON, all optional):
          {}   — just triggers the send

        Response:
          {"ok": true}  or  {"error": "..."}
        """
        try:
            import threading
            threading.Thread(target=send_session_report_fn, daemon=True).start()
            return jsonify({"ok": True, "message": "Shift report queued for delivery."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    app.register_blueprint(reports_bp)