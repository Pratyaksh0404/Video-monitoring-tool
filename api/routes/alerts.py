"""
api/routes/alerts.py
──────────────────────
Alert retrieval endpoints:

  GET /api/v1/alerts          — paginated alert history with filters
  GET /api/v1/alerts/live     — SSE stream (proxies existing /alerts/stream)

The alert log is the same list that the dashboard reads from.
No duplication — this route just exposes it as a proper REST API.
"""

from flask import Blueprint, jsonify, request, Response, stream_with_context
from api.middleware.auth import require_api_key

alerts_bp = Blueprint("alerts_v1", __name__)


def register(app, alert_log: list, alert_log_lock, sse_generator_fn):
    """
    alert_log:       the _alert_log list from flask_app
    alert_log_lock:  its threading.Lock()
    sse_generator_fn: the existing SSE generator function from flask_app
    """

    @alerts_bp.route("/api/v1/alerts", methods=["GET"])
    @require_api_key
    def get_alerts():
        """
        Return recent alerts with optional filtering.

        Query params:
          severity   str    Filter by severity: "high" | "medium" | "low"
          type       str    Filter by alert type substring (case-insensitive)
          limit      int    Max results to return (default 50, max 500)
          offset     int    Pagination offset (default 0)

        Response:
          {
            "alerts":  [...],
            "total":   int,
            "limit":   int,
            "offset":  int
          }
        """
        severity_filter = request.args.get("severity", "").lower()
        type_filter     = request.args.get("type", "").lower()
        limit           = min(int(request.args.get("limit", 50)), 500)
        offset          = int(request.args.get("offset", 0))

        with alert_log_lock:
            alerts = list(alert_log)

        # Apply filters
        if severity_filter:
            alerts = [a for a in alerts if a.get("severity") == severity_filter]
        if type_filter:
            alerts = [a for a in alerts if type_filter in a.get("type", "").lower()]

        total   = len(alerts)
        page    = alerts[offset: offset + limit]

        return jsonify({
            "alerts": page,
            "total":  total,
            "limit":  limit,
            "offset": offset,
        })

    @alerts_bp.route("/api/v1/alerts/live", methods=["GET"])
    @require_api_key
    def alerts_live():
        """
        Server-Sent Events stream of alerts in real time.
        Same as the existing /alerts/stream but under the v1 namespace
        and protected by API key.

        Connect with:
            const es = new EventSource('/api/v1/alerts/live?api_key=<key>');
            es.onmessage = e => console.log(JSON.parse(e.data));
        """
        return Response(
            stream_with_context(sse_generator_fn()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":  "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    app.register_blueprint(alerts_bp)