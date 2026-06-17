"""
api/routes/analytics.py
─────────────────────────
Analytics endpoints:

  GET /api/v1/analytics/summary    — compliance scores, dwell times, occupancy
  GET /api/v1/analytics/guards     — per-guard stats
  GET /api/v1/analytics/snapshots  — snapshot listing with metadata
"""

from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key

analytics_bp = Blueprint("analytics_v1", __name__)


def register(app, streamer, snap_mgr, alert_log, alert_log_lock):

    @analytics_bp.route("/api/v1/analytics/summary", methods=["GET"])
    @require_api_key
    def analytics_summary():
        """
        Return live analytics summary.

        Response:
          {
            "people_count":      int,
            "guards_detected":   int,
            "active_violations": int,
            "alerts_today":      int,
            "compliance_scores": {"Name": float, ...},
            "dwell_times":       {"Name": int_seconds, ...},
            "fps":               float,
            "source_label":      str
          }
        """
        stats = dict(streamer.stats)
        return jsonify({
            "people_count":      stats.get("people_count", 0),
            "guards_detected":   stats.get("guards_detected", 0),
            "active_violations": stats.get("active_violations", 0),
            "alerts_today":      stats.get("alerts_today", 0),
            "compliance_scores": stats.get("compliance_scores", {}),
            "dwell_times":       stats.get("dwell_times", {}),
            "fps":               stats.get("fps", 0),
            "source_label":      stats.get("source_label", "—"),
        })

    @analytics_bp.route("/api/v1/analytics/guards", methods=["GET"])
    @require_api_key
    def analytics_guards():
        """
        Per-guard breakdown from the current session's alert log.

        Response:
          {
            "guards": [
              {
                "name":          "Pratyaksh",
                "alert_count":   12,
                "high_count":    3,
                "dwell_seconds": 1840,
                "compliance":    87.5,
                "alerts":        [{...}, ...]
              }
            ]
          }
        """
        stats = dict(streamer.stats)
        compliance = stats.get("compliance_scores", {})
        dwell      = stats.get("dwell_times", {})

        with alert_log_lock:
            all_alerts = list(alert_log)

        # Group by guard_id (skip Camera/Post)
        by_guard: dict = {}
        for a in all_alerts:
            gid = a.get("guard_id", "")
            if gid in ("Camera", "Post", ""):
                continue
            if gid not in by_guard:
                by_guard[gid] = {"name": gid, "alerts": [], "high_count": 0}
            by_guard[gid]["alerts"].append(a)
            if a.get("severity") == "high":
                by_guard[gid]["high_count"] += 1

        result = []
        for name, data in by_guard.items():
            result.append({
                "name":          name,
                "alert_count":   len(data["alerts"]),
                "high_count":    data["high_count"],
                "dwell_seconds": dwell.get(name, 0),
                "compliance":    compliance.get(name, 0.0),
                "alerts":        data["alerts"][-20:],  # last 20 for this guard
            })

        return jsonify({"guards": result})

    @analytics_bp.route("/api/v1/analytics/snapshots", methods=["GET"])
    @require_api_key
    def analytics_snapshots():
        """
        List snapshots from the current session.

        Query params:
          limit   int  Max snapshots to return (default 50)

        Response:
          {
            "snapshots": [
              {
                "filename": "090012_guard_sleeping_1.jpg",
                "label":    "Guard Sleeping",
                "size_kb":  82.4,
                "ts":       "09:00:12",
                "url":      "/api/snapshots/090012_guard_sleeping_1.jpg"
              }
            ]
          }
        """
        limit = int(request.args.get("limit", 50))
        snaps = snap_mgr.list_snapshots()[:limit]

        # Add URL for direct fetch via existing /api/snapshots/<filename> route
        for s in snaps:
            s["url"] = f"/api/snapshots/{s['filename']}"

        return jsonify({"snapshots": snaps})

    app.register_blueprint(analytics_bp)