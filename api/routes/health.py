"""
api/routes/health.py
──────────────────────
GET /api/v1/health

Returns system status, model readiness, and uptime.
No auth required — safe to use as a load-balancer health check.
"""

import time
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

_start_time = time.time()


def register(app, pipeline_state: dict):
    """
    pipeline_state: shared dict updated by the pipeline thread with keys:
        running       bool
        source_label  str
        anomaly_ready bool
        clip_ready    bool
    """

    @health_bp.route("/api/v1/health", methods=["GET"])
    def health():
        uptime_secs = int(time.time() - _start_time)
        hours, rem  = divmod(uptime_secs, 3600)
        mins, secs  = divmod(rem, 60)

        return jsonify({
            "status":  "ok",
            "version": "1.0.0",
            "product": "NoviSentra",
            "uptime":  f"{hours:02d}:{mins:02d}:{secs:02d}",
            "uptime_seconds": uptime_secs,
            "pipeline": {
                "running":      pipeline_state.get("running", False),
                "source":       pipeline_state.get("source_label", "—"),
            },
            "models": {
                "weapon_fire":  pipeline_state.get("anomaly_ready", False),
                "behavior_clip": pipeline_state.get("clip_ready", False),
            },
        })

    app.register_blueprint(health_bp)