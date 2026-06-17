"""
api/routes/config.py
──────────────────────
Config endpoints:

  GET   /api/v1/config            — read current rules_config.yaml
  PATCH /api/v1/config            — partial update (same as existing /api/config/update)
  GET   /api/v1/config/profiles   — list available profiles
  POST  /api/v1/config/profile    — activate a profile
"""

from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key

config_bp = Blueprint("config_v1", __name__)

_AVAILABLE_PROFILES = {
    "guard_monitoring": {
        "name":        "Guard Monitoring",
        "description": "Security guard compliance — sleeping, phone, patrol, weapon, fire.",
        "detectors":   ["person", "face", "behavior", "weapon", "fire",
                        "tamper", "crowd", "loitering", "fight"],
    },
    "retail_analytics": {
        "name":        "Retail Analytics",
        "description": "Customer tracking — dwell time, crowd, queue monitoring.",
        "detectors":   ["person", "crowd", "loitering", "tamper"],
        "status":      "coming_soon",
    },
    "warehouse_ops": {
        "name":        "Warehouse Operations",
        "description": "Safety compliance — uniform detection, supervisor presence.",
        "detectors":   ["person", "face", "behavior", "crowd", "tamper"],
        "status":      "coming_soon",
    },
    "bank_security": {
        "name":        "Bank Security",
        "description": "Vault monitoring, occupancy, camera tamper.",
        "detectors":   ["person", "face", "weapon", "tamper", "crowd"],
        "status":      "coming_soon",
    },
}


def register(app, existing_config_update_fn, existing_config_read_fn, main_web):
    """
    existing_config_update_fn: the flask route handler for /api/config/update
    existing_config_read_fn:   the flask route handler for /api/config
    main_web:                  the main_web module (has reload_config())
    """

    @config_bp.route("/api/v1/config", methods=["GET"])
    @require_api_key
    def get_config():
        """Return current rules_config.yaml as JSON."""
        return existing_config_read_fn()

    @config_bp.route("/api/v1/config", methods=["PATCH"])
    @require_api_key
    def update_config():
        """
        Partial update of rules_config.yaml. Deep-merges the provided keys.
        Changes are applied live to the running pipeline without restart.

        Body: any subset of rules_config.yaml sections.

        Example:
          PATCH /api/v1/config
          {
            "behavior": {
              "clip_thresholds": {"SLEEPING": 0.80}
            },
            "alerts": {
              "cooldown": 45
            }
          }
        """
        return existing_config_update_fn()

    @config_bp.route("/api/v1/config/profiles", methods=["GET"])
    @require_api_key
    def list_profiles():
        """List all available use-case profiles."""
        return jsonify({
            "profiles": _AVAILABLE_PROFILES,
            "active":   "guard_monitoring",   # always active for now
        })

    @config_bp.route("/api/v1/config/profile", methods=["POST"])
    @require_api_key
    def set_profile():
        """
        Activate a profile. Currently only guard_monitoring is fully implemented.

        Body: {"profile": "guard_monitoring"}
        """
        body = request.get_json(silent=True) or {}
        profile = body.get("profile", "")

        if profile not in _AVAILABLE_PROFILES:
            return jsonify({
                "error": f"Unknown profile '{profile}'",
                "available": list(_AVAILABLE_PROFILES.keys())
            }), 400

        meta = _AVAILABLE_PROFILES[profile]
        if meta.get("status") == "coming_soon":
            return jsonify({
                "error":   f"Profile '{profile}' is not yet implemented.",
                "message": "Currently available: guard_monitoring",
            }), 501

        return jsonify({
            "ok":      True,
            "profile": profile,
            "message": f"Profile '{meta['name']}' is already active.",
        })

    app.register_blueprint(config_bp)