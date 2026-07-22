"""
api/routes/config.py
──────────────────────
Config endpoints:

  GET   /api/v1/config            — read current rules_config.yaml
  PATCH /api/v1/config            — partial update (same as existing /api/config/update)
  GET   /api/v1/config/profiles   — list available profiles (from novisentra.profiles registry)
  POST  /api/v1/config/profile    — activate a profile for EVERY camera in this deployment

── This actually changes camera behavior now ────────────────────────────────
POST /api/v1/config/profile used to only update a cosmetic "which profile
is active" label — it never touched any camera's real, running profile.
It now calls camera_manager.set_camera_profile() for every configured
camera, which writes into the live camera→profile registry in
alerts/alert_manager.py that main_web.py's capability checks actually
read. This is what makes a switch take effect on an ALREADY-RUNNING
camera within one frame, with no restart needed — matches this
deployment's business model (one deployment = one profile applied
uniformly across every camera it has).

── Licensed-profile lock (multi-tenant business model) ──────────────────────
Set the environment variable NOVISENTRA_LICENSED_PROFILE on a CUSTOMER
deployment to lock that entire installation to ONE profile — the one
they're licensed for. Once set:
  - GET /api/v1/config/profiles only returns that one profile (others
    aren't even visible, let alone selectable)
  - POST /api/v1/config/profile rejects any value other than the
    licensed one
  - The admin panel's profile switcher only shows that one profile card

Leave NOVISENTRA_LICENSED_PROFILE unset for OUR internal/dev/demo
deployments — that gives full access to every registered profile, for
testing and for showing prospective customers what each scenario looks
like before they buy one.

  export NOVISENTRA_LICENSED_PROFILE=retail_analytics   # customer's .env
  # (unset)                                              # our internal builds
"""

import os
from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key
from novisentra.profiles import get_profile, list_profiles, is_valid_profile

config_bp = Blueprint("config_v1", __name__)


def _licensed_profile_id():
    """Returns the locked profile id for this deployment, or None if
    this is an unrestricted (internal/dev) build."""
    return os.environ.get("NOVISENTRA_LICENSED_PROFILE", "").strip() or None


def register(app, existing_config_update_fn, existing_config_read_fn, main_web,
             camera_manager=None, send_session_report_fn=None):
    """
    existing_config_update_fn: the flask route handler for /api/config/update
    existing_config_read_fn:   the flask route handler for /api/config
    main_web:                  the main_web module (has reload_config())
    camera_manager:             the CameraManager singleton — needed so a
    send_session_report_fn:    flask_app.send_session_report — called on a
                                successful profile switch (background
                                thread, non-blocking) so the OUTGOING
                                profile's segment gets its shift-report
                                email now instead of only at final
                                process shutdown. Per-profile session LOG
                                FILES no longer depend on this at all
                                (2026-07 fix: those write immediately from
                                alert_manager.send_alert()) — this is only
                                about the email cadence. If not passed,
                                falls back to the old behavior (email
                                only at shutdown), logged clearly.
                                profile switch actually reaches every
                                camera, not just a cosmetic label. If not
                                passed, the endpoint falls back to the
                                old cosmetic-only behavior (logged clearly
                                so this misconfiguration isn't silent).
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
        """
        return existing_config_update_fn()

    @config_bp.route("/api/v1/config/profiles", methods=["GET"])
    @require_api_key
    def list_profiles_route():
        """
        List available use-case profiles. On a licensed customer
        deployment, this returns ONLY their one licensed profile —
        they never see that other scenarios exist on this system.
        """
        licensed = _licensed_profile_id()
        all_profiles = list_profiles()

        if licensed:
            if licensed not in all_profiles:
                return jsonify({
                    "profiles": {},
                    "active": None,
                    "error": f"NOVISENTRA_LICENSED_PROFILE='{licensed}' is not "
                             f"a valid profile id.",
                }), 500
            return jsonify({
                "profiles": {licensed: all_profiles[licensed]},
                "active": licensed,
                "licensed": True,
            })

        return jsonify({
            "profiles": all_profiles,
            "active": _get_active_profile_id(),
            "licensed": False,
        })

    @config_bp.route("/api/v1/config/profile", methods=["POST"])
    @require_api_key
    def set_profile():
        """
        Activate a profile for EVERY camera in this deployment — this now
        actually reaches the running cameras (see module docstring), not
        just a display label. Takes effect within one frame on each
        camera, no restart.

        On a licensed customer deployment, only the licensed profile
        itself is a valid value — attempting to switch to anything else
        is rejected with 403.
        """
        body = request.get_json(silent=True) or {}
        profile_id = body.get("profile", "")

        licensed = _licensed_profile_id()
        if licensed and profile_id != licensed:
            return jsonify({
                "error": f"This deployment is licensed for '{licensed}' only. "
                         f"Cannot switch to '{profile_id}'.",
            }), 403

        if not is_valid_profile(profile_id):
            return jsonify({
                "error": f"Unknown profile '{profile_id}'",
                "available": list(list_profiles().keys()),
            }), 400

        profile = get_profile(profile_id)

        from novisentra.capabilities.registry import get_capability
        runnable, pending = [], []
        for cap_key in profile.capabilities:
            cap = get_capability(cap_key)
            (runnable if cap["status"] == "available" else pending).append(cap_key)

        _set_active_profile_id(profile_id)

        cameras_updated = []
        if camera_manager is not None:
            for cam_id in list(camera_manager.cameras.keys()):
                camera_manager.set_camera_profile(cam_id, profile_id)
                cameras_updated.append(cam_id)
        else:
            # No camera_manager was passed to register() — the switch only
            # updated the display label, not any real camera. Surface this
            # loudly rather than silently pretending it worked.
            import logging
            logging.getLogger("NoviSentra.flask_app").warning(
                "POST /api/v1/config/profile: camera_manager not wired in — "
                "profile label updated but NO camera was actually switched. "
                "Check flask_app.py's call to api.routes.config.register()."
            )

        # Close out the OUTGOING profile's segment now, rather than
        # waiting for final shutdown. Bug found 2026-07: across a real
        # 4-profile test run, this never fired mid-run at all — only the
        # final Ctrl+C triggered it — so only one shift-report email ever
        # went out for the whole run instead of one per profile segment.
        # Per-profile session LOG FILES don't depend on this (they write
        # immediately per-alert now), so this is purely about email
        # cadence; safe to run in the background and not block the
        # response on it.
        if send_session_report_fn is not None and cameras_updated:
            import threading
            threading.Thread(target=send_session_report_fn, daemon=True).start()
        elif send_session_report_fn is None:
            import logging
            logging.getLogger("NoviSentra.flask_app").warning(
                "POST /api/v1/config/profile: send_session_report_fn not "
                "wired in — shift-report emails will only be sent at final "
                "process shutdown, not per profile switch."
            )

        return jsonify({
            "ok": True,
            "profile": profile_id,
            "message": (f"Profile '{profile.name}' activated on "
                       f"{len(cameras_updated)} camera(s) — takes effect within "
                       f"a second, no restart needed."
                       if cameras_updated else
                       f"Profile '{profile.name}' label set, but no cameras "
                       f"were actually switched (camera_manager not wired) — "
                       f"this is a configuration problem, not a click-and-wait "
                       f"situation."),
            "cameras_updated": cameras_updated,
            "capabilities_running": runnable,
            "capabilities_pending": pending,
        })

    app.register_blueprint(config_bp)


# ── Active profile state ─────────────────────────────────────────────────────
_active_profile_id = os.environ.get("NOVISENTRA_LICENSED_PROFILE", "").strip() or "guard_monitoring"


def _get_active_profile_id() -> str:
    return _active_profile_id


def _set_active_profile_id(profile_id: str) -> None:
    global _active_profile_id
    _active_profile_id = profile_id
