"""
api/routes/admin.py
──────────────────────
Admin-only operations:

  PATCH  /api/v1/admin/cameras/<cam_id>              — rename a camera
  PATCH  /api/v1/admin/cameras/<cam_id>/zones/<zone_id>  — rename a zone
  GET    /api/v1/admin/users                          — list users
  POST   /api/v1/admin/users                          — create a user
  DELETE /api/v1/admin/users/<user_id>                — remove a user

All gated by @require_admin — a logged-in viewer-role user gets 403.
"""

from flask import Blueprint, jsonify, request, g
from api.middleware.session_auth import require_admin
from novisentra.auth import create_user, list_users, delete_user, revoke_all_sessions_for_user

admin_bp = Blueprint("admin_v1", __name__)


def register(app, camera_manager):
    """camera_manager: the CameraManager singleton from flask_app.py"""

    @admin_bp.route("/api/v1/admin/cameras/<cam_id>", methods=["PATCH"])
    @require_admin
    def rename_camera(cam_id):
        """
        Body: {"name": "New Camera Name"}
        """
        body = request.get_json(silent=True) or {}
        new_name = body.get("name", "").strip()
        if not new_name:
            return jsonify({"error": "name is required"}), 400

        ok = camera_manager.rename_camera(cam_id, new_name)
        if not ok:
            return jsonify({"error": f"Camera '{cam_id}' not found"}), 404

        return jsonify({"ok": True, "cam_id": cam_id, "name": new_name})

    @admin_bp.route("/api/v1/admin/cameras/<cam_id>/zones/<zone_id>", methods=["PATCH"])
    @require_admin
    def rename_zone(cam_id, zone_id):
        """
        Body: {"label": "New Zone Label"}
        Zone `id` stays stable (historical alert data references zones by
        id) — only the human-readable label changes.
        """
        body = request.get_json(silent=True) or {}
        new_label = body.get("label", "").strip()
        if not new_label:
            return jsonify({"error": "label is required"}), 400

        ok = camera_manager.rename_zone(cam_id, zone_id, new_label)
        if not ok:
            return jsonify({"error": f"Zone '{zone_id}' not found on camera '{cam_id}'"}), 404

        return jsonify({"ok": True, "cam_id": cam_id, "zone_id": zone_id, "label": new_label})

    @admin_bp.route("/api/v1/admin/users", methods=["GET"])
    @require_admin
    def get_users():
        return jsonify({"users": list_users()})

    @admin_bp.route("/api/v1/admin/users", methods=["POST"])
    @require_admin
    def add_user():
        """
        Body: {"username": "...", "password": "...", "role": "admin"|"viewer"}
        """
        body = request.get_json(silent=True) or {}
        username = body.get("username", "").strip()
        password = body.get("password", "")
        role = body.get("role", "viewer")

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        if len(password) < 8:
            return jsonify({"error": "password must be at least 8 characters"}), 400

        try:
            user = create_user(username, password, role)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        return jsonify({"ok": True, "user": user.to_dict()}), 201

    @admin_bp.route("/api/v1/admin/users/<int:user_id>", methods=["DELETE"])
    @require_admin
    def remove_user(user_id):
        if g.current_user.id == user_id:
            return jsonify({"error": "Cannot delete your own account while logged in"}), 400

        revoke_all_sessions_for_user(user_id)
        ok = delete_user(user_id)
        if not ok:
            return jsonify({"error": f"User {user_id} not found"}), 404
        return jsonify({"ok": True})

    app.register_blueprint(admin_bp)
