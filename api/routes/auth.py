"""
api/routes/auth.py
─────────────────────
Login, logout, and "who am I" endpoints for the dashboard.

  POST /api/v1/auth/login    — {username, password} → sets session cookie
  POST /api/v1/auth/logout   — clears session cookie, revokes token
  GET  /api/v1/auth/me       — current logged-in user info (or 401)
"""

from flask import Blueprint, jsonify, request, make_response
from novisentra.auth import get_user_by_username, verify_password, create_session, revoke_session
from api.middleware.session_auth import get_current_user, SESSION_COOKIE_NAME

auth_bp = Blueprint("auth_v1", __name__)


def register(app):

    @auth_bp.route("/api/v1/auth/login", methods=["POST"])
    def login():
        """
        Body: {"username": "...", "password": "..."}
        On success, sets an HttpOnly session cookie and returns user info.
        """
        body = request.get_json(silent=True) or {}
        username = body.get("username", "").strip()
        password = body.get("password", "")

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        user = get_user_by_username(username)
        if not user or not verify_password(user, password):
            # Same error for "no such user" and "wrong password" — don't
            # leak which one it was, standard practice against user enumeration.
            return jsonify({"error": "Invalid username or password"}), 401

        token = create_session(user.id)

        resp = make_response(jsonify({"ok": True, "user": user.to_dict()}))
        resp.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            httponly=True,
            samesite="Lax",
            secure=False,   # set True once deployed behind HTTPS (Hetzner + reverse proxy)
            max_age=7 * 24 * 3600,
        )
        return resp

    @auth_bp.route("/api/v1/auth/logout", methods=["POST"])
    def logout():
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            revoke_session(token)
        resp = make_response(jsonify({"ok": True}))
        resp.delete_cookie(SESSION_COOKIE_NAME)
        return resp

    @auth_bp.route("/api/v1/auth/me", methods=["GET"])
    def me():
        user = get_current_user()
        if user is None:
            return jsonify({"error": "Not authenticated"}), 401
        return jsonify({"user": user.to_dict()})

    app.register_blueprint(auth_bp)
