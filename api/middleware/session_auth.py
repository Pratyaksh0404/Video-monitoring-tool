"""
api/middleware/session_auth.py
─────────────────────────────────
Cookie-based session auth for the dashboard/admin panel — separate from
api/middleware/auth.py's X-API-Key check, which is for the REST API used
by external integrations. Dashboard users log in with username/password
and get an HttpOnly session cookie; API integrations use an API key.
Different audiences, different mechanisms, both can be active at once.
"""

import functools
from flask import request, jsonify, g
from novisentra.auth import get_session_user_id, get_user_by_id

SESSION_COOKIE_NAME = "novisentra_session"


def get_current_user():
    """Returns the logged-in User object for this request, or None."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    user_id = get_session_user_id(token)
    if user_id is None:
        return None
    return get_user_by_id(user_id)


def require_login(f):
    """Any logged-in user (admin or viewer) can access this route."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "Not authenticated", "login_required": True}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    """Only admin-role users can access this route (camera/zone rename,
    user management, etc.)."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return jsonify({"error": "Not authenticated", "login_required": True}), 401
        if not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated
