"""
api/middleware/auth.py
───────────────────────
Optional API key middleware for the v1 REST API.

When API_KEY is set in environment (or config), all /api/v1/* requests
must include the header:
    X-API-Key: <key>

If API_KEY is not set, auth is skipped entirely (dev/local mode).
The dashboard routes (/  /video_feed  /alerts/stream  /api/stats etc.)
are never gated — they stay publicly accessible on the local network.
"""

import os
import functools
from flask import request, jsonify


def _get_configured_key() -> str:
    """Read API key from environment variable. Returns empty string if not set."""
    return os.environ.get("NOVISENTRA_API_KEY", "").strip()


def require_api_key(f):
    """
    Decorator for Flask route functions.

    Usage:
        @app.route("/api/v1/something")
        @require_api_key
        def something():
            ...

    If NOVISENTRA_API_KEY env var is not set, this decorator is a no-op.
    If it is set, requests without a matching X-API-Key header get 401.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        key = _get_configured_key()
        if not key:
            # Auth disabled — dev/local mode
            return f(*args, **kwargs)

        provided = (
            request.headers.get("X-API-Key")
            or request.args.get("api_key")   # also accept ?api_key= in URL
        )
        if not provided or provided != key:
            return jsonify({
                "error": "Unauthorized",
                "message": "Missing or invalid X-API-Key header."
            }), 401

        return f(*args, **kwargs)
    return decorated