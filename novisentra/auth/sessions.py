"""
novisentra/auth/sessions.py
──────────────────────────────
Server-side session tokens (opaque random strings, stored in SQLite with
an expiry), not JWTs. For a single-server, self-hosted deployment this is
simpler to reason about than JWT — sessions can be immediately revoked
(e.g. on logout, or an admin forcing a user out) by just deleting/marking
the row, with no signature/key-rotation machinery to manage.
"""

import secrets
import datetime
from .db import get_connection

SESSION_LIFETIME_DAYS = 7


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.datetime.utcnow()
    expires = now + datetime.timedelta(days=SESSION_LIFETIME_DAYS)

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at, revoked) "
            "VALUES (?, ?, ?, ?, 0)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def get_session_user_id(token: str):
    """Return the user_id for a valid, non-expired, non-revoked session
    token, or None if invalid/expired/revoked."""
    if not token:
        return None

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, expires_at, revoked FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["revoked"]:
            return None
        expires_at = datetime.datetime.fromisoformat(row["expires_at"])
        if datetime.datetime.utcnow() > expires_at:
            return None
        return row["user_id"]
    finally:
        conn.close()


def revoke_session(token: str) -> None:
    """Used on logout."""
    conn = get_connection()
    try:
        conn.execute("UPDATE sessions SET revoked = 1 WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def revoke_all_sessions_for_user(user_id: int) -> None:
    """Used when an admin needs to force a user's sessions to expire
    (e.g. after a password change, or removing a user's access)."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user_id,)
        )
        conn.commit()
    finally:
        conn.close()


def revoke_all_sessions() -> int:
    """
    Invalidate every existing session, regardless of user. Call this once
    at server startup — the person explicitly wants a fresh login required
    every time the system starts, not a 7-day "stay logged in" that
    survives a restart. Returns the number of sessions cleared.
    """
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM sessions")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def cleanup_expired_sessions() -> int:
    """Housekeeping — call periodically (e.g. a daily scheduled task) to
    keep the sessions table from growing unbounded. Returns rows deleted."""
    conn = get_connection()
    try:
        now = datetime.datetime.utcnow().isoformat()
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
