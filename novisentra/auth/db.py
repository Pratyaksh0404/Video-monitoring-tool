"""
novisentra/auth/db.py
───────────────────────
SQLite-backed user/session storage. SQLite (not a separate DB server) is
the right call here — NoviSentra is single-tenant per deployment (one
database per client install), so there's no need for a heavier database,
and it keeps "everything lives on the client's own machine" (see the
data-storage/privacy requirements) trivially true — it's just a file.
"""

import sqlite3
import os
import threading

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "novisentra.db"
)

_lock = threading.Lock()
_db_path = _DEFAULT_DB_PATH


def set_db_path(path: str):
    """Override the DB file location (e.g. from an env var at startup)."""
    global _db_path
    _db_path = path


def get_connection():
    os.makedirs(os.path.dirname(os.path.abspath(_db_path)), exist_ok=True)
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _lock:
        conn = get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consent_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    accepted_by TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    policy_version TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()
