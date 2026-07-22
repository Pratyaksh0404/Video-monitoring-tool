"""
novisentra/auth/models.py
────────────────────────────
User account management. Roles are deliberately simple — "admin" (can
rename cameras/zones, manage other users, change config) and "viewer"
(can see the dashboard, cannot change settings). No complex RBAC —
this is a single-deployment-per-client system, not a multi-org SaaS
control plane, so a 2-role model is enough.
"""

import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from .db import get_connection


class User:
    def __init__(self, row):
        self.id            = row["id"]
        self.username      = row["username"]
        self.password_hash = row["password_hash"]
        self.role          = row["role"]
        self.created_at    = row["created_at"]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self):
        return {"id": self.id, "username": self.username, "role": self.role,
                "created_at": self.created_at}


def create_user(username: str, password: str, role: str = "viewer") -> User:
    """
    Create a new user. Raises ValueError if username already exists or
    role is invalid — caller (CLI script or admin API route) should
    catch and report this cleanly.
    """
    if role not in ("admin", "viewer"):
        raise ValueError(f"Invalid role '{role}' — must be 'admin' or 'viewer'")

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        password_hash = generate_password_hash(password)
        now = datetime.datetime.utcnow().isoformat()
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, password_hash, role, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return User(row)
    finally:
        conn.close()


def get_user_by_username(username: str):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return User(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return User(row) if row else None
    finally:
        conn.close()


def verify_password(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)


def list_users() -> list:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [User(r).to_dict() for r in rows]
    finally:
        conn.close()


def user_count() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    finally:
        conn.close()


def change_password(user_id: int, new_password: str) -> bool:
    conn = get_connection()
    try:
        password_hash = generate_password_hash(new_password)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def delete_user(user_id: int) -> bool:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()
