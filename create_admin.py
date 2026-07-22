"""
create_admin.py
──────────────────
One-time setup script — creates the first admin account for a new
NoviSentra deployment. Run this once after installation, before starting
flask_app.py for the first time.

Deliberately a standalone script rather than an open API endpoint —
having "create the first admin" reachable over HTTP, even briefly, is an
avoidable attack surface. This matches how other self-hosted tools handle
bootstrap (e.g. Django's `createsuperuser`).

Usage:
    python create_admin.py
    (interactive prompts for username/password)

    python create_admin.py --username admin --password "..."
    (non-interactive, e.g. for scripted deployment)
"""

import sys
import os
import argparse
import getpass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from novisentra.auth import init_db, create_user, user_count


def main():
    parser = argparse.ArgumentParser(description="Create the first NoviSentra admin account.")
    parser.add_argument("--username", help="Admin username (omit for interactive prompt)")
    parser.add_argument("--password", help="Admin password (omit for interactive prompt)")
    args = parser.parse_args()

    init_db()

    existing = user_count()
    if existing > 0:
        print(f"[create_admin] {existing} user(s) already exist in this deployment.")
        confirm = input("Create another admin account anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            print("[create_admin] Cancelled.")
            return

    username = args.username or input("Admin username: ").strip()
    if not username:
        print("[create_admin] Username cannot be empty.")
        sys.exit(1)

    if args.password:
        password = args.password
    else:
        password = getpass.getpass("Admin password (min 8 characters): ")
        confirm_pw = getpass.getpass("Confirm password: ")
        if password != confirm_pw:
            print("[create_admin] Passwords do not match.")
            sys.exit(1)

    if len(password) < 8:
        print("[create_admin] Password must be at least 8 characters.")
        sys.exit(1)

    try:
        user = create_user(username, password, role="admin")
        print(f"[create_admin] ✓ Admin account created: {user.username} (id={user.id})")
        print(f"[create_admin] You can now log in at the NoviSentra dashboard.")
    except ValueError as e:
        print(f"[create_admin] ✗ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
