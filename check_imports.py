"""
check_imports.py
───────────────────
Run this ONE script to check the entire novisentra import chain at once.
Tells you exactly which file is broken, if any, instead of restarting the
whole Flask server and reading through walls of log output per error.

Usage:
    python check_imports.py
"""

import sys
sys.path.insert(0, "src")   # alerts.alert_manager and others live under src/

CHECKS = [
    ("novisentra.capabilities.registry", ["validate_capability_keys", "get_capability", "list_capabilities"]),
    ("novisentra.profiles.base", ["BaseProfile", "Zone"]),
    ("novisentra.profiles.guard_monitoring", ["GUARD_MONITORING"]),
    ("novisentra.profiles.bank_security", ["BANK_SECURITY"]),
    ("novisentra.profiles.retail_analytics", ["RETAIL_ANALYTICS"]),
    ("novisentra.profiles.warehouse_ops", ["WAREHOUSE_OPS"]),
    ("novisentra.profiles.loader", ["get_profile", "list_profiles", "is_valid_profile"]),
    ("novisentra.profiles", ["get_profile", "list_profiles", "is_valid_profile", "BaseProfile", "Zone"]),
    ("novisentra.auth.db", ["init_db", "get_connection"]),
    ("novisentra.auth.models", ["User", "create_user", "verify_password"]),
    ("novisentra.auth.sessions", ["create_session", "get_session_user_id", "revoke_session"]),
    ("novisentra.auth", ["get_session_user_id", "get_user_by_id", "create_user", "init_db"]),
    ("novisentra.alerting.base", ["AlertBackend"]),
    ("novisentra.alerting.webhook_backend", ["WebhookBackend"]),
    ("novisentra.alerting", ["AlertRouter", "WebhookBackend"]),
    ("novisentra.pipeline.batch_jobs", ["batch_queue", "init_batch_jobs_table"]),
    # alerts.alert_manager — the file that just broke camera startup. Every
    # name main_web.py / camera_manager.py / snapshot_manager.py actually
    # imports from it is checked here, so a missing function shows up
    # BEFORE you try to start the camera pipeline, not after a black screen.
    ("alerts.alert_manager", [
        "AlertManager", "alert_queue",
        "set_camera_context", "get_camera_context",
        "set_profile_context", "get_profile_context",
        "register_camera_profile", "get_camera_profile",
    ]),
    ("api.middleware.session_auth", ["get_current_user", "require_login", "require_admin"]),
    ("api.routes.auth", ["register"]),
    ("api.routes.admin", ["register"]),
    ("api.routes.config", ["register"]),
    ("api.routes.recordings", ["register"]),
    # camera_manager.py itself — at project root, not under src/ or
    # novisentra/, so this also confirms the root-level sys.path is set
    # up correctly (it is, by default, when running `python flask_app.py`
    # from the project root — this just double-checks it explicitly).
    ("camera_manager", ["CameraManager", "CameraInstance", "CameraStreamer"]),
]

print("=" * 70)
print("  NoviSentra Import Chain Check")
print("=" * 70)
print()

failed = False

for module_name, expected_names in CHECKS:
    try:
        module = __import__(module_name, fromlist=expected_names)
        missing = [n for n in expected_names if not hasattr(module, n)]
        if missing:
            print(f"✗ {module_name}")
            print(f"    Imported OK, but missing: {missing}")
            print(f"    → This file's content is incomplete/wrong version.")
            failed = True
        else:
            print(f"✓ {module_name}")
    except Exception as e:
        print(f"✗ {module_name}")
        print(f"    {type(e).__name__}: {e}")
        print(f"    → Fix this one first — everything after it in the chain")
        print(f"      will also fail until this is resolved.")
        failed = True
        # Don't break — show ALL failures in one pass so you can fix
        # everything at once instead of one-at-a-time.

print()
print("=" * 70)
if failed:
    print("  RESULT: One or more modules failed — see ✗ marks above.")
    print("  Fix the FIRST failure in the list first (later ones may be")
    print("  downstream symptoms of it), then rerun this script.")
else:
    print("  RESULT: ALL modules import cleanly. ✓")
    print("  If flask_app.py still shows errors, the problem is elsewhere")
    print("  (e.g. a stale __pycache__ — delete all __pycache__ folders")
    print("  and try again).")
print("=" * 70)

sys.exit(1 if failed else 0)
