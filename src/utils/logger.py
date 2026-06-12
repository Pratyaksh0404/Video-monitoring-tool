"""
src/utils/logger.py
───────────────────
Structured logging for the GMS system.

Usage:
    from utils.logger import get_logger
    log = get_logger("main_web")
    log.info("Camera ready")

Log levels:
    DEBUG   — detailed per-frame info (disabled in production)
    INFO    — normal operation milestones
    WARNING — threshold violations, recoverable issues
    ERROR   — exceptions, failed operations
    CRITICAL — pipeline crash

Output:
    - Console (INFO and above)
    - src/logs/system.log  ← always the CURRENT live session (single file)

Session archiving:
    On exit (Ctrl+C or normal shutdown), flask_app calls save_session_log()
    which copies system.log → logs/sessions/session_YYYY-MM-DD_HH-MM-SS.log
    then truncates system.log so the next run starts clean.

    No rotation, no ghost files, no backup suffixes.
    One live file.  One archive per session.
"""

import os
import shutil
import logging
import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR     = os.path.normpath(os.path.join(_HERE, "..", "logs"))
_SESSION_DIR = os.path.join(_LOG_DIR, "sessions")
_LOG_FILE    = os.path.join(_LOG_DIR, "system.log")

os.makedirs(_LOG_DIR,     exist_ok=True)
os.makedirs(_SESSION_DIR, exist_ok=True)

# ── Formatters ──────────────────────────────────────────────────────────────
_FMT_FILE    = "[%(asctime)s] %(levelname)-8s %(name)-16s %(message)s"
_FMT_CONSOLE = "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ── Root handler setup (done once) ──────────────────────────────────────────
_configured  = False
_file_handler = None   # kept so save_session_log() can flush it


def _configure_root():
    global _configured, _file_handler
    if _configured:
        return
    _configured = True

    root = logging.getLogger("gms")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # ── Single plain file handler (no rotation) ────────────────────────────
    try:
        _file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(_FMT_FILE, _DATE_FMT))
        root.addHandler(_file_handler)
    except Exception as e:
        print(f"[logger] Could not open log file {_LOG_FILE}: {e}")

    # ── Console handler ────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT_CONSOLE, _DATE_FMT))
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger under the 'gms' namespace.

        log = get_logger("main_web")
        log = get_logger("flask_app")
    """
    _configure_root()
    return logging.getLogger(f"gms.{name}")


# ── Session archiving ────────────────────────────────────────────────────────

def save_session_log(session_dt: datetime.datetime = None) -> str | None:
    """
    Copy system.log → logs/sessions/session_YYYY-MM-DD_HH-MM-SS.log
    then truncate system.log so the next session starts with a clean file.

    Called by flask_app._on_exit() / _sigint_handler() just before shutdown,
    and by send_session_report() on source-switch.

    Returns the path of the saved archive file, or None if system.log was
    empty / didn't exist (nothing worth saving).
    """
    global _file_handler

    if not os.path.exists(_LOG_FILE) or os.path.getsize(_LOG_FILE) == 0:
        return None

    # ── Flush the file handler so nothing is buffered ─────────────────────
    if _file_handler:
        try:
            _file_handler.flush()
        except Exception:
            pass

    # ── Build archive filename ─────────────────────────────────────────────
    ts  = (session_dt or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    dst = os.path.join(_SESSION_DIR, f"session_{ts}.log")

    # Avoid overwriting if two sessions somehow share the same second
    if os.path.exists(dst):
        ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dst = os.path.join(_SESSION_DIR, f"session_{ts}.log")

    # ── Copy then truncate (not move — handler keeps the fd open on Windows) ─
    try:
        shutil.copy2(_LOG_FILE, dst)
    except Exception as e:
        print(f"[logger] Could not archive session log: {e}")
        return None

    # Truncate system.log in-place (keeps the file descriptor valid)
    try:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            pass   # opening in "w" mode truncates
    except Exception as e:
        print(f"[logger] Could not truncate system.log: {e}")

    return dst


# ── Stubs kept for backward-compat with any old call sites ───────────────────
# flask_app.py references begin_session_log / end_session_log from an earlier
# iteration.  They are no-ops here — all the real work is in save_session_log.

def begin_session_log(session_dt: datetime.datetime = None):
    """No-op stub — session tracking is implicit in system.log."""
    pass


def end_session_log() -> str | None:
    """
    Alias for save_session_log() — called by flask_app on source-switch
    and by _on_exit_guarded() for the final session.
    """
    return save_session_log()


def set_debug_mode(enabled: bool):
    """Enable or disable DEBUG level console output."""
    root = logging.getLogger("gms")
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.DEBUG if enabled else logging.INFO)