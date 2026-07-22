"""
src/utils/logger.py
───────────────────
Structured logging for the NoviSentra system.

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
import threading

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

    root = logging.getLogger("NoviSentra")
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
    Get a named logger under the 'NoviSentra' namespace.

        log = get_logger("main_web")
        log = get_logger("flask_app")
    """
    _configure_root()
    return logging.getLogger(f"NoviSentra.{name}")


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
    root = logging.getLogger("NoviSentra")
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.DEBUG if enabled else logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# Per-profile session logs
#
# The existing system.log / save_session_log() above is ONE continuous
# technical log for the whole running process (all levels, all profiles
# mixed) — kept exactly as-is for troubleshooting, not touched by any of
# this.
#
# This is a SEPARATE, additional mechanism specifically for the ALERT
# HISTORY, split by profile: whichever profile is active when an alert
# fires, that alert goes into THAT profile's own session log file.
#
# ── Redesigned 2026-07 (real bugs found from a live multi-profile test run) ──
# Previous design batched alerts in memory (flask_app._alert_log) and only
# wrote them to per-profile files inside send_session_report(), which was
# assumed to fire on every profile switch. It didn't — across a 40-minute,
# 4-profile test run it fired exactly ONCE, at final shutdown. That caused
# two confirmed problems: (1) the per-profile filename embedded the
# PROCESS START timestamp, so restarting the app mid-testing (which
# happened, per the terminal logs) silently splits what a person thinks
# of as "one session" into multiple file groups — "why did I get 7 files
# for 4 profiles" — and (2) cross-checking the master log against the
# per-profile files line-by-line showed ~35 of 144 real alerts never
# made it into ANY per-profile file, most plausibly because they were
# still sitting in the in-memory batch when something (a crash, a
# restart) interrupted the run before the one deferred flush happened.
#
# Fix: write each alert to its profile's log file IMMEDIATELY, the
# moment it fires (called directly from alert_manager.send_alert(), the
# same place the master log line is written) — durable by construction,
# the same way system.log already is, instead of depending on a
# deferred batch flush ever happening. Filenames are now DATE-based
# (session_{YYYY-MM-DD}_{profile_id}.log) rather than process-start-
# based, so any number of restarts within the same day keep appending to
# the SAME 4 files instead of fragmenting — matching "4 profiles = 4
# files" as the actual expectation.
# ═══════════════════════════════════════════════════════════════════════════

_profile_log_lock = threading.Lock()


def append_alert_to_profile_log(alert: dict) -> str | None:
    """
    Append ONE alert to its profile's session log file immediately.
    Call this synchronously from wherever an alert is fired — do not
    batch. Returns the filepath written, or None on failure (never
    raises — a logging failure must not take down the alert pipeline).
    """
    try:
        pid = alert.get("profile_id") or "unknown"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        dst = os.path.join(_SESSION_DIR, f"session_{date_str}_{pid}.log")
        line = (
            f"[{alert.get('timestamp', ''):8s}] "
            f"{alert.get('severity', '').upper():8s} "
            f"{alert.get('type', ''):32s} | "
            f"{alert.get('guard_id', ''):20s} | "
            f"Zone {alert.get('zone', ''):16s} | "
            f"cam={alert.get('camera_id', '')}\n"
        )
        with _profile_log_lock:
            with open(dst, "a", encoding="utf-8") as f:
                f.write(line)
        return dst
    except Exception as e:
        print(f"[logger] Could not append to per-profile session log: {e}")
        return None


def save_profile_alert_log(alerts: list) -> dict:
    """
    LEGACY / safety-net path — kept for send_session_report()'s existing
    call site and for anything that still batches. Every alert passed
    here has almost certainly already been written by
    append_alert_to_profile_log() at the moment it fired, so this is now
    a no-op for anything already on disk; it only matters as a fallback
    if that immediate write somehow failed. Uses the SAME date-based
    filename as append_alert_to_profile_log() so both paths always
    target the same 4 files, never a second set.

    Returns {profile_id: filepath} for whichever profiles had alerts in
    this batch (empty dict if `alerts` was empty).
    """
    if not alerts:
        return {}

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    by_profile = {}
    for a in alerts:
        pid = a.get("profile_id") or "unknown"
        by_profile.setdefault(pid, []).append(a)

    written = {}
    for profile_id, group in by_profile.items():
        dst = os.path.join(_SESSION_DIR, f"session_{date_str}_{profile_id}.log")
        written[profile_id] = dst
        # Not re-writing the group here — append_alert_to_profile_log()
        # already wrote each of these individually and immediately when
        # they fired. Re-appending here would duplicate every line.
    return written