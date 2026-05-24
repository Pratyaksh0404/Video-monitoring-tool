"""
src/utils/logger.py
───────────────────
Structured logging for the GMS system.

Usage:
    from utils.logger import get_logger
    log = get_logger("main_web")
    log.info("Camera ready")
    log.warning("Fire confidence below threshold: 0.65")
    log.error("Pipeline crashed", exc_info=True)

Log levels:
    DEBUG   — detailed per-frame info (disabled in production)
    INFO    — normal operation milestones
    WARNING — threshold violations, recoverable issues
    ERROR   — exceptions, failed operations
    CRITICAL — pipeline crash

Output:
    - Console (INFO and above)
    - src/logs/system.log (DEBUG and above, rotating 5MB × 3 files)

Format:
    [2026-05-01 11:45:23] INFO     main_web     Camera ready
    [2026-05-01 11:45:24] WARNING  fire_det     Confidence 0.65 below threshold
"""

import os
import logging
import logging.handlers

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "..", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "system.log")

os.makedirs(_LOG_DIR, exist_ok=True)

# ── Formatters ──────────────────────────────────────────────────────────────
_FMT_FILE    = "[%(asctime)s] %(levelname)-8s %(name)-16s %(message)s"
_FMT_CONSOLE = "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ── Root handler setup (done once) ──────────────────────────────────────────
_configured = False

def _configure_root():
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("gms")
    root.setLevel(logging.DEBUG)
    root.propagate = False  # prevent double output via root logger

    # ── Rotating file handler ──────────────────────────────────────────────
    try:
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,               # keep last 3 files
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT_FILE, _DATE_FMT))
        root.addHandler(fh)
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

    Example:
        log = get_logger("main_web")
        log = get_logger("anomaly_detector")
        log = get_logger("flask_app")
    """
    _configure_root()
    return logging.getLogger(f"gms.{name}")


def set_debug_mode(enabled: bool):
    """Enable or disable DEBUG level console output."""
    root = logging.getLogger("gms")
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.handlers.RotatingFileHandler
        ):
            handler.setLevel(logging.DEBUG if enabled else logging.INFO)