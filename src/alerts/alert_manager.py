import queue
import datetime
import sys
import os
import re
import threading
import contextvars

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from utils.logger import get_logger
    _log = get_logger("alert_manager")
except Exception:
    _log = None

alert_queue = queue.Queue()

# ── Multi-camera context ─────────────────────────────────────────────────────
# Each camera's pipeline runs in its own dedicated thread (see
# CameraInstance._run_pipeline in camera_manager.py). ContextVars are
# isolated per-thread by default, so setting these once at the start of
# each camera's thread means every alert generated inside main_web.run()
# for that camera is automatically tagged correctly — with ZERO changes
# needed inside main_web.py itself.
_camera_id_var         = contextvars.ContextVar("camera_id", default="cam_1")
_camera_name_var       = contextvars.ContextVar("camera_name", default="Camera 1")
_camera_has_zones_var  = contextvars.ContextVar("camera_has_zones", default=False)
_profile_id_var        = contextvars.ContextVar("profile_id", default="guard_monitoring")


def set_camera_context(camera_id: str, camera_name: str = None,
                        has_named_zones: bool = False) -> None:
    """
    Call once at the start of a camera's pipeline thread (and again at the
    top of each loop iteration, in case of a live profile/name change).

    camera_name / has_named_zones power the zone-as-camera fallback below:
    for profiles where one camera = one physical section (retail, warehouse,
    bank, attendance) and no sub-zone polygons are configured for that
    camera, the camera's own name IS the section/zone label — no polygon
    calibration needed. Polygons remain available for the rare case one
    camera genuinely spans multiple named areas.
    """
    _camera_id_var.set(camera_id)
    if camera_name is not None:
        _camera_name_var.set(camera_name)
    _camera_has_zones_var.set(has_named_zones)


def get_camera_context() -> str:
    return _camera_id_var.get()


def set_profile_context(profile_id: str) -> None:
    """
    Call once at the start of a camera's pipeline thread alongside
    set_camera_context(). Stamped onto every alert the same way camera_id
    is. NOTE: this only tags alerts with which profile was active — it
    does NOT yet gate which detectors run. That gating happens inside
    main_web.py's per-frame loop and requires main_web.py to check
    get_profile_context() before invoking each detector.
    """
    _profile_id_var.set(profile_id)


def get_profile_context() -> str:
    return _profile_id_var.get()


# ═══════════════════════════════════════════════════════════════════════════
# Live camera→profile registry — separate from the contextvar above.
#
# contextvars are per-thread and DO NOT update across threads (verified
# empirically — even calling set_profile_context() from the admin API's
# request-handling thread never reaches an already-running camera
# pipeline thread's view of that variable, since each thread has its own
# isolated context). A camera's pipeline thread runs main_web.run() in a
# single blocking call for the entire session, so the contextvar set
# once at session start is all that thread will EVER see, no matter what
# happens afterward in any other thread.
#
# This registry is a genuinely shared, lock-protected dict — the actual
# mechanism that makes "switch profile in the admin panel" take effect
# on an already-running camera, live, without needing a restart.
# camera_manager.py writes to it (at camera start AND on every live
# profile switch); main_web.py reads from it on every capability check.
# ═══════════════════════════════════════════════════════════════════════════
_camera_profile_registry: dict = {}
_camera_profile_lock = threading.Lock()


def register_camera_profile(camera_id: str, profile_id: str) -> None:
    """Call whenever a camera's assigned profile is set OR changed —
    at camera startup, and on every live admin-panel switch."""
    with _camera_profile_lock:
        _camera_profile_registry[camera_id] = profile_id


def get_camera_profile(camera_id: str, fallback: str = "guard_monitoring") -> str:
    """The actual live profile for a given camera right now — this is
    what main_web.py should check, NOT get_profile_context(), for
    anything that needs to reflect a mid-session profile switch."""
    with _camera_profile_lock:
        return _camera_profile_registry.get(camera_id, fallback)


_SEVERITY = {
    # HIGH
    "Guard Missing":                    "high",
    "Guard Sleeping":                   "high",
    "Phone Usage":                      "high",
    "Weapon Detected":                  "high",
    "Unattended Weapon Detected":       "high",
    "Fire Detected":                    "high",
    "Fight / Violence Detected":        "high",
    "Guard Under Attack":               "high",
    "Unknown Person Sleeping":          "high",
    "Unknown Person Using Phone":       "high",
    "Camera Tamper":                    "high",
    "Face Covering Detected":           "high",
    "Bill Mismatch Detected":           "high",
    # MEDIUM
    "Guard Idle":                       "medium",
    "Guard Smoking":                    "medium",
    "Guard Distracted":                 "medium",
    "Unknown Person Detected":          "medium",
    "Crowd Detected":                   "medium",
    "Suspicious Loitering Detected":    "medium",
    # LOW
    "Patrol":                           "low",
}


def _get_severity(alert_type: str) -> str:
    for key, sev in _SEVERITY.items():
        if key.lower() in alert_type.lower():
            return sev
    return "low"


# Placeholder values that mean "no zone was actually determined" — when
# main_web.py passes one of these (its existing behavior, unchanged) AND
# this camera has no calibrated named zones, we substitute the camera's
# own name so the alert still carries a meaningful section label.
_NO_ZONE_PLACEHOLDERS = {"—", "-", "", None}

# ── Terminology translation ───────────────────────────────────────────────
# "Guard Sleeping"/"Patrol" are guard_monitoring vocabulary and read wrong
# for other sectors — a warehouse doesn't have "guards", a bank has
# "staff". Applied HERE, centrally, so every consumer of the alert dict
# (email subject/body, WhatsApp template variables, the dashboard SSE
# stream, the analytics history) gets the correct wording automatically,
# with one change in one place — rather than needing every downstream
# file (email_alerter.py, whatsapp_alerter.py, dashboard.html) to each
# reimplement the same substitution.
_ROLE_NOUN = {
    "guard_monitoring": "Guard",
    "bank_security":    "Staff",
    "warehouse_ops":    "Worker",
    "retail_analytics": "Staff",
}
_MOVEMENT_LABEL = {
    "guard_monitoring": "Patrol",
    "bank_security":    "Movement",
    "warehouse_ops":    "Movement",
    "retail_analytics": "Movement",
}


def _translate_alert_type(raw_type: str, profile_id: str) -> str:
    role_noun  = _ROLE_NOUN.get(profile_id, "Guard")
    move_label = _MOVEMENT_LABEL.get(profile_id, "Patrol")
    t = raw_type
    if role_noun != "Guard":
        t = re.sub(r"\bGuard\b", role_noun, t)
    if move_label != "Patrol":
        t = re.sub(r"^Patrol:", f"{move_label}:", t)
    return t


class AlertManager:

    def send_alert(self, alert_type: str, guard_id: str, zone: str = "—",
                    camera_id: str = None, profile_id: str = None):
        """
        camera_id / profile_id: optional explicit overrides. If not passed
        (the normal case — main_web.py's existing call sites don't need to
        change), they're read from the per-thread context set by
        CameraInstance._run_pipeline() via set_camera_context().

        Zone-as-camera fallback: if `zone` is a placeholder (main_web.py's
        existing default when it has no sub-zone concept to report) and
        this camera has no calibrated named zones configured, the zone
        field is set to the camera's own name instead — e.g. "Men's
        Section" rather than "—". This makes multi-camera-per-section
        deployments (retail/warehouse/bank/attendance) report meaningful
        section labels automatically, with zero main_web.py changes,
        since each such camera typically covers exactly one section.
        """
        now       = datetime.datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        # Severity lookup MUST use the raw canonical type ("Guard Sleeping")
        # — translating first would break the lookup for non-guard profiles
        # since _SEVERITY's keys are the canonical English strings, not
        # every possible translated variant.
        severity  = _get_severity(alert_type)

        resolved_zone = zone
        if zone in _NO_ZONE_PLACEHOLDERS and not _camera_has_zones_var.get():
            resolved_zone = _camera_name_var.get()

        resolved_camera_id = camera_id if camera_id is not None else _camera_id_var.get()
        # Resolve profile from the LIVE registry (keyed by camera), not the
        # contextvar — the contextvar is only set once per camera session
        # and goes stale exactly like main_web.py's capability checks did
        # before this fix; every alert must reflect whatever profile this
        # camera is ACTUALLY running right now, including immediately
        # after a live admin-panel switch.
        resolved_profile_id = (profile_id if profile_id is not None
                               else get_camera_profile(resolved_camera_id))
        display_type = _translate_alert_type(alert_type, resolved_profile_id)

        alert = {
            "type":       display_type,   # profile-appropriate wording — this is
                                           # what email/WhatsApp/dashboard/analytics
                                           # all read, so translation flows everywhere
                                           # automatically from this one place
            "guard_id":   guard_id,
            "zone":       resolved_zone,
            "severity":   severity,
            "timestamp":  timestamp,
            "camera_id":  resolved_camera_id,
            "profile_id": resolved_profile_id,
        }

        alert_queue.put(alert)

        # Durable per-profile session log — written IMMEDIATELY, right
        # here, not batched (see src/utils/logger.py module docstring for
        # the full story: batching + a deferred flush that turned out to
        # only fire once per process lost ~25% of a real test run's
        # alerts and fragmented files across restarts). This makes the
        # per-profile log exactly as durable as system.log below, since
        # it's now written at the same moment via the same code path.
        try:
            from utils.logger import append_alert_to_profile_log
            append_alert_to_profile_log(alert)
        except Exception as e:
            print(f"[AlertManager] Could not write per-profile session log: {e}")

        msg = (f"[ALERT] {display_type} | {guard_id} | Zone {resolved_zone} | "
               f"{severity.upper()} | cam={alert['camera_id']}")
        print(f"[ALERT] [{timestamp}] {display_type} : {guard_id} "
              f"(cam={alert['camera_id']}, zone={resolved_zone})")

        if _log:
            if severity == "high":
                _log.warning(msg)
            elif severity == "medium":
                _log.info(msg)
            else:
                _log.debug(msg)
