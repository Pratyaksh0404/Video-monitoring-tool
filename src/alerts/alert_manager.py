import queue
import datetime
import sys
import os

# Add src to path so utils.logger is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from utils.logger import get_logger
    _log = get_logger("alert_manager")
except Exception:
    _log = None

alert_queue = queue.Queue()

# ── Severity mapping ──────────────────────────────────────────────────────────
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


class AlertManager:

    def send_alert(self, alert_type: str, guard_id: str, zone: str = "—"):
        now       = datetime.datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        severity  = _get_severity(alert_type)

        alert = {
            "type":      alert_type,
            "guard_id":  guard_id,
            "zone":      zone,
            "severity":  severity,
            "timestamp": timestamp,
        }

        # Send to SSE queue for dashboard
        alert_queue.put(alert)

        # Log to system.log with appropriate level
        msg = f"[ALERT] {alert_type} | {guard_id} | Zone {zone} | {severity.upper()}"
        print(f"[ALERT] [{timestamp}] {alert_type} : {guard_id}")

        if _log:
            if severity == "high":
                _log.warning(msg)
            elif severity == "medium":
                _log.info(msg)
            else:
                _log.debug(msg)