import time
import queue
import datetime

# Global queue — flask_app reads from this to stream alerts to browser
alert_queue = queue.Queue(maxsize=500)


class AlertManager:
    def __init__(self, cooldown=20):
        self.last     = {}
        self.cooldown = cooldown

    def send_alert(self, alert_type, guard_id, zone=None):
        key = f"{alert_type}:{guard_id}"
        now = time.time()

        if key in self.last and now - self.last[key] < self.cooldown:
            return  # suppress duplicate within cooldown window

        self.last[key] = now

        # ── Severity mapping ──────────────────────────────────────────────────
        t = alert_type.lower()

        if any(kw in t for kw in (
            "missing", "sleeping", "phone", "weapon", "threat",
            "fire", "smoke", "attack", "fight", "violence"
        )):
            severity = "high"

        elif any(kw in t for kw in (
            "distracted", "idle", "smoking", "loitering", "crowd", "unknown"
        )):
            severity = "medium"

        else:
            severity = "low"

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        alert = {
            "type":      alert_type,
            "guard_id":  guard_id,
            "zone":      zone or "—",
            "severity":  severity,
            "timestamp": timestamp,
            "epoch":     now,
        }

        # Push to browser queue (non-blocking — drop if full)
        try:
            alert_queue.put_nowait(alert)
        except queue.Full:
            pass

        # Keep terminal output
        print(f"[ALERT] [{timestamp}] {alert_type} : {guard_id}")