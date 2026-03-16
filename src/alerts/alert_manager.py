import time
import queue
import datetime

# Global queue — flask_app reads from this to stream alerts to browser
alert_queue = queue.Queue(maxsize=500)


class AlertManager:
    def __init__(self, cooldown=20):
        self.last = {}
        self.cooldown = cooldown

    def send_alert(self, alert_type, guard_id, zone=None):
        key = f"{alert_type}:{guard_id}"
        now = time.time()

        if key in self.last and now - self.last[key] < self.cooldown:
            return  # suppress duplicate

        self.last[key] = now

        # Severity mapping
        severity = "high"
        if alert_type in ("Guard Missing", "Guard Sleeping", "Phone Usage"):
            severity = "high"
        elif alert_type in ("Guard Distracted", "Guard Idle"):
            severity = "medium"
        else:
            severity = "low"

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        alert = {
            "type": alert_type,
            "guard_id": guard_id,
            "zone": zone or "—",
            "severity": severity,
            "timestamp": timestamp,
            "epoch": now,
        }

        # Push to browser queue (non-blocking — drop if full)
        try:
            alert_queue.put_nowait(alert)
        except queue.Full:
            pass

        # Keep terminal output working exactly as before
        print(f"[ALERT] [{timestamp}] {alert_type} : {guard_id}")