import time

class AlertManager:
    def __init__(self, cooldown=20):
        self.last = {}
        self.cooldown = cooldown

    def send_alert(self, alert_type, guard_id):
        key = f"{alert_type}:{guard_id}"
        now = time.time()

        if key in self.last and now - self.last[key] < self.cooldown:
            return  # suppress duplicate

        self.last[key] = now
        print(f"[ALERT] {alert_type} : {guard_id}")