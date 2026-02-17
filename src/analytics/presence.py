import time


class PresenceMonitor:
    def __init__(self, absence_threshold=8, confirm_time=3):
        self.absence_threshold = absence_threshold
        self.confirm_time = confirm_time

        self.last_seen_time = time.time()
        self.first_seen_time = None

    def update(self, centroids):
        current_time = time.time()

        if not centroids:
            if current_time - self.last_seen_time > self.absence_threshold:
                return "ABSENT"
            return "TEMPORARILY_EMPTY"

        self.last_seen_time = current_time

        if self.first_seen_time is None:
            self.first_seen_time = current_time
            return "TEMPORARILY_EMPTY"

        if current_time - self.first_seen_time < self.confirm_time:
            return "TEMPORARILY_EMPTY"

        return "PRESENT"
