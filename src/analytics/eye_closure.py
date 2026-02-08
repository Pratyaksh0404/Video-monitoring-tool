import time


class EyeClosureMonitor:
    def __init__(self, ear_threshold=0.50, closed_time_threshold=10):
        self.ear_threshold = ear_threshold
        self.closed_time_threshold = closed_time_threshold
        self.closed_since = {}

    def update(self, subject_id, ear_value):
        now = time.time()

        if ear_value is None:
            self._reset(subject_id)
            return "UNKNOWN"

        if ear_value < self.ear_threshold:
            if subject_id not in self.closed_since:
                self.closed_since[subject_id] = now
                return "CLOSED_SHORT"

            if now - self.closed_since[subject_id] >= self.closed_time_threshold:
                return "CLOSED_LONG"

            return "CLOSED_SHORT"

        self._reset(subject_id)
        return "OPEN"

    def _reset(self, subject_id):
        if subject_id in self.closed_since:
            del self.closed_since[subject_id]
