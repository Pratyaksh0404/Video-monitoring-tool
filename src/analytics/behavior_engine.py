import time

class BehaviorEngine:
    def __init__(self):
        self.history = {}
        self.possible_time = 8
        self.confirm_time = 15

    def update(self, track_id, label):

        now = time.time()

        if track_id not in self.history:
            self.history[track_id] = {
                "label": label,
                "start": now
            }
            return "ANALYZING"

        if self.history[track_id]["label"] != label:
            self.history[track_id] = {
                "label": label,
                "start": now
            }
            return "ANALYZING"

        duration = now - self.history[track_id]["start"]

        if label == "NORMAL":
            return "NORMAL"

        if duration >= self.confirm_time:
            return f"CONFIRMED_{label}"

        if duration >= self.possible_time:
            return f"POSSIBLE_{label}"

        return "ANALYZING"
