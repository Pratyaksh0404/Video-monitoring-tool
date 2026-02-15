import time

class BehaviorEngine:
    def __init__(self):
        self.history = {}
        self.confirm_time = 20
        self.possible_time = 10

    def update(self, track_id, label):
        current_time = time.time()

        if track_id not in self.history:
            self.history[track_id] = {
                "label": label,
                "start": current_time
            }
            return "ANALYZING"

        if self.history[track_id]["label"] != label:
            self.history[track_id] = {
                "label": label,
                "start": current_time
            }
            return "ANALYZING"

        duration = current_time - self.history[track_id]["start"]

        if label == "NORMAL":
            return "NORMAL"

        if duration >= self.confirm_time:
            return f"CONFIRMED_{label}"

        if duration >= self.possible_time:
            return f"POSSIBLE_{label}"

        return "ANALYZING"
