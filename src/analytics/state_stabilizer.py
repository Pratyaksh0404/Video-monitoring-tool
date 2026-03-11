import time
from collections import defaultdict

class StateStabilizer:

    def __init__(self, confirm_time=5):

        self.states = defaultdict(lambda: {
            "label": None,
            "start": 0
        })

        self.confirm_time = confirm_time

    def update(self, track_id, label):

        now = time.time()
        state = self.states[track_id]

        if state["label"] != label:

            state["label"] = label
            state["start"] = now
            return "ANALYZING"

        if now - state["start"] >= self.confirm_time:
            return f"CONFIRMED_{label}"

        return f"POSSIBLE_{label}"