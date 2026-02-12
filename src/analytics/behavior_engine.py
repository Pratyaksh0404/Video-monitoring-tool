from collections import deque


class BehaviorEngine:
    def __init__(self, history_size=20, confirm_threshold=0.7):
        self.history = {}
        self.history_size = history_size
        self.confirm_threshold = confirm_threshold

    def update(self, track_id, label, confidence):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_size)

        self.history[track_id].append((label, confidence))

        # Count label frequency
        counts = {}
        for l, c in self.history[track_id]:
            if c >= self.confirm_threshold:
                counts[l] = counts.get(l, 0) + 1

        if not counts:
            return "ANALYZING"

        final_label = max(counts, key=counts.get)

        # Require majority in history
        if counts[final_label] > self.history_size * 0.6:
            return f"CONFIRMED_{final_label}"

        return f"POSSIBLE_{final_label}"
