import time
from collections import deque


class BehaviorEngine:
    WINDOW_SIZE    = 8
    CONFIRM_RATIO  = 0.60   # 60% of readings must agree → CONFIRMED
    POSSIBLE_RATIO = 0.40   # 40% of readings must agree → POSSIBLE

    # Per-label confirm ratio overrides
    _CONFIRM_RATIO_OVERRIDE = {
        "SMOKING":          0.75,   # raised from 0.70 — very strict
        "SLEEPING":         0.60,
        "PHONE_USE":        0.60,
        "IDLE":             0.55,
        "DISTRACTED_OTHER": 0.55,
    }

    # Minimum readings in window before we confirm anything
    MIN_READINGS_FOR_CONFIRM  = 6   # needs ~36s of consistent behavior before CONFIRMED
    MIN_READINGS_FOR_POSSIBLE = 3   # needs ~18s before POSSIBLE

    def __init__(self):
        # track_id → deque of (timestamp, label) readings
        self._windows: dict = {}

    def update(self, track_id, label) -> str:
        """
        Feed a new CLIP label reading and return the current state.
        Returns: ANALYZING | NORMAL | POSSIBLE_X | CONFIRMED_X
        """
        now = time.time()

        if track_id not in self._windows:
            self._windows[track_id] = deque(maxlen=self.WINDOW_SIZE)

        window = self._windows[track_id]

        # Add new reading (skip ANALYZING — not a real label)
        if label != "ANALYZING":
            window.append(label)

        if not window:
            return "ANALYZING"

        # Count label frequencies in window
        counts = {}
        for lbl in window:
            counts[lbl] = counts.get(lbl, 0) + 1

        total = len(window)

        # Find dominant label (not NORMAL)
        best_label = None
        best_count = 0
        for lbl, cnt in counts.items():
            if lbl != "NORMAL" and cnt > best_count:
                best_count = cnt
                best_label = lbl

        # If NORMAL dominates → person is normal
        normal_count = counts.get("NORMAL", 0)
        if normal_count > best_count:
            return "NORMAL"

        if best_label is None:
            return "NORMAL"

        ratio = best_count / total
        confirm_ratio  = self._CONFIRM_RATIO_OVERRIDE.get(best_label, self.CONFIRM_RATIO)
        possible_ratio = self.POSSIBLE_RATIO

        if total >= self.MIN_READINGS_FOR_CONFIRM and ratio >= confirm_ratio:
            return f"CONFIRMED_{best_label}"
        elif total >= self.MIN_READINGS_FOR_POSSIBLE and ratio >= possible_ratio:
            return f"POSSIBLE_{best_label}"
        else:
            return "ANALYZING"

    def reset(self, track_id=None):
        if track_id is None:
            self._windows.clear()
        else:
            self._windows.pop(track_id, None)