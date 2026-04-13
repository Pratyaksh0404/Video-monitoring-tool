import time
import math
from collections import deque


class InactivityMonitor:
    """
    Tracks whether a person is genuinely inactive (not moving).

    Fixes for 2.5fps operation
    ──────────────────────────
    - position_threshold raised 40px → 80px
      At 2.5fps, a walking person moves ~20-40px per frame. The old 40px
      threshold was so low that normal walking was classified as ACTIVE only
      sometimes. 80px requires clear, deliberate movement to reset inactivity.

    - inactivity_threshold raised 30s → 45s
      Give more time before flagging as INACTIVE — combined with BehaviorEngine
      CONFIRMED_IDLE at 25s, this prevents double-firing.

    - window_time raised 5s → 10s
      Wider window averages out the low-fps jitter in position readings.
    """

    def __init__(
        self,
        inactivity_threshold=45,   # raised from 30
        position_threshold=80,     # raised from 40
        window_time=10             # raised from 5
    ):
        self.inactivity_threshold = inactivity_threshold
        self.position_threshold   = position_threshold
        self.window_time          = window_time

        self.last_significant_move_time = time.time()
        self.positions = deque()

    def update(self, centroids):
        current_time = time.time()

        if not centroids:
            self.positions.clear()
            self.last_significant_move_time = current_time
            return "NO_PERSON"

        cx, cy = centroids[0]
        self.positions.append((current_time, cx, cy))

        while self.positions and current_time - self.positions[0][0] > self.window_time:
            self.positions.popleft()

        if len(self.positions) >= 2:
            avg_x = sum(p[1] for p in self.positions) / len(self.positions)
            avg_y = sum(p[2] for p in self.positions) / len(self.positions)

            dist = math.dist((cx, cy), (avg_x, avg_y))

            if dist > self.position_threshold:
                self.last_significant_move_time = current_time
                self.positions.clear()
                return "ACTIVE"

        if current_time - self.last_significant_move_time >= self.inactivity_threshold:
            return "INACTIVE"

        return "ACTIVE"