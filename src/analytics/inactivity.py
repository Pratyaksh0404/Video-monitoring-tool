import time
import math
from collections import deque


class InactivityMonitor:
    def __init__(
        self,
        inactivity_threshold=30,
        position_threshold=40,
        window_time=5
    ):
        self.inactivity_threshold = inactivity_threshold
        self.position_threshold = position_threshold
        self.window_time = window_time

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
