import time
import math
from collections import deque


class PresenceMonitor:
    def __init__(
        self,
        absence_threshold=5,
        confirm_time=2,
        min_motion=15,
        min_area_stability=0.7,
        min_height_stability=0.7
    ):
        self.absence_threshold = absence_threshold
        self.confirm_time = confirm_time
        self.min_motion = min_motion
        self.min_area_stability = min_area_stability
        self.min_height_stability = min_height_stability

        self.last_seen_time = time.time()
        self.first_seen_time = None

        self.centroids = deque(maxlen=20)
        self.areas = deque(maxlen=20)
        self.heights = deque(maxlen=20)

    def update(self, valid_detections):
        current_time = time.time()

        if not valid_detections:
            self.centroids.clear()
            self.areas.clear()
            self.heights.clear()
            self.first_seen_time = None

            if current_time - self.last_seen_time > self.absence_threshold:
                return "ABSENT"
            return "TEMPORARILY_EMPTY"

        self.last_seen_time = current_time

        for (cx, cy, area, height) in valid_detections:
            self.centroids.append((cx, cy))
            self.areas.append(area)
            self.heights.append(height)

        if self.first_seen_time is None:
            self.first_seen_time = current_time
            return "TEMPORARILY_EMPTY"

        if current_time - self.first_seen_time < self.confirm_time:
            return "TEMPORARILY_EMPTY"

        total_motion = 0
        for i in range(1, len(self.centroids)):
            total_motion += math.dist(
                self.centroids[i - 1],
                self.centroids[i]
            )

        if total_motion < self.min_motion:
            return "ABSENT"

        avg_area = sum(self.areas) / len(self.areas)
        stable_area = sum(
            1 for a in self.areas
            if abs(a - avg_area) / avg_area < 0.25
        )

        if stable_area / len(self.areas) < self.min_area_stability:
            return "ABSENT"

        avg_height = sum(self.heights) / len(self.heights)
        stable_height = sum(
            1 for h in self.heights
            if abs(h - avg_height) / avg_height < 0.2
        )

        if stable_height / len(self.heights) < self.min_height_stability:
            return "ABSENT"

        return "PRESENT"
