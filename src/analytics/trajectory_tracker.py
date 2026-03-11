from collections import defaultdict
import time


class TrajectoryTracker:

    def __init__(self, roi=None, grid_size=2):

        self.roi = roi
        self.grid_size = grid_size

        self.paths = defaultdict(list)
        self.last_zone = {}
        self.last_change_time = {}

        # minimum seconds before zone change is accepted
        self.zone_stability_time = 1.2


    def update(self, objects):

        for track_id, box in objects.items():

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            zone = self._get_zone(cx, cy)

            if zone is None:
                continue

            prev_zone = self.last_zone.get(track_id)

            now = time.time()

            # first zone
            if prev_zone is None:
                self.paths[track_id].append(zone)
                self.last_zone[track_id] = zone
                self.last_change_time[track_id] = now
                continue

            # same zone → ignore
            if zone == prev_zone:
                continue

            # check stability time
            last_time = self.last_change_time.get(track_id, 0)

            if now - last_time < self.zone_stability_time:
                return

            # valid zone change
            self.paths[track_id].append(zone)

            self.last_zone[track_id] = zone
            self.last_change_time[track_id] = now


    def get_current_zone(self, track_id):

        return self.last_zone.get(track_id, None)


    def get_path(self, track_id):

        path = self.paths.get(track_id, [])

        if not path:
            return ""

        return " -> ".join(path)


    def _get_zone(self, x, y):

        if not self.roi:
            return None

        rx1, ry1, rx2, ry2 = self.roi

        if not (rx1 <= x <= rx2 and ry1 <= y <= ry2):
            return None

        width = (rx2 - rx1) / self.grid_size
        height = (ry2 - ry1) / self.grid_size

        col = int((x - rx1) / width)
        row = int((y - ry1) / height)

        zones = ["A", "B", "C", "D"]

        index = row * self.grid_size + col

        if index < len(zones):
            return zones[index]

        return None