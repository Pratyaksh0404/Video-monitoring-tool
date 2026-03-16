from collections import defaultdict
import time


class TrajectoryTracker:

    def __init__(self, roi=None, grid_size=2):
        self.roi = roi
        self.grid_size = grid_size

        self.paths = defaultdict(list)
        self.last_zone = {}
        self.last_change_time = {}
        self.pending_zone = {}          # candidate zone before it's confirmed stable
        self.pending_since = {}         # when we first saw the candidate

        # A zone change is accepted only after guard stays there this long (seconds)
        self.zone_stability_time = 2.5

    def update(self, objects):
        now = time.time()

        for track_id, box in objects.items():
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            zone = self._get_zone(cx, cy)

            if zone is None:
                continue

            prev_zone = self.last_zone.get(track_id)

            # ── First ever detection for this track ───────────────────────
            if prev_zone is None:
                self.last_zone[track_id] = zone
                self.last_change_time[track_id] = now
                self.paths[track_id].append(zone)
                continue

            # ── Same zone as confirmed zone → reset any pending candidate ─
            if zone == prev_zone:
                self.pending_zone.pop(track_id, None)
                self.pending_since.pop(track_id, None)
                continue

            # ── Different zone — start or continue tracking a candidate ───
            if self.pending_zone.get(track_id) != zone:
                # New candidate — start the stability clock
                self.pending_zone[track_id] = zone
                self.pending_since[track_id] = now
                continue

            # ── Same candidate long enough → accept the zone change ───────
            if now - self.pending_since[track_id] >= self.zone_stability_time:
                self.last_zone[track_id] = zone
                self.last_change_time[track_id] = now
                self.pending_zone.pop(track_id, None)
                self.pending_since.pop(track_id, None)

                # Only append if different from last recorded zone (no consecutive dupes)
                path = self.paths[track_id]
                if not path or path[-1] != zone:
                    self.paths[track_id].append(zone)

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

        width  = (rx2 - rx1) / self.grid_size
        height = (ry2 - ry1) / self.grid_size

        col = int((x - rx1) / width)
        row = int((y - ry1) / height)

        # Clamp to grid bounds
        col = min(col, self.grid_size - 1)
        row = min(row, self.grid_size - 1)

        zones = ["A", "B", "C", "D"]
        index = row * self.grid_size + col

        if index < len(zones):
            return zones[index]
        return None