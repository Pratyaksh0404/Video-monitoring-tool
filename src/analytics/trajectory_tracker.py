from collections import defaultdict
import time


class TrajectoryTracker:

    def __init__(self, roi=None, grid_size=2, exit_timeout=20):

        self.roi = roi
        self.grid_size = grid_size

        # ordered zone path
        self.zone_paths = defaultdict(list)

        # current zone
        self.current_zone = {}

        # last time inside ROI
        self.last_seen = {}

        # allow guard temporary exit
        self.exit_timeout = exit_timeout

        # stability buffer
        self.zone_buffer = defaultdict(list)

        self.buffer_size = 5


    def update(self, objects):

        now = time.time()

        for track_id, box in objects.items():

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            zone = self._get_zone(cx, cy)

            if zone is not None:

                self.last_seen[track_id] = now

                # collect zones for stabilization
                self.zone_buffer[track_id].append(zone)

                if len(self.zone_buffer[track_id]) > self.buffer_size:
                    self.zone_buffer[track_id].pop(0)

                stable_zone = max(
                    set(self.zone_buffer[track_id]),
                    key=self.zone_buffer[track_id].count
                )

                prev_zone = self.current_zone.get(track_id)

                if prev_zone != stable_zone:

                    self.zone_paths[track_id].append(stable_zone)

                    self.current_zone[track_id] = stable_zone

            else:

                if track_id in self.last_seen:

                    if now - self.last_seen[track_id] > self.exit_timeout:

                        self.reset(track_id)


    def get_current_zone(self, track_id):

        zone = self.current_zone.get(track_id)

        if zone is None:
            return "-"

        return self._zone_name(zone)


    def get_path(self, track_id):

        path = self.zone_paths.get(track_id, [])

        if not path:
            return "-"

        return " -> ".join(self._zone_name(z) for z in path)


    def reset(self, track_id):

        self.zone_paths.pop(track_id, None)
        self.current_zone.pop(track_id, None)
        self.last_seen.pop(track_id, None)
        self.zone_buffer.pop(track_id, None)


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

        return row * self.grid_size + col


    def _zone_name(self, zone_id):

        names = ["A", "B", "C", "D"]

        if zone_id < len(names):
            return names[zone_id]

        return str(zone_id)