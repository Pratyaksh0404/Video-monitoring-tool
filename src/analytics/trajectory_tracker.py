from collections import defaultdict
import time


class TrajectoryTracker:

    def __init__(self, roi=None, grid_size=2, zone_mode="grid", zones=None):
        """
        zone_mode: "grid"  — legacy behavior, unchanged. Divides roi into
                             a grid_size × grid_size grid, labeled A/B/C/D.
                             This is what guard_monitoring uses.
                   "named" — arbitrary named zones with polygons, e.g.
                             [{"id": "kitchen", "polygon": [(x1,y1), ...]}, ...]
                             Used by retail/warehouse/bank profiles. If no
                             zones are configured yet (not calibrated for
                             this camera), get_current_zone/get_path simply
                             return None/empty — never crashes.
        """
        self.roi       = roi
        self.grid_size = grid_size
        self.zone_mode = zone_mode
        self.zones     = zones or []   # list of {"id": str, "polygon": [(x,y), ...]}

        self.paths            = defaultdict(list)
        self.last_zone        = {}
        self.last_change_time = {}
        self.pending_zone     = {}
        self.pending_since    = {}

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

            if prev_zone is None:
                self.last_zone[track_id]        = zone
                self.last_change_time[track_id] = now
                self.paths[track_id].append(zone)
                continue

            if zone == prev_zone:
                self.pending_zone.pop(track_id, None)
                self.pending_since.pop(track_id, None)
                continue

            if self.pending_zone.get(track_id) != zone:
                self.pending_zone[track_id]  = zone
                self.pending_since[track_id] = now
                continue

            if now - self.pending_since[track_id] >= self.zone_stability_time:
                self.last_zone[track_id]        = zone
                self.last_change_time[track_id] = now
                self.pending_zone.pop(track_id, None)
                self.pending_since.pop(track_id, None)

                path = self.paths[track_id]
                if not path or path[-1] != zone:
                    self.paths[track_id].append(zone)

    def reset_track(self, track_id):
        """Clear all state for a track_id when it is deregistered.
        Prevents path/zone data bleeding into the next person assigned
        the same track_id by the centroid tracker."""
        self.paths.pop(track_id, None)
        self.last_zone.pop(track_id, None)
        self.last_change_time.pop(track_id, None)
        self.pending_zone.pop(track_id, None)
        self.pending_since.pop(track_id, None)

    def get_current_zone(self, track_id):
        return self.last_zone.get(track_id, None)

    def get_path(self, track_id):
        path = self.paths.get(track_id, [])
        if not path:
            return ""
        return " -> ".join(path)

    # ── Zone resolution ───────────────────────────────────────────────────────

    def _get_zone(self, x, y):
        if self.zone_mode == "named":
            return self._get_named_zone(x, y)
        return self._get_grid_zone(x, y)

    def _get_grid_zone(self, x, y):
        """Original fixed-grid logic — unchanged, still the default."""
        if not self.roi:
            return None

        rx1, ry1, rx2, ry2 = self.roi

        if not (rx1 <= x <= rx2 and ry1 <= y <= ry2):
            return None

        width  = (rx2 - rx1) / self.grid_size
        height = (ry2 - ry1) / self.grid_size

        col = min(int((x - rx1) / width),  self.grid_size - 1)
        row = min(int((y - ry1) / height), self.grid_size - 1)

        zones = ["A", "B", "C", "D"]
        index = row * self.grid_size + col

        if index < len(zones):
            return zones[index]
        return None

    def _get_named_zone(self, x, y):
        """
        Point-in-polygon zone lookup for retail/warehouse/bank profiles.
        Returns None if no zones are configured yet (camera not calibrated)
        or if the point falls outside every defined zone polygon —
        never raises, so an uncalibrated deployment just doesn't get zone
        tracking rather than crashing.
        """
        if not self.zones:
            return None
        for zone in self.zones:
            polygon = zone.get("polygon")
            if polygon and self._point_in_polygon(x, y, polygon):
                return zone["id"]
        return None

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        """Standard ray-casting point-in-polygon test. No external deps."""
        n = len(polygon)
        inside = False
        px, py = polygon[0]
        for i in range(1, n + 1):
            qx, qy = polygon[i % n]
            if y > min(py, qy) and y <= max(py, qy) and x <= max(px, qx):
                if py != qy:
                    xinters = (y - py) * (qx - px) / (qy - py) + px
                if px == qx or x <= xinters:
                    inside = not inside
            px, py = qx, qy
        return inside
