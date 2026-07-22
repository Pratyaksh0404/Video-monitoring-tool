class PatrolAnalyzer:

    def __init__(self, roi=None, zone_mode="grid", zones=None):
        """
        zone_mode: "grid"  — legacy behavior, unchanged. roi is required,
                             divided into 4 fixed quadrants A/B/C/D.
                   "named" — arbitrary named zones with polygons, e.g.
                             [{"id": "table_1", "polygon": [(x1,y1), ...]}, ...]
                             coverage() is then computed as a fraction of
                             however many named zones are configured.
        """
        self.zone_mode    = zone_mode
        self.zones_named  = zones or []
        self.visited      = {}

        if zone_mode == "grid":
            if roi is None:
                raise ValueError(
                    "PatrolAnalyzer: roi is required when zone_mode='grid' "
                    "(the default). Pass zone_mode='named' with a zones "
                    "list instead for non-grid profiles."
                )
            x1, y1, x2, y2 = roi
            self.zones = {
                "A": (x1, y1, (x1 + x2) // 2, (y1 + y2) // 2),
                "B": ((x1 + x2) // 2, y1, x2, (y1 + y2) // 2),
                "C": (x1, (y1 + y2) // 2, (x1 + x2) // 2, y2),
                "D": ((x1 + x2) // 2, (y1 + y2) // 2, x2, y2),
            }
            self.total_zones = 4
        else:
            self.zones = {}
            self.total_zones = max(len(self.zones_named), 1)

    def _point_in_zone_rect(self, point, zone):
        zx1, zy1, zx2, zy2 = zone
        x, y = point
        return zx1 <= x <= zx2 and zy1 <= y <= zy2

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        """Standard ray-casting point-in-polygon test. No external deps —
        deliberately duplicated from trajectory_tracker.py rather than
        shared, matching this project's existing style of small
        self-contained analytics modules."""
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

    def update(self, track_id, centroid):
        if self.zone_mode == "grid":
            for zone_name, zone in self.zones.items():
                if self._point_in_zone_rect(centroid, zone):
                    self.visited.setdefault(track_id, set()).add(zone_name)
                    return zone_name
            return None

        x, y = centroid
        for zone in self.zones_named:
            polygon = zone.get("polygon")
            if polygon and self._point_in_polygon(x, y, polygon):
                self.visited.setdefault(track_id, set()).add(zone["id"])
                return zone["id"]
        return None

    def coverage(self, track_id):
        if track_id not in self.visited:
            return 0
        return int((len(self.visited[track_id]) / self.total_zones) * 100)

    def visited_zones(self, track_id):
        return sorted(list(self.visited.get(track_id, [])))
