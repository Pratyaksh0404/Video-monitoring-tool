class PatrolAnalyzer:

    def __init__(self, roi):

        x1, y1, x2, y2 = roi

        self.zones = {
            "A": (x1, y1, (x1+x2)//2, (y1+y2)//2),
            "B": ((x1+x2)//2, y1, x2, (y1+y2)//2),
            "C": (x1, (y1+y2)//2, (x1+x2)//2, y2),
            "D": ((x1+x2)//2, (y1+y2)//2, x2, y2),
        }

        self.visited = {}

    def _point_in_zone(self, point, zone):

        zx1, zy1, zx2, zy2 = zone
        x, y = point

        return zx1 <= x <= zx2 and zy1 <= y <= zy2

    def update(self, track_id, centroid):

        for zone_name, zone in self.zones.items():

            if self._point_in_zone(centroid, zone):

                if track_id not in self.visited:
                    self.visited[track_id] = set()

                self.visited[track_id].add(zone_name)

                return zone_name

        return None

    def coverage(self, track_id):

        if track_id not in self.visited:
            return 0

        return int((len(self.visited[track_id]) / 4) * 100)

    def visited_zones(self, track_id):

        return sorted(list(self.visited.get(track_id, [])))