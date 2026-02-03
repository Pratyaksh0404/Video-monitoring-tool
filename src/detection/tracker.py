import math


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = [self._centroid(box) for box in boxes]

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            used_rows = set()
            used_cols = set()

            for i, obj_centroid in enumerate(object_centroids):
                min_dist = float("inf")
                min_j = -1

                for j, input_centroid in enumerate(input_centroids):
                    if j in used_cols:
                        continue

                    dist = math.dist(obj_centroid, input_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j

                if min_j != -1:
                    self.objects[object_ids[i]] = input_centroids[min_j]
                    self.disappeared[object_ids[i]] = 0
                    used_cols.add(min_j)
                    used_rows.add(i)

            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
