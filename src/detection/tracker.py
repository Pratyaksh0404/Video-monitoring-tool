import math


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=60,
                 duplicate_iou_threshold=0.5):
        self.next_object_id          = 0
        self.objects                 = {}
        self.centroids               = {}
        self.disappeared             = {}
        self.max_disappeared         = max_disappeared
        self.max_distance            = max_distance
        self.duplicate_iou_threshold = duplicate_iou_threshold

        # Track IDs deregistered in the last update() call
        # — consumed by main_web.py to clean up trajectory / identity state
        self.recently_deregistered = []

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA);    interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def register(self, box):
        self.objects[self.next_object_id]    = box
        self.centroids[self.next_object_id]  = self._centroid(box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        self.recently_deregistered.append(object_id)
        del self.objects[object_id]
        del self.centroids[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        # Clear last frame's deregistered list
        self.recently_deregistered = []

        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Remove near-duplicate detections
        filtered_boxes = []
        for box in boxes:
            duplicate = False
            for existing_box in self.objects.values():
                if self._iou(box, existing_box) >= self.duplicate_iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                filtered_boxes.append(box)
        boxes = filtered_boxes

        input_centroids = [self._centroid(box) for box in boxes]

        if len(self.objects) == 0:
            for box in boxes:
                self.register(box)
            return self.objects

        object_ids       = list(self.objects.keys())
        object_centroids = list(self.centroids.values())

        used_objects = set()
        used_inputs  = set()

        for i, (ox, oy) in enumerate(object_centroids):
            min_dist = float("inf")
            min_j    = -1
            for j, (ix, iy) in enumerate(input_centroids):
                if j in used_inputs:
                    continue
                dist = math.dist((ox, oy), (ix, iy))
                if dist < min_dist:
                    min_dist = dist
                    min_j    = j

            if min_j != -1 and min_dist <= self.max_distance:
                object_id = object_ids[i]
                self.objects[object_id]    = boxes[min_j]
                self.centroids[object_id]  = input_centroids[min_j]
                self.disappeared[object_id] = 0
                used_objects.add(object_id)
                used_inputs.add(min_j)

        for object_id in list(self.objects.keys()):
            if object_id not in used_objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        for j in range(len(boxes)):
            if j not in used_inputs:
                self.register(boxes[j])

        return self.objects