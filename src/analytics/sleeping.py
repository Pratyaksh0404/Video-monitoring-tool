import time


class SleepingMonitor:
    def __init__(
        self,
        sleep_threshold=60,
        min_face_missing_time=15,
        min_height_ratio=0.75,
        min_centroid_drop=0.15
    ):
        self.sleep_threshold = sleep_threshold
        self.min_face_missing_time = min_face_missing_time
        self.min_height_ratio = min_height_ratio
        self.min_centroid_drop = min_centroid_drop

        self.sleep_start_times = {}
        self.last_face_seen = {}
        self.initial_geometry = {}

    def update(self, guard_id, inactive, face_box, person_box):
        current_time = time.time()

        if guard_id == "UNKNOWN" or not inactive or person_box is None:
            self._reset(guard_id)
            return False

        px1, py1, px2, py2 = person_box
        height = py2 - py1
        if height <= 0:
            self._reset(guard_id)
            return False

        centroid_y = (py1 + py2) / 2

        if guard_id not in self.initial_geometry:
            self.initial_geometry[guard_id] = (height, centroid_y)

        base_height, base_centroid = self.initial_geometry[guard_id]

        height_ratio = height / base_height
        centroid_drop = (centroid_y - base_centroid) / base_height

        face_missing = face_box is None

        if face_missing:
            if guard_id not in self.last_face_seen:
                self.last_face_seen[guard_id] = current_time
        else:
            self.last_face_seen[guard_id] = current_time

        prolonged_face_missing = (
            guard_id in self.last_face_seen and
            current_time - self.last_face_seen[guard_id] >= self.min_face_missing_time
        )

        posture_abnormal = (
            height_ratio < self.min_height_ratio or
            centroid_drop > self.min_centroid_drop
        )

        if not (prolonged_face_missing or posture_abnormal):
            self._reset(guard_id)
            return False

        if guard_id not in self.sleep_start_times:
            self.sleep_start_times[guard_id] = current_time
            return False

        if current_time - self.sleep_start_times[guard_id] >= self.sleep_threshold:
            return True

        return False

    def _reset(self, guard_id):
        if guard_id in self.sleep_start_times:
            del self.sleep_start_times[guard_id]
        if guard_id in self.last_face_seen:
            del self.last_face_seen[guard_id]
        if guard_id in self.initial_geometry:
            del self.initial_geometry[guard_id]
