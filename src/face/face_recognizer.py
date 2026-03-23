import os
import numpy as np
import face_recognition


class FaceRecognizer:
    def __init__(
        self,
        faces_dir="data/enrolled_faces",
        tolerance=0.55,
        min_confidence_distance=0.45   # faces further than this are always UNKNOWN
    ):
        self.faces_dir = faces_dir
        self.tolerance = tolerance
        self.min_confidence_distance = min_confidence_distance
        self.known_encodings = []
        self.known_ids = []
        self._load_faces()

    def _load_faces(self):
        if not os.path.exists(self.faces_dir):
            print(f"[FaceRecognizer] WARNING: faces_dir not found: {self.faces_dir}")
            return

        loaded = 0
        for guard_id in os.listdir(self.faces_dir):
            guard_path = os.path.join(self.faces_dir, guard_id)
            if not os.path.isdir(guard_path):
                continue

            for img_name in os.listdir(guard_path):
                img_path = os.path.join(guard_path, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encs  = face_recognition.face_encodings(image)
                    for enc in encs:
                        self.known_encodings.append(enc)
                        self.known_ids.append(guard_id)
                        loaded += 1
                except Exception as e:
                    print(f"[FaceRecognizer] Skipping {img_path}: {e}")

        print(f"[FaceRecognizer] Loaded {loaded} encodings for "
              f"{len(set(self.known_ids))} identities: {sorted(set(self.known_ids))}")

    def recognize(self, face_encodings):
        results = []

        for enc in face_encodings:
            if not self.known_encodings:
                results.append("UNKNOWN")
                continue

            distances = face_recognition.face_distance(self.known_encodings, enc)

            if len(distances) == 0:
                results.append("UNKNOWN")
                continue

            best_idx  = np.argmin(distances)
            best_dist = distances[best_idx]

            # Hard distance gate — anything above this is always UNKNOWN
            # regardless of compare_faces result
            if best_dist > self.min_confidence_distance:
                results.append("UNKNOWN")
                continue

            # Secondary check using tolerance
            matches = face_recognition.compare_faces(
                self.known_encodings, enc, tolerance=self.tolerance
            )
            if matches[best_idx]:
                results.append(self.known_ids[best_idx])
            else:
                results.append("UNKNOWN")

        return results