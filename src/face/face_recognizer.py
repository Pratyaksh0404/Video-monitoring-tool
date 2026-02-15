import os
import cv2
import numpy as np
import face_recognition


class FaceRecognizer:
    def __init__(
        self,
        faces_dir="data/enrolled_faces",
        tolerance=0.65
    ):
        self.faces_dir = faces_dir
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_ids = []

        self._load_faces()

    def _load_faces(self):
        if not os.path.exists(self.faces_dir):
            return

        for guard_id in os.listdir(self.faces_dir):
            guard_path = os.path.join(self.faces_dir, guard_id)
            if not os.path.isdir(guard_path):
                continue

            for img_name in os.listdir(guard_path):
                img_path = os.path.join(guard_path, img_name)

                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)

                for enc in encs:
                    self.known_encodings.append(enc)
                    self.known_ids.append(guard_id)

    def recognize(self, face_encodings):
        results = []

        for enc in face_encodings:
            if not self.known_encodings:
                results.append("UNKNOWN")
                continue

            matches = face_recognition.compare_faces(
                self.known_encodings,
                enc,
                tolerance=self.tolerance
            )

            distances = face_recognition.face_distance(
                self.known_encodings,
                enc
            )

            if len(distances) == 0:
                results.append("UNKNOWN")
                continue

            best_idx = np.argmin(distances)

            if matches[best_idx]:
                results.append(self.known_ids[best_idx])
            else:
                results.append("UNKNOWN")

        return results
