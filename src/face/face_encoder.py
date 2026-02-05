import cv2
import numpy as np
import face_recognition


class FaceEncoder:
    def __init__(self, model="large"):
        self.model = model

    def encode(self, frame, face_boxes):
        encodings = []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = []
        for (x1, y1, x2, y2, _) in face_boxes:
            locations.append((y1, x2, y2, x1))

        if not locations:
            return encodings

        vectors = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
            model=self.model
        )

        for vec in vectors:
            encodings.append(vec)

        return encodings
