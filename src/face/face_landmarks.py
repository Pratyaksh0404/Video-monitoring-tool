import cv2
import math
import os
import numpy as np


class FaceLandmarkAnalyzer:
    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LBF model not found at: {model_path}")

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(model_path)

        self.left_eye = [36, 37, 38, 39, 40, 41]
        self.right_eye = [42, 43, 44, 45, 46, 47]

    def _ear(self, landmarks, idx):
        p1 = landmarks[idx[1]]
        p2 = landmarks[idx[2]]
        p3 = landmarks[idx[4]]
        p4 = landmarks[idx[5]]
        p5 = landmarks[idx[0]]
        p6 = landmarks[idx[3]]

        vertical = math.dist(p1, p3) + math.dist(p2, p4)
        horizontal = math.dist(p5, p6)

        if horizontal == 0:
            return 0.0

        return vertical / (2.0 * horizontal)

    def analyze(self, frame, face_box):
        x1, y1, x2, y2 = face_box
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = np.array([[x1, y1, w, h]], dtype=np.int32)

        ok, landmarks = self.facemark.fit(gray, faces)

        if not ok:
            return None

        points = landmarks[0][0]

        left_ear = self._ear(points, self.left_eye)
        right_ear = self._ear(points, self.right_eye)
        ear = (left_ear + right_ear) / 2.0

        eyes_closed = ear < 0.20

        nose = points[30]
        chin = points[8]

        dy = chin[1] - nose[1]
        dx = chin[0] - nose[0]

        pitch = math.degrees(math.atan2(dy, dx))

        return {
            "ear": ear,
            "eyes_closed": eyes_closed,
            "pitch": pitch
        }

