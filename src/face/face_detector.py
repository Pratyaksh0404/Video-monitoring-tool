import cv2
import os
import numpy as np


class FaceDetector:
    def __init__(
        self,
        model_path=None,
        input_size=(320, 320),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(
                base_dir,
                "models",
                "face_detection_yunet_2023mar.onnx"
            )

        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold,
            nms_threshold,
            top_k
        )

    def detect(self, frame, person_boxes=None):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))

        faces = []

        if person_boxes is None:
            _, detections = self.detector.detect(frame)
            if detections is None:
                return faces

            for d in detections:
                x, y, bw, bh = map(int, d[:4])
                faces.append((x, y, x + bw, y + bh, float(d[4])))

            return faces

        for (px1, py1, px2, py2) in person_boxes:
            roi = frame[py1:py2, px1:px2]
            if roi.size == 0:
                continue

            rh, rw = roi.shape[:2]
            self.detector.setInputSize((rw, rh))

            _, detections = self.detector.detect(roi)
            if detections is None:
                continue

            for d in detections:
                x, y, bw, bh = map(int, d[:4])
                fx1 = px1 + x
                fy1 = py1 + y
                fx2 = fx1 + bw
                fy2 = fy1 + bh
                faces.append((fx1, fy1, fx2, fy2, float(d[4])))

        return faces
