from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        persons = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # YOLO class 0 = person
            if cls_id == 0 and confidence >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2, confidence))

        return persons
