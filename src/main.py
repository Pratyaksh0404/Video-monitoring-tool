import cv2
from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker


def main():
    stream = VideoStreamReader(source=0)
    detector = PersonDetector(conf_threshold=0.5)
    tracker = CentroidTracker(max_disappeared=30)

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        detections = detector.detect(frame)
        boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in detections]

        objects = tracker.update(boxes)

        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for obj_id, (cx, cy) in objects.items():
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {obj_id}",
                (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        cv2.imshow("Person Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
