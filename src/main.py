import cv2

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from face.face_recognizer import FaceRecognizer

POST_AREA = (100, 100, 400, 400)

MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH = 40


def is_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def main():
    stream = VideoStreamReader(source=0)
    detector = PersonDetector(conf_threshold=0.5)

    presence_monitor = PresenceMonitor(
        absence_threshold=5,
        confirm_time=2,
        min_motion=5
    )

    inactivity_monitor = InactivityMonitor(
        inactivity_threshold=30,
        position_threshold=40,
        window_time=5
    )

    face_detector = FaceDetector()
    face_encoder = FaceEncoder()
    face_recognizer = FaceRecognizer()

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        detections = detector.detect(frame)

        valid_centroids = []
        person_boxes = []

        for (x1, y1, x2, y2, conf) in detections:
            width = x2 - x1
            height = y2 - y1

            if height < MIN_PERSON_HEIGHT or width < MIN_PERSON_WIDTH:
                continue

            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if is_inside_roi((x1, y1, x2, y2), POST_AREA):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                valid_centroids.append((cx, cy))

        post_status = presence_monitor.update(valid_centroids)

        if post_status == "PRESENT":
            activity_status = inactivity_monitor.update(valid_centroids)
        else:
            activity_status = "NO_PERSON"

        face_boxes = face_detector.detect(frame, person_boxes)
        face_encodings = face_encoder.encode(frame, face_boxes)
        identities = face_recognizer.recognize(face_encodings)

        for i, (fx1, fy1, fx2, fy2, score) in enumerate(face_boxes):
            label = identities[i] if i < len(identities) else "UNKNOWN"

            if label == "UNKNOWN":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
            cv2.putText(
                frame,
                label,
                (fx1, fy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        rx1, ry1, rx2, ry2 = POST_AREA
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        if post_status == "PRESENT":
            post_color = (0, 255, 0)
        elif post_status == "TEMPORARILY_EMPTY":
            post_color = (0, 255, 255)
        else:
            post_color = (0, 0, 255)

        if activity_status == "ACTIVE":
            activity_color = (0, 255, 0)
        elif activity_status == "INACTIVE":
            activity_color = (0, 0, 255)
        else:
            activity_color = (255, 255, 255)

        cv2.putText(
            frame,
            f"Post Status: {post_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            post_color,
            2
        )

        cv2.putText(
            frame,
            f"Activity Status: {activity_status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            activity_color,
            2
        )

        cv2.imshow("Guard Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
