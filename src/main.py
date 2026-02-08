import cv2

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.sleeping import SleepingMonitor
from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from face.face_recognizer import FaceRecognizer
from face.face_landmarks import FaceLandmarkAnalyzer
from analytics.eye_closure import EyeClosureMonitor

POST_AREA = (100, 100, 400, 400)

MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH = 40


def is_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def main():
    stream = VideoStreamReader(source=0)
    detector = PersonDetector(conf_threshold=0.5)
    tracker = CentroidTracker(max_disappeared=30, max_distance=60)

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

    sleeping_monitor = SleepingMonitor(
        sleep_threshold=60,
        min_face_missing_time=15,
        min_height_ratio=0.75,
        min_centroid_drop=0.15
    )

    eye_closure_monitor = EyeClosureMonitor(
        ear_threshold=0.50,
        closed_time_threshold=10
    )

    face_detector = FaceDetector()
    face_encoder = FaceEncoder()
    face_recognizer = FaceRecognizer()
    face_landmarks = FaceLandmarkAnalyzer(
        model_path="face/models/lbfmodel.yaml"
    )

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        detections = detector.detect(frame)

        person_boxes = []
        valid_centroids = []

        for (x1, y1, x2, y2, conf) in detections:
            if (x2 - x1) < MIN_PERSON_WIDTH or (y2 - y1) < MIN_PERSON_HEIGHT:
                continue

            box = (x1, y1, x2, y2)
            person_boxes.append(box)

            if is_inside_roi(box, POST_AREA):
                valid_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        post_status = presence_monitor.update(valid_centroids)

        if post_status == "PRESENT":
            activity_status = inactivity_monitor.update(valid_centroids)
        else:
            activity_status = "NO_PERSON"

        tracked_objects = tracker.update(person_boxes)

        face_boxes = face_detector.detect(frame, list(tracked_objects.values()))
        face_encodings = face_encoder.encode(frame, face_boxes)
        identities = face_recognizer.recognize(face_encodings)

        for track_id, person_box in tracked_objects.items():
            best_iou = 0.0
            face_box = None
            guard_id = "UNKNOWN"

            for i, fb in enumerate(face_boxes):
                overlap = iou(person_box, fb[:4])
                if overlap > best_iou:
                    best_iou = overlap
                    face_box = fb[:4]
                    guard_id = identities[i] if i < len(identities) else "UNKNOWN"

            x1, y1, x2, y2 = person_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if face_box is not None:
                state = face_landmarks.analyze(frame, face_box)

                if state is not None:
                    ear_value = state["ear"]
                    eye_state = eye_closure_monitor.update(track_id, ear_value)

                    ear_text = f"{ear_value:.3f}"
                    pitch_text = f"{state['pitch']:.1f}"

                    if eye_state == "CLOSED_LONG":
                        eye_text = "POSSIBLE NEGLIGENCE (EYES CLOSED)"
                        color = (0, 0, 255)
                    elif eye_state == "CLOSED_SHORT":
                        eye_text = "EYES TEMPORARILY CLOSED"
                        color = (0, 255, 255)
                    elif eye_state == "OPEN":
                        eye_text = "EYES OPEN"
                        color = (0, 255, 0)
                    else:
                        eye_text = "EYES UNKNOWN"
                        color = (255, 255, 255)

                    fx1, fy1, fx2, fy2 = face_box
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                    cv2.putText(
                        frame,
                        f"{guard_id} | {eye_text} | EAR:{ear_text} | PITCH:{pitch_text}",
                        (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )



        rx1, ry1, rx2, ry2 = POST_AREA
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        post_color = (0, 255, 0) if post_status == "PRESENT" else (0, 255, 255) if post_status == "TEMPORARILY_EMPTY" else (0, 0, 255)
        activity_color = (0, 255, 0) if activity_status == "ACTIVE" else (0, 0, 255) if activity_status == "INACTIVE" else (255, 255, 255)

        cv2.putText(frame, f"Post Status: {post_status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, post_color, 2)
        cv2.putText(frame, f"Activity Status: {activity_status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, activity_color, 2)

        cv2.imshow("Guard Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
