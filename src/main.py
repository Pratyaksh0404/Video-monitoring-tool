import cv2

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from analytics.presence import PresenceMonitor

# ROI
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

    stream = VideoStreamReader(source=0)  # Webcam
    detector = PersonDetector(conf_threshold=0.5)

    presence_monitor = PresenceMonitor(
        absence_threshold=5,
        confirm_time=2,
        min_motion=5
    )

    while True:
        frame = stream.read_frame()
        if frame is None:
            print("Failed to read frame. Exiting...")
            break

        detections = detector.detect(frame)

        valid_centroids = []

        for (x1, y1, x2, y2, conf) in detections:
            width = x2 - x1
            height = y2 - y1

            if height < MIN_PERSON_HEIGHT or width < MIN_PERSON_WIDTH:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if is_inside_roi((x1, y1, x2, y2), POST_AREA):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                height = y2 - y1
                valid_centroids.append((cx, cy, area, height))

        status = presence_monitor.update(valid_centroids)

        # Draw ROI
        rx1, ry1, rx2, ry2 = POST_AREA
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        if status == "PRESENT":
            color = (0, 255, 0)
        elif status == "TEMPORARILY_EMPTY":
            color = (0, 255, 255)
        else:  # ABSENT
            color = (0, 0, 255)

        cv2.putText(
            frame,
            f"Post Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("Presence Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested by user.")
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
