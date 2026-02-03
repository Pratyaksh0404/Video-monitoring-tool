import cv2
from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector


def main():
    print("Starting Video Monitoring Tool – Step 2")

    stream = VideoStreamReader(source=0)
    detector = PersonDetector(conf_threshold=0.5)

    while True:
        frame = stream.read_frame()
        if frame is None:
            break

        persons = detector.detect(frame)

        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
