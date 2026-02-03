import cv2
from video.stream_reader import VideoStreamReader


def main():
    print("Starting Video Monitoring Tool – Step 1")

    stream = VideoStreamReader(source=0)  # Webcam

    while True:
        frame = stream.read_frame()
        if frame is None:
            print("Failed to read frame. Exiting...")
            break

        cv2.imshow("Live Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested by user.")
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
