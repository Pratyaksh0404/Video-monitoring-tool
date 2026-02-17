import cv2


class VideoStreamReader:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")


    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
