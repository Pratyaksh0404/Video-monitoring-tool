import cv2


class VideoStreamReader:
    def __init__(self, source=0):
        """
        source = 0 for webcam
        source = path/RTSP for CCTV later
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
