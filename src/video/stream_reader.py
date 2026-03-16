import cv2
import threading


class VideoStreamReader:
    """
    Threaded video reader. The background thread continuously pulls frames
    from the capture device so read_frame() always returns the latest frame
    instantly — no queue buildup, no lag.

    Public API is identical to the original so nothing else needs to change.
    """

    def __init__(self, source=0):
        self.source = source
        self.cap    = cv2.VideoCapture(self.source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep OS buffer minimal

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        self._frame   = None
        self._lock    = threading.Lock()
        self._stopped = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def read_frame(self):
        """Return the latest frame, or None if the stream has ended."""
        if self._stopped:
            return None
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        self.cap.release()