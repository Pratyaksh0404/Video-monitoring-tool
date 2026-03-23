import cv2
import threading
import time


class VideoStreamReader:
    """
    Threaded video reader for both webcam and recorded video files.

    Webcam mode: reads as fast as possible, always serves latest frame.
    File mode:   throttles to the video's native FPS so frames aren't
                 consumed instantly, and loops back to start at end-of-file.
    """

    def __init__(self, source=0):
        self.source  = source
        self.is_file = isinstance(source, str)

        self.cap = cv2.VideoCapture(self.source)

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        # Throttle file playback to native FPS
        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_interval = 1.0 / fps if fps > 0 else 1.0 / 25.0
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total / fps if fps > 0 else 0
            print(f"[VideoStreamReader] File: {source} | {fps:.1f} fps | {total} frames | {duration:.1f}s")
        else:
            self._frame_interval = 0.0

        self._frame   = None
        self._lock    = threading.Lock()
        self._stopped = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stopped:
            t_start = time.time()

            ret, frame = self.cap.read()

            if not ret:
                if self.is_file:
                    # End of file — loop back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._stopped = True
                    break

            with self._lock:
                self._frame = frame

            # Throttle to native FPS for file sources
            if self._frame_interval > 0:
                elapsed = time.time() - t_start
                sleep_t = self._frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

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