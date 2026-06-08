import cv2
import threading
import time


class VideoStreamer:
    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_bytes = None
        self.fps          = 0.0
        self.frame_count  = 0
        self.source_type  = "webcam"
        self.source_label = "Camera 0"
        self._fps_counter = []
        self._guards_detected   = 0
        self._active_violations = 0
        self._alerts_today      = 0
        self._people_count      = 0
        self._dwell_times       = {}
        self._compliance_scores = {}

    def push_frame(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        now = time.time()
        self._fps_counter.append(now)
        self._fps_counter = [t for t in self._fps_counter if now - t < 2.0]
        self.fps = round(len(self._fps_counter) / 2.0, 1)
        self.frame_count += 1
        with self._lock:
            self._frame_bytes = buf.tobytes()

    def get_frame(self):
        with self._lock:
            return self._frame_bytes

    def get_frame_raw(self):
        """Return latest frame as decoded BGR numpy array (for burst snapshots)."""
        import numpy as np
        with self._lock:
            buf = self._frame_bytes
        if buf is None:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def generate_mjpeg(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
            time.sleep(0.033)

    def set_source(self, source_type, label=""):
        self.source_type  = source_type
        self.source_label = label or source_type

    def update_stats(self, guards_detected=None, active_violations=None,
                     alerts_today=None, people_count=None,
                     dwell_times=None, compliance_scores=None):
        if guards_detected   is not None: self._guards_detected   = guards_detected
        if active_violations is not None: self._active_violations = active_violations
        if alerts_today      is not None: self._alerts_today      = alerts_today
        if people_count      is not None: self._people_count      = people_count
        if dwell_times       is not None: self._dwell_times       = dwell_times
        if compliance_scores is not None: self._compliance_scores = compliance_scores

    @property
    def stats(self):
        return {
            "fps":               self.fps,
            "frame_count":       self.frame_count,
            "source_type":       self.source_type,
            "source_label":      self.source_label,
            "guards_detected":   self._guards_detected,
            "active_violations": self._active_violations,
            "alerts_today":      self._alerts_today,
            "people_count":      self._people_count,
            "dwell_times":       self._dwell_times,
            "compliance_scores": self._compliance_scores,
        }


streamer = VideoStreamer()