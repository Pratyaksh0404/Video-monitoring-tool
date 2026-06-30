"""
camera_manager.py
─────────────────
Manages multiple camera pipelines for the NoviSentra dashboard.

Each camera gets its own:
  - VideoStreamReader
  - Detection pipeline thread (runs main_web.run)
  - Frame buffer (for MJPEG streaming)
  - Stats + alert attribution

The dashboard can switch between cameras or view them in a grid.

Architecture:
  CameraManager
    ├── CameraInstance("cam_1", source=0, name="Main Entrance")
    │     ├── thread → main_web.run(source=0, streamer=per_cam_streamer)
    │     └── streamer → /video_feed/cam_1
    ├── CameraInstance("cam_2", source="rtsp://...", name="Parking Lot")
    │     ├── thread → main_web.run(source="rtsp://...", streamer=per_cam_streamer)
    │     └── streamer → /video_feed/cam_2
    └── ...

Usage:
  manager = CameraManager(config)
  manager.start_all()
  manager.get_cameras()  → list of camera info dicts
  manager.get_streamer("cam_1")  → CameraStreamer instance
  manager.switch_source("cam_1", new_source)
"""

import os
import threading
import time
import cv2
import queue


class CameraStreamer:
    """
    Per-camera frame buffer + MJPEG generator.
    Replacement for the global 'streamer' when in multi-camera mode.
    """

    def __init__(self, cam_id, name):
        self.cam_id  = cam_id
        self.name    = name
        self._frame  = None
        self._lock   = threading.Lock()
        self._stats  = {
            "source_type":      "webcam",
            "source_label":     name,
            "fps":              "—",
            "guards_detected":  0,
            "active_violations": 0,
            "alerts_today":     0,
        }

        # FPS calculation
        self._frame_count = 0
        self._fps_start   = time.time()
        self._fps         = 0

    @property
    def fps(self):
        return self._fps

    @property
    def stats(self):
        s = dict(self._stats)
        s["fps"] = self._fps
        return s

    def set_source(self, source_type, source_label):
        self._stats["source_type"]  = source_type
        self._stats["source_label"] = source_label

    def update_stats(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._stats:
                self._stats[k] = v

    def push_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()

        # FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_start = time.time()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def generate_mjpeg(self):
        """Generator for Flask MJPEG streaming."""
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
            time.sleep(0.03)  # ~30fps cap


class CameraInstance:
    """
    Represents one camera and its pipeline thread.
    """

    def __init__(self, cam_id, config):
        self.cam_id   = cam_id
        self.name     = config.get("name", cam_id)
        self.source   = config.get("source", 0)
        self.enabled  = config.get("enabled", True)
        self.loop     = config.get("loop", True)
        self.streamer = CameraStreamer(cam_id, self.name)

        self._thread      = None
        self._stop_event  = threading.Event()

    def start(self):
        """Start the detection pipeline for this camera."""
        if not self.enabled:
            print(f"[CameraManager] {self.cam_id} ({self.name}) — disabled, skipping.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_pipeline,
            daemon=True
        )
        self._thread.start()
        print(f"[CameraManager] {self.cam_id} ({self.name}) — started. Source: {self.source}")

    def stop(self):
        """Stop the pipeline thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print(f"[CameraManager] {self.cam_id} ({self.name}) — stopped.")

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _run_pipeline(self):
        """
        Runs the main_web pipeline for this camera.
        Imports main_web and calls run() with this camera's streamer.
        """
        import main_web
        from analytics.behavior_classifier import BehaviorClassifier
        from analytics.anomaly_detector import AnomalyDetector

        beh_label_cache = {}
        beh_cache_lock  = threading.Lock()
        worker          = main_web.BehaviorWorker(beh_label_cache, beh_cache_lock)
        anomaly_det     = AnomalyDetector()

        source_label = self.name

        while not self._stop_event.is_set():
            main_web.run(
                source=self.source,
                source_label=source_label,
                beh_worker=worker,
                beh_label_cache=beh_label_cache,
                beh_cache_lock=beh_cache_lock,
                anomaly_det=anomaly_det,
                stop_event=self._stop_event,
                cam_streamer=self.streamer,
            )
            if self._stop_event.is_set():
                break

            # For video files — loop or stop
            if isinstance(self.source, str) and self.loop:
                print(f"[CameraManager] {self.cam_id} — looping video: {self.source}")
                time.sleep(1)
            else:
                print(f"[CameraManager] {self.cam_id} — stream ended.")
                break

        worker.stop()
        anomaly_det.stop()

    def to_dict(self):
        return {
            "cam_id":  self.cam_id,
            "name":    self.name,
            "source":  str(self.source),
            "enabled": self.enabled,
            "alive":   self.is_alive(),
            "stats":   self.streamer.stats,
        }


class CameraManager:
    """
    Manages all camera instances.
    """

    def __init__(self, config_path=None):
        self.cameras = {}  # cam_id → CameraInstance
        self._config = {}

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            print(f"[CameraManager] Loaded config: {config_path}")
        except ImportError:
            print("[CameraManager] PyYAML not installed — cannot load camera config.")
            return
        except Exception as e:
            print(f"[CameraManager] Config error: {e}")
            return

        cams = self._config.get("cameras", {})
        for cam_id, cam_cfg in cams.items():
            self.cameras[cam_id] = CameraInstance(cam_id, cam_cfg)

    @property
    def is_multi_camera(self):
        return self._config.get("multi_camera_enabled", False)

    @property
    def default_camera(self):
        return self._config.get("default_camera", "cam_1")

    def start_all(self):
        """Start all enabled cameras."""
        for cam_id, cam in self.cameras.items():
            if cam.enabled:
                cam.start()

    def stop_all(self):
        """Stop all cameras."""
        for cam in self.cameras.values():
            cam.stop()

    def get_camera(self, cam_id):
        return self.cameras.get(cam_id)

    def get_streamer(self, cam_id):
        cam = self.cameras.get(cam_id)
        return cam.streamer if cam else None

    def get_cameras_list(self):
        """Return list of camera info dicts for the API."""
        return [cam.to_dict() for cam in self.cameras.values()]

    def add_camera(self, cam_id, name, source, enabled=True):
        """Dynamically add a camera."""
        cfg = {"name": name, "source": source, "enabled": enabled}
        cam = CameraInstance(cam_id, cfg)
        self.cameras[cam_id] = cam
        if enabled:
            cam.start()
        return cam.to_dict()

    def remove_camera(self, cam_id):
        """Stop and remove a camera."""
        cam = self.cameras.pop(cam_id, None)
        if cam:
            cam.stop()
            return True
        return False
