"""
camera_manager.py
─────────────────
Manages multiple camera pipelines for the NoviSentra dashboard.

v3 changes:
  - Zone-as-camera model (see alerts/alert_manager.py): passes camera name
    and whether named zones are configured into the per-thread context,
    so alerts get a meaningful section label ("Men's Section") even when
    no sub-zone polygons are calibrated for that camera.
  - rename_camera() / rename_zone() — used by the admin panel so clients
    can rename their own cameras/zones without touching YAML directly.
    Both persist back to camera_config.yaml using ruamel.yaml (preserves
    comments/formatting), same convention as rules_config.yaml saves.

Each camera gets its own:
  - VideoStreamReader
  - Detection pipeline thread (runs main_web.run)
  - Frame buffer (for MJPEG streaming)
  - Stats + alert attribution (via camera_id/name context, see above)

Usage:
  manager = CameraManager(config)
  manager.start_all()
  manager.get_cameras()  → list of camera info dicts
  manager.get_streamer("cam_1")  → CameraStreamer instance
  manager.rename_camera("cam_1", "New Name")
  manager.rename_zone("cam_2", "kitchen", "Prep Kitchen")
"""

import os
import re
import threading
import time
import cv2
import queue


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitute_env_vars(text: str) -> str:
    """
    Replaces ${VAR_NAME} in a string with the value of environment
    variable VAR_NAME. Used so camera_config.yaml can reference
    rtsp://${CAM1_USER}:${CAM1_PASS}@192.168.1.10/stream1 instead of
    embedding real credentials in plaintext YAML — matching the same
    env-var convention already used for email/Twilio secrets.
    Missing env vars log a warning and substitute empty string rather
    than crashing, so a misconfigured camera doesn't take down startup.
    """
    if not isinstance(text, str):
        return text

    def _replace(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            print(f"[CameraManager] ⚠ Env var '{var_name}' referenced in "
                  f"camera_config.yaml but not set — substituting empty string.")
            return ""
        return value

    return _ENV_VAR_PATTERN.sub(_replace, text)


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
            time.sleep(0.03)


class CameraInstance:
    """
    Represents one camera and its pipeline thread.
    """

    def __init__(self, cam_id, config):
        self.cam_id     = cam_id
        self.name       = config.get("name", cam_id)
        self.source     = config.get("source", 0)
        self.enabled    = config.get("enabled", True)
        self.loop       = config.get("loop", True)
        self.profile_id = config.get("profile", "guard_monitoring")
        self.zones      = config.get("zones", [])   # named zones for this camera, if any
        self.streamer   = CameraStreamer(cam_id, self.name)

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
        print(f"[CameraManager] {self.cam_id} ({self.name}) — started. "
              f"Source: {self.source} | Profile: {self.profile_id} | "
              f"Named zones: {len(self.zones)}")

    def stop(self):
        """Stop the pipeline thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print(f"[CameraManager] {self.cam_id} ({self.name}) — stopped.")

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def set_profile(self, profile_id: str):
        """
        Change this camera's active profile. Takes effect on the VERY
        NEXT capability check inside main_web.py's already-running
        pipeline thread — register_camera_profile() writes to a genuinely
        shared, cross-thread registry (see alerts/alert_manager.py),
        which is what actually makes a live admin-panel switch work
        without restarting the camera stream. (A plain contextvar update
        here would NOT work — verified empirically — since the camera's
        pipeline thread is already blocked inside a single long-running
        run() call and never re-reads a per-thread contextvar set from
        this different thread.)
        """
        self.profile_id = profile_id
        from alerts.alert_manager import register_camera_profile
        register_camera_profile(self.cam_id, profile_id)
        print(f"[CameraManager] {self.cam_id} — profile switched LIVE to '{profile_id}' "
              f"(no restart needed).")

    def rename(self, new_name: str):
        """Update this camera's display name (in-memory only — caller
        (CameraManager.rename_camera) handles persisting to YAML)."""
        self.name = new_name
        self.streamer.name = new_name
        self.streamer.set_source(self.streamer.stats.get("source_type", "webcam"), new_name)

    def rename_zone(self, zone_id: str, new_label: str) -> bool:
        """Update a zone's display label (id stays stable — alerts/history
        reference zones by id, only the human-readable label changes)."""
        for z in self.zones:
            if z.get("id") == zone_id:
                z["label"] = new_label
                return True
        return False

    def _run_pipeline(self):
        """Runs the main_web pipeline for this camera."""
        import main_web
        from analytics.behavior_classifier import BehaviorClassifier
        from analytics.anomaly_detector import AnomalyDetector
        from alerts.alert_manager import (set_camera_context, set_profile_context,
                                          register_camera_profile)

        set_camera_context(self.cam_id, camera_name=self.name,
                           has_named_zones=bool(self.zones))
        set_profile_context(self.profile_id)
        # Seed the LIVE registry too — this is what main_web.py's
        # capability checks actually read (the contextvar above is only
        # used for alert-tagging fallback and initial log messages).
        register_camera_profile(self.cam_id, self.profile_id)

        beh_label_cache = {}
        beh_cache_lock  = threading.Lock()
        worker          = main_web.BehaviorWorker(beh_label_cache, beh_cache_lock)
        anomaly_det     = AnomalyDetector()

        source_label = self.name

        # Bounded reconnect-with-backoff for webcam (integer) sources.
        # Previously ANY failure here — including the dual-camera open
        # hang this was written to fix — permanently killed the camera's
        # thread for the rest of the process's life (the "else: stream
        # ended; break" branch below). Converting the hang into a clean
        # RuntimeError (see src/video/stream_reader.py) only helps if the
        # camera then gets another chance instead of staying dark.
        RECONNECT_DELAY_SEC      = 5
        MAX_CONSECUTIVE_FAILURES = 20   # ~100s of retrying before giving up
        consecutive_failures     = 0

        while not self._stop_event.is_set():
            # Re-assert context at the top of every loop — picks up any
            # rename()/set_profile() call made mid-run on the NEXT iteration.
            set_camera_context(self.cam_id, camera_name=self.name,
                               has_named_zones=bool(self.zones))
            set_profile_context(self.profile_id)

            _run_started = time.time()
            main_web.run(
                source=self.source,
                source_label=source_label,
                beh_worker=worker,
                beh_label_cache=beh_label_cache,
                beh_cache_lock=beh_cache_lock,
                anomaly_det=anomaly_det,
                stop_event=self._stop_event,
                cam_streamer=self.streamer,
                zones=self.zones,   # named zones for this camera, if calibrated
            )
            _ran_for = time.time() - _run_started
            if self._stop_event.is_set():
                break

            from video.stream_reader import is_stream_url
            _is_rtsp = isinstance(self.source, str) and is_stream_url(self.source)

            if isinstance(self.source, str) and self.loop and not _is_rtsp:
                print(f"[CameraManager] {self.cam_id} — looping video: {self.source}")
                time.sleep(1)
                consecutive_failures = 0
            elif isinstance(self.source, int) or _is_rtsp:
                # Reconnect-with-backoff for BOTH webcam (int) sources and
                # RTSP/DVR (str URL) sources — a dropped DVR channel is
                # the expected, routine failure mode for a real
                # multi-camera deployment (see stream_reader.py's inner
                # reconnect logic, which handles most blips without ever
                # reaching here; this outer retry is for when the DVR
                # itself is down long enough to exhaust that inner budget).
                #
                # A run that actually streamed for a while before dropping
                # (e.g. a genuine hardware/network hiccup after hours of
                # uptime) isn't the same failure mode as an instant open
                # failure — don't let it eat into the same short retry budget.
                if _ran_for >= 30:
                    consecutive_failures = 0
                consecutive_failures += 1
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    print(f"[CameraManager] {self.cam_id} — giving up after "
                          f"{consecutive_failures} consecutive failures. "
                          f"Camera will stay offline until manually restarted.")
                    break
                print(f"[CameraManager] {self.cam_id} — stream stopped, "
                      f"reconnecting in {RECONNECT_DELAY_SEC}s "
                      f"(attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})...")
                time.sleep(RECONNECT_DELAY_SEC)
            else:
                print(f"[CameraManager] {self.cam_id} — stream ended.")
                break

        worker.stop()
        anomaly_det.stop()

    def to_dict(self):
        return {
            "cam_id":     self.cam_id,
            "name":       self.name,
            "source":     str(self.source),
            "enabled":    self.enabled,
            "alive":      self.is_alive(),
            "profile":    self.profile_id,
            "zones":      [{"id": z.get("id"), "label": z.get("label")} for z in self.zones],
            "stats":      self.streamer.stats,
        }


class CameraManager:
    """
    Manages all camera instances.
    """

    def __init__(self, config_path=None):
        self.cameras     = {}
        self._config     = {}
        self._config_path = config_path

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
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
            # Substitute ${VAR} placeholders in the source URL with real
            # credentials from environment variables — keeps passwords
            # out of camera_config.yaml entirely.
            if "source" in cam_cfg:
                cam_cfg["source"] = _substitute_env_vars(cam_cfg["source"])
            self.cameras[cam_id] = CameraInstance(cam_id, cam_cfg)

    @property
    def is_multi_camera(self):
        return self._config.get("multi_camera_enabled", False)

    @property
    def default_camera(self):
        return self._config.get("default_camera", "cam_1")

    def start_all(self):
        for cam_id, cam in self.cameras.items():
            if cam.enabled:
                cam.start()

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()

    def get_camera(self, cam_id):
        return self.cameras.get(cam_id)

    def get_streamer(self, cam_id):
        cam = self.cameras.get(cam_id)
        return cam.streamer if cam else None

    def get_cameras_list(self):
        return [cam.to_dict() for cam in self.cameras.values()]

    def set_camera_profile(self, cam_id, profile_id):
        cam = self.cameras.get(cam_id)
        if not cam:
            return False
        cam.set_profile(profile_id)
        return True

    # ── Admin panel operations — rename camera/zone, persist to YAML ────────

    def rename_camera(self, cam_id: str, new_name: str) -> bool:
        """Rename a camera and persist the change to camera_config.yaml."""
        cam = self.cameras.get(cam_id)
        if not cam:
            return False
        cam.rename(new_name)
        self._persist_config()
        return True

    def rename_zone(self, cam_id: str, zone_id: str, new_label: str) -> bool:
        """Rename a zone within a camera and persist to camera_config.yaml."""
        cam = self.cameras.get(cam_id)
        if not cam:
            return False
        ok = cam.rename_zone(zone_id, new_label)
        if ok:
            self._persist_config()
        return ok

    def _persist_config(self):
        """
        Write current camera names/zone labels back to camera_config.yaml,
        preserving comments/formatting via ruamel.yaml — same convention
        already used for rules_config.yaml saves from the Settings tab.
        """
        if not self._config_path:
            print("[CameraManager] No config_path set — cannot persist rename.")
            return
        try:
            from ruamel.yaml import YAML
            yaml_rt = YAML()
            yaml_rt.preserve_quotes = True

            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml_rt.load(f)

            for cam_id, cam in self.cameras.items():
                if cam_id not in data.get("cameras", {}):
                    continue
                data["cameras"][cam_id]["name"] = cam.name
                if cam.zones:
                    for zone_entry in data["cameras"][cam_id].get("zones", []):
                        for z in cam.zones:
                            if z.get("id") == zone_entry.get("id"):
                                zone_entry["label"] = z.get("label")

            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml_rt.dump(data, f)
            print(f"[CameraManager] Persisted rename to {self._config_path}")
        except ImportError:
            print("[CameraManager] ruamel.yaml not installed — rename applied "
                  "in-memory only, will revert on restart. Run: pip install ruamel.yaml")
        except Exception as e:
            print(f"[CameraManager] Failed to persist rename: {e}")

    def add_camera(self, cam_id, name, source, enabled=True, profile="guard_monitoring"):
        cfg = {"name": name, "source": source, "enabled": enabled, "profile": profile}
        cam = CameraInstance(cam_id, cfg)
        self.cameras[cam_id] = cam
        if enabled:
            cam.start()
        return cam.to_dict()

    def remove_camera(self, cam_id):
        cam = self.cameras.pop(cam_id, None)
        if cam:
            cam.stop()
            return True
        return False
