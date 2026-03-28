"""
anomaly_detector.py
───────────────────
Real-time threat and anomaly detection using two pre-trained YOLOv8 models
from HuggingFace. No training required. CPU-friendly.

Models used:
  1. Subh775/Threat-Detection-YOLOv8n
     → Detects: Gun (96.7%), Grenade (93.1%), Knife, other weapons
     → ~2-5ms per frame on CPU (YOLOv8 Nano)


Usage in main_web.py / main.py:
    detector = AnomalyDetector()
    detector.submit(track_id, frame_crop)
    result = detector.get_result(track_id)
    # result = {"label": str, "is_alert": bool, "threat": str|None}
"""

import os
import sys
import threading
import queue
import time
import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
THREAT_MODEL_REPO   = "Subh775/Threat-Detection-YOLOv8n"
FIGHT_MODEL_REPO    = "Musawer14/fight_detection_yolov8"
THREAT_MODEL_FILE   = "weights/best.pt"
FIGHT_MODEL_FILE    = "weights/best.pt"

THREAT_CONFIDENCE   = 0.75   # min confidence to fire weapon alert
FIGHT_CONFIDENCE    = 0.75   # min confidence to fire fight alert

# Set to False to disable fight detection (if model unavailable)
ENABLE_FIGHT_DETECTION = False

# Cache downloaded weights here
_HERE       = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR  = os.path.join(_HERE, "..", "..", "model_cache", "yolo_threat")


class AnomalyDetector:
    """
    Background thread running two YOLOv8 models:
      - Weapon detector (gun, knife, grenade)
      - Fight detector (violence vs normal)

    submit() is non-blocking. get_result() returns latest cached result.
    """

    def __init__(self):
        self._q          = queue.Queue(maxsize=8)
        self._results    = {}   # track_id → result dict
        self._lock       = threading.Lock()
        self._stopped    = False
        self._ready      = False
        self._threat_model = None
        self._fight_model  = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, track_id, frame_crop):
        """Submit a BGR crop for analysis. Non-blocking, drops if queue full."""
        if not self._ready or frame_crop is None or frame_crop.size == 0:
            return
        try:
            self._q.put_nowait((track_id, frame_crop.copy()))
        except queue.Full:
            pass

    def get_result(self, track_id):
        """
        Returns latest result for this track or None.
        {
          "label":    "NORMAL" | "WEAPON" | "FIGHT",
          "threat":   "Gun" | "Knife" | "Fight" | None,
          "score":    float,
          "is_alert": bool,
        }
        """
        with self._lock:
            return self._results.get(track_id)

    def is_ready(self):
        return self._ready

    def stop(self):
        self._stopped = True

    # ── Internal ──────────────────────────────────────────────────────────────

    def _download_model(self, repo_id, filename):
        """Download model weights from HuggingFace hub to local cache."""
        os.makedirs(_CACHE_DIR, exist_ok=True)
        local_name = repo_id.replace("/", "_") + "_" + filename.replace("/", "_")
        local_path = os.path.join(_CACHE_DIR, local_name)

        if os.path.exists(local_path):
            print(f"[AnomalyDetector] Using cached: {local_name}")
            return local_path

        print(f"[AnomalyDetector] Downloading {repo_id}...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=_CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            # Copy to flat name for easy caching
            import shutil
            shutil.copy(path, local_path)
            return local_path
        except Exception as e:
            print(f"[AnomalyDetector] Download failed for {repo_id}: {e}")
            return None

    def _load_models(self):
        try:
            from ultralytics import YOLO

            # ── Weapon detection model ─────────────────────────────────────────
            threat_path = self._download_model(
                THREAT_MODEL_REPO, THREAT_MODEL_FILE)
            if threat_path:
                self._threat_model = YOLO(threat_path)
                self._threat_model.overrides["verbose"] = False
                print("[AnomalyDetector] Weapon detector ready.")
            else:
                print("[AnomalyDetector] Weapon model unavailable — skipping.")

            # ── Fight detection model ──────────────────────────────────────────
            if ENABLE_FIGHT_DETECTION:
                fight_path = self._download_model(
                    FIGHT_MODEL_REPO, FIGHT_MODEL_FILE)
                if fight_path:
                    self._fight_model = YOLO(fight_path)
                    self._fight_model.overrides["verbose"] = False
                    print("[AnomalyDetector] Fight detector ready.")
                else:
                    print("[AnomalyDetector] Fight model unavailable — skipping.")
            else:
                print("[AnomalyDetector] Fight detection disabled (ENABLE_FIGHT_DETECTION=False)")

            if self._threat_model or self._fight_model:
                self._ready = True
                print("[AnomalyDetector] Anomaly detection active.")
            else:
                print("[AnomalyDetector] No models loaded — "
                      "install huggingface_hub: pip install huggingface_hub")

        except ImportError:
            print("[AnomalyDetector] ultralytics not found — "
                  "pip install ultralytics")
        except Exception as e:
            print(f"[AnomalyDetector] Load error: {e}")

    def _analyze(self, track_id, crop):
        """Run both models on a crop and store result."""
        label    = "NORMAL"
        threat   = None
        score    = 0.0
        is_alert = False

        # ── Weapon detection ───────────────────────────────────────────────────
        if self._threat_model:
            try:
                results = self._threat_model(crop, verbose=False)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf >= THREAT_CONFIDENCE:
                        cls_name = results.names[int(box.cls[0])]
                        if conf > score:
                            score    = conf
                            threat   = cls_name
                            label    = "WEAPON"
                            is_alert = True
            except Exception as e:
                pass

        # ── Fight detection ────────────────────────────────────────────────────
        # Only run if no weapon already detected (saves CPU)
        if self._fight_model and not is_alert:
            try:
                results = self._fight_model(crop, verbose=False)[0]
                for box in results.boxes:
                    conf     = float(box.conf[0])
                    cls_name = results.names[int(box.cls[0])].lower()
                    # Fight model classes: "Violence"/"Fight" vs "NoViolence"/"NoFight"
                    if ("violence" in cls_name or "fight" in cls_name) \
                            and "no" not in cls_name \
                            and conf >= FIGHT_CONFIDENCE:
                        if conf > score:
                            score    = conf
                            threat   = "Fighting"
                            label    = "FIGHT"
                            is_alert = True
            except Exception as e:
                pass

        with self._lock:
            self._results[track_id] = {
                "label":     label,
                "threat":    threat,
                "score":     score,
                "is_alert":  is_alert,
                "timestamp": time.time(),
            }

        if is_alert:
            print(f"[AnomalyDetector] ALERT track {track_id}: "
                  f"{label} — {threat} ({score:.0%})")

    def _loop(self):
        self._load_models()
        if not self._ready:
            return

        while not self._stopped:
            try:
                track_id, crop = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._analyze(track_id, crop)