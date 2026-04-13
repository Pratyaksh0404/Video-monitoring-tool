"""
anomaly_detector.py
───────────────────
Weapon detection + best-effort fire/smoke detection.

Fire model strategy (REVISED)
──────────────────────────────
HuggingFace access fails on this network (all repos return 404/private).
We now check for a LOCAL fire model first before trying any downloads.

To enable fire detection:
  1. Download any YOLOv8 fire/smoke model manually (e.g. from Roboflow,
     GitHub releases, or any public source)
  2. Rename it to  fire_model.pt
  3. Place it in:  model_cache/yolo_threat/fire_model.pt
  4. Restart the app — fire detection will enable automatically.

Recommended free models to try:
  - https://huggingface.co/Liang44/fire-detection/resolve/main/best.pt
    (download in browser, rename to fire_model.pt)
  - Any YOLOv8n/s trained on fire/smoke dataset from Roboflow Universe

The HuggingFace download attempts are kept as fallback but will skip
quickly when network is unavailable (timeout added).
"""

import os
import shutil
import threading
import queue
import time

WEAPON_MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
WEAPON_MODEL_FILE = "weights/best.pt"
WEAPON_CONFIDENCE = 0.75
WEAPON_EXCLUDED   = {"grenade", "Grenade", "bomb", "Bomb",
                     "explosion", "Explosion"}

FIRE_CONFIDENCE   = 0.50   # raised from 0.45 for fewer false positives
FIRE_VALID_CLASSES = {"fire", "Fire", "smoke", "Smoke",
                      "flames", "Flames", "wildfire", "Wildfire"}
FIRE_REJECT_CLASSES = {"gun", "Gun", "knife", "Knife",
                       "grenade", "Grenade", "explosion", "Explosion"}

# LOCAL fire model path — check this FIRST before any downloads
LOCAL_FIRE_MODEL_NAME = "fire_model.pt"

# Ordered list of (repo_id, filename) to try for fire detection
# These are kept as fallback but will fail fast on blocked networks
FIRE_MODEL_CANDIDATES = [
    ("arnabdhar/YOLOv8-Fire-Detection",
     "runs/detect/train/weights/best.pt"),
    ("AndreyGermanov/yolov8_obb_fire_smoke",
     "best.pt"),
    ("keremberke/yolov8s-fire-smoke-detection",
     "best.pt"),
]

_HERE      = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, "..", "..", "model_cache", "yolo_threat")


def _purge_stale_subdirs():
    """Remove leftover hf_hub_download subdirectories that cause cache collisions."""
    for d in ("weights", "runs", "_tmp_download"):
        p = os.path.join(_CACHE_DIR, d)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


class AnomalyDetector:

    def __init__(self):
        self._weapon_q    = queue.Queue(maxsize=8)
        self._results     = {}
        self._fire_q      = queue.Queue(maxsize=4)
        self._fire_result = None
        self._lock        = threading.Lock()
        self._stopped     = False
        self._ready       = False
        self._fire_ready  = False
        self._weapon_model = None
        self._fire_model   = None

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, track_id, frame_crop):
        if not self._ready or frame_crop is None or frame_crop.size == 0:
            return
        try:
            self._weapon_q.put_nowait((track_id, frame_crop.copy()))
        except queue.Full:
            pass

    def get_result(self, track_id):
        with self._lock:
            return self._results.get(track_id)

    def submit_frame(self, frame):
        if not self._fire_ready or frame is None or frame.size == 0:
            return
        try:
            self._fire_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_fire_result(self):
        with self._lock:
            return self._fire_result

    def is_ready(self):
        return self._ready

    def stop(self):
        self._stopped = True

    # ── Model loading ─────────────────────────────────────────────────────────

    def _flat_path(self, repo_id, filename):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        safe = repo_id.replace("/", "_") + "__" + filename.replace("/", "_")
        return os.path.join(_CACHE_DIR, safe)

    def _try_hf_download(self, repo_id, filename):
        local = self._flat_path(repo_id, filename)
        if os.path.exists(local):
            print(f"[AnomalyDetector] Cache hit: {os.path.basename(local)}")
            return local
        print(f"[AnomalyDetector] Trying HF: {repo_id}/{filename}")
        try:
            from huggingface_hub import hf_hub_download
            tmp = os.path.join(_CACHE_DIR, "_tmp")
            os.makedirs(tmp, exist_ok=True)
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   local_dir=tmp, local_dir_use_symlinks=False)
            shutil.copy(path, local)
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"[AnomalyDetector] Downloaded: {os.path.basename(local)}")
            return local
        except Exception as e:
            print(f"[AnomalyDetector] HF failed ({repo_id}): {type(e).__name__}")
            return None

    def _is_valid_fire_model(self, model):
        """Returns True if model has fire/smoke classes, False if it's a weapon model."""
        names = list(model.names.values())
        names_lower = {n.lower() for n in names}
        is_weapon = names_lower & {c.lower() for c in FIRE_REJECT_CLASSES}
        is_fire   = names_lower & {c.lower() for c in FIRE_VALID_CLASSES}
        if is_weapon and not is_fire:
            print(f"[AnomalyDetector] ✗ Rejected (weapon model classes): {names}")
            return False
        return True

    def _try_load_local_fire_model(self):
        """
        Check if user has manually placed a fire model in model_cache/yolo_threat/.
        This is the recommended path since HuggingFace is blocked.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        local_path = os.path.join(_CACHE_DIR, LOCAL_FIRE_MODEL_NAME)
        if not os.path.exists(local_path):
            return False

        print(f"[AnomalyDetector] Found local fire model: {LOCAL_FIRE_MODEL_NAME}")
        try:
            from ultralytics import YOLO
            m = YOLO(local_path)
            m.overrides["verbose"] = False
            if self._is_valid_fire_model(m):
                self._fire_model  = m
                self._fire_ready  = True
                print(f"[AnomalyDetector] ✓ Fire detector ready (local). "
                      f"Classes: {list(m.names.values())}")
                return True
            else:
                print(f"[AnomalyDetector] ✗ Local fire model rejected "
                      f"(wrong classes). Rename correct model to fire_model.pt")
                return False
        except Exception as e:
            print(f"[AnomalyDetector] ✗ Local fire model load failed: {e}")
            return False

    def _load_models(self):
        _purge_stale_subdirs()

        try:
            from ultralytics import YOLO

            # ── Weapon model ──────────────────────────────────────────────────
            path = self._try_hf_download(WEAPON_MODEL_REPO, WEAPON_MODEL_FILE)
            if path:
                self._weapon_model = YOLO(path)
                self._weapon_model.overrides["verbose"] = False
                names  = list(self._weapon_model.names.values())
                active = [n for n in names if n not in WEAPON_EXCLUDED]
                print(f"[AnomalyDetector] ✓ Weapon detector ready. Active: {active}")
                self._ready = True
            else:
                print("[AnomalyDetector] ✗ Weapon model unavailable.")

            # ── Fire model — check local file FIRST ───────────────────────────
            fire_loaded = self._try_load_local_fire_model()

            # ── Fire model — try HuggingFace candidates (fallback) ────────────
            if not fire_loaded:
                for repo_id, filename in FIRE_MODEL_CANDIDATES:
                    path = self._try_hf_download(repo_id, filename)
                    if path:
                        try:
                            m = YOLO(path)
                            m.overrides["verbose"] = False
                            if self._is_valid_fire_model(m):
                                self._fire_model  = m
                                self._fire_ready  = True
                                fire_loaded       = True
                                print(f"[AnomalyDetector] ✓ Fire detector ready "
                                      f"({repo_id}). Classes: "
                                      f"{list(m.names.values())}")
                                break
                        except Exception as e:
                            print(f"[AnomalyDetector] Load failed ({repo_id}): {e}")

            if not fire_loaded:
                print("[AnomalyDetector] ✗ Fire detection disabled.")
                print("[AnomalyDetector] → To enable: download a YOLOv8 fire model,")
                print(f"[AnomalyDetector] → rename it to '{LOCAL_FIRE_MODEL_NAME}',")
                print(f"[AnomalyDetector] → place it in: {_CACHE_DIR}")

        except ImportError:
            print("[AnomalyDetector] ultralytics not installed.")
        except Exception as e:
            import traceback
            print(f"[AnomalyDetector] Load error: {e}")
            traceback.print_exc()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _analyze_weapon(self, track_id, crop):
        label = "NORMAL"; threat = None; score = 0.0; is_alert = False
        if self._weapon_model:
            try:
                results = self._weapon_model(crop, verbose=False)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls  = results.names[int(box.cls[0])]
                    if cls in WEAPON_EXCLUDED:
                        continue
                    if conf >= WEAPON_CONFIDENCE and conf > score:
                        score = conf; threat = cls; label = "WEAPON"; is_alert = True
            except Exception:
                pass
        with self._lock:
            self._results[track_id] = {
                "label": label, "threat": threat,
                "score": score, "is_alert": is_alert,
                "timestamp": time.time(),
            }
        if is_alert:
            print(f"[AnomalyDetector] WEAPON track={track_id} {threat} ({score:.0%})")

    def _analyze_fire(self, frame):
        label = "NORMAL"; threat = None; score = 0.0; is_alert = False
        if self._fire_model:
            try:
                results = self._fire_model(frame, verbose=False)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls  = results.names[int(box.cls[0])]
                    if cls not in FIRE_VALID_CLASSES:
                        continue
                    if conf >= FIRE_CONFIDENCE and conf > score:
                        score = conf; threat = cls.capitalize()
                        label = "FIRE"; is_alert = True
            except Exception as e:
                print(f"[AnomalyDetector] Fire inference error: {e}")
        with self._lock:
            self._fire_result = {
                "label": label, "threat": threat,
                "score": score, "is_alert": is_alert,
                "timestamp": time.time(),
            }
        if is_alert:
            print(f"[AnomalyDetector] FIRE: {threat} ({score:.0%})")

    def clear_result(self, track_id):
        """Clear stale weapon result for a track (call when track disappears)."""
        with self._lock:
            self._results.pop(track_id, None)

    def _loop(self):
        self._load_models()
        while not self._stopped:
            processed = False
            try:
                track_id, crop = self._weapon_q.get_nowait()
                self._analyze_weapon(track_id, crop)
                processed = True
            except queue.Empty:
                pass
            try:
                frame = self._fire_q.get_nowait()
                self._analyze_fire(frame)
                processed = True
            except queue.Empty:
                pass
            if not processed:
                time.sleep(0.01)