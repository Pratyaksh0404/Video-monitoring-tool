import os
import shutil
import threading
import queue
import time
import cv2
import numpy as np

WEAPON_MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
WEAPON_MODEL_FILE = "weights/best.pt"
WEAPON_CONFIDENCE = 0.75
WEAPON_EXCLUDED   = {"grenade", "Grenade", "bomb", "Bomb",
                     "explosion", "Explosion"}

FIRE_CONFIDENCE       = 0.55   # min confidence for fire/flames class
SMOKE_CONFIDENCE      = 0.85   # very high threshold for smoke — most false positives
DISABLE_SMOKE_CLASS   = True   # set True to ignore smoke class entirely (indoor use)
FIRE_VALID_CLASSES = {"fire", "Fire", "smoke", "Smoke",
                      "flames", "Flames", "wildfire", "Wildfire"}
FIRE_REJECT_CLASSES = {"gun", "Gun", "knife", "Knife",
                       "grenade", "Grenade", "explosion", "Explosion"}

# ── Fire color pre-filter thresholds ──────────────────────────────────────
FIRE_COLOR_MIN_FRACTION = 0.08

LOCAL_FIRE_MODEL_NAME = "fire_model.pt"

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

# ═══════════════════════════════════════════════════════════════════════════
# Shared model pool — multi-camera fix
#
# Previously every AnomalyDetector instance (one per camera, via
# camera_manager.py) loaded its OWN copy of the weapon YOLO model and the
# fire YOLO model from disk. With 2+ cameras starting simultaneously, this
# meant 2x full model loads competing for the same CPU cores at once —
# which is what caused the freeze with 2 cameras on a single-CPU laptop.
#
# Fix: the actual loaded model objects (self._weapon_model / self._fire_model)
# are now MODULE-LEVEL singletons, loaded once regardless of camera count.
# Every camera's AnomalyDetector still has its own independent queues,
# results dict, and background thread — only the expensive model loading
# is shared. Ultralytics YOLO models are safe for concurrent read-only
# inference calls from multiple threads (no shared mutable state is
# written during a forward pass), so this is safe for the CPU inference
# this project uses.
# ═══════════════════════════════════════════════════════════════════════════
_shared_lock          = threading.Lock()
_shared_weapon_model  = None
_shared_weapon_ready  = False
_shared_fire_model    = None
_shared_fire_ready    = False
_shared_load_started  = False
_shared_load_complete = False   # set True only after BOTH weapon+fire attempts finish


def _purge_stale_subdirs():
    for d in ("weights", "runs", "_tmp_download"):
        p = os.path.join(_CACHE_DIR, d)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def _has_fire_colors(frame, box_xyxy) -> bool:
    """
    Check if the region inside box_xyxy contains enough fire-colored pixels.
    Returns True if the region looks like actual fire/flames (red-orange-yellow).
    Returns False for gray/white regions (lighting, blur, glare).
    """
    try:
        h_img, w_img = frame.shape[:2]
        x1 = max(0, int(box_xyxy[0]))
        y1 = max(0, int(box_xyxy[1]))
        x2 = min(w_img, int(box_xyxy[2]))
        y2 = min(h_img, int(box_xyxy[3]))
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return True

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        lower_fire1 = np.array([0,  100,  80], dtype=np.uint8)
        upper_fire1 = np.array([35, 255, 255], dtype=np.uint8)
        lower_fire2 = np.array([170, 100,  80], dtype=np.uint8)
        upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = region.shape[0] * region.shape[1]

        if total_pixels == 0:
            return True

        fraction = fire_pixels / total_pixels
        return fraction >= FIRE_COLOR_MIN_FRACTION

    except Exception:
        return True


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

    # ── Public API (unchanged — no call sites need to change) ──────────────

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

    # ── Shared model loading ─────────────────────────────────────────────────

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
        names = list(model.names.values())
        names_lower = {n.lower() for n in names}
        is_weapon = names_lower & {c.lower() for c in FIRE_REJECT_CLASSES}
        is_fire   = names_lower & {c.lower() for c in FIRE_VALID_CLASSES}
        if is_weapon and not is_fire:
            print(f"[AnomalyDetector] ✗ Rejected (weapon model classes): {names}")
            return False
        return True

    def _try_load_local_fire_model_shared(self):
        """Loads into the module-level shared slot, not self."""
        global _shared_fire_model, _shared_fire_ready
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
                _shared_fire_model = m
                _shared_fire_ready = True
                print(f"[AnomalyDetector] ✓ Fire detector ready (local, shared). "
                      f"Classes: {list(m.names.values())}")
                return True
            else:
                print(f"[AnomalyDetector] ✗ Local fire model rejected.")
                return False
        except Exception as e:
            print(f"[AnomalyDetector] ✗ Local fire model load failed: {e}")
            return False

    def _load_models(self):
        """
        Loads weapon + fire models ONCE into the module-level shared slots,
        no matter how many AnomalyDetector instances (i.e. how many cameras)
        call this. The first camera to reach here does the real work; every
        camera after that just points self._weapon_model/self._fire_model
        at the already-loaded shared objects — instant, no duplicate load.
        """
        global _shared_weapon_model, _shared_weapon_ready
        global _shared_fire_model, _shared_fire_ready, _shared_load_started
        global _shared_load_complete

        with _shared_lock:
            already_loaded = _shared_load_started
            _shared_load_started = True

        if already_loaded:
            # Another camera already loaded (or is loading) the shared
            # models — wait for that FULL load attempt to finish (both
            # weapon AND fire, not just whichever succeeds first — a
            # previous version of this wait broke as soon as EITHER was
            # ready, which meant a slower fire-model load could be missed
            # entirely by the second camera). Then reuse whatever the
            # first camera ended up with.
            for _ in range(600):   # up to 60s wait for the first load to finish
                with _shared_lock:
                    if _shared_load_complete:
                        break
                time.sleep(0.1)

            with _shared_lock:
                self._weapon_model = _shared_weapon_model
                self._ready        = _shared_weapon_ready
                self._fire_model   = _shared_fire_model
                self._fire_ready   = _shared_fire_ready

            if self._ready:
                print("[AnomalyDetector] ✓ Weapon detector ready (shared, reused).")
            if self._fire_ready:
                print("[AnomalyDetector] ✓ Fire detector ready (shared, reused).")
            return

        # ── This camera is first — do the real loading ──────────────────────
        try:
            _purge_stale_subdirs()
            try:
                from ultralytics import YOLO

                # ── Weapon model ──────────────────────────────────────────
                path = self._try_hf_download(WEAPON_MODEL_REPO, WEAPON_MODEL_FILE)
                if path:
                    model = YOLO(path)
                    model.overrides["verbose"] = False
                    names  = list(model.names.values())
                    active = [n for n in names if n not in WEAPON_EXCLUDED]
                    print(f"[AnomalyDetector] ✓ Weapon detector ready. Active: {active}")
                    with _shared_lock:
                        _shared_weapon_model = model
                        _shared_weapon_ready = True
                    self._weapon_model = model
                    self._ready = True
                else:
                    print("[AnomalyDetector] ✗ Weapon model unavailable.")

                # ── Fire model ──────────────────────────────────────────────
                fire_loaded = self._try_load_local_fire_model_shared()
                if not fire_loaded:
                    for repo_id, filename in FIRE_MODEL_CANDIDATES:
                        path = self._try_hf_download(repo_id, filename)
                        if path:
                            try:
                                m = YOLO(path)
                                m.overrides["verbose"] = False
                                if self._is_valid_fire_model(m):
                                    with _shared_lock:
                                        _shared_fire_model = m
                                        _shared_fire_ready = True
                                    fire_loaded = True
                                    print(f"[AnomalyDetector] ✓ Fire detector ready "
                                          f"({repo_id}).")
                                    break
                            except Exception as e:
                                print(f"[AnomalyDetector] Load failed ({repo_id}): {e}")
                if not fire_loaded:
                    print("[AnomalyDetector] ✗ Fire detection disabled.")
                    print(f"[AnomalyDetector] → Place fire_model.pt in: {_CACHE_DIR}")
                else:
                    with _shared_lock:
                        self._fire_model = _shared_fire_model
                        self._fire_ready = _shared_fire_ready

            except ImportError:
                print("[AnomalyDetector] ultralytics not installed.")
            except Exception as e:
                import traceback
                print(f"[AnomalyDetector] Load error: {e}")
                traceback.print_exc()
        finally:
            # ALWAYS set this, even if loading failed/errored — otherwise
            # every other camera waiting in the "already_loaded" branch
            # above would hang for the full 60s timeout on every startup.
            with _shared_lock:
                _shared_load_complete = True

    # ── Inference (unchanged) ────────────────────────────────────────────────

    def _analyze_weapon(self, track_id, crop):
        label = "NORMAL"; threat = None; score = 0.0; is_alert = False
        boxes = []
        if self._weapon_model:
            try:
                results = self._weapon_model(crop, verbose=False)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls  = results.names[int(box.cls[0])]
                    if cls in WEAPON_EXCLUDED:
                        continue
                    if conf >= WEAPON_CONFIDENCE:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        boxes.append((int(xyxy[0]), int(xyxy[1]),
                                      int(xyxy[2]), int(xyxy[3])))
                        if conf > score:
                            score = conf; threat = cls
                            label = "WEAPON"; is_alert = True
            except Exception:
                pass
        with self._lock:
            self._results[track_id] = {
                "label": label, "threat": threat,
                "score": score, "is_alert": is_alert,
                "boxes": boxes,
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

                    cls_lower = cls.lower()
                    is_smoke_class = cls_lower in ("smoke",)

                    if is_smoke_class and DISABLE_SMOKE_CLASS:
                        continue

                    min_conf = SMOKE_CONFIDENCE if is_smoke_class else FIRE_CONFIDENCE
                    if conf < min_conf:
                        continue

                    if not is_smoke_class:
                        xyxy = box.xyxy[0].cpu().numpy()
                        if not _has_fire_colors(frame, xyxy):
                            continue

                    if conf > score:
                        score = conf
                        threat = "Fire"
                        label = "FIRE"
                        is_alert = True

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
