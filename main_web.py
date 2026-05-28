import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import time
import threading
import queue
import numpy as np
import face_recognition
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.behavior_engine import BehaviorEngine
from analytics.trajectory_tracker import TrajectoryTracker
from analytics.loitering_detector import LoiteringDetector
from analytics.fight_detector import FightDetector
from analytics.unknown_tracker import UnknownTracker
from alerts.alert_manager import AlertManager
from analytics.anomaly_detector import AnomalyDetector
from analytics.camera_tamper import CameraTamper
from video_streamer import streamer
from utils.logger import get_logger
from utils.timer import PipelineTimer

log = get_logger("main_web")

# ── Config loader ─────────────────────────────────────────────────────────────
def _load_config():
    """Load rules_config.yaml if it exists, otherwise return empty dict."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "rules_config.yaml"
    )
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            log.info(f"Loaded config from {config_path}")
            return cfg
        except ImportError:
            log.warning("PyYAML not installed — using defaults. pip install pyyaml")
        except Exception as e:
            log.error(f"Config load error: {e} — using defaults")
    return {}

_CFG = _load_config()

def _cfg(section, key, default):
    """Get a config value with fallback to default."""
    return _CFG.get(section, {}).get(key, default)

# ── Thresholds (from config or defaults) ──────────────────────────────────────
MIN_PERSON_HEIGHT    = _cfg("person", "min_height", 60)
MIN_PERSON_WIDTH     = _cfg("person", "min_width", 30)
FACE_TOLERANCE       = _cfg("person", "face_tolerance", 0.5)  # raised 0.4→0.5 for appearance variation
FACE_SCALE           = 0.5
CROWD_THRESHOLD      = _cfg("crowd", "threshold", 4)
FILE_PRESENCE_WARMUP = 30
PRESENCE_HOLD        = 5
FIRE_FRAME_INTERVAL  = _cfg("fire", "frame_interval", 20)
FIRE_RESULT_MAX_AGE  = _cfg("fire", "result_max_age", 8.0)
FIRE_MIN_CONFIDENCE  = _cfg("fire", "min_confidence", 0.70)     # per-reading threshold; rolling window of 3 hits in 6s is the real guard
FIRE_SUSTAIN_SECS    = _cfg("fire", "sustain_seconds", 6.0)     # raised from 4.0
FIRE_MIN_CONSECUTIVE = _cfg("fire", "min_consecutive", 3)       # raised from 2
UNKNOWN_GRACE_SECS   = _cfg("unknown", "grace_seconds", 12)
IDLE_MAX_SPEED       = _cfg("behavior", "idle_max_speed", 12.0)
IDLE_USE_CLIP        = _cfg("behavior", "idle_use_clip", False)  # NEW: disable CLIP for idle
PATROL_DISPLAY_ZONES = _cfg("patrol", "display_zones", 3)
MISSING_COOLDOWN     = _cfg("alerts", "missing_cooldown", 30)
IDLE_COOLDOWN        = _cfg("alerts", "idle_cooldown", 60)       # NEW: longer gap between idle alerts
IDLE_STARTUP_GRACE   = _cfg("alerts", "idle_startup_grace", 90)  # NEW: suppress idle for first 90s after pipeline start
SIDEWAYS_RADIUS      = _cfg("unknown", "sideways_radius", 200)   # raised from 150px
SINGLE_PERSON_GRACE  = _cfg("unknown", "single_person_grace", 15.0)  # raised from 8s — suppress unknown if 1 person + guard recently seen
POST_CROWD_GRACE     = _cfg("crowd", "post_crowd_grace", 10.0)
POST_CROWD_LOITER    = _cfg("crowd", "post_crowd_loiter", 15.0)
DRAW_KNIFE_BBOX      = _cfg("weapon", "draw_knife_bbox", True)  # NEW: knife overlay
TRACKER_MAX_DISAP    = _cfg("tracker", "max_disappeared", 40)   # raised from 30
TRACKER_MAX_DIST     = _cfg("tracker", "max_distance", 300)

_stop_event     = threading.Event()
_thread_lock    = threading.Lock()
_current_thread = None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    return inter / float(
        (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) +
        (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter)


def box_centroid(box):
    return (int((box[0]+box[2])/2), int((box[1]+box[3])/2))


def load_enrolled_faces(faces_dir):
    known_encodings, known_names = [], []
    if not os.path.exists(faces_dir):
        log.warning(f"enrolled_faces dir not found: {faces_dir}")
        return known_encodings, known_names
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            try:
                img  = face_recognition.load_image_file(
                    os.path.join(person_dir, img_file))
                encs = face_recognition.face_encodings(img)
                for enc in encs:
                    known_encodings.append(enc)
                    known_names.append(person_name)
            except Exception:
                pass
    log.info(f"Face recognizer: {len(known_encodings)} encodings for "
          f"{len(set(known_names))} identities: {sorted(set(known_names))}")
    return known_encodings, known_names


def recognize_faces_in_frame(frame, known_encodings, known_names):
    small = cv2.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs  = face_recognition.face_locations(rgb)
    encs  = face_recognition.face_encodings(rgb, locs)
    results = []
    scale = 1.0 / FACE_SCALE
    for (top, right, bottom, left), enc in zip(locs, encs):
        name = "UNKNOWN"
        if known_encodings:
            dists    = face_recognition.face_distance(known_encodings, enc)
            best_idx = np.argmin(dists)
            if dists[best_idx] < FACE_TOLERANCE:
                matches = face_recognition.compare_faces(
                    known_encodings, enc, tolerance=FACE_TOLERANCE)
                if matches[best_idx]:
                    name = known_names[best_idx]
        results.append((int(top*scale), int(right*scale),
                         int(bottom*scale), int(left*scale), name))
    return results


class IdentityTrajectory:
    STABILITY  = 2.5
    IDLE_RESET = 12.0

    def __init__(self):
        self.paths     = defaultdict(list)
        self.last_zone = {}
        self.pending   = {}
        self.last_seen = {}

    def update(self, name, zone):
        if zone is None:
            return
        now = time.time()
        if name in self.last_seen and now - self.last_seen[name] > self.IDLE_RESET:
            self.pending.pop(name, None)
        self.last_seen[name] = now
        prev = self.last_zone.get(name)
        if prev is None:
            self.last_zone[name] = zone; self.paths[name].append(zone); return
        if zone == prev:
            self.pending.pop(name, None); return
        cand, since = self.pending.get(name, (None, 0))
        if cand != zone:
            self.pending[name] = (zone, now); return
        if now - since >= self.STABILITY:
            self.last_zone[name] = zone
            self.pending.pop(name, None)
            if not self.paths[name] or self.paths[name][-1] != zone:
                self.paths[name].append(zone)
                if len(self.paths[name]) > 20:
                    self.paths[name] = self.paths[name][-20:]

    def is_active(self, name):
        last = self.last_seen.get(name)
        return last is not None and (time.time() - last) < self.IDLE_RESET

    def get_path(self, name):
        return " -> ".join(self.paths.get(name, []))

    def get_path_short(self, name, n=3):
        zones = self.paths.get(name, [])
        return " -> ".join(zones[-n:]) if zones else ""

    def get_zone(self, name):
        return self.last_zone.get(name, None)

    def last_added_zone(self, name):
        zones = self.paths.get(name, [])
        return zones[-1] if zones else None


class BehaviorWorker:
    def __init__(self, cache, cache_lock):
        self._q          = queue.Queue(maxsize=4)
        self._cache      = cache
        self._lock       = cache_lock
        self._stopped    = False
        self._classifier = None
        self._ready      = False
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def ready(self):
        return self._ready

    def submit(self, track_id, frame, box):
        if not self._ready:
            return
        try:
            self._q.put_nowait((track_id, frame, box))
        except queue.Full:
            pass

    def _loop(self):
        try:
            from analytics.behavior_classifier import BehaviorClassifier
            self._classifier = BehaviorClassifier(device="cpu")
            self._ready = True
            log.info("CLIP behavior model ready.")
        except Exception as e:
            print(f"[BehaviorWorker] Failed: {e}")
            return
        while not self._stopped:
            try:
                track_id, frame, box = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                label, _ = self._classifier.predict(frame, box)
            except Exception:
                label = "ANALYZING"
            with self._lock:
                self._cache[track_id] = label

    def stop(self):
        self._stopped = True


def run(source=0, source_label="Camera 0", beh_worker=None,
        beh_label_cache=None, beh_cache_lock=None, anomaly_det=None,
        stop_event=None, cam_streamer=None):
    """
    Main pipeline loop.

    stop_event:   threading.Event — if provided, used instead of global _stop_event
    cam_streamer: CameraStreamer   — if provided, used instead of global streamer
                                    (for multi-camera mode)
    """
    global _stop_event

    # Use provided stop_event or global
    effective_stop = stop_event if stop_event is not None else _stop_event
    effective_stop.clear()

    # Use provided streamer or global
    effective_streamer = cam_streamer if cam_streamer is not None else streamer

    is_file = isinstance(source, str)
    effective_streamer.set_source("file" if is_file else "webcam", source_label)

    faces_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "data", "enrolled_faces"
    )
    known_encodings, known_names = load_enrolled_faces(faces_dir)
    enrolled_names = set(known_names)

    try:
        stream = VideoStreamReader(source=source)
    except RuntimeError as e:
        log.error(f"Camera error: {e}")
        time.sleep(2)
        return

    detector        = PersonDetector(conf_threshold=0.5)
    # max_disappeared from config (default 40, raised from 30)
    tracker         = CentroidTracker(max_disappeared=TRACKER_MAX_DISAP,
                                      max_distance=TRACKER_MAX_DIST)
    presence_mon    = PresenceMonitor(absence_threshold=10, confirm_time=3)
    inactivity_mon  = InactivityMonitor(inactivity_threshold=45,
                                        position_threshold=80, window_time=10)
    beh_engine      = BehaviorEngine()
    alert_manager   = AlertManager()
    loitering_det   = LoiteringDetector()
    fight_det       = FightDetector()
    unknown_tracker = UnknownTracker()
    camera_tamper   = CameraTamper()
    timer           = PipelineTimer()

    log.info("Waiting for first frame...")
    for _ in range(50):
        if stream.read_frame() is not None:
            break
        time.sleep(0.1)
    else:
        log.error("Camera not responding — aborting.")
        stream.release()
        return

    log.info("Camera ready.")
    frame_count      = 0
    behavior_cache   = {}
    identity_memory  = {}
    track_birth      = {}
    track_last_pos   = {}
    slot_birth       = {}

    presence_buffer  = FILE_PRESENCE_WARMUP if is_file else 0
    warmup_frames    = 10 if is_file else 120
    last_alert_time  = {}
    ALERT_COOLDOWN   = _cfg("alerts", "cooldown", 30)
    last_beh_frame   = {}
    BEH_INTERVAL     = _cfg("behavior", "interval_frames", 50)

    post_area     = None
    zone_detector = None
    id_traj       = IdentityTrajectory()
    last_printed  = {}
    face_results  = []

    last_seen_guard_name     = "Post"
    last_patrol_alerted_zone = {}

    # Crowd state tracking for post-crowd grace periods
    crowd_ended_at   = 0.0
    was_crowd        = False

    # Fire sustained detection — rolling window of confident detection timestamps
    fire_hits = []   # timestamps of readings that hit FIRE_MIN_CONFIDENCE

    # ── Dwell time + compliance tracking ─────────────────────────────────────
    identity_first_seen = {}   # name → first time seen this session (epoch)

    compliance = {}            # name → compliance metrics dict
    def get_compliance(name):
        if name not in compliance:
            compliance[name] = {
                "patrol_zones":   set(),
                "idle_frames":    0,
                "phone_frames":   0,
                "sleeping_frames":0,
                "present_frames": 0,
            }
        return compliance[name]

    # Pipeline start time — for startup grace periods
    pipeline_start_time = time.time()

    try:
        while not effective_stop.is_set():
            frame_count += 1
            frame = stream.read_frame()
            if frame is None:
                break

            if frame_count < warmup_frames:
                effective_streamer.push_frame(frame)
                continue

            if post_area is None:
                h, w = frame.shape[:2]
                post_area     = (0, 0, w, h)
                zone_detector = TrajectoryTracker(roi=post_area, grid_size=2)
                log.info(f"Frame: {w}x{h}, POST_AREA={post_area}")

            # ── Camera tamper check ───────────────────────────────────────────
            tamper_event = camera_tamper.update(frame)
            if tamper_event:
                alert_manager.send_alert(
                    f"Camera Tamper: {tamper_event}", "Camera", zone="—")
                active_violations.add("Tamper")

            # ── Detection ────────────────────────────────────────────────────
            timer.start("person_detection")
            detections   = detector.detect(frame)
            person_boxes = []
            for (x1, y1, x2, y2, conf) in detections:
                if (x2-x1) < MIN_PERSON_WIDTH or (y2-y1) < MIN_PERSON_HEIGHT:
                    continue
                person_boxes.append((x1, y1, x2, y2))
            timer.stop("person_detection")

            valid_centroids = [box_centroid(b) for b in person_boxes]

            presence_buffer = PRESENCE_HOLD if valid_centroids else max(0, presence_buffer-1)
            post_status     = presence_mon.update(
                valid_centroids if presence_buffer > 0 else [])
            activity_status = (inactivity_mon.update(valid_centroids)
                               if post_status == "PRESENT" else "NO_PERSON")

            tracked_objects = tracker.update(person_boxes)
            zone_detector.update(tracked_objects)

            # ── Dead track cleanup ────────────────────────────────────────────
            for dead_id in tracker.recently_deregistered:
                zone_detector.reset_track(dead_id)
                behavior_cache.pop(dead_id, None)
                beh_label_cache.pop(dead_id, None)
                last_beh_frame.pop(dead_id, None)
                track_birth.pop(dead_id, None)
                loitering_det.reset(dead_id)
                fight_det.reset(dead_id)
                beh_engine.reset(dead_id)
                last_pos = track_last_pos.pop(dead_id, None)
                unknown_tracker.on_track_lost(dead_id, last_pos)
                if anomaly_det:
                    anomaly_det.clear_result(dead_id)

            # ── Face recognition every 8 frames ──────────────────────────────
            if frame_count % 8 == 0:
                timer.start("face_recognition")
                face_results = recognize_faces_in_frame(
                    frame, known_encodings, known_names)
                timer.stop("face_recognition")

            # ─────────────────────────────────────────────────────────────────
            # FRAME-LEVEL DETECTIONS
            # ─────────────────────────────────────────────────────────────────
            active_guard_ids  = set()
            active_violations = set()
            now               = time.time()
            is_crowd          = len(person_boxes) >= CROWD_THRESHOLD

            # Track crowd state transitions for post-crowd grace
            if was_crowd and not is_crowd:
                crowd_ended_at = now
            was_crowd = is_crowd

            in_post_crowd_grace  = (now - crowd_ended_at) < POST_CROWD_GRACE
            in_post_crowd_loiter = (now - crowd_ended_at) < POST_CROWD_LOITER

            # ── 1. Crowd ──────────────────────────────────────────────────────
            if is_crowd:
                if now - last_alert_time.get("crowd", 0) > ALERT_COOLDOWN:
                    alert_manager.send_alert("Crowd Detected", "Camera", zone="—")
                    last_alert_time["crowd"] = now
                    active_violations.add("Crowd")

            # ── 2. Fire detection ─────────────────────────────────────────────
            # Strategy: rolling time window of fire model readings.
            # A reading is "positive" if score >= FIRE_MIN_CONFIDENCE (0.70).
            # If 3+ positive readings in the last FIRE_SUSTAIN_SECS (6s) → alert.
            # Much more robust than consecutive counter — handles gaps between
            # fire model runs (every 20 frames) without false resets.
            if anomaly_det and anomaly_det._fire_ready:
                if frame_count % FIRE_FRAME_INTERVAL == 0:
                    anomaly_det.submit_frame(frame)

                fire_res = anomaly_det.get_fire_result()
                if (fire_res
                        and fire_res.get("is_alert")
                        and (now - fire_res.get("timestamp", 0)) < FIRE_RESULT_MAX_AGE):

                    score = fire_res.get("score", 0)
                    if score >= FIRE_MIN_CONFIDENCE:
                        fire_hits.append(now)

                    # Purge hits older than FIRE_SUSTAIN_SECS
                    fire_hits = [t for t in fire_hits
                                 if now - t < FIRE_SUSTAIN_SECS]

                    # Alert if enough positive readings in window
                    if len(fire_hits) >= FIRE_MIN_CONSECUTIVE:
                        if now - last_alert_time.get("fire", 0) > ALERT_COOLDOWN:
                            t = fire_res.get("threat") or "Fire"
                            alert_manager.send_alert(
                                f"Fire Detected: {t}", "Camera", zone="—")
                            last_alert_time["fire"] = now
                            active_violations.add("Fire")

                    cv2.putText(frame,
                                f"FIRE DETECTED: {fire_res.get('threat','')} "
                                f"({score:.0%})",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 255), 2)
                else:
                    # Decay stale hits
                    fire_hits = [t for t in fire_hits if now - t < FIRE_SUSTAIN_SECS]

            # ── 3. Full-frame weapon scan ─────────────────────────────────────
            if anomaly_det and anomaly_det.is_ready():
                if frame_count % 20 == 0:
                    anomaly_det.submit(track_id=-1, frame_crop=frame.copy())

                full_res = anomaly_det.get_result(-1)
                if (full_res
                        and full_res.get("is_alert")
                        and (now - full_res.get("timestamp", 0)) < 5.0):
                    threat = full_res.get("threat") or "Weapon"

                    # NEW: Draw knife bounding box on frame
                    if DRAW_KNIFE_BBOX and full_res.get("boxes"):
                        for kbox in full_res["boxes"]:
                            kx1, ky1, kx2, ky2 = kbox
                            cv2.rectangle(frame, (kx1, ky1), (kx2, ky2),
                                          (0, 0, 255), 2)
                            cv2.putText(frame,
                                        f"{threat} ({full_res.get('score',0):.0%})",
                                        (kx1, max(ky1-8, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                        (0, 0, 255), 2)

                    if not tracked_objects:
                        # No persons — unattended weapon
                        key = f"weapon:frame:{threat}"
                        if now - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                            alert_manager.send_alert(
                                f"Unattended Weapon Detected: {threat}",
                                "Camera", zone="—")
                            last_alert_time[key] = now
                            active_violations.add("Weapon")
                    else:
                        per_person_caught = any(
                            anomaly_det.get_result(tid) and
                            anomaly_det.get_result(tid).get("is_alert")
                            for tid in tracked_objects
                        )
                        if not per_person_caught:
                            gid = last_seen_guard_name if last_seen_guard_name != "Post" else "Camera"
                            key = f"weapon:fullframe:{threat}"
                            if now - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                                alert_manager.send_alert(
                                    f"Weapon Detected: {threat}", gid, zone="—")
                                last_alert_time[key] = now
                                active_violations.add("Weapon")

            # ── 4. Fight (suppressed in crowd) ────────────────────────────────
            fight_pairs = fight_det.update(tracked_objects,
                                           suppress=(is_crowd or in_post_crowd_grace))
            for (id1, id2) in fight_pairs:
                name1    = identity_memory.get(id1,
                           unknown_tracker.get_slot(id1) or f"Guard_{id1}")
                name2    = identity_memory.get(id2,
                           unknown_tracker.get_slot(id2) or f"Guard_{id2}")
                pair_key = f"fight:{min(id1,id2)}:{max(id1,id2)}"

                if now - last_alert_time.get(pair_key, 0) > ALERT_COOLDOWN:
                    known_in_pair = [n for t, n in identity_memory.items()
                                     if t in (id1, id2)]
                    if known_in_pair:
                        msg = "Guard Under Attack"; gid = known_in_pair[0]
                    else:
                        msg = "Fight / Violence Detected"
                        gid = f"{name1} & {name2}"

                    zone1 = zone_detector.get_current_zone(id1) or "—"
                    alert_manager.send_alert(msg, gid, zone=zone1)
                    last_alert_time[pair_key] = now
                    active_violations.add("Fight")

                if id1 in tracked_objects and id2 in tracked_objects:
                    c1 = box_centroid(tracked_objects[id1])
                    c2 = box_centroid(tracked_objects[id2])
                    cv2.line(frame, c1, c2, (0, 0, 255), 2)
                    mid = ((c1[0]+c2[0])//2-30, (c1[1]+c2[1])//2-12)
                    cv2.putText(frame, "FIGHT", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

            # ─────────────────────────────────────────────────────────────────
            # PER-PERSON LOOP
            # ─────────────────────────────────────────────────────────────────
            for track_id, person_box in tracked_objects.items():
                if track_id not in track_birth:
                    track_birth[track_id] = time.time()

                centroid = box_centroid(person_box)
                track_last_pos[track_id] = centroid

                # ── Identity ──────────────────────────────────────────────────
                # Once a track is identified via face recognition, it keeps that
                # identity FOREVER (even if face is no longer visible due to
                # turning sideways, looking down, etc.)
                if track_id not in identity_memory:
                    # Only try face matching for tracks that haven't been identified yet
                    best_score, recognized_name = 0.0, "UNKNOWN"
                    for (ft, fr, fb, fl, fname) in face_results:
                        score = iou(person_box, (fl, ft, fr, fb))
                        if score > best_score:
                            best_score, recognized_name = score, fname

                    if recognized_name != "UNKNOWN" and best_score > 0.1:
                        identity_memory[track_id] = recognized_name
                        last_seen_guard_name = recognized_name
                        if unknown_tracker.get_slot(track_id):
                            unknown_tracker.on_track_lost(track_id, centroid)
                else:
                    # Track already identified — just update the last seen guard name
                    last_seen_guard_name = identity_memory[track_id]
                    # Still try to match faces (in case a DIFFERENT face appears
                    # overlapping this track — e.g. two people very close)
                    for (ft, fr, fb, fl, fname) in face_results:
                        if fname != "UNKNOWN":
                            score = iou(person_box, (fl, ft, fr, fb))
                            if score > 0.3:
                                identity_memory[track_id] = fname
                                last_seen_guard_name = fname

                guard_id = identity_memory.get(track_id, "UNKNOWN")

                if guard_id == "UNKNOWN":
                    alert_name = unknown_tracker.get_or_assign(track_id, centroid)
                    if alert_name not in slot_birth:
                        slot_birth[alert_name] = time.time()
                else:
                    alert_name = guard_id
                    # Track when we last saw a known guard (for single-person grace)
                    last_alert_time["last_guard_seen_at"] = time.time()

                active_guard_ids.add(alert_name)

                # ── Dwell time tracking ───────────────────────────────────────
                if alert_name not in identity_first_seen:
                    identity_first_seen[alert_name] = now
                dwell_secs = now - identity_first_seen[alert_name]

                # ── Compliance frame counting ─────────────────────────────────
                if guard_id != "UNKNOWN":
                    c = get_compliance(guard_id)
                    c["present_frames"] += 1

                # ── Behavior ──────────────────────────────────────────────────
                # Don't run behavior for unknowns during crowd — too many false hits
                run_behavior = (post_status == "PRESENT" and
                                not (guard_id == "UNKNOWN" and is_crowd))
                if run_behavior:
                    last_f = last_beh_frame.get(track_id, 0)
                    if frame_count - last_f >= BEH_INTERVAL:
                        beh_worker.submit(track_id, frame, person_box)
                        last_beh_frame[track_id] = frame_count

                with beh_cache_lock:
                    raw_label = beh_label_cache.get(track_id, "ANALYZING")

                label       = raw_label
                final_state = (beh_engine.update(track_id, label)
                               if post_status == "PRESENT" else "ANALYZING")

                # ── Weapon (per-person crop) ───────────────────────────────────
                anomaly_label  = "NORMAL"
                anomaly_threat = None
                if anomaly_det and anomaly_det.is_ready():
                    if frame_count % 12 == track_id % 12:
                        x1a, y1a, x2a, y2a = person_box
                        crop = frame[max(0,y1a):y2a, max(0,x1a):x2a]
                        anomaly_det.submit(track_id, crop)
                    res = anomaly_det.get_result(track_id)
                    if res:
                        anomaly_label  = res["label"]
                        anomaly_threat = res.get("threat")

                # ── Trajectory ────────────────────────────────────────────────
                current_zone = zone_detector.get_current_zone(track_id)

                if guard_id != "UNKNOWN":
                    id_traj.update(guard_id, current_zone)
                    display_zone  = id_traj.get_zone(guard_id) or current_zone or "?"
                    display_path  = id_traj.get_path(guard_id)
                    latest_zone   = id_traj.last_added_zone(guard_id)
                    if (id_traj.is_active(guard_id)
                            and " -> " in display_path
                            and latest_zone is not None
                            and last_patrol_alerted_zone.get(guard_id) != latest_zone):
                        short_path = id_traj.get_path_short(guard_id, PATROL_DISPLAY_ZONES)
                        log.info(f"PATROL {guard_id} — {short_path}")
                        last_patrol_alerted_zone[guard_id] = latest_zone
                        # Track patrol zone for compliance
                        get_compliance(guard_id)["patrol_zones"].add(latest_zone)
                        alert_manager.send_alert(
                            f"Patrol: {short_path}", guard_id, zone=display_zone)
                else:
                    slot = alert_name
                    id_traj.update(slot, current_zone)
                    display_zone = id_traj.get_zone(slot) or current_zone or "?"
                    display_path = id_traj.get_path(slot)

                # ── Loitering ─────────────────────────────────────────────────
                person_speed = fight_det.speed(track_id)
                is_known     = guard_id != "UNKNOWN"
                is_loitering = loitering_det.update(
                    track_id, current_zone,
                    is_known=is_known, current_speed=person_speed)

                # ── Draw overlays ─────────────────────────────────────────────
                x1, y1, x2, y2 = person_box
                color = (0, 255, 0)
                if anomaly_label == "WEAPON":               color = (0, 0, 255)
                elif "CONFIRMED_SLEEPING" in final_state:   color = (0, 0, 255)
                elif "CONFIRMED" in final_state:            color = (0, 0, 255)
                elif "POSSIBLE" in final_state:             color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{alert_name} | {final_state}"
                if anomaly_label != "NORMAL":
                    label_text += f" | {anomaly_threat or anomaly_label}"

                cv2.putText(frame, label_text,
                            (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if display_zone and display_zone != "?":
                    dwell_secs_now = round(now - identity_first_seen.get(alert_name, now))
                    dwell_str = f"{dwell_secs_now//60}m{dwell_secs_now%60:02d}s" if dwell_secs_now >= 60 else f"{dwell_secs_now}s"
                    cv2.putText(frame, f"Zone:{display_zone}  {dwell_str}",
                                (x1, y1+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

                for (ft, fr, fb, fl, fname) in face_results:
                    fc = (0,255,0) if fname != "UNKNOWN" else (0,0,255)
                    cv2.rectangle(frame, (fl,ft), (fr,fb), fc, 1)

                # ── Per-person alerts ─────────────────────────────────────────
                now = time.time()

                def send_alert(key, msg,
                               _gid=alert_name, _zone=display_zone, _t=now):
                    if _t - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                        alert_manager.send_alert(msg, _gid, zone=_zone)
                        last_alert_time[key] = _t
                        active_violations.add(msg)

                is_moving = person_speed > IDLE_MAX_SPEED

                if guard_id != "UNKNOWN":
                    c = get_compliance(guard_id)

                    # SLEEPING: requires person to be stationary (not just looking down)
                    if "CONFIRMED_SLEEPING" in final_state and not is_moving:
                        send_alert(f"sleep:{guard_id}", "Guard Sleeping")
                        c["sleeping_frames"] += 1
                    elif "CONFIRMED_PHONE_USE" in final_state:
                        send_alert(f"phone:{guard_id}", "Phone Usage")
                        c["phone_frames"] += 1
                    elif "CONFIRMED_SMOKING" in final_state and not is_moving:
                        # SMOKING: back to CLIP-based detection.
                        # SmokingDetector (skin/hand detection) had too many false
                        # positives from eating, resting chin on hand, etc.
                        # CLIP with not_is_moving gate + 85% confirm ratio works better.
                        # Future: replace with cigarette object detection model.
                        send_alert(f"smoking:{guard_id}", "Guard Smoking")

                    # IDLE
                    if (not is_moving and activity_status == "INACTIVE"
                          and (now - pipeline_start_time) > IDLE_STARTUP_GRACE):
                        c["idle_frames"] += 1
                        idle_key = f"idle:{guard_id}"
                        if now - last_alert_time.get(idle_key, 0) > IDLE_COOLDOWN:
                            alert_manager.send_alert("Guard Idle", alert_name, zone=display_zone)
                            last_alert_time[idle_key] = now
                            active_violations.add("Guard Idle")

                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:{guard_id}:{anomaly_threat}",
                                   f"Weapon Detected: {anomaly_threat}")

                else:
                    # Unknown person alerts — suppressed during crowd + grace
                    sideways_suppressed = False

                    # Check 1: Is a known guard's track centroid nearby?
                    for tid, tbox in tracked_objects.items():
                        if tid == track_id:
                            continue
                        if identity_memory.get(tid) in enrolled_names:
                            tc = box_centroid(tbox)
                            dist = ((centroid[0]-tc[0])**2 + (centroid[1]-tc[1])**2)**0.5
                            if dist < SIDEWAYS_RADIUS:
                                sideways_suppressed = True
                                break

                    # Check 2: Only 1 person + guard was recently seen
                    if (not sideways_suppressed
                            and len(tracked_objects) == 1
                            and last_seen_guard_name != "Post"
                            and (now - last_alert_time.get("last_guard_seen_at", 0)) < SINGLE_PERSON_GRACE):
                        sideways_suppressed = True

                    # Check 3: This track_id recently HAD a known identity
                    # (face was recognized before but now lost — same person turned)
                    if (not sideways_suppressed
                            and track_id in track_birth
                            and (now - track_birth[track_id]) < 5.0):
                        # Track is very new (< 5s) — might be a re-registered track
                        # from a guard who just turned. Suppress if guard was just seen.
                        if (last_seen_guard_name != "Post"
                                and (now - last_alert_time.get("last_guard_seen_at", 0)) < SINGLE_PERSON_GRACE):
                            sideways_suppressed = True

                    if not is_crowd and not in_post_crowd_grace and not sideways_suppressed:
                        slot_age = now - slot_birth.get(alert_name, now)
                        if (slot_age > UNKNOWN_GRACE_SECS
                                and unknown_tracker.can_alert(alert_name)):
                            alert_manager.send_alert(
                                "Unknown Person Detected", alert_name,
                                zone=display_zone)
                            unknown_tracker.mark_alerted(alert_name)

                    if "CONFIRMED_SLEEPING" in final_state:
                        send_alert(f"sleep:{alert_name}",
                                   "Unknown Person Sleeping")
                    elif "CONFIRMED_PHONE_USE" in final_state:
                        send_alert(f"phone:{alert_name}",
                                   "Unknown Person Using Phone")

                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:{alert_name}:{anomaly_threat}",
                                   f"Weapon Detected: {anomaly_threat}")

            # ── Guard Missing ─────────────────────────────────────────────────
            if post_status == "ABSENT":
                now = time.time()
                if now - last_alert_time.get("post_missing", 0) > MISSING_COOLDOWN:
                    alert_manager.send_alert(
                        "Guard Missing", last_seen_guard_name, zone="—")
                    last_alert_time["post_missing"] = now

            # ── Stats ─────────────────────────────────────────────────────────
            # Compute compliance scores (0-100) for each known guard
            compliance_scores = {}
            for name, c in compliance.items():
                if c["present_frames"] < 10:
                    continue
                pf = max(c["present_frames"], 1)
                patrol_pct   = min(len(c["patrol_zones"]) / 4 * 100, 100)
                idle_pct     = min(c["idle_frames"]     / pf * 100, 100)
                phone_pct    = min(c["phone_frames"]    / pf * 100, 100)
                sleeping_pct = min(c["sleeping_frames"] / pf * 100, 100)
                # Score: 40% patrol + 20% not-idle + 20% not-phone + 20% not-sleeping
                score = (patrol_pct * 0.4
                         + (100 - idle_pct)     * 0.2
                         + (100 - phone_pct)    * 0.2
                         + (100 - sleeping_pct) * 0.2)
                compliance_scores[name] = round(score, 1)

            # Dwell times in seconds for each identity currently in frame
            dwell_now = {
                name: round(now - t)
                for name, t in identity_first_seen.items()
                if name in active_guard_ids
            }

            effective_streamer.update_stats(
                guards_detected=len(active_guard_ids),
                active_violations=len(active_violations),
                people_count=len(tracked_objects),
                dwell_times=dwell_now,
                compliance_scores=compliance_scores,
            )

            # ── HUD ───────────────────────────────────────────────────────────
            rx1, ry1, rx2, ry2 = post_area
            mid_x = (rx1+rx2)//2; mid_y = (ry1+ry2)//2
            cv2.line(frame, (mid_x,ry1), (mid_x,ry2), (255,100,0), 1)
            cv2.line(frame, (rx1,mid_y), (rx2,mid_y), (255,100,0), 1)
            for lbl, pos in [("A",(rx1+8,ry1+20)), ("B",(mid_x+8,ry1+20)),
                              ("C",(rx1+8,mid_y+20)), ("D",(mid_x+8,mid_y+20))]:
                cv2.putText(frame, lbl, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 2)

            post_color = ((0,255,0) if post_status=="PRESENT" else
                          (0,255,255) if "TEMP" in post_status else (0,0,255))
            cv2.putText(frame, post_status, (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, post_color, 1)
            cv2.putText(frame, f"FPS:{effective_streamer.fps}", (8,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

            effective_streamer.push_frame(frame)

            # Log performance stats every 300 frames (~55s)
            if frame_count % 300 == 0 and frame_count > 0:
                log.debug(f"Perf stats | {timer.summary()}")

    except Exception as e:
        import traceback
        log.error(f"Pipeline CRASH: {e}", exc_info=True)
    finally:
        stream.release()
        loitering_det.reset_all()
        fight_det.reset()
        unknown_tracker.reset()
        camera_tamper.reset()
        log.info("Pipeline loop ended.")


def _run_with_restart(source, source_label):
    beh_label_cache = {}
    beh_cache_lock  = threading.Lock()
    worker          = BehaviorWorker(beh_label_cache, beh_cache_lock)
    anomaly_det     = AnomalyDetector()

    while not _stop_event.is_set():
        run(source, source_label,
            beh_worker=worker,
            beh_label_cache=beh_label_cache,
            beh_cache_lock=beh_cache_lock,
            anomaly_det=anomaly_det,
            stop_event=_stop_event,
            cam_streamer=None)   # None → uses global streamer (single-camera mode)
        if _stop_event.is_set():
            break
        log.info("Restarting pipeline in 3s...")
        time.sleep(3)

    worker.stop()
    anomaly_det.stop()


def start(source=0, source_label="Camera 0"):
    global _current_thread, _stop_event
    with _thread_lock:
        if _current_thread and _current_thread.is_alive():
            _stop_event.set()
            _current_thread.join(timeout=5)
        _stop_event = threading.Event()
        t = threading.Thread(target=_run_with_restart,
                             args=(source, source_label), daemon=True)
        t.start()
        _current_thread = t


def stop():
    _stop_event.set()