"""
main_web.py  —  CV pipeline for the Flask dashboard.

Face recognition uses face_recognition.face_locations() directly on the full
frame (same approach as the working attendance system / src/main.py).
Trajectory is keyed by guard identity name, not track_id, so paths never
mix between people even when track IDs are reassigned.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import time
import threading
import queue
import numpy as np
import face_recognition
from collections import defaultdict

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.behavior_engine import BehaviorEngine
from analytics.trajectory_tracker import TrajectoryTracker
from alerts.alert_manager import AlertManager
from analytics.anomaly_detector import AnomalyDetector
from video_streamer import streamer

MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH  = 40
FACE_TOLERANCE    = 0.4
FACE_SCALE        = 0.5   # resize for face_locations speed

_stop_event     = threading.Event()
_thread_lock    = threading.Lock()
_current_thread = None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    if interArea == 0:
        return 0.0
    return interArea / float(
        (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) +
        (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - interArea
    )


# ── Load enrolled faces ────────────────────────────────────────────────────────
def load_enrolled_faces(faces_dir):
    known_encodings, known_names = [], []
    if not os.path.exists(faces_dir):
        print(f"[FaceRecognizer] WARNING: {faces_dir} not found")
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
    print(f"[FaceRecognizer] Loaded {len(known_encodings)} encodings for "
          f"{len(set(known_names))} identities: {sorted(set(known_names))}")
    return known_encodings, known_names


# ── Recognise all faces in a frame ────────────────────────────────────────────
def recognize_faces_in_frame(frame, known_encodings, known_names):
    """Returns [(top,right,bottom,left,name), ...] for every face found."""
    small = cv2.resize(frame, (0,0), fx=FACE_SCALE, fy=FACE_SCALE)
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


# ── Per-identity trajectory (keyed by name, survives track_id swaps) ──────────
class IdentityTrajectory:
    STABILITY  = 2.5
    IDLE_RESET = 12.0  # seconds without being seen → stop zone updates

    def __init__(self):
        self.paths     = defaultdict(list)
        self.last_zone = {}
        self.pending   = {}       # name → (candidate_zone, since_time)
        self.last_seen = {}       # name → timestamp last actively updated

    def update(self, name, zone):
        if zone is None or name == "UNKNOWN":
            return

        now = time.time()

        # If this identity hasn't been seen for IDLE_RESET seconds,
        # reset their pending state so stale zone changes don't fire
        if name in self.last_seen:
            if now - self.last_seen[name] > self.IDLE_RESET:
                self.pending.pop(name, None)

        self.last_seen[name] = now

        prev = self.last_zone.get(name)
        if prev is None:
            self.last_zone[name] = zone
            self.paths[name].append(zone)
            return
        if zone == prev:
            self.pending.pop(name, None)
            return
        cand, since = self.pending.get(name, (None, 0))
        if cand != zone:
            self.pending[name] = (zone, now)
            return
        if now - since >= self.STABILITY:
            self.last_zone[name] = zone
            self.pending.pop(name, None)
            if not self.paths[name] or self.paths[name][-1] != zone:
                self.paths[name].append(zone)
                # Keep only last 20 zones — prevents infinite growth
                if len(self.paths[name]) > 20:
                    self.paths[name] = self.paths[name][-20:]

    def is_active(self, name):
        """True if this identity was seen within the last IDLE_RESET seconds."""
        last = self.last_seen.get(name)
        return last is not None and (time.time() - last) < self.IDLE_RESET

    def get_path(self, name):
        return " -> ".join(self.paths.get(name, []))

    def get_zone(self, name):
        return self.last_zone.get(name, None)


# ── Background behavior inference thread ──────────────────────────────────────
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
            print("[BehaviorWorker] CLIP model ready.")
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


# ── Main CV loop ───────────────────────────────────────────────────────────────
def run(source=0, source_label="Camera 0", beh_worker=None,
        beh_label_cache=None, beh_cache_lock=None, anomaly_det=None):
    global _stop_event
    _stop_event.clear()

    is_file = isinstance(source, str)
    streamer.set_source("file" if is_file else "webcam", source_label)

    # Load enrolled faces
    faces_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "data", "enrolled_faces"
    )
    known_encodings, known_names = load_enrolled_faces(faces_dir)

    # Open stream — catch failure cleanly
    try:
        stream = VideoStreamReader(source=source)
    except RuntimeError as e:
        print(f"[main_web] Camera error: {e}")
        time.sleep(2)
        return

    detector     = PersonDetector(conf_threshold=0.5)
    tracker      = CentroidTracker(max_disappeared=60, max_distance=300)
    presence_mon = PresenceMonitor(absence_threshold=10, confirm_time=3)
    inactivity_mon = InactivityMonitor(inactivity_threshold=30,
                                       position_threshold=40, window_time=5)
    beh_engine     = BehaviorEngine()
    alert_manager  = AlertManager()

    # Wait for first frame — threaded reader needs a moment to capture
    print("[main_web] Waiting for first frame...")
    for _ in range(50):
        if stream.read_frame() is not None:
            break
        time.sleep(0.1)
    else:
        print("[main_web] Camera not responding — aborting.")
        stream.release()
        return

    print("[main_web] Camera ready.")
    frame_count     = 0
    behavior_cache  = {}
    identity_memory = {}   # track_id → name, reset each run()
    track_birth     = {}   # track_id → first-seen timestamp
    presence_buffer = 0
    warmup_frames   = 10 if is_file else 120
    last_alert_time = {}
    ALERT_COOLDOWN  = 15
    last_beh_frame  = {}
    BEH_INTERVAL    = 25

    # POST_AREA and zone tracker initialised from first real frame
    post_area    = None
    zone_detector = None           # TrajectoryTracker — track_id keyed zones
    id_traj       = IdentityTrajectory()   # name-keyed paths
    last_printed  = {}             # guard_name → last printed path string

    # Face recognition results, updated every 5 frames
    face_results  = []

    try:
        while not _stop_event.is_set():
            frame_count += 1
            frame = stream.read_frame()
            if frame is None:
                break

            if frame_count < warmup_frames:
                streamer.push_frame(frame)
                continue

            # Initialise POST_AREA from actual frame dimensions
            if post_area is None:
                h, w = frame.shape[:2]
                post_area    = (0, 0, w, h)
                zone_detector = TrajectoryTracker(roi=post_area, grid_size=2)
                print(f"[main_web] Frame: {w}x{h}, POST_AREA={post_area}")

            # ── Detection ─────────────────────────────────────────────────────
            detections      = detector.detect(frame)
            person_boxes    = []
            valid_centroids = []

            for (x1, y1, x2, y2, conf) in detections:
                if (x2-x1) < MIN_PERSON_WIDTH or (y2-y1) < MIN_PERSON_HEIGHT:
                    continue
                box = (x1, y1, x2, y2)
                person_boxes.append(box)
                if iou(box, post_area) > 0.25:
                    valid_centroids.append((int((x1+x2)/2), int((y1+y2)/2)))

            presence_buffer = 20 if valid_centroids else max(0, presence_buffer-1)
            post_status     = presence_mon.update(
                valid_centroids if presence_buffer > 0 else [])
            activity_status = (inactivity_mon.update(valid_centroids)
                               if post_status == "PRESENT" else "NO_PERSON")

            tracked_objects = tracker.update(person_boxes)
            zone_detector.update(tracked_objects)

            # Clean up dead tracks
            for dead_id in tracker.recently_deregistered:
                zone_detector.reset_track(dead_id)
                identity_memory.pop(dead_id, None)
                behavior_cache.pop(dead_id, None)
                beh_label_cache.pop(dead_id, None)
                last_beh_frame.pop(dead_id, None)
                track_birth.pop(dead_id, None)

            # ── Face recognition every 5 frames ───────────────────────────────
            if frame_count % 5 == 0:
                face_results = recognize_faces_in_frame(
                    frame, known_encodings, known_names)

            active_guard_ids  = set()
            active_violations = set()

            for track_id, person_box in tracked_objects.items():
                if track_id not in track_birth:
                    track_birth[track_id] = time.time()

                # ── Identity: match face results to person box ─────────────────
                best_score, recognized_name = 0.0, "UNKNOWN"
                for (ft, fr, fb, fl, fname) in face_results:
                    face_box = (fl, ft, fr, fb)
                    score    = iou(person_box, face_box)
                    if score > best_score:
                        best_score, recognized_name = score, fname

                if recognized_name != "UNKNOWN" and best_score > 0.1:
                    identity_memory[track_id] = recognized_name

                guard_id   = identity_memory.get(track_id, "UNKNOWN")
                alert_name = guard_id if guard_id != "UNKNOWN" else f"Guard_{track_id}"
                active_guard_ids.add(alert_name)

                # ── Behavior ───────────────────────────────────────────────────
                if post_status == "PRESENT":
                    last_f = last_beh_frame.get(track_id, 0)
                    if frame_count - last_f >= BEH_INTERVAL:
                        beh_worker.submit(track_id, frame, person_box)
                        last_beh_frame[track_id] = frame_count

                with beh_cache_lock:
                    raw_label = beh_label_cache.get(track_id, "ANALYZING")
                if track_id not in behavior_cache:
                    behavior_cache[track_id] = {"label": raw_label}
                else:
                    behavior_cache[track_id]["label"] = raw_label

                label       = behavior_cache[track_id]["label"]
                final_state = (beh_engine.update(track_id, label)
                               if post_status == "PRESENT" else "ANALYZING")

                # ── Anomaly detection ──────────────────────────────────────────
                anomaly_label = "NORMAL"
                anomaly_threat = None
                if anomaly_det and anomaly_det.is_ready():
                    # Submit crop every 8 frames — YOLOv8 Nano is fast on CPU
                    if frame_count % 8 == 0:
                        x1a, y1a, x2a, y2a = person_box
                        crop = frame[max(0,y1a):y2a, max(0,x1a):x2a]
                        anomaly_det.submit(track_id, crop)
                    # Read latest result
                    res = anomaly_det.get_result(track_id)
                    if res:
                        anomaly_label  = res["label"]
                        anomaly_threat = res.get("threat")

                # ── Trajectory ─────────────────────────────────────────────────
                current_zone = zone_detector.get_current_zone(track_id)

                if guard_id != "UNKNOWN":
                    id_traj.update(guard_id, current_zone)
                    display_zone = id_traj.get_zone(guard_id) or current_zone or "?"
                    display_path = id_traj.get_path(guard_id)

                    # Only log patrol when guard is actively present
                    if (id_traj.is_active(guard_id)
                            and " -> " in display_path
                            and last_printed.get(guard_id) != display_path):
                        print(f"[PATROL] {guard_id} — {display_path}")
                        last_printed[guard_id] = display_path
                        alert_manager.send_alert(
                            f"Patrol: {display_path}", guard_id,
                            zone=display_zone)
                else:
                    display_zone = current_zone or "?"
                    display_path = zone_detector.get_path(track_id)

                # ── Draw overlays ──────────────────────────────────────────────
                x1, y1, x2, y2 = person_box
                color = (0, 255, 0)
                if anomaly_label == "WEAPON":    color = (0, 0, 255)  # Red for weapons
                elif anomaly_label == "THREAT":    color = (0, 0, 255)
                elif anomaly_label == "ANOMALY": color = (0, 165, 255)
                elif "CONFIRMED" in final_state: color = (0, 0, 255)
                elif "POSSIBLE" in final_state:  color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{alert_name} | {final_state}"
                if anomaly_label != "NORMAL":
                    threat_str = anomaly_threat or anomaly_label
                    label_text += f" | {threat_str}"
                cv2.putText(frame, label_text,
                            (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if display_zone and display_zone != "?":
                    cv2.putText(frame, f"Zone: {display_zone}",
                                (x1, y1+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                # Draw face recognition boxes
                for (ft, fr, fb, fl, fname) in face_results:
                    fc = (0,255,0) if fname != "UNKNOWN" else (0,0,255)
                    cv2.rectangle(frame, (fl,ft), (fr,fb), fc, 1)

                # ── Alerts ─────────────────────────────────────────────────────
                now = time.time()

                def send_alert(key, msg,
                               _gid=alert_name, _zone=display_zone, _t=now):
                    if _t - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                        alert_manager.send_alert(msg, _gid, zone=_zone)
                        last_alert_time[key] = _t
                        active_violations.add(msg)

                if guard_id != "UNKNOWN":
                    if "CONFIRMED_SLEEPING" in final_state:
                        send_alert(f"sleep:{guard_id}", "Guard Sleeping")
                    elif "CONFIRMED_PHONE_USE" in final_state:
                        send_alert(f"phone:{guard_id}", "Phone Usage")
                    elif "CONFIRMED_DISTRACTED" in final_state:
                        send_alert(f"distracted:{guard_id}", "Guard Distracted")
                    elif activity_status == "INACTIVE":
                        send_alert(f"idle:{guard_id}", "Guard Idle")
                    # Anomaly alerts for known guards
                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:{guard_id}:{anomaly_threat}",
                                   f"Weapon Detected: {anomaly_threat}")
                    elif anomaly_label == "THREAT" and anomaly_threat:
                        send_alert(f"threat:{guard_id}:{anomaly_threat}",
                                   f"Threat Detected: {anomaly_threat}")
                    elif anomaly_label == "ANOMALY":
                        send_alert(f"anomaly:{guard_id}",
                                   "Suspicious Activity Detected")
                else:
                    # Unknown person alert after 6s grace period
                    track_age = now - track_birth.get(track_id, now)
                    if track_age > 6.0:
                        send_alert(f"unknown:{track_id}",
                                   "Unknown Person Detected")
                    # Weapon detection fires immediately for unknowns
                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:unknown:{track_id}",
                                   f"Weapon Detected: {anomaly_threat}")
                    # Threat detection fires immediately for unknowns
                    elif anomaly_label == "THREAT" and anomaly_threat:
                        send_alert(f"threat:unknown:{track_id}",
                                   f"Threat Detected: {anomaly_threat}")

            # Guard Missing — post level only
            if post_status == "ABSENT":
                now = time.time()
                if now - last_alert_time.get("post_missing", 0) > ALERT_COOLDOWN:
                    alert_manager.send_alert("Guard Missing", "Post", zone="—")
                    last_alert_time["post_missing"] = now

            # ── Stats ──────────────────────────────────────────────────────────
            streamer.update_stats(
                guards_detected=len(active_guard_ids),
                active_violations=len(active_violations),
            )

            # ── HUD ────────────────────────────────────────────────────────────
            rx1, ry1, rx2, ry2 = post_area
            mid_x = (rx1 + rx2) // 2
            mid_y = (ry1 + ry2) // 2
            cv2.line(frame, (mid_x, ry1), (mid_x, ry2), (255,100,0), 1)
            cv2.line(frame, (rx1, mid_y), (rx2, mid_y), (255,100,0), 1)
            for lbl, pos in [("A",(rx1+8, ry1+20)), ("B",(mid_x+8, ry1+20)),
                              ("C",(rx1+8, mid_y+20)), ("D",(mid_x+8, mid_y+20))]:
                cv2.putText(frame, lbl, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 2)

            post_color = ((0,255,0)   if post_status == "PRESENT" else
                          (0,255,255) if "TEMP" in post_status else (0,0,255))
            cv2.putText(frame, post_status,
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, post_color, 1)
            cv2.putText(frame, f"FPS:{streamer.fps}",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

            streamer.push_frame(frame)

    except Exception as e:
        import traceback
        print(f"[main_web] CRASH: {e}")
        traceback.print_exc()
    finally:
        stream.release()
        print("[main_web] Loop ended.")


def _run_with_restart(source, source_label):
    # Create BehaviorWorker ONCE — CLIP loads once, survives restarts
    beh_label_cache = {}
    beh_cache_lock  = threading.Lock()
    worker          = BehaviorWorker(beh_label_cache, beh_cache_lock)

    # AnomalyDetector also loads once — shares no state with BehaviorWorker
    anomaly_det = AnomalyDetector()

    while not _stop_event.is_set():
        run(source, source_label,
            beh_worker=worker,
            beh_label_cache=beh_label_cache,
            beh_cache_lock=beh_cache_lock,
            anomaly_det=anomaly_det)
        if _stop_event.is_set():
            break
        print("[main_web] Restarting in 3s...")
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
        t = threading.Thread(
            target=_run_with_restart,
            args=(source, source_label),
            daemon=True
        )
        t.start()
        _current_thread = t


def stop():
    _stop_event.set()