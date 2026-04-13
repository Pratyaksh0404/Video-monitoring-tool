"""
main_web.py  —  CV pipeline for the Flask dashboard.

Fixes in this version (v3)
──────────────────────────
1. GHOST BOXES after crowd leaves
   - max_disappeared reduced 150 → 30 (at 4fps = ~7.5s before track dies)
   - This stops dead tracks from lingering 60+ seconds and generating false alerts

2. UNKNOWN FLOOD in crowd
   - Unknown alerts suppressed entirely when is_crowd == True
   - When crowd disperses, a 10s post-crowd cooldown prevents immediate
     unknown alerts from the freshly-deregistered tracks re-appearing

3. GUARD MISSING cooldown raised 15s → 30s
   - Prevents spamming during a genuine long absence

4. SMOKING false positives — additional movement gate
   - Smoking alert only fires when person is NOT moving (speed < IDLE_MAX_SPEED)
   - A walking/patrolling person cannot be classified as smoking

5. LOITERING suppressed during and briefly after crowd
   - Loitering alerts suppressed for 15s after crowd disperses
   - Prevents freshly-arrived unknown persons from immediately triggering loitering

6. BEHAVIOR suppressed for unknowns during crowd
   - CLIP behavior (smoking, idle, distracted) suppressed for unknown persons
     when crowd is active — too many false triggers in a busy scene

7. Guard Missing cooldown separate from ALERT_COOLDOWN
   - Uses MISSING_COOLDOWN = 30s instead of 15s ALERT_COOLDOWN
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
from analytics.loitering_detector import LoiteringDetector
from analytics.fight_detector import FightDetector
from analytics.unknown_tracker import UnknownTracker
from alerts.alert_manager import AlertManager
from analytics.anomaly_detector import AnomalyDetector
from video_streamer import streamer

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_PERSON_HEIGHT    = 60
MIN_PERSON_WIDTH     = 30
FACE_TOLERANCE       = 0.4
FACE_SCALE           = 0.5
CROWD_THRESHOLD      = 4
FILE_PRESENCE_WARMUP = 30
PRESENCE_HOLD        = 5
FIRE_FRAME_INTERVAL  = 20
FIRE_RESULT_MAX_AGE  = 8.0
FIRE_MIN_CONFIDENCE  = 0.65   # FIX: raised from 0.50 — reduces lighting false alarms
FIRE_SUSTAIN_SECS    = 4.0    # FIX: fire must be detected for 4s before alert fires
UNKNOWN_GRACE_SECS   = 12
IDLE_MAX_SPEED       = 12.0
PATROL_DISPLAY_ZONES = 3
MISSING_COOLDOWN     = 30    # FIX: raised from 15s — don't spam during long absence
POST_CROWD_GRACE     = 10.0  # FIX: seconds after crowd ends before unknown alerts resume
POST_CROWD_LOITER    = 15.0  # FIX: seconds after crowd ends before loitering alerts resume

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


def run(source=0, source_label="Camera 0", beh_worker=None,
        beh_label_cache=None, beh_cache_lock=None, anomaly_det=None):
    global _stop_event
    _stop_event.clear()

    is_file = isinstance(source, str)
    streamer.set_source("file" if is_file else "webcam", source_label)

    faces_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "data", "enrolled_faces"
    )
    known_encodings, known_names = load_enrolled_faces(faces_dir)
    # FIX: set of enrolled guard names — used to suppress "Unknown" when face is just sideways
    enrolled_names = set(known_names)

    try:
        stream = VideoStreamReader(source=source)
    except RuntimeError as e:
        print(f"[main_web] Camera error: {e}")
        time.sleep(2)
        return

    detector        = PersonDetector(conf_threshold=0.5)
    # FIX: max_disappeared reduced 150→30. At 4fps = ~7.5s before ghost track dies.
    # Old value of 150 kept tracks alive ~37s, causing false alerts after crowd leaves.
    tracker         = CentroidTracker(max_disappeared=30, max_distance=300)
    presence_mon    = PresenceMonitor(absence_threshold=10, confirm_time=3)
    inactivity_mon  = InactivityMonitor(inactivity_threshold=45,
                                        position_threshold=80, window_time=10)
    beh_engine      = BehaviorEngine()
    alert_manager   = AlertManager()
    loitering_det   = LoiteringDetector()
    fight_det       = FightDetector()
    unknown_tracker = UnknownTracker()

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
    frame_count      = 0
    behavior_cache   = {}
    identity_memory  = {}
    track_birth      = {}
    track_last_pos   = {}
    slot_birth       = {}

    presence_buffer  = FILE_PRESENCE_WARMUP if is_file else 0
    warmup_frames    = 10 if is_file else 120
    last_alert_time  = {}
    ALERT_COOLDOWN   = 30    # raised from 15s — prevents rapid repeat alerts
    last_beh_frame   = {}
    BEH_INTERVAL     = 50    # raised from 30 frames — ~9s between CLIP reads at 5.5fps
                              # With window=8 readings, confirmation takes ~40-55s

    post_area     = None
    zone_detector = None
    id_traj       = IdentityTrajectory()
    last_printed  = {}
    face_results  = []

    last_seen_guard_name     = "Post"
    last_patrol_alerted_zone = {}

    # FIX: crowd state tracking for post-crowd grace periods
    crowd_ended_at   = 0.0   # timestamp when crowd last ended
    was_crowd        = False

    # FIX: fire sustained detection — must detect for FIRE_SUSTAIN_SECS before alert
    fire_first_seen  = 0.0   # timestamp when fire was first detected in current episode

    try:
        while not _stop_event.is_set():
            frame_count += 1
            frame = stream.read_frame()
            if frame is None:
                break

            if frame_count < warmup_frames:
                streamer.push_frame(frame)
                continue

            if post_area is None:
                h, w = frame.shape[:2]
                post_area     = (0, 0, w, h)
                zone_detector = TrajectoryTracker(roi=post_area, grid_size=2)
                print(f"[main_web] Frame: {w}x{h}, POST_AREA={post_area}")

            # ── Detection ────────────────────────────────────────────────────
            detections   = detector.detect(frame)
            person_boxes = []
            for (x1, y1, x2, y2, conf) in detections:
                if (x2-x1) < MIN_PERSON_WIDTH or (y2-y1) < MIN_PERSON_HEIGHT:
                    continue
                person_boxes.append((x1, y1, x2, y2))

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
                beh_engine.reset(dead_id)   # FIX: clear majority-vote window for dead track
                last_pos = track_last_pos.pop(dead_id, None)
                unknown_tracker.on_track_lost(dead_id, last_pos)
                if anomaly_det:
                    anomaly_det.clear_result(dead_id)

            # ── Face recognition every 8 frames ──────────────────────────────
            if frame_count % 8 == 0:
                face_results = recognize_faces_in_frame(
                    frame, known_encodings, known_names)

            # ─────────────────────────────────────────────────────────────────
            # FRAME-LEVEL DETECTIONS
            # ─────────────────────────────────────────────────────────────────
            active_guard_ids  = set()
            active_violations = set()
            now               = time.time()
            is_crowd          = len(person_boxes) >= CROWD_THRESHOLD

            # FIX: Track crowd state transitions for post-crowd grace
            if was_crowd and not is_crowd:
                crowd_ended_at = now   # crowd just ended
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
            if anomaly_det and anomaly_det._fire_ready:
                if frame_count % FIRE_FRAME_INTERVAL == 0:
                    anomaly_det.submit_frame(frame)

                fire_res = anomaly_det.get_fire_result()
                # FIX: higher confidence threshold + sustained detection required
                fire_confident = (fire_res
                        and fire_res.get("is_alert")
                        and fire_res.get("score", 0) >= FIRE_MIN_CONFIDENCE
                        and (now - fire_res.get("timestamp", 0)) < FIRE_RESULT_MAX_AGE)

                if fire_confident:
                    if fire_first_seen == 0.0:
                        fire_first_seen = now   # start sustained timer
                    # Only alert after sustained for FIRE_SUSTAIN_SECS
                    elif (now - fire_first_seen) >= FIRE_SUSTAIN_SECS:
                        if now - last_alert_time.get("fire", 0) > ALERT_COOLDOWN:
                            t = fire_res.get("threat") or "Fire"
                            alert_manager.send_alert(
                                f"Fire / Smoke Detected: {t}", "Camera", zone="—")
                            last_alert_time["fire"] = now
                            active_violations.add("Fire")
                    cv2.putText(frame,
                                f"FIRE/SMOKE: {fire_res.get('threat','')} "
                                f"({fire_res.get('score',0):.0%})",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 255), 2)
                else:
                    fire_first_seen = 0.0   # reset sustained timer if detection drops

            # ── 3. Full-frame weapon scan ─────────────────────────────────────
            if anomaly_det and anomaly_det.is_ready():
                if frame_count % 20 == 0:
                    anomaly_det.submit(track_id=-1, frame_crop=frame.copy())

                full_res = anomaly_det.get_result(-1)
                if (full_res
                        and full_res.get("is_alert")
                        and (now - full_res.get("timestamp", 0)) < 5.0):
                    threat = full_res.get("threat") or "Weapon"
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
                        # FIX: Persons present — fire as weapon alert if no per-person
                        # result already caught it. This is the knife fix — knife is often
                        # missed in small person crops but caught on full frame.
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

                guard_id = identity_memory.get(track_id, "UNKNOWN")

                if guard_id == "UNKNOWN":
                    alert_name = unknown_tracker.get_or_assign(track_id, centroid)
                    if alert_name not in slot_birth:
                        slot_birth[alert_name] = time.time()
                else:
                    alert_name = guard_id

                active_guard_ids.add(alert_name)

                # ── Behavior ──────────────────────────────────────────────────
                # FIX: Don't run behavior for unknowns during crowd — too many false hits
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
                        print(f"[PATROL] {guard_id} — {short_path}")
                        last_patrol_alerted_zone[guard_id] = latest_zone
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
                elif is_loitering:                          color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{alert_name} | {final_state}"
                if anomaly_label != "NORMAL":
                    label_text += f" | {anomaly_threat or anomaly_label}"
                if is_loitering:
                    secs = loitering_det.time_in_zone(track_id)
                    label_text += f" | LOITER {int(secs//60)}m{int(secs%60)}s"

                cv2.putText(frame, label_text,
                            (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if display_zone and display_zone != "?":
                    cv2.putText(frame, f"Zone:{display_zone}",
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
                    if "CONFIRMED_SLEEPING" in final_state:
                        send_alert(f"sleep:{guard_id}", "Guard Sleeping")
                    elif "CONFIRMED_PHONE_USE" in final_state:
                        send_alert(f"phone:{guard_id}", "Phone Usage")
                    elif "CONFIRMED_DISTRACTED_OTHER" in final_state:
                        send_alert(f"distracted:{guard_id}", "Guard Distracted")
                    elif "CONFIRMED_SMOKING" in final_state and not is_moving:
                        # FIX: smoking only when NOT moving — walking guard can't smoke
                        send_alert(f"smoking:{guard_id}", "Guard Smoking")
                    elif not is_moving and (
                            "CONFIRMED_IDLE" in final_state
                            and activity_status == "INACTIVE"):
                        send_alert(f"idle:{guard_id}", "Guard Idle")

                    if is_loitering and not in_post_crowd_loiter:
                        send_alert(f"loiter:{guard_id}", "Guard Loitering")

                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:{guard_id}:{anomaly_threat}",
                                   f"Weapon Detected: {anomaly_threat}")

                else:
                    # FIX: unknown alerts suppressed during crowd AND post-crowd grace.
                    # Also suppressed if ANY currently-tracked known guard has a box
                    # overlapping this unknown person (sideways face = same person).
                    # We compare centroids — if a known guard's centroid is within
                    # 150px of this unknown, it's likely the same person turned sideways.
                    sideways_suppressed = False
                    for tid, tbox in tracked_objects.items():
                        if tid == track_id:
                            continue
                        if identity_memory.get(tid) in enrolled_names:
                            tc = box_centroid(tbox)
                            dist = ((centroid[0]-tc[0])**2 + (centroid[1]-tc[1])**2)**0.5
                            if dist < 150:
                                sideways_suppressed = True
                                break

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
                    elif "CONFIRMED_DISTRACTED_OTHER" in final_state:
                        send_alert(f"distracted:{alert_name}",
                                   "Unknown Person Distracted")

                    # FIX: loitering for unknowns suppressed during/after crowd
                    if is_loitering and not is_crowd and not in_post_crowd_loiter:
                        send_alert(f"loiter:{alert_name}",
                                   "Suspicious Loitering Detected")

                    if anomaly_label == "WEAPON" and anomaly_threat:
                        send_alert(f"weapon:{alert_name}:{anomaly_threat}",
                                   f"Weapon Detected: {anomaly_threat}")

            # ── Guard Missing ─────────────────────────────────────────────────
            # FIX: uses MISSING_COOLDOWN (30s) not ALERT_COOLDOWN (15s)
            if post_status == "ABSENT":
                now = time.time()
                if now - last_alert_time.get("post_missing", 0) > MISSING_COOLDOWN:
                    alert_manager.send_alert(
                        "Guard Missing", last_seen_guard_name, zone="—")
                    last_alert_time["post_missing"] = now

            # ── Stats ─────────────────────────────────────────────────────────
            streamer.update_stats(
                guards_detected=len(active_guard_ids),
                active_violations=len(active_violations),
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
            cv2.putText(frame, f"FPS:{streamer.fps}", (8,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

            streamer.push_frame(frame)

    except Exception as e:
        import traceback
        print(f"[main_web] CRASH: {e}"); traceback.print_exc()
    finally:
        stream.release()
        loitering_det.reset_all()
        fight_det.reset()
        unknown_tracker.reset()
        print("[main_web] Loop ended.")


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
        t = threading.Thread(target=_run_with_restart,
                             args=(source, source_label), daemon=True)
        t.start()
        _current_thread = t


def stop():
    _stop_event.set()