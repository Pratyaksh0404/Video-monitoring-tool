import sys
print(sys.executable)

import os
import cv2
import time
import numpy as np
import face_recognition
from collections import defaultdict

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.behavior_classifier import BehaviorClassifier
from analytics.behavior_engine import BehaviorEngine
from analytics.trajectory_tracker import TrajectoryTracker
from alerts.alert_manager import AlertManager


POST_AREA         = (100, 100, 400, 400)
MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH  = 40
FACE_TOLERANCE    = 0.4   # same as your working attendance system
FACE_SCALE        = 0.5   # resize frame before face_locations for speed


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    if interArea == 0:
        return 0.0
    return interArea / float(
        (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) +
        (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - interArea
    )


# ── Load enrolled faces (same approach as your attendance system) ──────────────
def load_enrolled_faces(faces_dir):
    known_encodings = []
    known_names     = []

    if not os.path.exists(faces_dir):
        print(f"[FaceRecognizer] WARNING: {faces_dir} not found")
        return known_encodings, known_names

    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                encs  = face_recognition.face_encodings(image)
                for enc in encs:
                    known_encodings.append(enc)
                    known_names.append(person_name)
            except Exception as e:
                print(f"[FaceRecognizer] Skipping {img_path}: {e}")

    identities = sorted(set(known_names))
    print(f"[FaceRecognizer] Loaded {len(known_encodings)} encodings "
          f"for {len(identities)} identities: {identities}")
    return known_encodings, known_names


# ── Recognise all faces in a frame (like your attendance system) ───────────────
def recognize_faces_in_frame(frame, known_encodings, known_names):
    """
    Returns list of (top, right, bottom, left, name) for every face found.
    Unknown faces return name="UNKNOWN".
    Resizes frame for speed exactly like your attendance system.
    """
    small  = cv2.resize(frame, (0,0), fx=FACE_SCALE, fy=FACE_SCALE)
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []
    for (top, right, bottom, left), enc in zip(locations, encodings):
        name = "UNKNOWN"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx  = np.argmin(distances)

            if distances[best_idx] < FACE_TOLERANCE:
                matches = face_recognition.compare_faces(
                    known_encodings, enc, tolerance=FACE_TOLERANCE
                )
                if matches[best_idx]:
                    name = known_names[best_idx]

        # Scale coords back to original frame size
        scale = 1.0 / FACE_SCALE
        results.append((
            int(top*scale), int(right*scale),
            int(bottom*scale), int(left*scale),
            name
        ))

    return results


# ── Per-identity trajectory (keyed by name, not track_id) ─────────────────────
class IdentityTrajectory:
    STABILITY  = 2.5
    IDLE_RESET = 12.0

    def __init__(self):
        self.paths     = defaultdict(list)
        self.last_zone = {}
        self.pending   = {}
        self.last_seen = {}

    def update(self, name, zone):
        if zone is None or name == "UNKNOWN":
            return
        now = time.time()
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
                if len(self.paths[name]) > 20:
                    self.paths[name] = self.paths[name][-20:]

    def is_active(self, name):
        last = self.last_seen.get(name)
        return last is not None and (time.time() - last) < self.IDLE_RESET

    def get_path(self, name):
        return " -> ".join(self.paths.get(name, []))

    def get_zone(self, name):
        return self.last_zone.get(name, "?")


def main():
    # ── Setup ──────────────────────────────────────────────────────────────────
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    faces_dir  = os.path.join(base_dir, "data", "enrolled_faces")
    known_encodings, known_names = load_enrolled_faces(faces_dir)

    stream     = VideoStreamReader(source=0)
    detector   = PersonDetector(conf_threshold=0.5)
    tracker    = CentroidTracker(max_disappeared=60, max_distance=300)

    presence_monitor   = PresenceMonitor(absence_threshold=10, confirm_time=3)
    inactivity_monitor = InactivityMonitor(inactivity_threshold=30,
                                           position_threshold=40, window_time=5)
    behavior_classifier = BehaviorClassifier(device="cpu")
    behavior_engine     = BehaviorEngine()
    zone_detector       = TrajectoryTracker(roi=POST_AREA, grid_size=2)
    id_traj             = IdentityTrajectory()
    alert_manager       = AlertManager()

    frame_count     = 0
    behavior_cache  = {}
    identity_memory = {}       # track_id → guard name (persists while track lives)
    track_birth     = {}       # track_id → timestamp when first seen
    presence_buffer = 0
    warmup_frames   = 120
    last_alert_time = {}
    ALERT_COOLDOWN  = 15
    last_printed    = {}       # guard_name → last printed path

    # Face recognition results (updated every N frames)
    face_results    = []       # list of (top, right, bottom, left, name)

    while True:
        frame_count += 1
        frame = stream.read_frame()
        if frame is None:
            break

        if frame_count < warmup_frames:
            cv2.imshow("Guard Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Person detection & tracking ────────────────────────────────────────
        detections      = detector.detect(frame)
        person_boxes    = []
        valid_centroids = []

        for (x1, y1, x2, y2, conf) in detections:
            if (x2-x1) < MIN_PERSON_WIDTH or (y2-y1) < MIN_PERSON_HEIGHT:
                continue
            box = (x1, y1, x2, y2)
            person_boxes.append(box)
            if iou(box, POST_AREA) > 0.25:
                valid_centroids.append((int((x1+x2)/2), int((y1+y2)/2)))

        presence_buffer = 20 if valid_centroids else max(0, presence_buffer-1)
        post_status     = presence_monitor.update(
            valid_centroids if presence_buffer > 0 else []
        )
        activity_status = (inactivity_monitor.update(valid_centroids)
                           if post_status == "PRESENT" else "NO_PERSON")

        tracked_objects = tracker.update(person_boxes)
        zone_detector.update(tracked_objects)

        for dead_id in tracker.recently_deregistered:
            zone_detector.reset_track(dead_id)
            identity_memory.pop(dead_id, None)
            behavior_cache.pop(dead_id, None)
            track_birth.pop(dead_id, None)

        # ── Face recognition every 5 frames (direct on full frame) ───────────
        if frame_count % 5 == 0:
            face_results = recognize_faces_in_frame(frame, known_encodings, known_names)

        # ── Match face results to tracked person boxes ─────────────────────────
        # For each tracked person, find the face result with best overlap
        for track_id, person_box in tracked_objects.items():
            # Record when this track first appeared
            if track_id not in track_birth:
                track_birth[track_id] = time.time()

            best_iou_score = 0.0
            recognized_name = "UNKNOWN"

            for (ft, fr, fb, fl, fname) in face_results:
                face_box = (fl, ft, fr, fb)   # convert to (x1,y1,x2,y2)
                score    = iou(person_box, face_box)
                if score > best_iou_score:
                    best_iou_score  = score
                    recognized_name = fname

            # Only update identity if we got a confident face match
            if recognized_name != "UNKNOWN" and best_iou_score > 0.1:
                identity_memory[track_id] = recognized_name

            guard_id = identity_memory.get(track_id, "UNKNOWN")

            # ── Behavior ───────────────────────────────────────────────────────
            if track_id not in behavior_cache:
                behavior_cache[track_id] = {"label": "ANALYZING", "last_update": 0}
            if frame_count % 20 == 0 and post_status == "PRESENT":
                label, _ = behavior_classifier.predict(frame, person_box)
                behavior_cache[track_id]["label"]       = label
                behavior_cache[track_id]["last_update"] = frame_count
            label       = behavior_cache[track_id]["label"]
            final_state = (behavior_engine.update(track_id, label)
                           if post_status == "PRESENT" else "ANALYZING")

            # ── Trajectory (identity-keyed) ────────────────────────────────────
            current_zone = zone_detector.get_current_zone(track_id)
            id_traj.update(guard_id, current_zone)

            # For known guards: use identity-keyed path
            # For unknown: use track-level zone (resets when track dies — correct)
            if guard_id != "UNKNOWN":
                id_traj.update(guard_id, current_zone)
                display_zone = id_traj.get_zone(guard_id)
                display_path = id_traj.get_path(guard_id)
                if (id_traj.is_active(guard_id)
                        and " -> " in display_path
                        and last_printed.get(guard_id) != display_path):
                    print(f"[{guard_id}] Zone: {display_path}")
                    last_printed[guard_id] = display_path
            else:
                display_zone = current_zone or "?"
                display_path = zone_detector.get_path(track_id)
                # Print unknown person zone changes
                unk_key = f"UNKNOWN_{track_id}"
                if display_zone and display_zone != "?" \
                        and last_printed.get(unk_key) != display_zone:
                    print(f"[UNKNOWN person #{track_id}] Zone: {display_zone}")
                    last_printed[unk_key] = display_zone

            # ── Draw overlays ──────────────────────────────────────────────────
            x1, y1, x2, y2 = person_box
            color = (0,255,0)
            if "CONFIRMED" in final_state:  color = (0,0,255)
            elif "POSSIBLE" in final_state: color = (0,165,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{guard_id} | {final_state}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Zone: {display_zone}",
                        (x1, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            cv2.putText(frame, f"Path: {display_path}",
                        (x1, y1+44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            # ── Alerts ────────────────────────────────────────────────────────
            now = time.time()

            def send_alert(key, msg, _gid=guard_id, _t=now):
                if _t - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                    alert_manager.send_alert(msg, _gid)
                    last_alert_time[key] = _t

            if guard_id != "UNKNOWN":
                if post_status == "ABSENT":
                    send_alert(f"missing:{guard_id}", "Guard Missing")
                elif "CONFIRMED_SLEEPING" in final_state:
                    send_alert(f"sleep:{guard_id}", "Guard Sleeping")
                elif "CONFIRMED_PHONE_USE" in final_state:
                    send_alert(f"phone:{guard_id}", "Phone Usage")
                elif "CONFIRMED_DISTRACTED" in final_state:
                    send_alert(f"distracted:{guard_id}", "Guard Distracted")
                elif activity_status == "INACTIVE":
                    send_alert(f"idle:{guard_id}", "Guard Idle")
            else:
                # Unknown person — only alert after 4s grace period
                # (gives face recognition time to identify before alarming)
                track_age = time.time() - track_birth.get(track_id, now)
                if track_age > 4.0:
                    send_alert(f"unknown:{track_id}", "Unknown Person Detected")

        # ── Draw face recognition boxes on frame ───────────────────────────────
        for (ft, fr, fb, fl, fname) in face_results:
            fc = (0,255,0) if fname != "UNKNOWN" else (0,0,255)
            cv2.rectangle(frame, (fl,ft), (fr,fb), fc, 1)
            cv2.putText(frame, fname, (fl, ft-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, fc, 1)

        # ── HUD ────────────────────────────────────────────────────────────────
        rx1, ry1, rx2, ry2 = POST_AREA
        cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), (255,0,0), 2)

        post_color = ((0,255,0) if post_status=="PRESENT" else
                      (0,255,255) if post_status=="TEMPORARILY_EMPTY" else (0,0,255))
        act_color  = ((0,255,0) if activity_status=="ACTIVE" else
                      (0,0,255) if activity_status=="INACTIVE" else (255,255,255))

        cv2.putText(frame, f"Post: {post_status}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, post_color, 2)
        cv2.putText(frame, f"Activity: {activity_status}",
                    (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, act_color, 2)

        cv2.imshow("Guard Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()