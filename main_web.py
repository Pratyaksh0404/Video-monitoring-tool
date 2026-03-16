"""
main_web.py  —  CV pipeline for the Flask dashboard.

Key improvements over previous version:
  1. Behavior classifier runs in a background thread → main loop no longer
     blocks on CLIP inference → FPS improves significantly on CPU
  2. Patrol zone changes are sent to alert_manager so they appear in the
     dashboard log panel (as severity "low" patrol events)
  3. POST_AREA covers the full camera frame so all 4 zones A/B/C/D are
     reachable regardless of where the guard stands
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import time
import threading
import queue

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.behavior_engine import BehaviorEngine
from analytics.trajectory_tracker import TrajectoryTracker
from alerts.alert_manager import AlertManager
from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from face.face_recognizer import FaceRecognizer
from video_streamer import streamer

# ── ROI covers the full 640x480 frame so all zones are always reachable ───────
# Change these if your camera resolution is different
POST_AREA = (0, 0, 640, 480)

MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH  = 40

_stop_event     = threading.Event()
_thread_lock    = threading.Lock()
_current_thread = None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    return interArea / float(
        (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) +
        (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - interArea
    )


# ── Background behavior inference thread ──────────────────────────────────────
# Loads the CLIP model in its own thread so the camera starts instantly.
# The main loop shows "ANALYZING" until the model is ready, then switches
# to real predictions automatically — no blocking, no waiting.

class BehaviorWorker:
    """
    Lazy-loads BehaviorClassifier, then processes (track_id, frame, box) jobs.
    Results are written to a shared cache dict read by the main loop.
    """
    def __init__(self, cache, cache_lock):
        self._q          = queue.Queue(maxsize=4)
        self._cache      = cache
        self._lock       = cache_lock
        self._stopped    = False
        self._classifier = None          # loaded lazily inside thread
        self._ready      = False
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def ready(self):
        return self._ready

    def submit(self, track_id, frame, box):
        if not self._ready:
            return   # silently skip until model is loaded
        try:
            self._q.put_nowait((track_id, frame, box))
        except queue.Full:
            pass

    def _loop(self):
        # Load model here — does NOT block the main CV loop
        try:
            from analytics.behavior_classifier import BehaviorClassifier
            self._classifier = BehaviorClassifier(device="cpu")
            self._ready = True
            print("[BehaviorWorker] CLIP model ready.")
        except Exception as e:
            print(f"[BehaviorWorker] Failed to load classifier: {e}")
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


def run(source=0, source_label="Camera 0"):
    global _stop_event
    _stop_event.clear()

    is_file = isinstance(source, str)
    streamer.set_source("file" if is_file else "webcam", source_label)

    stream          = VideoStreamReader(source=source)
    detector        = PersonDetector(conf_threshold=0.5)
    tracker         = CentroidTracker(max_disappeared=300, max_distance=300)
    presence_mon    = PresenceMonitor(absence_threshold=10, confirm_time=3)
    inactivity_mon  = InactivityMonitor(inactivity_threshold=30,
                                        position_threshold=40, window_time=5)
    beh_engine      = BehaviorEngine()
    traj_tracker    = TrajectoryTracker(roi=POST_AREA, grid_size=2)
    alert_manager   = AlertManager()
    face_detector   = FaceDetector()
    face_encoder    = FaceEncoder()

    _faces_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "data", "enrolled_faces"
    )
    face_recognizer = FaceRecognizer(faces_dir=_faces_path, tolerance=0.55)

    # BehaviorWorker loads CLIP lazily in its own thread — camera starts instantly
    beh_label_cache = {}
    beh_cache_lock  = threading.Lock()
    beh_worker      = BehaviorWorker(beh_label_cache, beh_cache_lock)

    frame_count     = 0
    behavior_cache  = {}   # per-track {label, last_frame} for BehaviorEngine
    identity_memory = {}
    presence_buffer = 0
    warmup_frames   = 120
    face_boxes      = []
    identities      = []
    last_alert_time = {}
    ALERT_COOLDOWN  = 15
    last_path       = {}
    last_beh_frame  = {}   # track_id → frame_count when we last submitted

    BEH_INTERVAL = 25      # submit a new behavior job every N frames per track

    try:
        while not _stop_event.is_set():
            frame_count += 1
            frame = stream.read_frame()
            if frame is None:
                break

            if frame_count < warmup_frames:
                streamer.push_frame(frame)
                continue

            # ── Detection ─────────────────────────────────────────────────────
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
            post_status     = presence_mon.update(
                valid_centroids if presence_buffer > 0 else []
            )
            activity_status = (inactivity_mon.update(valid_centroids)
                               if post_status == "PRESENT" else "NO_PERSON")

            tracked_objects = tracker.update(person_boxes)
            traj_tracker.update(tracked_objects)

            # Face recognition every 15 frames
            if frame_count % 15 == 0:
                face_boxes  = face_detector.detect(frame, list(tracked_objects.values()))
                face_encs   = face_encoder.encode(frame, face_boxes)
                identities  = face_recognizer.recognize(face_encs)

            active_guard_ids  = set()
            active_violations = set()

            for track_id, person_box in tracked_objects.items():

                # ── Identity ───────────────────────────────────────────────────
                best_iou, recognized_id = 0.0, "UNKNOWN"
                for i, fb in enumerate(face_boxes):
                    ov = iou(person_box, fb[:4])
                    if ov > best_iou:
                        best_iou = ov
                        recognized_id = identities[i] if i < len(identities) else "UNKNOWN"
                if recognized_id != "UNKNOWN":
                    identity_memory[track_id] = recognized_id
                guard_id   = identity_memory.get(track_id, "UNKNOWN")
                alert_name = guard_id if guard_id != "UNKNOWN" else f"Guard_{track_id}"
                active_guard_ids.add(alert_name)

                # ── Submit crop to behavior worker every BEH_INTERVAL frames ──
                if post_status == "PRESENT":
                    last_f = last_beh_frame.get(track_id, 0)
                    if frame_count - last_f >= BEH_INTERVAL:
                        beh_worker.submit(track_id, frame, person_box)
                        last_beh_frame[track_id] = frame_count

                # ── Read latest label from worker cache ────────────────────────
                with beh_cache_lock:
                    raw_label = beh_label_cache.get(track_id, "ANALYZING")

                if track_id not in behavior_cache:
                    behavior_cache[track_id] = {"label": raw_label}
                else:
                    behavior_cache[track_id]["label"] = raw_label

                label       = behavior_cache[track_id]["label"]
                final_state = (beh_engine.update(track_id, label)
                               if post_status == "PRESENT" else "ANALYZING")

                # ── Trajectory → dashboard ─────────────────────────────────────
                zone = traj_tracker.get_current_zone(track_id)
                path = traj_tracker.get_path(track_id)
                if path and last_path.get(track_id) != path:
                    print(f"[PATROL] {alert_name} — {path}")
                    last_path[track_id] = path
                    # Send patrol event to dashboard log panel
                    alert_manager.send_alert(
                        f"Patrol: {path}", alert_name, zone=zone or "—"
                    )

                # ── Draw overlays ──────────────────────────────────────────────
                x1, y1, x2, y2 = person_box
                color = (0, 255, 0)
                if "CONFIRMED" in final_state:  color = (0, 0, 255)
                elif "POSSIBLE" in final_state: color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{alert_name} | {final_state}",
                            (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if zone:
                    cv2.putText(frame, f"Zone: {zone}",
                                (x1, y1+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                # ── Alerts ─────────────────────────────────────────────────────
                current_time = time.time()

                def send_alert(key, msg,
                               _gid=alert_name, _zone=zone, _t=current_time):
                    if _t - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                        alert_manager.send_alert(msg, _gid, zone=_zone)
                        last_alert_time[key] = _t
                        active_violations.add(msg)

                if post_status == "ABSENT":
                    send_alert(f"missing:{alert_name}", "Guard Missing")
                elif "CONFIRMED_SLEEPING" in final_state:
                    send_alert(f"sleep:{alert_name}", "Guard Sleeping")
                elif "CONFIRMED_PHONE_USE" in final_state:
                    send_alert(f"phone:{alert_name}", "Phone Usage")
                elif "CONFIRMED_DISTRACTED" in final_state:
                    send_alert(f"distracted:{alert_name}", "Guard Distracted")
                elif activity_status == "INACTIVE":
                    send_alert(f"idle:{alert_name}", "Guard Idle")

            # ── Stats ──────────────────────────────────────────────────────────
            streamer.update_stats(
                guards_detected=len(active_guard_ids),
                active_violations=len(active_violations),
            )

            # ── HUD ────────────────────────────────────────────────────────────
            rx1, ry1, rx2, ry2 = POST_AREA
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 100, 0), 1)

            # Draw zone grid lines
            mid_x = (rx1 + rx2) // 2
            mid_y = (ry1 + ry2) // 2
            cv2.line(frame, (mid_x, ry1), (mid_x, ry2), (255,100,0), 1)
            cv2.line(frame, (rx1, mid_y), (rx2, mid_y), (255,100,0), 1)
            for lbl, pos in [("A",(rx1+8, ry1+18)), ("B",(mid_x+8, ry1+18)),
                              ("C",(rx1+8, mid_y+18)), ("D",(mid_x+8, mid_y+18))]:
                cv2.putText(frame, lbl, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1)

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
        beh_worker.stop()
        stream.release()
        print("[main_web] Loop ended.")


def _run_with_restart(source, source_label):
    """Wraps run() so a crash auto-restarts after 3 seconds."""
    while not _stop_event.is_set():
        run(source, source_label)
        if _stop_event.is_set():
            break
        print("[main_web] Restarting in 3s...")
        time.sleep(3)


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