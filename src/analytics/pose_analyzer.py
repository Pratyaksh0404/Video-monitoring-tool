"""
pose_analyzer.py
────────────────
MediaPipe-based pose analysis for sleeping and smoking detection.

Replaces CLIP for:
  - SLEEPING: Measures head tilt angle using nose/ear/eye landmarks.
              Head tilted >40° downward + no movement for 30s = sleeping.
              Much more reliable than CLIP's "looks like sleeping" guess.

  - SMOKING:  Detects hand near mouth region. If a hand landmark is within
              the mouth bounding box for sustained time = possible smoking.
              This is a bridge until a dedicated YOLO cigarette model is added.

Keeps CLIP for:
  - PHONE_USE: CLIP works reasonably well here (phone-to-face is distinctive)

Dependencies:
  pip install mediapipe
"""

import threading
import queue
import time
import math
import numpy as np

_mp_available = False
try:
    import mediapipe as mp
    _mp_available = True
except ImportError:
    pass


class PoseAnalyzer:
    """
    Analyzes person crops using MediaPipe Pose + Face Mesh for:
    - Head tilt angle (sleeping detection)
    - Hand-near-mouth (smoking detection)

    Runs in a background thread to avoid blocking the main pipeline.
    Results are cached per track_id.
    """

    # Head tilt thresholds
    HEAD_TILT_SLEEPING = 35.0    # degrees — head tilted down more than this
    HEAD_TILT_LOOKING_DOWN = 20.0  # degrees — just looking down (phone, etc.)

    # Hand-near-mouth detection
    MOUTH_HAND_DIST_RATIO = 0.15  # fraction of crop height — hand must be this close to mouth

    def __init__(self):
        self._q       = queue.Queue(maxsize=4)
        self._results = {}
        self._lock    = threading.Lock()
        self._stopped = False
        self._ready   = False

        self._pose     = None
        self._face     = None

        if _mp_available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            print("[PoseAnalyzer] mediapipe not installed. "
                  "Install with: pip install mediapipe")
            print("[PoseAnalyzer] Falling back to CLIP-only mode.")

    @property
    def ready(self):
        return self._ready

    def submit(self, track_id, frame_crop):
        """Submit a person crop for pose analysis."""
        if not self._ready or frame_crop is None or frame_crop.size == 0:
            return
        try:
            self._q.put_nowait((track_id, frame_crop.copy()))
        except queue.Full:
            pass

    def get_result(self, track_id):
        """Get the latest pose analysis result for a track."""
        with self._lock:
            return self._results.get(track_id)

    def clear(self, track_id):
        """Clear result for a dead track."""
        with self._lock:
            self._results.pop(track_id, None)

    def stop(self):
        self._stopped = True

    def _loop(self):
        """Background worker thread."""
        try:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,    # fastest model
                min_detection_confidence=0.5,
            )
            self._face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
            )
            self._ready = True
            print("[PoseAnalyzer] MediaPipe ready (pose + face mesh).")
        except Exception as e:
            print(f"[PoseAnalyzer] Init failed: {e}")
            return

        while not self._stopped:
            try:
                track_id, crop = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                result = self._analyze(crop)
            except Exception as e:
                result = self._empty_result()

            with self._lock:
                self._results[track_id] = result

    def _empty_result(self):
        return {
            "head_tilt": 0.0,
            "head_down": False,
            "head_sleeping": False,
            "hand_near_mouth": False,
            "pose_detected": False,
            "face_detected": False,
            "timestamp": time.time(),
        }

    def _analyze(self, crop):
        """Run MediaPipe on a person crop and extract head tilt + hand position."""
        import cv2
        h, w = crop.shape[:2]
        if h < 40 or w < 20:
            return self._empty_result()

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = self._empty_result()

        # ── Face Mesh — head tilt ─────────────────────────────────────────
        face_out = self._face.process(rgb)
        if face_out.multi_face_landmarks:
            result["face_detected"] = True
            landmarks = face_out.multi_face_landmarks[0].landmark

            # Use nose tip (1), left ear (234), right ear (454),
            # forehead (10), chin (152) to compute head tilt
            nose  = landmarks[1]
            chin  = landmarks[152]
            forehead = landmarks[10]

            # Vertical tilt: angle between forehead-chin line and vertical
            # If chin is significantly lower than nose relative to the
            # forehead-chin distance, head is tilted forward (sleeping)
            dy = chin.y - forehead.y
            dx = chin.x - forehead.x

            if dy > 0.01:
                # Compute tilt angle from vertical
                # nose.y relative to forehead-chin line indicates forward tilt
                nose_rel = (nose.y - forehead.y) / dy  # 0=at forehead, 1=at chin
                # In neutral pose, nose is at ~0.4-0.5 of forehead-chin
                # When head tilts forward, nose moves up relative to this line
                # We use the nose-z (depth) and y-position for tilt estimation

                # Simpler approach: angle of nose relative to eye line
                left_eye  = landmarks[33]   # left eye inner corner
                right_eye = landmarks[263]  # right eye inner corner
                eye_mid_y = (left_eye.y + right_eye.y) / 2

                # When looking straight: nose.y > eye_mid_y by ~0.05-0.1
                # When head tilts down: nose.y >> eye_mid_y
                # When head tilts up: nose.y ~ eye_mid_y
                nose_drop = (nose.y - eye_mid_y) * h  # in pixels

                # Approximate tilt angle
                face_height = dy * h
                if face_height > 10:
                    tilt_ratio = nose_drop / face_height
                    # Map ratio to approximate degrees
                    # Neutral: ratio ~0.3-0.4 → 0°
                    # Looking down: ratio >0.5 → 20-40°
                    # Sleeping: ratio >0.7 → 40°+
                    neutral_ratio = 0.38
                    tilt_deg = max(0, (tilt_ratio - neutral_ratio) * 120)
                    result["head_tilt"] = round(tilt_deg, 1)
                    result["head_down"] = tilt_deg > self.HEAD_TILT_LOOKING_DOWN
                    result["head_sleeping"] = tilt_deg > self.HEAD_TILT_SLEEPING

            # ── Hand near mouth ───────────────────────────────────────────
            # Mouth center from face landmarks
            mouth_top    = landmarks[13]   # upper lip
            mouth_bottom = landmarks[14]   # lower lip
            mouth_cx = (mouth_top.x + mouth_bottom.x) / 2 * w
            mouth_cy = (mouth_top.y + mouth_bottom.y) / 2 * h
            threshold = h * self.MOUTH_HAND_DIST_RATIO

        # ── Pose — hand positions ─────────────────────────────────────────
        pose_out = self._pose.process(rgb)
        if pose_out.pose_landmarks:
            result["pose_detected"] = True
            plm = pose_out.pose_landmarks.landmark

            # Wrist landmarks: 15 (left wrist), 16 (right wrist)
            # Index finger tip: 19 (left), 20 (right)
            hand_points = []
            for idx in [15, 16, 19, 20]:
                lm = plm[idx]
                if lm.visibility > 0.5:
                    hand_points.append((lm.x * w, lm.y * h))

            # Check if any hand point is near the mouth region
            if face_out.multi_face_landmarks and hand_points:
                for (hx, hy) in hand_points:
                    dist = math.sqrt((hx - mouth_cx)**2 + (hy - mouth_cy)**2)
                    if dist < threshold:
                        result["hand_near_mouth"] = True
                        break

        result["timestamp"] = time.time()
        return result
