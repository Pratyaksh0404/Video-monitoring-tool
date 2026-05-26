import cv2
import numpy as np
import time

STILL_SECS = 4.0        # person must be still this long before check starts
CONFIRM_SECS = 12.0     # hand-near-mouth must persist this long to confirm
MOVE_THRESHOLD = 15.0   # px — centroid shift that resets the still timer
SKIN_MIN_PIXELS = 80
MOUTH_REGION_TOP = 0.15
MOUTH_REGION_BOT = 0.45
MOUTH_REGION_L = 0.25
MOUTH_REGION_R = 0.75


class SmokingDetector:

    def __init__(self):
        # keyed on identity NAME, not track_id
        self._states = {}

    def _get_state(self, name):
        if name not in self._states:
            self._states[name] = {
                "still_since": 0.0,
                "hand_since": 0.0,
            }
        return self._states[name]

    def reset(self, name=None):
        """Reset state for a specific identity, or all if name is None."""
        if name is None:
            self._states.clear()
        else:
            self._states.pop(name, None)

    def reset_track(self, track_id):
        """Called when a track dies — does NOT reset state (state is identity-based)."""
        pass  # intentionally no-op: state persists across track re-registrations

    def update_by_name(self, name, frame, person_box, person_speed=0.0, clip_label=None):
        """
        Update smoking state keyed on identity name.
        Returns: 'CONFIRMED_SMOKING', 'POSSIBLE_SMOKING', or 'NORMAL'
        """
        now = time.time()
        state = self._get_state(name)

        # Phone use gate — hands near face but it's a phone, not a cigarette
        if clip_label == "PHONE_USE":
            state["hand_since"] = 0.0
            return "NORMAL"

        # Must be still
        if person_speed >= MOVE_THRESHOLD:
            state["still_since"] = 0.0
            state["hand_since"] = 0.0
            return "NORMAL"

        if state["still_since"] == 0.0:
            state["still_since"] = now

        if (now - state["still_since"]) < STILL_SECS:
            return "NORMAL"

        # Check hand near mouth
        if not self._hand_near_mouth(frame, person_box):
            state["hand_since"] = 0.0
            return "NORMAL"

        # Hand is near mouth and person is still
        if state["hand_since"] == 0.0:
            state["hand_since"] = now

        sustained = now - state["hand_since"]
        if sustained >= CONFIRM_SECS:
            return "CONFIRMED_SMOKING"
        elif sustained >= CONFIRM_SECS * 0.4:
            return "POSSIBLE_SMOKING"
        return "NORMAL"

    def _hand_near_mouth(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 15:
            return False

        h, w = crop.shape[:2]
        r_top = int(h * MOUTH_REGION_TOP)
        r_bot = int(h * MOUTH_REGION_BOT)
        r_l = int(w * MOUTH_REGION_L)
        r_r = int(w * MOUTH_REGION_R)

        region = crop[r_top:r_bot, r_l:r_r]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 20, 70], dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 20, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return cv2.countNonZero(mask) >= SKIN_MIN_PIXELS