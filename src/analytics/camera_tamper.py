import cv2
import numpy as np
import time


# ── Thresholds ──────────────────────────────────────────────────────────────
BLACKOUT_THRESHOLD    = 15      # mean pixel brightness below this = blackout
BLUR_THRESHOLD        = 40.0    # Laplacian variance below this = blurry/obstructed
SCENE_DIFF_THRESHOLD  = 55.0    # mean absolute pixel diff above this = scene changed
CONFIRM_SECS          = 3.0     # seconds condition must persist before alert fires
COOLDOWN_SECS         = 60.0    # seconds between repeated tamper alerts
SCENE_HISTORY_FRAMES  = 5       # compare against frame from N frames ago (not prev frame)
                                 # avoids false triggers from fast movement


class CameraTamper:
    """
    Stateful per-camera tampering detector.
    Call update(frame) every frame. Returns tamper type string or None.
    """

    def __init__(self):
        self._blackout_since     = 0.0
        self._blur_since         = 0.0
        self._scene_change_since = 0.0
        self._last_alert_at      = 0.0
        self._last_alert_type    = None

        # Ring buffer of recent grayscale frames for scene change comparison
        self._frame_history = []

    def update(self, frame) -> str | None:
        """
        Analyze a BGR frame for tampering.
        Returns: "Blackout" | "Obstruction" | "Camera Moved" | None
        """
        now  = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Blackout detection ─────────────────────────────────────────
        mean_brightness = float(np.mean(gray))
        if mean_brightness < BLACKOUT_THRESHOLD:
            if self._blackout_since == 0.0:
                self._blackout_since = now
            if (now - self._blackout_since) >= CONFIRM_SECS:
                return self._fire("Blackout", now)
        else:
            self._blackout_since = 0.0

        # ── 2. Blur / obstruction detection ──────────────────────────────
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < BLUR_THRESHOLD:
            if self._blur_since == 0.0:
                self._blur_since = now
            if (now - self._blur_since) >= CONFIRM_SECS:
                return self._fire("Obstruction", now)
        else:
            self._blur_since = 0.0

        # ── 3. Scene change / camera moved detection ──────────────────────
        # Downsample for speed
        small = cv2.resize(gray, (80, 60))
        self._frame_history.append(small)
        if len(self._frame_history) > SCENE_HISTORY_FRAMES:
            self._frame_history.pop(0)

        if len(self._frame_history) == SCENE_HISTORY_FRAMES:
            diff = float(np.mean(
                np.abs(small.astype(np.float32) -
                       self._frame_history[0].astype(np.float32))
            ))
            if diff > SCENE_DIFF_THRESHOLD:
                if self._scene_change_since == 0.0:
                    self._scene_change_since = now
                if (now - self._scene_change_since) >= CONFIRM_SECS:
                    return self._fire("Camera Moved", now)
            else:
                self._scene_change_since = 0.0

        return None

    def _fire(self, tamper_type: str, now: float) -> str | None:
        """Fire alert if cooldown has passed."""
        if (now - self._last_alert_at) < COOLDOWN_SECS:
            return None   # still in cooldown
        self._last_alert_at   = now
        self._last_alert_type = tamper_type
        return tamper_type

    def reset(self):
        self._blackout_since     = 0.0
        self._blur_since         = 0.0
        self._scene_change_since = 0.0
        self._last_alert_at      = 0.0
        self._frame_history      = []