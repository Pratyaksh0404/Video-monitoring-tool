"""
loitering_detector.py
─────────────────────
Detects suspicious loitering — a person who has been moving around
but keeps lingering in the same zone.

Key distinction
───────────────
  LOITERING  = person has been MOVING (total travel > MIN_MOVEMENT_PX)
               AND has stayed in the same zone beyond the time threshold.

  STATIONARY = person has barely moved at all.
               → Not flagged as loitering. A person who walks in and
                 stands still is not "loitering" — they may be a guard
                 at their post, a visitor, or someone being interviewed.

Thresholds
──────────
  Known guard  : KNOWN_THRESHOLD    = 90 s
  Unknown       : UNKNOWN_THRESHOLD  = 30 s
  Min movement  : MIN_MOVEMENT_PX    = 80 px total distance traveled

Usage
─────
    loitering = LoiteringDetector()

    # Each frame — also pass the person's current speed from FightDetector:
    is_loitering = loitering.update(
        track_id, zone, is_known=False, current_speed=fight_det.speed(track_id))

    secs = loitering.time_in_zone(track_id)

    # When track disappears:
    loitering.reset(track_id)
"""

import time


class LoiteringDetector:
    KNOWN_THRESHOLD   = 90    # seconds — enrolled guard
    UNKNOWN_THRESHOLD = 30    # seconds — unrecognised person
    MIN_MOVEMENT_PX   = 80    # total px traveled required to flag loitering

    def __init__(self):
        # track_id → {"zone": str, "since": float, "total_dist": float}
        self._state: dict = {}

    def update(self, track_id, zone: str, is_known: bool = True,
               current_speed: float = 0.0) -> bool:
        """
        Returns True when the loitering threshold is exceeded.

        current_speed: px/frame from FightDetector.speed(track_id).
                       Used to accumulate total movement distance.
        """
        if not zone or zone in ("?", "—", "-", "None", ""):
            return False

        now       = time.time()
        threshold = self.KNOWN_THRESHOLD if is_known else self.UNKNOWN_THRESHOLD
        state     = self._state.get(track_id)

        if state is None:
            # First sighting
            self._state[track_id] = {
                "zone":       zone,
                "since":      now,
                "total_dist": 0.0,
            }
            return False

        if state["zone"] != zone:
            # Zone changed — reset timer but keep accumulated distance
            # (the person is still lingering around this area)
            state["zone"]  = zone
            state["since"] = now
            return False

        # Same zone — accumulate movement (speed × 1 frame ≈ pixels this frame)
        state["total_dist"] += current_speed

        # Only flag loitering if person has actually been moving around
        time_exceeded     = (now - state["since"]) >= threshold
        movement_exceeded = state["total_dist"] >= self.MIN_MOVEMENT_PX

        return time_exceeded and movement_exceeded

    def time_in_zone(self, track_id) -> float:
        state = self._state.get(track_id)
        if state is None:
            return 0.0
        return time.time() - state["since"]

    def total_movement(self, track_id) -> float:
        """Total pixels traveled by this track (across all zones)."""
        state = self._state.get(track_id)
        return state["total_dist"] if state else 0.0

    def reset(self, track_id):
        self._state.pop(track_id, None)

    def reset_all(self):
        self._state.clear()