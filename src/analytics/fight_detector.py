"""
fight_detector.py
─────────────────
Detects physical altercations using motion magnitude + spatial proximity.

Confirmed fight requires ALL conditions simultaneously:
  1. PROXIMITY  — centroid distance ≤ PROXIMITY_FACTOR × avg box height
  2. BOTH MOVING FAST — each person ≥ MOTION_THRESHOLD px/frame
  3. SUSTAINED  — conditions 1+2 hold for CONFIRM_SECONDS continuously

Fixes in this version
─────────────────────
- When suppress=True (crowd mode), _fight_since is CLEARED entirely.
  Previously, a fight countdown could reach 3.5s while persons were
  entering a crowd, then fire the moment crowd threshold was hit (the
  suppress flag was set too late). Now crowd wipes the fight history.

- MOTION_THRESHOLD raised 15 → 20 to reduce false triggers at 2.5fps
  where camera jitter can produce apparent motion.

- CONFIRM_SECONDS raised 3.5 → 5.0 seconds for more certainty.
"""

import time
from collections import deque


class FightDetector:
    PROXIMITY_FACTOR = 0.8    # × avg box height
    MOTION_THRESHOLD = 20.0   # px/frame — raised from 15
    CONFIRM_SECONDS  = 5.0    # seconds — raised from 3.5
    HISTORY_FRAMES   = 18     # rolling centroid history window

    def __init__(self):
        self._history: dict     = {}   # track_id → deque of (x, y)
        self._fight_since: dict = {}   # frozenset → timestamp

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tracked_objects: dict, suppress: bool = False) -> list:
        """
        Call once per frame with {track_id: (x1,y1,x2,y2)}.

        suppress: if True (e.g. during crowd), clears all fight history and
                  returns empty list. This prevents a fight that was building
                  up before crowd threshold from firing when crowd is detected.

        Returns list of (id1, id2) confirmed fight pairs.
        """
        self._update_history(tracked_objects)

        if suppress:
            # FIX: clear fight history so stale countdowns don't fire later
            self._fight_since.clear()
            return []

        ids  = list(tracked_objects.keys())
        now  = time.time()
        active_pairs    = set()
        confirmed_pairs = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                b1  = tracked_objects[id1]
                b2  = tracked_objects[id2]

                if self._is_close(b1, b2) and self._both_moving_fast(id1, id2):
                    pair = frozenset({id1, id2})
                    active_pairs.add(pair)

                    if pair not in self._fight_since:
                        self._fight_since[pair] = now
                    elif now - self._fight_since[pair] >= self.CONFIRM_SECONDS:
                        confirmed_pairs.append((id1, id2))

        # Clean up pairs no longer active
        for pair in list(self._fight_since):
            if pair not in active_pairs:
                del self._fight_since[pair]

        return confirmed_pairs

    def speed(self, track_id: int) -> float:
        """Average frame-to-frame speed (px/frame) for this track."""
        return self._speed(track_id)

    def reset(self, track_id=None):
        if track_id is None:
            self._history.clear()
            self._fight_since.clear()
        else:
            self._history.pop(track_id, None)
            for pair in list(self._fight_since):
                if track_id in pair:
                    del self._fight_since[pair]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _centroid(self, box):
        return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

    def _box_height(self, box):
        return max(1, box[3]-box[1])

    def _update_history(self, tracked_objects):
        seen = set(tracked_objects.keys())
        for tid, box in tracked_objects.items():
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.HISTORY_FRAMES)
            self._history[tid].append(self._centroid(box))
        for dead in list(self._history):
            if dead not in seen:
                del self._history[dead]

    def _speed(self, track_id) -> float:
        hist = self._history.get(track_id)
        if not hist or len(hist) < 5:
            return 0.0
        pts   = list(hist)
        total = sum(
            ((pts[k][0]-pts[k-1][0])**2 + (pts[k][1]-pts[k-1][1])**2)**0.5
            for k in range(1, len(pts))
        )
        return total / (len(pts)-1)

    def _is_close(self, b1, b2) -> bool:
        cx1, cy1 = self._centroid(b1)
        cx2, cy2 = self._centroid(b2)
        dist  = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
        avg_h = (self._box_height(b1) + self._box_height(b2)) / 2.0
        return dist <= self.PROXIMITY_FACTOR * avg_h

    def _both_moving_fast(self, id1, id2) -> bool:
        return (self._speed(id1) >= self.MOTION_THRESHOLD and
                self._speed(id2) >= self.MOTION_THRESHOLD)