"""
unknown_tracker.py
──────────────────
Gives unknown persons a STABLE persistent label across track_id reassignments.

Fixes in this version
─────────────────────
- TIME_TOLERANCE raised 12s → 30s  (at 2.5fps, track churn is slow)
- SPATIAL_TOLERANCE raised 150px → 200px  (person can drift between reassignments)
- Per-slot alert_fired tracking: once a slot fires an "Unknown Person Detected"
  alert, it won't fire again for ALERT_SUPPRESS_SECS (60s) even if track dies
  and a new track_id picks up the same slot. This is what stops the flood.
- on_track_lost now always records last_centroid even if not provided (uses
  last known position from get_or_assign calls)
"""

import time


class UnknownTracker:
    SPATIAL_TOLERANCE    = 200    # px — how close centroids must be to re-link
    TIME_TOLERANCE       = 30.0   # seconds — how long we remember a lost slot
    ALERT_SUPPRESS_SECS  = 60.0   # seconds — minimum gap between alerts per slot

    def __init__(self):
        # track_id → slot_name  (active tracks)
        self._active: dict = {}

        # slot_name → {"centroid": (x,y), "lost_at": float}  (recently lost)
        self._lost: dict = {}

        # slot_name → last time alert was fired for this slot
        self._alerted: dict = {}

        # track_id → last known centroid (so on_track_lost has position)
        self._last_centroid: dict = {}

        self._next_slot = 1   # monotonic counter for new slot IDs

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_assign(self, track_id: int, centroid: tuple) -> str:
        """
        Return the stable slot name for this unknown track.
        Creates a new slot or reuses a recently-lost nearby slot.
        """
        # Track the latest centroid for this track_id
        self._last_centroid[track_id] = centroid

        # Already assigned this frame
        if track_id in self._active:
            return self._active[track_id]

        # Try to match to a recently-lost slot by spatial proximity
        slot = self._find_nearby_slot(centroid)

        if slot is None:
            slot = f"Unknown_{self._next_slot}"
            self._next_slot += 1

        self._active[track_id] = slot
        self._lost.pop(slot, None)   # slot is active again, remove from lost
        return slot

    def on_track_lost(self, track_id: int, last_centroid: tuple = None):
        """Call when CentroidTracker deregisters a track."""
        slot = self._active.pop(track_id, None)
        # Use provided centroid, or fall back to last recorded centroid
        pos = last_centroid or self._last_centroid.pop(track_id, None)
        if slot and pos:
            self._lost[slot] = {
                "centroid": pos,
                "lost_at":  time.time(),
            }
        elif track_id in self._last_centroid:
            self._last_centroid.pop(track_id, None)

    def can_alert(self, slot_name: str) -> bool:
        """
        Returns True if this slot is allowed to fire an alert now.
        Call this before firing 'Unknown Person Detected'.
        If True, you must call mark_alerted() to register the alert.
        """
        last = self._alerted.get(slot_name, 0)
        return (time.time() - last) >= self.ALERT_SUPPRESS_SECS

    def mark_alerted(self, slot_name: str):
        """Record that an alert was just fired for this slot."""
        self._alerted[slot_name] = time.time()

    def reset(self):
        """Clear all state — call on source switch."""
        self._active.clear()
        self._lost.clear()
        self._alerted.clear()
        self._last_centroid.clear()
        # Keep _next_slot to avoid reusing old IDs

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_nearby_slot(self, centroid: tuple):
        """
        Return the name of the nearest recently-lost slot within tolerance,
        or None if none found. Also prunes expired slots.
        """
        now      = time.time()
        best     = None
        best_dist = float("inf")

        expired = []
        for slot, info in self._lost.items():
            age = now - info["lost_at"]
            if age > self.TIME_TOLERANCE:
                expired.append(slot)
                continue

            cx, cy = centroid
            lx, ly = info["centroid"]
            dist = ((cx-lx)**2 + (cy-ly)**2) ** 0.5

            if dist < self.SPATIAL_TOLERANCE and dist < best_dist:
                best_dist = dist
                best      = slot

        for s in expired:
            self._lost.pop(s, None)

        return best

    def get_slot(self, track_id: int):
        """Return current slot for a track, or None if not assigned."""
        return self._active.get(track_id)