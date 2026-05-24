import time
import math


class UnknownTracker:
    """
    Assigns stable 'Unknown_N' labels to unrecognized persons.

    Slot reuse logic:
    - If new centroid is within SPATIAL_TOLERANCE px of a recently-seen
      slot's last centroid AND the slot died within TIME_TOLERANCE seconds,
      reuse that slot number.
    - Otherwise assign a new slot, recycling numbers above MAX_SLOT_NUMBER.
    """

    SPATIAL_TOLERANCE   = 200    # px — slot reuse proximity
    TIME_TOLERANCE      = 30.0   # seconds — slot memory after track dies
    ALERT_SUPPRESS_SECS = 60.0   # seconds — minimum gap between alerts per slot
    MAX_SLOT_NUMBER     = 10     # slot numbers never exceed this

    def __init__(self):
        # track_id → slot name e.g. "Unknown_3"
        self._track_to_slot: dict = {}
        # slot_name → { centroid, died_at, last_alert_at }
        self._slot_memory: dict = {}
        # slot_name → track_id currently using it (None if free)
        self._active_slots: dict = {}
        # counter for next slot number (wraps at MAX_SLOT_NUMBER)
        self._next_slot = 1

    # ── Public API ────────────────────────────────────────────────────────

    def get_or_assign(self, track_id: int, centroid: tuple) -> str:
        """
        Return existing slot for track_id, or assign a new one.
        Reuses nearby recently-dead slots when possible.
        """
        if track_id in self._track_to_slot:
            return self._track_to_slot[track_id]

        # Try to reuse a dead slot at similar location
        slot = self._find_reusable_slot(centroid)

        if slot is None:
            slot = self._allocate_slot()

        self._track_to_slot[track_id] = slot
        self._active_slots[slot] = track_id

        # Initialize slot memory if new
        if slot not in self._slot_memory:
            self._slot_memory[slot] = {
                "centroid":      centroid,
                "died_at":       0.0,
                "last_alert_at": 0.0,
            }

        return slot

    def get_slot(self, track_id: int):
        """Return slot name for track, or None if not assigned."""
        return self._track_to_slot.get(track_id)

    def on_track_lost(self, track_id: int, last_centroid=None):
        """Call when a track is deregistered."""
        slot = self._track_to_slot.pop(track_id, None)
        if slot:
            self._active_slots.pop(slot, None)
            if slot in self._slot_memory:
                self._slot_memory[slot]["died_at"] = time.time()
                if last_centroid:
                    self._slot_memory[slot]["centroid"] = last_centroid

    def can_alert(self, slot: str) -> bool:
        """Return True if enough time has passed since last alert for this slot."""
        mem = self._slot_memory.get(slot)
        if not mem:
            return True
        return (time.time() - mem["last_alert_at"]) >= self.ALERT_SUPPRESS_SECS

    def mark_alerted(self, slot: str):
        """Record that an alert just fired for this slot."""
        if slot in self._slot_memory:
            self._slot_memory[slot]["last_alert_at"] = time.time()

    def reset(self):
        """Clear all state."""
        self._track_to_slot.clear()
        self._slot_memory.clear()
        self._active_slots.clear()
        self._next_slot = 1

    # ── Internal ──────────────────────────────────────────────────────────

    def _find_reusable_slot(self, centroid: tuple):
        """
        Look for a recently-dead slot whose last centroid is within
        SPATIAL_TOLERANCE pixels and died within TIME_TOLERANCE seconds.
        Returns slot name or None.
        """
        now = time.time()
        best_slot = None
        best_dist = float("inf")

        for slot, mem in self._slot_memory.items():
            # Skip active slots
            if self._active_slots.get(slot) is not None:
                continue
            # Skip if too old
            if mem["died_at"] == 0.0:
                continue
            if (now - mem["died_at"]) > self.TIME_TOLERANCE:
                continue
            # Check proximity
            sc = mem.get("centroid")
            if sc is None:
                continue
            dist = math.dist(centroid, sc)
            if dist < self.SPATIAL_TOLERANCE and dist < best_dist:
                best_dist = dist
                best_slot = slot

        return best_slot

    def _allocate_slot(self) -> str:
        """
        Allocate a slot number, recycling if we've exceeded MAX_SLOT_NUMBER.
        Strategy: try numbers 1..MAX_SLOT_NUMBER in order, skip any that
        are currently active. If all active, evict the one with oldest
        last_alert_at (least recently used).
        """
        # Try to find a free slot number (not currently active)
        for _ in range(self.MAX_SLOT_NUMBER):
            candidate = f"Unknown_{self._next_slot}"
            self._next_slot = (self._next_slot % self.MAX_SLOT_NUMBER) + 1

            if self._active_slots.get(candidate) is None:
                return candidate

        # All slots active — evict the one with oldest last_alert_at
        oldest_slot = None
        oldest_time = float("inf")
        for slot, track_id in list(self._active_slots.items()):
            mem = self._slot_memory.get(slot, {})
            lat = mem.get("last_alert_at", 0.0)
            if lat < oldest_time:
                oldest_time = lat
                oldest_slot = slot

        if oldest_slot:
            # Evict: remove the old track→slot mapping
            old_track = self._active_slots.pop(oldest_slot, None)
            if old_track is not None:
                self._track_to_slot.pop(old_track, None)
            return oldest_slot

        # Fallback (should never happen)
        return f"Unknown_{self._next_slot}"