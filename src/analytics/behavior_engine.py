import time
from collections import deque


class BehaviorEngine:
    WINDOW_SIZE    = 8      # rolling window of CLIP readings (was 6)
                            # At BEH_INTERVAL=50 frames, 5.5fps → ~9s per reading
                            # 8 readings = ~72s window
    CONFIRM_RATIO  = 0.60   # 60% of readings must agree → CONFIRMED
    POSSIBLE_RATIO = 0.35   # 35% of readings must agree → POSSIBLE

    # Per-label confirm ratio overrides
    _CONFIRM_RATIO_OVERRIDE = {
        "SMOKING":          0.85,
        "SLEEPING":         0.70,
        "PHONE_USE":        0.60,
        "IDLE":             0.60,
        "DISTRACTED_OTHER": 0.55,
    }

    # Bug found 2026-07: same disconnect as BehaviorClassifier._MIN_CONF
    # (see that file) — the Settings tab's "confirm ratio" sliders wrote
    # to rules_config.yaml's behavior.confirm_ratios and nothing ever
    # read it back; _CONFIRM_RATIO_OVERRIDE above was pure hardcoded
    # dead weight as far as the dashboard was concerned. Module-level
    # (not per-instance) so main_web.py's reload_config() can push a live
    # Settings save to every camera's engine at once, matching the
    # dashboard's "updated live" claim. An empty dict here means "use
    # _CONFIRM_RATIO_OVERRIDE above" — this is populated at startup from
    # config, not left empty in normal operation.
    _CONFIRM_RATIO_LIVE_OVERRIDE: dict = {}

    # Labels to ignore — these are handled by dedicated systems, not CLIP
    _IGNORED_LABELS = {"IDLE"}

    # Minimum readings in window before we confirm anything
    MIN_READINGS_FOR_CONFIRM  = 7
    MIN_READINGS_FOR_POSSIBLE = 4

    def __init__(self):
        # track_id → deque of (timestamp, label) readings
        self._windows: dict = {}

    def update(self, track_id, label) -> str:
        """
        Feed a new CLIP label reading and return the current state.
        Returns: ANALYZING | NORMAL | POSSIBLE_X | CONFIRMED_X
        """
        now = time.time()

        if track_id not in self._windows:
            self._windows[track_id] = deque(maxlen=self.WINDOW_SIZE)

        window = self._windows[track_id]

        # Add new reading (skip ANALYZING — not a real label)
        # Also skip labels handled by dedicated systems (e.g. IDLE)
        if label != "ANALYZING" and label not in self._IGNORED_LABELS:
            window.append(label)

        if not window:
            return "ANALYZING"

        # Count label frequencies in window
        counts = {}
        for lbl in window:
            counts[lbl] = counts.get(lbl, 0) + 1

        total = len(window)

        # Find dominant label (not NORMAL)
        best_label = None
        best_count = 0
        for lbl, cnt in counts.items():
            if lbl != "NORMAL" and cnt > best_count:
                best_count = cnt
                best_label = lbl

        # If NORMAL dominates → person is normal
        normal_count = counts.get("NORMAL", 0)
        if normal_count > best_count:
            return "NORMAL"

        if best_label is None:
            return "NORMAL"

        ratio = best_count / total
        confirm_ratio  = (self._CONFIRM_RATIO_LIVE_OVERRIDE.get(best_label)
                          or self._CONFIRM_RATIO_OVERRIDE.get(best_label, self.CONFIRM_RATIO))
        possible_ratio = self.POSSIBLE_RATIO

        if total >= self.MIN_READINGS_FOR_CONFIRM and ratio >= confirm_ratio:
            return f"CONFIRMED_{best_label}"
        elif total >= self.MIN_READINGS_FOR_POSSIBLE and ratio >= possible_ratio:
            return f"POSSIBLE_{best_label}"
        else:
            return "ANALYZING"

    def reset(self, track_id=None):
        if track_id is None:
            self._windows.clear()
        else:
            self._windows.pop(track_id, None)


def set_confirm_ratio_overrides(overrides: dict) -> None:
    """
    Called by main_web.py (at startup and from reload_config()) with
    whatever behavior.confirm_ratios the Settings tab has saved to
    rules_config.yaml — e.g. {"SLEEPING": 0.90, "PHONE_USE": 0.6, ...}.
    Takes effect immediately for every camera's next behavior update.
    """
    BehaviorEngine._CONFIRM_RATIO_LIVE_OVERRIDE = dict(overrides or {})