"""
snapshot_manager.py
────────────────────
3-burst snapshots per alert, session-isolated folders.

Burst: frame at t=0 (immediate), t+1s, t+2s from live stream.
Each session gets its own folder: src/snapshots/session_YYYYMMDD_HHMMSS/
"""

import os
import cv2
import time
import threading
import datetime

MAX_PER_SESSION = 200
SNAPSHOT_BASE   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src", "snapshots"
)

SNAPSHOT_TRIGGERS = {
    "weapon detected", "unattended weapon", "fire detected",
    "camera tamper", "guard missing", "guard sleeping",
    "fight", "guard absent", "crowd detected",
    "phone usage", "guard smoking",
}

os.makedirs(SNAPSHOT_BASE, exist_ok=True)


class SnapshotManager:

    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_fn    = None   # callable → latest BGR frame
        self._session_dir = None

    def new_session(self):
        """Create a fresh session folder. Call at pipeline start."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(SNAPSHOT_BASE, f"session_{ts}")
        os.makedirs(self._session_dir, exist_ok=True)

    def set_frame_source(self, fn):
        """Register callable that returns latest BGR frame."""
        self._frame_fn = fn

    def should_snap(self, alert_type: str) -> bool:
        al = alert_type.lower()
        return any(t in al for t in SNAPSHOT_TRIGGERS)

    def save(self, frame, alert_type: str, guard_id: str = "", zone: str = ""):
        """
        Save 3-burst snapshot asynchronously (non-blocking).
        Frame 1: provided frame (t=0)
        Frame 2: live frame at t+1s
        Frame 3: live frame at t+2s
        """
        if not self.should_snap(alert_type):
            return
        if frame is None or frame.size == 0:
            return
        if self._session_dir is None:
            self.new_session()

        threading.Thread(
            target=self._do_burst,
            args=(frame.copy(), alert_type, guard_id, zone,
                  datetime.datetime.now()),
            daemon=True
        ).start()

    def save_and_get_paths(self, frame, alert_type: str,
                           guard_id: str = "", zone: str = "") -> list:
        """
        Save 3-burst synchronously, return file paths for email attachment.
        Grabs frame 1 immediately, frames 2+3 from live stream after 1s/2s.
        """
        if not self.should_snap(alert_type):
            return []
        if frame is None or frame.size == 0:
            return []
        if self._session_dir is None:
            self.new_session()

        ts    = datetime.datetime.now()
        paths = []

        for i, gap in enumerate([0, 1.0, 2.0], 1):
            if gap > 0:
                time.sleep(1.0)   # sleep 1s between each frame
            src = frame if i == 1 else self._live_frame(frame)
            p   = self._write(src, alert_type, guard_id, zone, ts, i)
            if p:
                paths.append(p)

        return paths

    def _do_burst(self, f0, alert_type, guard_id, zone, ts):
        """Background burst writer: t=0, t+1s, t+2s."""
        self._write(f0, alert_type, guard_id, zone, ts, 1)
        time.sleep(1.0)
        self._write(self._live_frame(f0), alert_type, guard_id, zone, ts, 2)
        time.sleep(1.0)
        self._write(self._live_frame(f0), alert_type, guard_id, zone, ts, 3)

    def _live_frame(self, fallback):
        """Get latest frame from live stream, or use fallback."""
        if self._frame_fn:
            try:
                f = self._frame_fn()
                if f is not None and f.size > 0:
                    return f.copy()
            except Exception:
                pass
        return fallback.copy()

    def _write(self, frame, alert_type, guard_id, zone, ts, num) -> str:
        """Annotate and write one burst frame. Returns filepath."""
        try:
            h, w  = frame.shape[:2]
            out   = frame.copy()

            # Red banner
            cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 180), -1)
            label = alert_type
            if guard_id and guard_id not in ("Camera", "Post", ""):
                label += f"  ·  {guard_id}"
            if zone and zone not in ("—", "-", ""):
                label += f"  ·  Zone {zone}"
            cv2.putText(out, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Timestamp + burst indicator bottom-right
            cap_ts = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(out, f"{cap_ts}  [{num}/3]",
                        (w - 120, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)

            safe = "".join(c if c.isalnum() else "_"
                           for c in alert_type.lower())[:28]
            filename = f"{ts.strftime('%H%M%S')}_{safe}_{num}.jpg"
            filepath = os.path.join(self._session_dir, filename)

            with self._lock:
                cv2.imwrite(filepath, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
                self._prune()

            return filepath
        except Exception as e:
            print(f"[SnapshotManager] Write error: {e}")
            return ""

    def _prune(self):
        try:
            files = sorted(f for f in os.listdir(self._session_dir)
                           if f.endswith(".jpg"))
            while len(files) > MAX_PER_SESSION:
                os.remove(os.path.join(self._session_dir, files.pop(0)))
        except Exception:
            pass

    def get_recent_paths(self, alert_type: str, max_age_secs: float = 10.0) -> list:
        """
        Return file paths of the most recent burst for a given alert type.
        Used by email_alerter to attach snapshots taken in the last N seconds.
        """
        if not self._session_dir:
            return []
        safe = "".join(c if c.isalnum() else "_"
                       for c in alert_type.lower())[:28]
        now = time.time()
        try:
            matches = []
            for f in os.listdir(self._session_dir):
                if not f.endswith(".jpg"):
                    continue
                if safe not in f:
                    continue
                path = os.path.join(self._session_dir, f)
                if (now - os.path.getmtime(path)) <= max_age_secs:
                    matches.append(path)
            return sorted(matches)[:3]
        except Exception:
            return []

    def list_snapshots(self) -> list:
        if not self._session_dir:
            return []
        try:
            files = sorted(
                (f for f in os.listdir(self._session_dir)
                 if f.endswith(".jpg")),
                reverse=True
            )
            result = []
            for f in files:
                path = os.path.join(self._session_dir, f)
                try:
                    stat  = os.stat(path)
                    label = f.replace(".jpg", "")[7:].rsplit("_", 1)[0].replace("_", " ").title()
                    result.append({
                        "filename": f,
                        "label":    label,
                        "size_kb":  round(stat.st_size / 1024, 1),
                        "mtime":    stat.st_mtime,
                        "ts":       datetime.datetime.fromtimestamp(
                                        stat.st_mtime).strftime("%H:%M:%S"),
                    })
                except Exception:
                    pass
            return result
        except Exception:
            return []

    def get_path(self, filename: str) -> str:
        if not self._session_dir:
            return None
        safe = os.path.basename(filename)
        if not safe.endswith(".jpg"):
            return None
        path = os.path.join(self._session_dir, safe)
        return path if os.path.exists(path) else None


# Module-level singleton — shared across main_web and flask_app
snap_mgr = SnapshotManager()