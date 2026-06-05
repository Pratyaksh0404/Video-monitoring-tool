import os
import cv2
import time
import threading
import datetime

MAX_SNAPSHOTS  = 20     # keep last N snapshots on disk
SNAPSHOT_DIR   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "snapshots"
)

# Alert types that trigger a snapshot (HIGH severity)
SNAPSHOT_TRIGGERS = {
    "weapon detected", "unattended weapon", "fire detected",
    "camera tamper", "guard missing", "guard sleeping",
    "fight", "guard under attack", "unknown person sleeping",
    "phone usage", "guard smoking",
    "unknown person using phone",
}

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class SnapshotManager:

    def __init__(self):
        self._lock = threading.Lock()

    def should_snap(self, alert_type: str) -> bool:
        al = alert_type.lower()
        return any(t in al for t in SNAPSHOT_TRIGGERS)

    def save(self, frame, alert_type: str, guard_id: str = "", zone: str = ""):
        """Save an annotated snapshot. Non-blocking — runs in background thread."""
        if frame is None or frame.size == 0:
            return
        if not self.should_snap(alert_type):
            return
        threading.Thread(
            target=self._write,
            args=(frame.copy(), alert_type, guard_id, zone),
            daemon=True
        ).start()

    def _write(self, frame, alert_type: str, guard_id: str, zone: str):
        """Write frame to disk with annotation overlay."""
        try:
            now      = datetime.datetime.now()
            ts_file  = now.strftime("%Y%m%d_%H%M%S")
            ts_label = now.strftime("%H:%M:%S")

            # ── Annotation overlay ────────────────────────────────────────
            h, w = frame.shape[:2]

            # Red banner at top
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 200), -1)

            # Alert type text
            label = alert_type
            if guard_id and guard_id not in ("Camera", "Post", ""):
                label += f"  ·  {guard_id}"
            if zone and zone not in ("—", "-", ""):
                label += f"  ·  Zone {zone}"

            cv2.putText(frame, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Timestamp bottom-right
            cv2.putText(frame, ts_label, (w - 80, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                        cv2.LINE_AA)

            # ── Filename ──────────────────────────────────────────────────
            # Sanitize alert type for filename
            safe_type = alert_type.lower()
            safe_type = "".join(c if c.isalnum() else "_" for c in safe_type)
            safe_type = safe_type[:30]
            filename  = f"{ts_file}_{safe_type}.jpg"
            filepath  = os.path.join(SNAPSHOT_DIR, filename)

            with self._lock:
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                self._prune()

        except Exception as e:
            print(f"[SnapshotManager] Save failed: {e}")

    def _prune(self):
        """Keep only the last MAX_SNAPSHOTS files."""
        try:
            files = sorted([
                f for f in os.listdir(SNAPSHOT_DIR)
                if f.endswith(".jpg")
            ])
            while len(files) > MAX_SNAPSHOTS:
                os.remove(os.path.join(SNAPSHOT_DIR, files.pop(0)))
        except Exception:
            pass

    def list_snapshots(self) -> list:
        """Return list of snapshot metadata dicts, newest first."""
        try:
            files = sorted([
                f for f in os.listdir(SNAPSHOT_DIR)
                if f.endswith(".jpg")
            ], reverse=True)

            result = []
            for f in files:
                path = os.path.join(SNAPSHOT_DIR, f)
                try:
                    stat = os.stat(path)
                    # Parse filename: 20260603_143722_weapon_detected_gun.jpg
                    parts = f.replace(".jpg", "").split("_", 2)
                    label = parts[2].replace("_", " ").title() if len(parts) >= 3 else f
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
        """Return full path to a snapshot file (validated)."""
        # Security: strip path separators
        safe = os.path.basename(filename)
        if not safe.endswith(".jpg"):
            return None
        path = os.path.join(SNAPSHOT_DIR, safe)
        return path if os.path.exists(path) else None