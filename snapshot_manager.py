"""
snapshot_manager.py
────────────────────
3-burst snapshots per alert, session-isolated folders.

Burst: frame at t=0 (immediate), t+1s, t+2s from live stream.
Each session gets its own folder: src/snapshots/session_YYYYMMDD_HHMMSS/

v2 — camera + profile aware:
  - Every filename now embeds WHICH CAMERA produced it, using "__" as a
    delimiter (camera_ids like "cam_bank_lobby" already contain single
    underscores, so "__" is the only reliable, unambiguous separator).
    This is what makes it possible to filter "only this camera's
    snapshots" in the analytics tab — previously camera identity wasn't
    recorded anywhere, so filtering was impossible no matter what.
  - The on-image banner and filename both use PROFILE-APPROPRIATE
    wording (e.g. "Worker Sleeping" not "Guard Sleeping" under
    warehouse_ops) via the same translation alert_manager.py applies.

IMPORTANT threading note (verified empirically, not assumed): Python's
contextvars do NOT propagate into a plain threading.Thread — a background
thread sees only DEFAULT values, not whatever was set in the thread that
spawned it. save() spawns a background thread for the burst writer, so
camera_id/profile_id are captured HERE, in the calling (correct) thread,
and passed through explicitly as arguments — never re-read via
get_camera_context()/get_profile_context() inside the background thread,
which would silently give the wrong (default) values.
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

# Max length of the sanitized alert-type "slug" embedded in each filename.
#
# Bug found 2026-07 (handoff point #3, weapon-alert missing-snapshot
# investigation): this used to be a bare "[:28]" hardcoded independently
# in BOTH _write() and get_recent_paths() below. 28 chars is too short —
# "Unattended Weapon Detected: Gun" and "Unattended Weapon Detected:
# Knife" BOTH truncate to the identical 28-char slug
# "unattended_weapon_detected__", silently losing the weapon-type suffix
# entirely. If two different weapon sub-types are flagged as "Unattended"
# on the same camera within the same second, their burst files can
# literally collide on disk (same filename → one overwrites the other),
# and get_recent_paths()'s substring match can no longer tell them apart.
# Raised to 60, which comfortably fits every alert type in the current
# capability registry (the longest, "Unattended Weapon Detected: Knife",
# is 34 chars once slugified) with headroom for future additions.
SLUG_MAX_LEN = 60

# (Previous versions had a hand-maintained SNAPSHOT_TRIGGERS keyword set
# here — replaced by severity-based should_snap() below, which can't
# drift out of sync with the alert type universe the way a manually
# maintained list inevitably does.)

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
        """
        Severity-based, not keyword-based: medium/high always get a
        screenshot burst, low (Patrol/Movement) never does. Replaces the
        SNAPSHOT_TRIGGERS keyword list, which had real gaps — "Guard
        Idle" (medium severity) was never in that list, so it silently
        never got a screenshot even though it should have. This is
        called with the RAW canonical alert_type (main_web.py's call
        sites pass it directly, before alert_manager.py's translation),
        so severity lookup against the raw type is safe here — unlike
        email_alerter.py's should_send(), which is called with the
        already-translated type and must use the alert dict's
        pre-computed severity instead.

        Explicit exclusion (2026-07): "Unknown Person ..." alerts never
        get a screenshot burst regardless of severity, per request — an
        unenrolled/unrecognized person is common and low-signal enough
        in most deployments that photographing every one of them is
        noise, not useful evidence. Checked against the raw type since
        this is always called pre-translation.
        """
        if alert_type.lower().startswith("unknown person"):
            return False
        try:
            from alerts.alert_manager import _get_severity
            return _get_severity(alert_type) != "low"
        except Exception:
            return True   # fail open — better to over-capture than silently miss one

    # ── Context capture (thread-safety helper) ──────────────────────────────

    def _capture_context(self):
        """
        Reads camera_id (contextvar — stable within a session, correct)
        and profile_id (LIVE registry, keyed by camera — reflects a
        mid-session admin-panel profile switch immediately, unlike the
        old per-thread contextvar snapshot which went stale the moment a
        camera's pipeline thread started).
        Safe to call directly from save()/save_and_get_paths() (both run
        in the real per-camera pipeline thread) — NOT safe to call from
        inside a background thread (_do_burst), which is exactly why
        save() captures this up front and passes it through explicitly.
        """
        try:
            from alerts.alert_manager import get_camera_context, get_camera_profile
            cam_id = get_camera_context()
            return cam_id, get_camera_profile(cam_id)
        except Exception:
            return "cam_1", "guard_monitoring"

    # ── Public API ────────────────────────────────────────────────────────────

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

        camera_id, profile_id = self._capture_context()   # captured HERE, correct thread

        threading.Thread(
            target=self._do_burst,
            args=(frame.copy(), alert_type, guard_id, zone,
                  datetime.datetime.now(), camera_id, profile_id),
            daemon=True
        ).start()

    def save_and_get_paths(self, frame, alert_type: str,
                           guard_id: str = "", zone: str = "") -> list:
        """
        Save 3-burst synchronously, return file paths for email attachment.
        Grabs frame 1 immediately, frames 2+3 from live stream after 1s/2s.
        Synchronous (no background thread) — safe to read context directly.
        """
        if not self.should_snap(alert_type):
            return []
        if frame is None or frame.size == 0:
            return []
        if self._session_dir is None:
            self.new_session()

        camera_id, profile_id = self._capture_context()
        ts    = datetime.datetime.now()
        paths = []

        for i, gap in enumerate([0, 0.4, 0.8], 1):
            if gap > 0:
                time.sleep(0.4)
            src = frame if i == 1 else self._live_frame(frame)
            p   = self._write(src, alert_type, guard_id, zone, ts, i,
                              camera_id, profile_id)
            if p:
                paths.append(p)

        return paths

    def _do_burst(self, f0, alert_type, guard_id, zone, ts, camera_id, profile_id):
        """Background burst writer: t=0, t+0.4s, t+0.8s (~1s total, down
        from ~2s) — shortened specifically to reduce the window where
        multiple alerts firing within a few seconds of each other (which
        your logs show happening — weapon+unattended weapon 2s apart,
        tamper alerts close together) have overlapping burst writes
        competing for CPU/disk I/O, which was contributing to some
        screenshots not finishing in time to attach to their email.
        camera_id/profile_id passed in as plain arguments since this
        thread can't read the caller's contextvars (see module
        docstring)."""
        self._write(f0, alert_type, guard_id, zone, ts, 1, camera_id, profile_id)
        time.sleep(0.4)
        self._write(self._live_frame(f0), alert_type, guard_id, zone, ts, 2,
                    camera_id, profile_id)
        time.sleep(0.4)
        self._write(self._live_frame(f0), alert_type, guard_id, zone, ts, 3,
                    camera_id, profile_id)

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

    def _write(self, frame, alert_type, guard_id, zone, ts, num,
              camera_id="cam_1", profile_id="guard_monitoring") -> str:
        """Annotate and write one burst frame. Returns filepath."""
        try:
            h, w  = frame.shape[:2]
            out   = frame.copy()

            # Defensive normalization: cv2.imwrite() silently returns
            # False (no exception) for arrays that aren't a plain
            # contiguous uint8 buffer — a plausible contributor to the
            # intermittent blackout-frame write failures, since a tamper
            # frame is more likely than a normal frame to have come
            # through an unusual code path (e.g. a diff/threshold buffer
            # upstream). Cheap to guard against unconditionally.
            if out.dtype != "uint8":
                out = out.astype("uint8")
            if not out.flags["C_CONTIGUOUS"]:
                out = out.copy(order="C")

            # Profile-appropriate wording for the banner + filename —
            # should_snap() upstream already matched on the raw canonical
            # type, so this translation doesn't affect whether a snapshot
            # gets taken, only how it's labeled.
            try:
                from alerts.alert_manager import _translate_alert_type
                display_type = _translate_alert_type(alert_type, profile_id)
            except Exception:
                display_type = alert_type

            # Red banner
            cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 180), -1)
            label = display_type
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
                           for c in display_type.lower())[:SLUG_MAX_LEN]
            # "__" delimits each segment — camera_ids ("cam_bank_lobby")
            # and profile_ids ("guard_monitoring") both contain single
            # underscores themselves, so a single-"_" split would be
            # ambiguous; "__" never appears inside either, making this
            # reliably parseable back out in list_snapshots()/
            # get_recent_paths() below. Now embeds profile_id too (not
            # just camera_id) — the analytics tab was mixing snapshots
            # from every profile ever run on a camera together, since
            # camera-only scoping doesn't separate "same camera, tested
            # under 3 different profiles in one session," which is
            # exactly the actual usage pattern.
            filename = f"{ts.strftime('%H%M%S')}__{camera_id}__{profile_id}__{safe}_{num}.jpg"
            filepath = os.path.join(self._session_dir, filename)

            with self._lock:
                # cv2.imwrite() returns False on failure — it does NOT
                # raise, so the outer try/except never caught this case.
                # Bug found 2026-07 (handoff point #2, "Camera Tamper:
                # Blackout — no snapshot attached in ~30% of cases"): the
                # write could silently fail and this function would still
                # return a filepath that points at a file that was never
                # actually created, which get_recent_paths() then can
                # never find (it lists the real directory contents) and
                # save_and_get_paths() would hand a dead path straight to
                # the email attacher.
                ok = cv2.imwrite(filepath, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    print(f"[SnapshotManager] \u2717 cv2.imwrite() returned False for "
                          f"{filename} (dtype={out.dtype}, shape={out.shape}) \u2014 "
                          f"retrying once.")
                    ok = cv2.imwrite(filepath, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok:
                        print(f"[SnapshotManager] \u2717 Retry also failed for {filename} "
                              f"\u2014 giving up on this frame (alert_type={alert_type!r}, "
                              f"num={num}).")
                        self._prune()
                        return ""
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

    def get_recent_paths(self, alert_type: str, max_age_secs: float = 10.0,
                         camera_id: str = None, profile_id: str = None) -> list:
        """
        Return file paths of the most recent burst for a given alert type.
        Used by email_alerter to attach snapshots taken in the last N seconds.

        camera_id / profile_id: pass these EXPLICITLY from the alert dict
        (which already carries the correct values) — do not rely on
        get_camera_context()/get_profile_context() here, since this is
        typically called from email_alerter's background send thread,
        which (like all plain threads) doesn't inherit the pipeline
        thread's context. Passing None for either disables that specific
        filter (matches any camera / uses untranslated matching).
        """
        if not self._session_dir:
            return []

        try:
            from alerts.alert_manager import _translate_alert_type
            display_type = (_translate_alert_type(alert_type, profile_id)
                            if profile_id else alert_type)
        except Exception:
            display_type = alert_type
        safe = "".join(c if c.isalnum() else "_"
                       for c in display_type.lower())[:SLUG_MAX_LEN]

        now = time.time()
        try:
            matches = []
            for f in os.listdir(self._session_dir):
                if not f.endswith(".jpg"):
                    continue

                if camera_id or profile_id:
                    # BUG FOUND 2026-07 (root cause of "Fire/Weapon/Camera
                    # Tamper never get snapshots attached" — confirmed
                    # against real filenames, not guessed): every alert
                    # type with a colon in its canonical name — "Camera
                    # Tamper: Blackout", "Fire Detected: Fire", "Weapon
                    # Detected: Gun" — slugifies to something containing
                    # its OWN internal "__" (": " -> two non-alnum chars
                    # -> "__"). A bare .split("__") on the full filename
                    # then produces 5 parts instead of the expected 4,
                    # so `len(parts) != 4` below used to be True even for
                    # a perfectly normal, correctly-written current-format
                    # file — it was silently treated as "old-format,
                    # can't verify camera/profile" and skipped. Every
                    # alert type WITHOUT a colon ("Guard Sleeping",
                    # "Crowd Detected", "Phone Usage") slugifies to a
                    # single underscore and was never affected — which is
                    # exactly the split the real logs showed between
                    # "always attaches" and "never attaches" types.
                    # Fix: maxsplit=3 caps the split at the first 3 "__"
                    # delimiters (which is all that's needed to isolate
                    # ts/camera_id/profile_id), leaving "{slug}_{num}"
                    # intact as the 4th element no matter how many more
                    # "__" sequences it contains internally.
                    parts = f.replace(".jpg", "").split("__", 3)
                    if len(parts) != 4:
                        continue   # genuinely old-format file, no camera_id/profile_id to check
                    if camera_id and parts[1] != camera_id:
                        continue
                    if profile_id and parts[2] != profile_id:
                        continue

                if safe not in f:
                    continue
                path = os.path.join(self._session_dir, f)
                if (now - os.path.getmtime(path)) <= max_age_secs:
                    matches.append(path)
            return sorted(matches)[:3]
        except Exception:
            return []

    def list_snapshots(self, camera_id: str = None, profile_id: str = None) -> list:
        """
        camera_id: optional filter — only return snapshots from that one
        camera.
        profile_id: optional filter — only return snapshots taken while
        that profile was active. Pass the CURRENT profile for the
        requested camera (via get_camera_profile()) to get "only what
        this camera's active profile actually produced" — otherwise a
        camera tested under 3 different profiles in one session shows
        all three profiles' snapshots mixed together, which is exactly
        what was happening before this filter existed.
        """
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
                stem = f.replace(".jpg", "")
                # Same maxsplit fix as get_recent_paths() above — see that
                # comment for the full story. Without maxsplit=3, any
                # colon-bearing alert type (Fire/Weapon/Camera Tamper)
                # fell through to the "else" branch below as if it were a
                # pre-camera_id legacy file, corrupting f_camera_id/
                # f_profile_id (both wrongly set to None, silently
                # dropped by any profile-filtered listing) AND the
                # display label (stem[7:] assumed a legacy prefix width
                # that doesn't match the current filename format).
                parts = stem.split("__", 3)

                if len(parts) == 4:
                    # Current format: HHMMSS__camera_id__profile_id__slug_num
                    f_camera_id  = parts[1]
                    f_profile_id = parts[2]
                    label_part   = parts[3].rsplit("_", 1)[0]
                elif len(parts) == 3:
                    # Format from before profile_id was added to the
                    # filename — has camera_id, not profile_id. Still
                    # listed when no profile filter is requested;
                    # excluded when one is (can't safely guess which
                    # profile an old file belongs to).
                    f_camera_id  = parts[1]
                    f_profile_id = None
                    label_part   = parts[2].rsplit("_", 1)[0]
                else:
                    # Original format, before camera_id existed at all.
                    f_camera_id  = None
                    f_profile_id = None
                    label_part   = stem[7:].rsplit("_", 1)[0] if len(stem) > 7 else stem

                if camera_id and f_camera_id != camera_id:
                    continue
                if profile_id and f_profile_id != profile_id:
                    continue

                path = os.path.join(self._session_dir, f)
                try:
                    stat  = os.stat(path)
                    label = label_part.replace("_", " ").title()
                    result.append({
                        "filename":   f,
                        "label":      label,
                        "camera_id":  f_camera_id,
                        "profile_id": f_profile_id,
                        "size_kb":   round(stat.st_size / 1024, 1),
                        "mtime":     stat.st_mtime,
                        "ts":        datetime.datetime.fromtimestamp(
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
