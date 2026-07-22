import cv2
import platform
import threading
import time


# Real deployments (per 2026-07 direction) don't use 2-3 USB webcams —
# they use a DVR/NVR that serves every physical camera as its own RTSP
# channel. That source is a plain Python string, e.g.:
#   "rtsp://user:pass@192.168.1.50:554/cam/realmonitor?channel=1&subtype=0"
# Before this fix, ANY string source (RTSP URL OR a local video file path
# like "uploads/test.mp4") was treated identically via `is_file = True`,
# which is wrong for a live network stream in two concrete ways:
#   1. FPS-throttling to cap.get(CAP_PROP_FPS) — meaningless/unreliable
#      for a live feed, and actively wrong to rate-limit consumption of.
#   2. "End of stream -> seek back to frame 0" — there IS no frame 0 on a
#      live RTSP feed; ret=False there means a real connection drop and
#      needs a RECONNECT, not a seek.
_STREAM_URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")


def is_stream_url(source) -> bool:
    """True for a live network camera source (RTSP/HTTP — DVR/NVR
    channel or a direct IP camera), False for anything else (webcam
    index, local file path). Shared with camera_manager.py so both
    files agree on what counts as reconnect-worthy vs. a local file."""
    return isinstance(source, str) and source.lower().startswith(_STREAM_URL_SCHEMES)


class VideoStreamReader:
    """
    Threaded video reader for webcams, DVR/NVR-fed RTSP cameras, and
    recorded video files — three distinct behaviors, not two:

    Webcam (int):        reads as fast as possible, always serves latest frame.
    RTSP/DVR (str, URL):  reads as fast as frames arrive (no throttling — it's
                          live); a dropped connection triggers an in-place
                          reconnect with backoff instead of stopping.
    File (str, path):    throttles to the video's native FPS so frames
                          aren't consumed instantly, and loops back to
                          start at end-of-file.

    ── Dual-camera-stop fix (2026-07) ──────────────────────────────────────
    Root cause (confirmed against the actual code, not just theory):
    `cv2.VideoCapture()` is a blocking, synchronous call with NO built-in
    timeout. On Windows, opening two USB cameras from two different
    threads at nearly the same time is a well-known trigger for the
    default backend (MSMF) to hang indefinitely — and because that hang
    happens inside a C/C++ call, it can hold the GIL and freeze the
    ENTIRE Flask process, not just this one camera's thread. That matches
    the reported symptom exactly: both cameras start, shared models
    finish loading (proof both pipeline threads were alive), then
    everything goes silent with zero further log output — because the
    open() call for the second camera never returns and never raises.

    Two independent fixes, both applied below:
      1. Open on cv2.CAP_DSHOW explicitly for integer (webcam-index)
         sources on Windows — DSHOW does not exhibit the concurrent-open
         hang the way MSMF does. (Irrelevant for RTSP sources, which
         always use the FFMPEG backend regardless of platform.)
      2. Open inside a watchdog thread with a hard timeout, so even if a
         hang still occurs (e.g. a bad index, a camera in use by another
         app, or an RTSP source that's unreachable), it degrades into a
         clean RuntimeError that main_web.run() already catches and logs,
         instead of freezing the process. (The watchdog thread itself
         can't be force-killed if the underlying OS call is truly stuck —
         that's an accepted tradeoff: one leaked daemon thread beats the
         whole app going dark.)

    A SECOND, independently real bug was found and fixed while
    diagnosing this: the capture loop's "read failed" path previously
    had ZERO log output before setting self._stopped = True — a camera
    could go quiet with no diagnostic trace at all. Every stop/retry path
    below now logs a reason.
    """

    OPEN_TIMEOUT_SEC             = 8.0
    MAX_CONSECUTIVE_READ_ERRORS  = 10   # webcam: give up and stop after this many
    RTSP_RECONNECT_DELAY_SEC     = 2.0
    RTSP_MAX_RECONNECT_ATTEMPTS  = 30   # ~1 min of retrying a dropped DVR channel
                                         # before handing off to camera_manager's
                                         # own outer retry (which waits longer)

    def __init__(self, source=0):
        self.source   = source
        self.is_rtsp  = is_stream_url(source)
        self.is_file  = isinstance(source, str) and not self.is_rtsp

        self.cap = self._open_with_timeout(source, self.OPEN_TIMEOUT_SEC)

        if not self.is_file:
            # Applies to both webcam AND rtsp — neither should buffer
            # stale frames; always serve the latest.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Best-effort — only honored by backends that support it (mainly
        # FFMPEG, which is what handles RTSP — this is the main thing
        # that makes a genuinely stalled DVR connection surface as a
        # read failure instead of hanging cap.read() indefinitely).
        try:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        # Throttle file playback to native FPS — NOT applied to RTSP
        # (live feed, no meaningful "native FPS" to throttle to) or
        # webcam (already real-time).
        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_interval = 1.0 / fps if fps > 0 else 1.0 / 25.0
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total / fps if fps > 0 else 0
            print(f"[VideoStreamReader] File: {source} | {fps:.1f} fps | {total} frames | {duration:.1f}s")
        else:
            self._frame_interval = 0.0
            if self.is_rtsp:
                print(f"[VideoStreamReader] RTSP/DVR stream connected: {self._redact(source)}")

        self._frame   = None
        self._lock    = threading.Lock()
        self._stopped = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _redact(url: str) -> str:
        """user:pass@ credentials out of logs — DVR URLs carry real
        camera passwords in plain text in the source string."""
        if "@" not in url:
            return url
        scheme_sep = url.find("://")
        if scheme_sep == -1:
            return url
        scheme = url[:scheme_sep + 3]
        rest   = url[scheme_sep + 3:]
        creds, _, host = rest.partition("@")
        return f"{scheme}***:***@{host}" if creds else url

    def _open_with_timeout(self, source, timeout_sec):
        """
        Opens cv2.VideoCapture on a background thread and waits up to
        timeout_sec for it to finish, instead of letting a hang block
        this thread (and potentially the whole process) forever.
        """
        result = {}

        def _do_open():
            try:
                if isinstance(source, int) and platform.system() == "Windows":
                    # CAP_DSHOW avoids the MSMF concurrent-open hang that
                    # causes the dual-USB-camera stop-after-startup bug.
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    # RTSP URLs and local files both go through here —
                    # OpenCV picks FFMPEG for URL strings automatically.
                    cap = cv2.VideoCapture(source)
                result["cap"] = cap
            except Exception as e:
                result["error"] = e

        opener = threading.Thread(target=_do_open, daemon=True)
        opener.start()
        opener.join(timeout=timeout_sec)

        if opener.is_alive():
            print(f"[VideoStreamReader] \u2717 Timed out opening source "
                  f"{self._redact(source) if isinstance(source, str) else source!r} "
                  f"after {timeout_sec}s \u2014 treating as unavailable. "
                  f"(If this is a USB webcam index, another process or camera "
                  f"thread may be holding it open. If this is an RTSP/DVR URL, "
                  f"check the DVR is reachable and the channel/credentials are correct.)")
            raise RuntimeError(
                f"Timed out opening video source after {timeout_sec}s "
                f"(camera may be in use by another process, or the index/URL is invalid)"
            )

        if "error" in result:
            raise RuntimeError(f"Error opening video source: {result['error']}")

        return result["cap"]

    def _attempt_rtsp_reconnect(self) -> bool:
        """
        Re-opens the SAME RTSP URL in place after a dropped connection.
        Returns True on success. This is the expected, routine failure
        mode for a real DVR/NVR deployment (network blips, DVR reboots,
        Wi-Fi cameras going briefly out of range) — treating it as fatal
        (old behavior: stop the whole camera thread on ANY read failure)
        would mean a single dropped frame permanently kills a camera
        until a manual restart, which doesn't hold up against "a hell of
        a lot of cameras" all feeding through one DVR.
        """
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            new_cap = cv2.VideoCapture(self.source)
            if new_cap.isOpened():
                self.cap = new_cap
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except Exception:
                    pass
                return True
            new_cap.release()
        except Exception:
            pass
        return False

    def _capture_loop(self):
        consecutive_errors = 0
        rtsp_reconnect_attempts = 0
        while not self._stopped:
            t_start = time.time()

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                consecutive_errors += 1
                print(f"[VideoStreamReader] \u2717 cap.read() raised "
                      f"{type(e).__name__}: {e} (source="
                      f"{self._redact(self.source) if self.is_rtsp else self.source}, "
                      f"consecutive_errors={consecutive_errors})")
                ret = False
                frame = None
                if not self.is_rtsp:
                    if consecutive_errors >= self.MAX_CONSECUTIVE_READ_ERRORS:
                        print(f"[VideoStreamReader] \u2717 {consecutive_errors} consecutive "
                              f"read errors \u2014 stopping stream.")
                        self._stopped = True
                        break
                    time.sleep(0.1)
                    continue

            if not ret:
                if self.is_file:
                    # End of file — loop back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                elif self.is_rtsp:
                    # Expected, routine for a live network stream — try
                    # to reconnect in place rather than killing the
                    # camera. camera_manager.py also has an OUTER retry
                    # for when this inner budget is exhausted (e.g. the
                    # DVR itself is down for an extended period).
                    rtsp_reconnect_attempts += 1
                    if rtsp_reconnect_attempts > self.RTSP_MAX_RECONNECT_ATTEMPTS:
                        print(f"[VideoStreamReader] \u2717 RTSP stream "
                              f"({self._redact(self.source)}) failed to "
                              f"reconnect after {rtsp_reconnect_attempts} "
                              f"attempts \u2014 giving up at this level; "
                              f"camera_manager will retry the whole pipeline.")
                        self._stopped = True
                        break
                    print(f"[VideoStreamReader] \u26a0 RTSP stream "
                          f"({self._redact(self.source)}) read failed \u2014 "
                          f"reconnecting (attempt {rtsp_reconnect_attempts}/"
                          f"{self.RTSP_MAX_RECONNECT_ATTEMPTS}) in "
                          f"{self.RTSP_RECONNECT_DELAY_SEC}s...")
                    time.sleep(self.RTSP_RECONNECT_DELAY_SEC)
                    if self._attempt_rtsp_reconnect():
                        print(f"[VideoStreamReader] \u2713 RTSP stream "
                              f"({self._redact(self.source)}) reconnected.")
                        rtsp_reconnect_attempts = 0
                    continue

                else:
                    # Webcam
                    consecutive_errors += 1
                    print(f"[VideoStreamReader] \u2717 cap.read() returned ret=False "
                          f"(source={self.source}) \u2014 stream ended, stopping.")
                    self._stopped = True
                    break

            consecutive_errors = 0
            rtsp_reconnect_attempts = 0
            with self._lock:
                self._frame = frame

            # Throttle to native FPS for file sources only.
            if self._frame_interval > 0:
                elapsed = time.time() - t_start
                sleep_t = self._frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    def read_frame(self):
        """Return the latest frame, or None if the stream has ended."""
        if self._stopped:
            return None
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2)
        try:
            self.cap.release()
        except Exception:
            pass
