"""
whatsapp_alerter.py
────────────────────
WhatsApp batch alerter — sends exactly 5 alerts per message, no timeout.
Session summary sent only at explicit session end (not mid-session).
"""

import os
import sys
import time
import queue
import threading
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from utils.logger import get_logger
    _log = get_logger("whatsapp_alerter")
except Exception:
    import logging
    _log = logging.getLogger("whatsapp_alerter")


class WhatsAppAlerter:

    def __init__(self, config: dict):
        self._enabled     = config.get("enabled", False)
        self._transport   = config.get("transport", "stub")
        self._page_wait   = config.get("page_load_wait", 19)
        self._chrome_prof = config.get("chrome_profile", "Default")
        self._batch_size  = config.get("batch_size", 5)
        self._alert_types = set(config.get("alert_types", []))

        raw = config.get("recipient_number", "")
        self._recipient = self._parse_number(str(raw))

        self._buffer      = []
        self._buffer_lock = threading.Lock()
        self._send_queue  = queue.Queue(maxsize=20)

        if not self._enabled:
            _log.info("WhatsApp alerts disabled (whatsapp.enabled=false in config).")
            return

        if self._transport == "chrome":
            if not self._recipient:
                _log.warning("WhatsApp: recipient_number not configured.")
                self._enabled = False
                return
            try:
                import pyautogui
            except ImportError:
                _log.warning("pyautogui not installed. Run: pip install pyautogui")
                self._enabled = False
                return

        # Log without exposing the recipient number
        _log.info(
            f"WhatsApp ready | transport: {self._transport} | "
            f"1 recipient configured | batch size: {self._batch_size}"
        )

        # Single worker thread — serialized sends, no concurrent browser control
        threading.Thread(target=self._worker, daemon=True, name="wa-worker").start()
        # NOTE: No _flush_timer thread — only flush when batch of 5 is full

    @staticmethod
    def _parse_number(raw) -> str:
        digits = "".join(c for c in str(raw) if c.isdigit())
        return "+" + digits if digits else ""

    def _matches(self, alert_type: str) -> bool:
        if not self._alert_types:
            return True
        return any(t.lower() in alert_type.lower() for t in self._alert_types)

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, alert: dict):
        """Add alert to batch. Sends immediately when batch_size reached."""
        if not self._enabled:
            return
        if not self._matches(alert.get("type", "")):
            return

        with self._buffer_lock:
            self._buffer.append(alert)
            size = len(self._buffer)

        _log.debug(f"WA buffer {size}/{self._batch_size} — {alert.get('type','')}")

        if size >= self._batch_size:
            self._flush()

    def send_session_summary(self, alert_log: list, stats: dict):
        """
        Send session summary. Only includes:
        weapon, phone, sleeping, missing, tamper, smoking, fire, crowd.
        For unknown persons: only weapon alerts.
        """
        if not self._enabled:
            return

        import datetime
        from collections import Counter

        now   = datetime.datetime.now()
        total = len(alert_log)
        high_cnt = sum(1 for a in alert_log if a.get("severity") == "high")

        # Named guards (not Unknown, not Camera/Post)
        guard_names = set(
            a.get("guard_id", "") for a in alert_log
            if a.get("guard_id", "") not in ("Camera", "Post", "", "Unknown")
            and not a.get("guard_id", "").startswith("Unknown_")
        )

        # Only these types in summary
        _SUMMARY_KEYWORDS = [
            "weapon", "phone usage", "sleeping", "missing",
            "tamper", "smoking", "fire",
        ]

        def _is_relevant(a):
            atype = a.get("type", "").lower()
            guard = a.get("guard_id", "")
            is_unknown = guard.startswith("Unknown") or guard == "Unknown"
            # For unknown persons, only show weapon
            if is_unknown:
                return "weapon" in atype
            return any(kw in atype for kw in _SUMMARY_KEYWORDS)

        relevant = [a for a in alert_log if _is_relevant(a)]
        types = Counter(a.get("type", "") for a in relevant)
        top = types.most_common(7)
        top_str = "\n".join(f"  • {t}: {c}" for t, c in top) if top else "  None"

        message = (
            f"📊 NoviSentra Shift Summary\n\n"
            f"Date: {now.strftime('%d %b %Y  %H:%M')}\n"
            f"Total Alerts: {total}\n"
            f"High Severity: {high_cnt}\n"
            f"Guards Monitored: {len(guard_names)}\n\n"
            f"Key Incidents:\n{top_str}\n\n"
            f"Full report sent via email."
        )

        try:
            self._send_queue.put_nowait(message)
            _log.info("WhatsApp session summary queued.")
        except Exception:
            _log.warning("WhatsApp queue full — session summary dropped.")

    # ── Batch ─────────────────────────────────────────────────────────────────

    def _flush(self):
        """Grab buffer, format, queue for sending."""
        with self._buffer_lock:
            if not self._buffer:
                return
            alerts = list(self._buffer)
            self._buffer = []

        message = self._format_batch(alerts)
        _log.info(f"WhatsApp flushing {len(alerts)} alerts")
        try:
            self._send_queue.put_nowait(message)
        except queue.Full:
            _log.warning("WhatsApp send queue full — batch dropped.")

    def _format_batch(self, alerts: list) -> str:
        has_high = any(a.get("severity") == "high" for a in alerts)
        has_med  = any(a.get("severity") == "medium" for a in alerts)
        emoji    = "🔴" if has_high else "🟡" if has_med else "🟢"

        lines = [f"{emoji} NoviSentra Security Alert", ""]
        for a in alerts:
            lines.append(f"Alert: {a.get('type', '')}")
            guard = a.get("guard_id", "")
            zone  = a.get("zone", "")
            ts    = a.get("timestamp", "")
            if guard and guard not in ("Camera", "Post", ""):
                lines.append(f"Guard: {guard}")
            if zone and zone not in ("—", "-", ""):
                lines.append(f"Zone: {zone}")
            if ts:
                lines.append(f"Time: {ts}")
            lines.append("")
        lines.append("Check the NoviSentra dashboard for live video.")
        return "\n".join(lines)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        """Single worker — one send at a time, 5s gap between sends."""
        while True:
            try:
                message = self._send_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._do_send(message)
            except Exception as e:
                _log.error(f"WhatsApp send error: {e}", exc_info=True)
            finally:
                self._send_queue.task_done()
                time.sleep(5)

    def _do_send(self, message: str):
        if self._transport == "stub":
            _log.info("[WA STUB] Message would be sent")
            return
        if self._transport == "chrome":
            self._send_chrome(message)

    # ── Chrome transport ──────────────────────────────────────────────────────

    def _send_chrome(self, message: str):
        import pyautogui

        try:
            # Copy to clipboard via PowerShell
            ps_cmd = (
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'[System.Windows.Forms.Clipboard]::SetText(@\'\n{message}\n\'@)'
            )
            subprocess.run(["powershell", "-Command", ps_cmd],
                           capture_output=True, timeout=5)
            _log.info("Message copied to clipboard.")

            # Open WhatsApp Web with saved Chrome profile
            chrome_dir = os.path.expanduser("~") + "\\AppData\\Local\\Google\\Chrome\\User Data"
            subprocess.Popen([
                "cmd", "/c", "start",
                "https://web.whatsapp.com",
                "--profile-directory=" + self._chrome_prof,
                "--user-data-dir=" + chrome_dir
            ])
            _log.info(f"WhatsApp Web opening... waiting {self._page_wait}s")
            time.sleep(self._page_wait)

            # Tab x4 → search box
            for _ in range(4):
                pyautogui.press("tab")
                time.sleep(0.1)

            # Type number, select contact
            pyautogui.write(self._recipient, interval=0.1)
            time.sleep(2)
            pyautogui.press("down")
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(1)

            # Paste and send
            pyautogui.hotkey("ctrl", "v")
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(2)

            # Close tab + minimize
            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.5)

            _log.info("WhatsApp alert sent")
            print("[WhatsAppAlerter] ✓ Batch sent")

        except Exception as e:
            _log.error(f"WhatsApp chrome send failed: {e}")
            try:
                pyautogui.hotkey("ctrl", "w")
            except Exception:
                pass