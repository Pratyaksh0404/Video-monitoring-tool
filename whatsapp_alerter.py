"""
whatsapp_alerter.py
────────────────────
WhatsApp batch alerter using your proven Chrome + pyautogui approach.

Batches 5 alerts (or 5 minutes) into one message, then sends.
Uses contact number in the search box — same as contact name, just a number.
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
        self._enabled       = config.get("enabled", False)
        self._transport     = config.get("transport", "stub")
        self._page_wait     = config.get("page_load_wait", 19)
        self._chrome_prof   = config.get("chrome_profile", "Default")
        self._batch_size    = config.get("batch_size", 5)
        self._batch_timeout = config.get("batch_timeout", 300)
        self._alert_types   = set(config.get("alert_types", []))

        # Phone number used as search term (e.g. "+919876543210")
        raw = config.get("recipient_number", "")
        self._recipient = self._parse_number(str(raw))

        self._buffer           = []
        self._buffer_lock      = threading.Lock()
        self._first_alert_time = 0.0

        self._send_queue = queue.Queue(maxsize=20)

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

        _log.info(
            f"WhatsApp ready | transport: {self._transport} | "
            f"recipient: {self._recipient} | "
            f"batch: {self._batch_size} alerts or {self._batch_timeout}s"
        )

        threading.Thread(target=self._worker,       daemon=True, name="wa-worker").start()
        threading.Thread(target=self._flush_timer,  daemon=True, name="wa-flush").start()

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
        if not self._enabled:
            return
        if not self._matches(alert.get("type", "")):
            return

        with self._buffer_lock:
            if not self._buffer:
                self._first_alert_time = time.time()
            self._buffer.append(alert)
            size = len(self._buffer)

        _log.debug(f"WA buffer {size}/{self._batch_size} — {alert.get('type','')}")

        if size >= self._batch_size:
            self._flush("batch full")

    # ── Batch management ──────────────────────────────────────────────────────

    def _flush_timer(self):
        while True:
            time.sleep(10)
            with self._buffer_lock:
                if not self._buffer:
                    continue
                elapsed = time.time() - self._first_alert_time
            if elapsed >= self._batch_timeout:
                self._flush("timeout")

    def _flush(self, reason: str):
        with self._buffer_lock:
            if not self._buffer:
                return
            alerts = list(self._buffer)
            self._buffer = []
            self._first_alert_time = 0.0

        message = self._format_batch(alerts)
        _log.info(f"WhatsApp flushing {len(alerts)} alerts ({reason})")
        try:
            self._send_queue.put_nowait(message)
        except queue.Full:
            _log.warning("WhatsApp send queue full — batch dropped.")

    def _format_batch(self, alerts: list) -> str:
        has_high = any(a.get("severity") == "high" for a in alerts)
        has_med  = any(a.get("severity") == "medium" for a in alerts)
        emoji    = "🔴" if has_high else "🟡" if has_med else "🟢"

        lines = [f"{emoji} GMS Security Alert", ""]
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
        lines.append("Check the GMS dashboard for live video.")
        return "\n".join(lines)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
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
            _log.info(f"[WA STUB] → {self._recipient}:\n{message}")
            print(f"[WhatsAppAlerter] STUB → {self._recipient}")
            return
        if self._transport == "chrome":
            self._send_chrome(message)

    # ── Chrome transport (clipboard paste — fast) ─────────────────────────────

    def _send_chrome(self, message: str):
        import pyautogui
        import subprocess

        try:
            # ── Step 1: Copy message to clipboard ─────────────────────────
            # Use PowerShell to set clipboard — no extra library needed
            ps_cmd = (
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'[System.Windows.Forms.Clipboard]::SetText(@\'\n{message}\n\'@)'
            )
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, timeout=5
            )
            _log.info("Message copied to clipboard.")

            # ── Step 2: Open WhatsApp Web in Chrome ───────────────────────
            chrome_dir = os.path.expanduser("~") + "\\AppData\\Local\\Google\\Chrome\\User Data"
            subprocess.Popen([
                "cmd", "/c", "start",
                "https://web.whatsapp.com",
                "--profile-directory=" + self._chrome_prof,
                "--user-data-dir=" + chrome_dir
            ])
            _log.info(f"WhatsApp Web opening... waiting {self._page_wait}s")

            # ── Step 3: Wait for page to load ─────────────────────────────
            time.sleep(self._page_wait)

            # ── Step 4: Tab x4 → search box ──────────────────────────────
            for _ in range(4):
                pyautogui.press("tab")
                time.sleep(0.1)

            # ── Step 5: Type recipient number, select contact ─────────────
            pyautogui.write(self._recipient, interval=0.1)
            time.sleep(2)
            pyautogui.press("down")
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(1)

            # ── Step 6: Paste message (instant) ──────────────────────────
            pyautogui.hotkey("ctrl", "v")
            time.sleep(1)

            # ── Step 7: Send ──────────────────────────────────────────────
            pyautogui.press("enter")
            time.sleep(2)

            # ── Step 8: Close tab + minimize ─────────────────────────────
            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.5)
            _log.info(f"WhatsApp batch sent → {self._recipient}")
            print(f"[WhatsAppAlerter] ✓ Sent → {self._recipient}")

        except Exception as e:
            _log.error(f"WhatsApp chrome send failed: {e}")
            try:
                pyautogui.hotkey("ctrl", "w")
            except Exception:
                pass