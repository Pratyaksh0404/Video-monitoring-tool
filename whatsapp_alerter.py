"""
whatsapp_alerter.py
────────────────────
v2 — Transport-agnostic WhatsApp delivery.

  transport: "twilio"  → Twilio WhatsApp Business API (recommended for
                          production — no browser, no screen, works on
                          any server). Sends one templated message per
                          HIGH-severity alert, immediately (no batching —
                          see note below on why).
  transport: "chrome"  → original Chrome + pyautogui automation, kept for
                          local/dev use only. NOT usable on a server —
                          requires a physical screen and logged-in
                          WhatsApp Web session.
  transport: "stub"    → logs what would be sent, sends nothing. Useful
                          for testing the alert pipeline without touching
                          any real transport.

Why per-alert instead of batched, for the twilio transport:
  WhatsApp Business API requires any business-initiated message (i.e. any
  message that isn't a reply within 24h of the customer messaging you
  first) to use a pre-approved message TEMPLATE with fixed variable slots
  ({{1}}, {{2}}, ...). The old "buffer 5 alerts, format free text, flush"
  approach doesn't fit a template's fixed shape — Meta won't approve a
  template for an arbitrary, variable-length list of alerts. So each
  HIGH-severity alert is sent as its own templated message immediately.
  This is also just better UX — real-time instead of waiting for a batch
  of 5 or a timeout.

  The full multi-alert shift summary (previously send_session_summary)
  stays on email, which already renders it properly as an HTML report —
  no need to force that into WhatsApp's template constraints too. For
  the twilio transport, send_session_summary() is a no-op that logs a
  pointer to the email report instead.

One-time setup required before going live with transport=twilio:
  1. pip install twilio
  2. Create a Twilio account, get a WhatsApp-enabled sender (sandbox for
     testing, or an approved WhatsApp Business sender for production).
  3. In Twilio Console → Content Editor, create ONE template, e.g.:
       "🔴 NoviSentra Alert: {{1}} — {{2}}, Zone {{3}}, {{4}}"
     Submit for WhatsApp approval (Meta reviews it, usually within hours).
  4. Copy the resulting Content SID (starts with "HX...").
  5. Set environment variables:
       TWILIO_ACCOUNT_SID=ACxxxxxxxx
       TWILIO_AUTH_TOKEN=xxxxxxxx
  6. In rules_config.yaml:
       whatsapp:
         enabled: true
         transport: twilio
         twilio:
           from_number: "whatsapp:+14155238886"   # your Twilio WhatsApp sender
           content_sid: "HXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
         recipient_number: "+919876543210"
         alert_types: [...]   # unchanged — filters which alerts go to WhatsApp
"""

import os
import sys
import time
import json
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
        self._transport   = config.get("transport", "twilio")   # "twilio" | "chrome" | "stub"
        self._alert_types = set(config.get("alert_types", []))

        raw = config.get("recipient_number", "")
        self._recipient = self._parse_number(str(raw))

        # Twilio config
        twilio_cfg          = config.get("twilio", {})
        self._twilio_from     = twilio_cfg.get("from_number", "")
        self._twilio_content_sid = twilio_cfg.get("content_sid", "")
        self._twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self._twilio_auth_token  = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self._twilio_client       = None

        # Chrome (legacy/dev) config
        self._page_wait   = config.get("page_load_wait", 19)
        self._chrome_prof = config.get("chrome_profile", "Default")
        self._batch_size  = config.get("batch_size", 5)
        self._buffer      = []
        self._buffer_lock = threading.Lock()

        self._send_queue = queue.Queue(maxsize=50)

        if not self._enabled:
            _log.info("WhatsApp alerts disabled (whatsapp.enabled=false in config).")
            return

        if not self._recipient:
            _log.warning("WhatsApp: recipient_number not configured.")
            self._enabled = False
            return

        if self._transport == "twilio":
            self._init_twilio()
        elif self._transport == "chrome":
            self._init_chrome()
        elif self._transport == "stub":
            _log.info("WhatsApp ready | transport: stub (logs only, sends nothing)")
        else:
            _log.warning(f"Unknown WhatsApp transport '{self._transport}'.")
            self._enabled = False
            return

        # Single worker thread — serialized sends
        threading.Thread(target=self._worker, daemon=True, name="wa-worker").start()

    @staticmethod
    def _parse_number(raw) -> str:
        digits = "".join(c for c in str(raw) if c.isdigit())
        return "+" + digits if digits else ""

    def _matches(self, alert_type: str) -> bool:
        if not self._alert_types:
            return True
        return any(t.lower() in alert_type.lower() for t in self._alert_types)

    # ── Transport init ───────────────────────────────────────────────────────

    def _init_twilio(self):
        if not self._twilio_account_sid or not self._twilio_auth_token:
            _log.warning("transport=twilio but TWILIO_ACCOUNT_SID / "
                        "TWILIO_AUTH_TOKEN env vars not set.")
            self._enabled = False
            return
        if not self._twilio_from:
            _log.warning("transport=twilio but twilio.from_number not configured "
                        "in rules_config.yaml.")
            self._enabled = False
            return
        try:
            from twilio.rest import Client
            self._twilio_client = Client(self._twilio_account_sid, self._twilio_auth_token)
        except ImportError:
            _log.warning("'twilio' package not installed. Run: pip install twilio")
            self._enabled = False
            return

        _log.info(
            f"WhatsApp ready | transport: twilio | 1 recipient configured | "
            f"per-alert immediate send (no batching — see module docstring)"
        )

    def _init_chrome(self):
        try:
            import pyautogui  # noqa: F401
        except ImportError:
            _log.warning("pyautogui not installed. Run: pip install pyautogui")
            self._enabled = False
            return
        _log.info(
            f"WhatsApp ready | transport: chrome (DEV/LOCAL ONLY — will not "
            f"work on a server) | batch size: {self._batch_size}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, alert: dict):
        """
        Dispatch an alert.
          twilio/stub: sent immediately, one templated message per alert.
          chrome:      buffered, flushed in batches of batch_size (legacy).
        """
        if not self._enabled:
            return
        if not self._matches(alert.get("type", "")):
            return

        if self._transport in ("twilio", "stub"):
            try:
                self._send_queue.put_nowait(("alert", alert))
            except queue.Full:
                _log.warning("WhatsApp send queue full — alert dropped.")
            return

        # chrome legacy batching path
        with self._buffer_lock:
            self._buffer.append(alert)
            size = len(self._buffer)
        _log.debug(f"WA buffer {size}/{self._batch_size} — {alert.get('type','')}")
        if size >= self._batch_size:
            self._flush_chrome_batch()

    def send_session_summary(self, alert_log: list, stats: dict):
        """
        Session-end summary. For twilio, this is intentionally a no-op —
        WhatsApp template messaging doesn't fit a variable-length multi-alert
        summary. The full shift report already goes out via email
        (EmailAlerter.send_report) with proper HTML formatting. For chrome
        (dev/legacy), the original free-text summary behavior is kept.
        """
        if not self._enabled:
            return

        if self._transport in ("twilio", "stub"):
            _log.info(
                "Session summary not sent via WhatsApp (template messaging "
                "doesn't support variable-length summaries) — see the full "
                "shift report in email instead."
            )
            return

        # chrome legacy path — original free-text summary behavior
        import datetime
        from collections import Counter

        now   = datetime.datetime.now()
        total = len(alert_log)
        high_cnt = sum(1 for a in alert_log if a.get("severity") == "high")

        guard_names = set(
            a.get("guard_id", "") for a in alert_log
            if a.get("guard_id", "") not in ("Camera", "Post", "", "Unknown")
            and not a.get("guard_id", "").startswith("Unknown_")
        )

        _SUMMARY_KEYWORDS = ["weapon", "phone usage", "sleeping", "missing",
                            "tamper", "smoking", "fire"]

        def _is_relevant(a):
            atype = a.get("type", "").lower()
            guard = a.get("guard_id", "")
            is_unknown = guard.startswith("Unknown") or guard == "Unknown"
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
            f"Persons Monitored: {len(guard_names)}\n\n"
            f"Key Incidents:\n{top_str}\n\n"
            f"Full report sent via email."
        )
        try:
            self._send_queue.put_nowait(("chrome_text", message))
            _log.info("WhatsApp session summary queued (chrome transport).")
        except queue.Full:
            _log.warning("WhatsApp queue full — session summary dropped.")

    # ── Chrome batch flush (legacy) ──────────────────────────────────────────

    def _flush_chrome_batch(self):
        with self._buffer_lock:
            if not self._buffer:
                return
            alerts = list(self._buffer)
            self._buffer = []

        message = self._format_batch(alerts)
        _log.info(f"WhatsApp flushing {len(alerts)} alerts (chrome)")
        try:
            self._send_queue.put_nowait(("chrome_text", message))
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
                lines.append(f"Person: {guard}")
            if zone and zone not in ("—", "-", ""):
                lines.append(f"Zone: {zone}")
            if ts:
                lines.append(f"Time: {ts}")
            lines.append("")
        lines.append("Check the NoviSentra dashboard for live video.")
        return "\n".join(lines)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        """Single worker — one send at a time."""
        while True:
            try:
                kind, payload = self._send_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if kind == "alert":
                    self._send_alert_now(payload)
                elif kind == "chrome_text":
                    self._send_chrome(payload)
            except Exception as e:
                _log.error(f"WhatsApp send error: {e}", exc_info=True)
            finally:
                self._send_queue.task_done()
                # Small gap between sends — courteous to rate limits either way
                time.sleep(1 if self._transport == "twilio" else 5)

    # ── Immediate per-alert send (twilio / stub) ──────────────────────────────

    def _send_alert_now(self, alert: dict):
        alert_type = alert.get("type", "")
        guard_id   = alert.get("guard_id", "—")
        zone       = alert.get("zone", "—")
        timestamp  = alert.get("timestamp", "")
        severity   = alert.get("severity", "HIGH")
        camera     = alert.get("camera_id", "")

        if self._transport == "stub":
            _log.info(
                f"[WA STUB] Would send: {alert_type} | {guard_id} | "
                f"Zone {zone} | {severity.upper()} | {timestamp}"
            )
            return

        if self._transport == "twilio":
            self._send_twilio_template(alert_type, guard_id, zone,
                                       timestamp, severity, camera)

    def _send_twilio_template(self, alert_type: str, guard_id: str, zone: str,
                               timestamp: str, severity: str = "HIGH", camera: str = ""):
        """
        Send one alert via Twilio WhatsApp.

        Two modes — chosen automatically based on whether content_sid is set:

        SANDBOX / TESTING (content_sid is empty):
          Sends a free-form text message. Works immediately with the Twilio
          sandbox number (+14155238886) — no template approval needed.
          The recipient must have joined the sandbox first by sending the
          join keyword to the sandbox number.

        PRODUCTION (content_sid is set to "HX..."):
          Sends via a Meta-approved Content Template. Required once you
          switch from the sandbox to your real WhatsApp Business number.
          Template must be created in Twilio Console → Content Editor and
          approved by Meta before this works.

        Both modes send immediately, fully behind the scenes — no browser,
        no screen, no manual intervention of any kind.
        """
        try:
            to_number = f"whatsapp:{self._recipient}"

            if self._twilio_content_sid:
                # ── Production path: approved Meta template ──────────────────
                # Template slots: {{1}} alert_type, {{2}} person, {{3}} zone,
                # {{4}} severity, {{5}} time. Matches the template you create
                # in Twilio Console → Content Editor.
                content_variables = json.dumps({
                    "1": alert_type,
                    "2": guard_id if guard_id not in ("Camera", "Post", "—", "") else "System",
                    "3": zone if zone not in ("—", "-", "") else "—",
                    "4": severity.upper(),
                    "5": timestamp,
                })
                message = self._twilio_client.messages.create(
                    from_=self._twilio_from,
                    content_sid=self._twilio_content_sid,
                    content_variables=content_variables,
                    to=to_number,
                )
            else:
                # ── Sandbox / testing path: free-form text ───────────────────
                # Works with sandbox number only. Switch to content_sid path
                # when you move to your WhatsApp Business number.
                person_line = (
                    f"\n👤 *Person:* {guard_id}"
                    if guard_id not in ("Camera", "Post", "—", "", None) else ""
                )
                zone_line = (
                    f"\n📍 *Zone:* {zone}"
                    if zone not in ("—", "-", "", None) else ""
                )
                cam_line = f"\n📷 *Camera:* {camera}" if camera else ""

                body = (
                    f"⚠️ *NoviSentra Security Alert*\n\n"
                    f"🚨 *Alert:* {alert_type}"
                    f"{person_line}"
                    f"{zone_line}"
                    f"{cam_line}\n"
                    f"🔴 *Severity:* {severity.upper()}\n"
                    f"🕐 *Time:* {timestamp}\n\n"
                    f"Check the NoviSentra dashboard for live video."
                )
                message = self._twilio_client.messages.create(
                    from_=self._twilio_from,
                    body=body,
                    to=to_number,
                )

            _log.info(
                f"[WhatsApp ✓] Sent: {alert_type} | {guard_id} | "
                f"Zone {zone} | {timestamp} (sid={message.sid[:12]}...)"
            )

        except Exception as e:
            _log.error(f"Twilio WhatsApp send failed: {type(e).__name__}: {e}")

    # ── Chrome transport (legacy, dev/local only) ─────────────────────────────

    def _send_chrome(self, message: str):
        import pyautogui

        try:
            ps_cmd = (
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'[System.Windows.Forms.Clipboard]::SetText(@\'\n{message}\n\'@)'
            )
            subprocess.run(["powershell", "-Command", ps_cmd],
                           capture_output=True, timeout=5)
            _log.info("Message copied to clipboard.")

            chrome_dir = os.path.expanduser("~") + "\\AppData\\Local\\Google\\Chrome\\User Data"
            subprocess.Popen([
                "cmd", "/c", "start",
                "https://web.whatsapp.com",
                "--profile-directory=" + self._chrome_prof,
                "--user-data-dir=" + chrome_dir
            ])
            _log.info(f"WhatsApp Web opening... waiting {self._page_wait}s")
            time.sleep(self._page_wait)

            for _ in range(4):
                pyautogui.press("tab")
                time.sleep(0.1)

            pyautogui.write(self._recipient, interval=0.1)
            time.sleep(2)
            pyautogui.press("down")
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(1)

            pyautogui.hotkey("ctrl", "v")
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(2)

            pyautogui.hotkey("ctrl", "w")
            time.sleep(0.5)

            _log.info("WhatsApp alert sent (chrome)")

        except Exception as e:
            _log.error(f"WhatsApp chrome send failed: {e}")
            try:
                pyautogui.hotkey("ctrl", "w")
            except Exception:
                pass
