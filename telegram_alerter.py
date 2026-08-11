"""
telegram_alerter.py
────────────────────
Telegram bot alert delivery — no template approval, no business
verification, works immediately. Used as a fast testing path while
WhatsApp Business Sender registration (Meta approval) is pending.

Same send(alert) interface as WhatsAppAlerter/EmailAlerter, so it drops
into flask_app.py's alert_dispatcher with one extra call, nothing else
changes.

One-time setup (see chat for full walkthrough):
  1. Message @BotFather on Telegram -> /newbot -> copy the token
  2. Start a chat with your new bot (search its username, hit Start)
  3. Message @userinfobot to get your numeric chat ID
  4. Set environment variables:
       TELEGRAM_BOT_TOKEN=7123456789:AAHk...
       TELEGRAM_CHAT_ID=987654321
  5. pip install requests   (almost always already installed)
  6. In rules_config.yaml:
       telegram:
         enabled: true
         alert_types: [...]   # same filtering convention as whatsapp/email
"""

import os
import sys
import time
import queue
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from utils.logger import get_logger
    _log = get_logger("telegram_alerter")
except Exception:
    import logging
    _log = logging.getLogger("telegram_alerter")


class TelegramAlerter:

    def __init__(self, config: dict):
        self._enabled     = config.get("enabled", False)
        self._alert_types = set(config.get("alert_types", []))

        self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")

        self._send_queue = queue.Queue(maxsize=50)

        if not self._enabled:
            _log.info("Telegram alerts disabled (telegram.enabled=false in config).")
            return

        if not self._bot_token or not self._chat_id:
            _log.warning("Telegram enabled but TELEGRAM_BOT_TOKEN / "
                        "TELEGRAM_CHAT_ID env vars not set.")
            self._enabled = False
            return

        try:
            import requests  # noqa: F401
        except ImportError:
            _log.warning("'requests' package not installed. Run: pip install requests")
            self._enabled = False
            return

        # Verify the bot token is valid and reachable before declaring ready
        try:
            import requests
            r = requests.get(
                f"https://api.telegram.org/bot{self._bot_token}/getMe",
                timeout=8
            )
            if r.status_code == 200 and r.json().get("ok"):
                bot_name = r.json()["result"].get("username", "unknown")
                _log.info(f"Telegram ready | bot: @{bot_name} | chat_id configured")
            else:
                _log.warning(f"Telegram bot token check failed: {r.text[:200]}")
                self._enabled = False
                return
        except Exception as e:
            _log.warning(f"Could not verify Telegram bot token (will still try to "
                        f"send): {type(e).__name__}: {e}")

        threading.Thread(target=self._worker, daemon=True, name="telegram-worker").start()

    def _matches(self, alert_type: str) -> bool:
        if not self._alert_types:
            return True
        return any(t.lower() in alert_type.lower() for t in self._alert_types)

    def send(self, alert: dict):
        """Queue an alert for immediate Telegram delivery."""
        if not self._enabled:
            return
        if not self._matches(alert.get("type", "")):
            return
        # Same exclusion convention as email/WhatsApp — never notify on
        # unrecognized/unenrolled person detections regardless of severity.
        if alert.get("type", "").lower().startswith("unknown person"):
            return
        try:
            self._send_queue.put_nowait(alert)
        except queue.Full:
            _log.warning("Telegram send queue full — alert dropped.")

    def _worker(self):
        while True:
            try:
                alert = self._send_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._send_alert_now(alert)
            except Exception as e:
                _log.error(f"Telegram send error: {e}", exc_info=True)
            finally:
                self._send_queue.task_done()
                time.sleep(1)   # small gap, courteous to Telegram's rate limits

    def _send_alert_now(self, alert: dict):
        import requests

        alert_type = alert.get("type", "")
        guard_id   = alert.get("guard_id", "—")
        zone       = alert.get("zone", "—")
        severity   = alert.get("severity", "high")
        timestamp  = alert.get("timestamp", "")
        camera     = alert.get("camera_id", "")

        sev_emoji = {"high": "🔴", "medium": "🟠", "low": "⚪"}.get(severity, "🔴")

        person_line = (
            f"\n👤 <b>Person:</b> {guard_id}"
            if guard_id not in ("Camera", "Post", "—", "", None) else ""
        )
        zone_line = (
            f"\n📍 <b>Zone:</b> {zone}"
            if zone not in ("—", "-", "", None) else ""
        )
        cam_line = f"\n📷 <b>Camera:</b> {camera}" if camera else ""

        text = (
            f"⚠️ <b>NoviSentra Security Alert</b>\n\n"
            f"🚨 <b>Alert:</b> {alert_type}"
            f"{person_line}"
            f"{zone_line}"
            f"{cam_line}\n"
            f"{sev_emoji} <b>Severity:</b> {severity.upper()}\n"
            f"🕐 <b>Time:</b> {timestamp}"
        )

        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if r.status_code == 200 and r.json().get("ok"):
                _log.info(f"[Telegram ✓] Sent: {alert_type} | {guard_id} | Zone {zone}")
            else:
                _log.error(f"Telegram send failed: {r.status_code} {r.text[:300]}")
        except Exception as e:
            _log.error(f"Telegram send failed: {type(e).__name__}: {e}")
