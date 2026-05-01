"""
email_alerter.py
────────────────
Sends email alerts for high-severity events (weapons, fire, fight, guard missing).

Configuration is loaded from config/rules_config.yaml under the 'email' section.
If email is not configured, this module silently does nothing.

Example config:
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    sender: your-gms-alerts@gmail.com
    password: your-app-password     # Use Gmail App Password, NOT your real password
    recipients:
      - supervisor@company.com
      - security-ops@company.com
    # Only these alert types trigger emails (others stay in dashboard only)
    alert_types:
      - Weapon Detected
      - Unattended Weapon Detected
      - Fire / Smoke Detected
      - Fight / Violence Detected
      - Guard Under Attack
      - Guard Missing
    # Minimum seconds between emails of the same type (prevents spam)
    cooldown: 300

For Gmail: Enable 2FA, then create an App Password at
https://myaccount.google.com/apppasswords
"""

import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailAlerter:
    """
    Thread-safe email alerter with per-type cooldown to prevent spam.
    """

    def __init__(self, config: dict):
        """
        config: the 'email' section from rules_config.yaml
        """
        self._enabled     = config.get("enabled", False)
        self._smtp_server = config.get("smtp_server", "")
        self._smtp_port   = config.get("smtp_port", 587)
        self._sender      = config.get("sender", "")
        self._password    = config.get("password", "")
        self._recipients  = config.get("recipients", [])
        self._alert_types = set(config.get("alert_types", []))
        self._cooldown    = config.get("cooldown", 300)  # 5 min default

        self._last_sent   = {}  # alert_type → last send timestamp
        self._lock        = threading.Lock()

        if self._enabled:
            if not self._smtp_server or not self._sender or not self._recipients:
                print("[EmailAlerter] ✗ Enabled but missing smtp_server/sender/recipients.")
                self._enabled = False
            else:
                print(f"[EmailAlerter] ✓ Ready. Recipients: {self._recipients}")
                print(f"[EmailAlerter]   Alert types: {self._alert_types}")
                print(f"[EmailAlerter]   Cooldown: {self._cooldown}s")
        else:
            print("[EmailAlerter] Disabled (email.enabled=false in config).")

    def should_send(self, alert_type: str) -> bool:
        """Check if this alert type should trigger an email."""
        if not self._enabled:
            return False

        # Check if alert type matches any configured type (partial match)
        matched = False
        for configured_type in self._alert_types:
            if configured_type.lower() in alert_type.lower():
                matched = True
                break
        if not matched:
            return False

        # Check cooldown
        with self._lock:
            now = time.time()
            last = self._last_sent.get(alert_type, 0)
            if (now - last) < self._cooldown:
                return False

        return True

    def send(self, alert: dict):
        """
        Send an email alert in a background thread.
        alert: dict with keys type, guard_id, zone, severity, timestamp
        """
        alert_type = alert.get("type", "")
        if not self.should_send(alert_type):
            return

        with self._lock:
            self._last_sent[alert_type] = time.time()

        # Send in background — don't block the pipeline
        threading.Thread(
            target=self._send_email,
            args=(alert,),
            daemon=True
        ).start()

    def _send_email(self, alert: dict):
        """Actually send the email (runs in background thread)."""
        alert_type = alert.get("type", "Alert")
        guard_id   = alert.get("guard_id", "Unknown")
        zone       = alert.get("zone", "—")
        severity   = alert.get("severity", "high")
        timestamp  = alert.get("timestamp", "")

        subject = f"[GMS ALERT] {alert_type} — {guard_id}"

        body = f"""
        <html>
        <body style="font-family: -apple-system, Arial, sans-serif; padding: 20px;">
            <div style="max-width: 500px; margin: 0 auto; border: 2px solid #ef4444;
                        border-radius: 8px; overflow: hidden;">
                <div style="background: #ef4444; color: white; padding: 16px 20px;">
                    <h2 style="margin: 0; font-size: 18px;">
                        ⚠️ GMS Security Alert
                    </h2>
                </div>
                <div style="padding: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <tr>
                            <td style="padding: 8px 0; color: #666; width: 100px;">Alert</td>
                            <td style="padding: 8px 0; font-weight: 600; color: #dc2626;">
                                {alert_type}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Guard</td>
                            <td style="padding: 8px 0;">{guard_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Zone</td>
                            <td style="padding: 8px 0;">{zone}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Severity</td>
                            <td style="padding: 8px 0; text-transform: uppercase;
                                color: {'#dc2626' if severity == 'high' else '#f59e0b'};">
                                {severity}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Time</td>
                            <td style="padding: 8px 0;">{timestamp}</td>
                        </tr>
                    </table>
                    <p style="margin-top: 16px; font-size: 12px; color: #999;">
                        This is an automated alert from the Guard Monitoring System.
                        Check the dashboard for live video and details.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self._sender
        msg["To"]      = ", ".join(self._recipients)
        msg.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(self._smtp_server, self._smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self._sender, self._password)
                server.sendmail(self._sender, self._recipients, msg.as_string())
            print(f"[EmailAlerter] ✓ Sent: {alert_type} → {self._recipients}")
        except Exception as e:
            print(f"[EmailAlerter] ✗ Send failed: {type(e).__name__}: {e}")
