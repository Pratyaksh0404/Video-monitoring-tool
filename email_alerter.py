"""
email_alerter.py
────────────────
Sends email alerts with snapshot attachments + end-of-session report delivery.

New in v3:
  - Alert emails now attach up to 3 snapshot JPEGs (burst)
  - send_report() sends the shift report HTML as attachment at session end
"""

import threading
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders


class EmailAlerter:

    def __init__(self, config: dict):
        self._enabled     = config.get("enabled", False)
        self._smtp_server = config.get("smtp_server", "")
        self._smtp_port   = config.get("smtp_port", 587)
        self._sender      = config.get("sender", "")
        self._password    = config.get("password", "")
        self._recipients  = config.get("recipients", [])
        self._alert_types = set(config.get("alert_types", []))
        self._cooldown    = config.get("cooldown", 300)

        self._last_sent = {}
        self._lock      = threading.Lock()

        if self._enabled:
            if not self._smtp_server or not self._sender or not self._recipients:
                print("[EmailAlerter] ✗ Enabled but missing smtp_server/sender/recipients.")
                self._enabled = False
            else:
                print(f"[EmailAlerter] ✓ Ready. {len(self._recipients)} recipient(s) configured.")
                print(f"[EmailAlerter]   {len(self._alert_types)} alert type(s) configured.")
                print(f"[EmailAlerter]   Cooldown: {self._cooldown}s")
        else:
            print("[EmailAlerter] Disabled (email.enabled=false in config).")

    # Alert types to EXCLUDE from email — everything else gets emailed
    _EMAIL_EXCLUDE = {
        "guard idle",
        "patrol",
        "unknown person detected",       # generic unknown — not actionable
        "unknown person sleeping",       # not actionable
        "unknown person using phone",    # not actionable
    }

    def should_send(self, alert_type: str) -> bool:
        if not self._enabled:
            return False
        al = alert_type.lower()
        # Exclude guard idle, patrol, and low-value unknown alerts
        if any(ex in al for ex in self._EMAIL_EXCLUDE):
            return False
        with self._lock:
            now = time.time()
            if (now - self._last_sent.get(alert_type, 0)) < self._cooldown:
                return False
        return True

        # Reference to SnapshotManager — set by flask_app after init
        self._snap_mgr = None

    def set_snapshot_manager(self, snap_mgr):
        """Wire in the SnapshotManager so emails can attach burst snapshots."""
        self._snap_mgr = snap_mgr

    def send(self, alert: dict, snapshot_paths: list = None):
        """
        Send an alert email. Snapshot attachments are found automatically
        from the SnapshotManager (recent files for this alert type).
        snapshot_paths: optional explicit list (used when paths are pre-known)
        """
        alert_type = alert.get("type", "")
        if not self.should_send(alert_type):
            return

        with self._lock:
            self._last_sent[alert_type] = time.time()

        # Find burst snapshots — wait 2.5s so all 3 frames finish writing
        # (burst saves at t=0, t+1s, t+2s; we wait 2.5s to ensure all are on disk)
        paths = snapshot_paths or []
        threading.Thread(
            target=self._send_with_snap_wait,
            args=(alert, paths),
            daemon=True
        ).start()

    def _send_with_snap_wait(self, alert: dict, explicit_paths: list):
        """Wait for burst snapshots to be written, then send email."""
        alert_type = alert.get("type", "")
        # Wait 3s so all 3 burst frames (t=0, t+1, t+2) finish writing to disk
        time.sleep(3)
        paths = explicit_paths
        if not paths and self._snap_mgr:
            try:
                paths = self._snap_mgr.get_recent_paths(alert_type, max_age_secs=10.0)
            except Exception:
                paths = []
        if paths:
            print(f"[EmailAlerter] Attaching {len(paths)} snapshot(s) for: {alert_type}")
        else:
            print(f"[EmailAlerter] No snapshots found for: {alert_type}")
        self._send_email(alert, paths)

    def send_report(self, report_html: str, stats: dict):
        """
        Send the shift report as an email attachment at end of session.
        Non-blocking — runs in background thread.
        """
        if not self._enabled:
            return
        threading.Thread(
            target=self._send_report_email,
            args=(report_html, stats),
            daemon=True
        ).start()

    def _send_email(self, alert: dict, snapshot_paths: list):
        """Send alert email with up to 3 snapshot attachments."""
        alert_type = alert.get("type", "Alert")
        guard_id   = alert.get("guard_id", "Unknown")
        zone       = alert.get("zone", "—")
        severity   = alert.get("severity", "high")
        timestamp  = alert.get("timestamp", "")
        sev_color  = "#dc2626" if severity == "high" else "#f59e0b"

        subject = f"[NoviSentra ALERT] {alert_type} — {guard_id}"

        snap_note = ""
        if snapshot_paths:
            snap_note = f'<p style="margin-top:12px;font-size:12px;color:#666;">📷 {len(snapshot_paths)} snapshot(s) attached below.</p>'

        body = f"""
        <html>
        <body style="font-family: -apple-system, Arial, sans-serif; padding: 20px;">
            <div style="max-width: 520px; margin: 0 auto; border: 2px solid #ef4444;
                        border-radius: 8px; overflow: hidden;">
                <div style="background: #ef4444; color: white; padding: 16px 20px;">
                    <h2 style="margin: 0; font-size: 18px;">⚠️ NoviSentra Security Alert</h2>
                </div>
                <div style="padding: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <tr>
                            <td style="padding: 8px 0; color: #666; width: 100px;">Alert</td>
                            <td style="padding: 8px 0; font-weight: 600; color: {sev_color};">{alert_type}</td>
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
                            <td style="padding: 8px 0; text-transform: uppercase; color: {sev_color};">{severity}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Time</td>
                            <td style="padding: 8px 0;">{timestamp}</td>
                        </tr>
                    </table>
                    {snap_note}
                    <p style="margin-top: 16px; font-size: 12px; color: #999;">
                        Automated alert from Guard Monitoring System.
                        Check dashboard for live video.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # Build mixed message so we can have both HTML body + image attachments
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = self._sender
        msg["To"]      = ", ".join(self._recipients)

        # Attach HTML body
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body, "html"))
        msg.attach(alt)

        # Attach snapshot images
        for i, path in enumerate(snapshot_paths[:3], 1):
            try:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    img_data = f.read()
                img = MIMEImage(img_data, name=os.path.basename(path))
                img.add_header("Content-Disposition", "attachment",
                               filename=f"snapshot_{i}_{os.path.basename(path)}")
                msg.attach(img)
            except Exception as e:
                print(f"[EmailAlerter] Snapshot attach failed ({path}): {e}")

        self._smtp_send(msg, alert_type)

    def _send_report_email(self, report_html: str, stats: dict):
        """Send shift report as HTML attachment."""
        import datetime
        now      = datetime.datetime.now()
        date_str = now.strftime("%d %B %Y")
        time_str = now.strftime("%H:%M")
        filename = f"NoviSentra_ShiftReport_{now.strftime('%Y%m%d_%H%M')}.html"

        total   = stats.get("alerts_today", 0)
        guards  = stats.get("guards_detected", 0)
        subject = f"[NoviSentra] Shift Report — {date_str} {time_str}"

        body = f"""
        <html>
        <body style="font-family: -apple-system, Arial, sans-serif; padding: 20px;">
            <div style="max-width: 520px; margin: 0 auto; border: 2px solid #22c55e;
                        border-radius: 8px; overflow: hidden;">
                <div style="background: #0d1117; color: white; padding: 16px 20px;">
                    <h2 style="margin: 0; font-size: 18px;">
                        <span style="color:#22c55e">●</span> NoviSentra Shift Report
                    </h2>
                </div>
                <div style="padding: 20px;">
                    <p style="font-size: 14px; color: #374151; margin-bottom: 16px;">
                        The shift report for <strong>{date_str}</strong> is attached.
                    </p>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <tr>
                            <td style="padding: 8px 0; color: #666; width: 140px;">Session ended</td>
                            <td style="padding: 8px 0;">{time_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Total alerts</td>
                            <td style="padding: 8px 0; font-weight: 600;">{total}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #666;">Guards monitored</td>
                            <td style="padding: 8px 0;">{guards}</td>
                        </tr>
                    </table>
                    <p style="margin-top: 16px; font-size: 12px; color: #999;">
                        Open the attached HTML file in any browser to view the full report.
                        Use Print → Save as PDF to generate a PDF copy.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = self._sender
        msg["To"]      = ", ".join(self._recipients)

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body, "html"))
        msg.attach(alt)

        # Attach HTML report
        report_part = MIMEBase("text", "html")
        report_part.set_payload(report_html.encode("utf-8"))
        encoders.encode_base64(report_part)
        report_part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(report_part)

        self._smtp_send(msg, "Shift Report")

    def _smtp_send(self, msg, label: str):
        """Send via SMTP. Shared by all email types."""
        try:
            with smtplib.SMTP(self._smtp_server, self._smtp_port, timeout=15) as server:
                server.starttls()
                server.login(self._sender, self._password)
                server.sendmail(self._sender, self._recipients, msg.as_string())
            print(f"[EmailAlerter] ✓ Sent: {label}")
        except Exception as e:
            print(f"[EmailAlerter] ✗ Send failed ({label}): {type(e).__name__}: {e}")