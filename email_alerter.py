"""
email_alerter.py
────────────────
Sends email alerts with snapshot attachments + end-of-session report delivery.

v6 — Pure SMTP, no vendor REST API dependency. Works reliably on Hetzner
(or any cloud host) IF pointed at the right kind of SMTP server. This is
the one thing that actually matters here — the protocol was never the
problem, the SERVER you point it at is:

  ✗ smtp.gmail.com          — DO NOT use in production. Gmail treats
                               cloud-datacenter IPs as suspicious and
                               silently drops/defers mail. Works fine
                               from your laptop, breaks silently once
                               deployed. Fine for local dev testing only.

  ✓ smtp-relay.brevo.com    — Brevo's SMTP relay (free tier: 300/day).
    (or any transactional     Still 100% plain SMTP/STARTTLS — no API,
     ESP's SMTP endpoint)     no SDK. Built specifically to be reliable
                               from cloud server IPs.

  ✓ Client's own mail server — For clients who want zero third parties
                               (data-locality/privacy requirements). Same
                               code, just point smtp_server at their
                               server and use their credentials.

Nothing else about this file changed from the original design — same
public interface (send(), send_report(), set_snapshot_manager(),
should_send()), same 3-snapshot-burst attachment behavior, same cooldown
logic. Only the delivery mechanism target changed.
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
        self._password    = os.environ.get("SMTP_PASSWORD", config.get("password", ""))
        self._recipients  = config.get("recipients", [])
        self._alert_types = set(config.get("alert_types", []))
        self._cooldown    = config.get("cooldown", 300)

        self._last_sent = {}
        self._lock      = threading.Lock()
        self._snap_mgr  = None

        if not self._enabled:
            print("[EmailAlerter] Disabled (email.enabled=false in config).")
            return

        if not self._smtp_server or not self._sender or not self._recipients:
            print("[EmailAlerter] ✗ Enabled but missing smtp_server/sender/recipients.")
            self._enabled = False
            return

        if "gmail.com" in self._smtp_server.lower():
            print("[EmailAlerter] ⚠ WARNING: smtp_server is Gmail. This is NOT "
                  "reliable from a cloud server (Hetzner, AWS, etc.) — Gmail "
                  "silently drops/defers mail from datacenter IPs. Use this "
                  "only for local development. For production, point "
                  "smtp_server at a transactional relay (e.g. "
                  "smtp-relay.brevo.com) or the client's own mail server.")

        print(f"[EmailAlerter] ✓ Ready. Server: {self._smtp_server}:{self._smtp_port}. "
              f"{len(self._recipients)} recipient(s) configured.")
        print(f"[EmailAlerter]   {len(self._alert_types)} alert type(s) configured. "
              f"Cooldown: {self._cooldown}s")

    # Alert types to EXCLUDE from email — everything else gets emailed.
    # (Previous versions had a hand-maintained _EMAIL_EXCLUDE keyword list
    # here — replaced by the severity check in should_send() below. Every
    # medium/high alert now always emails, per explicit instruction;
    # only low severity (Patrol/Movement) is skipped.)

    # Severity-based, not keyword-based: "medium" and "high" always send,
    # "low" (Patrol/Movement zone-transition logging) never does. This
    # replaces a hand-maintained exclusion list that had two real bugs —
    # it silently broke for translated alert types ("patrol" stopped
    # matching once alert_manager.py started rendering it as "Movement:"
    # for non-guard profiles), and it wasn't kept in sync with every
    # alert type that should have been excluded/included. Severity is
    # computed ONCE, correctly, in alert_manager.py against the raw
    # canonical type before translation — reusing that value here (from
    # the alert dict, not recomputed from the type string) means this
    # can never drift out of sync with what actually happened.
    def should_send(self, alert_type: str, severity: str = None) -> bool:
        if not self._enabled:
            return False
        if severity == "low":
            return False
        # Explicit exclusion (2026-07): never email "Unknown Person ..."
        # alerts, regardless of severity ("Unknown Person Sleeping" is
        # HIGH, so the severity check above wouldn't have caught it) —
        # per request, an unenrolled/unrecognized person is common
        # enough that emailing every occurrence is noise. Checked against
        # alert_type as passed here (already translated, but "Unknown
        # Person" is never translated per-profile, so this is safe for
        # every profile).
        if alert_type.lower().startswith("unknown person"):
            return False
        with self._lock:
            now = time.time()
            if (now - self._last_sent.get(alert_type, 0)) < self._cooldown:
                return False
        return True

    def reset_cooldowns(self) -> None:
        """
        Reset all per-alert-type email cooldowns. Called on profile switch so
        each profile gets a fresh cooldown window — prevents cross-profile
        cooldown bleed where e.g. a Camera Tamper fired in guard_monitoring
        blocks the same alert type from emailing for 200 seconds in
        bank_security, even though it's a completely different operational context.
        """
        with self._lock:
            self._last_sent.clear()

    def set_snapshot_manager(self, snap_mgr):
        """Wire in the SnapshotManager so emails can attach burst snapshots."""
        self._snap_mgr = snap_mgr

    def send(self, alert: dict, snapshot_paths: list = None):
        """
        Send an alert email. Snapshot attachments are found automatically
        from the SnapshotManager (recent files for this alert type).
        """
        alert_type = alert.get("type", "")
        severity   = alert.get("severity")   # already correct — computed
                                              # against the RAW type in
                                              # alert_manager.py, before
                                              # translation; do NOT
                                              # recompute this from
                                              # alert_type here, it may
                                              # already be translated
        if not self.should_send(alert_type, severity):
            return

        with self._lock:
            self._last_sent[alert_type] = time.time()

        paths = snapshot_paths or []
        threading.Thread(
            target=self._send_with_snap_wait,
            args=(alert, paths),
            daemon=True
        ).start()

    def _send_with_snap_wait(self, alert: dict, explicit_paths: list):
        """Wait for burst snapshots to be written, then send email."""
        alert_type = alert.get("type", "")
        paths = explicit_paths
        if not paths and self._snap_mgr:
            try:
                paths = self._wait_for_snapshots(
                    alert_type,
                    camera_id=alert.get("camera_id"),
                    profile_id=alert.get("profile_id"),
                )
            except Exception:
                paths = []
        if paths:
            print(f"[EmailAlerter] Attaching {len(paths)} snapshot(s) for: {alert_type}")
        else:
            print(f"[EmailAlerter] No snapshots found for: {alert_type}")
        self._send_email(alert, paths)

    def _wait_for_snapshots(self, alert_type: str, camera_id: str = None,
                            profile_id: str = None, timeout: float = 12.0,
                            poll_interval: float = 0.3, expected: int = 3) -> list:
        """
        Poll for the burst snapshot files rather than a fixed sleep-then-
        check. The previous fixed 3-second sleep was a guess at how long
        the 3-frame burst write (t=0, t+1s, t+2s ≈ 2s minimum) takes — but
        under real load (multiple cameras, heavy models all competing for
        CPU, which this system genuinely experiences), the background
        writer thread can legitimately take longer than that guess. When
        it did, the fixed sleep expired first and the email went out with
        zero attachments even though the snapshot WAS being saved, just
        not fast enough. Polling waits exactly as long as actually needed
        (up to `timeout`), and returns whatever's available if the full
        set of 3 never completes in time — partial attachments beat none.
        """
        deadline = time.time() + timeout
        best = []
        while time.time() < deadline:
            found = self._snap_mgr.get_recent_paths(
                alert_type, max_age_secs=timeout + 2,
                camera_id=camera_id, profile_id=profile_id,
            )
            if len(found) >= expected:
                return found
            if len(found) > len(best):
                best = found
            time.sleep(poll_interval)
        return best   # whatever showed up, even if fewer than `expected`

    def send_report(self, report_html: str, stats: dict):
        """Send the shift report as an email attachment at end of session."""
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
                        <tr><td style="padding: 8px 0; color: #666; width: 100px;">Alert</td>
                            <td style="padding: 8px 0; font-weight: 600; color: {sev_color};">{alert_type}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Person</td>
                            <td style="padding: 8px 0;">{guard_id}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Zone</td>
                            <td style="padding: 8px 0;">{zone}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Severity</td>
                            <td style="padding: 8px 0; text-transform: uppercase; color: {sev_color};">{severity}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Time</td>
                            <td style="padding: 8px 0;">{timestamp}</td></tr>
                    </table>
                    {snap_note}
                    <p style="margin-top: 16px; font-size: 12px; color: #999;">
                        Automated alert from NoviSentra. Check dashboard for live video.
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
                        <tr><td style="padding: 8px 0; color: #666; width: 140px;">Session ended</td>
                            <td style="padding: 8px 0;">{time_str}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Total alerts</td>
                            <td style="padding: 8px 0; font-weight: 600;">{total}</td></tr>
                        <tr><td style="padding: 8px 0; color: #666;">Persons monitored</td>
                            <td style="padding: 8px 0;">{guards}</td></tr>
                    </table>
                    <p style="margin-top: 16px; font-size: 12px; color: #999;">
                        Open the attached HTML file in any browser to view the full report.
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
