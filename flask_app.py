import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["HUGGINGFACE_HUB_CACHE"] = _cache_dir
os.environ["HF_HOME"]               = _cache_dir
os.environ["TORCH_HOME"]            = _cache_dir
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import time
import queue
import threading
import datetime
import uuid
from collections import Counter, defaultdict

from flask import (
    Flask, render_template, Response,
    request, jsonify, stream_with_context
)
from werkzeug.utils import secure_filename

import main_web
from video_streamer import streamer
from alerts.alert_manager import alert_queue
from utils.logger import get_logger, save_session_log
from snapshot_manager import snap_mgr as _snap_mgr

log = get_logger("flask_app")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_rules_config = {}
_rules_config_path = os.path.join(_BASE_DIR, "config", "rules_config.yaml")
if os.path.exists(_rules_config_path):
    try:
        import yaml
        with open(_rules_config_path, "r", encoding="utf-8") as f:
            _rules_config = yaml.safe_load(f) or {}
    except Exception:
        pass

# Email alerter
from email_alerter import EmailAlerter
_email_alerter = EmailAlerter(_rules_config.get("email", {}))

# WhatsApp alerter
from whatsapp_alerter import WhatsAppAlerter
_whatsapp_alerter = WhatsAppAlerter(_rules_config.get("whatsapp", {}))

# Camera manager
from camera_manager import CameraManager
_cam_config_path = os.path.join(_BASE_DIR, "config", "camera_config.yaml")
_cam_manager = CameraManager(_cam_config_path)

# Shift report generator
from shift_report import generate_report

# Wire snapshot manager into email alerter so emails can attach burst snapshots
_email_alerter.set_snapshot_manager(_snap_mgr)

# Session start time — updated each time a new session begins
_session_start    = time.time()


def _get_session_start() -> float:
    """Live getter — api/routes/reports.py needs this to generate correct
    reports instead of the hardcoded 'time.time() - 3600' placeholder
    that was there before (an approximation that was never actually
    connected to the real session start time)."""
    return _session_start


def _dominant_profile(alerts: list) -> str:
    """
    Most common profile_id among a batch of alerts — used to pick which
    terminology (Guard/Worker/Staff, Patrol/Movement) the shift report
    for this batch should use. Falls back to guard_monitoring if the
    batch has no profile_id data (e.g. very old alerts from before that
    field existed).
    """
    from collections import Counter
    profiles = [a.get("profile_id") for a in alerts if a.get("profile_id")]
    if not profiles:
        return "guard_monitoring"
    return Counter(profiles).most_common(1)[0][0]


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

_alert_log      = []
_alert_log_lock = threading.Lock()
_sse_subscribers= []
_sse_lock       = threading.Lock()
_acknowledged_alerts = set()
_ack_lock            = threading.Lock()
_webhook_router = None   # set by _register_v1_api(); used by alert_dispatcher


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def alert_dispatcher():
    today       = datetime.date.today()
    daily_count = 0

    while True:
        try:
            alert = alert_queue.get(timeout=1)
        except queue.Empty:
            if datetime.date.today() != today:
                today = datetime.date.today()
                daily_count = 0
            continue

        alert["id"]           = str(uuid.uuid4())[:8]
        alert["acknowledged"] = False

        # Email ALL alerts except Guard Idle and Patrol
        # (should_send in EmailAlerter handles the filtering)
        _email_alerter.send(alert)

        # WhatsApp batches HIGH severity alerts
        if alert.get("severity") == "high":
            _whatsapp_alerter.send(alert)

        # Fanout to webhook/other backends via AlertRouter.
        # Must be called AFTER email/WA so alert already has id/acknowledged.
        # Uses a separate internal queue inside the router — does NOT race
        # with this dispatcher for the main alert_queue (that was the bug
        # that caused ~25% of alerts to be silently dropped: two threads
        # both calling .get() on the same queue).
        if _webhook_router is not None:
            _webhook_router.fanout(alert)

        daily_count += 1
        streamer.update_stats(alerts_today=daily_count)

        with _alert_log_lock:
            _alert_log.append(alert)
            if len(_alert_log) > 500:
                _alert_log.pop(0)

        payload = f"data: {json.dumps(alert)}\n\n"
        alert_camera  = alert.get("camera_id")
        alert_profile = alert.get("profile_id")
        with _sse_lock:
            dead = []
            for entry in _sse_subscribers:
                q         = entry["queue"]
                cam_filter = entry["camera_id"]
                prof_filter = entry.get("profile_id")
                # No camera filter = receive everything (external/API
                # firehose connections). A camera-scoped connection also
                # only gets alerts matching the profile that camera had
                # WHEN THE CONNECTION OPENED — this is what makes
                # switching profiles on the same camera correctly stop
                # showing that camera's older-profile alerts, live, not
                # just after a fresh reconnect.
                if cam_filter is not None and alert_camera != cam_filter:
                    continue
                if prof_filter is not None and alert_profile != prof_filter:
                    continue
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                _sse_subscribers[:] = [e for e in _sse_subscribers if e["queue"] is not q]


# Guards against send_session_report being triggered concurrently
# (e.g. source-switch + Ctrl+C arriving within milliseconds of each other)
_session_report_lock = threading.Lock()


def send_session_report():
    """
    Generate and deliver the shift report for the CURRENT (ending) session.
    Captures and CLEARS the alert log atomically, then closes the session log
    and opens a fresh one.

    Only runs for real sessions (>=3 alerts). Short/empty switches are ignored
    so no ghost log files are created.

    Re-entrant calls are dropped via _session_report_lock — the first caller
    owns the session boundary; any concurrent duplicate call returns immediately.
    """
    global _session_start

    # Drop duplicate concurrent calls (source-switch + atexit race)
    if not _session_report_lock.acquire(blocking=False):
        log.info("Session report already in progress — skipping duplicate call.")
        return

    try:
        # Capture AND clear the alert log atomically so a second call that
        # somehow slips through always sees an empty list.
        with _alert_log_lock:
            alerts = list(_alert_log)
            _alert_log.clear()

        # Write to each profile's own session log file, regardless of
        # whether this batch is big enough to also trigger an email below
        # — the per-profile log should capture everything, even a short
        # segment; only the EMAIL send has a minimum-size threshold.
        try:
            from utils.logger import save_profile_alert_log
            written = save_profile_alert_log(alerts)
            if written:
                log.info(f"Per-profile session log(s) updated: "
                        f"{', '.join(f'{p} ({os.path.basename(f)})' for p, f in written.items())}")
        except Exception as e:
            log.warning(f"Could not write per-profile session log: {e}")

        # Skip if there's nothing new to report (e.g. Ctrl+C fired shortly
        # after a manual "Send Report" from the dashboard already cleared the
        # alert log, or a source-switch happened with very few events).
        if len(alerts) < 3:
            log.info("Session too short (fewer than 3 alerts) — skipping report delivery.")
            return

        # ── Send the shift report ─────────────────────────────────────────
        # NOTE: we do NOT archive / rotate the log file here.
        # system.log runs continuously for the entire program run.
        # Archiving to sessions/session_YYYY-MM-DD_HH-MM-SS.log happens
        # only once — in _sigint_handler / _on_exit_guarded on real exit.
        # This means one source-switch = one report email, but still only
        # one session log file per run in the sessions/ folder.
        # Capture the CURRENT segment's actual start time for the report
        # about to be generated, THEN reset for the next segment — order
        # matters. The previous version reset _session_start to time.time()
        # BEFORE generating the report and passed that same freshly-reset
        # value in as session_start, so every report computed its own
        # duration against a timestamp taken moments before generation —
        # which is why duration always showed ~0m 00s regardless of how
        # long the segment actually ran (confirmed: 44 real alerts, but
        # "Session: 09:42 – 09:42 (0m 00s)").
        report_session_start = _session_start
        _session_start = time.time()   # reset timer for next segment's stats

        try:
            session_stats = dict(streamer.stats)

            html = generate_report(
                alert_log=alerts,
                stats=session_stats,
                session_start=report_session_start,
                profile_id=_dominant_profile(alerts),
            )

            # Email report (HTML attached)
            _email_alerter.send_report(html, session_stats)

            # WhatsApp summary (text only, high+smoking incidents)
            #_whatsapp_alerter.send_session_summary(alerts, session_stats)

            log.info("Session report sent via email")
        except Exception as e:
            log.error(f"Session report send failed: {e}", exc_info=True)

    finally:
        _session_report_lock.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/login")
def login_page():
    # If already logged in, don't show the login form again — go straight
    # to the dashboard (avoids confusing "why is it asking me to log in
    # when I already did" on a fresh page load).
    from api.middleware.session_auth import get_current_user
    from flask import redirect
    if get_current_user() is not None:
        return redirect("/")
    return render_template("login.html")


# ═══════════════════════════════════════════════════════════════════════════
# Universal login gate — EVERY route requires an authenticated session
# except the small explicit allowlist below. This closes the gap where
# only "/" was gated but /video_feed, /api/stats, /alerts/stream, etc.
# were still reachable without logging in first — "we can't see someone
# else's profile's things, everywhere should be like this."
#
# /api/v1/* is deliberately exempted from THIS check — those routes carry
# their OWN auth (X-API-Key for external business integrations via
# api.middleware.auth.require_api_key, or session-based @require_login /
# @require_admin for the auth/admin panel routes) — they're not
# unprotected, just protected by a different, more appropriate mechanism
# for programmatic/API-key access rather than a browser-session redirect.
# ═══════════════════════════════════════════════════════════════════════════
_PUBLIC_PATHS = {"/login", "/api/v1/auth/login", "/api/v1/health"}


@app.before_request
def _require_session_everywhere():
    path = request.path

    if path in _PUBLIC_PATHS:
        return None
    if path.startswith("/static/"):
        return None
    if path.startswith("/api/v1/"):
        # Own auth mechanism already applied at the route level — see
        # docstring above. Not a bypass, a different (correct) gate.
        return None

    from api.middleware.session_auth import get_current_user
    if get_current_user() is not None:
        return None   # logged in — proceed

    # Not logged in. JSON 401 for AJAX/streaming calls the dashboard's own
    # JS makes (so it fails cleanly instead of getting HTML back), redirect
    # to /login for an actual page load.
    if (path.startswith("/api/") or path.startswith("/video_feed")
            or path == "/alerts/stream"):
        return jsonify({"error": "Not authenticated", "login_required": True}), 401

    from flask import redirect
    return redirect("/login")


@app.route("/admin")
def admin_page():
    from api.middleware.session_auth import get_current_user
    from flask import redirect
    user = get_current_user()
    if user is None:
        return redirect("/login")
    if not user.is_admin:
        return redirect("/")
    return render_template("admin.html")


@app.route("/")
def index():
    from api.middleware.session_auth import get_current_user
    from flask import redirect
    if get_current_user() is None:
        return redirect("/login")
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(streamer.generate_mjpeg()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/alerts/stream")
def alerts_stream():
    # Camera-scoped AND profile-scoped. Camera-only scoping (the previous
    # version) assumed different profiles only ever run on different
    # cameras — true for a real customer (one licensed profile, applied
    # to every camera they have), but NOT true for our own testing
    # pattern of switching profiles on the SAME single camera to compare
    # scenarios. Without also checking the camera's CURRENT profile,
    # switching from guard_monitoring to retail_analytics on cam_1 would
    # still show cam_1's old guard_monitoring alerts mixed in — exactly
    # the bug confirmed by testing (alert panel showing "Guard Sleeping",
    # "Worker Sleeping", and "Staff Smoking" all together for one camera).
    requested_camera = request.args.get("camera") or None

    def _current_profile_for(cam_id):
        try:
            from alerts.alert_manager import get_camera_profile
            return get_camera_profile(cam_id)
        except Exception:
            return None

    def generate():
        q = queue.Queue(maxsize=200)
        current_profile = _current_profile_for(requested_camera) if requested_camera else None
        with _sse_lock:
            _sse_subscribers.append({
                "queue": q,
                "camera_id": requested_camera,
                "profile_id": current_profile,
            })

        with _alert_log_lock:
            if requested_camera:
                history = [a for a in _alert_log
                          if a.get("camera_id") == requested_camera
                          and (current_profile is None or a.get("profile_id") == current_profile)
                          ][-50:]
            else:
                history = list(_alert_log[-50:])
        for alert in history:
            yield f"data: {json.dumps(alert)}\n\n"

        try:
            while True:
                try:
                    yield q.get(timeout=20)
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with _sse_lock:
                _sse_subscribers[:] = [e for e in _sse_subscribers if e["queue"] is not q]

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(streamer.stats)


@app.route("/api/occupancy")
def api_occupancy():
    s = streamer.stats
    return jsonify({
        "people_count":      s.get("people_count", 0),
        "dwell_times":       s.get("dwell_times", {}),
        "compliance_scores": s.get("compliance_scores", {}),
    })


@app.route("/api/analytics")
def api_analytics():
    with _alert_log_lock:
        alerts = list(_alert_log)

    type_counts        = Counter()
    severity_counts    = Counter({"high": 0, "medium": 0, "low": 0})
    zone_counts        = Counter()
    guard_alert_counts = Counter()
    hourly_counts      = defaultdict(int)
    guard_zones        = defaultdict(set)

    for a in alerts:
        alert_type = a.get("type", "")
        severity   = a.get("severity", "low")
        zone       = a.get("zone", "")
        guard      = a.get("guard_id", "")
        timestamp  = a.get("timestamp", "")

        if alert_type.startswith("Patrol:"):
            path = alert_type.replace("Patrol:", "").strip()
            for z in path.replace(" ", "").split("->"):
                if z in ("A", "B", "C", "D"):
                    guard_zones[guard].add(z)
            type_counts["Patrol"] += 1
        elif alert_type.startswith("Weapon Detected"):
            type_counts["Weapon Detected"] += 1
        elif alert_type.startswith("Threat Detected"):
            type_counts["Threat Detected"] += 1
        else:
            type_counts[alert_type] += 1

        severity_counts[severity] += 1
        if zone and zone not in ("—", "-", ""):
            zone_counts[zone] += 1
        if guard and guard not in ("Post", ""):
            guard_alert_counts[guard] += 1
        if timestamp and len(timestamp) >= 2:
            hour_key = timestamp[:2] + ":00"
            hourly_counts[hour_key] += 1

    guard_compliance = {
        g: int(len(zones) / 4 * 100)
        for g, zones in guard_zones.items()
    }

    return jsonify({
        "type_counts":        dict(type_counts),
        "severity_counts":    dict(severity_counts),
        "zone_counts":        dict(zone_counts),
        "guard_alert_counts": dict(guard_alert_counts),
        "guard_compliance":   guard_compliance,
        "hourly_counts":      dict(hourly_counts),
        "total":              len(alerts),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    stats = streamer.stats
    if not stats.get("fps") or stats.get("fps") == "—":
        return jsonify({"error": "Camera not ready yet. Please wait a few seconds and try again."}), 503

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_bytes = f.read()

    def save_and_start():
        with open(filepath, "wb") as out:
            out.write(file_bytes)
        main_web.start(source=filepath, source_label=filename)

    threading.Thread(target=save_and_start, daemon=True).start()
    return jsonify({"ok": True, "file": filename})


@app.route("/api/source/webcam", methods=["POST"])
def api_webcam():
    main_web.start(source=0, source_label="Camera 0")
    return jsonify({"ok": True})


@app.route("/api/alerts/export")
def api_export():
    fmt = request.args.get("format", "json")
    with _alert_log_lock:
        alerts = list(_alert_log)
    if fmt == "csv":
        lines = ["timestamp,type,guard_id,zone,severity,acknowledged"]
        for a in alerts:
            ack = "yes" if a.get("acknowledged") else "no"
            lines.append(f"{a['timestamp']},{a['type']},{a['guard_id']},{a['zone']},{a['severity']},{ack}")
        return Response("\n".join(lines), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=alerts.csv"})
    return Response(json.dumps(alerts, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=alerts.json"})


@app.route("/api/alerts/acknowledge", methods=["POST"])
def api_acknowledge():
    data     = request.get_json(silent=True) or {}
    alert_id = data.get("id")
    if not alert_id:
        return jsonify({"error": "Missing alert id"}), 400

    with _alert_log_lock:
        for a in _alert_log:
            if a.get("id") == alert_id:
                a["acknowledged"] = True
                break

    with _ack_lock:
        _acknowledged_alerts.add(alert_id)

    return jsonify({"ok": True, "id": alert_id})


@app.route("/api/config")
def api_config():
    config_path = os.path.join(_BASE_DIR, "config", "rules_config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return jsonify(cfg)
        except ImportError:
            return jsonify({"error": "PyYAML not installed"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Config file not found"}), 404


@app.route("/api/config/update", methods=["POST"])
def api_config_update():
    """
    Receive a partial config dict from the Settings UI, deep-merge it into
    the existing YAML, write it back to disk, then hot-reload main_web globals.

    Uses ruamel.yaml for the read+write cycle so that all comments, blank lines,
    and key order in rules_config.yaml are preserved exactly.
    Only the keys sent by the client are updated — everything else is untouched.

    Falls back to plain PyYAML if ruamel.yaml is not installed (comments lost,
    but values are still saved correctly).
    """
    config_path = os.path.join(_BASE_DIR, "config", "rules_config.yaml")

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # ── Deep merge helper (works on both plain dicts and ruamel CommentedMaps) ─
    def deep_merge(base, updates):
        for k, v in updates.items():
            if isinstance(v, dict) and k in base and hasattr(base[k], 'items'):
                deep_merge(base[k], v)
            else:
                base[k] = v

    # ── Try ruamel.yaml first (preserves comments) ────────────────────────────
    try:
        from ruamel.yaml import YAML
        ry = YAML()
        ry.preserve_quotes = True

        # Load — ruamel returns a CommentedMap that carries all comment metadata
        with open(config_path, "r", encoding="utf-8") as f:
            current = ry.load(f) or {}

        deep_merge(current, data)

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(current, f)

        log.info("Config saved with ruamel.yaml (comments preserved)")

    except ImportError:
        # ── Fallback: plain PyYAML (comments will be lost) ────────────────────
        import yaml
        log.warning(
            "ruamel.yaml not installed — comments in rules_config.yaml will be "
            "stripped on save. Run: pip install ruamel.yaml"
        )

        current = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    current = yaml.safe_load(f) or {}
            except Exception as e:
                return jsonify({"error": f"Could not read config: {e}"}), 500

        deep_merge(current, data)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(current, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
        except Exception as e:
            return jsonify({"error": f"Could not write config: {e}"}), 500

    except Exception as e:
        return jsonify({"error": f"Could not write config: {e}"}), 500

    # ── Hot-reload pipeline globals ───────────────────────────────────────────
    try:
        main_web.reload_config()
    except Exception as e:
        log.warning(f"Config written but reload failed: {e}")
        return jsonify({"ok": True, "warning": f"Saved but reload failed: {e}"})

    log.info("Config updated and reloaded via UI")
    return jsonify({"ok": True})


@app.route("/api/perf")
def api_perf():
    log_path = os.path.join(_BASE_DIR, "src", "logs", "system.log")
    recent_logs = []
    perf_stats  = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            recent_logs = [l.rstrip() for l in lines[-100:]]
            for line in reversed(lines):
                if "Perf stats |" in line and "(no data)" not in line:
                    parts = line.split("Perf stats |", 1)
                    if len(parts) > 1:
                        for segment in parts[1].strip().split(" | "):
                            if ":" in segment and "ms" in segment:
                                key, val = segment.split(":", 1)
                                avg_ms = float(val.strip().replace("ms avg", "").replace("ms", "").strip())
                                perf_stats[key.strip()] = {"avg_ms": avg_ms, "calls": 1}
                    break
        except Exception:
            pass
    return jsonify({"log_lines": recent_logs, "log_path": log_path, "perf_stats": perf_stats})


# ── Shift Report ──────────────────────────────────────────────────────────────

@app.route("/api/report/send", methods=["POST"])
def api_report_send():
    """Manually trigger session report delivery via email and WhatsApp."""
    threading.Thread(target=send_session_report, daemon=True).start()
    return jsonify({"ok": True, "message": "Report delivery started"})


@app.route("/api/report/generate")
def api_report_generate():
    """Generate a Shift Intelligence Report and serve it as HTML."""
    with _alert_log_lock:
        alerts = list(_alert_log)

    html = generate_report(
        alert_log=alerts,
        stats=streamer.stats,
        session_start=_session_start,
        profile_id=_dominant_profile(alerts),
    )

    # Serve inline (opens in browser tab) or as download
    as_file = request.args.get("download", "false").lower() == "true"
    disposition = "attachment" if as_file else "inline"
    filename = f"NoviSentra_ShiftReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"

    return Response(
        html,
        mimetype="text/html",
        headers={
            "Content-Disposition": f"{disposition}; filename={filename}"
        }
    )


@app.route("/api/timeline")
def api_timeline():
    """
    Build per-guard activity timeline from the alert log.
    Returns events grouped by guard with timestamps, used to render
    a color-coded horizontal timeline strip in the Analytics tab.
    """
    with _alert_log_lock:
        alerts = list(_alert_log)

    if not alerts:
        return jsonify({"guards": {}, "session_start": None, "session_end": None})

    # Collect per-guard events sorted by time
    from collections import defaultdict
    guard_events = defaultdict(list)

    # Parse HH:MM:SS timestamps into comparable values (seconds since midnight)
    def ts_to_secs(ts):
        try:
            parts = ts.split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except Exception:
            return 0

    for a in alerts:
        guard = a.get("guard_id", "")
        atype = a.get("type", "")
        ts    = a.get("timestamp", "")
        sev   = a.get("severity", "low")

        # Skip Camera/Post entries and patrols for named guards
        if not guard or guard in ("Camera", "Post"):
            continue

        guard_events[guard].append({
            "type":      atype,
            "severity":  sev,
            "timestamp": ts,
            "secs":      ts_to_secs(ts),
            "zone":      a.get("zone", ""),
        })

    if not guard_events:
        return jsonify({"guards": {}, "session_start": None, "session_end": None})

    # Find session time range across all events
    all_secs = [e["secs"] for evs in guard_events.values() for e in evs if e["secs"] > 0]
    session_start = min(all_secs) if all_secs else 0
    session_end   = max(all_secs) if all_secs else 0

    # Sort events per guard
    result = {}
    for guard, events in guard_events.items():
        result[guard] = sorted(events, key=lambda e: e["secs"])

    return jsonify({
        "guards":        result,
        "session_start": session_start,
        "session_end":   session_end,
    })


@app.route("/api/snapshots")
def api_snapshots():
    """List saved alert snapshots, newest first. ?camera=<cam_id> scopes
    to just that camera's photos, AND automatically scopes to that
    camera's CURRENT profile too — without this second part, a camera
    tested under guard_monitoring, then retail_analytics, then
    bank_security in one session shows all three profiles' snapshots
    mixed together in the analytics tab, which is exactly what was
    happening (same root issue the alert history had, fixed the same way)."""
    camera_id = request.args.get("camera") or None
    profile_id = None
    if camera_id:
        try:
            from alerts.alert_manager import get_camera_profile
            profile_id = get_camera_profile(camera_id)
        except Exception:
            pass
    return jsonify(_snap_mgr.list_snapshots(camera_id=camera_id, profile_id=profile_id))


@app.route("/api/snapshots/<filename>")
def api_snapshot_image(filename):
    """Serve a snapshot image by filename."""
    path = _snap_mgr.get_path(filename)
    if not path:
        return "Not found", 404
    from flask import send_file
    return send_file(path, mimetype="image/jpeg")


# ── Multi-camera API ──────────────────────────────────────────────────────────

@app.route("/api/cameras")
def api_cameras():
    return jsonify({
        "multi_camera": _cam_manager.is_multi_camera,
        "default":      _cam_manager.default_camera,
        "cameras":      _cam_manager.get_cameras_list(),
    })


@app.route("/video_feed/<cam_id>")
def video_feed_cam(cam_id):
    cam_streamer = _cam_manager.get_streamer(cam_id)
    if not cam_streamer:
        return "Camera not found", 404
    return Response(
        stream_with_context(cam_streamer.generate_mjpeg()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/cameras/<cam_id>/stats")
def api_camera_stats(cam_id):
    cam_streamer = _cam_manager.get_streamer(cam_id)
    if not cam_streamer:
        return jsonify({"error": "Camera not found"}), 404
    return jsonify(cam_streamer.stats)


# ── Startup ───────────────────────────────────────────────────────────────────

_exit_triggered = False


def _on_exit():
    """Auto-send session report and archive the session log on normal exit."""
    print("\n[NoviSentra] Shutting down — sending session report...")
    try:
        send_session_report()
        import time as _t
        print("[NoviSentra] Waiting 20s for shift report email to send...")
        _t.sleep(20)
    except Exception as e:
        print(f"[NoviSentra] Report send on exit failed: {e}")



# ═══════════════════════════════════════════════════════════════════════════════
# NoviSentra REST API v1
# Registered here (after all existing routes/vars are defined) so imports
# can reference _alert_log, _alert_log_lock, send_session_report, etc.
# All /api/v1/* endpoints are additive — nothing existing changes.
# ═══════════════════════════════════════════════════════════════════════════════
def _register_v1_api():
    try:
        from novisentra.alerting.webhook_backend import WebhookBackend
        from novisentra.alerting.router import AlertRouter
        from novisentra.alerting.email_backend import EmailAlertBackend
        from novisentra.alerting.whatsapp_backend import WhatsAppAlertBackend

        # ── Webhook backend (singleton) ───────────────────────────────────────
        webhook_backend = WebhookBackend(
            persist_path=os.path.join(_BASE_DIR, "config", "webhooks.json")
        )

        # ── Alert router — fans out alert_queue to all backends ───────────────
        # NOTE: router now uses its own internal fanout queue (not alert_queue
        # directly) to avoid racing with alert_dispatcher. alert_dispatcher
        # calls router.fanout(alert) after each alert is processed.
        router = AlertRouter(alert_queue)
        router.register(webhook_backend)
        router.start()

        global _webhook_router
        _webhook_router = router

        # ── Pipeline state (shared with health endpoint) ──────────────────────
        pipeline_state = {
            "running":       False,
            "source_label":  "—",
            "anomaly_ready": False,
            "clip_ready":    False,
        }

        # Update pipeline state when pipeline starts
        _orig_start = main_web.start
        def _patched_start(source=0, source_label="Camera 0"):
            pipeline_state["running"]      = True
            pipeline_state["source_label"] = source_label
            _orig_start(source=source, source_label=source_label)
        main_web.start = _patched_start

        # ── SSE generator reusable function ───────────────────────────────────
        def _sse_gen():
            """
            Same SSE logic as /alerts/stream, reused by /api/v1/alerts/live.
            This one is deliberately UNFILTERED (camera_id=None) — it's the
            external/API-integration stream (protected by API key), meant
            to give a business consuming the REST API the full alert
            firehose for whichever camera(s) their key has access to, not
            scoped to "whichever camera a dashboard happens to be viewing."
            """
            import queue as _q
            sub_q = _q.Queue(maxsize=200)
            with _sse_lock:
                _sse_subscribers.append({"queue": sub_q, "camera_id": None})
            with _alert_log_lock:
                history = list(_alert_log[-50:])
            for alert in history:
                yield f"data: {json.dumps(alert)}\n\n"
            try:
                while True:
                    try:
                        yield sub_q.get(timeout=20)
                    except _q.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                with _sse_lock:
                    _sse_subscribers[:] = [e for e in _sse_subscribers if e["queue"] is not sub_q]

        # ── Register blueprints ───────────────────────────────────────────────
        import api.routes.health    as _r_health
        import api.routes.stream    as _r_stream
        import api.routes.alerts    as _r_alerts
        import api.routes.webhooks  as _r_webhooks
        import api.routes.analytics as _r_analytics
        import api.routes.config    as _r_config
        import api.routes.reports   as _r_reports
        import api.routes.auth        as _r_auth
        import api.routes.admin       as _r_admin
        import api.routes.recordings  as _r_recordings

        _r_health.register(app, pipeline_state)
        _r_stream.register(app, main_web, streamer)
        _r_alerts.register(app, _alert_log, _alert_log_lock, _sse_gen)
        _r_webhooks.register(app, webhook_backend)
        _r_analytics.register(app, streamer, _snap_mgr, _alert_log, _alert_log_lock)
        _r_config.register(app, api_config_update, api_config, main_web, _cam_manager,
                           email_alerter=_email_alerter)
        _r_reports.register(app, send_session_report, generate_report,
                            streamer, _alert_log, _alert_log_lock,
                            get_session_start_fn=_get_session_start,
                            dominant_profile_fn=_dominant_profile)

        # ── Auth + admin panel (user accounts, camera/zone rename) ────────────
        from novisentra.auth import init_db, revoke_all_sessions
        init_db()
        # Every server start requires a fresh login — a session cookie
        # from before a restart must not silently keep working. SQLite
        # sessions otherwise persist for 7 days across restarts by
        # design (that's fine for "stay logged in within one running
        # session"), but "system just started" should always mean
        # "log in again," explicitly requested.
        cleared = revoke_all_sessions()
        if cleared:
            log.info(f"Cleared {cleared} existing session(s) — fresh login required.")
        _r_auth.register(app)
        _r_admin.register(app, _cam_manager)

        # ── Recorded-video batch analysis (8-9hr shift processing) ────────────
        from novisentra.pipeline.batch_jobs import init_batch_jobs_table
        init_batch_jobs_table()
        _r_recordings.register(app)

        log.info("NoviSentra REST API v1 registered — /api/v1/*")

    except Exception as _err:
        log.warning(
            f"REST API v1 registration failed: {_err} "
            f"(dashboard continues to work normally)",
            exc_info=True,   # ← now logs the full traceback, not just the message
        )


_register_v1_api()


if __name__ == "__main__":
    import atexit
    import signal

    _exit_triggered = False

    def _sigint_handler(signum, frame):
        """Ctrl+C — archive log, send report, wait for email, exit."""
        global _exit_triggered
        if _exit_triggered:
            # Second Ctrl+C — force-quit immediately
            import sys
            sys.exit(1)
        _exit_triggered = True
        print("\n[NoviSentra] Caught Ctrl+C — saving session log and sending report...")

        # ── 1. Archive the log FIRST — safe regardless of what happens next ──
        archived = save_session_log(
            datetime.datetime.fromtimestamp(_session_start)
        )
        if archived:
            print(f"[NoviSentra] Session log archived → {archived}")

        # ── 2. Send the report ────────────────────────────────────────────────
        try:
            send_session_report()
        except Exception:
            pass

        # ── 3. Wait for email daemon thread to finish ─────────────────────────
        # EmailAlerter.send_report() fires SMTP on a daemon thread.
        # Daemon threads are killed the instant the process exits, so we must
        # hold the main thread open long enough for the send to complete.
        # SMTP connect + Gmail auth + send = typically 4-10s on a good connection.
        # We wait a flat 20s — safe margin with no false-early-exit risk.
        import time as _t
        print("[NoviSentra] Waiting 20s for shift report email to send...")
        _t.sleep(20)
        print("[NoviSentra] Done. Exiting.")

        import sys
        sys.exit(0)

    def _on_exit_guarded():
        """atexit wrapper — only runs if SIGINT didn't already handle everything."""
        if _exit_triggered:
            # SIGINT handler already archived the log and is waiting for email.
            # Nothing left to do here.
            return
        _on_exit()
        # Archive log after report is sent (normal exit path)
        archived = save_session_log(
            datetime.datetime.fromtimestamp(_session_start)
        )
        if archived:
            print(f"[NoviSentra] Session log archived → {archived}")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Register after _on_exit_guarded is defined
    atexit.register(_on_exit_guarded)

    threading.Thread(target=alert_dispatcher, daemon=True).start()

    # Pre-warm CLIP synchronously before starting any camera.
    # This means camera threads call get_shared_clip() and get an instant
    # cache hit — no model loading happens on the camera thread at all.
    # This avoids all previous multi-camera CLIP loading race conditions.
    print("[Startup] Pre-loading CLIP model (runs once)...")
    try:
        from analytics.behavior_classifier import get_shared_clip
        get_shared_clip()
        print("[Startup] CLIP ready.")
    except Exception as e:
        print(f"[Startup] CLIP pre-load failed (behavior detection disabled): {e}")

    # Pre-load face encodings synchronously. This is the main startup
    # bottleneck — 39 images × dlib encoding = ~5 min on CPU per camera
    # if not cached. After first run, the cache is read in <1s.
    print("[Startup] Pre-loading face encodings...")
    try:
        import os as _os
        _faces_dir = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "src", "data", "enrolled_faces"
        )
        main_web.load_enrolled_faces(_faces_dir)
        print("[Startup] Face encodings ready.")
    except Exception as e:
        print(f"[Startup] Face encoding pre-load failed: {e}")

    if _cam_manager.is_multi_camera:
        _cam_manager.start_all()
        print(f"\n  Multi-camera mode — {len(_cam_manager.cameras)} cameras configured")
    else:
        main_web.start(source=0, source_label="Camera 0")

    print("\n  NoviSentra → http://127.0.0.1:5000/login\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)