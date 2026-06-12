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
        with open(_rules_config_path, "r") as f:
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

        daily_count += 1
        streamer.update_stats(alerts_today=daily_count)

        with _alert_log_lock:
            _alert_log.append(alert)
            if len(_alert_log) > 500:
                _alert_log.pop(0)

        payload = f"data: {json.dumps(alert)}\n\n"
        with _sse_lock:
            dead = []
            for q in _sse_subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                _sse_subscribers.remove(q)


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
        _session_start = time.time()   # reset timer for next segment's stats

        try:
            session_stats = dict(streamer.stats)

            html = generate_report(
                alert_log=alerts,
                stats=session_stats,
                session_start=_session_start,
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

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(streamer.generate_mjpeg()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/alerts/stream")
def alerts_stream():
    def generate():
        q = queue.Queue(maxsize=200)
        with _sse_lock:
            _sse_subscribers.append(q)

        with _alert_log_lock:
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
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)

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
            with open(config_path, "r") as f:
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
                with open(config_path, "r") as f:
                    current = yaml.safe_load(f) or {}
            except Exception as e:
                return jsonify({"error": f"Could not read config: {e}"}), 500

        deep_merge(current, data)

        try:
            with open(config_path, "w") as f:
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
    )

    # Serve inline (opens in browser tab) or as download
    as_file = request.args.get("download", "false").lower() == "true"
    disposition = "attachment" if as_file else "inline"
    filename = f"GMS_ShiftReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"

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
    """List all saved alert snapshots, newest first."""
    return jsonify(_snap_mgr.list_snapshots())


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
    print("\n[GMS] Shutting down — sending session report...")
    try:
        send_session_report()
        import time as _t
        print("[GMS] Waiting 20s for shift report email to send...")
        _t.sleep(20)
    except Exception as e:
        print(f"[GMS] Report send on exit failed: {e}")


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
        print("\n[GMS] Caught Ctrl+C — saving session log and sending report...")

        # ── 1. Archive the log FIRST — safe regardless of what happens next ──
        archived = save_session_log(
            datetime.datetime.fromtimestamp(_session_start)
        )
        if archived:
            print(f"[GMS] Session log archived → {archived}")

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
        print("[GMS] Waiting 20s for shift report email to send...")
        _t.sleep(20)
        print("[GMS] Done. Exiting.")

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
            print(f"[GMS] Session log archived → {archived}")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Register after _on_exit_guarded is defined
    atexit.register(_on_exit_guarded)

    threading.Thread(target=alert_dispatcher, daemon=True).start()

    if _cam_manager.is_multi_camera:
        _cam_manager.start_all()
        print(f"\n  Multi-camera mode — {len(_cam_manager.cameras)} cameras configured")
    else:
        main_web.start(source=0, source_label="Camera 0")

    print("\n  Dashboard → http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)