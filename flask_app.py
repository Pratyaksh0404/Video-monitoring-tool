import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Set model cache BEFORE any other imports
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


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

_alert_log      = []
_alert_log_lock = threading.Lock()
_sse_subscribers= []
_sse_lock       = threading.Lock()

# Track acknowledged alert IDs
_acknowledged_alerts = set()
_ack_lock            = threading.Lock()


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def alert_dispatcher():
    """Drain alert_queue → log + fan out to all SSE clients."""
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

        # Add unique ID and acknowledged flag to each alert
        alert["id"] = str(uuid.uuid4())[:8]
        alert["acknowledged"] = False

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

        # Send recent history on connect so log panel isn't empty
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


@app.route("/api/analytics")
def api_analytics():
    """
    Compute analytics from the in-memory alert log.
    """
    with _alert_log_lock:
        log = list(_alert_log)

    type_counts       = Counter()
    severity_counts   = Counter({"high": 0, "medium": 0, "low": 0})
    zone_counts       = Counter()
    guard_alert_counts= Counter()
    hourly_counts     = defaultdict(int)
    guard_zones       = defaultdict(set)

    for a in log:
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
        "total":              len(log),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    # Guard: don't accept upload if pipeline hasn't started yet
    stats = streamer.stats
    if not stats.get("fps") or stats.get("fps") == "—":
        return jsonify({"error": "Camera not ready yet. Please wait a few seconds and try again."}), 503

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file_bytes = f.read()

    def save_and_start():
        with open(filepath, "wb") as out:
            out.write(file_bytes)
        print(f"[upload] Saved {filename} ({len(file_bytes)//1024}KB)")
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
        log = list(_alert_log)
    if fmt == "csv":
        lines = ["timestamp,type,guard_id,zone,severity,acknowledged"]
        for a in log:
            ack = "yes" if a.get("acknowledged") else "no"
            lines.append(f"{a['timestamp']},{a['type']},{a['guard_id']},{a['zone']},{a['severity']},{ack}")
        return Response("\n".join(lines), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=alerts.csv"})
    return Response(json.dumps(log, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=alerts.json"})


# ── NEW: Alert acknowledgement ────────────────────────────────────────────────

@app.route("/api/alerts/acknowledge", methods=["POST"])
def api_acknowledge():
    """Mark an alert as acknowledged by its ID."""
    data = request.get_json(silent=True) or {}
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


# ── NEW: Config endpoint ─────────────────────────────────────────────────────

@app.route("/api/config")
def api_config():
    """Return current rules_config.yaml values."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "rules_config.yaml"
    )
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


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=alert_dispatcher, daemon=True).start()
    main_web.start(source=0, source_label="Camera 0")
    print("\n  Dashboard → http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)