"""
api/routes/stream.py
──────────────────────
Stream control endpoints:

  POST /api/v1/stream/start   — start pipeline on a source
  POST /api/v1/stream/stop    — stop pipeline
  GET  /api/v1/stream/status  — current source, fps, pipeline state

These call the same main_web.start() / main_web.stop() that the
dashboard source-switch already uses. No new pipeline code needed.
"""

from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key

stream_bp = Blueprint("stream_v1", __name__)


def register(app, main_web, streamer):
    """
    main_web: the main_web module (has start() / stop())
    streamer: the VideoStreamer singleton (has .stats)
    """

    @stream_bp.route("/api/v1/stream/start", methods=["POST"])
    @require_api_key
    def stream_start():
        """
        Start the pipeline on a given source.

        Body (JSON):
          source        int | str   Camera index or file path or RTSP URL
          source_label  str         Human-readable label (optional)

        Example:
          {"source": 0, "source_label": "Front Gate"}
          {"source": "rtsp://192.168.1.10/stream1", "source_label": "Cam A"}
          {"source": "/path/to/video.mp4", "source_label": "Recorded"}
        """
        body = request.get_json(silent=True) or {}
        source = body.get("source", 0)
        label  = body.get("source_label", f"Camera {source}")

        # Convert string digit to int for webcam index
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        main_web.start(source=source, source_label=label)
        return jsonify({
            "ok":     True,
            "source": str(source),
            "label":  label,
        })

    @stream_bp.route("/api/v1/stream/stop", methods=["POST"])
    @require_api_key
    def stream_stop():
        """Stop the running pipeline."""
        main_web.stop()
        return jsonify({"ok": True})

    @stream_bp.route("/api/v1/stream/status", methods=["GET"])
    @require_api_key
    def stream_status():
        """Return current pipeline status and stats."""
        stats = dict(streamer.stats)
        is_running = (
            main_web._current_thread is not None
            and main_web._current_thread.is_alive()
        )
        return jsonify({
            "running":      is_running,
            "source_label": stats.get("source_label", "—"),
            "source_type":  stats.get("source_type", "—"),
            "fps":          stats.get("fps", 0),
            "frame_count":  stats.get("frame_count", 0),
            "people_count": stats.get("people_count", 0),
            "active_violations": stats.get("active_violations", 0),
        })

    app.register_blueprint(stream_bp)