"""
api/routes/recordings.py
────────────────────────────
Recorded-video batch analysis endpoints:

  POST /api/v1/recordings/analyze   — submit a video file for batch analysis
  GET  /api/v1/recordings/<job_id>  — check job status/progress
  GET  /api/v1/recordings           — list recent jobs
"""

from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key
from novisentra.pipeline.batch_jobs import batch_queue
from novisentra.profiles import is_valid_profile

recordings_bp = Blueprint("recordings_v1", __name__)


def register(app):

    @recordings_bp.route("/api/v1/recordings/analyze", methods=["POST"])
    @require_api_key
    def analyze_recording():
        """
        Submit a video file for batch (non-real-time) analysis — designed
        for full 8-9hr shift recordings. Returns immediately with a job_id;
        poll GET /api/v1/recordings/<job_id> for progress.

        Body:
          {
            "video_path": "/path/to/recording.mp4",   (required — must already
                                                        exist on this server,
                                                        e.g. uploaded separately)
            "profile": "guard_monitoring",             (required)
            "camera_id": "cam_1"                       (optional — for labeling)
          }
        """
        body = request.get_json(silent=True) or {}
        video_path = body.get("video_path", "")
        profile_id = body.get("profile", "")
        camera_id  = body.get("camera_id")

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400
        if not is_valid_profile(profile_id):
            return jsonify({"error": f"Unknown profile '{profile_id}'"}), 400

        job_id = batch_queue.submit(video_path, profile_id, camera_id)
        return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 201

    @recordings_bp.route("/api/v1/recordings/<job_id>", methods=["GET"])
    @require_api_key
    def get_recording_status(job_id):
        job = batch_queue.get_status(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({"job": job})

    @recordings_bp.route("/api/v1/recordings", methods=["GET"])
    @require_api_key
    def list_recordings():
        limit = int(request.args.get("limit", 50))
        return jsonify({"jobs": batch_queue.list_jobs(limit)})

    app.register_blueprint(recordings_bp)
