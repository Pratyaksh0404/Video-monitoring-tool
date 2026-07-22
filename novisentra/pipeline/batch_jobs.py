"""
novisentra/pipeline/batch_jobs.py
────────────────────────────────────
Job queue for recorded-video analysis — the "process an 8-9hr shift
recording" feature. This is the infrastructure AROUND the detection
pipeline (queueing, progress tracking, status, results storage) — the
actual frame-by-frame call into the pipeline is stubbed at
_run_detection_job() pending main_web.py, clearly marked below.

Why a job queue at all, rather than just calling the pipeline directly
from an API request: an 8-9hr video takes a long time to process even
without real-time throttling — far longer than an HTTP request should
ever block for. So this is fire-and-forget: submit a job, it runs in a
background thread, poll its status, fetch the report when done.

Uses the same SQLite database as the auth system (config/novisentra.db)
— one file, still lives entirely on the client's own machine.
"""

import os
import sys
import time
import uuid
import threading
import datetime
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from novisentra.auth.db import get_connection

try:
    from utils.logger import get_logger
    _log = get_logger("batch_jobs")
except Exception:
    import logging
    _log = logging.getLogger("batch_jobs")


def init_batch_jobs_table():
    """Call once at startup, alongside novisentra.auth.init_db()."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_jobs (
                job_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                camera_id TEXT,
                profile_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_pct REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error_message TEXT,
                report_path TEXT,
                alert_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
    finally:
        conn.close()


# Job statuses
QUEUED    = "queued"
RUNNING   = "running"
COMPLETE  = "complete"
FAILED    = "failed"


class BatchJobQueue:
    """
    Single-worker background queue — processes one recorded video at a
    time (sequential, not parallel, to avoid CPU contention with any
    live camera pipelines running simultaneously on the same server).
    """

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._lock = threading.Lock()
        init_batch_jobs_table()

    def submit(self, video_path: str, profile_id: str, camera_id: str = None) -> str:
        """
        Queue a video for batch analysis. Returns a job_id immediately —
        does not block. Processing starts as soon as a worker slot is free.
        """
        job_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()

        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO batch_jobs "
                "(job_id, video_path, camera_id, profile_id, status, progress_pct, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (job_id, video_path, camera_id, profile_id, QUEUED, now),
            )
            conn.commit()
        finally:
            conn.close()

        _log.info(f"Batch job queued: {job_id} — {video_path} (profile={profile_id})")

        threading.Thread(target=self._maybe_start_next, daemon=True).start()
        return job_id

    def get_status(self, job_id: str) -> dict:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_jobs(self, limit: int = 50) -> list:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def cancel(self, job_id: str) -> bool:
        """
        Only cancels a job still in QUEUED state — a RUNNING job's actual
        frame loop needs a cooperative stop signal from within main_web.py
        (same stop_event pattern CameraInstance already uses for live
        cameras) to cancel cleanly mid-processing. Flagged as part of the
        main_web.py wiring, not solvable from this file alone.
        """
        conn = get_connection()
        try:
            cur = conn.execute(
                "UPDATE batch_jobs SET status = ? WHERE job_id = ? AND status = ?",
                (FAILED, job_id, QUEUED),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_start_next(self):
        with self._lock:
            if self._active_count >= self.max_concurrent:
                return
            self._active_count += 1

        try:
            job = self._claim_next_queued_job()
            if job is None:
                return
            self._run_job(job)
        finally:
            with self._lock:
                self._active_count -= 1
            # In case more jobs are waiting and a slot just freed up
            threading.Thread(target=self._maybe_start_next, daemon=True).start()

    def _claim_next_queued_job(self):
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                return None
            job = dict(row)
            conn.execute(
                "UPDATE batch_jobs SET status = ?, started_at = ? WHERE job_id = ?",
                (RUNNING, datetime.datetime.utcnow().isoformat(), job["job_id"]),
            )
            conn.commit()
            return job
        finally:
            conn.close()

    def _update_progress(self, job_id: str, progress_pct: float):
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE batch_jobs SET progress_pct = ? WHERE job_id = ?",
                (progress_pct, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _mark_complete(self, job_id: str, report_path: str, alert_count: int):
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE batch_jobs SET status = ?, progress_pct = 100, "
                "finished_at = ?, report_path = ?, alert_count = ? WHERE job_id = ?",
                (COMPLETE, datetime.datetime.utcnow().isoformat(),
                 report_path, alert_count, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _mark_failed(self, job_id: str, error_message: str):
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE batch_jobs SET status = ?, finished_at = ?, error_message = ? "
                "WHERE job_id = ?",
                (FAILED, datetime.datetime.utcnow().isoformat(), error_message, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _run_job(self, job: dict):
        job_id     = job["job_id"]
        video_path = job["video_path"]
        profile_id = job["profile_id"]
        camera_id  = job.get("camera_id") or f"batch_{job_id[:8]}"

        _log.info(f"Batch job starting: {job_id} — {video_path}")

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # ═══════════════════════════════════════════════════════════════
            # PENDING main_web.py — the actual detection call goes here.
            #
            # Once main_web.py's interface is known, this becomes something
            # like:
            #
            #   from alerts.alert_manager import set_camera_context, set_profile_context
            #   set_camera_context(camera_id, camera_name=f"Recording: {os.path.basename(video_path)}")
            #   set_profile_context(profile_id)
            #
            #   main_web.run_batch(
            #       source=video_path,
            #       progress_callback=lambda pct: self._update_progress(job_id, pct),
            #       # ^ needs main_web.py to report progress as (current_frame / total_frames)
            #       #   rather than the live-mode "keep up with wall clock" behavior
            #   )
            #
            # For now, this stub demonstrates the queue mechanics end-to-end
            # (status transitions, progress updates, completion) using a
            # simulated processing delay, so the job queue itself is fully
            # testable before the real pipeline call is wired in.
            # ═══════════════════════════════════════════════════════════════
            for pct in [10, 30, 50, 70, 90, 100]:
                time.sleep(0.05)   # simulated processing — replace with real frame loop
                self._update_progress(job_id, pct)

            report_path = video_path + ".report.html"   # placeholder path convention
            alert_count = 0   # placeholder — real count comes from the alert log once wired

            self._mark_complete(job_id, report_path, alert_count)
            _log.info(f"Batch job complete: {job_id}")

        except Exception as e:
            _log.error(f"Batch job failed: {job_id} — {e}", exc_info=True)
            self._mark_failed(job_id, str(e))


# Module-level singleton — mirrors the pattern used by snap_mgr and streamer
batch_queue = BatchJobQueue(max_concurrent=1)
