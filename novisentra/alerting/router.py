"""
novisentra/alerting/router.py
───────────────────────────────
AlertRouter — dispatches every alert to all registered backends.

IMPORTANT: The AlertRouter does NOT consume from alert_manager.alert_queue
directly — that queue is consumed by alert_dispatcher in flask_app.py which
handles email, WhatsApp, SSE, and the dashboard alert log. Having two
consumers on the same queue causes a race where each alert is delivered to
only ONE consumer (whichever calls .get() first), silently dropping alerts
from the other. Instead, alert_dispatcher feeds a COPY of each alert to
router.fanout_queue, which the router consumes independently. This ensures
every alert reaches BOTH the main dispatcher (email/WA/SSE/log) AND the
router backends (webhooks, future Slack/SMS etc.) with zero interference.

Wiring in flask_app.py alert_dispatcher:
    # After processing the alert:
    if _webhook_router:
        _webhook_router.fanout(alert)
"""

import queue
import threading
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    from utils.logger import get_logger
    _log = get_logger("alert_router")
except Exception:
    import logging
    _log = logging.getLogger("alert_router")

from .base import AlertBackend


class AlertRouter:
    """
    Receives alert copies via fanout() and fans out to all registered backends.
    Does NOT consume from alert_manager.alert_queue — see module docstring.
    """

    def __init__(self, alert_queue: queue.Queue = None):
        # alert_queue param kept for backward compat but ignored — we use
        # our own internal fanout queue to avoid the dual-consumer race.
        self._fanout_queue = queue.Queue(maxsize=200)
        self._backends: list[AlertBackend] = []
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

    def register(self, backend: AlertBackend) -> None:
        with self._lock:
            self._backends.append(backend)
        _log.info(f"Alert backend registered: {type(backend).__name__}")

    def deregister(self, backend: AlertBackend) -> None:
        with self._lock:
            self._backends = [b for b in self._backends if b is not backend]

    def fanout(self, alert: dict) -> None:
        """
        Called by alert_dispatcher (flask_app.py) after each alert is
        processed. Puts a copy onto the internal fanout queue so the
        router thread can deliver it to webhook/other backends without
        racing with the main dispatcher.
        """
        try:
            self._fanout_queue.put_nowait(dict(alert))
        except queue.Full:
            _log.warning("AlertRouter fanout queue full — webhook delivery dropped for one alert.")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._consume,
            daemon=True,
            name="alert-router"
        )
        self._thread.start()
        _log.info("AlertRouter started.")

    def stop(self) -> None:
        self._running = False

    def _consume(self):
        while self._running:
            try:
                alert = self._fanout_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                backends = list(self._backends)

            for backend in backends:
                try:
                    if backend.is_enabled():
                        backend.send(alert)
                except Exception as e:
                    _log.error(
                        f"Backend {type(backend).__name__} raised on send: {e}",
                        exc_info=True
                    )

            self._fanout_queue.task_done()