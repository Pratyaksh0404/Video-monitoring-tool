"""
novisentra/alerting/router.py
───────────────────────────────
AlertRouter — dispatches every alert to all registered backends.

This is the single point that connects the pipeline's alert_queue
to all delivery channels (email, WhatsApp, webhook, future SMS/Slack/etc.).

Usage in flask_app.py:
    from novisentra.alerting.router import AlertRouter
    from novisentra.alerting.email_backend import EmailAlertBackend
    from novisentra.alerting.whatsapp_backend import WhatsAppAlertBackend
    from novisentra.alerting.webhook_backend import WebhookBackend

    router = AlertRouter()
    router.register(EmailAlertBackend(_email_alerter))
    router.register(WhatsAppAlertBackend(_whatsapp_alerter))
    router.register(webhook_backend)   # singleton shared with webhook routes
    router.start()                     # begins consuming from alert_queue

The router drains src/alerts/alert_manager.alert_queue and fans out
to every enabled backend. The pipeline itself does not change at all.
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
    Drains the global alert_queue and fans out to all registered backends.

    Thread-safe. Can register/deregister backends at runtime.
    """

    def __init__(self, alert_queue: queue.Queue):
        self._queue    = alert_queue
        self._backends: list[AlertBackend] = []
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

    def register(self, backend: AlertBackend) -> None:
        """Add a delivery backend. Can be called before or after start()."""
        with self._lock:
            self._backends.append(backend)
        _log.info(f"Alert backend registered: {type(backend).__name__}")

    def deregister(self, backend: AlertBackend) -> None:
        """Remove a backend."""
        with self._lock:
            self._backends = [b for b in self._backends if b is not backend]

    def start(self) -> None:
        """Start the background consumer thread."""
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _consume(self):
        while self._running:
            try:
                alert = self._queue.get(timeout=0.5)
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

            self._queue.task_done()