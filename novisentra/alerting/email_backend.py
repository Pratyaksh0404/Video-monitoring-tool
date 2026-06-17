"""
novisentra/alerting/email_backend.py
──────────────────────────────────────
AlertBackend wrapper around the existing EmailAlerter.

This makes EmailAlerter pluggable into the AlertRouter
without changing email_alerter.py at all.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .base import AlertBackend


class EmailAlertBackend(AlertBackend):
    """
    Wraps the existing EmailAlerter to conform to the AlertBackend interface.

    Usage:
        from novisentra.alerting.email_backend import EmailAlertBackend
        backend = EmailAlertBackend(email_alerter_instance)
    """

    def __init__(self, email_alerter):
        self._alerter = email_alerter

    def is_enabled(self) -> bool:
        return getattr(self._alerter, "_enabled", False)

    def send(self, alert: dict) -> None:
        self._alerter.send(alert)