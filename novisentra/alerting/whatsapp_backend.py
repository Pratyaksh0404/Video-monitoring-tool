"""
novisentra/alerting/whatsapp_backend.py
─────────────────────────────────────────
AlertBackend wrapper around the existing WhatsAppAlerter.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .base import AlertBackend


class WhatsAppAlertBackend(AlertBackend):
    """
    Wraps the existing WhatsAppAlerter to conform to the AlertBackend interface.
    """

    def __init__(self, whatsapp_alerter):
        self._alerter = whatsapp_alerter

    def is_enabled(self) -> bool:
        return getattr(self._alerter, "_enabled", False)

    def send(self, alert: dict) -> None:
        self._alerter.send(alert)