"""
novisentra/alerting/base.py
────────────────────────────
Abstract base for all alert delivery backends.

Any new channel (SMS, Slack, PagerDuty, etc.) implements this interface.
The AlertRouter dispatches to all registered backends automatically.
"""

from abc import ABC, abstractmethod


class AlertBackend(ABC):
    """
    Implement this to add a new alert delivery channel.

    Usage:
        class MyBackend(AlertBackend):
            def send(self, alert: dict) -> None:
                # deliver the alert
                pass

            def is_enabled(self) -> bool:
                return True
    """

    @abstractmethod
    def send(self, alert: dict) -> None:
        """
        Deliver one alert.

        alert dict keys:
            type       str   — e.g. "Guard Sleeping"
            guard_id   str   — person name or "Camera"
            zone       str   — "A", "B", "C", "D", or "—"
            severity   str   — "high" | "medium" | "low"
            timestamp  str   — "HH:MM:SS"
        """

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this backend is configured and active."""