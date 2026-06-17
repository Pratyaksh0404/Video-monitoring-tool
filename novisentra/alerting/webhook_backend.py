"""
novisentra/alerting/webhook_backend.py
────────────────────────────────────────
Delivers alerts to client-registered HTTP endpoints (webhooks).

Any external system — a client's own dashboard, mobile app, or SIEM —
can register a URL and receive every alert as a JSON POST in real time.

Features:
  - Multiple webhooks per deployment (each gets every alert)
  - Per-webhook secret for HMAC-SHA256 signature verification
  - Automatic retry (3 attempts, exponential backoff: 1s, 2s, 4s)
  - Per-webhook severity filter (e.g. HIGH only)
  - In-memory registry with optional JSON persistence
  - Non-blocking — sends on background thread pool

Webhook POST body:
  {
    "event":     "alert",
    "id":        "uuid4",
    "timestamp": "2026-06-17T09:00:12Z",
    "alert": {
        "type":      "Guard Sleeping",
        "guard_id":  "Pratyaksh",
        "zone":      "C",
        "severity":  "high",
        "timestamp": "09:00:12"
    }
  }

Signature header (if secret configured):
  X-NoviSentra-Signature: sha256=<hmac_hex>
"""

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from utils.logger import get_logger
    _log = get_logger("webhook_backend")
except Exception:
    import logging
    _log = logging.getLogger("webhook_backend")

from .base import AlertBackend


class WebhookBackend(AlertBackend):
    """
    Delivers alerts to one or more registered HTTP endpoints.

    Typical usage (in flask_app.py):
        from novisentra.alerting.webhook_backend import WebhookBackend
        webhook_alerter = WebhookBackend(persist_path="config/webhooks.json")
        # client registers via POST /api/v1/webhooks/register
    """

    def __init__(self, persist_path: Optional[str] = None, max_workers: int = 4):
        """
        persist_path: optional JSON file to persist webhook registrations
                      across restarts. If None, registrations are in-memory only.
        max_workers:  thread pool size for concurrent webhook deliveries.
        """
        self._webhooks: dict[str, dict] = {}   # id → webhook config
        self._lock = threading.Lock()
        self._persist_path = persist_path
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="webhook")

        if not _HAS_REQUESTS:
            _log.warning(
                "requests library not installed — webhook delivery disabled. "
                "Run: pip install requests"
            )

        if persist_path:
            self._load()

    # ── AlertBackend interface ────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return _HAS_REQUESTS and bool(self._webhooks)

    def send(self, alert: dict) -> None:
        """
        Dispatch alert to all registered webhooks that match the alert's
        severity filter. Non-blocking — fires on background thread pool.
        """
        if not _HAS_REQUESTS:
            return

        with self._lock:
            targets = list(self._webhooks.values())

        if not targets:
            return

        payload = self._build_payload(alert)

        for wh in targets:
            # Apply per-webhook severity filter
            allowed = wh.get("severity_filter", [])
            if allowed and alert.get("severity", "low") not in allowed:
                continue
            self._executor.submit(self._deliver, wh, payload)

    # ── Registration API ──────────────────────────────────────────────────────

    def register(self,
                 url: str,
                 secret: Optional[str] = None,
                 severity_filter: Optional[list] = None,
                 name: Optional[str] = None) -> dict:
        """
        Register a new webhook endpoint.

        Args:
            url:              The HTTPS URL to POST alerts to.
            secret:           Optional shared secret for HMAC-SHA256 signing.
                              Client verifies: sha256(body, secret) == X-NoviSentra-Signature
            severity_filter:  List of severities to deliver, e.g. ["high"].
                              Empty / None = all severities.
            name:             Human-readable label (for listing).

        Returns:
            The webhook registration dict including its generated id.
        """
        wh_id = str(uuid.uuid4())
        entry = {
            "id":              wh_id,
            "url":             url,
            "secret":          secret,
            "severity_filter": severity_filter or [],
            "name":            name or url,
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "delivery_count":  0,
            "failure_count":   0,
            "last_delivered":  None,
        }
        with self._lock:
            self._webhooks[wh_id] = entry
        self._save()
        _log.info(f"Webhook registered: {name or url} (id={wh_id[:8]})")
        return self._public(entry)

    def deregister(self, wh_id: str) -> bool:
        """Remove a webhook by id. Returns True if found and removed."""
        with self._lock:
            if wh_id not in self._webhooks:
                return False
            del self._webhooks[wh_id]
        self._save()
        _log.info(f"Webhook removed: {wh_id[:8]}")
        return True

    def list_webhooks(self) -> list:
        """Return all registered webhooks (secrets redacted)."""
        with self._lock:
            return [self._public(wh) for wh in self._webhooks.values()]

    def get_webhook(self, wh_id: str) -> Optional[dict]:
        """Return a single webhook (secret redacted), or None if not found."""
        with self._lock:
            wh = self._webhooks.get(wh_id)
            return self._public(wh) if wh else None

    # ── Delivery ──────────────────────────────────────────────────────────────

    def _build_payload(self, alert: dict) -> dict:
        return {
            "event":     "alert",
            "id":        str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert":     alert,
            "source":    "novisentra",
        }

    def _sign(self, body: bytes, secret: str) -> str:
        """HMAC-SHA256 signature over the raw JSON body."""
        return "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()

    def _deliver(self, wh: dict, payload: dict, attempt: int = 1):
        """Send one webhook with retry (up to 3 attempts, exponential backoff)."""
        url    = wh["url"]
        secret = wh.get("secret")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type":    "application/json",
            "User-Agent":      "NoviSentra-Webhook/1.0",
            "X-NoviSentra-Event": "alert",
        }
        if secret:
            headers["X-NoviSentra-Signature"] = self._sign(body, secret)

        try:
            resp = _requests.post(url, data=body, headers=headers, timeout=10)
            resp.raise_for_status()

            with self._lock:
                wh["delivery_count"] += 1
                wh["last_delivered"] = datetime.now(timezone.utc).isoformat()

            _log.debug(f"Webhook delivered → {url} [{resp.status_code}]")

        except Exception as e:
            with self._lock:
                wh["failure_count"] += 1

            if attempt < 3:
                delay = 2 ** (attempt - 1)   # 1s, 2s, 4s
                _log.warning(
                    f"Webhook failed (attempt {attempt}/3, retry in {delay}s): "
                    f"{url} — {type(e).__name__}: {e}"
                )
                time.sleep(delay)
                self._deliver(wh, payload, attempt + 1)
            else:
                _log.error(
                    f"Webhook permanently failed after 3 attempts: {url} — {e}"
                )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _public(self, wh: dict) -> dict:
        """Return webhook dict with secret replaced by a masked indicator."""
        out = dict(wh)
        out["secret"] = "***" if wh.get("secret") else None
        return out

    def _save(self):
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            # Save full data (including secrets) to disk — file should be chmod 600
            with self._lock:
                data = {k: dict(v) for k, v in self._webhooks.items()}
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _log.error(f"Webhook persist save failed: {e}")

    def _load(self):
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._webhooks = data
            _log.info(f"Loaded {len(data)} webhook(s) from {self._persist_path}")
        except Exception as e:
            _log.warning(f"Webhook persist load failed: {e}")