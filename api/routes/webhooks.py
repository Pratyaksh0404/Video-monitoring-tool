"""
api/routes/webhooks.py
────────────────────────
Webhook registration endpoints:

  POST   /api/v1/webhooks            — register a new webhook
  GET    /api/v1/webhooks            — list all registered webhooks
  GET    /api/v1/webhooks/<id>       — get one webhook by id
  DELETE /api/v1/webhooks/<id>       — remove a webhook
  POST   /api/v1/webhooks/<id>/test  — send a test alert to one webhook

Any external system calls POST /api/v1/webhooks to start receiving
real-time alert POSTs. NoviSentra will deliver every alert to their URL
with an optional HMAC-SHA256 signature for verification.

Webhook payload (delivered to client URL):
  {
    "event":     "alert",
    "id":        "uuid",
    "timestamp": "2026-06-17T09:00:12Z",
    "alert": {
      "type":      "Guard Sleeping",
      "guard_id":  "Pratyaksh",
      "zone":      "C",
      "severity":  "high",
      "timestamp": "09:00:12"
    },
    "source": "novisentra"
  }

Signature verification (Python example):
  import hmac, hashlib
  sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
  assert "sha256=" + sig == request.headers["X-NoviSentra-Signature"]
"""

import datetime
from flask import Blueprint, jsonify, request
from api.middleware.auth import require_api_key

webhooks_bp = Blueprint("webhooks_v1", __name__)


def register(app, webhook_backend):
    """webhook_backend: the WebhookBackend singleton from novisentra/alerting/"""

    @webhooks_bp.route("/api/v1/webhooks", methods=["POST"])
    @require_api_key
    def register_webhook():
        """
        Register a new webhook endpoint.

        Body (JSON):
          url              str    Required. HTTPS endpoint to POST alerts to.
          secret           str    Optional. Shared secret for HMAC-SHA256 signing.
          severity_filter  list   Optional. e.g. ["high"] to receive only HIGH alerts.
                                  Omit or [] to receive all severities.
          name             str    Optional. Human-readable label.

        Example:
          {
            "url": "https://yourapp.com/webhooks/novisentra",
            "secret": "your_shared_secret",
            "severity_filter": ["high"],
            "name": "Production Alert Handler"
          }

        Response:
          {
            "ok": true,
            "webhook": { "id": "...", "url": "...", ... }
          }
        """
        body = request.get_json(silent=True)
        if not body or not body.get("url"):
            return jsonify({"error": "url is required"}), 400

        url             = body["url"]
        secret          = body.get("secret")
        severity_filter = body.get("severity_filter", [])
        name            = body.get("name")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": "url must start with http:// or https://"}), 400

        # Validate severity_filter values
        valid_severities = {"high", "medium", "low"}
        for s in severity_filter:
            if s not in valid_severities:
                return jsonify({
                    "error": f"Invalid severity '{s}'. Must be: high, medium, low"
                }), 400

        wh = webhook_backend.register(
            url=url,
            secret=secret,
            severity_filter=severity_filter,
            name=name,
        )
        return jsonify({"ok": True, "webhook": wh}), 201

    @webhooks_bp.route("/api/v1/webhooks", methods=["GET"])
    @require_api_key
    def list_webhooks():
        """List all registered webhooks (secrets redacted)."""
        return jsonify({
            "webhooks": webhook_backend.list_webhooks(),
            "total":    len(webhook_backend.list_webhooks()),
        })

    @webhooks_bp.route("/api/v1/webhooks/<wh_id>", methods=["GET"])
    @require_api_key
    def get_webhook(wh_id):
        """Get a single webhook by id."""
        wh = webhook_backend.get_webhook(wh_id)
        if not wh:
            return jsonify({"error": "Webhook not found"}), 404
        return jsonify({"webhook": wh})

    @webhooks_bp.route("/api/v1/webhooks/<wh_id>", methods=["DELETE"])
    @require_api_key
    def delete_webhook(wh_id):
        """Remove a webhook by id."""
        removed = webhook_backend.deregister(wh_id)
        if not removed:
            return jsonify({"error": "Webhook not found"}), 404
        return jsonify({"ok": True, "id": wh_id})

    @webhooks_bp.route("/api/v1/webhooks/<wh_id>/test", methods=["POST"])
    @require_api_key
    def test_webhook(wh_id):
        """
        Send a test alert payload to one specific webhook.
        Useful for clients to verify their endpoint is receiving correctly.
        """
        wh = webhook_backend.get_webhook(wh_id)
        if not wh:
            return jsonify({"error": "Webhook not found"}), 404

        test_alert = {
            "type":      "Test Alert",
            "guard_id":  "NoviSentra",
            "zone":      "—",
            "severity":  "low",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        }

        # Use internal method to get the full webhook entry (with secret)
        with webhook_backend._lock:
            full_wh = webhook_backend._webhooks.get(wh_id)
        if not full_wh:
            return jsonify({"error": "Webhook not found"}), 404

        payload = webhook_backend._build_payload(test_alert)
        webhook_backend._executor.submit(webhook_backend._deliver, full_wh, payload)

        return jsonify({
            "ok":      True,
            "message": f"Test alert dispatched to {full_wh['url']}",
        })

    app.register_blueprint(webhooks_bp)