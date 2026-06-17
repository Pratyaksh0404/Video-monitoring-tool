"""
test_api.py
────────────
Quick smoke-test for all NoviSentra REST API v1 endpoints.
Run while flask_app.py is running in another terminal.

Usage:
    python test_api.py

No dependencies beyond stdlib + requests (already in requirements).
"""

import json
import sys
import time
import requests

BASE = "http://127.0.0.1:5000"
API  = f"{BASE}/api/v1"

# Set this if you've configured NOVISENTRA_API_KEY in your environment
API_KEY = ""   # leave empty if no key configured

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

PASS = "  ✓"
FAIL = "  ✗"
SKIP = "  -"

results = []


def test(name, method, path, body=None, expected_status=200, check_keys=None):
    url = f"{API}{path}"
    try:
        if method == "GET":
            r = requests.get(url, headers=HEADERS, timeout=5)
        elif method == "POST":
            r = requests.post(url, json=body, headers=HEADERS, timeout=5)
        elif method == "PATCH":
            r = requests.patch(url, json=body, headers=HEADERS, timeout=5)
        elif method == "DELETE":
            r = requests.delete(url, headers=HEADERS, timeout=5)

        ok = r.status_code == expected_status

        # Check expected keys in response
        if ok and check_keys:
            try:
                data = r.json()
                for key in check_keys:
                    parts = key.split(".")
                    val = data
                    for p in parts:
                        val = val[p]
            except (KeyError, TypeError) as e:
                ok = False
                print(f"{FAIL} {name}")
                print(f"       → Missing key in response: {e}")
                print(f"       → Got: {json.dumps(data, indent=2)[:300]}")
                results.append(False)
                return None

        icon = PASS if ok else FAIL
        status_note = f"[{r.status_code}]"
        if not ok:
            status_note += f" expected {expected_status}"

        print(f"{icon} {name}  {status_note}")
        if not ok:
            try:
                print(f"       → {r.json()}")
            except Exception:
                print(f"       → {r.text[:200]}")

        results.append(ok)
        return r

    except requests.exceptions.ConnectionError:
        print(f"{FAIL} {name}  [CONNECTION REFUSED]")
        print(f"       → Is flask_app.py running on port 5000?")
        results.append(False)
        return None
    except Exception as e:
        print(f"{FAIL} {name}  [{type(e).__name__}: {e}]")
        results.append(False)
        return None


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  NoviSentra REST API v1 — Smoke Test")
print("  Target:", BASE)
print("═" * 60)

# ── Health ────────────────────────────────────────────────────────────────────
print("\n── Health ──────────────────────────────────────────────────")
test("GET  /api/v1/health",
     "GET", "/health",
     check_keys=["status", "version", "product", "uptime", "pipeline", "models"])

# ── Stream ────────────────────────────────────────────────────────────────────
print("\n── Stream ──────────────────────────────────────────────────")
test("GET  /api/v1/stream/status",
     "GET", "/stream/status",
     check_keys=["running", "fps", "people_count"])

# Note: we don't actually start/stop the stream in the test
# since that would interrupt your live session
print(f"{SKIP} POST /api/v1/stream/start  (skipped — would interrupt live session)")
print(f"{SKIP} POST /api/v1/stream/stop   (skipped — would interrupt live session)")

# ── Alerts ────────────────────────────────────────────────────────────────────
print("\n── Alerts ──────────────────────────────────────────────────")
r = test("GET  /api/v1/alerts",
         "GET", "/alerts",
         check_keys=["alerts", "total", "limit", "offset"])

if r:
    data = r.json()
    print(f"       → {data['total']} alert(s) in current session")

test("GET  /api/v1/alerts  (filter severity=high)",
     "GET", "/alerts?severity=high&limit=5",
     check_keys=["alerts", "total"])

test("GET  /api/v1/alerts  (filter type=sleeping)",
     "GET", "/alerts?type=sleeping",
     check_keys=["alerts", "total"])

# SSE — just check it opens (don't actually subscribe)
try:
    r2 = requests.get(f"{API}/alerts/live", headers=HEADERS,
                      stream=True, timeout=2)
    ok = r2.status_code == 200
    print(f"{PASS if ok else FAIL} GET  /api/v1/alerts/live  (SSE)  [{r2.status_code}]")
    r2.close()
    results.append(ok)
except requests.exceptions.ReadTimeout:
    # Timeout is expected for SSE — means it's streaming correctly
    print(f"{PASS} GET  /api/v1/alerts/live  (SSE)  [streaming — timeout as expected]")
    results.append(True)
except Exception as e:
    print(f"{FAIL} GET  /api/v1/alerts/live  (SSE)  [{e}]")
    results.append(False)

# ── Webhooks ──────────────────────────────────────────────────────────────────
print("\n── Webhooks ────────────────────────────────────────────────")
r = test("GET  /api/v1/webhooks  (list)",
         "GET", "/webhooks",
         check_keys=["webhooks", "total"])

# Register a test webhook pointing to a public echo service
r_reg = test("POST /api/v1/webhooks  (register)",
             "POST", "/webhooks",
             body={
                 "url":             "https://webhook.site/test-novisentra",
                 "name":            "Test Webhook (smoke test)",
                 "severity_filter": ["high"],
             },
             expected_status=201,
             check_keys=["ok", "webhook.id", "webhook.url"])

wh_id = None
if r_reg:
    wh_id = r_reg.json().get("webhook", {}).get("id")
    print(f"       → Registered webhook id: {wh_id[:8] if wh_id else 'N/A'}...")

if wh_id:
    test(f"GET  /api/v1/webhooks/<id>",
         "GET", f"/webhooks/{wh_id}",
         check_keys=["webhook.id", "webhook.url"])

    test(f"POST /api/v1/webhooks/<id>/test",
         "POST", f"/webhooks/{wh_id}/test",
         check_keys=["ok", "message"])

    test(f"DELETE /api/v1/webhooks/<id>",
         "DELETE", f"/webhooks/{wh_id}",
         check_keys=["ok"])

# ── Analytics ─────────────────────────────────────────────────────────────────
print("\n── Analytics ───────────────────────────────────────────────")
r = test("GET  /api/v1/analytics/summary",
         "GET", "/analytics/summary",
         check_keys=["people_count", "guards_detected", "fps", "source_label"])

if r:
    d = r.json()
    print(f"       → fps={d['fps']}  people={d['people_count']}  "
          f"guards={d['guards_detected']}  source='{d['source_label']}'")

r = test("GET  /api/v1/analytics/guards",
         "GET", "/analytics/guards",
         check_keys=["guards"])

if r:
    guards = r.json().get("guards", [])
    if guards:
        for g in guards:
            print(f"       → {g['name']}: {g['alert_count']} alerts, "
                  f"compliance={g['compliance']}%")
    else:
        print("       → No guard data yet (pipeline may not be running)")

r = test("GET  /api/v1/analytics/snapshots",
         "GET", "/analytics/snapshots",
         check_keys=["snapshots"])

if r:
    snaps = r.json().get("snapshots", [])
    print(f"       → {len(snaps)} snapshot(s) in current session")
    if snaps:
        print(f"       → Latest: {snaps[0].get('filename', '?')} "
              f"at {snaps[0].get('ts', '?')}")

# ── Config ────────────────────────────────────────────────────────────────────
print("\n── Config ──────────────────────────────────────────────────")
r = test("GET  /api/v1/config",
         "GET", "/config",
         check_keys=["behavior", "alerts"])

# Test PATCH with a safe no-op change (reading current value and writing it back)
if r:
    current_cooldown = r.json().get("alerts", {}).get("cooldown", 30)
    test("PATCH /api/v1/config  (update alerts.cooldown)",
         "PATCH", "/config",
         body={"alerts": {"cooldown": current_cooldown}},   # write same value back
         check_keys=["ok"])
    print(f"       → alerts.cooldown={current_cooldown} (unchanged, written back)")

test("GET  /api/v1/config/profiles",
     "GET", "/config/profiles",
     check_keys=["profiles", "active"])

test("POST /api/v1/config/profile  (guard_monitoring)",
     "POST", "/config/profile",
     body={"profile": "guard_monitoring"},
     check_keys=["ok"])

test("POST /api/v1/config/profile  (unknown → 400)",
     "POST", "/config/profile",
     body={"profile": "nonexistent"},
     expected_status=400)

test("POST /api/v1/config/profile  (retail_analytics → 501 not yet implemented)",
     "POST", "/config/profile",
     body={"profile": "retail_analytics"},
     expected_status=501)

# ── Reports ───────────────────────────────────────────────────────────────────
print("\n── Reports ─────────────────────────────────────────────────")
r = test("GET  /api/v1/reports/generate",
         "GET", "/reports/generate",
         expected_status=200)

if r:
    size_kb = len(r.content) / 1024
    is_html = "<!DOCTYPE" in r.text[:100] or "<html" in r.text[:100]
    print(f"       → {size_kb:.1f} KB HTML report generated  (valid HTML: {is_html})")

# Don't actually send an email in the test unless explicitly requested
print(f"{SKIP} POST /api/v1/reports/send  (skipped — would send real email)")

# ─────────────────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
failed = total - passed

print("\n" + "═" * 60)
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
else:
    print("  — all good ✓")
print("═" * 60)

if failed:
    sys.exit(1)