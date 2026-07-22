"""
novisentra/capabilities/registry.py
──────────────────────────────────────
The single source of truth for every detection/analysis capability
NoviSentra can run. A "capability" is one togglable unit of detection —
weapon detection, crowd detection, loitering, etc.

Profiles (novisentra/profiles/) are just named bundles of capability keys
plus config overrides. This registry is what makes "give the customer only
what they need" possible — a deployment only loads the models and runs the
detectors listed in its active profile's capability list. Nothing else
runs, so nothing else costs CPU/GPU or shows up in that customer's alerts.

status values:
    "available"      — fully built, ships today
    "needs_sourcing"  — needs a pretrained model to be found/evaluated
                        (e.g. face covering detection)
    "needs_training"  — needs a custom model trained on customer-specific
                        data before it can run
"""

CAPABILITIES = {

    # ── Core presence ──────────────────────────────────────────────────────
    "person_detection": {
        "label": "Person Detection",
        "description": "YOLOv8 person bounding box detection. Required by nearly every other capability.",
        "status": "available",
        "produces_alerts": [],
    },
    "face_recognition": {
        "label": "Face Recognition",
        "description": "Identifies enrolled persons by face; unrecognized persons get stable Unknown_N labels.",
        "status": "available",
        "produces_alerts": [],
    },

    # ── Behavior (CLIP-based) ─────────────────────────────────────────────
    "behavior_sleeping": {
        "label": "Sleeping Detection",
        "description": "Zero-shot CLIP classification for sleeping-on-duty posture.",
        "status": "available",
        "produces_alerts": ["Guard Sleeping", "Unknown Person Sleeping"],
    },
    "behavior_phone": {
        "label": "Phone Usage Detection",
        "description": "Detects mobile phone use via object detection + CLIP context.",
        "status": "available",
        "produces_alerts": ["Phone Usage", "Unknown Person Using Phone"],
    },
    "behavior_smoking": {
        "label": "Smoking Detection",
        "description": "CLIP-based smoking behavior classification.",
        "status": "available",
        "produces_alerts": ["Guard Smoking"],
    },
    "behavior_idle": {
        "label": "Idle / Inactivity Detection",
        "description": "Centroid-movement-based inactivity monitor.",
        "status": "available",
        "produces_alerts": ["Guard Idle"],
    },
    "behavior_distracted": {
        "label": "Distraction Detection",
        "description": "CLIP-based distracted-behavior classification.",
        "status": "available",
        "produces_alerts": ["Guard Distracted"],
    },

    # ── Movement / zones ───────────────────────────────────────────────────
    "zone_movement_tracking": {
        "label": "Zone Movement Tracking",
        "description": "Tracks which zone(s) a person moves through. Powers patrol logging, "
                        "dwell-time analytics, and unauthorized-zone alerts, depending on profile config.",
        "status": "available",
        "produces_alerts": ["Patrol"],  # alert type varies by profile config
    },
    "loitering_detection": {
        "label": "Loitering Detection",
        "description": "Flags a person remaining in the same zone/area beyond a configurable "
                        "time threshold. Can run continuously (always-on analytics) or "
                        "alert-only past threshold, per profile.",
        "status": "available",
        "produces_alerts": ["Suspicious Loitering Detected"],
    },
    "dwell_time_analytics": {
        "label": "Dwell Time Analytics",
        "description": "Reports time spent per zone per person/session — business analytics, "
                        "not an alert by itself (retail traffic patterns, etc).",
        "status": "available",
        "produces_alerts": [],
    },

    # ── Threats ────────────────────────────────────────────────────────────
    "weapon_detection": {
        "label": "Weapon Detection",
        "description": "YOLOv8 fine-tuned gun/knife detector, dual-scan (full-frame + per-person crop).",
        "status": "available",
        "produces_alerts": ["Weapon Detected", "Unattended Weapon Detected"],
    },
    "fire_detection": {
        "label": "Fire / Smoke Detection",
        "description": "YOLOv8 fire model with HSV color pre-filter to reject lighting false positives.",
        "status": "available",
        "produces_alerts": ["Fire Detected"],
    },
    "fight_detection": {
        "label": "Fight / Violence Detection",
        "description": "Proximity + motion-magnitude based fight detection.",
        "status": "available",
        "produces_alerts": ["Fight / Violence Detected", "Guard Under Attack"],
    },
    "camera_tamper": {
        "label": "Camera Tamper Detection",
        "description": "Blackout / blur / obstruction / scene-change detection.",
        "status": "available",
        "produces_alerts": ["Camera Tamper"],
    },
    "crowd_detection": {
        "label": "Crowd Detection",
        "description": "Person-count threshold with post-crowd grace period.",
        "status": "available",
        "produces_alerts": ["Crowd Detected"],
    },
    "unknown_person_risk": {
        "label": "Unknown Person Risk Flagging",
        "description": "Flags risky behavior specifically by unrecognized persons "
                        "(sleeping, phone use, weapon) with a grace period before alerting.",
        "status": "available",
        "produces_alerts": ["Unknown Person Detected", "Unknown Person Sleeping",
                            "Unknown Person Using Phone"],
    },
    "face_covering_detection": {
        "label": "Face Covering / Mask Detection",
        "description": "Flags deliberately concealed faces (scarf, balaclava, helmet indoors) — "
                        "a robbery/security indicator, distinct from surgical-mask detectors. "
                        "No off-the-shelf model reliably covers this use case yet; needs "
                        "evaluation/sourcing or light fine-tuning before going live.",
        "status": "needs_sourcing",
        "produces_alerts": ["Face Covering Detected"],
    },
}


def get_capability(key: str) -> dict:
    """Return capability metadata, or raise KeyError if unknown."""
    return CAPABILITIES[key]


def list_capabilities(status: str = None) -> dict:
    """Return all capabilities, optionally filtered by status."""
    if status is None:
        return dict(CAPABILITIES)
    return {k: v for k, v in CAPABILITIES.items() if v["status"] == status}


def validate_capability_keys(keys: list) -> list:
    """Return any keys in the list that are NOT valid registry entries."""
    return [k for k in keys if k not in CAPABILITIES]
