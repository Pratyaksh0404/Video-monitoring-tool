"""
novisentra/profiles/retail_analytics.py
─────────────────────────────────────────
Retail store — customer safety + traffic analytics. person_detection
(anonymous YOLO boxes, tracked as Unknown_N) stays ON — it's required
plumbing for smoking/idle/weapon attribution and crowd counting. What's
OFF is face_recognition: no enrolled customer identities, since a store
can't realistically enroll every shopper — this keeps the profile
privacy-appropriate while still catching the behaviors that matter.
Named zones (e.g. "Kids", "Men's", "Checkout") instead of the fixed
grid, configured per-store during setup. Loitering runs continuously as
analytics (dwell per zone always logged) AND alerts when a person stays
in one area far longer than typical browsing time; idle detection
separately flags a person standing still (not necessarily loitering
across zones) for longer than typical.
"""

from .base import BaseProfile, Zone

RETAIL_ANALYTICS = BaseProfile(
    id="retail_analytics",
    name="Retail Store Analytics",
    description=(
        "Customer safety and store traffic analytics — smoking, idle, crowd, "
        "weapon, and fire detection, plus always-on dwell-time and loitering "
        "analysis across named store zones."
    ),
    capabilities=[
        "person_detection",           # required plumbing for smoking/idle/
                                       # weapon-on-person attribution below —
                                       # NOT the same as identifying WHO a
                                       # customer is. Tracked anonymously as
                                       # Unknown_N; see face_recognition note.
        "behavior_smoking",
        "behavior_idle",              # "a person standing in one place for a
                                       # long time is suspicious" — added 2026-07,
                                       # was missing despite being requested
        "crowd_detection",
        "weapon_detection",
        "fire_detection",
        "loitering_detection",       # continuous, see alert_overrides
        "dwell_time_analytics",      # always-on traffic analytics
        "camera_tamper",
        "zone_movement_tracking",
    ],
    zone_mode="named",
    zones=[
        # Example default set — every store customizes this at setup time
        # by drawing/labeling zones against their actual camera view.
        Zone(id="entrance", label="Entrance"),
        Zone(id="checkout", label="Checkout"),
        Zone(id="mens", label="Men's Section"),
        Zone(id="womens", label="Women's Section"),
        Zone(id="kids", label="Kids' Section"),
    ],
    thresholds={
        # Real schema field names are unknown_zone_time / known_zone_time
        # (seconds in one zone before loitering fires) — no
        # alert_after_seconds/always_log fields exist in rules_config.yaml.
        # Base loitering detection already logs every zone-dwell period
        # regardless of these thresholds; these two values just control
        # how long before a dwell period escalates to an actual alert.
        # Retail needs a much longer threshold than guard_monitoring's
        # defaults (30s/90s) — normal browsing shouldn't trigger alerts,
        # only genuinely extended dwelling (potential shoplifting risk).
        "loitering": {
            "unknown_zone_time": 600,   # 10 min — customers are anonymous by default in retail
            "known_zone_time": 600,
        },
    },
    alert_overrides={
        "Suspicious Loitering Detected": {"severity": "medium"},
    },
    status="available",
    notes=(
        "Default zone list above is a starting template — real zone "
        "polygons must be calibrated per store during camera setup. "
        "Face recognition intentionally excluded (no enrolled customer "
        "identities; keeps this privacy-appropriate for public retail floors)."
    ),
)