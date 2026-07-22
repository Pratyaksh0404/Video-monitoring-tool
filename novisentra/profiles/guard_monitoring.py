"""
novisentra/profiles/guard_monitoring.py
───────────────────────────────────────
The original NoviSentra use case — security guard compliance monitoring.
This is exactly the current pipeline behavior. Nothing changes for
existing deployments running this profile.
"""

from .base import BaseProfile

GUARD_MONITORING = BaseProfile(
    id="guard_monitoring",
    name="Guard Monitoring",
    description=(
        "Security guard compliance — presence, sleeping, phone use, smoking, "
        "patrol coverage, weapon/fire/tamper detection."
    ),
    capabilities=[
        "person_detection",
        "face_recognition",
        "behavior_sleeping",
        "behavior_phone",
        "behavior_smoking",
        "behavior_idle",
        "behavior_distracted",
        "zone_movement_tracking",   # used for patrol logging in this profile
        "weapon_detection",
        "fire_detection",
        "camera_tamper",
        "crowd_detection",
        "loitering_detection",
        "fight_detection",
        "unknown_person_risk",
    ],
    zone_mode="grid",   # current fixed A/B/C/D 2x2 grid
    thresholds={},      # uses rules_config.yaml defaults as-is
    alert_overrides={},
    status="available",
    notes="Default profile — matches current production behavior exactly.",
)
