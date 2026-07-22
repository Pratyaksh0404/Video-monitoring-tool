"""
novisentra/profiles/warehouse_ops.py
──────────────────────────────────────
Warehouse operations safety — sleeping, phone use, crowd near machinery,
fire, fights, and unauthorized zone movement. No PPE/uniform detection
in v1 (flagged in notes — genuinely feasible to explore later, but
skipped for now per explicit scope decision, not a technical blocker
like face covering or restaurant food).
"""

from .base import BaseProfile, Zone

WAREHOUSE_OPS = BaseProfile(
    id="warehouse_ops",
    name="Warehouse Operations",
    description=(
        "Worker safety monitoring — sleeping/idle detection, phone use, crowd "
        "gathering near machinery, fire, fights, and unauthorized zone entry."
    ),
    capabilities=[
        "person_detection",
        "face_recognition",          # worker identification
        "behavior_sleeping",
        "behavior_phone",
        "crowd_detection",
        "fire_detection",
        "fight_detection",
        "camera_tamper",
        "zone_movement_tracking",    # powers unauthorized-zone alerting
        "unknown_person_risk",       # unrecognized person in restricted area
    ],
    zone_mode="named",
    zones=[
        Zone(id="loading_dock", label="Loading Dock"),
        Zone(id="storage_a", label="Storage Area A"),
        Zone(id="restricted", label="Restricted Zone"),
        Zone(id="machinery", label="Machinery Floor"),
    ],
    thresholds={
        "zone_restricted": {
            # Unauthorized zone entry alerts immediately — this is a
            # different trigger from loitering, it fires on ENTRY to a
            # zone flagged "restricted", not on dwell time.
            "restricted_zone_ids": ["restricted"],
            "alert_on_entry": True,
        },
    },
    alert_overrides={
        # NOTE: zone_movement_tracking's alert is intentionally NOT
        # suppressed here (see bank_security.py for the full explanation
        # of the same bug/fix) — it fires as "Movement: ..." via
        # _MOVEMENT_LABEL in alert_manager.py, not "Patrol".
    },
    status="available",
    notes=(
        "PPE (hard hat/vest) and uniform detection intentionally excluded "
        "from v1 — uniform detection especially is a harder, low-accuracy "
        "problem (variable lighting, partial occlusion, similar-colored "
        "plain clothing). Worth a small feasibility test later if a client "
        "specifically needs it, but not blocking this profile's launch."
    ),
)
