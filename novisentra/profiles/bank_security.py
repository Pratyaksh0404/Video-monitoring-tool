"""
novisentra/profiles/bank_security.py
──────────────────────────────────────
Bank security — weapon-first priority, stricter confidence thresholds
("more sure") with faster repeat alerting ("more alert") than guard
monitoring. Zone movement is tracked and alerted as "Movement" (not
"Patrol" — that's guard_monitoring's vocabulary; see _MOVEMENT_LABEL in
alert_manager.py). Face covering detection included but flagged
needs_sourcing — profile runs without it until a suitable model is
evaluated; can be enabled per-deployment once ready.
"""

from .base import BaseProfile

BANK_SECURITY = BaseProfile(
    id="bank_security",
    name="Bank Security",
    description=(
        "Weapon detection, face covering flagging, crowd/fight monitoring, "
        "smoking, fire and camera tamper for bank branches. Higher confidence "
        "thresholds to minimize false alarms, faster re-alert on sustained threats."
    ),
    capabilities=[
        "person_detection",
        "face_recognition",
        "weapon_detection",
        "face_covering_detection",   # status: needs_sourcing — see notes
        "crowd_detection",
        "behavior_smoking",
        "fight_detection",
        "zone_movement_tracking",    # fires as "Movement", not "Patrol" — see alert_overrides
        "fire_detection",
        "camera_tamper",
        "unknown_person_risk",
    ],
    zone_mode="named",   # e.g. "Lobby", "Teller Counter", "Vault Approach"
    thresholds={
        # "More sure" — raise confidence bar to cut false positives on
        # high-consequence alerts before they reach a branch manager.
        "weapon": {"confidence": 0.82},   # up from guard_monitoring default 0.75
        "fire":   {"min_confidence": 0.65},  # up from 0.55
    },
    alert_overrides={
        # "More alert" — shorter cooldowns so repeat/sustained threats
        # aren't suppressed as long as in guard_monitoring.
        "Weapon Detected":     {"cooldown": 60},   # vs default ~300s elsewhere
        "Camera Tamper":       {"cooldown": 60},
        "Fight / Violence Detected": {"cooldown": 60},
        # NOTE: zone_movement_tracking's alert is NOT suppressed here.
        # Bug found 2026-07: this used to set
        # {"Patrol": {"alert_enabled": False}}, which fully silenced the
        # alert before _translate_alert_type() ever got a chance to
        # rename it — despite _MOVEMENT_LABEL (alert_manager.py) already
        # correctly mapping bank_security's zone-transition alert to
        # "Movement: ..." wording. The alert should fire, just under
        # "Movement" instead of "Patrol" — that's the whole point of the
        # translation layer. Removing the override is the fix.
    },
    status="partial",
    notes=(
        "face_covering_detection is included in this profile's capability list "
        "but its underlying model still needs sourcing/evaluation (no reliable "
        "off-the-shelf balaclava/scarf detector exists yet — COVID-era mask "
        "detectors are trained for surgical masks, not concealment). Runs "
        "without this capability until resolved; everything else in this "
        "profile is fully available today."
    ),
)
