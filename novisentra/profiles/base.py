"""
novisentra/profiles/base.py
──────────────────────────────
BaseProfile — a named bundle of capabilities + config overrides.

A profile is what gets assigned to a customer deployment. Only the
capabilities listed in the active profile are loaded and run — this is
what makes "give the customer only what they need" real: a retail
customer's deployment never loads the weapon model's threshold tuning
for guard patrol, never runs face recognition if it's not in their
capability list, etc.
"""

from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from novisentra.capabilities.registry import validate_capability_keys


@dataclass
class Zone:
    """One named zone for profiles using zone_mode='named'."""
    id: str                      # short key, e.g. "kitchen"
    label: str                   # display name, e.g. "Kitchen"
    polygon: Optional[list] = None  # [[x,y], ...] — set per-deployment during camera calibration


@dataclass
class BaseProfile:
    id: str
    name: str
    description: str
    capabilities: list                 # list of capability keys from the registry
    zone_mode: str = "none"            # "grid" | "named" | "none"
    zones: list = field(default_factory=list)   # list[Zone], only used if zone_mode == "named"
    thresholds: dict = field(default_factory=dict)     # overrides merged onto rules_config.yaml
    alert_overrides: dict = field(default_factory=dict)  # per-alert-type severity/cooldown overrides
    status: str = "available"          # "available" | "partial" | "planned"
    notes: str = ""                    # human-readable caveats (e.g. capability needs training)

    def __post_init__(self):
        bad = validate_capability_keys(self.capabilities)
        if bad:
            raise ValueError(
                f"Profile '{self.id}' references unknown capabilities: {bad}. "
                f"Check novisentra/capabilities/registry.py for valid keys."
            )

    def has_capability(self, key: str) -> bool:
        return key in self.capabilities

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "zone_mode": self.zone_mode,
            "zones": [
                {"id": z.id, "label": z.label} for z in self.zones
            ] if self.zones else [],
            "thresholds": self.thresholds,
            "alert_overrides": self.alert_overrides,
            "status": self.status,
            "notes": self.notes,
        }
