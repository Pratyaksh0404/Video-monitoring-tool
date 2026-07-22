"""
novisentra/profiles/loader.py
────────────────────────────────
Central profile registry. This is what api/routes/config.py and the
pipeline both call to get/activate a profile.
"""

from .guard_monitoring import GUARD_MONITORING
from .bank_security import BANK_SECURITY
from .retail_analytics import RETAIL_ANALYTICS
from .warehouse_ops import WAREHOUSE_OPS

_PROFILES = {
    p.id: p for p in [
        GUARD_MONITORING,
        BANK_SECURITY,
        RETAIL_ANALYTICS,
        WAREHOUSE_OPS,
    ]
}


def get_profile(profile_id: str):
    """Return a BaseProfile instance, or None if not found."""
    return _PROFILES.get(profile_id)


def list_profiles() -> dict:
    """Return {profile_id: profile.to_dict()} for every registered profile."""
    return {pid: p.to_dict() for pid, p in _PROFILES.items()}


def is_valid_profile(profile_id: str) -> bool:
    return profile_id in _PROFILES
