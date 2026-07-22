"""
novisentra/utils/model_loader.py
───────────────────────────────────
Fixes the "resnet50 / CLIP fetched from HuggingFace every startup" bug.

Root cause: even when HF_HOME/HUGGINGFACE_HUB_CACHE point to a local folder,
the huggingface_hub / open_clip loader still makes a network call on every
startup to check for a newer model version (an ETag check), unless told not to.
If there is no internet, or HF is briefly unreachable, this either hangs or
throws — and previously nothing caught it, so the whole app crashed.

Fix: try loading 100% from local cache first (local_files_only=True).
Only fall back to a real network fetch if nothing is cached yet.
Once cached, the app never touches the network again for that model.

Usage:
    from novisentra.utils.model_loader import load_clip_safely

    model, preprocess = load_clip_safely(
        model_name="RN50",
        pretrained="openai",
        cache_dir=CACHE_DIR,
    )
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    from utils.logger import get_logger
    _log = get_logger("model_loader")
except Exception:
    import logging
    _log = logging.getLogger("model_loader")


def load_clip_safely(model_name: str, pretrained: str, cache_dir: str):
    """
    Load an OpenCLIP model — local cache first, network only if truly needed.

    Returns (model, preprocess) or (None, None) if loading fails entirely
    (caller should treat this as "behavior detection disabled" and continue
    running rather than crashing the whole pipeline).
    """
    import open_clip

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("TORCH_HOME", cache_dir)

    # ── Attempt 1: fully offline, from local cache ──────────────────────────
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        _log.info(f"CLIP model loaded from local cache (offline): {model_name}/{pretrained}")
        return model, preprocess
    except Exception as e:
        _log.warning(f"CLIP offline load failed (not cached yet?): {type(e).__name__}: {e}")

    # ── Attempt 2: allow network, cache it for next time ─────────────────────
    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        _log.info(f"CLIP model downloaded and cached: {model_name}/{pretrained}")
        return model, preprocess
    except Exception as e:
        _log.error(
            f"CLIP model load failed completely (no cache, no internet): "
            f"{type(e).__name__}: {e}. Behavior detection will be disabled."
        )
        return None, None
    finally:
        # Always return to offline-preferred mode after this point
        os.environ["HF_HUB_OFFLINE"] = "1"


def warm_cache(model_name: str, pretrained: str, cache_dir: str) -> bool:
    """
    Call this once during Docker build (or a setup script) with internet
    available, so the running container never needs network access for
    model loading. Returns True if the model is now cached locally.
    """
    model, preprocess = load_clip_safely(model_name, pretrained, cache_dir)
    return model is not None
