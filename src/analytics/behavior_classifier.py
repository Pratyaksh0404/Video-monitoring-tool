import os
import sys
import torch
from PIL import Image
import cv2

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "model_cache"
)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.abspath(_CACHE_DIR))
os.environ.setdefault("TORCH_HOME",            os.path.abspath(_CACHE_DIR))

# Reach novisentra/ at the project root from src/analytics/behavior_classifier.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from novisentra.utils.model_loader import load_clip_safely

# ═══════════════════════════════════════════════════════════════════════════
# Shared model singleton — multi-camera fix
#
# Every camera gets its own BehaviorWorker (own queue, own thread, own
# per-track cache — all cheap, stay independent). But loading the actual
# CLIP model (RN50, torch-based) is expensive and was previously repeated
# per camera. With 2+ cameras starting together, that meant 2 simultaneous
# CPU-bound model loads competing for the same cores — this is what caused
# the multi-camera startup freeze. Fix: load it once, share the object.
# Thread-safe via a lock so two cameras starting at the same instant don't
# both trigger a redundant load.
# ═══════════════════════════════════════════════════════════════════════════
import threading as _threading
_shared_lock  = _threading.Lock()
_shared_model = None
_shared_preprocess = None
_shared_load_attempted = False


def get_shared_clip(device="cpu"):
    """
    Returns (model, preprocess) — loaded once and reused by every camera's
    BehaviorClassifier. Safe to call concurrently from multiple camera
    threads at startup.
    """
    global _shared_model, _shared_preprocess, _shared_load_attempted

    with _shared_lock:
        if _shared_load_attempted:
            return _shared_model, _shared_preprocess
        _shared_load_attempted = True

        model, preprocess = load_clip_safely(
            model_name="RN50",
            pretrained="openai",
            cache_dir=os.path.abspath(_CACHE_DIR),
        )
        _shared_model = model
        _shared_preprocess = preprocess
        return _shared_model, _shared_preprocess


class BehaviorClassifier:
    _LABEL_MAP = [
        "NORMAL",
        "SLEEPING",
        "PHONE_USE",
        "IDLE",
        "DISTRACTED_OTHER",
        "SMOKING",
    ]

    _MIN_CONF = {
        "NORMAL":           0.25,
        "SLEEPING":         0.55,   # raised from 0.45 — too many false positives on "looking down"
        "PHONE_USE":        0.50,
        "IDLE":             0.50,
        "DISTRACTED_OTHER": 0.45,
        "SMOKING":          0.70,   # raised from 0.65 — hand-near-face triggers too easily
    }

    # Bug found 2026-07: the Settings tab's "Behavior Detection (CLIP)"
    # sliders (Sleeping/Phone/Smoking/Idle — CLIP threshold) write to
    # rules_config.yaml's behavior.clip_thresholds, and the UI claims
    # "Settings saved — thresholds updated live" — but nothing in this
    # file ever read that config. _MIN_CONF above was a pure hardcoded
    # constant with zero connection to the config file at all, so tuning
    # it from the dashboard (e.g. raising the Sleeping threshold to cut
    # false positives) silently did nothing. Fixed by reading through
    # this module-level override dict instead, which main_web.py
    # populates from _cfg() at startup AND repopulates inside
    # reload_config() on every live "Save & Apply" — see main_web.py's
    # apply_behavior_clip_thresholds(). A plain module-level dict (not
    # per-instance) matches how the CLIP model itself is already a
    # shared singleton across every camera (get_shared_clip above), and
    # the Settings tab is a single global panel, not per-camera anyway.
    _MIN_CONF_LIVE_OVERRIDE: dict = {}

    def __init__(self, device="cpu"):
        self.device   = device
        self.is_ready = False

        # ── Safe load: shared singleton across all cameras first, local ──
        # cache fallback second. Previously this called
        # open_clip.create_model_and_transforms() directly PER CAMERA,
        # which is what caused the multi-camera startup freeze — 2+
        # cameras each loading a full CLIP model at once, competing for
        # the same CPU. Now the first camera to reach here loads it once;
        # every camera after that reuses the same loaded model object.
        self.model, self.preprocess = get_shared_clip(device=device)

        if self.model is None:
            # No cache AND no internet (or some other load failure).
            # Degrade gracefully — behavior detection is disabled for this
            # session, but the rest of the pipeline (person detection,
            # weapon, fire, tamper, etc.) keeps running normally.
            print("[BehaviorClassifier] ✗ CLIP model unavailable — "
                  "behavior detection disabled for this session. "
                  "All other detectors continue running normally.")
            return

        self.model = self.model.to(self.device)
        self.model.eval()

        import open_clip
        self.tokenizer = open_clip.get_tokenizer("RN50")

        # v5: More specific prompts to reduce false positives
        self.labels = [
            "a security guard standing upright alert and watching attentively",
            "a security guard with eyes closed head drooping asleep on duty",
            "a security guard holding and looking at a mobile phone screen",
            "a security guard sitting or standing idle and inactive",
            "a distracted security guard looking away or talking to someone",
            "a person with a lit cigarette with visible smoke near their mouth",
        ]

        self.text_tokens = self.tokenizer(self.labels).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.is_ready = True

    def predict(self, frame, person_box):
        if not self.is_ready:
            # Model failed to load — return a neutral, non-alerting result
            # rather than crashing the caller. The pipeline treats this
            # exactly like a normal "nothing detected yet" frame.
            return "ANALYZING", 0.0

        x1, y1, x2, y2 = person_box
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if crop.size == 0:
            return "ANALYZING", 0.0

        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()

        best_idx   = int(probs.argmax())
        confidence = float(probs[best_idx])
        label      = self._LABEL_MAP[best_idx]

        min_conf = self._MIN_CONF_LIVE_OVERRIDE.get(
            label, self._MIN_CONF.get(label, 0.35))
        if confidence < min_conf:
            return "NORMAL", confidence

        return label, confidence


def set_min_conf_overrides(overrides: dict) -> None:
    """
    Called by main_web.py (at startup and from reload_config()) with
    whatever behavior.clip_thresholds the Settings tab has saved to
    rules_config.yaml — e.g. {"SLEEPING": 0.80, "PHONE_USE": 0.5, ...}.
    Takes effect immediately for every camera's next classification call,
    same live-reload behavior the Settings UI already claims.
    """
    BehaviorClassifier._MIN_CONF_LIVE_OVERRIDE = dict(overrides or {})
