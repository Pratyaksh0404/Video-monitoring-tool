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
_shared_lock       = _threading.Lock()
_shared_model      = None
_shared_preprocess = None
_shared_loaded     = False


def get_shared_clip(device="cpu"):
    """
    Returns (model, preprocess) — loaded exactly once, shared by every
    BehaviorClassifier instance (i.e. every camera thread).

    Simple correct implementation: use a lock, check if already loaded,
    if not load, release. The key insight that was missing in previous
    attempts: this function is called from BehaviorWorker._loop() which
    runs on a BACKGROUND THREAD — it does NOT block Flask or the main
    thread. The only thread it blocks is the BehaviorWorker's own thread
    while CLIP loads, and that is exactly what we want (the worker can't
    classify behavior until CLIP is loaded anyway).

    With the 1.5s stagger between camera starts in camera_manager.py,
    the second camera's BehaviorWorker starts 1.5s after the first, so
    by the time it calls get_shared_clip(), the first camera's load is
    almost certainly complete — it returns instantly from the `if
    _shared_loaded:` fast path. No events, no races, no starvation.
    """
    global _shared_model, _shared_preprocess, _shared_loaded

    with _shared_lock:
        if _shared_loaded:
            return _shared_model, _shared_preprocess

        model, preprocess = load_clip_safely(
            model_name="RN50",
            pretrained="openai",
            cache_dir=os.path.abspath(_CACHE_DIR),
        )
        _shared_model      = model
        _shared_preprocess = preprocess
        _shared_loaded     = True
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
        "SLEEPING":         0.85,   # strict — was 0.55, causing false positives
        "PHONE_USE":        0.50,
        "IDLE":             0.50,
        "DISTRACTED_OTHER": 0.45,
        "SMOKING":          0.70,
    }

    # Live override dict — populated from rules_config.yaml at startup and
    # on every Settings "Save & Apply". Takes effect immediately for all
    # camera threads on the next CLIP call.
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
    Called by main_web.py at startup and from reload_config() to push
    behavior.clip_thresholds from rules_config.yaml into the classifier.
    Takes effect immediately for every camera's next CLIP call.
    """
    BehaviorClassifier._MIN_CONF_LIVE_OVERRIDE = dict(overrides or {})


def set_confirm_ratio_overrides(overrides: dict) -> None:
    """
    Stub kept for import compatibility — confirm ratios live in
    BehaviorEngine, not BehaviorClassifier. See behavior_engine.py.
    main_web.py imports this name from here for convenience; the real
    implementation is in analytics.behavior_engine.
    """
    # Delegate to BehaviorEngine if available, otherwise no-op.
    try:
        from analytics.behavior_engine import set_confirm_ratio_overrides as _engine_fn
        _engine_fn(overrides)
    except Exception:
        pass
