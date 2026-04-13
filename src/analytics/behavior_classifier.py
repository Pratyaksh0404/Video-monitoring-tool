import os
import torch
import open_clip
from PIL import Image
import cv2

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "model_cache"
)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.abspath(_CACHE_DIR))
os.environ.setdefault("TORCH_HOME",            os.path.abspath(_CACHE_DIR))


class BehaviorClassifier:
    """
    CLIP-based behavior classifier.

    Threshold revisions (v3)
    ─────────────────────────
    Smoking: 0.70 → 0.60. The 0.70 threshold was too aggressive and blocked
    ALL smoking detection. 0.60 is still above the previous 0.58 that caused
    false positives, but allows genuine smoking to pass. The new BehaviorEngine
    majority-vote system provides the second layer of protection.

    Sleeping: 0.50 → 0.45. Was blocking sleeping detection. BehaviorEngine
    now handles false positive filtering via majority vote.

    Distracted: kept at 0.40 — fine for the new engine.
    """

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
        "SLEEPING":         0.45,   # restored from 0.50 — was blocking detection
        "PHONE_USE":        0.50,
        "IDLE":             0.50,
        "DISTRACTED_OTHER": 0.40,
        "SMOKING":          0.60,   # lowered from 0.70 — was blocking ALL smoking
    }

    def __init__(self, device="cpu"):
        self.device   = device
        self.is_ready = False

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50",
            pretrained="openai",
            cache_dir=os.path.abspath(_CACHE_DIR),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer("RN50")

        self.labels = [
            "a security guard standing alert and watching",
            "a security guard sleeping or dozing off on duty",
            "a security guard using a mobile phone",
            "a security guard sitting or standing idle and inactive",
            "a distracted security guard looking away or talking to someone",
            "a person smoking a cigarette or holding one to their mouth",
        ]

        self.text_tokens = self.tokenizer(self.labels).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.is_ready = True

    def predict(self, frame, person_box):
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

        min_conf = self._MIN_CONF.get(label, 0.35)
        if confidence < min_conf:
            return "NORMAL", confidence

        return label, confidence