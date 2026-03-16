import os
import torch
import open_clip
from PIL import Image
import cv2

# Cache weights locally in the project so they are never re-downloaded
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # → src/
    "..", "model_cache"                                            # → project root/model_cache
)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.abspath(_CACHE_DIR))
os.environ.setdefault("TORCH_HOME",            os.path.abspath(_CACHE_DIR))


class BehaviorClassifier:
    def __init__(self, device="cpu"):
        self.device = device
        self.is_ready = False   # callers can poll this before using predict()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50",
            pretrained="openai",
            cache_dir=os.path.abspath(_CACHE_DIR),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer("RN50")

        self.labels = [
            "a security guard standing alert",
            "a security guard sleeping on duty",
            "a security guard using a mobile phone",
            "a security guard sitting idle",
            "a distracted security guard talking to someone"
        ]

        self.text_tokens = self.tokenizer(self.labels).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.is_ready = True

    def predict(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        crop = frame[max(0,y1):y2, max(0,x1):x2]

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

        best_idx   = probs.argmax()
        confidence = float(probs[best_idx])

        if confidence < 0.40:
            return "NORMAL", confidence

        mapped = [
            "NORMAL",
            "SLEEPING",
            "PHONE_USE",
            "IDLE",
            "DISTRACTED_OTHER"
        ]
        return mapped[best_idx], confidence