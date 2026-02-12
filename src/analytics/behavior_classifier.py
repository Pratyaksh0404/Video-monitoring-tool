import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


class BehaviorClassifier:
    def __init__(self, device="cpu"):
        self.device = device

        # Load MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Replace classifier head
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 4)

        self.model.to(self.device)
        self.model.eval()

        # Temporary random weights for now
        # Later you fine-tune this

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.classes = [
            "NORMAL",
            "SLEEPING",
            "PHONE_USE",
            "IDLE"
        ]

    def predict(self, frame, person_box):
        x1, y1, x2, y2 = person_box

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "NORMAL", 0.0

        image = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return self.classes[predicted.item()], confidence.item()
