import torch
import torchvision.transforms as T
from torchvision.models.video import r3d_18
import collections
import cv2


class ActionRecognitionEngine:
    def __init__(self,
                 device="cpu",
                 clip_length=16):

        self.device = torch.device(device)
        self.clip_length = clip_length

        self.model = r3d_18(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        self.frame_buffers = {}

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
        ])

        self.kinetics_classes = self._load_kinetics_labels()

    def _load_kinetics_labels(self):
        from torchvision.datasets.utils import download_url
        import os

        label_path = "kinetics_labels.txt"

        if not os.path.exists(label_path):
            download_url(
                "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt",
                ".",
                filename=label_path
            )

        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        return labels

    def update(self, track_id, frame, person_box):

        if track_id not in self.frame_buffers:
            self.frame_buffers[track_id] = collections.deque(maxlen=self.clip_length)

        x1, y1, x2, y2 = person_box
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        processed = self.transform(person_crop)
        self.frame_buffers[track_id].append(processed)

        if len(self.frame_buffers[track_id]) < self.clip_length:
            return None

        clip = torch.stack(list(self.frame_buffers[track_id]), dim=1)
        clip = clip.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(clip)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.topk(probs, 1)

        action_label = self.kinetics_classes[top_idx.item()]
        confidence = top_prob.item()

        return {
            "action": action_label,
            "confidence": confidence
        }