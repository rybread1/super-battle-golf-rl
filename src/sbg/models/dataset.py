"""Dataset for icon detection training.

Annotation format (JSON):
{
    "frames": [
        {
            "file": "screenshot_001.png",
            "ball": [x, y] or null,
            "pin": [x, y] or null
        },
        ...
    ]
}

Coordinates in annotations are in original frame pixels (1280x720).
The dataset resizes frames to model input size and normalizes coordinates.
"""

import json
import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sbg.models.icon_net import IconNet


class IconDataset(Dataset):
    """Dataset of annotated game frames for icon detection training."""

    def __init__(self, annotations_path: str | pathlib.Path,
                 augment: bool = False):
        self.annotations_path = pathlib.Path(annotations_path)
        self.base_dir = self.annotations_path.parent
        self.augment = augment

        with open(self.annotations_path) as f:
            data = json.load(f)
        self.frames = data["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        entry = self.frames[idx]
        img_path = self.base_dir / entry["file"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Resize to model input
        img = cv2.resize(img, (IconNet.INPUT_W, IconNet.INPUT_H))

        if self.augment:
            img = self._augment(img)

        # HWC uint8 → CHW float32 [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Build targets: (present, x_norm, y_norm) for each icon
        targets = {}
        for key in ("ball", "pin"):
            coords = entry.get(key)
            if coords is not None:
                x_norm = coords[0] / orig_w
                y_norm = coords[1] / orig_h
                targets[key] = torch.tensor([1.0, x_norm, y_norm])
            else:
                targets[key] = torch.tensor([0.0, 0.0, 0.0])

        return {
            "image": tensor,
            "ball": targets["ball"],
            "pin": targets["pin"],
        }

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Simple augmentations that don't move icon positions."""
        # Random brightness
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        # Random channel noise
        if np.random.random() < 0.3:
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
