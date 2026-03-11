"""Dataset for icon/object detection training.

Annotation format (JSON):
{
    "frames": [
        {
            "file": "screenshot_001.png",
            "ball_icon": [x, y] or null,
            "pin_icon": [x, y] or null,
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

from sbg.models.icon_net import IconNet, TARGETS


class IconDataset(Dataset):
    """Dataset of annotated game frames for icon/object detection training."""

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

        # Normalize coordinates to [0, 1] before augmentation
        coords = {}
        for key in TARGETS:
            raw = entry.get(key)
            if raw is not None:
                coords[key] = (raw[0] / orig_w, raw[1] / orig_h)
            else:
                coords[key] = None

        # Resize to model input
        img = cv2.resize(img, (IconNet.INPUT_W, IconNet.INPUT_H))

        if self.augment:
            img, coords = self._augment(img, coords)

        # HWC uint8 -> CHW float32 [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        result = {"image": tensor}
        for key in TARGETS:
            if coords[key] is not None:
                x_norm, y_norm = coords[key]
                result[key] = torch.tensor([1.0, x_norm, y_norm])
            else:
                result[key] = torch.tensor([0.0, 0.0, 0.0])

        return result

    def _augment(self, img: np.ndarray,
                 coords: dict[str, tuple[float, float] | None],
                 ) -> tuple[np.ndarray, dict]:
        """Augmentations for small-dataset training."""
        # Horizontal flip (mirror x coordinates)
        if np.random.random() < 0.5:
            img = img[:, ::-1].copy()
            coords = {
                k: (1.0 - x, y) if v is not None else None
                for k, v in coords.items()
                for x, y in [v if v is not None else (0, 0)]
            }

        # Random brightness
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Random contrast
        if np.random.random() < 0.4:
            factor = np.random.uniform(0.7, 1.3)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random hue/saturation shift (in HSV space)
        if np.random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(-20, 20), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random crop/zoom (scale 85-100%, recentered)
        if np.random.random() < 0.4:
            h, w = img.shape[:2]
            scale = np.random.uniform(0.85, 1.0)
            crop_h, crop_w = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            img = cv2.resize(img[top:top+crop_h, left:left+crop_w], (w, h))
            # Remap coordinates: shift by crop origin, scale to new size
            new_coords = {}
            for k, v in coords.items():
                if v is not None:
                    x, y = v
                    nx = (x * w - left) / crop_w
                    ny = (y * h - top) / crop_h
                    # Drop if coordinate lands outside the crop
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        new_coords[k] = (nx, ny)
                    else:
                        new_coords[k] = None
                else:
                    new_coords[k] = None
            coords = new_coords

        # Gaussian blur
        if np.random.random() < 0.2:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Channel noise
        if np.random.random() < 0.3:
            noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img, coords
