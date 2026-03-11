"""CNN for detecting ball and pin icons in game frames.

Input: 320x180 RGB image (resized from 1280x720).
Output per icon (ball, pin): presence logit + normalized (x, y) coordinates.

Architecture: lightweight backbone (4 conv blocks) → two detection heads.
Coordinates are normalized to [0, 1] relative to frame dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetectionHead(nn.Module):
    """Predicts (present_logit, x, y) for one icon class."""

    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),  # [present_logit, x, y]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        # Sigmoid on coordinates to keep them in [0, 1]
        out[:, 1:] = torch.sigmoid(out[:, 1:])
        return out


class IconNet(nn.Module):
    """Lightweight CNN for ball and pin icon detection.

    Input shape: (B, 3, 180, 320)  — RGB, channels-first.
    Output: dict with 'ball' and 'pin' tensors, each (B, 3):
        [0] = presence logit (raw, use sigmoid for probability)
        [1] = x coordinate (0-1, fraction of frame width)
        [2] = y coordinate (0-1, fraction of frame height)
    """

    INPUT_H = 180
    INPUT_W = 320

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),     # 180x320 → 90x160
            ConvBlock(32, 64),    # → 45x80
            ConvBlock(64, 128),   # → 22x40
            ConvBlock(128, 256),  # → 11x20
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ball_head = DetectionHead(256)
        self.pin_head = DetectionHead(256)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return {
            "ball": self.ball_head(pooled),
            "pin": self.pin_head(pooled),
        }

    @torch.no_grad()
    def predict(self, frame_rgb, device: str = "cuda",
                threshold: float = 0.5) -> dict[str, tuple[int, int] | None]:
        """Run inference on a single numpy RGB frame (H, W, 3).

        Returns dict with 'ball' and 'pin' keys, each (x, y) in original
        frame coordinates or None if not detected.
        """
        import numpy as np

        h, w = frame_rgb.shape[:2]
        # Resize to model input size
        import cv2
        resized = cv2.resize(frame_rgb, (self.INPUT_W, self.INPUT_H))
        # HWC uint8 → CHW float32, normalized to [0, 1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        self.eval()
        out = self.forward(tensor)

        result: dict[str, tuple[int, int] | None] = {}
        for key in ("ball", "pin"):
            pred = out[key][0]  # (3,)
            prob = torch.sigmoid(pred[0]).item()
            if prob >= threshold:
                x = int(pred[1].item() * w)
                y = int(pred[2].item() * h)
                result[key] = (x, y)
            else:
                result[key] = None

        return result
