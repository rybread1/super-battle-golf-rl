"""CNN for detecting ball/pin icons and objects in game frames.

Input: 640x360 RGB image (resized from 1280x720).
Output per target: presence logit + normalized (x, y).

Targets:
    ball_icon — UI ball indicator overlay
    pin_icon  — UI flag indicator overlay
    ball      — actual golf ball in the scene
    pin       — actual flagstick on the green

Architecture: 5-block backbone → FPN (feature pyramid) with top-down pathway
→ spatial heatmap heads with soft-argmax. Icon heads use coarser P4 features
(22x40), object heads use finer P3 features (45x80) for better localization.
"""

import torch
import torch.nn as nn

TARGETS = (
    "ball_icon",
    "pin_icon",
    "ball",
    # "pin",
)

# Object heads get finer P3 features; icon heads get coarser P4
OBJECT_TARGETS = ("ball", "pin")


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""

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


class SpatialDetectionHead(nn.Module):
    """Detection head with spatial heatmap and soft-argmax localization.

    Produces a heatmap from spatial features, then computes expected (x, y)
    via soft-argmax. Presence is predicted from the global pooled vector.
    """

    def __init__(self, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.feat_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # -> (B, 1, H, W) raw heatmap
        )
        self.presence_fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        features: (B, C, H, W) spatial feature map
        pooled: (B, C) global pooled features
        Returns: (pred (B, 3), heatmap (B, 1, H, W))
        """
        B, C, H, W = features.shape

        features = self.feat_dropout(features)
        heatmap = self.conv(features)                  # (B, 1, H, W)
        heatmap_flat = heatmap.view(B, -1)             # (B, H*W)
        weights = torch.softmax(heatmap_flat, dim=1)   # (B, H*W)
        weights = weights.view(B, 1, H, W)

        # Coordinate grids — cell centers mapped to [0, 1]
        gy = ((torch.arange(H, device=features.device).float() + 0.5) / H).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = ((torch.arange(W, device=features.device).float() + 0.5) / W).view(1, 1, 1, W).expand(B, 1, H, W)

        x_coord = (weights * gx).sum(dim=(2, 3)).squeeze(1)
        y_coord = (weights * gy).sum(dim=(2, 3)).squeeze(1)
        presence = self.presence_fc(pooled).squeeze(1)

        pred = torch.stack([presence, x_coord, y_coord], dim=1)
        return pred, heatmap


class IconNet(nn.Module):
    """CNN for ball/pin icon and object detection.

    5-block backbone with Feature Pyramid Network (FPN) providing multi-scale
    features. Icon heads (ball_icon, pin_icon) use P4 (22x40) for coarse
    localization. Object heads (ball, pin) use P3 (45x80) for finer spatial
    precision.

    Input shape: (B, 3, 360, 640) — RGB, channels-first.
    Output: dict with tensors per target, each (B, 3):
        [0] = presence logit (raw, use sigmoid for probability)
        [1] = x coordinate (0-1, fraction of frame width)
        [2] = y coordinate (0-1, fraction of frame height)
    """

    INPUT_H = 360
    INPUT_W = 640

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        # 5 conv blocks with pooling
        self.block1 = ConvBlock(3, 32)      # 360x640 -> 180x320
        self.block2 = ConvBlock(32, 64)     # -> 90x160
        self.block3 = ConvBlock(64, 128)    # -> 45x80
        self.block4 = ConvBlock(128, 128)   # -> 22x40
        self.block5 = ConvBlock(128, 128)   # -> 11x20

        # FPN: lateral connections (project to common channel count)
        fpn_ch = 128
        self.lateral5 = nn.Conv2d(128, fpn_ch, 1)
        self.lateral4 = nn.Conv2d(128, fpn_ch, 1)
        self.lateral3 = nn.Conv2d(128, fpn_ch, 1)

        # FPN: smoothing convs after top-down addition (reduce aliasing)
        self.smooth4 = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Icon heads on P4 (22x40), object heads on P3 (45x80)
        self.heads = nn.ModuleDict({
            name: SpatialDetectionHead(fpn_ch, dropout=dropout)
            for name in TARGETS
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Backbone
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)   # 45x80, 128ch
        x4 = self.block4(x3)   # 22x40, 128ch
        x5 = self.block5(x4)   # 11x20, 128ch

        # FPN top-down pathway
        p5 = self.lateral5(x5)                                      # 11x20
        p4 = self.lateral4(x4) + nn.functional.interpolate(
            p5, size=x4.shape[2:], mode="bilinear", align_corners=False
        )
        p4 = self.smooth4(p4)                                       # 22x40
        p3 = self.lateral3(x3) + nn.functional.interpolate(
            p4, size=x3.shape[2:], mode="bilinear", align_corners=False
        )
        p3 = self.smooth3(p3)                                       # 45x80

        # Pool from P4 for all heads (global context)
        pooled = self.pool(p4).flatten(1)

        result = {}
        heatmaps = {}
        for name, head in self.heads.items():
            # Object heads get finer P3 features, icon heads get P4
            features = p3 if name in OBJECT_TARGETS else p4
            pred, heatmap = head(features, pooled)
            result[name] = pred
            heatmaps[name] = heatmap

        # Store heatmaps for loss computation (accessed via model._heatmaps)
        self._heatmaps = heatmaps
        return result

    @torch.no_grad()
    def predict(self, frame_rgb, device: str = "cuda",
                threshold: float = 0.5) -> dict[str, tuple[int, int] | None]:
        """Run inference on a single numpy RGB frame (H, W, 3).

        Returns dict with keys for each target, each (x, y) in original
        frame coordinates or None if not detected.
        """
        import cv2

        h, w = frame_rgb.shape[:2]
        resized = cv2.resize(frame_rgb, (self.INPUT_W, self.INPUT_H))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        self.eval()
        out = self.forward(tensor)

        result: dict[str, tuple[int, int] | None] = {}
        for key in TARGETS:
            pred = out[key][0]  # (3,)
            prob = torch.sigmoid(pred[0]).item()
            if prob >= threshold:
                x = int(pred[1].item() * w)
                y = int(pred[2].item() * h)
                result[key] = (x, y)
            else:
                result[key] = None

        return result
